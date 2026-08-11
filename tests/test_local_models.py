from __future__ import annotations

import hashlib
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
import zipfile

import numpy as np
from PIL import Image

from photoredactor.local_generative import LocalGenerationOptions, LocalImageClient, _LocalServer
from photoredactor.model_manager import LCM_ACCELERATOR, MODEL_CATALOG, LocalModelSpec, ModelManagerError, ModelStore


class StubLocalClient(LocalImageClient):
    def _run(self, image, mask, prompt, negative, seed):
        self.last_request = (image.copy(), mask.copy(), prompt, negative, seed)
        result = np.full_like(image, (210, 40, 30, 255))
        return result


class LocalModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def model_spec(self, payload: bytes, source: Path) -> LocalModelSpec:
        return LocalModelSpec(
            model_id="test-model", name="Test", description="Test", filename="test.safetensors",
            url=source.as_uri(), source_url="https://example.invalid/model",
            license_name="Test license", license_url="https://example.invalid/license",
            size=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
        )

    def test_model_download_is_verified_and_removable(self) -> None:
        payload = b"real model payload" * 1024
        source = self.root / "source.safetensors"
        source.write_bytes(payload)
        spec = self.model_spec(payload, source)
        store = ModelStore(self.root / "store")
        progress = []

        installed = store.install_model(spec, lambda stage, done, total: progress.append((stage, done, total)))

        self.assertEqual(installed.read_bytes(), payload)
        self.assertTrue(store.model_installed(spec))
        self.assertTrue(any(item[0] == "Проверка SHA-256" for item in progress))
        store.remove_model(spec)
        self.assertFalse(store.model_installed(spec))

    def test_model_with_wrong_digest_is_rejected(self) -> None:
        source = self.root / "source.safetensors"
        source.write_bytes(b"payload")
        spec = self.model_spec(b"different", source)
        spec = LocalModelSpec(**{**spec.__dict__, "size": source.stat().st_size})

        with self.assertRaises(ModelManagerError):
            ModelStore(self.root / "store").install_model(spec)

    def test_completed_partial_download_is_verified_without_network(self) -> None:
        payload = b"complete partial model"
        missing_source = self.root / "missing.safetensors"
        spec = self.model_spec(payload, missing_source)
        store = ModelStore(self.root / "store")
        target = store.model_path(spec)
        target.parent.mkdir(parents=True)
        target.with_suffix(target.suffix + ".part").write_bytes(payload)

        installed = store.install_model(spec)

        self.assertEqual(installed.read_bytes(), payload)
        self.assertTrue(store.model_installed(spec))

    def test_engine_archive_is_verified_and_safely_extracted(self) -> None:
        archive = self.root / "engine.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("bin/sd-cli.exe", b"executable")
        payload = archive.read_bytes()

        class TestStore(ModelStore):
            def _latest_engine_assets(inner_self, backend):
                return [{
                    "tag": "test-release", "name": archive.name, "url": archive.as_uri(),
                    "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "backend": backend,
                }]

        store = TestStore(self.root / "store")
        executable = store.install_engine("cpu")
        self.assertEqual(executable.read_bytes(), b"executable")
        self.assertEqual(store.engine_executable("cpu"), executable)

    def test_ready_install_always_includes_lcm_accelerator(self) -> None:
        installed = []

        class RecordingStore(ModelStore):
            def install_engine(inner_self, backend, progress=None, cancel=None):
                return self.root / "sd-cli.exe"

            def install_model(inner_self, model, progress=None, cancel=None):
                installed.append(model.model_id)
                return inner_self.model_path(model)

        store = RecordingStore(self.root / "store")
        store.ensure_ready(MODEL_CATALOG[0].model_id, "cpu")

        self.assertEqual(installed, [MODEL_CATALOG[0].model_id, LCM_ACCELERATOR.model_id])

    def test_local_inpaint_passes_text_and_mask_to_model(self) -> None:
        executable, model = self.root / "sd-cli.exe", self.root / "model.safetensors"
        executable.write_bytes(b"x")
        model.write_bytes(b"x")
        client = StubLocalClient(executable, model, "cpu", LocalGenerationOptions(max_side=512))
        image = np.full((120, 160, 4), (20, 30, 40, 255), dtype=np.uint8)
        mask = np.zeros((120, 160), dtype=np.uint8)
        mask[45:75, 65:95] = 255

        result = client.inpaint(image, mask, "красная чашка", "текст", 123, "photographic")

        self.assertEqual(result.shape, image.shape)
        self.assertEqual(client.last_request[2:], ("красная чашка, photographic", "текст", 123))
        self.assertGreater(np.count_nonzero(client.last_request[1]), 0)
        np.testing.assert_array_equal(result[mask == 0], image[mask == 0])

    def test_local_outpaint_keeps_original_pixels_exact(self) -> None:
        executable, model = self.root / "sd-cli.exe", self.root / "model.safetensors"
        executable.write_bytes(b"x")
        model.write_bytes(b"x")
        client = StubLocalClient(executable, model, "cpu")
        image = np.full((24, 30, 4), (15, 70, 120, 255), dtype=np.uint8)

        result = client.outpaint(image, (5, 4, 7, 6), "лес", "люди", 77)

        self.assertEqual(result.shape, (34, 42, 4))
        np.testing.assert_array_equal(result[4:28, 5:35], image)
        self.assertEqual(client.last_request[2:], ("лес", "люди", 77))

    def test_local_server_sends_prompt_mask_and_generation_controls(self) -> None:
        received = {}
        generated = Image.new("RGB", (64, 64), (11, 22, 33))
        encoded = io.BytesIO()
        generated.save(encoded, format="PNG")

        class Handler(BaseHTTPRequestHandler):
            def do_POST(inner_self):
                size = int(inner_self.headers["Content-Length"])
                received.update(json.loads(inner_self.rfile.read(size)))
                body = json.dumps({"images": [base64.b64encode(encoded.getvalue()).decode("ascii")]})
                inner_self.send_response(200)
                inner_self.send_header("Content-Type", "application/json")
                inner_self.end_headers()
                inner_self.wfile.write(body.encode("utf-8"))

            def log_message(self, *_args):
                pass

        http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        lora = self.root / "lcm.safetensors"
        lora.write_bytes(b"real lora")
        server = _LocalServer(self.root / "sd-cli.exe", self.root / "model", "cpu", "CPU", lora)
        server.port = http.server_address[1]
        server.start = lambda: None
        options = LocalGenerationOptions(steps=6, cfg_scale=1.5, strength=0.72, sampler="LCM")
        image = np.full((64, 64, 4), (1, 2, 3, 255), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[12:40, 20:48] = 255
        try:
            result = server.generate(image, mask, "красная кружка", "текст", 456, options, None)
        finally:
            http.shutdown()
            http.server_close()

        self.assertEqual(result.shape, image.shape)
        self.assertEqual(received["prompt"], "красная кружка")
        self.assertEqual(received["negative_prompt"], "текст")
        self.assertEqual(received["steps"], 6)
        self.assertEqual(received["cfg_scale"], 1.5)
        self.assertEqual(received["denoising_strength"], 0.72)
        self.assertEqual(received["sampler_name"], "LCM")
        self.assertEqual(received["lora"], [{"path": "lcm.safetensors", "multiplier": 1.0}])
        self.assertEqual(received["seed"], 456)
        decoded_mask = Image.open(io.BytesIO(base64.b64decode(received["mask"])))
        self.assertEqual(np.asarray(decoded_mask)[20, 30], 255)


if __name__ == "__main__":
    unittest.main()
