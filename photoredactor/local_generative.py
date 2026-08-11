from __future__ import annotations

import atexit
import base64
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image

from .generative_api import GeneratedVariant, GenerativeAPIError, strict_outpaint_result, variant_seeds


SAMPLERS = {
    "DPM++ 2M": "dpm++2m",
    "Euler a": "euler_a",
    "Euler": "euler",
    "LCM": "lcm",
}
PERFORMANCE_LABELS = {"Быстро": "fast", "Баланс": "balanced", "Качество": "quality", "Вручную": "custom"}
PERFORMANCE_VALUES = {"fast": (4, 1.2, "LCM"), "balanced": (6, 1.5, "LCM"), "quality": (22, 7.0, "DPM++ 2M")}


def _creation_flags() -> int:
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def _image_base64(image: Image.Image) -> str:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


class _LocalServer:
    def __init__(self, executable: Path, model: Path, backend: str, device: str, lora: Path | None = None) -> None:
        self.executable = executable
        self.model = model
        self.backend = backend
        self.device = device
        self.lora = lora
        self.process: subprocess.Popen | None = None
        self.port = 0
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.executable.with_name("sd-server.exe").is_file()

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _backend_assignment(self) -> str:
        if self.device.upper() == "CPU":
            return "CPU"
        return f"diffusion={self.device},clip=CPU,vae=CPU"

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def start(self) -> None:
        with self._lock:
            if self.alive:
                return
            server = self.executable.with_name("sd-server.exe")
            if not server.is_file():
                raise GenerativeAPIError("Сервер локальной модели не найден")
            self.port = self._free_port()
            command = [
                str(server), "--model", str(self.model), "--listen-ip", "127.0.0.1",
                "--listen-port", str(self.port), "--backend", self._backend_assignment(),
                "--rng", "cpu", "--mmap", "--vae-tiling", "--fa",
            ]
            if self.lora is not None:
                command.extend(("--lora-model-dir", str(self.lora.parent)))
            self.process = subprocess.Popen(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=_creation_flags(),
            )
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if not self.alive:
                raise GenerativeAPIError("Локальный сервер модели не запустился")
            try:
                endpoint = "loras" if self.lora is not None else "options"
                with urlopen(f"http://127.0.0.1:{self.port}/sdapi/v1/{endpoint}", timeout=1):
                    return
            except (OSError, URLError):
                time.sleep(0.2)
        self.stop()
        raise GenerativeAPIError("Локальная модель слишком долго загружается")

    def stop(self) -> None:
        with self._lock:
            process, self.process = self.process, None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=6)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)

    def generate(
        self, image: np.ndarray, mask: np.ndarray, prompt: str, negative: str,
        seed: int, options: "LocalGenerationOptions", cancel: threading.Event | None,
    ) -> np.ndarray:
        self.start()
        payload = {
            "prompt": prompt,
            "negative_prompt": negative,
            "init_images": [_image_base64(Image.fromarray(image, "RGBA").convert("RGB"))],
            "mask": _image_base64(Image.fromarray(mask.astype(np.uint8), "L")),
            "width": image.shape[1],
            "height": image.shape[0],
            "steps": options.steps,
            "cfg_scale": options.cfg_scale,
            "denoising_strength": options.strength,
            "sampler_name": options.sampler,
            "seed": seed,
            "batch_size": 1,
        }
        if options.sampler == "LCM" and self.lora is not None:
            payload["lora"] = [{"path": self.lora.name, "multiplier": 1.0}]
        request = Request(
            f"http://127.0.0.1:{self.port}/sdapi/v1/img2img",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        state: dict[str, object] = {}

        def worker() -> None:
            try:
                with urlopen(request, timeout=3600) as response:
                    state["result"] = json.load(response)
            except Exception as exc:
                state["error"] = exc

        thread = threading.Thread(target=worker, name="local-image-request", daemon=True)
        thread.start()
        while thread.is_alive():
            if cancel is not None and cancel.is_set():
                self.stop()
                raise GenerativeAPIError("Локальная генерация отменена")
            thread.join(0.15)
        if "error" in state:
            error = state["error"]
            if isinstance(error, HTTPError):
                try:
                    details = error.read().decode("utf-8", errors="replace")
                except OSError:
                    details = str(error)
                raise GenerativeAPIError(f"Локальная модель вернула ошибку: {details}") from error
            raise GenerativeAPIError(f"Связь с локальной моделью прервана: {error}") from error
        result = state.get("result")
        images = result.get("images", []) if isinstance(result, dict) else []
        if not images:
            raise GenerativeAPIError("Локальная модель не вернула изображение")
        try:
            raw = base64.b64decode(str(images[0]).split(",", 1)[-1], validate=True)
            with Image.open(io.BytesIO(raw)) as generated:
                return np.asarray(generated.convert("RGBA"), dtype=np.uint8)
        except (OSError, ValueError) as exc:
            raise GenerativeAPIError("Локальная модель вернула повреждённое изображение") from exc


class _LocalServerPool:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._server: _LocalServer | None = None

    def get(self, executable: Path, model: Path, backend: str, device: str, lora: Path | None) -> _LocalServer:
        key = (executable.resolve(), model.resolve(), backend, device, None if lora is None else lora.resolve())
        with self._lock:
            current = self._server
            current_key = None if current is None else (
                current.executable.resolve(), current.model.resolve(), current.backend, current.device,
                None if current.lora is None else current.lora.resolve(),
            )
            if current_key == key and current is not None:
                return current
            self._server = _LocalServer(executable, model, backend, device, lora)
        if current is not None:
            current.stop()
        return self._server

    def stop(self) -> None:
        with self._lock:
            current, self._server = self._server, None
        if current is not None:
            current.stop()


_SERVER_POOL = _LocalServerPool()


def shutdown_local_servers() -> None:
    _SERVER_POOL.stop()


atexit.register(shutdown_local_servers)


@dataclass(frozen=True)
class LocalGenerationOptions:
    steps: int = 24
    cfg_scale: float = 7.0
    strength: float = 0.88
    sampler: str = "DPM++ 2M"
    max_side: int = 768

    def normalized(self) -> "LocalGenerationOptions":
        return LocalGenerationOptions(
            steps=max(4, min(80, int(self.steps))),
            cfg_scale=max(1.0, min(20.0, float(self.cfg_scale))),
            strength=max(0.05, min(1.0, float(self.strength))),
            sampler=self.sampler if self.sampler in SAMPLERS else "DPM++ 2M",
            max_side=max(512, min(1280, int(self.max_side))),
        )


def _multiple_of_64(value: float) -> int:
    return max(64, int(round(value / 64.0)) * 64)


def _fit_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    scale = min(1.0, max_side / max(width, height))
    return _multiple_of_64(width * scale), _multiple_of_64(height * scale)


def _selection_region(mask: np.ndarray, padding_ratio: float = 0.55) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        raise GenerativeAPIError("Маска локального заполнения пуста")
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    padding = max(96, round(max(x2 - x1, y2 - y1) * padding_ratio))
    return max(0, x1 - padding), max(0, y1 - padding), min(mask.shape[1], x2 + padding), min(mask.shape[0], y2 + padding)


def _outpaint_canvas(image: np.ndarray, margins: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    left, top, right, bottom = margins
    if not any(margins):
        raise GenerativeAPIError("Укажите расширение хотя бы с одной стороны")
    rgb = image[:, :, :3]
    canvas = cv2.copyMakeBorder(rgb, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    blur = cv2.GaussianBlur(canvas, (0, 0), sigmaX=max(8.0, min(canvas.shape[:2]) / 32.0))
    canvas = np.dstack((blur, np.full(blur.shape[:2], 255, dtype=np.uint8)))
    canvas[top:top + image.shape[0], left:left + image.shape[1]] = image
    mask = np.full(canvas.shape[:2], 255, dtype=np.uint8)
    mask[top:top + image.shape[0], left:left + image.shape[1]] = 0
    return canvas, mask


class LocalImageClient:
    def __init__(
        self, executable: str | Path, model: str | Path, backend: str,
        options: LocalGenerationOptions | None = None, cancel: threading.Event | None = None,
        lora: str | Path | None = None,
    ) -> None:
        self.executable = Path(executable)
        self.model = Path(model)
        self.backend = backend
        self.options = (options or LocalGenerationOptions()).normalized()
        self.cancel = cancel
        self.lora = None if lora is None else Path(lora)
        if not self.executable.is_file() or not self.model.is_file():
            raise GenerativeAPIError("Локальная модель или движок не установлены")
        if self.options.sampler == "LCM" and (self.lora is None or not self.lora.is_file()):
            raise GenerativeAPIError("Ускоритель LCM не установлен")

    def _device(self) -> str:
        if self.backend == "cpu":
            return "CPU"
        try:
            result = subprocess.run(
                [str(self.executable), "--list-devices"], capture_output=True, text=True,
                timeout=20, creationflags=_creation_flags(),
            )
            devices = []
            for line in (result.stdout + "\n" + result.stderr).splitlines():
                if "\t" in line:
                    name, description = line.split("\t", 1)
                    if name.lower().startswith(self.backend):
                        devices.append((name, description.lower()))
            preferred = next((name for name, text in devices if "nvidia" in text or "amd" in text or "radeon" in text), None)
            return preferred or (devices[0][0] if devices else self.backend)
        except (OSError, subprocess.SubprocessError):
            return self.backend

    def _run(self, image: np.ndarray, mask: np.ndarray, prompt: str, negative: str, seed: int) -> np.ndarray:
        options = self.options
        width, height = _fit_size(image.shape[1], image.shape[0], options.max_side)
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
        resized_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        device = self._device()
        server = _SERVER_POOL.get(self.executable, self.model, self.backend, device, self.lora)
        if server.available:
            return server.generate(resized, resized_mask, prompt, negative, seed, options, self.cancel)
        with tempfile.TemporaryDirectory(prefix="photoredactor-ai-") as temp:
            root = Path(temp)
            input_path, mask_path, output_path = root / "input.png", root / "mask.png", root / "result.png"
            Image.fromarray(resized, "RGBA").convert("RGB").save(input_path)
            Image.fromarray(resized_mask.astype(np.uint8), "L").save(mask_path)
            backend = device if device.upper() == "CPU" else f"diffusion={device},clip=CPU,vae=CPU"
            command = [
                str(self.executable), "--model", str(self.model), "--init-img", str(input_path),
                "--mask", str(mask_path), "--prompt", prompt, "--negative-prompt", negative,
                "--output", str(output_path), "--width", str(width), "--height", str(height),
                "--steps", str(options.steps), "--cfg-scale", str(options.cfg_scale),
                "--strength", str(options.strength), "--sampling-method", SAMPLERS[options.sampler],
                "--seed", str(seed), "--rng", "cpu", "--backend", backend, "--mmap", "--vae-tiling", "--fa",
            ]
            if options.sampler == "LCM" and self.lora is not None:
                command.extend(("--lora-model-dir", str(self.lora.parent)))
                command[command.index("--prompt") + 1] = f"{prompt}<lora:{self.lora.stem}:1>"
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                creationflags=_creation_flags(),
            )
            while True:
                if self.cancel is not None and self.cancel.is_set():
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise GenerativeAPIError("Локальная генерация отменена")
                try:
                    stdout, stderr = process.communicate(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    continue
            if process.returncode != 0 or not output_path.is_file():
                details = (stderr or stdout).strip().splitlines()
                message = details[-1] if details else f"код {process.returncode}"
                raise GenerativeAPIError(f"Локальная генерация завершилась с ошибкой: {message}")
            with Image.open(output_path) as result:
                return np.asarray(result.convert("RGBA"), dtype=np.uint8)

    def inpaint(self, image: np.ndarray, mask: np.ndarray, prompt: str, negative: str, seed: int, style: str = "") -> np.ndarray:
        x1, y1, x2, y2 = _selection_region(mask)
        crop = image[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]
        styled_prompt = f"{prompt}, {style}" if style else prompt
        generated = self._run(crop, crop_mask, styled_prompt, negative, seed)
        generated = cv2.resize(generated, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4)
        generated[crop_mask == 0] = crop[crop_mask == 0]
        output = image.copy()
        output[y1:y2, x1:x2] = generated
        return output

    def outpaint(
        self, image: np.ndarray, margins: tuple[int, int, int, int], prompt: str,
        negative: str, seed: int, style: str = "",
    ) -> np.ndarray:
        canvas, mask = _outpaint_canvas(image, margins)
        styled_prompt = f"{prompt}, {style}" if style else prompt
        generated = self._run(canvas, mask, styled_prompt, negative, seed)
        generated = cv2.resize(generated, canvas.shape[1::-1], interpolation=cv2.INTER_LANCZOS4)
        return strict_outpaint_result(image, generated, margins)

    def variants(self, operation, seed: int, count: int) -> list[GeneratedVariant]:
        return [GeneratedVariant(operation(value), value) for value in variant_seeds(seed, count)]


__all__ = [
    "LocalGenerationOptions", "LocalImageClient", "PERFORMANCE_LABELS", "PERFORMANCE_VALUES",
    "SAMPLERS", "shutdown_local_servers",
]
