from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from photoredactor.automation import ActionRecorder, ActionRunner
from photoredactor.color_management import cmyk_to_rgb, lab_to_rgb, rgb_to_cmyk, rgb_to_lab
from photoredactor.core import Document, generative_expand_pixels
from photoredactor.large_document import MipmapPyramid, ScratchCache, gpu_status
from photoredactor.plugins import PluginRegistry


def sample_rgba(width: int = 32, height: int = 24) -> np.ndarray:
    yy, xx = np.indices((height, width))
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    pixels[:, :, 0] = (xx * 7) % 256
    pixels[:, :, 1] = (yy * 11) % 256
    pixels[:, :, 2] = ((xx + yy) * 5) % 256
    pixels[:, :, 3] = 255
    return pixels


def test_smart_object_uses_original_payload_and_roundtrips(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    original = sample_rgba()
    Image.fromarray(original, "RGBA").save(source)
    document = Document.new(100, 80, (0, 0, 0, 0))
    layer = document.place_image(source)
    document.transform_active_layer(width=9, height=7)
    document.transform_active_layer(width=32, height=24)
    assert np.array_equal(layer.pixels, original)

    project = tmp_path / "smart.prdx"
    document.save_project(project)
    restored = Document.open_project(project)
    assert restored.layer.smart_source is not None
    assert np.array_equal(restored.layer.smart_source, original)


def test_linked_object_reports_external_change(tmp_path: Path) -> None:
    source = tmp_path / "linked.png"
    Image.fromarray(sample_rgba(), "RGBA").save(source)
    document = Document.new(64, 64)
    document.place_image(source, linked=True)
    assert document.linked_layer_status()["status"] == "current"
    changed = sample_rgba()
    changed[0, 0, 0] ^= 255
    Image.fromarray(changed, "RGBA").save(source)
    assert document.linked_layer_status()["status"] == "modified"
    source.unlink()
    assert document.linked_layer_status()["status"] == "missing"


def test_color_models_and_high_precision_project_data(tmp_path: Path) -> None:
    original = sample_rgba(20, 16)
    lab_roundtrip = lab_to_rgb(rgb_to_lab(original))
    cmyk_roundtrip = cmyk_to_rgb(rgb_to_cmyk(original))
    normalized = original.astype(np.float32) / 255.0
    assert np.max(np.abs(lab_roundtrip - normalized)) < 0.03
    assert np.max(np.abs(cmyk_roundtrip - normalized)) < 0.002

    document = Document.new(20, 16)
    document.layer.pixels = original
    document.set_bit_depth(16)
    document.set_color_model("Lab")
    lab_working = document.layer.working_pixels.copy()
    document.assign_color_profile("sRGB")
    project = tmp_path / "color.prdx"
    document.save_project(project)
    restored = Document.open_project(project)
    assert restored.bit_depth == 16
    assert restored.color_model == "Lab"
    assert restored.layer.working_pixels is not None
    assert restored.layer.working_model == "Lab"
    assert restored.layer.working_pixels.dtype == np.float32
    np.testing.assert_allclose(restored.layer.working_pixels, lab_working)
    assert restored.metadata["color_management"]["profile_name"] == "sRGB"


def test_scratch_cache_spills_and_mipmap_selects_level(tmp_path: Path) -> None:
    cache = ScratchCache(memory_limit=600, directory=tmp_path / "scratch")
    first = np.arange(400, dtype=np.uint8).reshape(20, 20)
    second = np.full((20, 20), 7, dtype=np.uint8)
    cache.put("first", first)
    cache.put("second", second)
    assert cache.stats["disk_items"] >= 1
    assert np.array_equal(cache.get("first"), first)
    pyramid = MipmapPyramid(cache)
    image = sample_rgba(64, 64)
    reduced, level = pyramid.for_zoom("image", image, 0.2)
    assert level == 2
    assert reduced.shape[:2] == (16, 16)
    assert {"available", "enabled", "devices", "backend", "device", "mode"} <= set(gpu_status())
    cache.close()


def test_actions_are_replayable_and_batchable(tmp_path: Path) -> None:
    recorder = ActionRecorder()
    recorder.start()
    recorder.record("resize_image", {"width": 12, "height": 10}, "Размер")
    recorder.record("set_bit_depth", {"bit_depth": 16}, "Глубина")
    recorder.stop()
    action = tmp_path / "action.json"
    recorder.save(action)
    data = json.loads(action.read_text(encoding="utf-8"))
    assert data["format"] == "PhotoRedactor action v3"
    document = Document.new(30, 20)
    assert ActionRunner().run(document, action) == 2
    assert (document.width, document.height, document.bit_depth) == (12, 10, 16)


def test_python_filter_plugin_is_discovered_and_validated(tmp_path: Path) -> None:
    plugin = tmp_path / "invert.py"
    plugin.write_text(
        "import numpy as np\n"
        "def register(api):\n"
        "    def invert(pixels, params):\n"
        "        result = pixels.copy(); result[:, :, :3] = 255 - result[:, :, :3]; return result\n"
        "    api.register_filter('Инверсия плагина', invert, 'test')\n",
        encoding="utf-8",
    )
    registry = PluginRegistry([tmp_path])
    assert registry.discover() == 1
    original = sample_rgba(8, 6)
    result = registry.apply_filter("Инверсия плагина", original)
    assert np.array_equal(result[:, :, :3], 255 - original[:, :, :3])
    assert np.array_equal(result[:, :, 3], original[:, :, 3])


def test_generative_expand_preserves_original_and_document_layers() -> None:
    original = sample_rgba(18, 12)
    expanded = generative_expand_pixels(original, 5, 4, 7, 3, "content-aware")
    assert expanded.shape == (19, 30, 4)
    assert np.array_equal(expanded[4:16, 5:23], original)
    assert np.all(expanded[:, :, 3] == 255)

    document = Document.new(18, 12, (0, 0, 0, 0))
    document.layer.pixels = original.copy()
    old_id = document.layer.id
    document.generative_expand(5, 4, 7, 3)
    assert (document.width, document.height) == (30, 19)
    assert document.layers[1].id == old_id
    assert (document.layers[1].x, document.layers[1].y) == (5, 4)
    assert np.array_equal(document.composite(False)[4:16, 5:23], original)
