from __future__ import annotations

import copy
import gc
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
from PIL import Image

from uzyro.core import Document, Layer
from uzyro.project_tiles import is_tiled_array, read_tiled_array
from uzyro.rendering import RenderEngine


def patterned_rgba(width: int, height: int, offset: int = 0, alpha: int = 255) -> np.ndarray:
    yy, xx = np.indices((height, width), dtype=np.uint16)
    result = np.empty((height, width, 4), dtype=np.uint8)
    result[:, :, 0] = ((xx + offset) % 251).astype(np.uint8)
    result[:, :, 1] = ((yy * 3 + offset) % 241).astype(np.uint8)
    result[:, :, 2] = (((xx // 7) + (yy // 5) + offset) % 239).astype(np.uint8)
    result[:, :, 3] = alpha
    return result


@pytest.mark.security
def test_tiled_project_roundtrip_preserves_all_raster_payloads_and_rejects_invalid_hash(tmp_path: Path) -> None:
    document = Document.new(1100, 730, (0, 0, 0, 0))
    layer = document.layer
    layer.pixels = patterned_rgba(1100, 730)
    layer.mask = np.tile(np.linspace(0, 255, 1100, dtype=np.uint8), (730, 1))
    layer.smart_source = patterned_rgba(540, 360, 17)
    layer.transform_source = patterned_rgba(620, 410, 29)
    layer.transform_mask_source = np.full((410, 620), 173, dtype=np.uint8)
    document.selection_mask = np.zeros((730, 1100), dtype=np.uint8)
    document.selection_mask[90:640, 130:980] = 255
    document.saved_selections["Объект"] = document.selection_mask.copy()
    document.set_bit_depth(16)
    expected = {
        "pixels": layer.pixels.copy(),
        "mask": layer.mask.copy(),
        "working": layer.working_pixels.copy(),
        "smart": layer.smart_source.copy(),
        "transform": layer.transform_source.copy(),
        "transform_mask": layer.transform_mask_source.copy(),
        "selection": document.selection_mask.copy(),
    }
    project = tmp_path / "tiled-roundtrip.prdx"
    document.save_project(project)
    assert not (tmp_path / ".tiled-roundtrip.prdx.saving").exists()

    with zipfile.ZipFile(project, "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        raw = manifest["layers"][0]
        assert manifest["format_version"] == 3
        assert manifest["storage"]["format"] == "tiles-v1"
        assert all(is_tiled_array(raw[key]) for key in ("pixels", "mask", "working_pixels", "smart_source", "transform_source", "transform_mask_source"))
        assert len(raw["pixels"]["tiles"]) == 6
        broken = copy.deepcopy(raw["pixels"])
        broken["tiles"][0]["sha256"] = "0" * 64
        with pytest.raises(ValueError, match="checksum"):
            read_tiled_array(archive, broken)

    progress: list[tuple[int, int, str]] = []
    restored = Document.open_project(project, lambda current, total, name: progress.append((current, total, name)))
    np.testing.assert_array_equal(restored.layer.pixels, expected["pixels"])
    np.testing.assert_array_equal(restored.layer.mask, expected["mask"])
    np.testing.assert_array_equal(restored.layer.working_pixels, expected["working"])
    np.testing.assert_array_equal(restored.layer.smart_source, expected["smart"])
    np.testing.assert_array_equal(restored.layer.transform_source, expected["transform"])
    np.testing.assert_array_equal(restored.layer.transform_mask_source, expected["transform_mask"])
    np.testing.assert_array_equal(restored.selection_mask, expected["selection"])
    np.testing.assert_array_equal(restored.saved_selections["Объект"], expected["selection"])
    assert progress == [(1, 1, "Background")]
    info = restored.project_storage_info(project)
    assert info["format"] == "tiles-v1"
    assert info["tiles"] >= 20
    assert info["bytes"] > 10_000_000


def test_legacy_single_image_project_remains_readable(tmp_path: Path) -> None:
    pixels = patterned_rgba(73, 51, 11)
    buffer = io.BytesIO()
    Image.fromarray(pixels, "RGBA").save(buffer, "PNG")
    manifest = {
        "width": 73,
        "height": 51,
        "layers": [{"id": "legacy-layer", "name": "Legacy", "pixels": "layers/0000.png"}],
    }
    path = tmp_path / "legacy.prdx"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("layers/0000.png", buffer.getvalue())
        archive.writestr("manifest.json", json.dumps(manifest))
    restored = Document.open_project(path)
    assert restored.layer.id == "legacy-layer"
    np.testing.assert_array_equal(restored.layer.pixels, pixels)
    assert restored.project_storage_info(path)["format"] == "legacy"


def test_large_multilayer_project_loads_and_renders_reduced_without_full_composite(tmp_path: Path) -> None:
    width, height = 4096, 3072
    colors = [(22, 35, 48, 255), (210, 55, 42, 84), (40, 170, 95, 72), (50, 105, 225, 64)]
    document = Document.new(width, height, colors[0])
    for index, color in enumerate(colors[1:], 1):
        document.layers.append(Layer(f"Large {index}", np.full((height, width, 4), color, dtype=np.uint8)))
    document.active_layer = len(document.layers) - 1
    project = tmp_path / "large-multilayer.prdx"
    document.save_project(project)
    expected_bytes = width * height * 4 * len(colors)
    del document
    gc.collect()

    loaded_layers: list[str] = []
    restored = Document.open_project(project, lambda _current, _total, name: loaded_layers.append(name))
    assert (restored.width, restored.height) == (width, height)
    assert len(restored.layers) == len(colors)
    assert loaded_layers == ["Background", "Large 1", "Large 2", "Large 3"]
    info = restored.project_storage_info(project)
    assert info["bytes"] == expected_bytes
    assert info["tiles"] == 48 * len(colors)

    engine = RenderEngine(tile_size=256)
    reduced, level = engine.render_for_zoom(restored, 0.08, checker=True)
    assert level == 3
    assert reduced.shape == (384, 512, 4)
    assert reduced.nbytes < 1_000_000
    assert engine._composites == {}
    assert engine.cache_status()["memory_bytes"] < 32 * 1024 * 1024
    region = engine.composite_region(restored, (1800, 1200, 2056, 1456), checker=False)
    assert region.shape == (256, 256, 4)
    assert np.any(region[:, :, 3] > 0)
    engine.scratch.close()
