from __future__ import annotations

from pathlib import Path

import numpy as np

from photoredactor.core import Document, mesh_warp_pixels, perspective_warp_pixels, render_shape_layer, render_text_layer


def sample_pixels(width: int = 96, height: int = 72) -> np.ndarray:
    yy, xx = np.indices((height, width))
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    pixels[:, :, 0] = (xx * 3) % 255
    pixels[:, :, 1] = (yy * 4) % 255
    pixels[:, :, 2] = 120
    pixels[:, :, 3] = 255
    return pixels


def test_perspective_warp_preserves_rgba_and_reports_offset() -> None:
    source = sample_pixels()
    corners = [(-8, 6), (103, 0), (91, 81), (4, 67)]
    output, offset = perspective_warp_pixels(source, corners)
    assert offset == (-8, 0)
    assert output.shape[1] == 112
    assert output.shape[0] == 82
    assert np.count_nonzero(output[:, :, 3]) > source.shape[0] * source.shape[1] * 0.7


def test_mesh_warp_uses_all_editable_nodes() -> None:
    source = sample_pixels()
    points = []
    for row, y in enumerate(np.linspace(0, source.shape[0] - 1, 4)):
        for column, x in enumerate(np.linspace(0, source.shape[1] - 1, 4)):
            points.append((x + (8 if row == 1 else -5 if row == 2 else 0), y + (6 if column in {1, 2} else 0)))
    output, _ = mesh_warp_pixels(source, points, 4, 4)
    assert output.shape != source.shape
    assert np.count_nonzero(output[:, :, 3]) > 0
    regular = [(x, y) for y in np.linspace(0, source.shape[0] - 1, 4) for x in np.linspace(0, source.shape[1] - 1, 4)]
    unwarped, _ = mesh_warp_pixels(source, regular, 4, 4)
    assert not np.array_equal(output, unwarped)


def test_shape_stays_editable_after_perspective_and_roundtrip(tmp_path: Path) -> None:
    document = Document.new(260, 180, (0, 0, 0, 0))
    document.layers.clear()
    layer = document.add_shape_layer("ellipse", (55, 40, 190, 140), (220, 55, 42, 255))
    corners = [(48, 52), (205, 32), (184, 151), (63, 137)]
    document.set_active_layer_advanced_transform("perspective", corners)
    assert layer.kind == "shape"
    assert layer.shape_data is not None
    assert layer.transform_data["mode"] == "perspective"
    transformed_before = layer.pixels.copy()
    layer.shape_data["fill"] = [35, 210, 95, 255]
    render_shape_layer(layer)
    assert layer.kind == "shape"
    assert layer.transform_data is not None
    assert not np.array_equal(layer.pixels, transformed_before)

    project = tmp_path / "shape-transform.prdx"
    document.save_project(project)
    restored = Document.open_project(project)
    restored_layer = restored.layer
    assert restored_layer.kind == "shape"
    assert restored_layer.transform_data["points"] == layer.transform_data["points"]
    assert np.array_equal(restored_layer.transform_source, layer.transform_source)
    assert np.array_equal(restored_layer.pixels, layer.pixels)


def test_text_mesh_rerenders_and_reset_restores_editable_base() -> None:
    document = Document.new(300, 200, (0, 0, 0, 0))
    document.layers.clear()
    layer = document.add_text_layer("Editable", 45, 80, (245, 245, 245, 255), 32)
    visible = layer.pixels[:, :, 3] > 0
    ys, xs = np.where(visible)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
    points = []
    for row, y in enumerate(np.linspace(y1, y2, 4)):
        for column, x in enumerate(np.linspace(x1, x2, 4)):
            points.append((x, y - (12 if row == 1 and column in {1, 2} else 0)))
    document.set_active_layer_advanced_transform("mesh", points, 4, 4)
    assert layer.kind == "text"
    assert layer.transform_data["mode"] == "mesh"
    layer.text_data["text"] = "Changed"
    render_text_layer(layer)
    assert layer.transform_data is not None
    assert np.count_nonzero(layer.pixels[:, :, 3]) > 0
    assert document.reset_active_layer_advanced_transform()
    assert layer.transform_data is None
    assert layer.transform_source is None
    assert layer.kind == "text"
    assert layer.text_data["text"] == "Changed"
    assert layer.pixels.shape == (200, 300, 4)


def test_raster_transform_reset_is_exact() -> None:
    document = Document.new(96, 72, (0, 0, 0, 0))
    layer = document.layer
    layer.pixels = sample_pixels()
    original = layer.pixels.copy()
    corners = [(4, 2), (90, 8), (94, 68), (0, 71)]
    document.set_active_layer_advanced_transform("perspective", corners)
    assert not np.array_equal(layer.pixels, original)
    assert document.reset_active_layer_advanced_transform()
    assert np.array_equal(layer.pixels, original)


def test_selected_pixels_support_perspective_and_update_selection() -> None:
    document = Document.new(140, 100, (40, 90, 150, 255))
    layer = document.layer
    layer.pixels[30:65, 45:85, :3] = (225, 55, 40)
    original = layer.pixels.copy()
    selection = np.zeros((100, 140), dtype=np.uint8)
    selection[30:65, 45:85] = 255
    document.selection_mask = selection
    corners = [(62, 20), (111, 29), (104, 76), (58, 67)]
    assert document.transform_selected_pixels_advanced("perspective", corners)
    assert int(layer.pixels[40, 50, 3]) < int(original[40, 50, 3])
    assert document.selection_mask is not None
    assert int(document.selection_mask[45, 80]) > 0
    assert int(document.selection_mask[10, 10]) == 0


def test_adjustment_layer_transform_creates_and_resets_coverage_mask() -> None:
    document = Document.new(120, 90, (80, 100, 120, 255))
    document.add_adjustment_layer("Light", {"type": "brightness_contrast", "brightness": 25, "contrast": 1.0})
    layer = document.layer
    assert layer.mask is None
    document.set_active_layer_advanced_transform("perspective", [(18, 12), (101, 8), (108, 78), (12, 82)])
    assert layer.kind == "adjustment"
    assert layer.mask is not None
    assert np.count_nonzero(layer.mask) > 0
    assert np.count_nonzero(layer.mask == 0) > 0
    assert document.reset_active_layer_advanced_transform()
    assert layer.mask is None
