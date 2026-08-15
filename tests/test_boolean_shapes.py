from __future__ import annotations

from pathlib import Path

import numpy as np

from uzyro.core import (
    Document,
    boolean_shape_mask,
    shape_data_bounds,
    transform_shape_data_to_box,
)


def overlapping_document() -> Document:
    document = Document.new(100, 80, (0, 0, 0, 0))
    document.layers.clear()
    document.add_shape_layer("rectangle", (10, 10, 50, 50), (220, 40, 40, 255))
    document.add_shape_layer("rectangle", (30, 20, 70, 60), (40, 120, 220, 255))
    return document


def test_all_boolean_modes_have_correct_regions() -> None:
    expected = {
        "union": (True, True, True),
        "subtract": (True, False, False),
        "intersect": (False, True, False),
        "xor": (True, False, True),
    }
    for mode, values in expected.items():
        document = overlapping_document()
        data = document.boolean_shape_data_with_lower(mode)
        assert data is not None
        mask = boolean_shape_mask(data, (80, 100))
        assert (mask[15, 15] > 0, mask[30, 40] > 0, mask[55, 60] > 0) == values


def test_boolean_combination_respects_moved_layer_positions() -> None:
    document = overlapping_document()
    document.layers[0].x = 8
    document.layers[0].y = 4
    document.layers[1].x = -3
    document.layers[1].y = 7
    data = document.boolean_shape_data_with_lower("union")
    assert data is not None
    assert shape_data_bounds(data) == (18, 14, 67, 67)
    assert data["children"][0]["box"] == [18, 14, 58, 54]
    assert data["children"][1]["box"] == [27, 27, 67, 67]


def test_compound_resize_scales_every_source_contour() -> None:
    document = overlapping_document()
    data = document.boolean_shape_data_with_lower("union")
    assert data is not None
    resized = transform_shape_data_to_box(data, (20, 30, 140, 130))
    assert shape_data_bounds(resized) == (20, 30, 140, 130)
    first = shape_data_bounds(resized["children"][0])
    second = shape_data_bounds(resized["children"][1])
    assert first == (20, 30, 100, 110)
    assert second == (60, 50, 140, 130)
    assert np.count_nonzero(boolean_shape_mask(resized, (160, 180))) > 0


def test_disabled_and_reordered_sources_remain_editable() -> None:
    document = overlapping_document()
    data = document.boolean_shape_data_with_lower("subtract")
    assert data is not None
    data["children"][1]["_enabled"] = False
    mask = boolean_shape_mask(data, (80, 100))
    assert mask[30, 40] == 255
    data["children"][1]["_enabled"] = True
    data["children"].reverse()
    reversed_mask = boolean_shape_mask(data, (80, 100))
    assert reversed_mask[55, 60] == 255
    assert reversed_mask[15, 15] == 0


def test_nested_boolean_project_roundtrip_preserves_vectors(tmp_path: Path) -> None:
    document = overlapping_document()
    assert document.boolean_active_shape_with_lower("union")
    document.add_shape_layer("ellipse", (40, 15, 85, 65), (30, 200, 110, 255))
    assert document.boolean_active_shape_with_lower("subtract")
    before_data = document.layer.shape_data
    before_pixels = document.layer.pixels.copy()
    project = tmp_path / "boolean.prdx"
    document.save_project(project)
    restored = Document.open_project(project)
    assert restored.layer.shape_data == before_data
    assert np.array_equal(restored.layer.pixels, before_pixels)
    assert len(restored.layer.shape_data["children"]) == 2
    assert restored.layer.shape_data["children"][0]["shape"] == "boolean"
