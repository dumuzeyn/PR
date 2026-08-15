from __future__ import annotations

import numpy as np

from uzyro.core import Document, draw_brush, selection_contour_points


def test_ellipse_selection_does_not_select_bounding_box_corners() -> None:
    document = Document.new(100, 100)
    document.set_ellipse_selection((20, 20, 80, 80))
    mask = document.selection_mask
    assert mask is not None
    assert mask[50, 50] == 255
    assert mask[20, 20] == 0
    assert mask[50, 21] > 0


def test_polygon_and_lasso_keep_irregular_shape() -> None:
    document = Document.new(100, 100)
    document.set_polygon_selection([(10, 10), (90, 20), (45, 90)])
    mask = document.selection_mask
    assert mask is not None
    assert mask[40, 45] > 0
    assert mask[80, 80] == 0


def test_add_subtract_and_intersect_operate_on_masks() -> None:
    document = Document.new(120, 100)
    document.set_ellipse_selection((10, 20, 70, 80))
    document.set_ellipse_selection((50, 20, 110, 80), "add")
    assert document.selection_mask is not None
    assert document.selection_mask[20, 10] == 0
    assert document.selection_mask[50, 30] > 0
    assert document.selection_mask[50, 90] > 0

    document.set_ellipse_selection((40, 30, 80, 70), "subtract")
    assert document.selection_mask[50, 60] == 0
    assert document.selection_mask[50, 30] > 0

    document.set_ellipse_selection((10, 20, 70, 80), "intersect")
    assert document.selection_mask[50, 90] == 0
    assert document.selection_mask[50, 30] > 0


def test_feather_preserves_intermediate_coverage() -> None:
    document = Document.new(80, 80)
    document.set_ellipse_selection((15, 15, 65, 65), feather=4)
    mask = document.selection_mask
    assert mask is not None
    assert np.any((mask > 0) & (mask < 255))


def test_magic_wand_does_not_fill_its_bounding_box() -> None:
    document = Document.new(40, 40, (0, 0, 0, 255))
    layer = document.layer
    layer.pixels[5:30, 5:10, :3] = 200
    layer.pixels[25:30, 5:30, :3] = 200
    document.magic_wand_selection(layer, 7, 7, 0)
    assert document.selection_mask is not None
    assert document.selection_mask[7, 7] == 255
    assert document.selection_mask[20, 20] == 0


def test_magic_wand_contiguous_option_controls_disconnected_colors() -> None:
    document = Document.new(40, 40, (0, 0, 0, 255))
    layer = document.layer
    layer.pixels[4:10, 4:10, :3] = 180
    layer.pixels[28:34, 28:34, :3] = 180
    document.magic_wand_selection(layer, 6, 6, 0, contiguous=True)
    assert document.selection_mask is not None and document.selection_mask[30, 30] == 0
    document.magic_wand_selection(layer, 6, 6, 0, contiguous=False)
    assert document.selection_mask is not None and document.selection_mask[30, 30] == 255


def test_selection_contours_include_disconnected_areas_and_holes() -> None:
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[10:40, 10:40] = 255
    mask[20:30, 20:30] = 0
    mask[45:70, 60:90] = 255
    contours = selection_contour_points(mask)
    assert len(contours) == 3


def test_brush_respects_exact_ellipse_and_feather_coverage() -> None:
    document = Document.new(80, 80, (255, 255, 255, 255))
    document.set_ellipse_selection((15, 15, 65, 65), feather=3)
    selection = document.layer_selection_mask(document.layer)
    before = document.layer.pixels.copy()
    draw_brush(document.layer, 40, 40, 40, (0, 0, 0, 255), 1.0, selection_mask=selection)
    assert np.array_equal(document.layer.pixels[15, 15], before[15, 15])
    assert np.all(document.layer.pixels[40, 40, :3] == 0)
    edge = document.layer.pixels[40, 14, 0]
    assert 0 < int(edge) < 255
