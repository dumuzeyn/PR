from __future__ import annotations

import math

from photoredactor.core import (
    Document,
    layer_contains_point,
    resize_box_from_handle,
    shape_drag_is_meaningful,
    shape_geometry_from_drag,
    star_points,
    topmost_layer_at,
)


def test_ellipse_and_rectangle_normalize_all_drag_directions() -> None:
    for tool in ("ellipse_shape", "rect_shape"):
        for end in ((80, 70), (20, 70), (80, 10), (20, 10)):
            geometry = shape_geometry_from_drag(tool, (50, 40), end)
            x1, y1, x2, y2 = geometry["box"]
            assert x1 <= x2 and y1 <= y2


def test_shift_and_alt_shape_geometry() -> None:
    square = shape_geometry_from_drag("rect_shape", (50, 50), (80, 70), shift=True)
    assert square["box"] == (50, 50, 80, 80)
    centered = shape_geometry_from_drag("ellipse_shape", (50, 50), (70, 60), alt=True)
    assert centered["box"] == (30, 40, 70, 60)
    both = shape_geometry_from_drag("ellipse_shape", (50, 50), (70, 60), shift=True, alt=True)
    assert both["box"] == (30, 30, 70, 70)


def test_shift_line_snaps_to_45_degrees() -> None:
    geometry = shape_geometry_from_drag("line_shape", (10, 10), (47, 31), shift=True)
    x1, y1, x2, y2 = geometry["line"]
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    assert angle in {0.0, 45.0, 90.0}


def test_polygon_star_custom_and_bezier_return_real_geometry() -> None:
    polygon = shape_geometry_from_drag("polygon_shape", (10, 10), (90, 70), sides=8)
    star = shape_geometry_from_drag("star_shape", (10, 10), (90, 70), sides=7, inner_ratio=0.4)
    custom = shape_geometry_from_drag("custom_shape", (10, 10), (90, 70), custom_points=[[0, 0], [1, 0.5], [0, 1]])
    bezier = shape_geometry_from_drag("bezier_shape", (10, 10), (90, 70))
    assert len(polygon["points"]) == 8
    assert len(star["points"]) == 14
    assert len(custom["points"]) == 3
    assert len(bezier["points"]) == 65


def test_tiny_shape_is_rejected() -> None:
    assert not shape_drag_is_meaningful(shape_geometry_from_drag("ellipse_shape", (10, 10), (11, 11)))
    assert shape_drag_is_meaningful(shape_geometry_from_drag("ellipse_shape", (10, 10), (20, 20)))


def test_preview_geometry_matches_final_shape_data() -> None:
    geometry = shape_geometry_from_drag("star_shape", (90, 70), (20, 10), sides=8, inner_ratio=0.4)
    document = Document.new(120, 90, (0, 0, 0, 0))
    document.add_shape_layer("star", geometry["box"], (220, 30, 40, 255), (20, 60, 220, 255), 3, 8, 0.4)
    data = document.layer.shape_data
    assert data is not None
    assert tuple(data["box"]) == geometry["box"]
    assert data["sides"] == 8
    assert data["inner_ratio"] == 0.4
    assert star_points(tuple(data["box"]), data["sides"], data["inner_ratio"]) == geometry["points"]


def test_corner_creation_acceptance_coordinates() -> None:
    forward = shape_geometry_from_drag("ellipse_shape", (100, 100), (300, 200))
    reverse = shape_geometry_from_drag("ellipse_shape", (300, 200), (100, 100))
    assert forward["box"] == (100, 100, 300, 200)
    assert reverse["box"] == forward["box"]


def test_resize_handles_support_proportion_and_center_modifiers() -> None:
    assert resize_box_from_handle((10, 20, 110, 70), "se", (160, 100)) == (10, 20, 160, 100)
    proportional = resize_box_from_handle((10, 20, 110, 70), "se", (160, 80), keep_proportions=True)
    assert proportional == (10, 20, 160, 95)
    centered = resize_box_from_handle((10, 20, 110, 70), "e", (160, 45), from_center=True)
    assert centered == (-40, 20, 160, 70)


def test_real_shape_hit_testing_and_topmost_selection() -> None:
    document = Document.new(240, 180, (0, 0, 0, 0))
    document.layers.clear()
    ellipse = document.add_shape_layer("ellipse", (20, 20, 140, 100), (220, 40, 40, 255))
    rectangle = document.add_shape_layer("rectangle", (70, 45, 180, 130), (40, 120, 220, 255))
    assert layer_contains_point(ellipse, (80, 60), tolerance=0)
    assert not layer_contains_point(ellipse, (21, 21), tolerance=0)
    assert topmost_layer_at(document, (90, 60)) == 1
    rectangle.visible = False
    assert topmost_layer_at(document, (90, 60)) == 0
    ellipse.locked = True
    assert topmost_layer_at(document, (80, 60)) == 0


def test_thin_line_hit_testing_uses_screen_tolerance() -> None:
    document = Document.new(200, 120, (0, 0, 0, 0))
    document.layers.clear()
    line = document.add_shape_layer("line", (20, 20, 170, 90), (0, 0, 0, 0), (255, 255, 255, 255), 2)
    assert layer_contains_point(line, (95, 57), tolerance=4)
    assert not layer_contains_point(line, (95, 90), tolerance=4)
