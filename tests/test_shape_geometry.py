from __future__ import annotations

import math

from photoredactor.core import Document, shape_drag_is_meaningful, shape_geometry_from_drag, star_points


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
