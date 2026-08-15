from __future__ import annotations

import math

import numpy as np

from photoredactor.core import Document, editable_bezier_path_points, render_shape_layer, shape_data_to_mask, translated_shape_data
from photoredactor.vector_geometry import (
    adaptive_cubic_bezier,
    cubic_bezier_point,
    nearest_cubic_parameter,
    point_to_polyline_distance,
    split_cubic_bezier,
)
from photoredactor.vector_rasterizer import stroke_polyline_mask


def cubic_point(points, amount: float) -> tuple[float, float]:
    inverse = 1.0 - amount
    weights = (inverse**3, 3 * inverse * inverse * amount, 3 * inverse * amount * amount, amount**3)
    return tuple(sum(float(points[index][axis]) * weights[index] for index in range(4)) for axis in range(2))


def test_adaptive_bezier_uses_few_segments_for_lines_and_respects_error() -> None:
    straight = [(0, 0), (30, 0), (70, 0), (100, 0)]
    curved = [(0, 0), (0, 100), (100, 100), (100, 0)]
    assert len(adaptive_cubic_bezier(straight, 0.25)) == 2
    flattened = adaptive_cubic_bezier(curved, 0.25)
    assert len(flattened) > 12
    error = max(point_to_polyline_distance(cubic_point(curved, value / 1000), flattened) for value in range(1001))
    assert error <= 0.3


def test_bezier_tessellation_tracks_zoom_in_screen_space() -> None:
    data = {
        "shape": "bezier", "box": [0, 0, 300, 200],
        "path_nodes": [
            {"anchor": [0, 180], "in": [0, 180], "out": [20, -80], "linked": True},
            {"anchor": [300, 20], "in": [260, 280], "out": [300, 20], "linked": True},
        ],
    }
    low_zoom = editable_bezier_path_points(data, zoom=0.25)
    high_zoom = editable_bezier_path_points(data, zoom=4.0)
    assert len(low_zoom) < len(high_zoom)


def test_nearest_parameter_and_de_casteljau_split_preserve_curve() -> None:
    control = [(0, 80), (25, -30), (105, 150), (140, 20)]
    target = cubic_bezier_point(control, 0.37)
    amount, distance = nearest_cubic_parameter(control, target)
    assert abs(amount - 0.37) < 0.002 and distance < 0.01
    left, right = split_cubic_bezier(control, amount)
    original = [cubic_bezier_point(control, value / 100) for value in range(101)]
    split = adaptive_cubic_bezier(left, 0.05) + adaptive_cubic_bezier(right, 0.05)[1:]
    assert max(point_to_polyline_distance(point, split) for point in original) < 0.08


def test_supersampled_rotated_shape_has_fractional_edge_alpha() -> None:
    data = {"shape": "ellipse", "box": [20, 15, 105, 75], "rotation": 31.0, "stroke_width": 0}
    mask = shape_data_to_mask(data, (100, 130))
    assert np.any(mask == 255)
    assert np.count_nonzero((mask > 0) & (mask < 255)) > 80


def test_line_caps_dash_and_joins_change_real_raster_geometry() -> None:
    butt = stroke_polyline_mask((80, 140), [(30, 40), (100, 40)], 10, cap="butt")
    round_cap = stroke_polyline_mask((80, 140), [(30, 40), (100, 40)], 10, cap="round")
    square = stroke_polyline_mask((80, 140), [(30, 40), (100, 40)], 10, cap="square")
    dashed = stroke_polyline_mask((80, 140), [(30, 40), (100, 40)], 10, cap="butt", dash_pattern=[8, 8])
    assert butt[40, 25] == 0
    assert round_cap[40, 25] > 200 and square[40, 25] > 200
    assert np.count_nonzero(dashed) < np.count_nonzero(butt) * 0.8

    points = [(20, 90), (60, 20), (100, 90)]
    bevel = stroke_polyline_mask((120, 120), points, 14, join="bevel")
    miter = stroke_polyline_mask((120, 120), points, 14, join="miter", miter_limit=10)
    assert np.where(miter > 0)[0].min() < np.where(bevel > 0)[0].min()


def test_vector_stroke_settings_render_and_survive_project_roundtrip(tmp_path) -> None:
    document = Document.new(180, 100, (0, 0, 0, 0))
    layer = document.add_shape_layer("line", (20, 50, 160, 50), (0, 0, 0, 0), (240, 70, 40, 255), 8)
    layer.shape_data.update({
        "stroke_cap": "square", "stroke_join": "miter", "miter_limit": 7.0,
        "dash_pattern": [10.0, 5.0], "dash_offset": 3.0,
    })
    render_shape_layer(layer)
    assert np.count_nonzero(layer.pixels[:, :, 3]) > 300
    path = tmp_path / "vector-stroke.prdx"
    document.save_project(path)
    restored = Document.open_project(path)
    for key in ("stroke_cap", "stroke_join", "miter_limit", "dash_pattern", "dash_offset"):
        assert restored.layer.shape_data[key] == layer.shape_data[key]


def test_translated_bezier_moves_nodes_and_handles_together() -> None:
    data = {
        "shape": "bezier", "box": [10, 20, 90, 80],
        "path_nodes": [
            {"anchor": [10, 70], "in": [8, 72], "out": [30, 15], "linked": True},
            {"anchor": [90, 30], "in": [70, 90], "out": [92, 28], "linked": True},
        ],
    }
    moved = translated_shape_data(data, 13, -7)
    assert moved["path_nodes"][0]["anchor"] == [23.0, 63.0]
    assert moved["path_nodes"][0]["out"] == [43.0, 8.0]
    original_distance = math.dist(data["path_nodes"][0]["anchor"], data["path_nodes"][0]["out"])
    assert math.isclose(math.dist(moved["path_nodes"][0]["anchor"], moved["path_nodes"][0]["out"]), original_distance)
