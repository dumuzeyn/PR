from __future__ import annotations

import numpy as np

from photoredactor.core import Document, GradientEngine, editable_bezier_nodes, render_shape_layer, shape_data_bounds, transform_shape_data_to_box
from photoredactor.history import History, LayerGroupMoveCommand


BLACK = (0, 0, 0, 255)
WHITE = (255, 255, 255, 255)


def alpha_bounds(pixels: np.ndarray) -> tuple[int, int]:
    ys, xs = np.where(pixels[:, :, 3] > 0)
    return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def test_gradient_supports_color_opacity_midpoint_reverse_and_dither() -> None:
    stops = [
        {"position": 0.0, "color": [255, 0, 0, 255], "midpoint": 0.25},
        {"position": 0.5, "color": [0, 255, 0, 255], "midpoint": 0.75},
        {"position": 1.0, "color": [0, 0, 255, 255]},
    ]
    opacity = [{"position": 0.0, "opacity": 0.0}, {"position": 1.0, "opacity": 1.0}]
    normal = GradientEngine.render(101, 3, (0, 1), (100, 1), stops, opacity_stops=opacity)
    reversed_image = GradientEngine.render(101, 3, (0, 1), (100, 1), stops, opacity_stops=opacity, reverse=True)
    dithered = GradientEngine.render(101, 3, (0, 1), (100, 1), stops, opacity_stops=opacity, dither=True)
    assert tuple(normal[1, 0, :3]) == (255, 0, 0)
    assert tuple(normal[1, -1, :3]) == (0, 0, 255)
    assert normal[1, 0, 3] == 0 and normal[1, -1, 3] == 255
    assert tuple(reversed_image[1, 0, :3]) == (0, 0, 255)
    assert reversed_image[1, 0, 3] == 255
    assert normal[1, 13, 1] >= normal[1, 13, 0]
    assert not np.array_equal(normal, dithered)
    assert np.array_equal(dithered, GradientEngine.render(101, 3, (0, 1), (100, 1), stops, opacity_stops=opacity, dither=True))


def test_shape_appearance_rotation_rounding_and_stroke_alignment() -> None:
    document = Document.new(120, 100, (0, 0, 0, 0))
    layer = document.add_shape_layer("rectangle", (25, 25, 85, 70), (30, 140, 230, 255), (230, 40, 50, 255), 6)
    data = layer.shape_data
    assert data is not None
    data.update({"corner_radius": 14, "stroke_alignment": "outside", "opacity": 0.5, "rotation": 24.0})
    render_shape_layer(layer)
    assert layer.pixels[25, 25, 3] < 128
    assert 110 <= int(layer.pixels[:, :, 3].max()) <= 128
    assert np.count_nonzero(layer.pixels[:, :, 3]) > 1500


def test_editable_bezier_nodes_change_the_rendered_path() -> None:
    document = Document.new(180, 130, (0, 0, 0, 0))
    layer = document.add_shape_layer("bezier", (15, 20, 165, 110), (0, 0, 0, 0), WHITE, 4)
    nodes = editable_bezier_nodes(layer.shape_data or {})
    nodes.insert(1, {"anchor": [90, 35], "in": [60, 15], "out": [120, 85], "linked": False})
    layer.shape_data["path_nodes"] = nodes
    render_shape_layer(layer)
    first = layer.pixels.copy()
    layer.shape_data["path_nodes"][1]["anchor"] = [90, 100]
    render_shape_layer(layer)
    assert len(editable_bezier_nodes(layer.shape_data)) == 3
    assert not np.array_equal(first, layer.pixels)


def test_rotated_shape_and_editable_nodes_resize_without_losing_rotation() -> None:
    data = {
        "shape": "bezier", "box": [20, 30, 140, 90], "rotation": 25.0,
        "path_nodes": [
            {"anchor": [20, 90], "in": [20, 90], "out": [55, 20], "linked": True},
            {"anchor": [140, 30], "in": [105, 100], "out": [140, 30], "linked": True},
        ],
    }
    before = shape_data_bounds(data)
    assert before is not None
    target = (before[0] - 30, before[1] - 20, before[2] + 30, before[3] + 20)
    resized = transform_shape_data_to_box(data, target)
    after = shape_data_bounds(resized)
    assert resized["rotation"] == 25.0
    assert len(resized["path_nodes"]) == 2
    assert after is not None
    assert all(abs(actual - expected) <= 2 for actual, expected in zip(after, target))


def test_group_ids_roundtrip_and_group_move_is_one_compact_history_step(tmp_path) -> None:
    document = Document.new(40, 30, (0, 0, 0, 0))
    first = document.layer
    document.add_layer("Second")
    second = document.layer
    first.group_id = second.group_id = "group-a"
    path = tmp_path / "groups.prdx"
    document.save_project(path)
    restored = Document.open_project(path)
    assert [layer.group_id for layer in restored.layers] == ["group-a", "group-a"]
    before = {layer.id: (layer.x, layer.y) for layer in restored.layers}
    after = {layer.id: (layer.x + 12, layer.y - 3) for layer in restored.layers}
    command = LayerGroupMoveCommand("Переместить группу", before, after)
    history = History()
    history.push(command)
    command.redo(restored)
    assert [(layer.x, layer.y) for layer in restored.layers] == list(after.values())
    history.undo(restored)
    assert [(layer.x, layer.y) for layer in restored.layers] == list(before.values())
    history.redo(restored)
    assert history.memory_bytes < 1000


def test_advanced_typography_stays_editable_and_renders_vertical_text(tmp_path) -> None:
    document = Document.new(420, 300, (0, 0, 0, 0))
    layer = document.add_text_layer("Первый абзац для проверки\nВторой абзац", 20, 20, BLACK, 30, box_width=330)
    before_width, _ = alpha_bounds(layer.pixels)
    document.edit_text_layer(
        align="justify", kerning=2, horizontal_scale=160, vertical_scale=85,
        strike_through=True, indent_left=14, indent_right=10,
        first_line_indent=18, spacing_before=6, spacing_after=9,
    )
    scaled_width, _ = alpha_bounds(layer.pixels)
    assert scaled_width > before_width
    assert layer.text_data["align"] == "justify"
    assert layer.text_data["strike_through"] is True
    path = tmp_path / "typography.prdx"
    document.save_project(path)
    restored = Document.open_project(path)
    assert restored.layer.text_data == layer.text_data
    restored.edit_text_layer(text="ВЕРТИКАЛЬ", vertical=True, horizontal_scale=100, vertical_scale=100, box_width=0)
    width, height = alpha_bounds(restored.layer.pixels)
    assert height > width * 2
