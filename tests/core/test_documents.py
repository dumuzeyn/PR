from __future__ import annotations

import time

from uzyro.core import Document, render_shape_layer
from uzyro.document_manager import DocumentManager
from uzyro.history import History, LayerMoveCommand, ObjectStatesCommand
from uzyro.object_hit_testing import hit_test_document, hit_test_stack, layers_inside_box


def test_document_manager_keeps_history_selection_and_dirty_titles_independent() -> None:
    first = Document.new(120, 80, (255, 255, 255, 255))
    manager = DocumentManager(first, History())
    first_session = manager.active
    assert first_session is not None
    first_session.history.push(LayerMoveCommand("Первый шаг", first.layer.id, (0, 0), (4, 2)))
    first_session.selected_layer_ids = {first.layer.id}
    first.dirty = True

    second = Document.new(320, 200, (0, 0, 0, 0))
    second_session = manager.add(second)
    assert manager.active is second_session
    assert second_session.history.undo_stack == []
    assert second_session.display_title == "Новый документ 2"
    assert first_session.display_title.endswith("*")

    assert manager.activate(first_session.id).history.undo_stack[0].label == "Первый шаг"
    assert manager.close(first_session.id) is second_session


def test_unified_hit_test_respects_z_order_visibility_and_locking() -> None:
    document = Document.new(160, 120, (0, 0, 0, 0))
    document.layers.clear()
    bottom = document.add_shape_layer("rectangle", (20, 20, 110, 90), (230, 40, 40, 255))
    top = document.add_shape_layer("ellipse", (35, 30, 125, 105), (30, 120, 240, 255))

    hit = hit_test_document(document, (70, 60))
    assert hit is not None and hit.layer_id == top.id and hit.layer_index == 1
    assert [item.layer_id for item in hit_test_stack(document, (70, 60))] == [top.id, bottom.id]
    top.visible = False
    assert hit_test_document(document, (70, 60)).layer_id == bottom.id
    bottom.locked = True
    assert hit_test_document(document, (70, 60), include_locked=False) is None


def test_object_marquee_selects_only_fully_contained_editable_objects() -> None:
    document = Document.new(260, 180, (0, 0, 0, 0))
    document.layers.clear()
    first = document.add_shape_layer("rectangle", (20, 20, 70, 70), (255, 255, 255, 255))
    document.add_shape_layer("ellipse", (130, 80, 220, 160), (255, 255, 255, 255))
    selected = layers_inside_box(document, (5, 5, 100, 100))
    assert [layer.id for layer in selected] == [first.id]


def test_shape_fill_and_stroke_opacity_are_rendered_independently() -> None:
    document = Document.new(120, 100, (0, 0, 0, 0))
    document.layers.clear()
    layer = document.add_shape_layer("rectangle", (20, 20, 90, 75), (250, 40, 30, 255), (20, 220, 80, 255), 8)
    layer.shape_data.update(fill_opacity=0.25, stroke_opacity=0.75, stroke_enabled=True)
    render_shape_layer(layer)
    assert 55 <= int(layer.pixels[45, 50, 3]) <= 70
    assert int(layer.pixels[20, 50, 3]) > 150
    layer.shape_data["stroke_enabled"] = False
    render_shape_layer(layer)
    assert int(layer.pixels[15, 50, 3]) == 0


def test_group_object_transform_is_one_compact_undo_step() -> None:
    document = Document.new(180, 120, (0, 0, 0, 0))
    document.layers.clear()
    first = document.add_shape_layer("rectangle", (10, 10, 50, 50), (255, 255, 255, 255))
    second = document.add_shape_layer("ellipse", (80, 30, 130, 85), (255, 255, 255, 255))
    before = {
        first.id: {"x": first.x, "y": first.y, "shape_data": first.shape_data, "text_data": None},
        second.id: {"x": second.x, "y": second.y, "shape_data": second.shape_data, "text_data": None},
    }
    after = {
        first.id: {**before[first.id], "x": 12},
        second.id: {**before[second.id], "x": 82},
    }
    history = History()
    command = ObjectStatesCommand("Масштабировать объекты", before, after)
    history.push(command)
    command.redo(document)
    assert (first.x, second.x) == (12, 82)
    history.undo(document)
    assert (first.x, second.x) == (0, 0)
    assert len(history.redo_stack) == 1 and history.memory_bytes == 0


def test_hit_testing_remains_responsive_with_one_hundred_shapes() -> None:
    document = Document.new(800, 600, (0, 0, 0, 0))
    document.layers.clear()
    for index in range(100):
        x = (index % 10) * 70
        y = (index // 10) * 50
        document.add_shape_layer("rectangle", (x, y, x + 55, y + 40), (40, 120, 220, 255))

    started = time.perf_counter()
    for _ in range(25):
        for point in ((10, 10), (360, 260), (710, 510), (799, 599)):
            hit_test_document(document, point)
    elapsed = time.perf_counter() - started

    assert elapsed < 1.5
