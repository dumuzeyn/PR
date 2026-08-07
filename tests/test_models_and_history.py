import copy

import numpy as np

from photoredactor.core import Document, Layer
from photoredactor.history import History, LayersDeleteCommand, ShapeDataCommand


def test_color_range_is_global_and_perceptual() -> None:
    pixels = np.full((8, 10, 4), (20, 20, 20, 255), dtype=np.uint8)
    pixels[1, 1] = (200, 40, 30, 255)
    pixels[6, 8] = (202, 42, 31, 255)
    document = Document(10, 8, layers=[Layer("Colors", pixels)])
    document.color_range_selection(document.layer, 1, 1, 0)
    assert document.selection_mask[1, 1] == 255
    assert document.selection_mask[6, 8] == 0
    document.color_range_selection(document.layer, 1, 1, 8)
    assert document.selection_mask[6, 8] == 255
    document.color_range_selection(document.layer, 1, 1, 120)
    assert np.count_nonzero(document.selection_mask) >= 2


def test_crop_changes_document_only_when_called() -> None:
    document = Document.new(20, 16)
    document.set_rect_selection((2, 3, 15, 12))
    before = document.layers[0].pixels.copy()
    assert document.width == 20 and document.height == 16
    assert np.array_equal(document.layers[0].pixels, before)
    document.crop((2, 3, 15, 12))
    assert (document.width, document.height) == (13, 9)
    assert document.layers[0].pixels.shape[:2] == (9, 13)


def test_text_data_survives_project_roundtrip(tmp_path) -> None:
    document = Document.new(180, 120, (0, 0, 0, 0))
    layer = document.add_text_layer(
        "Первая строка\nВторая строка", 12, 18, (10, 30, 220, 255), 24,
        "Arial", 130, "center", 8, 3, True, True, True, rotation=12.5,
    )
    path = tmp_path / "text.prdx"
    document.save_project(path)
    restored = Document.open_project(path)
    loaded = restored.get_layer(layer.id)
    assert loaded is not None and loaded.kind == "text"
    assert loaded.text_data["box_width"] == 130
    assert loaded.text_data["align"] == "center"
    assert loaded.text_data["tracking"] == 3
    assert loaded.text_data["rotation"] == 12.5
    assert loaded.text_data["text"].startswith("Первая")


def test_multi_layer_delete_is_one_undoable_command() -> None:
    document = Document.new(16, 16)
    document.add_layer("Two")
    document.add_layer("Three")
    removed = [(1, copy.deepcopy(document.layers[1])), (2, copy.deepcopy(document.layers[2]))]
    command = LayersDeleteCommand("Delete layers", removed, document.layers[2].id)
    command.redo(document)
    assert len(document.layers) == 1
    command.undo(document)
    assert [layer.name for layer in document.layers] == ["Background", "Two", "Three"]


def test_shape_data_command_rebuilds_vector_layer() -> None:
    document = Document.new(64, 64, (0, 0, 0, 0))
    layer = document.add_shape_layer("ellipse", (8, 8, 30, 30), (255, 0, 0, 255))
    before = copy.deepcopy(layer.shape_data)
    after = copy.deepcopy(before)
    after["box"] = [20, 20, 50, 50]
    command = ShapeDataCommand("Move shape", layer.id, before, after, layer.name, layer.name)
    command.redo(document)
    assert document.get_layer(layer.id).pixels[35, 35, 3] > 0
    command.undo(document)
    assert document.get_layer(layer.id).pixels[35, 35, 3] == 0
