from __future__ import annotations

import copy

import numpy as np

from photoredactor.core import Document, Layer
from photoredactor.action_history import apply_history_payload, history_command_to_payload
from photoredactor.history import GeneratedExpandCommand, History


def test_generation_metadata_survives_project_roundtrip(tmp_path) -> None:
    document = Document.new(12, 9, (0, 0, 0, 0))
    document.layer.generation_data = {
        "operation": "fill",
        "provider": "stability-ai",
        "prompt": "красная чашка",
        "negative_prompt": "текст",
        "seed": 714,
        "style": "photographic",
        "variant_seeds": [714, 715, 716],
    }
    project = tmp_path / "generation.prdx"
    document.save_project(project)
    restored = Document.open_project(project)
    assert restored.layer.generation_data == document.layer.generation_data


def test_generated_expand_command_has_exact_undo_redo() -> None:
    document = Document.new(6, 4, (15, 25, 35, 255))
    document.add_layer("Foreground", np.full((2, 3, 4), (80, 90, 100, 255), dtype=np.uint8))
    document.layer.x, document.layer.y = 2, 1
    document.selection_mask = np.arange(24, dtype=np.uint8).reshape(4, 6)
    document.saved_selections["Объект"] = document.selection_mask.copy()
    before_layers = copy.deepcopy(document.layers)
    before_selection = document.selection_mask.copy()
    generated = Layer(
        "Генеративное расширение · 45",
        np.full((7, 11, 4), (180, 30, 20, 255), dtype=np.uint8),
        generation_data={"operation": "expand", "seed": 45, "margins": [2, 1, 3, 2]},
    )
    command = GeneratedExpandCommand(
        "Генеративное расширение", generated, (2, 1, 3, 2), (6, 4), document.active_layer,
        before_selection.copy(), {"Объект": before_selection.copy()},
    )
    history = History()
    command.redo(document)
    history.push(command)

    assert (document.width, document.height) == (11, 7)
    assert document.layers[0].id == generated.id
    assert [(layer.x, layer.y) for layer in document.layers[1:]] == [(2, 1), (4, 2)]
    np.testing.assert_array_equal(document.selection_mask[1:5, 2:8], before_selection)
    assert history.undo(document) == "Генеративное расширение"
    assert (document.width, document.height) == (6, 4)
    assert [layer.id for layer in document.layers] == [layer.id for layer in before_layers]
    assert [(layer.x, layer.y) for layer in document.layers] == [(layer.x, layer.y) for layer in before_layers]
    np.testing.assert_array_equal(document.selection_mask, before_selection)

    assert history.redo(document) == "Генеративное расширение"
    assert (document.width, document.height) == (11, 7)
    assert len(document.layers) == 3
    assert document.layers[0].id == generated.id
    np.testing.assert_array_equal(document.saved_selections["Объект"][1:5, 2:8], before_selection)


def test_generated_expand_metadata_and_pixels_survive_undo_redo() -> None:
    document = Document.new(3, 2, (5, 10, 15, 255))
    pixels = np.zeros((4, 6, 4), dtype=np.uint8)
    pixels[:, :, 0] = 210
    pixels[:, :, 3] = 255
    pixels[1:3, 1:4] = 0
    data = {"operation": "expand", "prompt": "лес", "seed": 991, "margins": [1, 1, 2, 1]}
    layer = Layer("Generated", pixels, generation_data=data)
    command = GeneratedExpandCommand("Expand", layer, (1, 1, 2, 1), (3, 2), 0, None, {})
    command.redo(document)
    command.undo(document)
    command.redo(document)
    restored = document.layers[0]
    assert restored.generation_data == data
    np.testing.assert_array_equal(restored.pixels, pixels)


def test_generated_expand_can_be_recorded_and_replayed() -> None:
    source = Document.new(4, 3, (10, 20, 30, 255))
    generated = Layer(
        "Generated",
        np.full((6, 8, 4), (90, 100, 110, 255), dtype=np.uint8),
        generation_data={"operation": "expand", "seed": 77, "margins": [1, 2, 3, 1]},
    )
    command = GeneratedExpandCommand("Expand", generated, (1, 2, 3, 1), (4, 3), 0, None, {})
    command.redo(source)
    payload = history_command_to_payload(command, source)

    target = Document.new(4, 3, (10, 20, 30, 255))
    apply_history_payload(target, payload, {})
    assert (target.width, target.height) == (8, 6)
    assert len(target.layers) == 2
    assert target.layers[0].generation_data["seed"] == 77
    assert (target.layers[1].x, target.layers[1].y) == (1, 2)
