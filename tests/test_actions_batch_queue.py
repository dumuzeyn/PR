from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from photoredactor.automation import ACTION_FORMAT, ActionRecorder, ActionRunner
from photoredactor.batch_queue import BatchQueue
from photoredactor.core import Document
from photoredactor.history import LayerInsertCommand, LayerPropertyCommand, PixelPatchCommand, SelectionMaskCommand


def test_history_commands_record_and_replay_pixels_properties_and_selection(tmp_path: Path) -> None:
    source = Document.new(12, 10, (255, 255, 255, 255))
    recorder = ActionRecorder()
    recorder.start()

    before = source.layer.pixels[2:5, 3:7].copy()
    after = np.zeros_like(before)
    after[:, :, 0] = 240
    after[:, :, 3] = 255
    source.layer.pixels[2:5, 3:7] = after
    recorder.record_history_command(PixelPatchCommand("Штрих", source.layer.id, (3, 2, 7, 5), before, after), source)

    source.layer.opacity = 0.4
    recorder.record_history_command(LayerPropertyCommand("Непрозрачность", source.layer.id, "opacity", 1.0, 0.4), source)
    selection = np.zeros((10, 12), dtype=np.uint8)
    selection[1:6, 2:8] = 255
    source.selection_mask = selection
    recorder.record_history_command(SelectionMaskCommand("Выделение", None, selection), source)
    recorder.stop()

    path = tmp_path / "paint.json"
    recorder.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == ACTION_FORMAT

    target = Document.new(12, 10, (255, 255, 255, 255))
    report = ActionRunner().run_with_report(target, path)
    assert report.executed == 3
    assert not report.errors
    np.testing.assert_array_equal(target.layer.pixels[2:5, 3:7], after)
    np.testing.assert_array_equal(target.selection_mask, selection)
    assert target.layer.opacity == 0.4


def test_inserted_layer_is_mapped_for_following_recorded_steps() -> None:
    source = Document.new(8, 8)
    source.add_layer("Рисунок", np.zeros((8, 8, 4), dtype=np.uint8))
    inserted = source.layer
    recorder = ActionRecorder()
    recorder.start()
    recorder.record_history_command(LayerInsertCommand("Новый слой", 1, inserted), source)
    inserted.visible = False
    recorder.record_history_command(LayerPropertyCommand("Скрыть", inserted.id, "visible", True, False), source)
    recorder.stop()

    target = Document.new(8, 8)
    action = {"format": ACTION_FORMAT, "name": "Слои", "steps": [step.__dict__ for step in recorder.steps]}
    assert ActionRunner().run(target, action) == 2
    assert len(target.layers) == 2
    assert target.layers[1].name == "Рисунок"
    assert target.layers[1].visible is False


def test_conditions_stops_and_error_policies_are_reported() -> None:
    action = {
        "format": ACTION_FORMAT,
        "name": "Условия",
        "on_error": "stop",
        "steps": [
            {"command": "unknown", "label": "Ошибка", "on_error": "continue"},
            {
                "command": "set_bit_depth",
                "params": {"bit_depth": 16},
                "condition": {"field": "document.width", "operator": "gte", "value": 10},
            },
            {
                "command": "set_color_model",
                "params": {"color_model": "GRAY"},
                "condition": {"field": "source.extension", "operator": "eq", "value": ".raw"},
            },
            {"command": "stop", "params": {"message": "Проверка оператором"}},
            {"command": "set_bit_depth", "params": {"bit_depth": 32}},
        ],
    }
    document = Document.new(20, 12)
    report = ActionRunner().run_with_report(document, action, context={"source_extension": ".png"})
    assert report.executed == 1
    assert report.skipped == 1
    assert len(report.errors) == 1
    assert report.stopped and report.stop_message == "Проверка оператором"
    assert document.bit_depth == 16


def test_batch_queue_handles_collisions_errors_progress_cancel_and_persistence(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    output = tmp_path / "output"
    sources.mkdir()
    Image.new("RGBA", (16, 12), "red").save(sources / "one.png")
    Image.new("RGBA", (16, 12), "blue").save(sources / "two.png")
    (sources / "broken.png").write_bytes(b"not an image")
    output.mkdir()
    Image.new("RGBA", (2, 2), "white").save(output / "one.png")
    action = {
        "format": ACTION_FORMAT,
        "name": "Размер",
        "steps": [{"command": "resize_image", "params": {"width": 8, "height": 6}}],
    }

    queue = BatchQueue()
    job = queue.enqueue(action, sorted(sources.glob("*.png")), output, conflict="rename")
    queue.run_all()
    assert job.state == "completed_with_errors"
    assert job.completed_count == 3
    assert job.error_count == 1
    assert (output / "one_2.png").exists()
    assert Image.open(output / "one_2.png").size == (8, 6)

    manifest = tmp_path / "queue.json"
    queue.save(manifest)
    restored = BatchQueue()
    assert restored.load(manifest) == 1
    assert restored.jobs[0].error_count == 1

    cancel_queue = BatchQueue()
    cancel_job = cancel_queue.enqueue(action, [sources / "one.png", sources / "two.png"], tmp_path / "cancelled")

    def cancel_after_first(_job, item) -> None:
        if item.state == "completed":
            cancel_queue.cancel()

    cancel_queue.run_all(cancel_after_first)
    assert cancel_job.state == "cancelled"
    assert cancel_job.items[0].state == "completed"
    assert cancel_job.items[1].state == "cancelled"
