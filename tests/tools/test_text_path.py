from __future__ import annotations

import copy

import numpy as np

from uzyro.core import Document, text_path_point_at_distance, text_path_samples
from uzyro.history import History, TextDataCommand


def horizontal_points() -> list[list[float]]:
    return [[30.0, 110.0], [120.0, 110.0], [220.0, 110.0], [330.0, 110.0]]


def test_bezier_path_sampling_uses_arc_length_and_tangents() -> None:
    points = [[20.0, 150.0], [90.0, 20.0], [240.0, 220.0], [340.0, 80.0]]
    positions, tangents, cumulative = text_path_samples(points, 600)
    midpoint, tangent = text_path_point_at_distance(positions, tangents, cumulative, cumulative[-1] / 2.0)
    assert np.allclose(positions[0], points[0])
    assert np.allclose(positions[-1], points[-1])
    assert cumulative[-1] > np.linalg.norm(np.asarray(points[-1]) - np.asarray(points[0]))
    assert 20.0 < midpoint[0] < 340.0
    assert np.isclose(np.linalg.norm(tangent), 1.0, atol=1e-5)


def test_text_is_rotated_and_laid_out_on_selected_path_segment() -> None:
    document = Document.new(360, 220, (0, 0, 0, 0))
    layer = document.add_text_layer(
        "Контур текста",
        20,
        20,
        (30, 40, 50, 255),
        30,
        path_mode="bezier",
        path_points=[[30, 160], [110, 15], [250, 210], [330, 70]],
        path_start=0.12,
        path_end=0.86,
    )
    ys, xs = np.where(layer.pixels[:, :, 3] > 0)
    assert len(xs) > 500
    assert xs.min() >= 20
    assert xs.max() <= 340
    assert ys.max() - ys.min() > 35


def test_text_can_switch_between_both_sides_and_reverse_direction() -> None:
    document = Document.new(380, 220, (0, 0, 0, 0))
    layer = document.add_text_layer(
        "Две стороны",
        30,
        40,
        (10, 20, 30, 255),
        28,
        path_mode="bezier",
        path_points=horizontal_points(),
        path_side=1,
    )
    upper = layer.pixels.copy()
    upper_y = float(np.where(upper[:, :, 3] > 0)[0].mean())
    document.edit_text_layer(path_side=-1, path_reverse=True)
    lower = layer.pixels.copy()
    lower_y = float(np.where(lower[:, :, 3] > 0)[0].mean())
    assert lower_y > upper_y + 10
    assert not np.array_equal(upper, lower)


def test_text_path_data_roundtrips_through_project(tmp_path) -> None:
    document = Document.new(420, 260, (255, 255, 255, 255))
    layer = document.add_text_layer(
        "Сохраняемый путь",
        25,
        45,
        (60, 70, 80, 255),
        32,
        path_mode="bezier",
        path_points=[[25, 130], [120, 35], [290, 210], [395, 100]],
        path_start=0.17,
        path_end=0.83,
        path_side=-1,
        path_reverse=True,
        baseline_shift=7,
    )
    expected = copy.deepcopy(layer.text_data)
    path = tmp_path / "text-path.prdx"
    document.save_project(path)
    restored = Document.open_project(path)
    assert restored.layer.text_data == expected
    assert np.array_equal(restored.layer.pixels, layer.pixels)


def test_text_path_edit_has_compact_undo_and_redo() -> None:
    document = Document.new(360, 220, (0, 0, 0, 0))
    layer = document.add_text_layer("Undo", 20, 30, (0, 0, 0, 255), 30)
    before = copy.deepcopy(layer.text_data)
    document.edit_text_layer(
        path_mode="bezier",
        path_points=horizontal_points(),
        path_start=0.2,
        path_end=0.75,
        path_side=-1,
    )
    after = copy.deepcopy(layer.text_data)
    history = History()
    history.push(TextDataCommand("Изменить контур текста", layer.id, before, after, layer.name, layer.name))
    history.undo(document)
    assert layer.text_data == before
    history.redo(document)
    assert layer.text_data == after
    assert history.memory_bytes < 20000


def test_text_box_transform_scales_editable_path_points() -> None:
    document = Document.new(480, 300, (0, 0, 0, 0))
    layer = document.add_text_layer(
        "Трансформация",
        40,
        60,
        (10, 20, 30, 255),
        34,
        path_mode="bezier",
        path_points=[[40, 160], [150, 50], [300, 230], [430, 120]],
    )
    before = np.asarray(layer.text_data["path_points"], dtype=np.float64)
    ys, xs = np.where(layer.pixels[:, :, 3] > 0)
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    assert document.transform_active_text_box(25, 35, width * 2, height * 2)
    after = np.asarray(layer.text_data["path_points"], dtype=np.float64)
    assert after.shape == (4, 2)
    assert np.ptp(after[:, 0]) > np.ptp(before[:, 0]) * 1.8
    assert np.ptp(after[:, 1]) > np.ptp(before[:, 1]) * 1.8
    assert np.count_nonzero(layer.pixels[:, :, 3]) > 500
