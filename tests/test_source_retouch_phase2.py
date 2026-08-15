from __future__ import annotations

import cv2
import numpy as np

from uzyro.core import (
    BrushSettings,
    CloneHealingStroke,
    Document,
    Layer,
    SourceTransform,
    clone_or_heal,
    spot_heal,
)


def directional_source() -> np.ndarray:
    pixels = np.zeros((96, 96, 4), dtype=np.uint8)
    yy, xx = np.mgrid[:96, :96]
    pixels[:, :, 0] = xx * 2
    pixels[:, :, 1] = yy * 2
    pixels[:, :, 3] = 255
    return pixels


def test_clone_source_transform_changes_sample_geometry() -> None:
    source = directional_source()
    plain = Layer("plain", np.full_like(source, (20, 20, 20, 255)))
    rotated = Layer("rotated", plain.pixels.copy())
    clone_or_heal(plain, 24, 24, 64, 64, 10, source_pixels=source, transform=SourceTransform())
    clone_or_heal(
        rotated,
        24,
        24,
        64,
        64,
        10,
        source_pixels=source,
        transform=SourceTransform(rotation=90, flip_horizontal=True),
    )
    assert not np.array_equal(plain.pixels[56:73, 56:73], rotated.pixels[56:73, 56:73])
    assert int(plain.pixels[64, 70, 0]) > int(plain.pixels[64, 58, 0])
    assert int(rotated.pixels[70, 64, 0]) < int(rotated.pixels[58, 64, 0])


def test_clone_healing_stroke_uses_flow_and_one_tile_history() -> None:
    source = directional_source()
    layer = Layer("target", np.full_like(source, (110, 110, 110, 255)))
    settings = BrushSettings(9, 0.4, 0.8, 0.25)
    stroke = CloneHealingStroke(layer, settings, source, heal=True)
    for x in range(42, 67, 4):
        stroke.dab(x, 60, x - 24, 24)
    assert stroke.before_tiles
    assert len(stroke.before_tiles) <= 2
    assert np.any(layer.pixels != 110)
    assert abs(float(layer.pixels[54:67, 45:64, :3].mean()) - 110.0) < 30.0


def test_spot_healing_modes_are_distinct_and_reduce_defect() -> None:
    rng = np.random.default_rng(12)
    pixels = np.full((90, 90, 4), (155, 142, 126, 255), dtype=np.uint8)
    pixels[:, :, :3] = np.clip(pixels[:, :, :3].astype(np.int16) + rng.normal(0, 9, (90, 90, 1)), 0, 255)
    cv2.line(pixels, (34, 45), (56, 45), (8, 8, 8, 255), 5)
    proximity = Layer("proximity", pixels.copy())
    content = Layer("content", pixels.copy())
    before = float(np.abs(pixels[42:49, 38:53, :3].astype(np.int16) - 141).mean())
    spot_heal(proximity, 45, 45, 13, 1.0, hardness=0.5, mode="proximity_match")
    spot_heal(content, 45, 45, 13, 1.0, hardness=0.5, mode="content_aware")
    proximity_error = float(np.abs(proximity.pixels[42:49, 38:53, :3].astype(np.int16) - 141).mean())
    content_error = float(np.abs(content.pixels[42:49, 38:53, :3].astype(np.int16) - 141).mean())
    assert proximity_error < before
    assert content_error < before
    assert not np.array_equal(proximity.pixels, content.pixels)


def test_patch_structure_and_color_adaptation_change_result() -> None:
    document = Document.new(120, 80, (180, 145, 118, 255))
    checker = ((np.indices((24, 24)).sum(axis=0) % 2) * 50 + 35).astype(np.uint8)
    document.layer.pixels[8:32, 8:32, :3] = checker[:, :, None]
    document.layer.pixels[38:62, 72:96, :3] = 5
    selection = np.zeros((80, 120), dtype=np.uint8)
    cv2.rectangle(selection, (72, 38), (95, 61), 255, -1)
    document.selection_mask = selection
    low = document.layer.pixels.copy()
    assert document.patch_active_selection(8, 8, True, 1.0, structure=1)
    low_result = document.layer.pixels.copy()
    document.layer.pixels[:] = low
    assert document.patch_active_selection(8, 8, True, 10.0, structure=7)
    high_result = document.layer.pixels.copy()
    active = selection > 0
    assert not np.array_equal(low_result[active], high_result[active])
    assert float(high_result[:, :, :3][active].std()) > float(low_result[:, :, :3][active].std())
