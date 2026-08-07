from __future__ import annotations

import numpy as np

from photoredactor.core import Layer, RetouchStroke, clone_or_heal, local_retouch, retouch_falloff_mask, spot_heal


def textured_pixels(width: int = 128, height: int = 96) -> np.ndarray:
    rng = np.random.default_rng(42)
    pixels = rng.integers(35, 220, (height, width, 4), dtype=np.uint8)
    pixels[:, :, 3] = 255
    return pixels


def changed_amount(before: np.ndarray, after: np.ndarray) -> float:
    return float(np.abs(after[:, :, :3].astype(np.int16) - before[:, :, :3].astype(np.int16)).mean())


def test_retouch_falloff_is_soft_and_cached() -> None:
    mask = retouch_falloff_mask(20, 0.35)
    assert mask is retouch_falloff_mask(20, 0.35)
    assert mask[20, 20] == 1.0
    assert 0.0 < mask[20, 35] < mask[20, 20]
    assert mask[0, 0] == 0.0


def test_dodge_and_burn_low_exposure_do_not_clip_midtones() -> None:
    source = np.full((64, 64, 4), (128, 112, 96, 255), dtype=np.uint8)
    dodge = Layer("dodge", source.copy())
    burn = Layer("burn", source.copy())
    local_retouch(dodge, 32, 32, 18, "dodge", 0.1, hardness=0.5)
    local_retouch(burn, 32, 32, 18, "burn", 0.1, hardness=0.5)
    assert 128 < int(dodge.pixels[32, 32, 0]) < 170
    assert 80 < int(burn.pixels[32, 32, 0]) < 128
    assert np.all(dodge.pixels[:, :, 3] == 255)
    assert np.all(burn.pixels[:, :, 3] == 255)


def test_retouch_strength_is_monotonic() -> None:
    source = textured_pixels()
    for mode in ("blur", "sharpen", "dodge", "burn"):
        amounts = []
        for strength in (0.1, 0.5, 1.0):
            layer = Layer(mode, source.copy())
            local_retouch(layer, 64, 48, 20, mode, strength, hardness=0.55)
            amounts.append(changed_amount(source, layer.pixels))
        assert amounts[0] < amounts[1] < amounts[2], (mode, amounts)


def test_blur_radius_changes_area_not_center_strength() -> None:
    source = textured_pixels()
    centers = []
    changed_counts = []
    for radius in (10, 28):
        layer = Layer("blur", source.copy())
        local_retouch(layer, 64, 48, radius, "blur", 0.3, hardness=0.5)
        centers.append(layer.pixels[48, 64, :3].astype(np.int16))
        changed_counts.append(int(np.count_nonzero(np.any(layer.pixels[:, :, :3] != source[:, :, :3], axis=2))))
    assert np.max(np.abs(centers[0] - centers[1])) <= 1
    assert changed_counts[1] > changed_counts[0] * 2


def apply_line(mode: str, event_count: int) -> np.ndarray:
    layer = Layer(mode, textured_pixels())
    stroke = RetouchStroke(layer, mode, 12, 0.45, 0.25)
    for x in np.linspace(18, 110, event_count):
        stroke.dab(round(float(x)), 48)
    return layer.pixels


def test_retouch_stroke_is_nearly_independent_of_pointer_rate() -> None:
    for mode in ("blur", "sharpen", "dodge", "burn"):
        sparse = apply_line(mode, 10)
        dense = apply_line(mode, 100)
        difference = np.abs(sparse.astype(np.int16) - dense.astype(np.int16))
        assert float(difference.mean()) < 0.2, mode
        assert int(difference.max()) < 12, mode


def test_selection_coverage_multiplies_falloff() -> None:
    source = np.full((64, 64, 4), (120, 120, 120, 255), dtype=np.uint8)
    selection = np.zeros((64, 64), dtype=np.uint8)
    selection[:, :32] = 128
    layer = Layer("selected", source.copy())
    local_retouch(layer, 32, 32, 18, "dodge", 0.8, selection, 0.3)
    left_delta = int(layer.pixels[32, 28, 0]) - 120
    right_delta = int(layer.pixels[32, 36, 0]) - 120
    assert left_delta > 0
    assert right_delta == 0


def test_clone_uses_soft_edge_falloff() -> None:
    pixels = np.full((80, 100, 4), (40, 40, 40, 255), dtype=np.uint8)
    pixels[20:61, 10:51, :3] = 210
    layer = Layer("clone", pixels.copy())
    clone_or_heal(layer, 30, 40, 75, 40, 16, 1.0, False, hardness=0.25)
    center = int(layer.pixels[40, 75, 0]) - 40
    edge = int(layer.pixels[40, 88, 0]) - 40
    outside = int(layer.pixels[40, 95, 0]) - 40
    assert center > edge > outside
    assert outside == 0


def test_healing_transfers_texture_while_adapting_target_tone() -> None:
    pixels = np.full((80, 100, 4), (170, 140, 115, 255), dtype=np.uint8)
    checker = (np.indices((31, 31)).sum(axis=0) % 2) * 30 - 15
    pixels[15:46, 10:41, :3] = np.clip(70 + checker[:, :, None], 0, 255)
    layer = Layer("heal", pixels.copy())
    clone_or_heal(layer, 25, 30, 70, 40, 14, 0.7, True, hardness=0.4)
    healed = layer.pixels[28:53, 58:83, :3]
    assert float(healed.mean()) > 105.0
    assert float(healed.std()) > 2.0


def test_spot_healing_strength_is_controlled() -> None:
    pixels = np.full((64, 64, 4), (170, 140, 120, 255), dtype=np.uint8)
    pixels[29:36, 29:36, :3] = 15
    weak = Layer("weak", pixels.copy())
    strong = Layer("strong", pixels.copy())
    spot_heal(weak, 32, 32, 9, 0.25, hardness=0.4)
    spot_heal(strong, 32, 32, 9, 0.8, hardness=0.4)
    assert int(weak.pixels[32, 32, 0]) < int(strong.pixels[32, 32, 0]) < 200
