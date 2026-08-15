from __future__ import annotations

import cv2
import numpy as np

from uzyro.core import BrushSettings, CloneHealingStroke, Layer, spot_heal
from uzyro.exemplar_inpaint import exemplar_inpaint


def striped_scene(size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    _yy, xx = np.mgrid[:size, :size]
    clean = np.zeros((size, size, 4), dtype=np.uint8)
    clean[:, :, :3] = ((((xx // 4) % 2) * 150 + 50)[:, :, None])
    clean[:, :, 3] = 255
    damaged = clean.copy()
    damaged[54:74, 54:74, :3] = 125
    return clean, damaged


def test_exemplar_fill_is_deterministic_and_preserves_pixels_outside_mask() -> None:
    clean, damaged = striped_scene(96)
    mask = np.zeros((96, 96), dtype=np.uint8)
    cv2.rectangle(mask, (38, 38), (57, 57), 255, -1)
    first = exemplar_inpaint(damaged[:, :, :3], mask, seed=17)
    second = exemplar_inpaint(damaged[:, :, :3], mask, seed=17)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[mask == 0], damaged[:, :, :3][mask == 0])
    before = np.abs(damaged[:, :, :3][mask > 0].astype(np.int16) - clean[:, :, :3][mask > 0]).mean()
    after = np.abs(first[mask > 0].astype(np.int16) - clean[:, :, :3][mask > 0]).mean()
    assert after < before * 0.25


def test_content_aware_spot_preserves_repeating_structure_better_than_fast_modes() -> None:
    clean, damaged = striped_scene()
    errors = {}
    outputs = {}
    for mode in ("fast_telea", "fast_navier_stokes", "content_aware"):
        layer = Layer(mode, damaged.copy())
        spot_heal(layer, 64, 64, 13, 1.0, hardness=0.8, mode=mode)
        outputs[mode] = layer.pixels
        errors[mode] = np.abs(layer.pixels[56:72, 56:72, :3].astype(np.int16) - clean[56:72, 56:72, :3]).mean()
    assert errors["content_aware"] < min(errors["fast_telea"], errors["fast_navier_stokes"]) * 0.2
    assert not np.array_equal(outputs["fast_telea"], outputs["fast_navier_stokes"])


def test_healing_diffusion_changes_texture_but_keeps_target_edge() -> None:
    yy, xx = np.mgrid[:61, :61]
    source = np.zeros((61, 61, 4), dtype=np.float32)
    source[:, :, :3] = (80 + ((xx + yy) % 2) * 70)[:, :, None]
    source[:, :, 3] = 255
    target = np.zeros_like(source)
    target[:, :30, :3] = 55
    target[:, 30:, :3] = 205
    target[:, :, 3] = 255
    precise = CloneHealingStroke._heal_patch(source, target, 28, diffusion=1)
    diffuse = CloneHealingStroke._heal_patch(source, target, 28, diffusion=7)
    assert float(precise[:, 8:24, :3].std()) > float(diffuse[:, 8:24, :3].std())
    assert float(diffuse[:, 35:50, :3].mean() - diffuse[:, 10:25, :3].mean()) > 110
    assert not np.array_equal(precise, diffuse)


def test_clone_healing_respects_layer_mask_selection_alpha_opacity_and_flow() -> None:
    pixels = np.full((64, 96, 4), (40, 40, 40, 255), dtype=np.uint8)
    source = pixels.copy()
    source[:, :30, :3] = (230, 80, 40)
    layer = Layer("masked", pixels.copy())
    layer.mask = np.full((64, 96), 255, dtype=np.uint8)
    layer.mask[:, 45:52] = 0
    layer.mask_enabled = True
    selection = np.full((64, 96), 128, dtype=np.uint8)
    stroke = CloneHealingStroke(
        layer, BrushSettings(14, hardness=1.0, opacity=0.5, flow=0.4), source,
        heal=False, selection_mask=selection, diffusion=4,
    )
    before = layer.pixels.copy()
    stroke.dab(50, 32, 18, 32)
    np.testing.assert_array_equal(layer.pixels[:, 45:52], before[:, 45:52])
    assert np.any(layer.pixels[:, 55:64, :3] != before[:, 55:64, :3])
    np.testing.assert_array_equal(layer.pixels[:, :, 3], before[:, :, 3])


def test_spot_healing_does_not_paint_transparent_or_masked_pixels() -> None:
    pixels = np.full((80, 80, 4), (120, 120, 120, 255), dtype=np.uint8)
    pixels[25:55, 25:40, 3] = 0
    pixels[35:45, 35:45, :3] = 5
    layer = Layer("alpha", pixels.copy())
    layer.mask = np.full((80, 80), 255, dtype=np.uint8)
    layer.mask[:, 40:] = 0
    layer.mask_enabled = True
    before = layer.pixels.copy()
    spot_heal(layer, 40, 40, 14, 1.0, mode="content_aware")
    np.testing.assert_array_equal(layer.pixels[25:55, 25:40], before[25:55, 25:40])
    np.testing.assert_array_equal(layer.pixels[:, 40:], before[:, 40:])
