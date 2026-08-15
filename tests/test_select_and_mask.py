from __future__ import annotations

import cv2
import numpy as np

from uzyro.core import decontaminate_edge_colors, refine_selection_brush


def synthetic_hair_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.zeros((180, 240, 4), dtype=np.uint8)
    image[:, :, :3] = (35, 90, 170)
    image[:, :, 3] = 255
    cv2.ellipse(image, (120, 105), (48, 58), 0, 0, 360, (175, 65, 45, 255), -1)
    for offset in range(-42, 43, 7):
        cv2.line(image, (120 + offset // 3, 60), (120 + offset, 12), (175, 65, 45, 255), 2, cv2.LINE_AA)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, (120, 108), (43, 53), 0, 0, 360, 255, -1)
    brush = np.zeros_like(mask)
    cv2.rectangle(brush, (66, 5), (174, 78), 255, -1)
    return image, mask, brush


def test_refine_brush_recovers_thin_hair_with_soft_alpha() -> None:
    image, mask, brush = synthetic_hair_scene()
    refined = refine_selection_brush(mask, image, brush, "refine", 9, 0.9)
    assert refined.shape == mask.shape
    assert np.count_nonzero(refined[8:62, 70:170]) > np.count_nonzero(mask[8:62, 70:170])
    assert np.count_nonzero((refined > 0) & (refined < 255)) > 30
    assert refined[108, 120] == 255
    assert refined[170, 10] == 0


def test_manual_brush_modes_are_local_and_preserve_soft_coverage() -> None:
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[20:60, 25:75] = 255
    image = np.zeros((80, 100, 4), dtype=np.uint8)
    image[:, :, 3] = 255
    brush = np.zeros_like(mask)
    cv2.circle(brush, (78, 40), 12, 180, -1, cv2.LINE_AA)
    added = refine_selection_brush(mask, image, brush, "add", 4, 0.8)
    subtracted = refine_selection_brush(mask, image, brush, "subtract", 4, 0.8)
    smoothed = refine_selection_brush(mask, image, brush, "smooth", 4, 1.0)
    assert added[40, 78] > mask[40, 78]
    assert subtracted[40, 70] < mask[40, 70]
    assert np.array_equal(added[:10, :10], mask[:10, :10])
    assert np.count_nonzero((smoothed > 0) & (smoothed < 255)) > 0


def test_edge_color_decontamination_keeps_alpha_and_solid_pixels() -> None:
    pixels = np.zeros((60, 80, 4), dtype=np.uint8)
    pixels[:, :, :3] = (20, 80, 190)
    pixels[:, :, 3] = 255
    pixels[15:45, 20:60, :3] = (200, 55, 45)
    pixels[15:18, 20:60, :3] = (90, 70, 140)
    pixels[42:45, 20:60, :3] = (90, 70, 140)
    pixels[15:45, 20:24, :3] = (90, 70, 140)
    pixels[15:45, 56:60, :3] = (90, 70, 140)
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[18:42, 24:56] = 255
    mask[15:45, 20:60] = np.maximum(mask[15:45, 20:60], 120)
    result = decontaminate_edge_colors(pixels, mask, 0.8, 5)
    assert np.array_equal(result[:, :, 3], pixels[:, :, 3])
    assert np.array_equal(result[30, 40], pixels[30, 40])
    assert result[16, 22, 0] >= pixels[16, 22, 0]
    assert not np.array_equal(result[16, 22, :3], pixels[16, 22, :3])
