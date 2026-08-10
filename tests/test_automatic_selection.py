from __future__ import annotations

import cv2
import numpy as np

from photoredactor.core import Document, background_selection_mask, sky_selection_mask, subject_selection_mask


def complex_subject_scene() -> tuple[np.ndarray, np.ndarray]:
    height, width = 240, 320
    yy, xx = np.indices((height, width))
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    pixels[:, :, :3] = np.stack((40 + xx // 8, 100 + yy // 5, 165 - xx // 10), axis=2)
    pixels[:, :, 3] = 255
    truth = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(truth, (160, 145), (58, 68), 0, 0, 360, 255, -1)
    for offset in range(-48, 49, 8):
        cv2.line(truth, (160 + offset // 4, 85), (160 + offset, 20), 255, 2, cv2.LINE_AA)
    cv2.circle(truth, (235, 170), 8, 255, -1)
    pixels[truth > 0, :3] = (190, 55, 40)
    return pixels, truth


def test_subject_selection_keeps_hair_and_small_disconnected_detail() -> None:
    pixels, truth = complex_subject_scene()
    mask = subject_selection_mask(pixels, 0.65)
    predicted = mask >= 128
    expected = truth >= 128
    intersection = np.count_nonzero(predicted & expected)
    union = np.count_nonzero(predicted | expected)
    assert intersection / union > 0.88
    assert mask[170, 235] >= 128
    assert np.count_nonzero(mask[18:80]) > 250
    assert np.count_nonzero((mask > 0) & (mask < 255)) > 100


def test_background_selection_is_soft_inverse_of_subject() -> None:
    pixels, _truth = complex_subject_scene()
    subject = subject_selection_mask(pixels, 0.55)
    background = background_selection_mask(pixels, 0.55)
    assert np.max(np.abs(subject.astype(np.int16) + background.astype(np.int16) - 255)) == 0
    assert np.count_nonzero((background > 0) & (background < 255)) > 100


def test_magic_wand_and_color_range_keep_antialiased_similarity() -> None:
    document = Document.new(180, 100, (0, 0, 0, 255))
    layer = document.layer
    gradient = np.linspace(80, 150, 180, dtype=np.uint8)
    layer.pixels[:, :, 0] = gradient[None, :]
    layer.pixels[:, :, 1] = 90
    layer.pixels[:, :, 2] = 100
    document.magic_wand_selection(layer, 70, 50, 36, contiguous=True)
    assert document.selection_mask is not None
    assert np.count_nonzero((document.selection_mask > 0) & (document.selection_mask < 255)) > 100
    document.color_range_selection(layer, 70, 50, 36)
    assert document.selection_mask is not None
    assert np.count_nonzero((document.selection_mask > 0) & (document.selection_mask < 255)) > 100


def test_sky_selection_follows_top_connection_around_foreground_branches() -> None:
    height, width = 220, 300
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    for y in range(height):
        pixels[y, :, :3] = (90 + y // 6, 150 + y // 8, 220 - y // 12)
    pixels[:, :, 3] = 255
    pixels[155:, :, :3] = (65, 105, 55)
    cv2.line(pixels, (150, 219), (145, 55), (42, 38, 32, 255), 9, cv2.LINE_AA)
    cv2.line(pixels, (147, 90), (100, 48), (42, 38, 32, 255), 5, cv2.LINE_AA)
    cv2.line(pixels, (146, 105), (205, 62), (42, 38, 32, 255), 5, cv2.LINE_AA)
    mask = sky_selection_mask(pixels, 0.65)
    assert mask[15, 20] >= 220
    assert mask[200, 20] == 0
    assert mask[80, 146] < 80
    assert np.count_nonzero((mask > 0) & (mask < 255)) > 100


def test_transparent_subject_preserves_original_alpha_and_small_parts() -> None:
    pixels = np.zeros((100, 140, 4), dtype=np.uint8)
    pixels[20:80, 30:90] = (180, 70, 45, 210)
    pixels[42:55, 105:118] = (180, 70, 45, 128)
    mask = subject_selection_mask(pixels)
    assert np.array_equal(mask, pixels[:, :, 3])
