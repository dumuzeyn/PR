from __future__ import annotations

import cv2
import numpy as np

from uzyro.core import content_aware_fill, content_aware_fill_variants


def source_guidance_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = 100, 160
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    pixels[:, :55, :3] = (215, 72, 42)
    pixels[:, 105:, :3] = (45, 105, 220)
    pixels[:, 55:105, :3] = (118, 118, 118)
    pixels[:, :, 3] = 255
    cv2.rectangle(pixels, (62, 28), (98, 72), (8, 8, 8, 255), -1)
    target = np.zeros((height, width), dtype=np.uint8)
    target[28:73, 62:99] = 255
    left = np.zeros_like(target)
    left[:, :55] = 255
    right = np.zeros_like(target)
    right[:, 105:] = 255
    return pixels, target, left, right


def test_source_mask_really_controls_texture_origin() -> None:
    pixels, target, left, right = source_guidance_scene()
    from_left = content_aware_fill(pixels, target, 5, left, 0.0, False, False)
    from_right = content_aware_fill(pixels, target, 5, right, 0.0, False, False)
    center = np.s_[38:63, 72:89, :3]
    left_mean = from_left[center].mean(axis=(0, 1))
    right_mean = from_right[center].mean(axis=(0, 1))
    assert left_mean[0] > right_mean[0] + 45
    assert right_mean[2] > left_mean[2] + 45
    assert np.array_equal(from_left[target == 0], pixels[target == 0])
    assert np.array_equal(from_right[target == 0], pixels[target == 0])


def test_variants_are_distinct_and_remove_target_content() -> None:
    pixels, target, left, _ = source_guidance_scene()
    variants = content_aware_fill_variants(
        pixels,
        target,
        left,
        radius=7,
        color_adaptation=0.25,
        rotation_adaptation=True,
        scale_adaptation=True,
        count=3,
        seed=19,
    )
    assert len(variants) == 3
    assert all(np.array_equal(item[target == 0], pixels[target == 0]) for item in variants)
    assert all(float(item[target > 0, :3].mean()) > 35 for item in variants)
    differences = [np.mean(np.abs(variants[index].astype(np.int16) - variants[0].astype(np.int16))[target > 0]) for index in (1, 2)]
    assert min(differences) > 1.0


def test_soft_selection_blends_boundary_and_preserves_transparency() -> None:
    height, width = 90, 130
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    pixels[:, :, :3] = (58, 145, 88)
    pixels[:, :, 3] = 220
    cv2.circle(pixels, (65, 45), 18, (230, 35, 45, 255), -1)
    hard = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(hard, (65, 45), 20, 255, -1)
    soft = cv2.GaussianBlur(hard, (0, 0), 2.2)
    source = np.zeros_like(hard)
    source[:, :35] = 255
    result = content_aware_fill(pixels, soft, 5, source, 0.4, True, True)
    assert np.array_equal(result[soft == 0], pixels[soft == 0])
    boundary = (soft > 0) & (soft < 255)
    assert np.any(result[boundary] != pixels[boundary])
    assert np.all(result[:, :, 3] <= 255)
    assert int(result[45, 65, 0]) < 200


def test_empty_selection_is_a_noop() -> None:
    pixels, _, _, _ = source_guidance_scene()
    empty = np.zeros(pixels.shape[:2], dtype=np.uint8)
    variants = content_aware_fill_variants(pixels, empty, count=4)
    assert len(variants) == 1
    assert np.array_equal(variants[0], pixels)
