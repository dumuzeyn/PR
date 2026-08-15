from __future__ import annotations

import cv2
import numpy as np


def rgba_to_perceptual(pixels: np.ndarray) -> np.ndarray:
    """Return CIE Lab plus alpha on a comparable perceptual scale."""
    rgb = np.asarray(pixels[:, :, :3], dtype=np.float32) / 255.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    alpha = np.asarray(pixels[:, :, 3], dtype=np.float32)[:, :, None] * (40.0 / 255.0)
    return np.concatenate((lab, alpha), axis=2)


def perceptual_color_mask(
    pixels: np.ndarray,
    samples: list[tuple[int, int, int, int]],
    fuzziness: int,
    antialias: bool = True,
) -> np.ndarray:
    if pixels.size == 0 or not samples:
        return np.zeros(pixels.shape[:2], dtype=np.uint8)
    fuzziness = max(0, min(128, int(fuzziness)))
    if fuzziness == 0:
        exact = np.zeros(pixels.shape[:2], dtype=bool)
        for sample in samples:
            exact |= np.all(pixels == np.asarray(sample, dtype=np.uint8), axis=2)
        return exact.astype(np.uint8) * 255
    rgba = np.asarray(samples, dtype=np.uint8).reshape((-1, 1, 4))
    sample_lab = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2LAB)[:, 0].astype(np.int16)
    lab = cv2.cvtColor(pixels[:, :, :3], cv2.COLOR_RGB2LAB).astype(np.int16)
    alpha = pixels[:, :, 3].astype(np.float32) * (40.0 / 255.0)
    distance_squared = np.full(pixels.shape[:2], np.inf, dtype=np.float32)
    for index, sample in enumerate(sample_lab):
        lightness = (lab[:, :, 0] - sample[0]).astype(np.float32) / 2.55
        axis_a = (lab[:, :, 1] - sample[1]).astype(np.float32)
        axis_b = (lab[:, :, 2] - sample[2]).astype(np.float32)
        alpha_delta = alpha - float(rgba[index, 0, 3]) * (40.0 / 255.0)
        current = lightness * lightness + axis_a * axis_a + axis_b * axis_b + alpha_delta * alpha_delta
        np.minimum(distance_squared, current, out=distance_squared)
    distance = np.sqrt(distance_squared, out=distance_squared)
    outer = max(1.0, float(fuzziness) * 0.9)
    if not antialias:
        return (distance <= outer).astype(np.uint8) * 255
    inner = outer * 0.55
    return np.clip((outer - distance) / max(1e-6, outer - inner) * 255.0, 0, 255).astype(np.uint8)


def combine_sample_masks(
    pixels: np.ndarray,
    included: list[tuple[int, int, int, int]],
    excluded: list[tuple[int, int, int, int]],
    fuzziness: int,
    antialias: bool = True,
) -> np.ndarray:
    mask = perceptual_color_mask(pixels, included, fuzziness, antialias)
    if excluded:
        removed = perceptual_color_mask(pixels, excluded, fuzziness, antialias)
        mask = np.clip(mask.astype(np.float32) * (1.0 - removed.astype(np.float32) / 255.0), 0, 255).astype(np.uint8)
    return mask
