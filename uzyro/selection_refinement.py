from __future__ import annotations

import cv2
import numpy as np


def signed_distance_field(binary: np.ndarray) -> np.ndarray:
    """Return a positive-inside signed distance field in pixel units."""
    inside = np.asarray(binary, dtype=np.uint8) > 0
    if not np.any(inside):
        return np.full(inside.shape, -1.0, dtype=np.float32)
    if np.all(inside):
        return np.full(inside.shape, 1.0, dtype=np.float32)
    foreground = inside.astype(np.uint8)
    background = 1 - foreground
    distance_in = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)
    distance_out = cv2.distanceTransform(background, cv2.DIST_L2, 5)
    return distance_in.astype(np.float32) - distance_out.astype(np.float32)


def _processing_bounds(mask: np.ndarray, margin: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return None
    height, width = mask.shape
    return (
        max(0, int(xs.min()) - margin),
        max(0, int(ys.min()) - margin),
        min(width, int(xs.max()) + margin + 1),
        min(height, int(ys.max()) + margin + 1),
    )


def _level_set_refine(alpha: np.ndarray, smooth: int, shift: int) -> np.ndarray:
    partial = np.any((alpha > 0.0) & (alpha < 1.0))
    levels = (0.125, 0.375, 0.625, 0.875) if partial else (0.5,)
    sigma = max(0.0, float(smooth) * 0.35)
    coverage = np.zeros(alpha.shape, dtype=np.float32)
    for level in levels:
        binary = alpha >= level
        field = signed_distance_field(binary)
        if sigma > 0.0 and np.any(binary) and not np.all(binary):
            field = cv2.GaussianBlur(field, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
        field += float(shift)
        coverage += np.clip(0.5 + field * 0.5, 0.0, 1.0)
    coverage /= float(len(levels))
    fine_detail = np.where((alpha > 0.0) & (alpha < 0.5), alpha, 0.0)
    return np.maximum(coverage, fine_detail)


def _apply_contrast(alpha: np.ndarray, contrast: float) -> np.ndarray:
    value = max(0.05, float(contrast))
    if abs(value - 1.0) <= 1e-6:
        return alpha
    lower = np.power(alpha, value)
    upper = np.power(1.0 - alpha, value)
    return np.divide(lower, lower + upper, out=np.zeros_like(alpha), where=(lower + upper) > 1e-8)


def refine_soft_selection(
    mask: np.ndarray,
    smooth: int = 0,
    feather: int = 0,
    contrast: float = 1.0,
    shift: int = 0,
) -> np.ndarray:
    """Refine geometry and alpha without reducing the mask to a binary contour."""
    source = np.clip(np.asarray(mask), 0, 255).astype(np.uint8)
    if source.size == 0 or not np.any(source):
        return np.zeros_like(source, dtype=np.uint8)
    smooth = max(0, int(smooth))
    feather = max(0, int(feather))
    shift = int(shift)
    margin = max(4, abs(shift) + smooth * 2 + feather * 4 + 4)
    bounds = _processing_bounds(source, margin)
    if bounds is None:
        return np.zeros_like(source, dtype=np.uint8)
    x1, y1, x2, y2 = bounds
    alpha = source[y1:y2, x1:x2].astype(np.float32) / 255.0
    if smooth > 0 or shift != 0:
        alpha = _level_set_refine(alpha, smooth, shift)
    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), max(0.1, float(feather)), borderType=cv2.BORDER_CONSTANT)
    alpha = _apply_contrast(np.clip(alpha, 0.0, 1.0), contrast)
    result = np.zeros_like(source, dtype=np.uint8)
    result[y1:y2, x1:x2] = np.rint(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    return result


__all__ = ["refine_soft_selection", "signed_distance_field"]
