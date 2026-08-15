from __future__ import annotations

import cv2
import numpy as np


def build_patch_edit(
    target_pixels: np.ndarray,
    target_origin: tuple[int, int],
    selection_mask: np.ndarray,
    source_x: int,
    source_y: int,
    *,
    source_pixels: np.ndarray | None = None,
    source_origin: tuple[int, int] | None = None,
    heal: bool = True,
    structure: int = 5,
    color_adaptation: float = 8.0,
) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    if selection_mask.shape != target_pixels.shape[:2] or not np.any(selection_mask):
        return None
    ys, xs = np.where(selection_mask > 0)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
    width, height = x2 - x1, y2 - y1
    source = target_pixels if source_pixels is None else source_pixels
    origin = target_origin if source_origin is None else source_origin
    sx1, sy1 = int(source_x) - int(origin[0]), int(source_y) - int(origin[1])
    sx2, sy2 = sx1 + width, sy1 + height
    if sx1 < 0 or sy1 < 0 or sx2 > source.shape[1] or sy2 > source.shape[0]:
        return None

    src = source[sy1:sy2, sx1:sx2].astype(np.float32)
    dst = target_pixels[y1:y2, x1:x2].astype(np.float32)
    mask = selection_mask[y1:y2, x1:x2].astype(np.float32) / 255.0
    if not heal:
        edited = src
    else:
        structure_amount = float(np.clip(structure, 1, 7)) / 7.0
        adaptation = float(np.clip(color_adaptation / 10.0 if color_adaptation > 1.0 else color_adaptation, 0.0, 1.0))
        sigma = max(1.0, min(18.0, max(width, height) * (0.16 - structure_amount * 0.11)))
        source_low = cv2.GaussianBlur(src[:, :, :3], (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
        ring = max(2, min(24, round(max(width, height) * 0.16)))
        inner_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring * 2 + 1, ring * 2 + 1))
        outer_radius = ring * 3
        outer_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer_radius * 2 + 1, outer_radius * 2 + 1))
        padding = outer_radius + 2
        rx1, ry1 = max(0, x1 - padding), max(0, y1 - padding)
        rx2, ry2 = min(target_pixels.shape[1], x2 + padding), min(target_pixels.shape[0], y2 + padding)
        binary = (selection_mask[ry1:ry2, rx1:rx2] > 0).astype(np.uint8)
        tone_ring = (cv2.dilate(binary, outer_kernel) > 0) & (cv2.dilate(binary, inner_kernel) == 0)
        target_low = _fit_tone_surface(
            target_pixels[ry1:ry2, rx1:rx2, :3],
            tone_ring,
            (x1 - rx1, y1 - ry1, x2 - rx1, y2 - ry1),
        )
        source_detail = src[:, :, :3] - source_low
        tone_matched = target_low + source_detail * (0.35 + structure_amount * 0.95)
        edited = src.copy()
        edited[:, :, :3] = np.clip(src[:, :, :3] * (1.0 - adaptation) + tone_matched * adaptation, 0, 255)
        edited[:, :, 3] = dst[:, :, 3]
    mixed = dst * (1.0 - mask[:, :, None]) + edited * mask[:, :, None]
    return (x1, y1, x2, y2), np.clip(mixed, 0, 255).astype(np.uint8)


def _fit_tone_surface(
    pixels: np.ndarray,
    ring: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    ys, xs = np.where(ring)
    x1, y1, x2, y2 = bounds
    if len(xs) < 3:
        fallback = np.median(pixels.reshape(-1, 3), axis=0)
        return np.broadcast_to(fallback, (y2 - y1, x2 - x1, 3)).astype(np.float32).copy()
    values = pixels[ys, xs].astype(np.float32)
    median = np.median(values, axis=0)
    deviation = np.median(np.abs(values - median), axis=0)
    threshold = np.maximum(18.0, deviation * 3.5)
    keep = np.all(np.abs(values - median) <= threshold, axis=1)
    if np.count_nonzero(keep) >= 3:
        xs, ys, values = xs[keep], ys[keep], values[keep]
    design = np.column_stack((xs, ys, np.ones(len(xs), dtype=np.float32))).astype(np.float32)
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    target_y, target_x = np.mgrid[y1:y2, x1:x2]
    target_design = np.stack((target_x, target_y, np.ones_like(target_x)), axis=-1).astype(np.float32)
    return np.clip(target_design @ coefficients, 0, 255).astype(np.float32)


__all__ = ["build_patch_edit"]
