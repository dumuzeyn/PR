from __future__ import annotations

import cv2
import numpy as np


def _feature_image(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    local = cv2.boxFilter(lab, cv2.CV_32F, (5, 5), borderType=cv2.BORDER_REFLECT_101)
    gray = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) * 0.35
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) * 0.35
    return np.dstack((lab, local, gx, gy)).astype(np.float32)


def _candidate_coordinates(mask: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    erosion = cv2.erode((mask == 0).astype(np.uint8), np.ones((5, 5), dtype=np.uint8)) > 0
    coordinates = np.column_stack(np.where(erosion))
    if not len(coordinates):
        coordinates = np.column_stack(np.where(mask == 0))
    if len(coordinates) > limit:
        coordinates = coordinates[rng.choice(len(coordinates), limit, replace=False)]
    return coordinates.astype(np.int32)


def _coherent_patch(source: np.ndarray, hard: np.ndarray, fallback: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    targets = np.column_stack(np.where(hard))
    ring_mask = cv2.dilate(hard.astype(np.uint8), np.ones((7, 7), dtype=np.uint8)) > 0
    ring_mask &= ~hard
    ring = np.column_stack(np.where(ring_mask))
    if not len(targets) or len(ring) < 4:
        return fallback
    candidates = _candidate_coordinates(hard.astype(np.uint8), 640, rng)
    center = np.rint(targets.mean(axis=0)).astype(np.int32)
    features = _feature_image(source)
    target_ring = features[ring[:, 0], ring[:, 1]]
    best_cost, best_offset = float("inf"), None
    height, width = hard.shape
    for candidate in candidates:
        offset = candidate - center
        shifted_targets = targets + offset
        shifted_ring = ring + offset
        if (
            shifted_targets[:, 0].min() < 0 or shifted_targets[:, 0].max() >= height
            or shifted_targets[:, 1].min() < 0 or shifted_targets[:, 1].max() >= width
            or shifted_ring[:, 0].min() < 0 or shifted_ring[:, 0].max() >= height
            or shifted_ring[:, 1].min() < 0 or shifted_ring[:, 1].max() >= width
            or np.any(hard[shifted_targets[:, 0], shifted_targets[:, 1]])
        ):
            continue
        candidate_ring = features[shifted_ring[:, 0], shifted_ring[:, 1]]
        delta = candidate_ring - target_ring
        cost = float(np.mean(np.square(delta[:, :6])) + np.mean(np.square(delta[:, 6:])) * 1.6)
        if cost < best_cost:
            best_cost, best_offset = cost, offset
    if best_offset is None:
        return fallback
    shifted = targets + best_offset
    result = fallback.copy()
    result[targets[:, 0], targets[:, 1]] = source[shifted[:, 0], shifted[:, 1]]
    return result


def _fill_level(
    source: np.ndarray,
    mask: np.ndarray,
    initial: np.ndarray | None,
    seed: int,
) -> np.ndarray:
    hard = np.asarray(mask, dtype=np.uint8) > 0
    if not np.any(hard) or np.all(hard):
        return source.copy()
    rng = np.random.default_rng(int(seed))
    result = source.copy() if initial is None else np.asarray(initial, dtype=np.uint8).copy()
    if initial is None:
        result[hard] = cv2.inpaint(source, hard.astype(np.uint8) * 255, 3.0, cv2.INPAINT_TELEA)[hard]
    result = _coherent_patch(source, hard, result, rng)
    candidates = _candidate_coordinates(hard.astype(np.uint8), 2048, rng)
    if not len(candidates):
        return result
    distance = cv2.distanceTransform(hard.astype(np.uint8), cv2.DIST_L2, 3)
    targets = np.column_stack(np.where(hard))
    targets = targets[np.argsort(distance[hard], kind="stable")]
    height, width = hard.shape
    spatial_weight = np.sqrt(180.0)
    candidate_base = _feature_image(source)[candidates[:, 0], candidates[:, 1]]
    candidate_desc = np.column_stack((
        candidate_base[:, :6], candidate_base[:, 6:] * 1.25,
        candidates[:, 0] / max(1, height) * spatial_weight,
        candidates[:, 1] / max(1, width) * spatial_weight,
    )).astype(np.float32)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    boundary_count = max(1, len(targets) // 4)
    for band in np.array_split(targets[:boundary_count], min(2, boundary_count)):
        if not len(band):
            continue
        target_base = _feature_image(result)[band[:, 0], band[:, 1]]
        target_desc = np.column_stack((
            target_base[:, :6], target_base[:, 6:] * 1.25,
            band[:, 0] / max(1, height) * spatial_weight,
            band[:, 1] / max(1, width) * spatial_weight,
        )).astype(np.float32)
        matches = matcher.match(target_desc, candidate_desc)
        source_points = candidates[np.asarray([match.trainIdx for match in matches], dtype=np.int32)]
        result[band[:, 0], band[:, 1]] = source[source_points[:, 0], source_points[:, 1]]
    return result


def exemplar_inpaint(rgb: np.ndarray, mask: np.ndarray, seed: int = 0) -> np.ndarray:
    """Deterministic coarse-to-fine exemplar fill using color, texture, and edge features."""
    source = np.ascontiguousarray(rgb, dtype=np.uint8)
    hard = np.asarray(mask, dtype=np.uint8)
    if hard.shape != source.shape[:2]:
        hard = cv2.resize(hard, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_NEAREST)
    if not np.any(hard):
        return source.copy()
    initial = None
    if min(source.shape[:2]) >= 48:
        small_size = max(8, source.shape[1] // 2), max(8, source.shape[0] // 2)
        small_source = cv2.resize(source, small_size, interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(hard, small_size, interpolation=cv2.INTER_NEAREST)
        coarse = _fill_level(small_source, small_mask, None, seed)
        initial = source.copy()
        upsampled = cv2.resize(coarse, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_CUBIC)
        initial[hard > 0] = upsampled[hard > 0]
    filled = _fill_level(source, hard, initial, seed + 1)
    boundary = cv2.dilate((hard > 0).astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) - cv2.erode((hard > 0).astype(np.uint8), np.ones((3, 3), dtype=np.uint8))
    blend = cv2.GaussianBlur(boundary.astype(np.float32), (0, 0), 0.7)[:, :, None]
    blend = np.clip(blend, 0.0, 1.0) * (hard > 0)[:, :, None]
    output = filled.astype(np.float32) * (1.0 - blend) + cv2.bilateralFilter(filled, 5, 12, 4).astype(np.float32) * blend
    output[hard == 0] = source[hard == 0]
    return np.clip(output, 0, 255).astype(np.uint8)


__all__ = ["exemplar_inpaint"]
