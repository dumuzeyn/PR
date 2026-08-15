from __future__ import annotations

import cv2
import numpy as np


def image_similarity(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    """Return tolerant visual metrics for equally sized RGB/RGBA images."""
    left = np.asarray(actual)
    right = np.asarray(expected)
    if left.shape != right.shape:
        raise ValueError(f"Image shapes differ: {left.shape} != {right.shape}")
    if left.ndim != 3 or left.shape[2] not in (3, 4):
        raise ValueError("Expected RGB or RGBA images")
    left = left.astype(np.float32)
    right = right.astype(np.float32)
    channels = left.shape[2]
    scores = []
    for channel in range(channels):
        x = left[:, :, channel]
        y = right[:, :, channel]
        mean_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mean_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        variance_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mean_x * mean_x
        variance_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mean_y * mean_y
        covariance = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mean_x * mean_y
        numerator = (2.0 * mean_x * mean_y + 6.5025) * (2.0 * covariance + 58.5225)
        denominator = (mean_x * mean_x + mean_y * mean_y + 6.5025) * (variance_x + variance_y + 58.5225)
        scores.append(float(np.mean(np.divide(numerator, denominator, out=np.ones_like(numerator), where=denominator != 0))))
    difference = np.abs(left - right)
    return {
        "ssim": float(np.mean(scores)),
        "mae": float(np.mean(difference)),
        "p99": float(np.percentile(difference, 99)),
        "max": float(np.max(difference)),
    }


def assert_similar_image(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    minimum_ssim: float = 0.985,
    maximum_mae: float = 2.0,
) -> dict[str, float]:
    metrics = image_similarity(actual, expected)
    if metrics["ssim"] < minimum_ssim or metrics["mae"] > maximum_mae:
        raise AssertionError(
            f"Visual mismatch: SSIM {metrics['ssim']:.5f} < {minimum_ssim:.5f} "
            f"or MAE {metrics['mae']:.3f} > {maximum_mae:.3f}; "
            f"p99={metrics['p99']:.1f}, max={metrics['max']:.1f}"
        )
    return metrics


def mask_quality(actual: np.ndarray, expected: np.ndarray, boundary_radius: int = 2) -> dict[str, float]:
    """Measure binary region and boundary quality while retaining soft-alpha error."""
    left = np.asarray(actual, dtype=np.uint8)
    right = np.asarray(expected, dtype=np.uint8)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Masks must be equally sized 2D arrays")
    predicted = left >= 128
    truth = right >= 128
    intersection = int(np.count_nonzero(predicted & truth))
    predicted_count = int(np.count_nonzero(predicted))
    truth_count = int(np.count_nonzero(truth))
    union = int(np.count_nonzero(predicted | truth))
    precision = intersection / max(1, predicted_count)
    recall = intersection / max(1, truth_count)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (boundary_radius * 2 + 1, boundary_radius * 2 + 1))
    predicted_edge = cv2.morphologyEx(predicted.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    truth_edge = cv2.morphologyEx(truth.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    predicted_near_truth = cv2.dilate(truth_edge.astype(np.uint8), kernel) > 0
    truth_near_predicted = cv2.dilate(predicted_edge.astype(np.uint8), kernel) > 0
    boundary_precision = np.count_nonzero(predicted_edge & predicted_near_truth) / max(1, np.count_nonzero(predicted_edge))
    boundary_recall = np.count_nonzero(truth_edge & truth_near_predicted) / max(1, np.count_nonzero(truth_edge))
    boundary_f1 = 2.0 * boundary_precision * boundary_recall / max(1e-9, boundary_precision + boundary_recall)
    return {
        "iou": intersection / max(1, union),
        "precision": precision,
        "recall": recall,
        "boundary_f1": float(boundary_f1),
        "alpha_mae": float(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))) / 255.0),
    }


__all__ = ["assert_similar_image", "image_similarity", "mask_quality"]
