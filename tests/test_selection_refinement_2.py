from __future__ import annotations

import cv2
import numpy as np

from photoredactor.core import cleanup_selection_edges, correct_selection_edges, refine_selection_mask
from photoredactor.selection_refinement import signed_distance_field


def jagged_soft_mask() -> np.ndarray:
    mask = np.zeros((120, 160), dtype=np.uint8)
    points = np.array([[32, 25], [126, 25], [126, 94], [32, 94]], dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    for y in range(28, 92, 6):
        mask[y:y + 3, 27:35] = 255
        mask[y + 3:y + 6, 32:38] = 0
    mask = cv2.GaussianBlur(mask, (5, 5), 0.9)
    cv2.line(mask, (80, 24), (92, 5), 92, 1, cv2.LINE_AA)
    return mask


def boundary_length(mask: np.ndarray) -> float:
    contours, _hierarchy = cv2.findContours((mask >= 128).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return sum(cv2.arcLength(contour, True) for contour in contours)


def test_signed_distance_has_expected_sign_and_order() -> None:
    binary = np.zeros((40, 50), dtype=np.uint8)
    binary[10:30, 12:38] = 255
    field = signed_distance_field(binary)
    assert field[20, 25] > field[10, 12] > 0
    assert field[0, 0] < 0


def test_smooth_reduces_jagged_geometry_without_binary_alpha() -> None:
    source = jagged_soft_mask()
    refined = refine_selection_mask(source, smooth=6)
    assert boundary_length(refined) < boundary_length(source)
    assert np.count_nonzero((refined > 0) & (refined < 255)) > 80
    assert refined[12, 87] > 0


def test_feather_contrast_and_shift_have_distinct_meanings() -> None:
    source = np.zeros((120, 160), dtype=np.uint8)
    source[35:85, 50:110] = 255
    base_area = np.count_nonzero(source >= 128)
    feathered = refine_selection_mask(source, feather=5)
    contrasted = refine_selection_mask(feathered, contrast=2.5)
    expanded = refine_selection_mask(source, shift=5)
    contracted = refine_selection_mask(source, shift=-5)
    assert np.count_nonzero((feathered > 0) & (feathered < 255)) > 500
    assert abs(np.count_nonzero(feathered >= 128) - base_area) < 100
    assert np.count_nonzero((contrasted > 32) & (contrasted < 223)) < np.count_nonzero((feathered > 32) & (feathered < 223))
    assert np.count_nonzero(expanded >= 128) > base_area
    assert np.count_nonzero(contracted >= 128) < base_area


def test_soft_matte_survives_combined_refinement() -> None:
    source = jagged_soft_mask()
    result = refine_selection_mask(source, smooth=3, feather=2, contrast=1.3, shift=-1)
    assert result.dtype == np.uint8
    assert result[60, 80] > 245
    assert result[12, 87] > 0
    assert np.unique(result).size > 32


def test_edge_cleanup_paths_keep_fractional_alpha() -> None:
    image = np.full((120, 160, 4), (40, 95, 170, 255), dtype=np.uint8)
    image[30:90, 45:115, :3] = (195, 60, 45)
    source = jagged_soft_mask()
    cleaned = cleanup_selection_edges(source, image, 4, 0.7)
    corrected = correct_selection_edges(source, image, 4, 0.7, 80)
    for result in (cleaned, corrected):
        assert result.dtype == np.uint8
        assert result[60, 80] > 200
        assert np.count_nonzero((result > 0) & (result < 255)) > 50
