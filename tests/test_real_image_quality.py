from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from photoredactor.core import Layer, content_aware_fill, spot_heal
from photoredactor.quality_metrics import image_similarity, mask_quality
from photoredactor.segmentation import OpenCvCpuSegmentationBackend, postprocess_segmentation
from photoredactor.selection_ops import sky_selection_mask


FIXTURES = Path(__file__).with_name("fixtures") / "quality"


def _read(name: str, suffix: str, mode: str = "RGBA") -> np.ndarray:
    return np.asarray(Image.open(FIXTURES / f"{name}_{suffix}.png").convert(mode), dtype=np.uint8)


def _quality_roi(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[slice, slice]]:
    image = _read(name, "input")
    expected = _read(name, "expected")
    mask = _read(name, "mask", "L")
    ys, xs = np.where(mask > 0)
    x1, y1 = max(0, int(xs.min()) - 8), max(0, int(ys.min()) - 8)
    x2, y2 = min(image.shape[1], int(xs.max()) + 9), min(image.shape[0], int(ys.max()) + 9)
    return image, expected, mask, (slice(y1, y2), slice(x1, x2))


def test_similar_background_object_selection_quality() -> None:
    image = _read("similar_object", "input")
    expected = _read("similar_object", "mask", "L")
    actual = OpenCvCpuSegmentationBackend().select_object(image, (55, 25, 185, 170), 0.65)
    metrics = mask_quality(actual, expected)
    assert metrics["iou"] >= 0.96
    assert metrics["precision"] >= 0.98
    assert metrics["recall"] >= 0.96
    assert metrics["boundary_f1"] >= 0.95


def test_sky_between_buildings_quality() -> None:
    image = _read("sky_buildings", "input")
    expected = _read("sky_buildings", "mask", "L")
    metrics = mask_quality(sky_selection_mask(image, 0.72), expected)
    assert metrics["iou"] >= 0.95
    assert metrics["precision"] >= 0.98
    assert metrics["boundary_f1"] >= 0.88


def test_thin_hair_and_soft_alpha_survive_postprocessing() -> None:
    image = _read("hair", "input")
    expected = _read("hair", "mask", "L")
    damaged = expected.copy()
    damaged[::19, ::17] = 0
    damaged[4, 4] = 255
    actual = postprocess_segmentation(damaged, image, "subject", 0.65)
    metrics = mask_quality(actual, expected)
    assert metrics["iou"] >= 0.98
    assert metrics["boundary_f1"] >= 0.92
    assert metrics["alpha_mae"] <= 0.01
    assert np.count_nonzero((actual > 0) & (actual < 255)) > 100


@pytest.mark.parametrize(
    ("name", "minimum_ssim_gain", "maximum_mae_ratio"),
    (("repeated_texture", 0.01, 0.82), ("wire_texture", 0.05, 0.2)),
)
def test_content_aware_improves_stored_texture_fixtures(
    name: str, minimum_ssim_gain: float, maximum_mae_ratio: float,
) -> None:
    image, expected, mask, roi = _quality_roi(name)
    output = content_aware_fill(
        image, mask, radius=5, color_adaptation=0.35,
        rotation_adaptation=False, scale_adaptation=False,
    )
    before = image_similarity(image[roi], expected[roi])
    after = image_similarity(output[roi], expected[roi])
    assert after["ssim"] >= before["ssim"] + minimum_ssim_gain
    assert after["mae"] <= before["mae"] * maximum_mae_ratio
    np.testing.assert_array_equal(output[mask == 0], image[mask == 0])


@pytest.mark.parametrize("name", ("skin_blemish", "sharp_edge"))
def test_spot_healing_improves_realistic_fixture_without_touching_outside(name: str) -> None:
    image, expected, mask, roi = _quality_roi(name)
    layer = Layer(name, image.copy())
    ys, xs = np.where(mask >= 128)
    x, y = round(float(xs.mean())), round(float(ys.mean()))
    radius = max(3, round(max(int(xs.max() - xs.min()), int(ys.max() - ys.min())) / 2) + 3)
    spot_heal(layer, x, y, radius, 1.0, selection_mask=mask, hardness=0.85, mode="content_aware")
    before = image_similarity(image[roi], expected[roi])
    after = image_similarity(layer.pixels[roi], expected[roi])
    assert after["ssim"] >= before["ssim"] + 0.12
    assert after["mae"] <= before["mae"] * 0.15
    outside = mask == 0
    np.testing.assert_array_equal(layer.pixels[outside], image[outside])


def test_quality_metric_detects_regression_and_soft_alpha_error() -> None:
    expected = np.zeros((48, 64, 4), dtype=np.uint8)
    expected[12:36, 16:48] = (40, 120, 220, 255)
    shifted = np.roll(expected, 5, axis=1)
    image_metrics = image_similarity(shifted, expected)
    mask_metrics = mask_quality(shifted[:, :, 3], expected[:, :, 3])
    assert image_metrics["ssim"] < 0.9
    assert image_metrics["mae"] > 5.0
    assert mask_metrics["iou"] < 0.75
    assert mask_metrics["boundary_f1"] < 0.8
