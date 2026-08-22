from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from uzyro.segmentation import (
    OpenCvCpuSegmentationBackend,
    SegmentationBackend,
    SegmentationService,
    postprocess_segmentation,
)
from uzyro.segmentation_config import SegmentationConfig
from uzyro.selection_ops import sky_selection_mask


class FixedBackend(SegmentationBackend):
    def __init__(self, subject: np.ndarray, sky: np.ndarray | None = None, fail: bool = False, name: str = "fixed") -> None:
        self.subject_mask = subject
        self.sky_mask = subject if sky is None else sky
        self.fail = fail
        self.name = name
        self.subject_calls = 0
        self.sky_calls = 0

    def select_subject(self, image: np.ndarray, sensitivity: float = 0.5) -> np.ndarray:
        self.subject_calls += 1
        if self.fail:
            raise RuntimeError("backend failed")
        return self.subject_mask.copy()

    def select_object(self, image: np.ndarray, box: tuple[int, int, int, int], sensitivity: float = 0.5) -> np.ndarray:
        mask = self.select_subject(image, sensitivity)
        gate = np.zeros_like(mask)
        x1, y1, x2, y2 = box
        gate[y1:y2, x1:x2] = 255
        return cv2.bitwise_and(mask, gate)

    def select_sky(self, image: np.ndarray, sensitivity: float = 0.5) -> np.ndarray:
        self.sky_calls += 1
        if self.fail:
            raise RuntimeError("backend failed")
        return self.sky_mask.copy()


def rgba(width: int = 120, height: int = 90) -> np.ndarray:
    image = np.full((height, width, 4), (70, 115, 165, 255), dtype=np.uint8)
    image[25:75, 35:85, :3] = (190, 60, 45)
    return image


def test_postprocess_removes_speck_but_keeps_soft_edge_and_thin_detail() -> None:
    image = rgba()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, (60, 50), (25, 28), 0, 0, 360, 255, -1, cv2.LINE_AA)
    cv2.line(mask, (60, 25), (95, 5), 220, 1, cv2.LINE_AA)
    mask[3, 3] = 120
    result = postprocess_segmentation(mask, image, "subject", 0.6)
    assert result[3, 3] == 0
    assert result[8, 90] > 0
    assert np.count_nonzero((result > 0) & (result < 255)) > 30


def test_accurate_mode_without_models_reports_real_cpu_fallback() -> None:
    image = rgba()
    service = SegmentationService(primary=None, fallback=OpenCvCpuSegmentationBackend(), requested_accurate=True)
    result = service.select(image, "Объект", 0.6)
    assert result.backend == "OpenCV CPU"
    assert result.fallback
    assert result.mask[50, 60] > 128


def test_subject_and_sky_use_independent_backends() -> None:
    image = rgba()
    subject_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    subject_mask[25:75, 35:85] = 255
    sky_mask = np.zeros_like(subject_mask)
    sky_mask[:35] = 230
    subject = FixedBackend(subject_mask, name="subject model")
    sky = FixedBackend(subject_mask, sky_mask, name="sky model")
    service = SegmentationService(subject, sky_primary=sky, requested_accurate=True)
    subject_result = service.select(image, "Объект")
    sky_result = service.select(image, "Небо")
    assert subject.subject_calls == 1 and subject.sky_calls == 0
    assert sky.sky_calls == 1 and sky.subject_calls == 0
    assert subject_result.backend == "subject model" and not subject_result.fallback
    assert sky_result.backend == "sky model" and not sky_result.fallback


def test_backend_failure_falls_back_without_breaking_selection() -> None:
    image = rgba()
    failing = FixedBackend(np.zeros(image.shape[:2], dtype=np.uint8), fail=True)
    service = SegmentationService(failing, fallback=OpenCvCpuSegmentationBackend())
    result = service.select(image, "Объект", 0.65)
    assert result.fallback and result.backend == "OpenCV CPU"
    assert np.any(result.mask)


def test_model_configuration_is_centralized_and_environment_driven(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UZYRO_SEGMENTATION_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("UZYRO_SUBJECT_MODEL", str(tmp_path / "subject-custom.onnx"))
    monkeypatch.setenv("UZYRO_SKY_MODEL", str(tmp_path / "sky-custom.onnx"))
    monkeypatch.setenv("UZYRO_SEGMENTATION_SIZE", "512")
    config = SegmentationConfig.from_environment()
    assert config.model_dir == tmp_path / "models"
    assert config.subject_model == tmp_path / "subject-custom.onnx"
    assert config.sky_model == tmp_path / "sky-custom.onnx"
    assert config.input_size == 512


def test_fast_sky_handles_gray_sunset_and_night_without_selecting_ground() -> None:
    for sky_color in ((105, 112, 125), (205, 115, 72), (22, 31, 48)):
        image = np.full((140, 180, 4), (*sky_color, 255), dtype=np.uint8)
        image[95:, :, :3] = (55, 72, 44)
        cv2.rectangle(image, (82, 28), (96, 140), (35, 38, 42, 255), -1)
        mask = sky_selection_mask(image, 0.72)
        assert mask[12, 20] > 180
        assert mask[120, 20] < 30
        assert mask[60, 89] < 80
        assert np.any((mask > 0) & (mask < 255))
