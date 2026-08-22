from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from uzyro.core import alpha_blend_inplace, apply_filter_stack
from uzyro.gpu_acceleration import (
    accelerated_alpha_blend,
    accelerated_gaussian_blur,
    acceleration_metrics,
    acceleration_status,
    benchmark_acceleration,
    calibrate_acceleration,
    reset_acceleration_metrics,
)


@pytest.fixture
def gpu_environment(monkeypatch: pytest.MonkeyPatch):
    reset_acceleration_metrics()
    monkeypatch.setenv("UZYRO_GPU", "force")
    monkeypatch.setenv("UZYRO_GPU_MIN_PIXELS", "1")
    status = acceleration_status()
    if status["backend"] == "cpu":
        pytest.skip("No real OpenCL or CUDA device is available")
    yield status
    reset_acceleration_metrics()


@pytest.mark.hardware
def test_real_gpu_blur_and_filter_stack_match_cpu(gpu_environment, monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(9812)
    source = rng.integers(0, 256, (768, 1024, 4), dtype=np.uint8)
    gpu_blur = accelerated_gaussian_blur(source, (9, 9), 4.0)
    cpu_blur = cv2.GaussianBlur(source, (9, 9), 4.0)
    assert np.max(np.abs(gpu_blur.astype(np.int16) - cpu_blur.astype(np.int16))) <= 1
    filters = [
        {"type": "blur", "radius": 4, "enabled": True},
        {"type": "sharpen", "amount": 0.65, "enabled": True},
        {"type": "emboss", "strength": 0.2, "enabled": True},
    ]
    gpu_result = apply_filter_stack(source, filters)
    monkeypatch.setenv("UZYRO_GPU", "off")
    cpu_result = apply_filter_stack(source, filters)
    assert np.max(np.abs(gpu_result.astype(np.int16) - cpu_result.astype(np.int16))) <= 2
    metrics = acceleration_metrics()
    assert metrics["counts"]["gpu.gaussian_blur"] >= 3
    assert metrics["counts"]["gpu.filter2d"] >= 1


def test_real_gpu_alpha_composite_matches_cpu(gpu_environment, monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(551)
    source = rng.integers(0, 256, (640, 960, 4), dtype=np.uint8)
    destination = rng.integers(0, 256, source.shape, dtype=np.uint8)
    mask = rng.integers(0, 256, source.shape[:2], dtype=np.uint8)
    gpu_result = accelerated_alpha_blend(source, destination, 0.73, mask, 0.64)
    assert gpu_result is not None
    monkeypatch.setenv("UZYRO_GPU", "off")
    cpu_result = destination.copy()
    alpha_blend_inplace(cpu_result, source, 0, 0, 0.73, mask, 0.64, "Normal")
    assert np.max(np.abs(gpu_result.astype(np.int16) - cpu_result.astype(np.int16))) <= 1


def test_auto_mode_calibrates_and_benchmark_reports_real_measurements(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_acceleration_metrics()
    monkeypatch.setenv("UZYRO_GPU", "auto")
    selected = calibrate_acceleration(force=True)
    assert isinstance(selected, bool)
    if not acceleration_status()["enabled"]:
        assert not selected
        return
    calibration = acceleration_metrics()["calibration"]
    assert calibration["cpu_ms"] > 0
    assert calibration["gpu_ms"] > 0
    report = benchmark_acceleration(1024, 768, 2)
    assert report["cpu_blur_ms"] > 0
    assert report["gpu_blur_ms"] > 0
    assert report["blur_max_error"] <= 1
    assert report["blend_executed"]


def test_gpu_can_be_disabled_without_changing_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UZYRO_GPU", "off")
    reset_acceleration_metrics()
    source = np.full((300, 400, 4), (90, 120, 150, 255), dtype=np.uint8)
    result = accelerated_gaussian_blur(source, (7, 7), 2.0)
    np.testing.assert_array_equal(result, cv2.GaussianBlur(source, (7, 7), 2.0))
    assert acceleration_status()["backend"] == "cpu"
    assert acceleration_metrics()["counts"]["cpu.gaussian_blur"] == 1
