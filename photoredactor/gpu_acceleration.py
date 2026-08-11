from __future__ import annotations

from collections import defaultdict
import os
import time
from typing import Callable

import cv2
import numpy as np


_timings: dict[str, list[float]] = defaultdict(list)
_counts: dict[str, int] = defaultdict(int)
_auto_decision: bool | None = None
_calibration: dict[str, float | bool] = {}


def acceleration_status() -> dict[str, object]:
    mode = os.environ.get("PHOTO_REDACTOR_GPU", "auto").lower()
    requested = mode not in {"0", "false", "off"}
    cuda_devices = 0
    try:
        cuda_devices = int(cv2.cuda.getCudaEnabledDeviceCount())
    except (AttributeError, cv2.error):
        pass
    opencl_available = bool(cv2.ocl.haveOpenCL())
    if requested and opencl_available:
        cv2.ocl.setUseOpenCL(True)
    opencl_enabled = requested and opencl_available and bool(cv2.ocl.useOpenCL())
    device_name = ""
    device_vendor = ""
    if opencl_available:
        try:
            device = cv2.ocl.Device_getDefault()
            device_name = str(device.name())
            device_vendor = str(device.vendorName())
        except (AttributeError, cv2.error):
            opencl_enabled = False
    backend = "cuda" if requested and cuda_devices > 0 else "opencl" if opencl_enabled else "cpu"
    return {
        "available": cuda_devices > 0 or opencl_available,
        "enabled": backend != "cpu",
        "backend": backend,
        "device": device_name,
        "vendor": device_vendor,
        "cuda_devices": cuda_devices,
        "opencl_available": opencl_available,
        "mode": "off" if not requested else "force" if mode in {"1", "true", "on", "force"} else "auto",
        "auto_selected": _auto_decision,
    }


def minimum_pixels() -> int:
    return max(1, int(os.environ.get("PHOTO_REDACTOR_GPU_MIN_PIXELS", "262144")))


def should_accelerate(array: np.ndarray) -> bool:
    status = acceleration_status()
    if not bool(status["enabled"]) or int(array.shape[0] * array.shape[1]) < minimum_pixels():
        return False
    if status["mode"] == "force":
        return True
    return calibrate_acceleration()


def calibrate_acceleration(force: bool = False) -> bool:
    global _auto_decision, _calibration
    if _auto_decision is not None and not force:
        return _auto_decision
    status = acceleration_status()
    if not status["enabled"]:
        _auto_decision = False
        _calibration = {"selected": False, "cpu_ms": 0.0, "gpu_ms": 0.0, "speedup": 0.0}
        return False
    rng = np.random.default_rng(31991)
    sample = rng.integers(0, 256, (768, 1024, 4), dtype=np.uint8)
    try:
        cv2.GaussianBlur(cv2.UMat(sample), (9, 9), 4.0).get()
        started = time.perf_counter()
        for _ in range(3):
            cv2.GaussianBlur(sample, (9, 9), 4.0)
        cpu_ms = (time.perf_counter() - started) * 1000.0 / 3.0
        started = time.perf_counter()
        for _ in range(3):
            cv2.GaussianBlur(cv2.UMat(sample), (9, 9), 4.0).get()
        gpu_ms = (time.perf_counter() - started) * 1000.0 / 3.0
        _auto_decision = gpu_ms < cpu_ms * 0.9
        _calibration = {
            "selected": _auto_decision,
            "cpu_ms": cpu_ms,
            "gpu_ms": gpu_ms,
            "speedup": cpu_ms / max(gpu_ms, 1e-9),
        }
    except (cv2.error, AttributeError):
        _auto_decision = False
        _calibration = {"selected": False, "cpu_ms": 0.0, "gpu_ms": 0.0, "speedup": 0.0}
    return _auto_decision


def _timed(name: str, backend: str, operation: Callable[[], np.ndarray]) -> np.ndarray:
    started = time.perf_counter()
    result = operation()
    elapsed = (time.perf_counter() - started) * 1000.0
    key = f"{backend}.{name}"
    _counts[key] += 1
    samples = _timings[key]
    samples.append(elapsed)
    if len(samples) > 100:
        del samples[:-100]
    return result


def _gpu_or_cpu(name: str, array: np.ndarray, gpu, cpu) -> np.ndarray:
    if should_accelerate(array):
        try:
            return _timed(name, "gpu", lambda: gpu(cv2.UMat(array)).get())
        except (cv2.error, AttributeError):
            _counts[f"gpu.{name}.fallback"] += 1
    return _timed(name, "cpu", cpu)


def accelerated_resize(array: np.ndarray, size: tuple[int, int], interpolation: int) -> np.ndarray:
    return _gpu_or_cpu(
        "resize",
        array,
        lambda source: cv2.resize(source, size, interpolation=interpolation),
        lambda: cv2.resize(array, size, interpolation=interpolation),
    )


def accelerated_gaussian_blur(
    array: np.ndarray,
    kernel: tuple[int, int],
    sigma: float,
    border_type: int = cv2.BORDER_DEFAULT,
) -> np.ndarray:
    return _gpu_or_cpu(
        "gaussian_blur",
        array,
        lambda source: cv2.GaussianBlur(source, kernel, sigma, borderType=border_type),
        lambda: cv2.GaussianBlur(array, kernel, sigma, borderType=border_type),
    )


def accelerated_median_blur(array: np.ndarray, size: int) -> np.ndarray:
    return _gpu_or_cpu(
        "median_blur",
        array,
        lambda source: cv2.medianBlur(source, int(size)),
        lambda: cv2.medianBlur(array, int(size)),
    )


def accelerated_filter2d(array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return _gpu_or_cpu(
        "filter2d",
        array,
        lambda source: cv2.filter2D(source, -1, kernel),
        lambda: cv2.filter2D(array, -1, kernel),
    )


def accelerated_canny(array: np.ndarray, low: float, high: float) -> np.ndarray:
    return _gpu_or_cpu(
        "canny",
        array,
        lambda source: cv2.Canny(source, low, high),
        lambda: cv2.Canny(array, low, high),
    )


def accelerated_alpha_blend(
    source: np.ndarray,
    destination: np.ndarray,
    opacity: float,
    alpha_mask: np.ndarray | None,
    mask_density: float,
) -> np.ndarray | None:
    if not should_accelerate(source):
        return None

    def blend() -> np.ndarray:
        source_channels = cv2.split(cv2.UMat(source.astype(np.float32)))
        destination_channels = cv2.split(cv2.UMat(destination.astype(np.float32)))
        source_alpha = cv2.multiply(source_channels[3], float(opacity) / 255.0)
        if alpha_mask is not None:
            mask = cv2.UMat(alpha_mask.astype(np.float32) / 255.0)
            coverage = cv2.add(float(1.0 - mask_density), cv2.multiply(mask, float(mask_density)))
            source_alpha = cv2.multiply(source_alpha, coverage)
        destination_alpha = cv2.multiply(destination_channels[3], 1.0 / 255.0)
        ones = cv2.UMat(np.ones(source.shape[:2], dtype=np.float32))
        inverse = cv2.subtract(ones, source_alpha)
        output_alpha = cv2.add(source_alpha, cv2.multiply(destination_alpha, inverse))
        denominator = cv2.max(output_alpha, 1e-6)
        output_channels = []
        destination_weight = cv2.multiply(destination_alpha, inverse)
        for source_channel, destination_channel in zip(source_channels[:3], destination_channels[:3]):
            numerator = cv2.add(
                cv2.multiply(source_channel, source_alpha),
                cv2.multiply(destination_channel, destination_weight),
            )
            output_channels.append(cv2.divide(numerator, denominator))
        output_channels.append(cv2.multiply(output_alpha, 255.0))
        return cv2.merge(output_channels).get()

    try:
        result = _timed("alpha_blend", "gpu", blend)
        return np.clip(result, 0, 255).astype(np.uint8)
    except (cv2.error, AttributeError):
        _counts["gpu.alpha_blend.fallback"] += 1
        return None


def acceleration_metrics() -> dict[str, object]:
    return {
        "counts": dict(_counts),
        "average_ms": {key: sum(values) / len(values) for key, values in _timings.items() if values},
        "calibration": dict(_calibration),
    }


def reset_acceleration_metrics() -> None:
    global _auto_decision, _calibration
    _counts.clear()
    _timings.clear()
    _auto_decision = None
    _calibration = {}


def benchmark_acceleration(width: int = 2048, height: int = 1536, iterations: int = 3) -> dict[str, object]:
    rng = np.random.default_rng(20260811)
    source = rng.integers(0, 256, (height, width, 4), dtype=np.uint8)
    destination = rng.integers(0, 256, source.shape, dtype=np.uint8)
    destination[:, :, 3] = 255
    previous_threshold = os.environ.get("PHOTO_REDACTOR_GPU_MIN_PIXELS")
    previous_mode = os.environ.get("PHOTO_REDACTOR_GPU")
    os.environ["PHOTO_REDACTOR_GPU_MIN_PIXELS"] = "1"
    os.environ["PHOTO_REDACTOR_GPU"] = "force"
    try:
        cv2.GaussianBlur(cv2.UMat(source), (9, 9), 4.0).get()
        started = time.perf_counter()
        cpu_blur = None
        for _ in range(max(1, iterations)):
            cpu_blur = cv2.GaussianBlur(source, (9, 9), 4.0)
        cpu_ms = (time.perf_counter() - started) * 1000.0 / max(1, iterations)
        started = time.perf_counter()
        gpu_blur = None
        for _ in range(max(1, iterations)):
            gpu_blur = cv2.GaussianBlur(cv2.UMat(source), (9, 9), 4.0).get()
        gpu_ms = (time.perf_counter() - started) * 1000.0 / max(1, iterations)
        gpu_blend = accelerated_alpha_blend(source, destination, 0.72, None, 1.0)
        return {
            "status": acceleration_status(),
            "size": [width, height],
            "cpu_blur_ms": cpu_ms,
            "gpu_blur_ms": gpu_ms,
            "blur_speedup": cpu_ms / max(gpu_ms, 1e-9),
            "blur_max_error": int(np.max(np.abs(cpu_blur.astype(np.int16) - gpu_blur.astype(np.int16)))),
            "blend_executed": gpu_blend is not None,
        }
    finally:
        if previous_threshold is None:
            os.environ.pop("PHOTO_REDACTOR_GPU_MIN_PIXELS", None)
        else:
            os.environ["PHOTO_REDACTOR_GPU_MIN_PIXELS"] = previous_threshold
        if previous_mode is None:
            os.environ.pop("PHOTO_REDACTOR_GPU", None)
        else:
            os.environ["PHOTO_REDACTOR_GPU"] = previous_mode


__all__ = [name for name in globals() if not name.startswith("__")]
