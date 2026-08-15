from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .segmentation_config import SegmentationConfig
from .selection_ops import background_selection_mask, sky_selection_mask, smart_radius_refine, subject_selection_mask


@dataclass(frozen=True)
class SegmentationResult:
    mask: np.ndarray
    backend: str
    fallback: bool = False


class SegmentationBackend(ABC):
    name = "segmentation"

    @abstractmethod
    def select_subject(self, image: np.ndarray, sensitivity: float = 0.5) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def select_object(self, image: np.ndarray, box: tuple[int, int, int, int], sensitivity: float = 0.5) -> np.ndarray:
        raise NotImplementedError

    def select_background(self, image: np.ndarray, sensitivity: float = 0.5) -> np.ndarray:
        return 255 - self.select_subject(image, sensitivity)

    def select_sky(self, image: np.ndarray, sensitivity: float = 0.5) -> np.ndarray:
        return sky_selection_mask(image, sensitivity)


class OpenCvCpuSegmentationBackend(SegmentationBackend):
    name = "OpenCV CPU"

    def select_subject(self, image: np.ndarray, sensitivity: float = 0.5) -> np.ndarray:
        return subject_selection_mask(image, sensitivity)

    def select_object(self, image: np.ndarray, box: tuple[int, int, int, int], sensitivity: float = 0.5) -> np.ndarray:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = box
        x1, x2 = sorted((max(0, min(width - 1, int(x1))), max(1, min(width, int(x2)))))
        y1, y2 = sorted((max(0, min(height - 1, int(y1))), max(1, min(height, int(y2)))))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return np.zeros((height, width), dtype=np.uint8)
        rgb = np.asarray(image[:, :, :3], dtype=np.uint8)
        alpha = np.asarray(image[:, :, 3], dtype=np.uint8)
        work_scale = min(1.0, 720.0 / max(height, width))
        work_size = (max(2, round(width * work_scale)), max(2, round(height * work_scale)))
        work = rgb if work_size == (width, height) else cv2.resize(rgb, work_size, interpolation=cv2.INTER_AREA)
        sx1, sy1 = round(x1 * work_scale), round(y1 * work_scale)
        sx2, sy2 = round(x2 * work_scale), round(y2 * work_scale)
        rect = (sx1, sy1, max(2, sx2 - sx1), max(2, sy2 - sy1))
        mask = np.zeros(work.shape[:2], dtype=np.uint8)
        bg_model = np.zeros((1, 65), dtype=np.float64)
        fg_model = np.zeros((1, 65), dtype=np.float64)
        try:
            cv2.grabCut(work, mask, rect, bg_model, fg_model, 4, cv2.GC_INIT_WITH_RECT)
            binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except cv2.error:
            binary = np.zeros(work.shape[:2], dtype=np.uint8)
            binary[sy1:sy2, sx1:sx2] = 255
        radius = max(1, round(min(work.shape[:2]) * (0.002 + (1.0 - float(sensitivity)) * 0.004)))
        soft = cv2.GaussianBlur(binary, (radius * 2 + 1, radius * 2 + 1), radius)
        if work_size != (width, height):
            soft = cv2.resize(soft, (width, height), interpolation=cv2.INTER_LINEAR)
        return np.minimum(soft, alpha).astype(np.uint8)


class OnnxForegroundSegmentationBackend(SegmentationBackend):
    """Runs a local one-channel foreground ONNX model through real OpenCV DNN."""

    def __init__(self, model_path: Path, prefer_gpu: bool = False, input_size: int = 320) -> None:
        self.model_path = Path(model_path)
        self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
        self.input_size = max(128, int(input_size))
        self.name = "ONNX GPU" if prefer_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0 else "ONNX CPU"
        if self.name == "ONNX GPU":
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def select_subject(self, image: np.ndarray, sensitivity: float = 0.5) -> np.ndarray:
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image[:, :, :3], 1.0 / 255.0, (self.input_size, self.input_size), swapRB=False, crop=False)
        self.net.setInput(blob)
        output = np.asarray(self.net.forward()).squeeze()
        if output.ndim != 2:
            raise ValueError(f"Неподдерживаемый выход ONNX-модели: {output.shape}")
        output = 1.0 / (1.0 + np.exp(-output)) if float(output.min()) < 0.0 or float(output.max()) > 1.0 else output
        threshold = 0.35 + (1.0 - float(np.clip(sensitivity, 0.0, 1.0))) * 0.3
        matte = np.clip((output - threshold + 0.18) / 0.36, 0.0, 1.0)
        mask = cv2.resize((matte * 255.0).astype(np.uint8), (width, height), interpolation=cv2.INTER_LINEAR)
        return np.minimum(mask, image[:, :, 3]).astype(np.uint8)

    def select_object(self, image: np.ndarray, box: tuple[int, int, int, int], sensitivity: float = 0.5) -> np.ndarray:
        mask = self.select_subject(image, sensitivity)
        gate = np.zeros(mask.shape, dtype=np.uint8)
        x1, y1, x2, y2 = box
        gate[max(0, y1):min(mask.shape[0], y2), max(0, x1):min(mask.shape[1], x2)] = 255
        return cv2.bitwise_and(mask, gate)


class OnnxSkySegmentationBackend(OnnxForegroundSegmentationBackend):
    def select_sky(self, image: np.ndarray, sensitivity: float = 0.5) -> np.ndarray:
        return super().select_subject(image, sensitivity)

    def select_subject(self, image: np.ndarray, sensitivity: float = 0.5) -> np.ndarray:
        return subject_selection_mask(image, sensitivity)


def postprocess_segmentation(mask: np.ndarray, image: np.ndarray, target: str, sensitivity: float) -> np.ndarray:
    alpha = np.clip(np.asarray(mask), 0, 255).astype(np.uint8)
    if alpha.shape != image.shape[:2]:
        alpha = cv2.resize(alpha, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    alpha = np.minimum(alpha, image[:, :, 3])
    if not np.any(alpha):
        return alpha
    height, width = alpha.shape
    work_scale = min(1.0, 1600.0 / max(height, width))
    if work_scale < 1.0:
        work_size = (max(2, round(width * work_scale)), max(2, round(height * work_scale)))
        work_mask = cv2.resize(alpha, work_size, interpolation=cv2.INTER_AREA)
        work_image = cv2.resize(image, work_size, interpolation=cv2.INTER_AREA)
        refined = postprocess_segmentation(work_mask, work_image, target, sensitivity)
        restored = cv2.resize(refined, (width, height), interpolation=cv2.INTER_LINEAR)
        return np.minimum(restored, image[:, :, 3]).astype(np.uint8)
    binary = (alpha >= 96).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    minimum_area = max(3, round(binary.size * (0.00008 + (1.0 - float(sensitivity)) * 0.00012)))
    keep = np.zeros_like(binary)
    sky = target in {"sky", "небо"}
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        component = labels == index
        top_connected = y <= max(2, image.shape[0] // 50)
        thin_detail = max(width, height) >= 4 * max(1, min(width, height)) and area >= 3
        high_confidence = int(alpha[component].max(initial=0)) >= 210
        if area >= minimum_area or thin_detail or high_confidence or (sky and top_connected):
            keep[component] = 1
    if not np.any(keep):
        return alpha
    holes = (keep == 0).astype(np.uint8)
    hole_count, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(holes, 8)
    for index in range(1, hole_count):
        x, y, width, height, area = (int(value) for value in hole_stats[index])
        touches_border = x == 0 or y == 0 or x + width == keep.shape[1] or y + height == keep.shape[0]
        if not touches_border and area <= minimum_area * 6:
            keep[hole_labels == index] = 1
    cleaned = np.where(keep > 0, alpha, 0).astype(np.uint8)
    refined = smart_radius_refine(cleaned, image, max(2, min(image.shape[:2]) // 180), 0.45)
    detail_support = cv2.dilate(keep, np.ones((3, 3), dtype=np.uint8)) > 0
    fine = np.where(detail_support & (alpha > 0) & (alpha < 160), alpha, 0).astype(np.uint8)
    return np.maximum(refined, np.rint(fine.astype(np.float32) * 0.85).astype(np.uint8))


class SegmentationService:
    def __init__(
        self, primary: SegmentationBackend | None = None, fallback: SegmentationBackend | None = None,
        sky_primary: SegmentationBackend | None = None, requested_accurate: bool = False,
        config: SegmentationConfig | None = None,
    ) -> None:
        self.primary = primary
        self.sky_primary = sky_primary
        self.fallback = fallback or OpenCvCpuSegmentationBackend()
        self.requested_accurate = bool(requested_accurate)
        self.config = config or SegmentationConfig.from_environment()

    @classmethod
    def from_local_resources(cls, quality: str = "accurate") -> "SegmentationService":
        config = SegmentationConfig.from_environment()
        if str(quality).lower() in {"fast", "быстрый"}:
            return cls(config=config)
        subject_backend = None
        sky_backend = None
        if config.subject_model.is_file():
            try:
                subject_backend = OnnxForegroundSegmentationBackend(config.subject_model, config.prefer_gpu, config.input_size)
            except (cv2.error, OSError, ValueError):
                pass
        if config.sky_model.is_file():
            try:
                sky_backend = OnnxSkySegmentationBackend(config.sky_model, config.prefer_gpu, config.input_size)
            except (cv2.error, OSError, ValueError):
                pass
        return cls(subject_backend, sky_primary=sky_backend, requested_accurate=True, config=config)

    @property
    def backend_name(self) -> str:
        active = [backend.name for backend in (self.primary, self.sky_primary) if backend is not None]
        return " / ".join(active) if active else self.fallback.name

    @staticmethod
    def model_folder() -> Path:
        return SegmentationConfig.from_environment().model_dir

    def select(
        self, image: np.ndarray, target: str, sensitivity: float = 0.5,
        box: tuple[int, int, int, int] | None = None,
    ) -> SegmentationResult:
        normalized = str(target).lower()
        sky_target = normalized in {"sky", "небо"}
        configured = self.sky_primary if sky_target else self.primary
        backend = configured or self.fallback
        missing_accurate = self.requested_accurate and configured is None
        try:
            if normalized in {"object roi", "объект в области"}:
                if box is None:
                    raise ValueError("Для выделения объекта требуется область")
                mask = backend.select_object(image, box, sensitivity)
            elif normalized in {"subject", "object", "объект"}:
                mask = backend.select_subject(image, sensitivity)
            elif normalized in {"background", "фон"}:
                mask = backend.select_background(image, sensitivity)
            elif normalized in {"sky", "небо"}:
                mask = backend.select_sky(image, sensitivity)
            else:
                raise ValueError(f"Неизвестная цель выделения: {target}")
            mask = postprocess_segmentation(mask, image, normalized, sensitivity)
            return SegmentationResult(mask.astype(np.uint8), backend.name, missing_accurate)
        except (cv2.error, OSError, RuntimeError, ValueError):
            if backend is self.fallback:
                raise
            mask = self.select_with(self.fallback, image, normalized, sensitivity, box)
            return SegmentationResult(postprocess_segmentation(mask, image, normalized, sensitivity), self.fallback.name, True)

    @staticmethod
    def select_with(
        backend: SegmentationBackend, image: np.ndarray, target: str, sensitivity: float,
        box: tuple[int, int, int, int] | None,
    ) -> np.ndarray:
        if target in {"object roi", "объект в области"} and box is not None:
            return backend.select_object(image, box, sensitivity)
        if target in {"background", "фон"}:
            return backend.select_background(image, sensitivity)
        if target in {"sky", "небо"}:
            return backend.select_sky(image, sensitivity)
        return backend.select_subject(image, sensitivity)
