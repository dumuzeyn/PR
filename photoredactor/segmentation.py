from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from pathlib import Path

import cv2
import numpy as np

from .selection_ops import background_selection_mask, sky_selection_mask, subject_selection_mask


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

    def __init__(self, model_path: Path, prefer_gpu: bool = False) -> None:
        self.model_path = Path(model_path)
        self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
        self.name = "ONNX GPU" if prefer_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0 else "ONNX CPU"
        if self.name == "ONNX GPU":
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def select_subject(self, image: np.ndarray, sensitivity: float = 0.5) -> np.ndarray:
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image[:, :, :3], 1.0 / 255.0, (320, 320), swapRB=False, crop=False)
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


class SegmentationService:
    def __init__(self, primary: SegmentationBackend | None = None, fallback: SegmentationBackend | None = None) -> None:
        self.primary = primary
        self.fallback = fallback or OpenCvCpuSegmentationBackend()

    @classmethod
    def from_local_resources(cls) -> "SegmentationService":
        root = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "PhotoRedactor" / "models" / "segmentation"
        model = root / "foreground.onnx"
        if model.is_file():
            try:
                return cls(OnnxForegroundSegmentationBackend(model, prefer_gpu=True))
            except (cv2.error, OSError, ValueError):
                pass
        return cls()

    @property
    def backend_name(self) -> str:
        return (self.primary or self.fallback).name

    @staticmethod
    def model_folder() -> Path:
        return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "PhotoRedactor" / "models" / "segmentation"

    def select(
        self, image: np.ndarray, target: str, sensitivity: float = 0.5,
        box: tuple[int, int, int, int] | None = None,
    ) -> SegmentationResult:
        normalized = str(target).lower()
        backend = self.primary or self.fallback
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
            return SegmentationResult(mask.astype(np.uint8), backend.name, False)
        except (cv2.error, OSError, RuntimeError, ValueError):
            if backend is self.fallback:
                raise
            return SegmentationResult(self.select_with(self.fallback, image, normalized, sensitivity, box), self.fallback.name, True)

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
