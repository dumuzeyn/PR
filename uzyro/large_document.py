from __future__ import annotations

from collections import OrderedDict
import hashlib
import os
from pathlib import Path
import shutil
from typing import Hashable

import cv2
import numpy as np

from .gpu_acceleration import acceleration_status, should_accelerate
from .security.temporary import temporary_root


def gpu_status() -> dict[str, object]:
    status = acceleration_status()
    return {
        **status,
        "devices": int(status["cuda_devices"]) if status["backend"] == "cuda" else int(bool(status["opencl_available"])),
    }


def pyr_down(array: np.ndarray, use_gpu: bool = True) -> np.ndarray:
    """Downsample on CUDA when available and fall back to the identical CPU path."""
    status = gpu_status()
    if use_gpu and should_accelerate(array):
        try:
            if status["backend"] == "cuda":
                source = cv2.cuda_GpuMat()
                source.upload(array)
                if hasattr(cv2.cuda, "pyrDown"):
                    return cv2.cuda.pyrDown(source).download()
                size = (max(1, array.shape[1] // 2), max(1, array.shape[0] // 2))
                return cv2.cuda.resize(source, size, interpolation=cv2.INTER_AREA).download()
            return cv2.pyrDown(cv2.UMat(array)).get()
        except (AttributeError, cv2.error):
            pass
    return cv2.pyrDown(array)


class ScratchCache:
    """LRU ndarray cache that spills older entries to a private scratch folder."""

    def __init__(self, memory_limit: int = 256 * 1024 * 1024, directory: str | Path | None = None) -> None:
        self.memory_limit = max(0, int(memory_limit))
        if directory is None:
            import tempfile
            self.directory = Path(tempfile.mkdtemp(prefix="cache-", dir=temporary_root()))
        else:
            self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._memory: OrderedDict[Hashable, np.ndarray] = OrderedDict()
        self._disk: dict[Hashable, Path] = {}
        self._bytes = 0

    def _path(self, key: Hashable) -> Path:
        token = hashlib.sha256(repr(key).encode("utf-8", "replace")).hexdigest()
        return self.directory / f"{token}.npy"

    def put(self, key: Hashable, value: np.ndarray) -> None:
        self.delete(key)
        array = np.ascontiguousarray(value)
        self._memory[key] = array
        self._bytes += int(array.nbytes)
        self._evict()

    def get(self, key: Hashable) -> np.ndarray | None:
        if key in self._memory:
            value = self._memory.pop(key)
            self._memory[key] = value
            return value
        path = self._disk.pop(key, None)
        if path is None or not path.exists():
            return None
        value = np.load(path, allow_pickle=False)
        path.unlink(missing_ok=True)
        self._memory[key] = value
        self._bytes += int(value.nbytes)
        self._evict(exclude=key)
        return value

    def _evict(self, exclude: Hashable | None = None) -> None:
        while self._bytes > self.memory_limit and self._memory:
            key, value = next(iter(self._memory.items()))
            if key == exclude and len(self._memory) == 1:
                break
            self._memory.pop(key)
            path = self._path(key)
            np.save(path, value, allow_pickle=False)
            self._disk[key] = path
            self._bytes -= int(value.nbytes)

    def delete(self, key: Hashable) -> None:
        value = self._memory.pop(key, None)
        if value is not None:
            self._bytes -= int(value.nbytes)
        path = self._disk.pop(key, None)
        if path is not None:
            path.unlink(missing_ok=True)

    def clear(self) -> None:
        self._memory.clear()
        self._bytes = 0
        for path in self._disk.values():
            path.unlink(missing_ok=True)
        self._disk.clear()

    @property
    def stats(self) -> dict[str, int]:
        return {"memory_bytes": self._bytes, "memory_items": len(self._memory), "disk_items": len(self._disk)}

    def close(self) -> None:
        self.clear()
        shutil.rmtree(self.directory, ignore_errors=True)


class MipmapPyramid:
    def __init__(self, scratch: ScratchCache | None = None) -> None:
        self.scratch = scratch or ScratchCache()
        self.gpu = gpu_status()

    @staticmethod
    def level_for_zoom(zoom: float) -> int:
        level = 0
        scale = max(float(zoom), 1e-6)
        while scale < 0.5:
            scale *= 2.0
            level += 1
        return level

    def get(self, key: Hashable, source: np.ndarray, level: int) -> np.ndarray:
        level = max(0, int(level))
        current = source
        for index in range(1, level + 1):
            cache_key = ("mipmap", key, index)
            cached = self.scratch.get(cache_key)
            if cached is None:
                current = pyr_down(current, bool(self.gpu["enabled"]))
                self.scratch.put(cache_key, current)
            else:
                current = cached
        return current

    def for_zoom(self, key: Hashable, source: np.ndarray, zoom: float) -> tuple[np.ndarray, int]:
        level = self.level_for_zoom(zoom)
        return self.get(key, source, level), level
