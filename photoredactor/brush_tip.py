from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from .render_ops import retouch_falloff_mask


def load_alpha_mask(path: str) -> np.ndarray | None:
    image = _load_image(path)
    if image is None:
        return None
    if image.ndim == 3 and image.shape[2] == 4:
        return np.asarray(image[:, :, 3], dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    return np.asarray(255 - gray if float(border.mean()) > 127.0 else gray, dtype=np.uint8)


def load_texture_mask(path: str) -> np.ndarray | None:
    image = _load_image(path)
    if image is None:
        return None
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3].astype(np.float32) / 255.0
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
        return np.rint(gray * alpha).astype(np.uint8)
    return np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image, dtype=np.uint8)


def _load_image(path: str) -> np.ndarray | None:
    if not path:
        return None
    try:
        payload = np.fromfile(Path(path), dtype=np.uint8)
        image = cv2.imdecode(payload, cv2.IMREAD_UNCHANGED)
    except (OSError, ValueError):
        return None
    if image is None or image.size == 0:
        return None
    return image


def _resource_signature(path: str) -> tuple[str, int, int]:
    if not path:
        return "", 0, 0
    try:
        stat = Path(path).stat()
        return str(Path(path).resolve()), int(stat.st_mtime_ns), int(stat.st_size)
    except OSError:
        return str(path), 0, 0


class BrushResourceCache:
    def __init__(self, capacity: int = 24) -> None:
        self.capacity = max(4, int(capacity))
        self._items: OrderedDict[tuple[object, ...], np.ndarray | None] = OrderedDict()

    def alpha(self, path: str) -> np.ndarray | None:
        return self._get(path, "tip", load_alpha_mask)

    def texture(self, path: str) -> np.ndarray | None:
        return self._get(path, "texture", load_texture_mask)

    def _get(self, path: str, kind: str, loader) -> np.ndarray | None:
        key = (kind, *_resource_signature(path))
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        value = loader(path)
        self._items[key] = value
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)
        return value


class BrushTipCache:
    def __init__(self, capacity: int = 256) -> None:
        self.capacity = max(32, int(capacity))
        self.resources = BrushResourceCache()
        self._items: OrderedDict[tuple[object, ...], np.ndarray] = OrderedDict()

    def stamp(
        self,
        radius: int,
        hardness: float,
        angle: float = 0.0,
        roundness: float = 1.0,
        flip_x: bool = False,
        flip_y: bool = False,
        custom_path: str = "",
        dual_path: str = "",
    ) -> np.ndarray:
        key = (
            max(1, int(radius)), round(float(hardness), 3), round(float(angle) % 360.0, 2),
            round(float(roundness), 3), bool(flip_x), bool(flip_y),
            _resource_signature(custom_path), _resource_signature(dual_path),
        )
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        result = self._build(*key[:6], custom_path, dual_path)
        self._items[key] = result
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)
        return result

    def _build(
        self,
        radius: int,
        hardness: float,
        angle: float,
        roundness: float,
        flip_x: bool,
        flip_y: bool,
        custom_path: str,
        dual_path: str,
    ) -> np.ndarray:
        size = radius * 2 + 1
        custom = self.resources.alpha(custom_path)
        if custom is None:
            base = retouch_falloff_mask(radius, hardness)
        else:
            base = self._fit_mask(custom, size)
            power = 1.8 - float(np.clip(hardness, 0.0, 1.0)) * 1.55
            base = np.power(np.clip(base, 0.0, 1.0), power)
        base = self._transform(base, angle, roundness, flip_x, flip_y)
        dual = self.resources.alpha(dual_path)
        if dual is not None:
            dual_mask = self._transform(self._fit_mask(dual, size), -angle, roundness, False, False)
            base *= dual_mask
        return np.ascontiguousarray(np.clip(base, 0.0, 1.0).astype(np.float32))

    @staticmethod
    def _fit_mask(mask: np.ndarray, size: int) -> np.ndarray:
        source = np.asarray(mask, dtype=np.uint8)
        scale = min(size / max(1, source.shape[1]), size / max(1, source.shape[0]))
        width, height = max(1, round(source.shape[1] * scale)), max(1, round(source.shape[0] * scale))
        resized = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
        canvas = np.zeros((size, size), dtype=np.float32)
        x, y = (size - width) // 2, (size - height) // 2
        canvas[y:y + height, x:x + width] = resized.astype(np.float32) / 255.0
        return canvas

    @staticmethod
    def _transform(mask: np.ndarray, angle: float, roundness: float, flip_x: bool, flip_y: bool) -> np.ndarray:
        result = mask
        if flip_x:
            result = np.fliplr(result)
        if flip_y:
            result = np.flipud(result)
        size = result.shape[0]
        roundness = float(np.clip(roundness, 0.01, 1.0))
        if roundness < 0.999:
            height = max(1, round(size * roundness))
            squeezed = cv2.resize(result, (size, height), interpolation=cv2.INTER_LINEAR)
            result = np.zeros_like(result)
            y = (size - height) // 2
            result[y:y + height] = squeezed
        if abs(float(angle) % 360.0) > 0.01:
            center = ((size - 1) / 2.0, (size - 1) / 2.0)
            matrix = cv2.getRotationMatrix2D(center, -float(angle), 1.0)
            result = cv2.warpAffine(result, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return result


class BrushTexture:
    def __init__(self, resources: BrushResourceCache | None = None) -> None:
        self.resources = resources or BrushResourceCache()

    def apply(
        self,
        mask: np.ndarray,
        path: str,
        document_rect: tuple[int, int, int, int],
        *,
        scale: float = 1.0,
        depth: float = 0.0,
        invert: bool = False,
        stroke_origin: tuple[int, int] | None = None,
        canvas_space: bool = True,
    ) -> np.ndarray:
        texture = self.resources.texture(path)
        depth = float(np.clip(depth, 0.0, 1.0))
        if texture is None or depth <= 0.0:
            return mask
        factor = max(0.05, float(scale))
        width = max(1, round(texture.shape[1] * factor))
        height = max(1, round(texture.shape[0] * factor))
        tiled = cv2.resize(texture, (width, height), interpolation=cv2.INTER_AREA if factor < 1.0 else cv2.INTER_LINEAR)
        if invert:
            tiled = 255 - tiled
        x1, y1, x2, y2 = document_rect
        if not canvas_space and stroke_origin is not None:
            x1 -= int(stroke_origin[0])
            y1 -= int(stroke_origin[1])
        ys = np.mod(np.arange(y1, y1 + mask.shape[0]), height)
        xs = np.mod(np.arange(x1, x1 + mask.shape[1]), width)
        sample = tiled[np.ix_(ys, xs)].astype(np.float32) / 255.0
        return mask * ((1.0 - depth) + sample * depth)


BRUSH_TIP_CACHE = BrushTipCache()
BRUSH_TEXTURE = BrushTexture(BRUSH_TIP_CACHE.resources)


__all__ = ["BRUSH_TEXTURE", "BRUSH_TIP_CACHE", "BrushResourceCache", "BrushTexture", "BrushTipCache", "load_alpha_mask", "load_texture_mask"]
