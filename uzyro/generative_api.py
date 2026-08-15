from __future__ import annotations

from dataclasses import dataclass
import math
import secrets

import cv2
import numpy as np


MAX_SEED = 4_294_967_294


class GenerativeAPIError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, code: str | None = None, request_id: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.request_id = request_id


@dataclass
class GeneratedVariant:
    pixels: np.ndarray
    seed: int


def variant_seeds(seed: int, count: int) -> list[int]:
    count = max(1, min(4, int(count)))
    base = int(seed)
    if base <= 0 or base > MAX_SEED:
        base = secrets.randbelow(MAX_SEED) + 1
    return [((base - 1 + index) % MAX_SEED) + 1 for index in range(count)]


def _resize_rgba(pixels: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if pixels.shape[1::-1] == size:
        return np.ascontiguousarray(pixels)
    return np.ascontiguousarray(cv2.resize(pixels, size, interpolation=cv2.INTER_LANCZOS4))


def inpaint_proxy(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, math.sqrt(8_500_000 / max(1, width * height)))
    if min(width, height) * scale < 64:
        scale = max(scale, 64 / min(width, height))
    size = max(64, round(width * scale)), max(64, round(height * scale))
    resized_image = _resize_rgba(image, size)
    resized_mask = mask if mask.shape == resized_image.shape[:2] else cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)
    return resized_image, np.ascontiguousarray(resized_mask.astype(np.uint8)), scale


def outpaint_proxy(image: np.ndarray, margins: tuple[int, int, int, int]) -> tuple[np.ndarray, tuple[int, int, int, int], float]:
    left, top, right, bottom = (max(0, int(value)) for value in margins)
    target_width = image.shape[1] + left + right
    target_height = image.shape[0] + top + bottom
    scale = min(1.0, math.sqrt(8_500_000 / max(1, target_width * target_height)))
    maximum_margin = max(left, top, right, bottom)
    if maximum_margin:
        scale = min(scale, 2000 / maximum_margin)
    minimum_scale = 64 / min(image.shape[1], image.shape[0])
    if minimum_scale <= 1.0 and maximum_margin * minimum_scale <= 2000:
        scale = max(scale, minimum_scale)
    proxy_size = max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))
    proxy_margins = tuple(max(0, min(2000, round(value * scale))) for value in (left, top, right, bottom))
    return _resize_rgba(image, proxy_size), proxy_margins, scale


def validate_outpaint_dimensions(image: np.ndarray, margins: tuple[int, int, int, int]) -> tuple[int, int]:
    left, top, right, bottom = (max(0, int(value)) for value in margins)
    if not any((left, top, right, bottom)):
        raise GenerativeAPIError("Укажите расширение хотя бы с одной стороны")
    width = image.shape[1] + left + right
    height = image.shape[0] + top + bottom
    if width > 50_000 or height > 50_000:
        raise GenerativeAPIError("Итоговый холст не должен превышать 50000 x 50000 px")
    ratio = width / max(1, height)
    if not 0.4 <= ratio <= 2.5:
        raise GenerativeAPIError("Соотношение сторон результата должно быть от 1:2,5 до 2,5:1")
    return width, height


def strict_inpaint_result(source: np.ndarray, generated: np.ndarray, mask: np.ndarray) -> np.ndarray:
    generated = _resize_rgba(generated, source.shape[1::-1])
    alpha = np.clip(mask.astype(np.float32) / 255.0, 0.0, 1.0)[:, :, None]
    return np.clip(source.astype(np.float32) * (1.0 - alpha) + generated.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def strict_outpaint_result(source: np.ndarray, generated: np.ndarray, margins: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = margins
    size = source.shape[1] + left + right, source.shape[0] + top + bottom
    output = _resize_rgba(generated, size)
    output[top:top + source.shape[0], left:left + source.shape[1]] = source
    return output


__all__ = [
    "GeneratedVariant", "GenerativeAPIError", "MAX_SEED", "inpaint_proxy", "outpaint_proxy",
    "strict_inpaint_result", "strict_outpaint_result", "validate_outpaint_dimensions", "variant_seeds",
]
