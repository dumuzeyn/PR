from __future__ import annotations

from dataclasses import dataclass
import io
import json
import math
import secrets
import time
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid

import cv2
import numpy as np
from PIL import Image


MAX_SEED = 4_294_967_294
STABILITY_STYLES = (
    "", "photographic", "cinematic", "digital-art", "anime", "comic-book",
    "fantasy-art", "line-art", "analog-film", "pixel-art", "3d-model",
)


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


def png_bytes(pixels: np.ndarray, grayscale: bool = False) -> bytes:
    image = Image.fromarray(pixels.astype(np.uint8), "L" if grayscale else "RGBA")
    buffer = io.BytesIO()
    image.save(buffer, "PNG", optimize=True)
    return buffer.getvalue()


def encoded_input_image(pixels: np.ndarray) -> tuple[str, bytes, str]:
    encoded = png_bytes(pixels)
    if len(encoded) <= 8 * 1024 * 1024:
        return "image.png", encoded, "image/png"
    image = Image.fromarray(pixels.astype(np.uint8), "RGBA").convert("RGB")
    for quality in (92, 84, 76):
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=quality, optimize=True)
        if len(buffer.getvalue()) <= 8 * 1024 * 1024:
            return "image.jpg", buffer.getvalue(), "image/jpeg"
    raise GenerativeAPIError("Изображение не удалось подготовить в пределах лимита API 10 МБ")


def decode_image(data: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as image:
            return np.asarray(image.convert("RGBA"), dtype=np.uint8)
    except Exception as exc:
        raise GenerativeAPIError("Провайдер вернул данные, которые не являются изображением") from exc


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


def multipart_body(fields: dict[str, object], files: dict[str, tuple[str, bytes, str]]) -> tuple[bytes, str]:
    boundary = f"PhotoRedactor-{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.extend((
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode("utf-8"),
        ))
    for name, (filename, data, content_type) in files.items():
        parts.extend((
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode(), data, b"\r\n",
        ))
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class StabilityImageClient:
    def __init__(self, api_key: str, base_url: str = "https://api.stability.ai", timeout: int = 150) -> None:
        if not str(api_key).strip():
            raise GenerativeAPIError("API-ключ Stability AI не настроен")
        self.api_key = str(api_key).strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = max(10, int(timeout))

    def account(self) -> dict[str, object]:
        request = Request(f"{self.base_url}/v1/user/account", headers=self._headers("application/json"))
        data, _headers = self._open(request)
        return json.loads(data.decode("utf-8"))

    def _headers(self, accept: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": accept,
            "Stability-Client-ID": "PhotoRedactor",
            "Stability-Client-Version": "1.0",
            "User-Agent": "PhotoRedactor/1.0",
        }

    def _open(self, request: Request) -> tuple[bytes, object]:
        for attempt in range(3):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    return response.read(), response.headers
            except HTTPError as exc:
                payload = exc.read()
                request_id = exc.headers.get("x-request-id")
                if exc.code == 429 or exc.code >= 500:
                    if attempt < 2:
                        time.sleep(min(4.0, float(exc.headers.get("retry-after", attempt + 1))))
                        continue
                raise self._api_error(payload, exc.code, request_id) from exc
            except (URLError, TimeoutError, OSError) as exc:
                if attempt < 2:
                    time.sleep(attempt + 1)
                    continue
                raise GenerativeAPIError(f"Не удалось связаться со Stability AI: {exc}") from exc
        raise GenerativeAPIError("Stability AI не ответил после повторных попыток")

    @staticmethod
    def _api_error(payload: bytes, status: int, request_id: str | None) -> GenerativeAPIError:
        try:
            data = json.loads(payload.decode("utf-8"))
            raw = data.get("errors") or data.get("message") or data.get("name") or data
            message = "; ".join(map(str, raw)) if isinstance(raw, list) else str(raw)
            code = str(data.get("name") or data.get("id") or "") or None
        except Exception:
            message, code = payload.decode("utf-8", "replace")[:500] or f"HTTP {status}", None
        labels = {401: "Неверный API-ключ", 403: "Запрос отклонён модерацией", 413: "Изображение слишком велико", 429: "Превышен лимит запросов"}
        return GenerativeAPIError(f"{labels.get(status, 'Ошибка Stability AI')}: {message}", status, code, request_id)

    def _post_image(self, endpoint: str, fields: dict[str, object], files: dict[str, tuple[str, bytes, str]]) -> np.ndarray:
        body, content_type = multipart_body(fields, files)
        headers = {**self._headers("image/*"), "Content-Type": content_type}
        data, _response_headers = self._open(Request(f"{self.base_url}{endpoint}", data=body, headers=headers, method="POST"))
        return decode_image(data)

    def inpaint(self, image: np.ndarray, mask: np.ndarray, prompt: str, negative_prompt: str, seed: int, style: str = "") -> np.ndarray:
        proxy, proxy_mask, _scale = inpaint_proxy(image, mask)
        fields: dict[str, object] = {"prompt": prompt, "seed": seed, "output_format": "png"}
        if negative_prompt.strip():
            fields["negative_prompt"] = negative_prompt.strip()
        if style in STABILITY_STYLES and style:
            fields["style_preset"] = style
        generated = self._post_image(
            "/v2beta/stable-image/edit/inpaint", fields,
            {"image": encoded_input_image(proxy), "mask": ("mask.png", png_bytes(proxy_mask, True), "image/png")},
        )
        return _resize_rgba(generated, image.shape[1::-1])

    def outpaint(self, image: np.ndarray, margins: tuple[int, int, int, int], prompt: str, seed: int, creativity: float, style: str = "") -> np.ndarray:
        validate_outpaint_dimensions(image, margins)
        proxy, proxy_margins, _scale = outpaint_proxy(image, margins)
        left, top, right, bottom = proxy_margins
        if not any(proxy_margins):
            raise GenerativeAPIError("Укажите расширение хотя бы с одной стороны")
        fields: dict[str, object] = {
            "left": left, "up": top, "right": right, "down": bottom,
            "prompt": prompt.strip(), "seed": seed, "creativity": float(np.clip(creativity, 0.0, 1.0)), "output_format": "png",
        }
        if style in STABILITY_STYLES and style:
            fields["style_preset"] = style
        generated = self._post_image(
            "/v2beta/stable-image/edit/outpaint", fields,
            {"image": encoded_input_image(proxy)},
        )
        return strict_outpaint_result(image, generated, margins)

    def variants(self, operation: Callable[[int], np.ndarray], seed: int, count: int) -> list[GeneratedVariant]:
        return [GeneratedVariant(operation(value), value) for value in variant_seeds(seed, count)]


__all__ = [
    "GeneratedVariant", "GenerativeAPIError", "MAX_SEED", "STABILITY_STYLES", "StabilityImageClient",
    "decode_image", "encoded_input_image", "inpaint_proxy", "multipart_body", "outpaint_proxy", "strict_inpaint_result",
    "strict_outpaint_result", "validate_outpaint_dimensions", "variant_seeds",
]
