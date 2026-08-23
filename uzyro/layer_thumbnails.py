from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .layer import Layer
from .render_ops import render_layer_pixels, render_layer_style
from .selection_ops import effective_layer_mask


def _display_rgba(pixels: np.ndarray) -> np.ndarray:
    source = np.asarray(pixels)
    if source.dtype != np.uint8:
        maximum = 1.0 if np.issubdtype(source.dtype, np.floating) and float(source.max(initial=0)) <= 1.0 else 255.0
        source = np.clip(source.astype(np.float32) * (255.0 / maximum), 0, 255).astype(np.uint8)
    if source.ndim == 2:
        source = cv2.cvtColor(source, cv2.COLOR_GRAY2RGBA)
    elif source.shape[2] == 3:
        source = cv2.cvtColor(source, cv2.COLOR_RGB2RGBA)
    return np.ascontiguousarray(source[:, :, :4])


def _apply_layer_alpha(layer: Layer, pixels: np.ndarray) -> np.ndarray:
    has_mask = layer.mask is not None and layer.mask_enabled
    if not has_mask and layer.opacity >= 1.0:
        return pixels
    result = pixels.copy()
    alpha = result[:, :, 3].astype(np.float32)
    if has_mask:
        mask = effective_layer_mask(layer).astype(np.float32) / 255.0
        density = float(np.clip(layer.mask_density, 0.0, 1.0))
        alpha *= (1.0 - density) + mask * density
    alpha *= float(np.clip(layer.opacity, 0.0, 1.0))
    result[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    return result


def _visible_part(
    pixels: np.ndarray,
    x: int,
    y: int,
    opacity: float,
    max_dimension: int | None = None,
) -> tuple[Image.Image, int, int] | None:
    source = _display_rgba(pixels)
    alpha = source[:, :, 3]
    left, top, width, height = cv2.boundingRect(alpha)
    if width <= 0 or height <= 0 or opacity <= 0:
        return None
    right, bottom = left + width, top + height
    cropped = source[top:bottom, left:right].copy()
    if opacity < 1.0:
        cropped[:, :, 3] = np.clip(cropped[:, :, 3].astype(np.float32) * opacity, 0, 255).astype(np.uint8)
    if max_dimension is not None and max(cropped.shape[:2]) > max_dimension:
        scale = max_dimension / max(cropped.shape[:2])
        output_size = max(1, round(cropped.shape[1] * scale)), max(1, round(cropped.shape[0] * scale))
        cropped = cv2.resize(cropped, output_size, interpolation=cv2.INTER_AREA)
    return Image.fromarray(cropped, "RGBA"), int(x) + left, int(y) + top


def _layer_content(layer: Layer) -> Image.Image | None:
    rendered = render_layer_pixels(layer)
    effects, styled = render_layer_style(layer, rendered)
    parts: list[tuple[Image.Image, int, int]] = []
    for pixels, x, y, opacity, _blend_mode in effects:
        part = _visible_part(pixels, x, y, float(opacity))
        if part is not None:
            parts.append(part)
    base = _visible_part(
        _apply_layer_alpha(layer, _display_rgba(styled)),
        layer.x,
        layer.y,
        1.0,
        max_dimension=256 if not effects else None,
    )
    if base is not None:
        parts.append(base)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0][0]

    left = min(x for _image, x, _y in parts)
    top = min(y for _image, _x, y in parts)
    right = max(x + image.width for image, x, _y in parts)
    bottom = max(y + image.height for image, _x, y in parts)
    content = Image.new("RGBA", (max(1, right - left), max(1, bottom - top)), (0, 0, 0, 0))
    for image, x, y in parts:
        content.alpha_composite(image, (x - left, y - top))
    return content


def _checkerboard(size: int) -> Image.Image:
    canvas = Image.new("RGBA", (size, size), (224, 226, 230, 255))
    draw = ImageDraw.Draw(canvas)
    tile = max(2, size // 8)
    for y in range(0, size, tile):
        for x in range(0, size, tile):
            if (x // tile + y // tile) % 2:
                draw.rectangle((x, y, min(size - 1, x + tile - 1), min(size - 1, y + tile - 1)), fill=(190, 194, 201, 255))
    return canvas


def build_layer_thumbnail(source: Layer | np.ndarray, size: int = 64) -> Image.Image:
    """Render a compact preview focused on the visible contents of one layer."""
    size = max(8, int(size))
    content = _layer_content(source) if isinstance(source, Layer) else Image.fromarray(_display_rgba(source), "RGBA")
    canvas = _checkerboard(size)
    if content is None or content.getbbox() is None:
        return canvas
    content.thumbnail((max(1, size - 4), max(1, size - 4)), Image.Resampling.LANCZOS)
    canvas.alpha_composite(content, ((size - content.width) // 2, (size - content.height) // 2))
    return canvas


__all__ = ["build_layer_thumbnail"]
