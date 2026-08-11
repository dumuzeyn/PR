from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image

from .core import mesh_warp_pixels, perspective_warp_pixels, rgba_array_to_pil


def preview_scale_for_bounds(
    canvas_size: tuple[int, int],
    bounds: tuple[float, float, float, float],
    padding: int = 48,
) -> float:
    width, height = canvas_size
    x1, y1, x2, y2 = bounds
    return min(
        max(1, width - padding) / max(1.0, x2 - x1),
        max(1, height - padding) / max(1.0, y2 - y1),
    )


def transform_preview_pixels(
    source: np.ndarray,
    source_cache: dict[float, np.ndarray],
    mode: str,
    pixel_scale: float,
    free_values: tuple[float, float, float, float, float],
    points: list[list[float]],
    rows: int,
    columns: int,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    row_positions: list[float] | None = None,
    column_positions: list[float] | None = None,
) -> tuple[np.ndarray, float, float, float]:
    """Render a screen-resolution transform preview in document coordinates."""
    pixel_scale = round(float(np.clip(pixel_scale, 0.025, 1.0)), 4)
    proxy = source_cache.get(pixel_scale)
    if proxy is None:
        if pixel_scale >= 0.9999:
            proxy = source
        else:
            proxy = cv2.resize(
                source,
                (max(1, round(source.shape[1] * pixel_scale)), max(1, round(source.shape[0] * pixel_scale))),
                interpolation=cv2.INTER_AREA,
            )
        source_cache.clear()
        source_cache[pixel_scale] = np.ascontiguousarray(proxy)

    x, y, width, height, angle = free_values
    if mode == "Свободная":
        image = rgba_array_to_pil(proxy).resize(
            (max(1, round(width * pixel_scale)), max(1, round(height * pixel_scale))),
            Image.Resampling.BILINEAR,
        )
        if flip_horizontal:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if flip_vertical:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if abs(angle) > 0.001:
            image = image.rotate(-angle, expand=True, resample=Image.Resampling.BILINEAR)
        logical_width = image.width / pixel_scale
        logical_height = image.height / pixel_scale
        return (
            np.asarray(image, dtype=np.uint8),
            x + (width - logical_width) / 2.0,
            y + (height - logical_height) / 2.0,
            pixel_scale,
        )

    scaled_points = [[point[0] * pixel_scale, point[1] * pixel_scale] for point in points]
    if mode == "Перспектива":
        output, offset = perspective_warp_pixels(proxy, scaled_points, cv2.INTER_LINEAR)
    else:
        output, offset = mesh_warp_pixels(
            proxy, scaled_points, rows, columns, cv2.INTER_LINEAR, row_positions, column_positions
        )
    return output, offset[0] / pixel_scale, offset[1] / pixel_scale, pixel_scale


def visible_document_rect(
    canvas,
    canvas_origin: tuple[float, float],
    zoom: float,
    document_size: tuple[int, int],
    padding: int = 2,
) -> tuple[int, int, int, int]:
    """Return the visible document rectangle, clipped to document bounds."""
    scale = max(0.01, float(zoom))
    origin_x, origin_y = canvas_origin
    left = canvas.canvasx(0)
    top = canvas.canvasy(0)
    right = canvas.canvasx(max(1, canvas.winfo_width()))
    bottom = canvas.canvasy(max(1, canvas.winfo_height()))
    width, height = document_size
    return (
        max(0, math.floor((left - origin_x) / scale) - padding),
        max(0, math.floor((top - origin_y) / scale) - padding),
        min(width, math.ceil((right - origin_x) / scale) + padding),
        min(height, math.ceil((bottom - origin_y) / scale) + padding),
    )
