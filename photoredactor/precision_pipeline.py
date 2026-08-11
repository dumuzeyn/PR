from __future__ import annotations

from typing import Any

import numpy as np

from .adjustment_ops import apply_adjustment
from .color_management import normalize_rgba
from .core_shared import blend_rgb, blank_rgba
from .filter_ops import apply_filter_stack
from .geometry_ops import checker_background
from .render_ops import render_layer_effects
from .selection_ops import effective_layer_mask


def render_layer_precision(layer) -> np.ndarray:
    source = layer.working_rgba()
    return normalize_rgba(apply_filter_stack(source, layer.filters)) if layer.filters else source.copy()


def alpha_blend_precision(
    destination: np.ndarray,
    source: np.ndarray,
    x: int,
    y: int,
    opacity: float = 1.0,
    alpha_mask: np.ndarray | None = None,
    mask_density: float = 1.0,
    blend_mode: str = "Normal",
) -> None:
    height, width = source.shape[:2]
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(destination.shape[1], int(x) + width), min(destination.shape[0], int(y) + height)
    if x1 >= x2 or y1 >= y2:
        return
    sx1, sy1 = x1 - int(x), y1 - int(y)
    sx2, sy2 = sx1 + x2 - x1, sy1 + y2 - y1
    source_region = normalize_rgba(source[sy1:sy2, sx1:sx2])
    target = destination[y1:y2, x1:x2]
    source_alpha = source_region[:, :, 3:4] * float(np.clip(opacity, 0.0, 1.0))
    if alpha_mask is not None:
        mask = alpha_mask[sy1:sy2, sx1:sx2].astype(np.float32) / 255.0
        source_alpha *= (1.0 - float(mask_density)) + mask[:, :, None] * float(mask_density)
    target_alpha = target[:, :, 3:4]
    output_alpha = source_alpha + target_alpha * (1.0 - source_alpha)
    blended = blend_rgb(source_region[:, :, :3], target[:, :, :3], blend_mode, 1.0).clip(0.0, 1.0)
    output_rgb = np.where(
        output_alpha > 1e-7,
        (blended * source_alpha + target[:, :, :3] * target_alpha * (1.0 - source_alpha)) / np.maximum(output_alpha, 1e-7),
        0.0,
    )
    target[:, :, :3] = np.clip(output_rgb, 0.0, 1.0)
    target[:, :, 3:4] = np.clip(output_alpha, 0.0, 1.0)


def _mask_canvas(document, layer, mask: np.ndarray) -> np.ndarray:
    canvas = np.zeros((document.height, document.width), dtype=np.uint8)
    x1, y1 = max(0, layer.x), max(0, layer.y)
    x2 = min(document.width, layer.x + mask.shape[1])
    y2 = min(document.height, layer.y + mask.shape[0])
    if x1 < x2 and y1 < y2:
        lx1, ly1 = x1 - layer.x, y1 - layer.y
        canvas[y1:y2, x1:x2] = mask[ly1 : ly1 + y2 - y1, lx1 : lx1 + x2 - x1]
    return canvas


def _layer_alpha_canvas(document, layer, pixels: np.ndarray, alpha_mask: np.ndarray | None) -> np.ndarray:
    alpha = np.clip(pixels[:, :, 3] * 255.0, 0, 255).astype(np.uint8)
    if alpha_mask is not None:
        coverage = ((1.0 - float(layer.mask_density)) + alpha_mask.astype(np.float32) / 255.0 * float(layer.mask_density))
        alpha = np.clip(alpha.astype(np.float32) * coverage, 0, 255).astype(np.uint8)
    return _mask_canvas(document, layer, alpha)


def _document_alpha_to_layer(alpha_canvas: np.ndarray, layer) -> np.ndarray:
    mask = np.zeros(layer.pixels.shape[:2], dtype=np.uint8)
    x1, y1 = max(0, layer.x), max(0, layer.y)
    x2 = min(alpha_canvas.shape[1], layer.x + layer.pixels.shape[1])
    y2 = min(alpha_canvas.shape[0], layer.y + layer.pixels.shape[0])
    if x1 < x2 and y1 < y2:
        lx1, ly1 = x1 - layer.x, y1 - layer.y
        mask[ly1 : ly1 + y2 - y1, lx1 : lx1 + x2 - x1] = alpha_canvas[y1:y2, x1:x2]
    return mask


def apply_adjustment_precision(
    output: np.ndarray,
    document,
    layer,
    clipping_mask: np.ndarray | None = None,
) -> None:
    if layer.adjustment is None:
        return
    adjusted = normalize_rgba(apply_adjustment(output, layer.adjustment))
    coverage = np.full(output.shape[:2], float(layer.opacity), dtype=np.float32)
    if layer.mask is not None and layer.mask_enabled:
        mask = _mask_canvas(document, layer, effective_layer_mask(layer)).astype(np.float32) / 255.0
        coverage *= (1.0 - float(layer.mask_density)) + mask * float(layer.mask_density)
    if clipping_mask is not None:
        coverage *= clipping_mask.astype(np.float32) / 255.0
    adjusted_rgb = adjusted[:, :, :3]
    if layer.blend_mode != "Normal":
        adjusted_rgb = blend_rgb(adjusted_rgb, output[:, :, :3], layer.blend_mode, 1.0)
    mix = np.clip(coverage, 0.0, 1.0)[:, :, None]
    output[:, :, :3] = np.clip(adjusted_rgb * mix + output[:, :, :3] * (1.0 - mix), 0.0, 1.0)
    if str(layer.adjustment.get("channel", "RGB")) == "Alpha":
        output[:, :, 3:4] = adjusted[:, :, 3:4] * mix + output[:, :, 3:4] * (1.0 - mix)


def composite_precision(document, checker: bool = False) -> np.ndarray:
    if checker:
        output = normalize_rgba(checker_background(document.width, document.height))
    else:
        output = normalize_rgba(blank_rgba(document.width, document.height, (0, 0, 0, 0)))
    previous_alpha: np.ndarray | None = None
    for layer in document.layers:
        if not layer.visible:
            continue
        if layer.kind == "adjustment" and layer.adjustment is not None:
            clipping = previous_alpha if layer.clipping else None
            apply_adjustment_precision(output, document, layer, clipping)
            continue
        pixels = render_layer_precision(layer)
        alpha_mask = effective_layer_mask(layer) if layer.mask is not None and layer.mask_enabled else None
        if layer.clipping and previous_alpha is not None:
            clipping = _document_alpha_to_layer(previous_alpha, layer)
            alpha_mask = clipping if alpha_mask is None else np.minimum(alpha_mask, clipping)
        display_pixels = np.rint(pixels * 255.0).astype(np.uint8)
        for effect, x, y, opacity, blend_mode in render_layer_effects(layer, display_pixels):
            alpha_blend_precision(output, normalize_rgba(effect), x, y, opacity, None, 1.0, blend_mode)
        alpha_blend_precision(output, pixels, layer.x, layer.y, layer.opacity, alpha_mask, layer.mask_density, layer.blend_mode)
        previous_alpha = _layer_alpha_canvas(document, layer, pixels, alpha_mask)
    return np.ascontiguousarray(output.astype(np.float32))


def precision_statistics(array: np.ndarray) -> dict[str, Any]:
    rgba = normalize_rgba(array)
    return {
        "minimum": [float(value) for value in rgba.min(axis=(0, 1))],
        "maximum": [float(value) for value in rgba.max(axis=(0, 1))],
        "mean": [float(value) for value in rgba.mean(axis=(0, 1))],
        "unique_luminance": int(np.unique(np.rint((rgba[:, :, :3] @ np.array([0.2126, 0.7152, 0.0722])) * 65535)).size),
    }


__all__ = [name for name in globals() if not name.startswith("__")]
