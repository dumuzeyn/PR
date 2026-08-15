from __future__ import annotations

from .core_shared import *
from .layer import Layer
from .filter_ops import *
from .selection_ops import *


def apply_adjustment(arr: np.ndarray, adjustment: dict[str, Any]) -> np.ndarray:
    kind = str(adjustment.get("type", "")).lower()
    if kind == "brightness_contrast":
        adjusted = adjust_brightness_contrast(arr, int(adjustment.get("brightness", 0)), float(adjustment.get("contrast", 1.0)))
    elif kind == "saturation":
        adjusted = adjust_saturation(arr, float(adjustment.get("saturation", 1.0)))
    elif kind == "vibrance":
        adjusted = adjust_vibrance(arr, float(adjustment.get("vibrance", 0.0)), float(adjustment.get("saturation", 1.0)))
    elif kind == "temperature_tint":
        adjusted = adjust_temperature_tint(arr, float(adjustment.get("temperature", 0.0)), float(adjustment.get("tint", 0.0)))
    elif kind == "hue_saturation":
        adjusted = adjust_hue_saturation(arr, int(adjustment.get("hue", 0)), float(adjustment.get("saturation", 1.0)), int(adjustment.get("lightness", 0)))
    elif kind == "exposure":
        adjusted = adjust_exposure(arr, float(adjustment.get("exposure", 0.0)), float(adjustment.get("offset", 0.0)), float(adjustment.get("gamma", 1.0)))
    elif kind == "color_balance":
        adjusted = adjust_color_balance(arr, int(adjustment.get("red", 0)), int(adjustment.get("green", 0)), int(adjustment.get("blue", 0)))
    elif kind == "black_white":
        adjusted = adjust_black_white(arr, float(adjustment.get("red", 0.299)), float(adjustment.get("green", 0.587)), float(adjustment.get("blue", 0.114)))
    elif kind == "threshold":
        adjusted = adjust_threshold(arr, int(adjustment.get("threshold", 128)))
    elif kind == "posterize":
        adjusted = adjust_posterize(arr, int(adjustment.get("levels", 6)))
    elif kind == "levels":
        adjusted = levels(arr, int(adjustment.get("black", 0)), int(adjustment.get("white", 255)), float(adjustment.get("gamma", 1.0)))
    elif kind == "curves":
        adjusted = curves(arr, int(adjustment.get("shadows", 64)), int(adjustment.get("midtones", 128)), int(adjustment.get("highlights", 192)))
    elif kind == "invert":
        normalized = normalize_rgba(arr)
        normalized[:, :, :3] = 1.0 - normalized[:, :, :3]
        adjusted = _restore_pixel_dtype(normalized, arr.dtype)
    elif kind == "grayscale":
        normalized = normalize_rgba(arr)
        gray = cv2.cvtColor(normalized[:, :, :3], cv2.COLOR_RGB2GRAY)
        normalized[:, :, :3] = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        adjusted = _restore_pixel_dtype(normalized, arr.dtype)
    else:
        return arr.copy()
    channel = str(adjustment.get("channel", "RGB"))
    if channel == "RGB":
        return adjusted
    before_normalized = normalize_rgba(arr)
    adjusted_normalized = normalize_rgba(adjusted)
    if channel == "Alpha":
        alpha_source = np.dstack([before_normalized[:, :, 3]] * 4).astype(np.float32)
        alpha_adjustment = dict(adjustment)
        alpha_adjustment["channel"] = "RGB"
        alpha_result = normalize_rgba(apply_adjustment(alpha_source, alpha_adjustment))
        before_normalized[:, :, 3] = alpha_result[:, :, 0]
    else:
        index = {"Red": 0, "Green": 1, "Blue": 2}.get(channel)
        if index is None:
            return adjusted
        before_normalized[:, :, index] = adjusted_normalized[:, :, index]
    return _restore_pixel_dtype(before_normalized, arr.dtype)

def apply_adjustment_layer(out: np.ndarray, layer: Layer, clipping_mask: np.ndarray | None = None) -> None:
    if layer.adjustment is None:
        return
    adjusted = apply_adjustment(out, layer.adjustment)
    alpha = np.full(out.shape[:2], float(layer.opacity), dtype=np.float32)
    if layer.mask is not None and layer.mask_enabled:
        mask_canvas = np.zeros(out.shape[:2], dtype=np.uint8)
        paste_mask(mask_canvas, effective_layer_mask(layer), layer.x, layer.y)
        mask_alpha = ((1.0 - float(layer.mask_density)) + (mask_canvas.astype(np.float32) / 255.0) * float(layer.mask_density))
        alpha *= mask_alpha
    if clipping_mask is not None:
        alpha *= (clipping_mask.astype(np.float32) / 255.0).clip(0, 1)
    alpha = alpha[:, :, None].clip(0, 1)
    adjustment_rgb = adjusted[:, :, :3].astype(np.float32)
    if layer.blend_mode != "Normal":
        adjustment_rgb = blend_rgb(adjustment_rgb, out[:, :, :3], layer.blend_mode)
    out[:, :, :3] = np.clip(adjustment_rgb * alpha + out[:, :, :3].astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)

__all__ = [name for name in globals() if not name.startswith("__")]
