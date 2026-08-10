from __future__ import annotations

from .core_shared import *
from .layer import Layer


def adjust_brightness_contrast(arr: np.ndarray, brightness: int, contrast: float) -> np.ndarray:
    out = normalize_rgba(arr)
    out[:, :, :3] = np.clip((out[:, :, :3] - 0.5) * float(contrast) + 0.5 + float(brightness) / 255.0, 0, 1)
    return _restore_pixel_dtype(out, arr.dtype)

def adjust_saturation(arr: np.ndarray, saturation: float) -> np.ndarray:
    out = normalize_rgba(arr)
    hsv = cv2.cvtColor(out[:, :, :3], cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(saturation), 0, 1)
    out[:, :, :3] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return _restore_pixel_dtype(out, arr.dtype)

def adjust_vibrance(arr: np.ndarray, vibrance: float = 0.0, saturation: float = 1.0) -> np.ndarray:
    out = normalize_rgba(arr)
    hsv = cv2.cvtColor(out[:, :, :3], cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    amount = float(np.clip(vibrance, -1.0, 1.0))
    if amount >= 0.0:
        sat = sat + (1.0 - sat) * (1.0 - sat) * amount
    else:
        sat = sat * (1.0 + amount)
    hsv[:, :, 1] = np.clip(sat * max(0.0, float(saturation)), 0.0, 1.0)
    out[:, :, :3] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return _restore_pixel_dtype(out, arr.dtype)

def adjust_temperature_tint(arr: np.ndarray, temperature: float = 0.0, tint: float = 0.0) -> np.ndarray:
    out = normalize_rgba(arr)
    rgb = out[:, :, :3]
    temperature = float(np.clip(temperature, -100.0, 100.0))
    tint = float(np.clip(tint, -100.0, 100.0))
    rgb[:, :, 0] += (temperature * 0.72 + tint * 0.18) / 255.0
    rgb[:, :, 1] += (tint * 0.52 - abs(temperature) * 0.06) / 255.0
    rgb[:, :, 2] -= (temperature * 0.72 + tint * 0.18) / 255.0
    out[:, :, :3] = np.clip(rgb, 0, 1)
    return _restore_pixel_dtype(out, arr.dtype)

def adjust_hue_saturation(arr: np.ndarray, hue: int, saturation: float, lightness: int) -> np.ndarray:
    out = normalize_rgba(arr)
    hsv = cv2.cvtColor(out[:, :, :3], cv2.COLOR_RGB2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + float(hue)) % 360.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(saturation), 0, 1)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + float(lightness) / 255.0, 0, 1)
    out[:, :, :3] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return _restore_pixel_dtype(out, arr.dtype)

def adjust_exposure(arr: np.ndarray, exposure: float, offset: float, gamma: float) -> np.ndarray:
    out = normalize_rgba(arr)
    rgb = out[:, :, :3]
    rgb = np.clip(rgb * (2.0 ** float(exposure)) + float(offset), 0, 1)
    rgb = np.power(rgb, 1.0 / max(0.01, float(gamma)))
    out[:, :, :3] = rgb
    return _restore_pixel_dtype(out, arr.dtype)

def adjust_color_balance(arr: np.ndarray, red: int, green: int, blue: int) -> np.ndarray:
    out = normalize_rgba(arr)
    shifts = np.array([red, green, blue], dtype=np.float32) / 255.0
    out[:, :, :3] = np.clip(out[:, :, :3] + shifts, 0, 1)
    return _restore_pixel_dtype(out, arr.dtype)

def adjust_threshold(arr: np.ndarray, threshold: int) -> np.ndarray:
    out = normalize_rgba(arr)
    gray = cv2.cvtColor(out[:, :, :3], cv2.COLOR_RGB2GRAY)
    bw = np.where(gray >= float(np.clip(threshold, 0, 255)) / 255.0, 1.0, 0.0).astype(np.float32)
    out[:, :, :3] = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
    return _restore_pixel_dtype(out, arr.dtype)

def adjust_posterize(arr: np.ndarray, levels_count: int) -> np.ndarray:
    levels_count = int(np.clip(levels_count, 2, 64))
    out = normalize_rgba(arr)
    out[:, :, :3] = np.round(out[:, :, :3] * (levels_count - 1)) / (levels_count - 1)
    return _restore_pixel_dtype(out, arr.dtype)

def levels(arr: np.ndarray, black: int, white: int, gamma: float) -> np.ndarray:
    out = normalize_rgba(arr)
    black_value = float(black) / 255.0
    white_value = float(white) / 255.0
    rgb = np.clip((out[:, :, :3] - black_value) / max(1e-7, white_value - black_value), 0, 1)
    rgb = np.power(rgb, 1.0 / max(0.01, gamma))
    out[:, :, :3] = rgb
    return _restore_pixel_dtype(out, arr.dtype)

def curves(arr: np.ndarray, shadows: int, midtones: int, highlights: int) -> np.ndarray:
    xs = np.array([0, 64, 128, 192, 255], dtype=np.float32) / 255.0
    ys = np.array([0, shadows, midtones, highlights, 255], dtype=np.float32) / 255.0
    out = normalize_rgba(arr)
    out[:, :, :3] = np.interp(out[:, :, :3], xs, ys).clip(0, 1)
    return _restore_pixel_dtype(out, arr.dtype)

def _restore_pixel_dtype(normalized: np.ndarray, dtype: np.dtype) -> np.ndarray:
    value = np.clip(normalized, 0.0, 1.0)
    if dtype == np.uint8:
        return np.rint(value * 255.0).astype(np.uint8)
    if dtype == np.uint16:
        return np.rint(value * 65535.0).astype(np.uint16)
    if np.issubdtype(dtype, np.floating):
        return value.astype(dtype)
    raise TypeError(f"Unsupported pixel dtype: {dtype}")

def blur(arr: np.ndarray, radius: int) -> np.ndarray:
    k = max(1, radius * 2 + 1)
    out = normalize_rgba(arr)
    out[:, :, :3] = cv2.GaussianBlur(out[:, :, :3], (k, k), radius)
    return _restore_pixel_dtype(out, arr.dtype)

def sharpen(arr: np.ndarray, amount: float) -> np.ndarray:
    out = normalize_rgba(arr)
    blurred = cv2.GaussianBlur(out[:, :, :3], (0, 0), 1.2)
    out[:, :, :3] = np.clip(out[:, :, :3] * (1 + amount) - blurred * amount, 0, 1)
    return _restore_pixel_dtype(out, arr.dtype)

def add_noise(arr: np.ndarray, amount: float) -> np.ndarray:
    out = normalize_rgba(arr)
    noise = np.random.normal(0, amount, out[:, :, :3].shape)
    out[:, :, :3] = np.clip(out[:, :, :3] + noise, 0, 1)
    return _restore_pixel_dtype(out, arr.dtype)

def apply_filter_stack(arr: np.ndarray, filters: list[dict[str, Any]]) -> np.ndarray:
    original_dtype = arr.dtype
    out = normalize_rgba(arr)
    for item in filters:
        if not bool(item.get("enabled", True)):
            continue
        before = out.copy()
        kind = str(item.get("type", "")).lower()
        channel = str(item.get("channel", "RGB"))
        source = before
        if channel == "Alpha":
            alpha = before[:, :, 3]
            source = np.dstack((alpha, alpha, alpha, alpha)).astype(np.float32)
        if kind == "blur":
            filtered = normalize_rgba(blur(source, int(item.get("radius", 3))))
        elif kind == "sharpen":
            filtered = normalize_rgba(sharpen(source, float(item.get("amount", 1.0))))
        elif kind == "noise":
            filtered = normalize_rgba(deterministic_noise(source, float(item.get("amount", 0.03)), int(item.get("seed", 12345))))
        elif kind == "median":
            filtered = normalize_rgba(median_filter(source, int(item.get("size", 3))))
        elif kind == "edge":
            filtered = normalize_rgba(edge_filter(source, float(item.get("strength", 1.0))))
        elif kind == "emboss":
            filtered = normalize_rgba(emboss_filter(source, float(item.get("strength", 1.0))))
        else:
            continue
        if channel == "Alpha":
            effect = before.copy()
            effect[:, :, 3] = filtered[:, :, 0]
            channel_indices: tuple[int, ...] = ()
        else:
            effect = filtered
            channel_indices = {"Red": (0,), "Green": (1,), "Blue": (2,), "RGB": (0, 1, 2)}.get(channel, (0, 1, 2))
            for index in {0, 1, 2} - set(channel_indices):
                effect[:, :, index] = before[:, :, index]
            effect[:, :, 3] = before[:, :, 3]
        opacity = float(np.clip(item.get("opacity", 1.0), 0.0, 1.0))
        blend_mode = str(item.get("blend_mode", "Normal"))
        blend_source = effect
        if blend_mode != "Normal":
            blend_source = effect.copy()
            blend_source[:, :, :3] = blend_rgb(effect[:, :, :3], before[:, :, :3], blend_mode, 1.0).clip(0, 1)
            for index in {0, 1, 2} - set(channel_indices):
                blend_source[:, :, index] = before[:, :, index]
        mask = filter_mask_from_item(item, out.shape[:2])
        if opacity <= 0.001:
            out = before
        else:
            mix = np.full(out.shape[:2], opacity, dtype=np.float32) if mask is None else mask * opacity
            out = before * (1.0 - mix[:, :, None]) + blend_source * mix[:, :, None]
            if channel == "Alpha":
                out[:, :, :3] = before[:, :, :3]
            elif channel != "RGB":
                for index in {0, 1, 2} - set(channel_indices):
                    out[:, :, index] = before[:, :, index]
    return _restore_pixel_dtype(out, original_dtype)

def filter_mask_from_item(item: dict[str, Any], shape: tuple[int, int]) -> np.ndarray | None:
    encoded = item.get("mask")
    if not isinstance(encoded, str) or not encoded:
        return None
    cached = _filter_mask_cache.get(encoded)
    if cached is None:
        try:
            cached = decode_png(encoded)[:, :, 0].astype(np.uint8)
        except Exception:
            return None
        if len(_filter_mask_cache) > 32:
            _filter_mask_cache.clear()
        _filter_mask_cache[encoded] = cached
    mask = cached
    target_h, target_w = shape
    if mask.shape != (target_h, target_w):
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    result = (mask.astype(np.float32) / 255.0).clip(0, 1)
    feather = max(0.0, float(item.get("mask_feather", 0.0)))
    if feather > 0.01:
        result = cv2.GaussianBlur(result, (0, 0), feather).clip(0, 1)
    if bool(item.get("mask_inverted", False)):
        result = 1.0 - result
    density = float(np.clip(item.get("mask_density", 1.0), 0.0, 1.0))
    return ((1.0 - density) + result * density).clip(0, 1)

def median_filter(arr: np.ndarray, size: int) -> np.ndarray:
    maximum = max(3, min(arr.shape[0], arr.shape[1]))
    k = min(maximum if maximum % 2 else maximum - 1, max(3, int(size) | 1))
    out = normalize_rgba(arr)
    passes = max(1, int(math.ceil((k - 1) / 4.0)))
    for _ in range(passes):
        out[:, :, :3] = cv2.medianBlur(out[:, :, :3], 3 if k == 3 else 5)
    return _restore_pixel_dtype(out, arr.dtype)

def deterministic_noise(arr: np.ndarray, amount: float, seed: int = 12345) -> np.ndarray:
    out = normalize_rgba(arr)
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(0, float(amount), out[:, :, :3].shape)
    out[:, :, :3] = np.clip(out[:, :, :3] + noise, 0, 1)
    return _restore_pixel_dtype(out, arr.dtype)

def edge_filter(arr: np.ndarray, strength: float = 1.0) -> np.ndarray:
    out = normalize_rgba(arr)
    gray = cv2.cvtColor(out[:, :, :3], cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(np.rint(gray * 255.0).astype(np.uint8), 80, 160).astype(np.float32) / 255.0
    edge_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    mix = np.clip(float(strength), 0, 1)
    out[:, :, :3] = out[:, :, :3] * (1.0 - mix) + edge_rgb * mix
    return _restore_pixel_dtype(out, arr.dtype)

def emboss_filter(arr: np.ndarray, strength: float = 1.0) -> np.ndarray:
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
    out = normalize_rgba(arr)
    embossed = cv2.filter2D(out[:, :, :3], -1, kernel) + 0.5
    mix = np.clip(float(strength), 0, 1)
    out[:, :, :3] = np.clip(out[:, :, :3] * (1.0 - mix) + embossed * mix, 0, 1)
    return _restore_pixel_dtype(out, arr.dtype)

__all__ = [name for name in globals() if not name.startswith("__")]
