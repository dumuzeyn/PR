from __future__ import annotations

from typing import Any

import cv2
import numpy as np


NOISE_COLOR_MODELS = ("rgb", "hsv", "grayscale")


def _channel_ranges(options: dict[str, Any], model: str) -> list[tuple[float, float]]:
    count = 1 if model == "grayscale" else 3
    source = options.get("channels", [])
    ranges: list[tuple[float, float]] = []
    for index in range(count):
        values = source[index] if isinstance(source, list) and index < len(source) else (0.0, 1.0)
        low, high = (float(values[0]), float(values[1])) if len(values) >= 2 else (0.0, 1.0)
        ranges.append((float(np.clip(min(low, high), 0.0, 1.0)), float(np.clip(max(low, high), 0.0, 1.0))))
    return ranges


def _hsv_to_rgb(values: np.ndarray) -> np.ndarray:
    hsv = values.reshape(1, -1, 3).astype(np.float32)
    hsv[:, :, 0] *= 360.0
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).reshape(-1, 3)


def noise_lookup(base: np.ndarray, options: dict[str, Any]) -> np.ndarray:
    """Create a stable one-dimensional noise gradient lookup table."""
    roughness = float(np.clip(options.get("roughness", 0.5), 0.0, 1.0))
    model = str(options.get("color_model", "rgb")).lower()
    if model not in NOISE_COLOR_MODELS:
        model = "rgb"
    rng = np.random.default_rng(int(options.get("seed", 0)))
    anchor_count = max(4, round(5 + roughness * 251))
    anchor_x = np.linspace(0.0, 1.0, anchor_count, dtype=np.float32)
    ranges = _channel_ranges(options, model)
    random_channels = [rng.uniform(low, high, anchor_count) for low, high in ranges]
    if model == "grayscale":
        anchor_rgb = np.repeat(random_channels[0][:, None], 3, axis=1)
    elif model == "hsv":
        anchor_rgb = _hsv_to_rgb(np.column_stack(random_channels))
    else:
        anchor_rgb = np.column_stack(random_channels)

    axis = np.linspace(0.0, 1.0, len(base), dtype=np.float32)
    generated = np.column_stack(
        [np.interp(axis, anchor_x, anchor_rgb[:, channel]) for channel in range(3)]
    ).astype(np.float32)
    if bool(options.get("restrict_colors", False)):
        low = np.min(base[:, :3], axis=0)
        high = np.max(base[:, :3], axis=0)
        generated = low + generated * (high - low)
    result = base.copy()
    result[:, :3] = np.clip(generated, 0.0, 1.0)
    return result
