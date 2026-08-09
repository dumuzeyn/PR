from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageCms


COLOR_MODELS = ("RGBA", "Lab", "CMYK")
BIT_DEPTHS = (8, 16, 32)


def normalize_rgba(array: np.ndarray) -> np.ndarray:
    """Convert 8/16-bit integer or float pixels to float32 RGBA in 0..1."""
    value = np.asarray(array)
    if value.dtype == np.uint8:
        out = value.astype(np.float32) / 255.0
    elif value.dtype == np.uint16:
        out = value.astype(np.float32) / 65535.0
    elif np.issubdtype(value.dtype, np.floating):
        out = value.astype(np.float32)
    else:
        raise TypeError(f"Unsupported pixel dtype: {value.dtype}")
    if out.ndim != 3 or out.shape[2] not in {3, 4}:
        raise ValueError("Expected an RGB or RGBA image")
    if out.shape[2] == 3:
        out = np.dstack((out, np.ones(out.shape[:2], dtype=np.float32)))
    return np.clip(out, 0.0, 1.0)


def quantize_rgba(array: np.ndarray, bit_depth: int) -> np.ndarray:
    normalized = normalize_rgba(array)
    if bit_depth == 8:
        return np.rint(normalized * 255.0).astype(np.uint8)
    if bit_depth == 16:
        return np.rint(normalized * 65535.0).astype(np.uint16)
    if bit_depth == 32:
        return normalized.astype(np.float32)
    raise ValueError("Bit depth must be 8, 16 or 32")


def display_rgba(array: np.ndarray) -> np.ndarray:
    return quantize_rgba(array, 8)


def rgb_to_lab(array: np.ndarray) -> np.ndarray:
    rgba = normalize_rgba(array)
    lab = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2Lab).astype(np.float32)
    return np.dstack((lab, rgba[:, :, 3]))


def lab_to_rgb(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    if value.ndim != 3 or value.shape[2] not in {3, 4}:
        raise ValueError("Expected a Lab or LabA image")
    rgb = cv2.cvtColor(value[:, :, :3], cv2.COLOR_Lab2RGB).clip(0.0, 1.0)
    alpha = value[:, :, 3] if value.shape[2] == 4 else np.ones(value.shape[:2], dtype=np.float32)
    return np.dstack((rgb, np.clip(alpha, 0.0, 1.0))).astype(np.float32)


def rgb_to_cmyk(array: np.ndarray) -> np.ndarray:
    rgb = normalize_rgba(array)[:, :, :3]
    key = 1.0 - np.max(rgb, axis=2)
    denominator = np.maximum(1.0 - key, 1e-7)
    cmy = (1.0 - rgb - key[:, :, None]) / denominator[:, :, None]
    cmy[key >= 1.0 - 1e-7] = 0.0
    return np.dstack((np.clip(cmy, 0.0, 1.0), np.clip(key, 0.0, 1.0))).astype(np.float32)


def cmyk_to_rgb(array: np.ndarray, alpha: np.ndarray | None = None) -> np.ndarray:
    cmyk = np.asarray(array, dtype=np.float32)
    if cmyk.ndim != 3 or cmyk.shape[2] != 4:
        raise ValueError("Expected a CMYK image")
    cmy, key = cmyk[:, :, :3], cmyk[:, :, 3:4]
    rgb = (1.0 - cmy) * (1.0 - key)
    alpha_channel = np.ones(cmyk.shape[:2], dtype=np.float32) if alpha is None else np.asarray(alpha, dtype=np.float32)
    return np.dstack((np.clip(rgb, 0.0, 1.0), np.clip(alpha_channel, 0.0, 1.0))).astype(np.float32)


def profile_bytes(profile: str | Path | bytes | None) -> bytes | None:
    if profile is None:
        return None
    if isinstance(profile, bytes):
        return profile
    path = Path(profile)
    return path.read_bytes() if path.exists() else None


def profile_name(profile: str | Path | bytes | None) -> str:
    if profile is None or str(profile).lower() in {"srgb", "s-rgb"}:
        return "sRGB"
    if isinstance(profile, (str, Path)) and Path(profile).exists():
        try:
            return ImageCms.getProfileName(ImageCms.ImageCmsProfile(str(profile))).strip()
        except Exception:
            return Path(profile).stem
    if isinstance(profile, bytes):
        try:
            return ImageCms.getProfileName(ImageCms.ImageCmsProfile(BytesIO(profile))).strip()
        except Exception:
            return "Встроенный ICC"
    return str(profile)


def _open_profile(profile: str | Path | bytes | None):
    if profile is None or str(profile).lower() in {"srgb", "s-rgb"}:
        return ImageCms.createProfile("sRGB")
    if isinstance(profile, bytes):
        return ImageCms.ImageCmsProfile(BytesIO(profile))
    return ImageCms.ImageCmsProfile(str(profile))


def convert_icc(array: np.ndarray, source: str | Path | bytes | None, destination: str | Path | bytes | None) -> np.ndarray:
    rgba = display_rgba(array)
    alpha = rgba[:, :, 3].copy()
    image = Image.fromarray(rgba[:, :, :3], "RGB")
    converted = ImageCms.profileToProfile(image, _open_profile(source), _open_profile(destination), outputMode="RGB")
    return np.dstack((np.asarray(converted, dtype=np.uint8), alpha))


def color_settings(metadata: dict[str, Any]) -> dict[str, Any]:
    settings = metadata.setdefault("color_management", {})
    settings.setdefault("profile_name", "sRGB")
    settings.setdefault("rendering_intent", "perceptual")
    return settings
