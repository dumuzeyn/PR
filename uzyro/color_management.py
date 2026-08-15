from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageCms


COLOR_MODELS = ("RGBA", "Lab", "CMYK")
BIT_DEPTHS = (8, 16, 32)
RENDERING_INTENTS = {
    "perceptual": ImageCms.Intent.PERCEPTUAL,
    "relative": ImageCms.Intent.RELATIVE_COLORIMETRIC,
    "saturation": ImageCms.Intent.SATURATION,
    "absolute": ImageCms.Intent.ABSOLUTE_COLORIMETRIC,
}


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


def working_to_rgba(array: np.ndarray, color_model: str = "RGBA") -> np.ndarray:
    value = np.asarray(array)
    if color_model == "Lab":
        return lab_to_rgb(value)
    if color_model == "CMYK":
        if value.ndim != 3 or value.shape[2] not in {4, 5}:
            raise ValueError("Expected CMYK or CMYKA working pixels")
        alpha = value[:, :, 4] if value.shape[2] == 5 else None
        return cmyk_to_rgb(value[:, :, :4], alpha)
    return normalize_rgba(value)


def rgba_to_working(array: np.ndarray, color_model: str = "RGBA", bit_depth: int = 8) -> np.ndarray:
    rgba = normalize_rgba(array)
    if color_model == "Lab":
        return rgb_to_lab(rgba)
    if color_model == "CMYK":
        return np.dstack((rgb_to_cmyk(rgba), rgba[:, :, 3])).astype(np.float32)
    return quantize_rgba(rgba, bit_depth)


def profile_bytes(profile: str | Path | bytes | None) -> bytes | None:
    if profile is None:
        return None
    if isinstance(profile, bytes):
        return profile
    path = Path(profile)
    return path.read_bytes() if path.exists() else None


def builtin_srgb_bytes() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


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


def profile_details(profile: str | Path | bytes | None) -> dict[str, str]:
    raw_profile = _open_profile(profile)
    opened = raw_profile if isinstance(raw_profile, ImageCms.ImageCmsProfile) else ImageCms.ImageCmsProfile(raw_profile)
    core = opened.profile
    return {
        "name": str(getattr(core, "profile_description", "") or profile_name(profile)).strip(),
        "color_space": str(getattr(core, "xcolor_space", "") or "").strip(),
        "device_class": str(getattr(core, "device_class", "") or "").strip(),
        "connection_space": str(getattr(core, "connection_space", "") or "").strip(),
        "copyright": str(getattr(core, "copyright", "") or "").strip(),
    }


def _intent(value: str | int) -> ImageCms.Intent:
    if isinstance(value, int):
        return ImageCms.Intent(int(value))
    return RENDERING_INTENTS.get(str(value).lower(), ImageCms.Intent.PERCEPTUAL)


def _apply_rgb_transform(normalized: np.ndarray, transform) -> np.ndarray:
    base = np.rint(np.clip(normalized[:, :, :3], 0.0, 1.0) * 255.0).astype(np.uint8)
    image = Image.fromarray(base, "RGB")
    converted = np.asarray(ImageCms.applyTransform(image, transform), dtype=np.uint8).astype(np.float32) / 255.0
    return converted


def _restore_precision(normalized: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if dtype == np.uint8:
        return np.rint(np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)
    if dtype == np.uint16:
        return np.rint(np.clip(normalized, 0.0, 1.0) * 65535.0).astype(np.uint16)
    if np.issubdtype(dtype, np.floating):
        return np.clip(normalized, 0.0, 1.0).astype(dtype)
    raise TypeError(f"Unsupported pixel dtype: {dtype}")


def _transform_with_residual(array: np.ndarray, transform) -> np.ndarray:
    normalized = normalize_rgba(array)
    rgb = _apply_rgb_transform(normalized, transform)
    quantized = np.rint(normalized[:, :, :3] * 255.0) / 255.0
    residual = normalized[:, :, :3] - quantized
    if array.dtype != np.uint8 and np.any(residual):
        step = 1.0 / 255.0
        for channel in range(3):
            perturbed = normalized.copy()
            direction = np.where(quantized[:, :, channel] < 1.0, step, -step).astype(np.float32)
            perturbed[:, :, channel] = np.clip(quantized[:, :, channel] + direction, 0.0, 1.0)
            transformed_step = _apply_rgb_transform(perturbed, transform)
            derivative = (transformed_step - rgb) / direction[:, :, None]
            rgb += derivative * residual[:, :, channel : channel + 1]
    result = np.dstack((np.clip(rgb, 0.0, 1.0), normalized[:, :, 3]))
    return _restore_precision(result, array.dtype)


def convert_icc(
    array: np.ndarray,
    source: str | Path | bytes | None,
    destination: str | Path | bytes | None,
    rendering_intent: str | int = "perceptual",
    black_point_compensation: bool = True,
) -> np.ndarray:
    source_profile = _open_profile(source)
    destination_profile = _open_profile(destination)
    if ImageCms.getProfileName(source_profile).strip() == ImageCms.getProfileName(destination_profile).strip():
        return array.copy()
    flags = ImageCms.Flags.BLACKPOINTCOMPENSATION if black_point_compensation else ImageCms.Flags.NONE
    transform = ImageCms.buildTransformFromOpenProfiles(
        source_profile,
        destination_profile,
        "RGB",
        "RGB",
        renderingIntent=_intent(rendering_intent),
        flags=flags,
    )
    return _transform_with_residual(array, transform)


def soft_proof_rgba(
    array: np.ndarray,
    source: str | Path | bytes | None,
    proof: str | Path | bytes,
    display: str | Path | bytes | None = "sRGB",
    rendering_intent: str | int = "relative",
    black_point_compensation: bool = True,
) -> np.ndarray:
    flags = ImageCms.Flags.SOFTPROOFING
    if black_point_compensation:
        flags |= ImageCms.Flags.BLACKPOINTCOMPENSATION
    transform = ImageCms.buildProofTransformFromOpenProfiles(
        _open_profile(source),
        _open_profile(display),
        _open_profile(proof),
        "RGB",
        "RGB",
        renderingIntent=_intent(rendering_intent),
        proofRenderingIntent=_intent(rendering_intent),
        flags=flags,
    )
    return _transform_with_residual(array, transform)


def rgb_to_profile_cmyk(
    array: np.ndarray,
    source: str | Path | bytes | None,
    destination: str | Path | bytes,
    rendering_intent: str | int = "relative",
    black_point_compensation: bool = True,
) -> np.ndarray:
    details = profile_details(destination)
    if details["color_space"].upper() != "CMYK":
        raise ValueError("Для цветоделения требуется CMYK ICC-профиль печати")
    flags = ImageCms.Flags.BLACKPOINTCOMPENSATION if black_point_compensation else ImageCms.Flags.NONE
    transform = ImageCms.buildTransformFromOpenProfiles(
        _open_profile(source),
        _open_profile(destination),
        "RGB",
        "CMYK",
        renderingIntent=_intent(rendering_intent),
        flags=flags,
    )
    rgba = display_rgba(array)
    image = Image.fromarray(rgba[:, :, :3], "RGB")
    return np.asarray(ImageCms.applyTransform(image, transform), dtype=np.uint8)


def gamut_warning_mask(original: np.ndarray, proofed: np.ndarray, threshold: float = 8.0) -> np.ndarray:
    original_lab = cv2.cvtColor(normalize_rgba(original)[:, :, :3], cv2.COLOR_RGB2Lab)
    proofed_lab = cv2.cvtColor(normalize_rgba(proofed)[:, :, :3], cv2.COLOR_RGB2Lab)
    delta = np.linalg.norm(original_lab - proofed_lab, axis=2)
    return np.where(delta >= max(0.1, float(threshold)), 255, 0).astype(np.uint8)


def color_settings(metadata: dict[str, Any]) -> dict[str, Any]:
    settings = metadata.setdefault("color_management", {})
    settings.setdefault("profile_name", "sRGB")
    settings.setdefault("rendering_intent", "perceptual")
    settings.setdefault("black_point_compensation", True)
    settings.setdefault("soft_proof_enabled", False)
    return settings
