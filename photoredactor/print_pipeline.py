from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from .color_management import (
    color_settings,
    display_rgba,
    gamut_warning_mask,
    profile_bytes,
    profile_details,
    rgb_to_profile_cmyk,
    soft_proof_rgba,
)


def document_source_profile(document) -> str | bytes:
    settings = color_settings(document.metadata)
    encoded = settings.get("icc_base64")
    return base64.b64decode(encoded) if encoded else str(settings.get("profile_name", "sRGB"))


def proof_document(
    document,
    proof_profile: str | Path | bytes,
    rendering_intent: str = "relative",
    black_point_compensation: bool = True,
    gamut_warning: bool = False,
    gamut_threshold: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    source = document.composite_precision(False)
    proofed = soft_proof_rgba(
        source,
        document_source_profile(document),
        proof_profile,
        "sRGB",
        rendering_intent,
        black_point_compensation,
    )
    warning = gamut_warning_mask(source, proofed, gamut_threshold)
    preview = display_rgba(proofed)
    if gamut_warning and np.any(warning):
        overlay = warning.astype(np.float32)[:, :, None] / 255.0 * 0.72
        warning_color = np.zeros_like(preview, dtype=np.float32)
        warning_color[:, :, :3] = (255, 0, 210)
        warning_color[:, :, 3] = preview[:, :, 3]
        preview = np.clip(preview.astype(np.float32) * (1.0 - overlay) + warning_color * overlay, 0, 255).astype(np.uint8)
    return preview, warning


def cmyk_separation(
    document,
    output_profile: str | Path | bytes,
    rendering_intent: str = "relative",
    black_point_compensation: bool = True,
) -> np.ndarray:
    return rgb_to_profile_cmyk(
        document.composite_precision(False),
        document_source_profile(document),
        output_profile,
        rendering_intent,
        black_point_compensation,
    )


def print_preflight(
    document,
    output_profile: str | Path | bytes,
    rendering_intent: str = "relative",
    black_point_compensation: bool = True,
    ink_limit: float = 300.0,
) -> dict[str, Any]:
    cmyk = cmyk_separation(document, output_profile, rendering_intent, black_point_compensation)
    total_ink = cmyk.astype(np.float32).sum(axis=2) / 255.0 * 100.0
    composite = document.composite_precision(False)
    transparent = int(np.count_nonzero(composite[:, :, 3] < 0.999))
    over_limit = int(np.count_nonzero(total_ink > float(ink_limit)))
    issues: list[str] = []
    if document.dpi < 150:
        issues.append(f"Низкое разрешение: {document.dpi} DPI")
    elif document.dpi < 240:
        issues.append(f"Разрешение ниже рекомендуемых 240 DPI: {document.dpi} DPI")
    if transparent:
        issues.append(f"Полупрозрачных пикселей: {transparent}")
    if over_limit:
        issues.append(f"Превышение лимита {ink_limit:.0f}%: {over_limit} пикселей")
    details = profile_details(output_profile)
    return {
        "profile": details,
        "dpi": int(document.dpi),
        "width": int(document.width),
        "height": int(document.height),
        "ink_limit": float(ink_limit),
        "maximum_ink": float(total_ink.max(initial=0.0)),
        "mean_ink": float(total_ink.mean()),
        "over_limit_pixels": over_limit,
        "transparent_pixels": transparent,
        "issues": issues,
        "ready": not issues,
    }


def export_cmyk_tiff(
    document,
    path: str | Path,
    output_profile: str | Path | bytes,
    rendering_intent: str = "relative",
    black_point_compensation: bool = True,
) -> None:
    cmyk = cmyk_separation(document, output_profile, rendering_intent, black_point_compensation)
    icc = profile_bytes(output_profile)
    if icc is None:
        raise ValueError("Печатный ICC-профиль должен быть выбран из файла")
    tifffile.imwrite(
        path,
        cmyk,
        photometric="separated",
        resolution=(float(document.dpi), float(document.dpi)),
        resolutionunit="INCH",
        iccprofile=icc,
        metadata=None,
    )


__all__ = [name for name in globals() if not name.startswith("__")]
