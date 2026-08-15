from __future__ import annotations

import base64
import json
from pathlib import Path
import re
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
from .render_ops import layer_alpha_canvas
from .spot_colors import document_spot_colors, spot_settings


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
    spots = spot_separations(document)
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
        "spot_colors": [
            {"name": item["color"].name, "source": item["color"].source, "coverage": item["coverage"]}
            for item in spots.values()
        ],
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


def spot_separations(document) -> dict[str, dict[str, Any]]:
    colors = {color.id: color for color in document_spot_colors(document)}
    assignments = spot_settings(document.metadata).get("assignments", {})
    plates: dict[str, dict[str, Any]] = {}
    for layer in document.layers:
        color = colors.get(assignments.get(layer.id))
        if color is None or not layer.visible:
            continue
        alpha = layer_alpha_canvas(document, layer).astype(np.float32) / 255.0
        alpha *= float(np.clip(layer.opacity, 0.0, 1.0))
        if color.id not in plates:
            plates[color.id] = {"color": color, "mask": np.zeros((document.height, document.width), dtype=np.float32)}
        existing = plates[color.id]["mask"]
        plates[color.id]["mask"] = 1.0 - (1.0 - existing) * (1.0 - alpha)
    for value in plates.values():
        mask = np.clip(value["mask"] * 255.0, 0, 255).astype(np.uint8)
        value["mask"] = mask
        value["coverage"] = float(np.count_nonzero(mask) / mask.size)
    return plates


def _safe_plate_name(value: str) -> str:
    cleaned = re.sub(r"[^\w .-]+", "_", value, flags=re.UNICODE).strip(" .")
    return cleaned or "Spot"


def export_color_separations(
    document,
    directory: str | Path,
    output_profile: str | Path | bytes,
    rendering_intent: str = "relative",
    black_point_compensation: bool = True,
) -> Path:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    cmyk = cmyk_separation(document, output_profile, rendering_intent, black_point_compensation)
    resolution = (float(document.dpi), float(document.dpi))
    files: list[dict[str, Any]] = []
    for index, name in enumerate(("Cyan", "Magenta", "Yellow", "Black")):
        path = target / f"{index + 1:02d}_{name}.tif"
        tifffile.imwrite(path, cmyk[:, :, index], photometric="minisblack", resolution=resolution, resolutionunit="INCH", metadata=None)
        files.append({"type": "process", "name": name, "file": path.name})
    for index, value in enumerate(spot_separations(document).values(), 5):
        color = value["color"]
        path = target / f"{index:02d}_Spot_{_safe_plate_name(color.name)}.tif"
        tifffile.imwrite(path, value["mask"], photometric="minisblack", resolution=resolution, resolutionunit="INCH", metadata=None)
        files.append(
            {
                "type": "spot",
                "name": color.name,
                "source": color.source,
                "lab": list(color.lab),
                "alternate_rgb": list(color.alternate_rgb),
                "coverage": value["coverage"],
                "file": path.name,
            }
        )
    manifest = target / "separations.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "UZYRO color separations",
                "version": 1,
                "document": {"width": document.width, "height": document.height, "dpi": document.dpi},
                "profile": profile_details(output_profile),
                "plates": files,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest


__all__ = [name for name in globals() if not name.startswith("__")]
