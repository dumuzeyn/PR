from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import base64
import hashlib
import io
import json
import math
import uuid
import zipfile

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageDraw, ImageFont

from .performance import profiled, profiler
from .color_management import BIT_DEPTHS, COLOR_MODELS, color_settings, convert_icc, display_rgba, normalize_rgba, profile_bytes, profile_name, quantize_rgba, rgb_to_cmyk, rgb_to_lab, rgba_to_working, working_to_rgba
from .gradient_color import INTERPOLATION_SPACES, color_lookup, ordered_dither
from .gradient_noise import noise_lookup


_checker_cache: dict[tuple[int, int, int], np.ndarray] = {}
_brush_mask_cache: dict[int, np.ndarray] = {}
_retouch_mask_cache: dict[tuple[int, int], np.ndarray] = {}
_filter_mask_cache: dict[str, np.ndarray] = {}
_gradient_lookup_cache: dict[tuple[Any, ...], np.ndarray] = {}
BLEND_MODES = [
    "Normal",
    "Multiply",
    "Screen",
    "Overlay",
    "Soft Light",
    "Linear Light",
    "Darken",
    "Lighten",
    "Difference",
    "Color",
    "Luminosity",
]
RAW_EXTENSIONS = {".3fr", ".arw", ".cr2", ".cr3", ".dng", ".erf", ".kdc", ".mef", ".mos", ".mrw", ".nef", ".nrw", ".orf", ".pef", ".raf", ".raw", ".rw2", ".sr2", ".srf", ".x3f"}


@dataclass
class SourceAnchor:
    point: tuple[int, int] | None = None
    stroke_source: tuple[int, int] | None = None
    stroke_target: tuple[int, int] | None = None
    aligned: bool = True
    sampling: str = "current"
    offset: tuple[int, int] | None = None

    def set_source(self, point: tuple[int, int]) -> None:
        self.point = (int(point[0]), int(point[1]))
        self.stroke_source = None
        self.stroke_target = None
        self.offset = None

    def reset(self) -> None:
        self.point = None
        self.stroke_source = None
        self.stroke_target = None
        self.offset = None

    def begin_stroke(self, target: tuple[int, int]) -> bool:
        if self.point is None:
            return False
        self.stroke_target = (int(target[0]), int(target[1]))
        if self.aligned and self.offset is not None:
            self.stroke_source = (self.stroke_target[0] + self.offset[0], self.stroke_target[1] + self.offset[1])
        else:
            self.stroke_source = self.point
        if self.aligned and self.offset is None:
            self.offset = (self.stroke_source[0] - self.stroke_target[0], self.stroke_source[1] - self.stroke_target[1])
        return True

    def source_for(self, target: tuple[int, int]) -> tuple[int, int] | None:
        if self.point is None:
            return None
        if self.stroke_source is None or self.stroke_target is None:
            return self.point
        return (
            self.stroke_source[0] + int(target[0]) - self.stroke_target[0],
            self.stroke_source[1] + int(target[1]) - self.stroke_target[1],
        )

    def end_stroke(self) -> None:
        self.stroke_source = None
        self.stroke_target = None


class GradientEngine:
    TYPES = ("linear", "radial", "reflected", "diamond", "angular")
    INTERPOLATION_SPACES = INTERPOLATION_SPACES

    @staticmethod
    def normalize_stops(stops: list[Any] | None) -> list[tuple[float, tuple[int, int, int, int]]]:
        normalized: list[tuple[float, tuple[int, int, int, int]]] = []
        for stop in stops or []:
            if isinstance(stop, dict):
                position, color = stop.get("position", 0.0), stop.get("color", [0, 0, 0, 255])
            else:
                position, color = stop
            rgba = tuple(int(np.clip(value, 0, 255)) for value in list(color)[:4])
            if len(rgba) == 3:
                rgba = (*rgba, 255)
            if len(rgba) != 4:
                continue
            normalized.append((float(np.clip(position, 0.0, 1.0)), rgba))
        if len(normalized) < 2:
            normalized = [(0.0, (0, 0, 0, 255)), (1.0, (255, 255, 255, 255))]
        normalized.sort(key=lambda item: item[0])
        return normalized

    @staticmethod
    def normalize_opacity_stops(stops: list[Any] | None) -> list[tuple[float, float]]:
        normalized: list[tuple[float, float]] = []
        for stop in stops or []:
            if isinstance(stop, dict):
                position, opacity = stop.get("position", 0.0), stop.get("opacity", 1.0)
            else:
                position, opacity = stop
            value = float(opacity)
            if value > 1.0:
                value /= 100.0
            normalized.append((float(np.clip(position, 0.0, 1.0)), float(np.clip(value, 0.0, 1.0))))
        if len(normalized) < 2:
            normalized = [(0.0, 1.0), (1.0, 1.0)]
        normalized.sort(key=lambda item: item[0])
        return normalized

    @staticmethod
    def midpoint_axis(axis: np.ndarray, stops: list[Any] | None, positions: np.ndarray) -> np.ndarray:
        if len(positions) < 2:
            return axis
        midpoints = []
        source = stops or []
        for index in range(len(positions) - 1):
            raw = source[index] if index < len(source) and isinstance(source[index], dict) else {}
            midpoints.append(float(np.clip(raw.get("midpoint", 0.5), 0.01, 0.99)))
        adjusted = axis.copy()
        for index, midpoint in enumerate(midpoints):
            left, right = float(positions[index]), float(positions[index + 1])
            selected = (axis >= left) & (axis <= right)
            local = (axis[selected] - left) / max(1e-8, right - left)
            curved = np.where(
                local <= midpoint,
                local * (0.5 / midpoint),
                0.5 + (local - midpoint) * (0.5 / (1.0 - midpoint)),
            )
            adjusted[selected] = left + curved * (right - left)
        return adjusted

    @staticmethod
    def coordinates(
        width: int,
        height: int,
        start: tuple[float, float],
        end: tuple[float, float],
        kind: str = "linear",
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        xx = np.arange(width, dtype=np.float32)[None, :] + float(origin[0])
        yy = np.arange(height, dtype=np.float32)[:, None] + float(origin[1])
        sx, sy = float(start[0]), float(start[1])
        dx, dy = float(end[0]) - sx, float(end[1]) - sy
        length = max(1e-6, math.hypot(dx, dy))
        ux, uy = dx / length, dy / length
        rx, ry = xx - sx, yy - sy
        along = (rx * ux + ry * uy) / length
        across = (-rx * uy + ry * ux) / length
        kind = kind if kind in GradientEngine.TYPES else "linear"
        if kind == "radial":
            values = np.sqrt(rx * rx + ry * ry) / length
        elif kind == "reflected":
            values = np.abs(along)
        elif kind == "diamond":
            values = np.abs(along) + np.abs(across)
        elif kind == "angular":
            base = math.atan2(dy, dx)
            values = (np.arctan2(ry, rx) - base) / (2.0 * math.pi)
            values = np.mod(values, 1.0)
        else:
            values = along
        return np.clip(values, 0.0, 1.0)

    @classmethod
    def render(
        cls,
        width: int,
        height: int,
        start: tuple[float, float],
        end: tuple[float, float],
        stops: list[Any] | None,
        kind: str = "linear",
        origin: tuple[float, float] = (0.0, 0.0),
        opacity_stops: list[Any] | None = None,
        reverse: bool = False,
        dither: bool = False,
        transparency: bool = True,
        output_depth: int = 8,
        interpolation_space: str = "srgb",
        noise: dict[str, Any] | None = None,
    ) -> np.ndarray:
        normalized = cls.normalize_stops(stops)
        noise_enabled = bool(noise and noise.get("enabled", False))
        source_stops = stops or []
        midpoints = tuple(
            float(np.clip(
                source_stops[index].get("midpoint", 0.5)
                if index < len(source_stops) and isinstance(source_stops[index], dict) else 0.5,
                0.01,
                0.99,
            ))
            for index in range(len(normalized) - 1)
        )
        cache_key = None
        if opacity_stops is None and not noise_enabled:
            cache_key = (tuple(normalized), midpoints, interpolation_space, bool(transparency))
        lookup = _gradient_lookup_cache.get(cache_key) if cache_key is not None else None
        if lookup is None:
            positions = np.array([item[0] for item in normalized], dtype=np.float32)
            colors = np.array([item[1] for item in normalized], dtype=np.float32) / 255.0
            axis = np.arange(65537, dtype=np.float32) / 65536.0
            interpolation_axis = cls.midpoint_axis(axis, stops, positions)
            lookup = color_lookup(interpolation_axis, positions, colors, interpolation_space)
            opacity = cls.normalize_opacity_stops(opacity_stops)
            opacity_positions = np.array([item[0] for item in opacity], dtype=np.float32)
            opacity_values = np.array([item[1] for item in opacity], dtype=np.float32)
            alpha = np.interp(axis, opacity_positions, opacity_values)
            lookup[:, 3] = np.clip(lookup[:, 3] * alpha, 0.0, 1.0)
            if not transparency:
                lookup[:, 3] = 1.0
            if noise_enabled:
                lookup = noise_lookup(lookup, noise)
            if cache_key is not None:
                if len(_gradient_lookup_cache) >= 32:
                    _gradient_lookup_cache.pop(next(iter(_gradient_lookup_cache)))
                lookup.setflags(write=False)
                _gradient_lookup_cache[cache_key] = lookup
        dtype = {8: np.uint8, 16: np.uint16, 32: np.float32}.get(output_depth)
        if dtype is None:
            raise ValueError("Bit depth must be 8, 16 or 32")
        result = np.empty((height, width, 4), dtype=dtype)
        rows_per_chunk = max(1, min(height, 1_048_576 // max(1, width)))
        for top in range(0, height, rows_per_chunk):
            bottom = min(height, top + rows_per_chunk)
            chunk_origin = (origin[0], origin[1] + top)
            values = cls.coordinates(width, bottom - top, start, end, kind, chunk_origin)
            if reverse:
                values = 1.0 - values
            indices = np.clip(values * 65536.0, 0.0, 65536.0).astype(np.uint32)
            output = lookup[indices]
            if dither and width > 1 and height > 1:
                output = ordered_dither(output, output_depth, chunk_origin)
            result[top:bottom] = quantize_rgba(output, output_depth)
        return np.ascontiguousarray(result)


def blank_rgba(width: int, height: int, color=(255, 255, 255, 255)) -> np.ndarray:
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    arr[:, :] = color
    return arr


def pil_to_rgba_array(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGBA"), dtype=np.uint8)


def rgba_array_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")


def encode_png(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    rgba_array_to_pil(arr).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_png(text: str) -> np.ndarray:
    return pil_to_rgba_array(Image.open(io.BytesIO(base64.b64decode(text))))


def encode_array(arr: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, arr, allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_array(text: str) -> np.ndarray:
    return np.load(io.BytesIO(base64.b64decode(text)), allow_pickle=False)


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    """Return a stable linked-file identity without keeping the file open."""
    source = Path(path)
    stat = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()}


def image_metadata(image: Image.Image, path: str | Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source_path": str(path),
        "format": image.format or Path(path).suffix.lstrip(".").upper(),
        "mode": image.mode,
        "size": [image.width, image.height],
    }
    if image.info.get("dpi"):
        metadata["dpi"] = list(image.info["dpi"])
    try:
        exif = image.getexif()
        if exif:
            metadata["exif"] = {}
            for key, value in exif.items():
                name = ExifTags.TAGS.get(key, str(key))
                if isinstance(value, bytes):
                    value = value[:128].hex()
                elif not isinstance(value, (str, int, float, bool, list, tuple)):
                    value = str(value)
                metadata["exif"][name] = value
    except Exception:
        metadata["exif_error"] = "Could not read EXIF"
    return metadata

def blend_rgb(src: np.ndarray, dst: np.ndarray, mode: str, value_max: float = 255.0) -> np.ndarray:
    mode = mode if mode in BLEND_MODES else "Normal"
    s = src.astype(np.float32)
    d = dst.astype(np.float32)
    value_max = max(1e-7, float(value_max))
    if mode == "Multiply":
        return s * d / value_max
    if mode == "Screen":
        return value_max - (value_max - s) * (value_max - d) / value_max
    if mode == "Overlay":
        midpoint = value_max * 0.5
        return np.where(d <= midpoint, 2.0 * s * d / value_max, value_max - 2.0 * (value_max - s) * (value_max - d) / value_max)
    if mode == "Soft Light":
        sn = s / value_max
        dn = d / value_max
        out = (1.0 - 2.0 * sn) * dn * dn + 2.0 * sn * dn
        return out * value_max
    if mode == "Linear Light":
        return np.clip(d + 2.0 * s - value_max, 0.0, value_max)
    if mode == "Darken":
        return np.minimum(s, d)
    if mode == "Lighten":
        return np.maximum(s, d)
    if mode == "Difference":
        return np.abs(d - s)
    if mode in {"Color", "Luminosity"}:
        src_hls = cv2.cvtColor(np.clip(s / value_max, 0, 1).astype(np.float32), cv2.COLOR_RGB2HLS)
        dst_hls = cv2.cvtColor(np.clip(d / value_max, 0, 1).astype(np.float32), cv2.COLOR_RGB2HLS)
        out_hls = dst_hls.copy()
        if mode == "Color":
            out_hls[:, :, 0] = src_hls[:, :, 0]
            out_hls[:, :, 2] = src_hls[:, :, 2]
        else:
            out_hls[:, :, 1] = src_hls[:, :, 1]
        return cv2.cvtColor(out_hls, cv2.COLOR_HLS2RGB).astype(np.float32) * value_max
    return s

__all__ = [name for name in globals() if not name.startswith("__")]
