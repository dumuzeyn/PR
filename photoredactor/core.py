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
from .color_management import BIT_DEPTHS, COLOR_MODELS, color_settings, convert_icc, display_rgba, profile_bytes, profile_name, quantize_rgba, rgb_to_cmyk, rgb_to_lab


_checker_cache: dict[tuple[int, int, int], np.ndarray] = {}
_brush_mask_cache: dict[int, np.ndarray] = {}
_retouch_mask_cache: dict[tuple[int, int], np.ndarray] = {}
_filter_mask_cache: dict[str, np.ndarray] = {}
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
    def coordinates(
        width: int,
        height: int,
        start: tuple[float, float],
        end: tuple[float, float],
        kind: str = "linear",
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        xx += float(origin[0])
        yy += float(origin[1])
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
    ) -> np.ndarray:
        values = cls.coordinates(width, height, start, end, kind, origin)
        normalized = cls.normalize_stops(stops)
        positions = np.array([item[0] for item in normalized], dtype=np.float32)
        colors = np.array([item[1] for item in normalized], dtype=np.float32)
        output = np.empty((height, width, 4), dtype=np.float32)
        flat = values.reshape(-1)
        for channel in range(4):
            output[:, :, channel] = np.interp(flat, positions, colors[:, channel]).reshape(height, width)
        return np.clip(output, 0, 255).astype(np.uint8)


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


@dataclass
class Layer:
    name: str
    pixels: np.ndarray
    x: int = 0
    y: int = 0
    opacity: float = 1.0
    visible: bool = True
    locked: bool = False
    mask: np.ndarray | None = None
    mask_enabled: bool = True
    mask_linked: bool = True
    mask_density: float = 1.0
    mask_feather: float = 0.0
    blend_mode: str = "Normal"
    clipping: bool = False
    effects: dict[str, Any] = field(default_factory=dict)
    filters: list[dict[str, Any]] = field(default_factory=list)
    kind: str = "raster"
    text_data: dict[str, Any] | None = None
    shape_data: dict[str, Any] | None = None
    adjustment: dict[str, Any] | None = None
    smart_data: dict[str, Any] | None = None
    smart_source: np.ndarray | None = field(default=None, repr=False, compare=False)
    working_pixels: np.ndarray | None = field(default=None, repr=False, compare=False)
    working_model: str = "RGBA"
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    pixels_revision: int = field(default=0, repr=False, compare=False)
    mask_revision: int = field(default=0, repr=False, compare=False)

    def touch_pixels(self) -> None:
        self.pixels_revision += 1
        if self.working_pixels is not None:
            if self.working_model == "Lab":
                self.working_pixels = rgb_to_lab(self.pixels)
            elif self.working_model == "CMYK":
                alpha = self.pixels[:, :, 3].astype(np.float32) / 255.0
                self.working_pixels = np.dstack((rgb_to_cmyk(self.pixels), alpha))
            elif self.working_pixels.dtype == np.uint16:
                self.working_pixels = self.pixels.astype(np.uint16) * 257
            elif self.working_pixels.dtype == np.float32:
                self.working_pixels = self.pixels.astype(np.float32) / 255.0

    def touch_mask(self) -> None:
        self.mask_revision += 1

    def clone(self) -> "Layer":
        return Layer(
            name=f"{self.name} copy",
            pixels=self.pixels.copy(),
            x=self.x,
            y=self.y,
            opacity=self.opacity,
            visible=self.visible,
            locked=self.locked,
            mask=None if self.mask is None else self.mask.copy(),
            mask_enabled=self.mask_enabled,
            mask_linked=self.mask_linked,
            mask_density=self.mask_density,
            mask_feather=self.mask_feather,
            blend_mode=self.blend_mode,
            clipping=self.clipping,
            effects=json.loads(json.dumps(self.effects)),
            filters=json.loads(json.dumps(self.filters)),
            kind=self.kind,
            text_data=None if self.text_data is None else json.loads(json.dumps(self.text_data)),
            shape_data=None if self.shape_data is None else json.loads(json.dumps(self.shape_data)),
            adjustment=None if self.adjustment is None else dict(self.adjustment),
            smart_data=None if self.smart_data is None else json.loads(json.dumps(self.smart_data, ensure_ascii=False)),
            smart_source=None if self.smart_source is None else self.smart_source.copy(),
            working_pixels=None if self.working_pixels is None else self.working_pixels.copy(),
            working_model=self.working_model,
        )


@dataclass
class Document:
    width: int
    height: int
    dpi: int = 300
    color_model: str = "RGBA"
    bit_depth: int = 8
    background: tuple[int, int, int, int] = (255, 255, 255, 255)
    layers: list[Layer] = field(default_factory=list)
    active_layer: int = 0
    path: str | None = None
    dirty: bool = False
    selection_mask: np.ndarray | None = None
    saved_selections: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(cls, width: int = 1280, height: int = 900, background=(255, 255, 255, 255)) -> "Document":
        doc = cls(width=width, height=height, background=background, metadata={"source": "new document"})
        doc.layers.append(Layer("Background", blank_rgba(width, height, background)))
        return doc

    @classmethod
    def from_image(cls, path: str | Path) -> "Document":
        if Path(path).suffix.lower() in RAW_EXTENSIONS:
            return cls.from_raw(path)
        image = Image.open(path)
        arr = pil_to_rgba_array(image)
        h, w = arr.shape[:2]
        dpi = image.info.get("dpi", (300, 300))[0] if image.info.get("dpi") else 300
        doc = cls(width=w, height=h, dpi=dpi, metadata=image_metadata(image, path))
        doc.layers.append(Layer(Path(path).stem, arr))
        doc.path = str(path)
        return doc

    @classmethod
    def from_raw(cls, path: str | Path) -> "Document":
        try:
            import rawpy
        except ImportError as exc:
            raise RuntimeError("Для открытия RAW требуется компонент rawpy. Переустановите полную сборку PhotoRedactor.") from exc
        with rawpy.imread(str(path)) as raw:
            rgb16 = raw.postprocess(use_camera_wb=True, output_bps=16, no_auto_bright=False)
            rgba16 = np.dstack((rgb16, np.full(rgb16.shape[:2], 65535, dtype=np.uint16)))
            metadata = raw.metadata
            white_balance = raw.camera_whitebalance
            raw_info = {
                "camera": " ".join(part for part in (str(metadata.make or ""), str(metadata.model or "")) if part).strip(),
                "iso": float(metadata.iso_speed or 0),
                "shutter": float(metadata.shutter or 0),
                "aperture": float(metadata.aperture or 0),
                "focal_length": float(metadata.focal_len or 0),
                "timestamp": str(metadata.timestamp or ""),
                "white_balance": [] if white_balance is None else [float(value) for value in white_balance],
            }
        h, w = rgba16.shape[:2]
        layer = Layer(Path(path).stem, display_rgba(rgba16), working_pixels=rgba16)
        doc = cls(
            width=w,
            height=h,
            bit_depth=16,
            metadata={"source": str(Path(path).resolve()), "format": "RAW", "raw": raw_info},
            layers=[layer],
            path=str(path),
        )
        doc.assign_color_profile("sRGB")
        doc.dirty = False
        return doc

    @property
    def layer(self) -> Layer:
        return self.layers[self.active_layer]

    def get_layer(self, layer_id: str) -> Layer | None:
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        return None

    def raw_state(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "dpi": self.dpi,
            "color_model": self.color_model,
            "bit_depth": self.bit_depth,
            "background": self.background,
            "active_layer": self.active_layer,
            "path": self.path,
            "selection_mask": None if self.selection_mask is None else self.selection_mask.copy(),
            "saved_selections": {name: mask.copy() for name, mask in self.saved_selections.items()},
            "metadata": json.loads(json.dumps(self.metadata, ensure_ascii=False)),
            "layers": [
                {
                    "id": layer.id,
                    "name": layer.name,
                    "x": layer.x,
                    "y": layer.y,
                    "opacity": layer.opacity,
                    "visible": layer.visible,
                    "locked": layer.locked,
                    "mask": None if layer.mask is None else layer.mask.copy(),
                    "mask_enabled": layer.mask_enabled,
                    "mask_linked": layer.mask_linked,
                    "mask_density": layer.mask_density,
                    "mask_feather": layer.mask_feather,
                    "blend_mode": layer.blend_mode,
                    "clipping": layer.clipping,
                    "effects": json.loads(json.dumps(layer.effects)),
                    "filters": json.loads(json.dumps(layer.filters)),
                    "kind": layer.kind,
                    "text_data": None if layer.text_data is None else json.loads(json.dumps(layer.text_data)),
                    "shape_data": None if layer.shape_data is None else dict(layer.shape_data),
                    "adjustment": None if layer.adjustment is None else dict(layer.adjustment),
                    "smart_data": None if layer.smart_data is None else json.loads(json.dumps(layer.smart_data, ensure_ascii=False)),
                    "smart_source": None if layer.smart_source is None else layer.smart_source.copy(),
                    "working_pixels": None if layer.working_pixels is None else layer.working_pixels.copy(),
                    "working_model": layer.working_model,
                    "pixels": layer.pixels.copy(),
                }
                for layer in self.layers
            ],
        }

    def restore_raw_state(self, data: dict[str, Any]) -> None:
        self.width = int(data["width"])
        self.height = int(data["height"])
        self.dpi = int(data.get("dpi", 300))
        self.color_model = data.get("color_model", "RGBA")
        self.bit_depth = int(data.get("bit_depth", 8))
        self.background = tuple(data.get("background", (255, 255, 255, 255)))
        self.active_layer = int(data.get("active_layer", 0))
        self.path = data.get("path")
        self.metadata = data.get("metadata", {})
        selection = data.get("selection_mask")
        self.selection_mask = None if selection is None else selection.copy()
        self.saved_selections = {name: mask.copy() for name, mask in data.get("saved_selections", {}).items()}
        self.layers = []
        for raw in data["layers"]:
            self.layers.append(
                Layer(
                    name=raw["name"],
                    pixels=raw["pixels"].copy(),
                    x=int(raw.get("x", 0)),
                    y=int(raw.get("y", 0)),
                    opacity=float(raw.get("opacity", 1.0)),
                    visible=bool(raw.get("visible", True)),
                    locked=bool(raw.get("locked", False)),
                    mask=None if raw.get("mask") is None else raw["mask"].copy(),
                    mask_enabled=bool(raw.get("mask_enabled", True)),
                    mask_linked=bool(raw.get("mask_linked", True)),
                    mask_density=float(raw.get("mask_density", 1.0)),
                    mask_feather=float(raw.get("mask_feather", 0.0)),
                    blend_mode=raw.get("blend_mode", "Normal"),
                    clipping=bool(raw.get("clipping", False)),
                    effects=raw.get("effects", {}),
                    filters=json.loads(json.dumps(raw.get("filters", []))),
                    kind=raw.get("kind", "raster"),
                    text_data=raw.get("text_data"),
                    shape_data=raw.get("shape_data"),
                    adjustment=raw.get("adjustment"),
                    smart_data=raw.get("smart_data"),
                    smart_source=None if raw.get("smart_source") is None else raw["smart_source"].copy(),
                    working_pixels=None if raw.get("working_pixels") is None else raw["working_pixels"].copy(),
                    working_model=raw.get("working_model", "RGBA"),
                    id=raw.get("id", uuid.uuid4().hex),
                )
            )
        self.active_layer = min(self.active_layer, max(0, len(self.layers) - 1))

    def snapshot(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "dpi": self.dpi,
            "color_model": self.color_model,
            "bit_depth": self.bit_depth,
            "background": list(self.background),
            "active_layer": self.active_layer,
            "selection": None if self.selection_mask is None else encode_png(np.dstack([self.selection_mask] * 4)),
            "saved_selections": {name: encode_png(np.dstack([mask] * 4)) for name, mask in self.saved_selections.items()},
            "metadata": self.metadata,
            "layers": [
                {
                    "name": layer.name,
                    "x": layer.x,
                    "y": layer.y,
                    "opacity": layer.opacity,
                    "visible": layer.visible,
                    "locked": layer.locked,
                    "mask": None if layer.mask is None else encode_png(np.dstack([layer.mask] * 4)),
                    "mask_enabled": layer.mask_enabled,
                    "mask_linked": layer.mask_linked,
                    "mask_density": layer.mask_density,
                    "mask_feather": layer.mask_feather,
                    "blend_mode": layer.blend_mode,
                    "clipping": layer.clipping,
                    "effects": layer.effects,
                    "filters": layer.filters,
                    "kind": layer.kind,
                    "text_data": layer.text_data,
                    "shape_data": layer.shape_data,
                    "adjustment": layer.adjustment,
                    "smart_data": layer.smart_data,
                    "smart_source": None if layer.smart_source is None else encode_png(layer.smart_source),
                    "pixels": encode_png(layer.pixels),
                }
                for layer in self.layers
            ],
        }

    @classmethod
    def restore(cls, data: dict[str, Any]) -> "Document":
        doc = cls(
            width=int(data["width"]),
            height=int(data["height"]),
            dpi=int(data.get("dpi", 300)),
            color_model=data.get("color_model", "RGBA"),
            bit_depth=int(data.get("bit_depth", 8)),
            background=tuple(data.get("background", [255, 255, 255, 255])),
        )
        doc.layers = []
        for raw in data["layers"]:
            doc.layers.append(
                Layer(
                    name=raw["name"],
                    pixels=decode_png(raw["pixels"]),
                    x=int(raw.get("x", 0)),
                    y=int(raw.get("y", 0)),
                    opacity=float(raw.get("opacity", 1.0)),
                    visible=bool(raw.get("visible", True)),
                    locked=bool(raw.get("locked", False)),
                    mask=None if raw.get("mask") is None else decode_png(raw["mask"])[:, :, 0],
                    mask_enabled=bool(raw.get("mask_enabled", True)),
                    mask_linked=bool(raw.get("mask_linked", True)),
                    mask_density=float(raw.get("mask_density", 1.0)),
                    mask_feather=float(raw.get("mask_feather", 0.0)),
                    blend_mode=raw.get("blend_mode", "Normal"),
                    clipping=bool(raw.get("clipping", False)),
                    effects=raw.get("effects", {}),
                    filters=raw.get("filters", []),
                    kind=raw.get("kind", "raster"),
                    text_data=raw.get("text_data"),
                    shape_data=raw.get("shape_data"),
                    adjustment=raw.get("adjustment"),
                    smart_data=raw.get("smart_data"),
                    smart_source=None if raw.get("smart_source") is None else decode_png(raw["smart_source"]),
                    id=raw.get("id", uuid.uuid4().hex),
                )
            )
        doc.active_layer = min(int(data.get("active_layer", 0)), max(0, len(doc.layers) - 1))
        if data.get("selection"):
            doc.selection_mask = decode_png(data["selection"])[:, :, 0]
        doc.saved_selections = {name: decode_png(mask)[:, :, 0] for name, mask in data.get("saved_selections", {}).items()}
        doc.metadata = data.get("metadata", {})
        return doc

    def save_project(self, path: str | Path) -> None:
        manifest = {
            "width": self.width,
            "height": self.height,
            "dpi": self.dpi,
            "color_model": self.color_model,
            "bit_depth": self.bit_depth,
            "background": list(self.background),
            "active_layer": self.active_layer,
            "selection": "selection.png" if self.selection_mask is not None else None,
            "saved_selections": {},
            "metadata": self.metadata,
            "layers": [],
        }
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            if self.selection_mask is not None:
                buf = io.BytesIO()
                rgba_array_to_pil(np.dstack([self.selection_mask] * 4)).save(buf, "PNG")
                zf.writestr("selection.png", buf.getvalue())
            for i, (name, mask) in enumerate(self.saved_selections.items()):
                selection_path = f"selections/{i:04d}.png"
                mask_buf = io.BytesIO()
                rgba_array_to_pil(np.dstack([mask] * 4)).save(mask_buf, "PNG")
                zf.writestr(selection_path, mask_buf.getvalue())
                manifest["saved_selections"][name] = selection_path
            for i, layer in enumerate(self.layers):
                layer_path = f"layers/{i:04d}.png"
                manifest["layers"].append(
                    {
                        "id": layer.id,
                        "name": layer.name,
                        "x": layer.x,
                        "y": layer.y,
                        "opacity": layer.opacity,
                        "visible": layer.visible,
                        "locked": layer.locked,
                        "mask": f"masks/{i:04d}.png" if layer.mask is not None else None,
                        "mask_enabled": layer.mask_enabled,
                        "mask_linked": layer.mask_linked,
                        "mask_density": layer.mask_density,
                        "mask_feather": layer.mask_feather,
                        "blend_mode": layer.blend_mode,
                        "clipping": layer.clipping,
                        "effects": layer.effects,
                        "filters": layer.filters,
                        "kind": layer.kind,
                        "text_data": layer.text_data,
                        "shape_data": layer.shape_data,
                        "adjustment": layer.adjustment,
                        "smart_data": layer.smart_data,
                        "smart_source": f"smart/{i:04d}.png" if layer.smart_source is not None else None,
                        "working_pixels": f"working/{i:04d}.npy" if layer.working_pixels is not None else None,
                        "working_model": layer.working_model,
                        "pixels": layer_path,
                    }
                )
                buf = io.BytesIO()
                rgba_array_to_pil(layer.pixels).save(buf, "PNG")
                zf.writestr(layer_path, buf.getvalue())
                if layer.smart_source is not None:
                    smart_buf = io.BytesIO()
                    rgba_array_to_pil(layer.smart_source).save(smart_buf, "PNG")
                    zf.writestr(f"smart/{i:04d}.png", smart_buf.getvalue())
                if layer.working_pixels is not None:
                    working_buf = io.BytesIO()
                    np.save(working_buf, layer.working_pixels, allow_pickle=False)
                    zf.writestr(f"working/{i:04d}.npy", working_buf.getvalue())
                if layer.mask is not None:
                    mask_buf = io.BytesIO()
                    rgba_array_to_pil(np.dstack([layer.mask] * 4)).save(mask_buf, "PNG")
                    zf.writestr(f"masks/{i:04d}.png", mask_buf.getvalue())
            zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False).encode("utf-8"))
        self.path = str(path)
        self.dirty = False

    @classmethod
    def open_project(cls, path: str | Path) -> "Document":
        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
            if "manifest.json" not in names and "document.json" in names:
                data = json.loads(zf.read("document.json").decode("utf-8"))
                doc = cls.restore(data)
                doc.path = str(path)
                return doc
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            doc = cls(
                width=int(manifest["width"]),
                height=int(manifest["height"]),
                dpi=int(manifest.get("dpi", 300)),
                color_model=manifest.get("color_model", "RGBA"),
                bit_depth=int(manifest.get("bit_depth", 8)),
                background=tuple(manifest.get("background", [255, 255, 255, 255])),
                metadata=manifest.get("metadata", {}),
            )
            doc.layers = []
            for raw in manifest["layers"]:
                pixels = pil_to_rgba_array(Image.open(io.BytesIO(zf.read(raw["pixels"]))))
                doc.layers.append(
                    Layer(
                        name=raw["name"],
                        pixels=pixels,
                        x=int(raw.get("x", 0)),
                        y=int(raw.get("y", 0)),
                        opacity=float(raw.get("opacity", 1.0)),
                        visible=bool(raw.get("visible", True)),
                        locked=bool(raw.get("locked", False)),
                        mask=None if raw.get("mask") is None else pil_to_rgba_array(Image.open(io.BytesIO(zf.read(raw["mask"]))))[:, :, 0],
                        mask_enabled=bool(raw.get("mask_enabled", True)),
                        mask_linked=bool(raw.get("mask_linked", True)),
                        mask_density=float(raw.get("mask_density", 1.0)),
                        mask_feather=float(raw.get("mask_feather", 0.0)),
                        blend_mode=raw.get("blend_mode", "Normal"),
                        clipping=bool(raw.get("clipping", False)),
                        effects=raw.get("effects", {}),
                        filters=raw.get("filters", []),
                        kind=raw.get("kind", "raster"),
                        text_data=raw.get("text_data"),
                        shape_data=raw.get("shape_data"),
                        adjustment=raw.get("adjustment"),
                        smart_data=raw.get("smart_data"),
                        smart_source=None if not raw.get("smart_source") else pil_to_rgba_array(Image.open(io.BytesIO(zf.read(raw["smart_source"])))),
                        working_pixels=None if not raw.get("working_pixels") else np.load(io.BytesIO(zf.read(raw["working_pixels"])), allow_pickle=False),
                        working_model=raw.get("working_model", "RGBA"),
                        id=raw.get("id", uuid.uuid4().hex),
                    )
                )
            doc.active_layer = min(int(manifest.get("active_layer", 0)), max(0, len(doc.layers) - 1))
            if manifest.get("selection"):
                doc.selection_mask = pil_to_rgba_array(Image.open(io.BytesIO(zf.read(manifest["selection"]))))[:, :, 0]
            doc.saved_selections = {}
            for name, selection_path in manifest.get("saved_selections", {}).items():
                doc.saved_selections[name] = pil_to_rgba_array(Image.open(io.BytesIO(zf.read(selection_path))))[:, :, 0]
        doc.path = str(path)
        return doc

    def composite(self, checker: bool = False) -> np.ndarray:
        with profiler.measure("render.composite.reference"):
            out = checker_background(self.width, self.height).copy() if checker else blank_rgba(self.width, self.height, (0, 0, 0, 0))
            previous_alpha: np.ndarray | None = None
            for layer in self.layers:
                if layer.visible:
                    if layer.kind == "adjustment" and layer.adjustment is not None:
                        clipping_mask = previous_alpha if layer.clipping and previous_alpha is not None else None
                        apply_adjustment_layer(out, layer, clipping_mask)
                    else:
                        layer_pixels = render_layer_pixels(layer)
                        alpha_mask = effective_layer_mask(layer) if layer.mask_enabled else None
                        if layer.clipping and previous_alpha is not None:
                            clipping_mask = document_alpha_to_layer_mask(previous_alpha, layer)
                            alpha_mask = clipping_mask if alpha_mask is None else np.minimum(alpha_mask, clipping_mask)
                        for pixels, x, y, opacity, blend_mode in render_layer_effects(layer, layer_pixels):
                            alpha_blend_inplace(out, pixels, x, y, opacity, None, 1.0, blend_mode)
                        alpha_blend_inplace(out, layer_pixels, layer.x, layer.y, layer.opacity, alpha_mask, layer.mask_density, layer.blend_mode)
                        previous_alpha = layer_alpha_canvas(self, layer, layer_pixels)
            return out

    def export_flat(self, path: str | Path, quality: int = 95) -> None:
        composite = self.composite(checker=False)
        img = rgba_array_to_pil(composite)
        suffix = Path(path).suffix.lower()
        if suffix in [".jpg", ".jpeg"]:
            img.convert("RGB").save(path, quality=max(1, min(100, int(quality))), subsampling=0)
        elif suffix == ".webp":
            img.save(path, quality=max(1, min(100, int(quality))))
        elif suffix in {".tif", ".tiff"} and self.bit_depth == 16:
            bgra = cv2.cvtColor(composite.astype(np.uint16) * 257, cv2.COLOR_RGBA2BGRA)
            if not cv2.imwrite(str(path), bgra):
                raise OSError(f"Could not write TIFF: {path}")
        else:
            img.save(path)
        self.dirty = False

    def set_bit_depth(self, bit_depth: int) -> None:
        bit_depth = int(bit_depth)
        if bit_depth not in BIT_DEPTHS:
            raise ValueError("Bit depth must be 8, 16 or 32")
        for layer in self.layers:
            if layer.working_model == "RGBA":
                source = layer.working_pixels if layer.working_pixels is not None else layer.pixels
                layer.working_pixels = None if bit_depth == 8 else quantize_rgba(source, bit_depth)
                layer.pixels = display_rgba(source)
            layer.pixels_revision += 1
        self.bit_depth = bit_depth
        self.metadata["bit_depth"] = bit_depth
        self.dirty = True

    def set_color_model(self, color_model: str) -> None:
        if color_model not in COLOR_MODELS:
            raise ValueError(f"Unsupported color model: {color_model}")
        for layer in self.layers:
            if layer.kind == "adjustment" or layer.pixels.size == 0:
                continue
            if color_model == "Lab":
                layer.working_pixels = rgb_to_lab(layer.pixels)
            elif color_model == "CMYK":
                alpha = layer.pixels[:, :, 3].astype(np.float32) / 255.0
                layer.working_pixels = np.dstack((rgb_to_cmyk(layer.pixels), alpha))
            else:
                layer.working_pixels = None if self.bit_depth == 8 else quantize_rgba(layer.pixels, self.bit_depth)
            layer.working_model = color_model
            layer.pixels_revision += 1
        self.color_model = color_model
        self.metadata["color_model"] = color_model
        self.dirty = True

    def assign_color_profile(self, profile: str | Path | bytes | None) -> None:
        settings = color_settings(self.metadata)
        settings["profile_name"] = profile_name(profile)
        raw = profile_bytes(profile)
        if raw is None:
            settings.pop("icc_base64", None)
        else:
            settings["icc_base64"] = base64.b64encode(raw).decode("ascii")
        self.dirty = True

    def convert_color_profile(self, destination: str | Path | bytes | None) -> None:
        settings = color_settings(self.metadata)
        encoded = settings.get("icc_base64")
        source: str | bytes | None = base64.b64decode(encoded) if encoded else settings.get("profile_name", "sRGB")
        for layer in self.layers:
            if layer.kind == "adjustment" or layer.pixels.size == 0:
                continue
            layer.pixels = convert_icc(layer.pixels, source, destination)
            layer.touch_pixels()
        self.assign_color_profile(destination)
        self.dirty = True

    def add_layer(self, name="Layer", pixels: np.ndarray | None = None) -> None:
        if pixels is None:
            pixels = blank_rgba(self.width, self.height, (0, 0, 0, 0))
        self.layers.append(Layer(name, pixels))
        self.active_layer = len(self.layers) - 1
        self.dirty = True

    def frequency_separate_active(self, radius: float = 8.0, texture_strength: float = 1.0) -> bool:
        source = self.layer
        if source.locked or source.kind == "adjustment" or source.pixels.size == 0:
            return False
        low_pixels, high_pixels = frequency_separation(render_layer_pixels(source), radius, texture_strength)
        low = Layer(
            name=f"Низкие частоты - {source.name}",
            pixels=low_pixels,
            x=source.x,
            y=source.y,
            mask=None if source.mask is None else source.mask.copy(),
            mask_enabled=source.mask_enabled,
            mask_linked=source.mask_linked,
            mask_density=source.mask_density,
            mask_feather=source.mask_feather,
        )
        high = Layer(
            name=f"Высокие частоты - {source.name}",
            pixels=high_pixels,
            x=source.x,
            y=source.y,
            mask=None if source.mask is None else source.mask.copy(),
            mask_enabled=source.mask_enabled,
            mask_linked=source.mask_linked,
            mask_density=source.mask_density,
            mask_feather=source.mask_feather,
            blend_mode="Linear Light",
        )
        source.visible = False
        insert_at = self.active_layer + 1
        self.layers[insert_at:insert_at] = [low, high]
        self.active_layer = insert_at + 1
        self.dirty = True
        return True

    def place_image(self, path: str | Path, linked: bool = False) -> Layer:
        image = Image.open(path)
        pixels = pil_to_rgba_array(image)
        h, w = pixels.shape[:2]
        source_path = str(Path(path).resolve())
        layer = Layer(
            Path(path).stem,
            pixels,
            x=(self.width - w) // 2,
            y=(self.height - h) // 2,
            kind="linked" if linked else "embedded",
            smart_data={
                "linked": bool(linked),
                "source_path": source_path,
                "original_size": [w, h],
                "fingerprint": file_fingerprint(source_path),
                "transform": {"width": w, "height": h, "angle": 0.0, "flip_horizontal": False, "flip_vertical": False},
            },
            smart_source=pixels.copy(),
        )
        self.layers.append(layer)
        self.active_layer = len(self.layers) - 1
        embedded = list(self.metadata.get("embedded_images", []))
        embedded.append({"name": layer.name, "source_path": source_path, "size": [w, h], "linked": bool(linked)})
        self.metadata["embedded_images"] = embedded
        self.dirty = True
        return layer

    def linked_layer_status(self, layer: Layer | None = None) -> dict[str, Any]:
        layer = layer or self.layer
        data = layer.smart_data or {}
        path = data.get("source_path")
        if layer.kind != "linked" or not path:
            return {"status": "embedded", "path": path}
        if not Path(path).exists():
            return {"status": "missing", "path": path}
        saved = data.get("fingerprint") or {}
        stat = Path(path).stat()
        if saved and int(saved.get("size", -1)) == stat.st_size and int(saved.get("mtime_ns", -1)) == stat.st_mtime_ns:
            return {"status": "current", "path": path, "saved": saved, "current": saved}
        current = file_fingerprint(path)
        modified = bool(saved) and (current.get("sha256") != saved.get("sha256"))
        return {"status": "modified" if modified else "current", "path": path, "saved": saved, "current": current}

    def update_linked_layer(self) -> bool:
        layer = self.layer
        smart_data = layer.smart_data or {}
        source_path = smart_data.get("source_path")
        if not source_path or not Path(source_path).exists():
            return False
        image = Image.open(source_path)
        pixels = pil_to_rgba_array(image)
        source_h, source_w = pixels.shape[:2]
        layer.smart_source = pixels.copy()
        layer.kind = "linked"
        layer.smart_data = {
            **smart_data,
            "linked": True,
            "source_path": str(Path(source_path).resolve()),
            "original_size": [source_w, source_h],
            "fingerprint": file_fingerprint(source_path),
        }
        render_smart_object(layer)
        layer.touch_pixels()
        self.dirty = True
        return True

    def relink_active_layer(self, path: str | Path) -> bool:
        if not Path(path).exists():
            return False
        layer = self.layer
        layer.kind = "linked"
        layer.smart_data = {
            **(layer.smart_data or {}),
            "linked": True,
            "source_path": str(Path(path).resolve()),
        }
        return self.update_linked_layer()

    def convert_active_smart_to_embedded(self) -> bool:
        layer = self.layer
        if layer.kind not in {"linked", "embedded"}:
            return False
        layer.kind = "embedded"
        layer.smart_data = {**(layer.smart_data or {}), "linked": False}
        self.dirty = True
        return True

    def replace_active_smart_contents(self, path: str | Path, linked: bool | None = None) -> bool:
        layer = self.layer
        if layer.kind not in {"linked", "embedded"} or not Path(path).exists():
            return False
        pixels = pil_to_rgba_array(Image.open(path))
        h, w = pixels.shape[:2]
        keep_linked = layer.kind == "linked" if linked is None else bool(linked)
        layer.smart_source = pixels.copy()
        layer.kind = "linked" if keep_linked else "embedded"
        layer.smart_data = {
            **(layer.smart_data or {}),
            "linked": keep_linked,
            "source_path": str(Path(path).resolve()),
            "original_size": [w, h],
            "fingerprint": file_fingerprint(path),
        }
        render_smart_object(layer)
        layer.touch_pixels()
        self.dirty = True
        return True

    def reset_active_smart_transform(self) -> bool:
        layer = self.layer
        if layer.kind not in {"linked", "embedded"} or layer.smart_source is None:
            return False
        center_x = layer.x + layer.pixels.shape[1] / 2.0
        center_y = layer.y + layer.pixels.shape[0] / 2.0
        h, w = layer.smart_source.shape[:2]
        layer.smart_data = {
            **(layer.smart_data or {}),
            "transform": {"width": w, "height": h, "angle": 0.0, "flip_horizontal": False, "flip_vertical": False},
        }
        render_smart_object(layer)
        layer.x = round(center_x - w / 2.0)
        layer.y = round(center_y - h / 2.0)
        layer.touch_pixels()
        self.dirty = True
        return True

    def add_text_layer(
        self,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int, int],
        size: int,
        font_family: str = "arial.ttf",
        box_width: int = 0,
        align: str = "left",
        line_spacing: int | None = None,
        tracking: int = 0,
        bold: bool = False,
        italic: bool = False,
        underline: bool = False,
        path_mode: str = "none",
        path_amount: int = 0,
        path_points: list[tuple[float, float]] | list[list[float]] | None = None,
        path_start: float = 0.0,
        path_end: float = 1.0,
        path_side: int = 1,
        path_reverse: bool = False,
        baseline_shift: int = 0,
        rotation: float = 0.0,
    ) -> Layer:
        normalized_path_start, normalized_path_end = normalize_text_path_range(path_start, path_end)
        layer = Layer(
            name=f"Text: {text[:24]}",
            pixels=blank_rgba(self.width, self.height, (0, 0, 0, 0)),
            kind="text",
            text_data={
                "text": text,
                "x": int(x),
                "y": int(y),
                "color": list(color),
                "size": int(size),
                "font_family": font_family,
                "box_width": int(box_width),
                "align": align,
                "line_spacing": int(line_spacing if line_spacing is not None else max(2, int(size) // 5)),
                "tracking": int(tracking),
                "bold": bool(bold),
                "italic": bool(italic),
                "underline": bool(underline),
                "path_mode": path_mode if path_mode in {"none", "arc", "wave", "bezier"} else "none",
                "path_amount": int(path_amount),
                "path_points": normalize_text_path_points(path_points, int(x), int(y) + int(size), max(int(box_width), int(size) * 8)),
                "path_start": normalized_path_start,
                "path_end": normalized_path_end,
                "path_side": -1 if int(path_side) < 0 else 1,
                "path_reverse": bool(path_reverse),
                "baseline_shift": int(baseline_shift),
                "rotation": float(rotation),
            },
        )
        render_text_layer(layer)
        self.layers.append(layer)
        self.active_layer = len(self.layers) - 1
        self.dirty = True
        return layer

    def add_shape_layer(
        self,
        shape: str,
        box: tuple[int, int, int, int],
        fill: tuple[int, int, int, int],
        stroke: tuple[int, int, int, int] | None = None,
        stroke_width: int = 0,
        sides: int = 5,
        inner_ratio: float = 0.5,
        control_points: list[tuple[float, float]] | None = None,
        custom_points: list[tuple[float, float]] | None = None,
        gradient: dict[str, Any] | None = None,
        texture: dict[str, Any] | None = None,
    ) -> Layer:
        shape_box = normalized_box(box)
        if shape == "bezier" and control_points is None:
            x1, y1, x2, y2 = shape_box
            control_points = [(x1, y2), (x1, y1), (x2, y1), (x2, y2)]
        layer = Layer(
            name=f"{shape.title()} shape",
            pixels=blank_rgba(self.width, self.height, (0, 0, 0, 0)),
            kind="shape",
            shape_data={
                "shape": shape,
                "box": [int(v) for v in shape_box],
                "fill": list(fill),
                "stroke": None if stroke is None else list(stroke),
                "stroke_width": int(stroke_width),
                "sides": int(sides),
                "inner_ratio": float(inner_ratio),
                "control_points": None if control_points is None else [[float(x), float(y)] for x, y in control_points],
                "custom_points": None if custom_points is None else [[float(x), float(y)] for x, y in custom_points],
                "gradient": None if gradient is None else json.loads(json.dumps(gradient)),
                "texture": None if texture is None else json.loads(json.dumps(texture)),
            },
        )
        render_shape_layer(layer)
        self.layers.append(layer)
        self.active_layer = len(self.layers) - 1
        self.dirty = True
        return layer

    def edit_shape_layer(
        self,
        shape: str | None = None,
        fill: tuple[int, int, int, int] | None = None,
        stroke: tuple[int, int, int, int] | None = None,
        stroke_width: int | None = None,
        sides: int | None = None,
        inner_ratio: float | None = None,
        control_points: list[tuple[float, float]] | None = None,
        custom_points: list[tuple[float, float]] | None = None,
    ) -> None:
        layer = self.layer
        if layer.locked or layer.kind != "shape" or layer.shape_data is None:
            return
        if shape is not None:
            layer.shape_data["shape"] = shape
            layer.name = f"{shape.title()} shape"
        if fill is not None:
            layer.shape_data["fill"] = list(fill)
        if stroke is not None:
            layer.shape_data["stroke"] = list(stroke)
        if stroke_width is not None:
            layer.shape_data["stroke_width"] = int(stroke_width)
        if sides is not None:
            layer.shape_data["sides"] = max(3, int(sides))
        if inner_ratio is not None:
            layer.shape_data["inner_ratio"] = float(np.clip(inner_ratio, 0.05, 0.95))
        if control_points is not None:
            layer.shape_data["control_points"] = [[float(x), float(y)] for x, y in control_points]
        if custom_points is not None:
            layer.shape_data["custom_points"] = [[float(x), float(y)] for x, y in custom_points]
        render_shape_layer(layer)
        self.dirty = True

    def boolean_shape_data_with_lower(self, mode: str) -> dict[str, Any] | None:
        if self.active_layer <= 0:
            return None
        upper = self.layer
        lower = self.layers[self.active_layer - 1]
        if upper.locked or lower.locked or upper.kind != "shape" or lower.kind != "shape" or upper.shape_data is None or lower.shape_data is None:
            return None
        mode = str(mode).lower().strip()
        if mode not in {"union", "subtract", "intersect", "xor"}:
            return None
        fill = upper.shape_data.get("fill", lower.shape_data.get("fill", [255, 255, 255, 255]))
        stroke = upper.shape_data.get("stroke")
        stroke_width = int(upper.shape_data.get("stroke_width", 0))
        lower_data = translated_shape_data(lower.shape_data, lower.x, lower.y)
        upper_data = translated_shape_data(upper.shape_data, upper.x, upper.y)
        lower_data.setdefault("_name", lower.name)
        upper_data.setdefault("_name", upper.name)
        lower_data.setdefault("_enabled", True)
        upper_data.setdefault("_enabled", True)
        result = {
            "shape": "boolean",
            "boolean_mode": mode,
            "children": [lower_data, upper_data],
            "fill": list(fill),
            "stroke": None if stroke is None else list(stroke),
            "stroke_width": max(0, stroke_width),
        }
        result["box"] = list(shape_data_bounds(result) or (0, 0, 1, 1))
        return result

    def boolean_active_shape_with_lower(self, mode: str, shape_data: dict[str, Any] | None = None) -> bool:
        data = self.boolean_shape_data_with_lower(mode) if shape_data is None else json.loads(json.dumps(shape_data))
        if data is None or self.active_layer <= 0:
            return False
        upper = self.layer
        lower = self.layers[self.active_layer - 1]
        if upper.locked or lower.locked or upper.kind != "shape" or lower.kind != "shape":
            return False
        data["shape"] = "boolean"
        data["boolean_mode"] = str(data.get("boolean_mode", mode)).lower()
        data["box"] = list(shape_data_bounds(data) or (0, 0, 1, 1))
        combined = Layer(
            name=f"Булева фигура: {data['boolean_mode']}",
            pixels=blank_rgba(self.width, self.height, (0, 0, 0, 0)),
            kind="shape",
            shape_data=data,
        )
        render_shape_layer(combined)
        self.layers[self.active_layer - 1] = combined
        del self.layers[self.active_layer]
        self.active_layer -= 1
        self.dirty = True
        return True

    def edit_text_layer(
        self,
        text: str | None = None,
        size: int | None = None,
        color: tuple[int, int, int, int] | None = None,
        font_family: str | None = None,
        box_width: int | None = None,
        align: str | None = None,
        line_spacing: int | None = None,
        tracking: int | None = None,
        bold: bool | None = None,
        italic: bool | None = None,
        underline: bool | None = None,
        path_mode: str | None = None,
        path_amount: int | None = None,
        path_points: list[tuple[float, float]] | list[list[float]] | None = None,
        path_start: float | None = None,
        path_end: float | None = None,
        path_side: int | None = None,
        path_reverse: bool | None = None,
        baseline_shift: int | None = None,
        rotation: float | None = None,
    ) -> None:
        layer = self.layer
        if layer.locked or layer.kind != "text" or layer.text_data is None:
            return
        if text is not None:
            layer.text_data["text"] = text
            layer.name = f"Text: {text[:24]}"
        if size is not None:
            layer.text_data["size"] = int(size)
        if color is not None:
            layer.text_data["color"] = list(color)
        if font_family is not None:
            layer.text_data["font_family"] = font_family
        if box_width is not None:
            layer.text_data["box_width"] = max(0, int(box_width))
        if align is not None:
            layer.text_data["align"] = align if align in {"left", "center", "right"} else "left"
        if line_spacing is not None:
            layer.text_data["line_spacing"] = max(0, int(line_spacing))
        if tracking is not None:
            layer.text_data["tracking"] = int(tracking)
        if bold is not None:
            layer.text_data["bold"] = bool(bold)
        if italic is not None:
            layer.text_data["italic"] = bool(italic)
        if underline is not None:
            layer.text_data["underline"] = bool(underline)
        if path_mode is not None:
            layer.text_data["path_mode"] = path_mode if path_mode in {"none", "arc", "wave", "bezier"} else "none"
        if path_amount is not None:
            layer.text_data["path_amount"] = int(path_amount)
        if path_points is not None:
            layer.text_data["path_points"] = normalize_text_path_points(
                path_points,
                int(layer.text_data.get("x", 0)),
                int(layer.text_data.get("y", 0)) + int(layer.text_data.get("size", 48)),
                max(int(layer.text_data.get("box_width", 0) or 0), int(layer.text_data.get("size", 48)) * 8),
            )
        if path_start is not None or path_end is not None:
            normalized_start, normalized_end = normalize_text_path_range(
                float(layer.text_data.get("path_start", 0.0)) if path_start is None else path_start,
                float(layer.text_data.get("path_end", 1.0)) if path_end is None else path_end,
            )
            layer.text_data["path_start"] = normalized_start
            layer.text_data["path_end"] = normalized_end
        if path_side is not None:
            layer.text_data["path_side"] = -1 if int(path_side) < 0 else 1
        if path_reverse is not None:
            layer.text_data["path_reverse"] = bool(path_reverse)
        if baseline_shift is not None:
            layer.text_data["baseline_shift"] = int(baseline_shift)
        if rotation is not None:
            layer.text_data["rotation"] = float(rotation)
        render_text_layer(layer)
        self.dirty = True

    def transform_active_text_box(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        angle: float = 0.0,
        flip_horizontal: bool = False,
        flip_vertical: bool = False,
    ) -> bool:
        layer = self.layer
        if layer.locked or layer.kind != "text" or layer.text_data is None or not np.any(layer.pixels[:, :, 3]):
            return False
        ys, xs = np.where(layer.pixels[:, :, 3] > 0)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
        old_width, old_height = max(1, x2 - x1), max(1, y2 - y1)
        scale_x = max(1, int(width)) / old_width
        scale_y = max(1, int(height)) / old_height
        data = layer.text_data
        data["x"] = int(data.get("x", 0)) + int(x) - x1
        data["y"] = int(data.get("y", 0)) + int(y) - y1
        data["size"] = max(4, round(int(data.get("size", 48)) * scale_y))
        if int(data.get("box_width", 0) or 0) > 0:
            data["box_width"] = max(1, round(int(data["box_width"]) * scale_x))
        data["line_spacing"] = max(0, round(int(data.get("line_spacing", 0)) * scale_y))
        data["tracking"] = round(int(data.get("tracking", 0)) * scale_x)
        raw_path = data.get("path_points")
        if isinstance(raw_path, list) and len(raw_path) == 4:
            data["path_points"] = [
                [
                    int(x) + (float(point[0]) - x1) * scale_x,
                    int(y) + (float(point[1]) - y1) * scale_y,
                ]
                for point in raw_path
                if isinstance(point, (list, tuple)) and len(point) >= 2
            ]
        data["rotation"] = float(data.get("rotation", 0.0)) + float(angle)
        if flip_horizontal:
            data["flip_horizontal"] = not bool(data.get("flip_horizontal", False))
        if flip_vertical:
            data["flip_vertical"] = not bool(data.get("flip_vertical", False))
        render_text_layer(layer)
        self.dirty = True
        return True

    def add_adjustment_layer(self, name: str, adjustment: dict[str, Any]) -> None:
        layer = Layer(
            name=name,
            pixels=blank_rgba(self.width, self.height, (0, 0, 0, 0)),
            kind="adjustment",
            adjustment=dict(adjustment),
        )
        self.layers.append(layer)
        self.active_layer = len(self.layers) - 1
        self.dirty = True

    def delete_active_layer(self) -> None:
        if len(self.layers) <= 1:
            return
        del self.layers[self.active_layer]
        self.active_layer = max(0, self.active_layer - 1)
        self.dirty = True

    def toggle_active_clipping(self) -> None:
        if self.active_layer <= 0:
            return
        self.layer.clipping = not self.layer.clipping
        self.dirty = True

    def set_active_layer_effects(self, effects: dict[str, Any]) -> None:
        self.layer.effects = json.loads(json.dumps(effects))
        self.dirty = True

    def set_active_layer_filters(self, filters: list[dict[str, Any]]) -> None:
        self.layer.filters = json.loads(json.dumps(filters))
        self.dirty = True

    def clear_active_layer_filters(self) -> None:
        self.layer.filters = []
        self.dirty = True

    def duplicate_active_layer(self) -> None:
        self.layers.insert(self.active_layer + 1, self.layer.clone())
        self.active_layer += 1
        self.dirty = True

    def move_active_layer(self, dx: int, dy: int) -> None:
        layer = self.layer
        if layer.locked:
            return
        dx, dy = int(dx), int(dy)
        if dx == 0 and dy == 0:
            return
        layer.x += dx
        layer.y += dy
        if layer.mask is not None and not layer.mask_linked:
            h, w = layer.mask.shape[:2]
            layer.mask = shifted_mask(layer.mask, w, h, w, h, -dx, -dy)
        self.dirty = True

    def merge_down(self) -> None:
        if self.active_layer <= 0:
            return
        lower = self.layers[self.active_layer - 1]
        upper = self.layer
        min_x = min(lower.x, upper.x)
        min_y = min(lower.y, upper.y)
        max_x = max(lower.x + lower.pixels.shape[1], upper.x + upper.pixels.shape[1])
        max_y = max(lower.y + lower.pixels.shape[0], upper.y + upper.pixels.shape[0])
        merged = blank_rgba(max_x - min_x, max_y - min_y, (0, 0, 0, 0))
        alpha_blend_inplace(merged, lower.pixels, lower.x - min_x, lower.y - min_y, lower.opacity, lower.mask if lower.mask_enabled else None, lower.mask_density, lower.blend_mode)
        alpha_blend_inplace(merged, upper.pixels, upper.x - min_x, upper.y - min_y, upper.opacity, upper.mask if upper.mask_enabled else None, upper.mask_density, upper.blend_mode)
        lower.pixels = merged
        lower.x = min_x
        lower.y = min_y
        lower.opacity = 1.0
        lower.mask = None
        del self.layers[self.active_layer]
        self.active_layer -= 1
        self.dirty = True

    def flatten(self) -> None:
        self.layers = [Layer("Flattened", self.composite(False))]
        self.active_layer = 0
        self.dirty = True

    def resize_image(self, width: int, height: int, interpolation=cv2.INTER_CUBIC) -> None:
        for layer in self.layers:
            new_w = max(1, round(layer.pixels.shape[1] * width / self.width))
            new_h = max(1, round(layer.pixels.shape[0] * height / self.height))
            if layer.kind in {"linked", "embedded"} and layer.smart_source is not None:
                data = dict(layer.smart_data or {})
                transform = dict(data.get("transform") or {})
                transform["width"] = max(1, round(int(transform.get("width", layer.smart_source.shape[1])) * width / self.width))
                transform["height"] = max(1, round(int(transform.get("height", layer.smart_source.shape[0])) * height / self.height))
                data["transform"] = transform
                layer.smart_data = data
                render_smart_object(layer)
            else:
                layer.pixels = cv2.resize(layer.pixels, (new_w, new_h), interpolation=interpolation)
            if layer.mask is not None:
                layer.mask = cv2.resize(layer.mask, (layer.pixels.shape[1], layer.pixels.shape[0]), interpolation=cv2.INTER_NEAREST)
            layer.x = round(layer.x * width / self.width)
            layer.y = round(layer.y * height / self.height)
            layer.touch_pixels()
        self.width, self.height = width, height
        if self.selection_mask is not None:
            self.selection_mask = cv2.resize(self.selection_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        for name, mask in list(self.saved_selections.items()):
            self.saved_selections[name] = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        self.dirty = True

    def resize_canvas(self, width: int, height: int, anchor="center") -> None:
        dx = (width - self.width) // 2 if anchor == "center" else 0
        dy = (height - self.height) // 2 if anchor == "center" else 0
        for layer in self.layers:
            layer.x += dx
            layer.y += dy
        self.width, self.height = width, height
        if self.selection_mask is not None:
            new_mask = np.zeros((height, width), dtype=np.uint8)
            x1, y1 = max(0, dx), max(0, dy)
            x2, y2 = min(width, dx + self.selection_mask.shape[1]), min(height, dy + self.selection_mask.shape[0])
            if x1 < x2 and y1 < y2:
                sx1, sy1 = x1 - dx, y1 - dy
                new_mask[y1:y2, x1:x2] = self.selection_mask[sy1 : sy1 + (y2 - y1), sx1 : sx1 + (x2 - x1)]
            self.selection_mask = new_mask
        for name, mask in list(self.saved_selections.items()):
            new_mask = np.zeros((height, width), dtype=np.uint8)
            x1, y1 = max(0, dx), max(0, dy)
            x2, y2 = min(width, dx + mask.shape[1]), min(height, dy + mask.shape[0])
            if x1 < x2 and y1 < y2:
                sx1, sy1 = x1 - dx, y1 - dy
                new_mask[y1:y2, x1:x2] = mask[sy1 : sy1 + (y2 - y1), sx1 : sx1 + (x2 - x1)]
            self.saved_selections[name] = new_mask
        self.dirty = True

    def generative_expand(self, left: int, top: int, right: int, bottom: int, method: str = "content-aware") -> Layer:
        left, top, right, bottom = (max(0, int(value)) for value in (left, top, right, bottom))
        if left + top + right + bottom == 0:
            raise ValueError("At least one expansion margin must be greater than zero")
        original = self.composite(False)
        expanded = generative_expand_pixels(original, left, top, right, bottom, method)
        generated = expanded.copy()
        generated[top : top + self.height, left : left + self.width] = 0
        for layer in self.layers:
            layer.x += left
            layer.y += top
        generated_layer = Layer("Генеративное расширение", generated)
        self.layers.insert(0, generated_layer)
        self.active_layer += 1
        self.width += left + right
        self.height += top + bottom
        if self.selection_mask is not None:
            selection = np.zeros((self.height, self.width), dtype=np.uint8)
            selection[top : top + self.selection_mask.shape[0], left : left + self.selection_mask.shape[1]] = self.selection_mask
            self.selection_mask = selection
        self.saved_selections = {
            name: _expanded_mask(mask, left, top, self.width, self.height) for name, mask in self.saved_selections.items()
        }
        self.metadata["last_generative_expand"] = {
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "method": method,
        }
        self.dirty = True
        return generated_layer

    def crop(self, box: tuple[int, int, int, int]) -> None:
        x1, y1, x2, y2 = normalized_box(box)
        x1, x2 = max(0, min(self.width, x1)), max(0, min(self.width, x2))
        y1, y2 = max(0, min(self.height, y1)), max(0, min(self.height, y2))
        if x1 == x2 or y1 == y2:
            return
        new_w, new_h = max(1, x2 - x1), max(1, y2 - y1)
        for layer in self.layers:
            canvas = blank_rgba(self.width, self.height, (0, 0, 0, 0))
            alpha_blend_inplace(canvas, layer.pixels, layer.x, layer.y, 1.0)
            if layer.mask is not None:
                mask_canvas = np.zeros((self.height, self.width), dtype=np.uint8)
                paste_mask(mask_canvas, layer.mask, layer.x, layer.y)
                layer.mask = mask_canvas[y1:y2, x1:x2].copy()
            layer.pixels = canvas[y1:y2, x1:x2].copy()
            layer.x = 0
            layer.y = 0
        self.width, self.height = new_w, new_h
        if self.selection_mask is not None:
            self.selection_mask = self.selection_mask[y1:y2, x1:x2].copy()
        for name, mask in list(self.saved_selections.items()):
            self.saved_selections[name] = mask[y1:y2, x1:x2].copy()
        self.dirty = True

    def trim_transparent(self) -> None:
        alpha = self.composite(False)[:, :, 3]
        if not np.any(alpha):
            return
        ys, xs = np.where(alpha > 0)
        self.crop((int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)))

    def reveal_all(self) -> None:
        min_x, min_y = 0, 0
        max_x, max_y = self.width, self.height
        for layer in self.layers:
            h, w = layer.pixels.shape[:2]
            min_x = min(min_x, layer.x)
            min_y = min(min_y, layer.y)
            max_x = max(max_x, layer.x + w)
            max_y = max(max_y, layer.y + h)
        if min_x == 0 and min_y == 0 and max_x == self.width and max_y == self.height:
            return
        dx, dy = -min_x, -min_y
        old_w, old_h = self.width, self.height
        new_w, new_h = max(1, max_x - min_x), max(1, max_y - min_y)
        for layer in self.layers:
            layer.x += dx
            layer.y += dy
        if self.selection_mask is not None:
            self.selection_mask = shifted_mask(self.selection_mask, old_w, old_h, new_w, new_h, dx, dy)
        for name, mask in list(self.saved_selections.items()):
            self.saved_selections[name] = shifted_mask(mask, old_w, old_h, new_w, new_h, dx, dy)
        self.width, self.height = new_w, new_h
        self.dirty = True

    def set_rect_selection(self, box: tuple[int, int, int, int], mode: str = "replace", feather: int = 0) -> None:
        x1, y1, x2, y2 = normalized_box(box)
        x1, x2 = max(0, min(self.width, x1)), max(0, min(self.width, x2))
        y1, y2 = max(0, min(self.height, y1)), max(0, min(self.height, y2))
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        if x1 < x2 and y1 < y2:
            mask[y1:y2, x1:x2] = 255
        if feather > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), max(0.1, float(feather)))
        self.apply_selection_mask(mask, mode)

    def set_ellipse_selection(self, box: tuple[int, int, int, int], mode: str = "replace", feather: int = 0, antialias: bool = True) -> None:
        x1, y1, x2, y2 = normalized_box(box)
        x1, x2 = max(0, min(self.width, x1)), max(0, min(self.width, x2))
        y1, y2 = max(0, min(self.height, y1)), max(0, min(self.height, y2))
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        if x1 < x2 and y1 < y2:
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1, lineType=cv2.LINE_AA if antialias else cv2.LINE_8)
        if feather > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), max(0.1, float(feather)))
        self.apply_selection_mask(mask, mode)

    def set_polygon_selection(self, points: list[tuple[int, int]], mode: str = "replace", feather: int = 0, antialias: bool = True) -> None:
        if len(points) < 3:
            return
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        pts = np.array([[(max(0, min(self.width - 1, int(x))), max(0, min(self.height - 1, int(y)))) for x, y in points]], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255, lineType=cv2.LINE_AA if antialias else cv2.LINE_8)
        if feather > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), max(0.1, float(feather)))
        self.apply_selection_mask(mask, mode)

    def set_single_row_selection(self, y: int, mode: str = "replace") -> None:
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        if 0 <= y < self.height:
            mask[y : y + 1, :] = 255
        self.apply_selection_mask(mask, mode)

    def set_single_column_selection(self, x: int, mode: str = "replace") -> None:
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        if 0 <= x < self.width:
            mask[:, x : x + 1] = 255
        self.apply_selection_mask(mask, mode)

    def magic_wand_selection(self, layer: Layer, x: int, y: int, tolerance: int, mode: str = "replace", contiguous: bool = True) -> None:
        lx, ly = int(x) - layer.x, int(y) - layer.y
        if lx < 0 or ly < 0 or lx >= layer.pixels.shape[1] or ly >= layer.pixels.shape[0]:
            return
        seed = layer.pixels[ly, lx].astype(np.int16)
        diff = np.abs(layer.pixels.astype(np.int16) - seed).max(axis=2)
        candidates = (diff <= int(tolerance)).astype(np.uint8)
        if contiguous:
            _, labels, _, _ = cv2.connectedComponentsWithStats(candidates, 4)
            local = (labels == labels[ly, lx]).astype(np.uint8) * 255
        else:
            local = candidates * 255
        self.apply_selection_mask(self._layer_mask_to_document(layer, local), mode)

    def color_range_selection(self, layer: Layer, x: int, y: int, tolerance: int, mode: str = "replace") -> None:
        lx, ly = int(x) - layer.x, int(y) - layer.y
        if lx < 0 or ly < 0 or lx >= layer.pixels.shape[1] or ly >= layer.pixels.shape[0]:
            return
        tolerance = max(0, int(tolerance))
        seed_rgba = layer.pixels[ly, lx]
        if tolerance == 0:
            local = np.all(layer.pixels == seed_rgba, axis=2).astype(np.uint8) * 255
        else:
            rgb = layer.pixels[:, :, :3].astype(np.float32) / 255.0
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            seed_lab = lab[ly, lx]
            color_distance = np.linalg.norm(lab - seed_lab, axis=2)
            alpha_distance = np.abs(layer.pixels[:, :, 3].astype(np.float32) - float(seed_rgba[3])) / 255.0 * 100.0
            perceptual_distance = np.sqrt(color_distance * color_distance + alpha_distance * alpha_distance)
            local = (perceptual_distance <= tolerance * 0.8).astype(np.uint8) * 255
        self.apply_selection_mask(self._layer_mask_to_document(layer, local), mode)

    def _quick_selection_mask(
        self,
        layer: Layer,
        points: list[tuple[int, int]],
        radius: int,
        tolerance: int,
        smooth: int = 0,
        edge_radius: int = 0,
        edge_strength: float = 0.0,
    ) -> np.ndarray:
        local_union = np.zeros(layer.pixels.shape[:2], dtype=np.uint8)
        radius = max(1, int(radius))
        tolerance = max(0, int(tolerance))
        for x, y in points:
            lx, ly = int(x) - layer.x, int(y) - layer.y
            if lx < 0 or ly < 0 or lx >= layer.pixels.shape[1] or ly >= layer.pixels.shape[0]:
                continue
            x1, y1 = max(0, lx - radius), max(0, ly - radius)
            x2, y2 = min(layer.pixels.shape[1], lx + radius + 1), min(layer.pixels.shape[0], ly + radius + 1)
            sample = layer.pixels[y1:y2, x1:x2]
            opaque = sample[:, :, 3] > 0
            if np.any(opaque):
                seed = sample[:, :, :3][opaque].mean(axis=0)
            else:
                seed = layer.pixels[ly, lx, :3].astype(np.float32)
            gate_radius = radius * 3
            gx1, gy1 = max(0, lx - gate_radius), max(0, ly - gate_radius)
            gx2 = min(layer.pixels.shape[1], lx + gate_radius + 1)
            gy2 = min(layer.pixels.shape[0], ly + gate_radius + 1)
            region = layer.pixels[gy1:gy2, gx1:gx2]
            diff = np.abs(region[:, :, :3].astype(np.float32) - seed).max(axis=2)
            candidates = ((diff <= tolerance) & (region[:, :, 3] > 0)).astype(np.uint8)
            _, labels, _, _ = cv2.connectedComponentsWithStats(candidates, 4)
            rx, ry = lx - gx1, ly - gy1
            label = labels[ry, rx]
            if label == 0 and candidates[ry, rx] == 0:
                continue
            component = (labels == label).astype(np.uint8) * 255
            brush_gate = np.zeros_like(component)
            cv2.circle(brush_gate, (rx, ry), gate_radius, 255, -1)
            gated = np.where(brush_gate > 0, component, 0).astype(np.uint8)
            local_union[gy1:gy2, gx1:gx2] = np.maximum(local_union[gy1:gy2, gx1:gx2], gated)
        if np.any(local_union):
            local_union = refine_selection_mask(local_union, max(0, int(smooth)), 0, 1.0, 0)
            edge_radius = max(0, int(edge_radius))
            edge_strength = float(np.clip(edge_strength, 0.0, 1.0))
            if edge_radius > 0 and edge_strength > 0.0:
                local_union = correct_selection_edges(local_union, layer.pixels, edge_radius, edge_strength, 96)
        return self._layer_mask_to_document(layer, local_union)

    def preview_quick_selection_brush(
        self,
        layer: Layer,
        points: list[tuple[int, int]],
        radius: int,
        tolerance: int,
        mode: str = "replace",
        smooth: int = 0,
        edge_radius: int = 0,
        edge_strength: float = 0.0,
    ) -> np.ndarray | None:
        mask = self._quick_selection_mask(layer, points, radius, tolerance, smooth, edge_radius, edge_strength)
        if not np.any(mask):
            return None if self.selection_mask is None else self.selection_mask.copy()
        current = self.selection_mask
        if mode == "add" and current is not None:
            return np.maximum(current, mask)
        if mode == "subtract" and current is not None:
            result = np.clip(current.astype(np.float32) * (1.0 - mask.astype(np.float32) / 255.0), 0, 255).astype(np.uint8)
            return result if np.any(result) else None
        if mode == "intersect" and current is not None:
            result = np.minimum(current, mask)
            return result if np.any(result) else None
        return mask

    def quick_selection_brush(
        self,
        layer: Layer,
        points: list[tuple[int, int]],
        radius: int,
        tolerance: int,
        mode: str = "replace",
        smooth: int = 0,
        edge_radius: int = 0,
        edge_strength: float = 0.0,
    ) -> None:
        mask = self._quick_selection_mask(layer, points, radius, tolerance, smooth, edge_radius, edge_strength)
        if np.any(mask):
            self.apply_selection_mask(mask, mode)

    def magnetic_edge_map(self, composite: np.ndarray | None = None) -> np.ndarray:
        composite = self.composite(False) if composite is None else composite
        gray = cv2.cvtColor(composite[:, :, :3], cv2.COLOR_RGB2GRAY)
        gray = np.where(composite[:, :, 3] > 0, gray, 0).astype(np.uint8)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 140)
        grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        if float(magnitude.max()) > 0.0:
            magnitude = (magnitude / magnitude.max() * 255.0).astype(np.uint8)
        else:
            magnitude = np.zeros_like(edges)
        return np.maximum(edges, magnitude)

    def snap_point_to_edge(self, point: tuple[int, int], edge_map: np.ndarray, radius: int = 14) -> tuple[int, int]:
        x = max(0, min(self.width - 1, int(point[0])))
        y = max(0, min(self.height - 1, int(point[1])))
        radius = max(1, int(radius))
        x1, y1 = max(0, x - radius), max(0, y - radius)
        x2, y2 = min(self.width, x + radius + 1), min(self.height, y + radius + 1)
        local = edge_map[y1:y2, x1:x2]
        if local.size == 0 or int(local.max()) <= 0:
            return x, y
        ys, xs = np.where(local > 0)
        if len(xs) == 0:
            return x, y
        doc_x = xs + x1
        doc_y = ys + y1
        distance = np.sqrt((doc_x - x) ** 2 + (doc_y - y) ** 2)
        score = local[ys, xs].astype(np.float32) - distance.astype(np.float32) * 4.0
        best = int(np.argmax(score))
        return int(doc_x[best]), int(doc_y[best])

    def select_opaque_pixels(self, layer: Layer, mode: str = "replace") -> None:
        local = (layer.pixels[:, :, 3] > 0).astype(np.uint8) * 255
        self.apply_selection_mask(self._layer_mask_to_document(layer, local), mode)

    def select_subject(self, layer: Layer, mode: str = "replace") -> None:
        local = subject_selection_mask(layer.pixels)
        if np.any(local):
            self.apply_selection_mask(self._layer_mask_to_document(layer, local), mode)

    def select_background(self, layer: Layer, mode: str = "replace") -> None:
        local = background_selection_mask(layer.pixels)
        if np.any(local):
            self.apply_selection_mask(self._layer_mask_to_document(layer, local), mode)

    def select_sky(self, layer: Layer, mode: str = "replace") -> None:
        local = sky_selection_mask(layer.pixels)
        if np.any(local):
            self.apply_selection_mask(self._layer_mask_to_document(layer, local), mode)

    def save_selection(self, name: str) -> None:
        if self.selection_mask is None:
            return
        self.saved_selections[name] = self.selection_mask.copy()
        self.dirty = True

    def load_selection(self, name: str, mode: str = "replace") -> None:
        mask = self.saved_selections.get(name)
        if mask is not None:
            self.apply_selection_mask(mask.copy(), mode)

    def delete_saved_selection(self, name: str) -> None:
        if name in self.saved_selections:
            del self.saved_selections[name]
            self.dirty = True

    def _layer_mask_to_document(self, layer: Layer, local_mask: np.ndarray) -> np.ndarray:
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        paste_mask(mask, local_mask, layer.x, layer.y)
        return mask

    def apply_selection_mask(self, mask: np.ndarray, mode: str = "replace") -> None:
        mask = np.clip(mask, 0, 255).astype(np.uint8)
        if mode == "add" and self.selection_mask is not None:
            self.selection_mask = np.maximum(self.selection_mask, mask)
        elif mode == "subtract" and self.selection_mask is not None:
            self.selection_mask = np.clip(
                self.selection_mask.astype(np.float32) * (1.0 - mask.astype(np.float32) / 255.0),
                0,
                255,
            ).astype(np.uint8)
        elif mode == "intersect" and self.selection_mask is not None:
            self.selection_mask = np.minimum(self.selection_mask, mask)
        else:
            self.selection_mask = mask
        if self.selection_mask is not None and not np.any(self.selection_mask):
            self.selection_mask = None

    def clear_selection(self) -> None:
        self.selection_mask = None

    def select_all(self) -> None:
        self.selection_mask = np.full((self.height, self.width), 255, dtype=np.uint8)

    def invert_selection(self) -> None:
        if self.selection_mask is None:
            self.selection_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.selection_mask = 255 - self.selection_mask
        if not np.any(self.selection_mask):
            self.selection_mask = None

    def feather_selection(self, radius: int) -> None:
        if self.selection_mask is None:
            return
        radius = max(1, int(radius))
        k = radius * 2 + 1
        self.selection_mask = cv2.GaussianBlur(self.selection_mask, (k, k), radius)
        self.dirty = True

    def grow_selection(self, pixels: int) -> None:
        if self.selection_mask is None:
            return
        pixels = max(1, int(pixels))
        kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), dtype=np.uint8)
        self.selection_mask = cv2.dilate(self.selection_mask, kernel)
        self.dirty = True

    def shrink_selection(self, pixels: int) -> None:
        if self.selection_mask is None:
            return
        pixels = max(1, int(pixels))
        kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), dtype=np.uint8)
        self.selection_mask = cv2.erode(self.selection_mask, kernel)
        if not np.any(self.selection_mask):
            self.selection_mask = None
        self.dirty = True

    def smooth_selection(self, radius: int) -> None:
        if self.selection_mask is None:
            return
        radius = max(1, int(radius))
        k = radius * 2 + 1
        mask = cv2.GaussianBlur(self.selection_mask, (k, k), radius)
        self.selection_mask = np.where(mask >= 128, 255, 0).astype(np.uint8)
        if not np.any(self.selection_mask):
            self.selection_mask = None
        self.dirty = True

    def border_selection(self, width: int) -> None:
        if self.selection_mask is None:
            return
        width = max(1, int(width))
        kernel = np.ones((width * 2 + 1, width * 2 + 1), dtype=np.uint8)
        outer = cv2.dilate(self.selection_mask, kernel)
        inner = cv2.erode(self.selection_mask, kernel)
        border = np.clip(outer.astype(np.int16) - inner.astype(np.int16), 0, 255).astype(np.uint8)
        self.selection_mask = border if np.any(border) else None
        self.dirty = True

    def refine_selection(self, smooth: int = 0, feather: int = 0, contrast: float = 1.0, shift: int = 0) -> None:
        if self.selection_mask is None:
            return
        mask = refine_selection_mask(self.selection_mask, smooth, feather, contrast, shift)
        self.selection_mask = mask if np.any(mask) else None
        self.dirty = True

    def cleanup_selection_edges(self, radius: int = 3, strength: float = 0.7) -> None:
        if self.selection_mask is None:
            return
        mask = cleanup_selection_edges(self.selection_mask, self.composite(False), radius, strength)
        self.selection_mask = mask if np.any(mask) else None
        self.dirty = True

    def correct_selection_edges(self, radius: int = 3, strength: float = 0.65, threshold: int = 96) -> None:
        if self.selection_mask is None:
            return
        mask = correct_selection_edges(self.selection_mask, self.composite(False), radius, strength, threshold)
        self.selection_mask = mask if np.any(mask) else None
        self.dirty = True

    def selection_bounds(self) -> tuple[int, int, int, int] | None:
        if self.selection_mask is None or not np.any(self.selection_mask):
            return None
        ys, xs = np.where(self.selection_mask > 0)
        return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)

    def layer_selection_mask(self, layer: Layer) -> np.ndarray | None:
        if self.selection_mask is None:
            return None
        mask = np.zeros(layer.pixels.shape[:2], dtype=np.uint8)
        x1, y1 = max(0, layer.x), max(0, layer.y)
        x2 = min(self.width, layer.x + layer.pixels.shape[1])
        y2 = min(self.height, layer.y + layer.pixels.shape[0])
        if x1 >= x2 or y1 >= y2:
            return mask
        lx1, ly1 = x1 - layer.x, y1 - layer.y
        mask[ly1 : ly1 + (y2 - y1), lx1 : lx1 + (x2 - x1)] = self.selection_mask[y1:y2, x1:x2]
        return mask

    def add_reveal_all_mask(self) -> None:
        layer = self.layer
        layer.mask = np.full(layer.pixels.shape[:2], 255, dtype=np.uint8)
        layer.mask_enabled = True
        layer.mask_linked = True

    def add_hide_all_mask(self) -> None:
        layer = self.layer
        layer.mask = np.zeros(layer.pixels.shape[:2], dtype=np.uint8)
        layer.mask_enabled = True
        layer.mask_linked = True

    def add_mask_from_selection(self) -> None:
        layer = self.layer
        mask = self.layer_selection_mask(layer)
        if mask is None:
            self.add_reveal_all_mask()
        else:
            layer.mask = mask
            layer.mask_enabled = True
            layer.mask_linked = True

    def invert_active_mask(self) -> None:
        layer = self.layer
        if layer.mask is not None:
            layer.mask = 255 - layer.mask

    def toggle_active_mask(self) -> None:
        layer = self.layer
        if layer.mask is not None:
            layer.mask_enabled = not layer.mask_enabled

    def toggle_active_mask_link(self) -> None:
        layer = self.layer
        if layer.mask is not None:
            layer.mask_linked = not layer.mask_linked
            self.dirty = True

    def set_active_mask_density(self, density: float) -> None:
        layer = self.layer
        if layer.mask is not None:
            layer.mask_density = float(np.clip(density, 0.0, 1.0))
            self.dirty = True

    def set_active_mask_feather(self, radius: float) -> None:
        layer = self.layer
        if layer.mask is not None:
            layer.mask_feather = max(0.0, float(radius))
            self.dirty = True

    def preview_active_mask_refinement(
        self,
        smooth: int = 0,
        feather: int = 0,
        contrast: float = 1.0,
        shift: int = 0,
        edge_radius: int = 0,
        edge_strength: float = 0.0,
        confidence_threshold: int = 96,
    ) -> np.ndarray | None:
        layer = self.layer
        if layer.mask is None:
            return None
        return refine_layer_mask(
            layer.mask,
            layer.pixels,
            smooth,
            feather,
            contrast,
            shift,
            edge_radius,
            edge_strength,
            confidence_threshold,
        )

    def refine_active_mask(
        self,
        smooth: int = 0,
        feather: int = 0,
        contrast: float = 1.0,
        shift: int = 0,
        edge_radius: int = 0,
        edge_strength: float = 0.0,
        confidence_threshold: int = 96,
    ) -> None:
        mask = self.preview_active_mask_refinement(
            smooth,
            feather,
            contrast,
            shift,
            edge_radius,
            edge_strength,
            confidence_threshold,
        )
        if mask is not None:
            self.layer.mask = mask
            self.dirty = True

    def delete_active_mask(self) -> None:
        self.layer.mask = None
        self.layer.mask_linked = True

    def apply_active_mask(self) -> None:
        layer = self.layer
        if layer.mask is None:
            return
        mask = effective_layer_mask(layer)
        density = float(np.clip(layer.mask_density, 0.0, 1.0))
        alpha = ((1.0 - density) + (mask.astype(np.float32) / 255.0) * density).clip(0, 1)
        layer.pixels[:, :, 3] = np.clip(layer.pixels[:, :, 3].astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        layer.mask = None
        layer.mask_feather = 0.0

    def patch_active_selection(self, source_x: int, source_y: int, heal: bool = True) -> None:
        layer = self.layer
        if layer.locked:
            return
        selection = self.layer_selection_mask(layer)
        if selection is None or not np.any(selection):
            return
        ys, xs = np.where(selection > 0)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
        w, h = x2 - x1, y2 - y1
        sx1 = int(source_x) - layer.x
        sy1 = int(source_y) - layer.y
        sx2, sy2 = sx1 + w, sy1 + h
        if sx1 < 0 or sy1 < 0 or sx2 > layer.pixels.shape[1] or sy2 > layer.pixels.shape[0]:
            return
        src = layer.pixels[sy1:sy2, sx1:sx2].astype(np.float32)
        dst = layer.pixels[y1:y2, x1:x2].astype(np.float32)
        mask = selection[y1:y2, x1:x2].astype(np.float32) / 255.0
        edited = src.copy()
        active = mask > 0
        if heal and np.any(active):
            src_mean = src[active, :3].mean(axis=0)
            dst_mean = dst[active, :3].mean(axis=0)
            edited[:, :, :3] = np.clip(src[:, :, :3] - src_mean + dst_mean, 0, 255)
        mixed = edited * mask[:, :, None] + dst * (1.0 - mask[:, :, None])
        layer.pixels[y1:y2, x1:x2] = np.clip(mixed, 0, 255).astype(np.uint8)
        self.dirty = True

    def transform_active_layer(
        self,
        x: int | None = None,
        y: int | None = None,
        width: int | None = None,
        height: int | None = None,
        angle: float = 0.0,
        flip_horizontal: bool = False,
        flip_vertical: bool = False,
    ) -> None:
        layer = self.layer
        if layer.locked:
            return
        if x is not None:
            layer.x = int(x)
        if y is not None:
            layer.y = int(y)
        if layer.kind in {"linked", "embedded"}:
            if layer.smart_source is None:
                layer.smart_source = layer.pixels.copy()
            data = dict(layer.smart_data or {})
            transform = dict(data.get("transform") or {})
            current_w = int(transform.get("width", layer.pixels.shape[1]))
            current_h = int(transform.get("height", layer.pixels.shape[0]))
            center_x = layer.x + layer.pixels.shape[1] / 2.0
            center_y = layer.y + layer.pixels.shape[0] / 2.0
            transform.update(
                {
                    "width": max(1, int(width if width is not None else current_w)),
                    "height": max(1, int(height if height is not None else current_h)),
                    "angle": float(transform.get("angle", 0.0)) + float(angle),
                    "flip_horizontal": bool(transform.get("flip_horizontal", False)) ^ bool(flip_horizontal),
                    "flip_vertical": bool(transform.get("flip_vertical", False)) ^ bool(flip_vertical),
                }
            )
            data["transform"] = transform
            layer.smart_data = data
            render_smart_object(layer)
            if x is None:
                layer.x = round(center_x - layer.pixels.shape[1] / 2.0)
            if y is None:
                layer.y = round(center_y - layer.pixels.shape[0] / 2.0)
            if layer.mask is not None and layer.mask.shape != layer.pixels.shape[:2]:
                layer.mask = cv2.resize(layer.mask, (layer.pixels.shape[1], layer.pixels.shape[0]), interpolation=cv2.INTER_NEAREST)
            layer.touch_pixels()
            self.dirty = True
            return
        target_w = max(1, int(width or layer.pixels.shape[1]))
        target_h = max(1, int(height or layer.pixels.shape[0]))
        if (target_w, target_h) != (layer.pixels.shape[1], layer.pixels.shape[0]):
            layer.pixels = cv2.resize(layer.pixels, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            if layer.mask is not None:
                layer.mask = cv2.resize(layer.mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        if flip_horizontal:
            layer.pixels = cv2.flip(layer.pixels, 1)
            if layer.mask is not None:
                layer.mask = cv2.flip(layer.mask, 1)
        if flip_vertical:
            layer.pixels = cv2.flip(layer.pixels, 0)
            if layer.mask is not None:
                layer.mask = cv2.flip(layer.mask, 0)
        if abs(float(angle)) > 0.001:
            center_x = layer.x + layer.pixels.shape[1] / 2.0
            center_y = layer.y + layer.pixels.shape[0] / 2.0
            layer.pixels = rotate_bound(layer.pixels, float(angle), cv2.INTER_CUBIC)
            if layer.mask is not None:
                layer.mask = rotate_bound(layer.mask, float(angle), cv2.INTER_LINEAR)
            layer.x = round(center_x - layer.pixels.shape[1] / 2.0)
            layer.y = round(center_y - layer.pixels.shape[0] / 2.0)
        self.dirty = True

    def transform_selected_pixels(
        self,
        x: int | None = None,
        y: int | None = None,
        width: int | None = None,
        height: int | None = None,
        angle: float = 0.0,
        flip_horizontal: bool = False,
        flip_vertical: bool = False,
    ) -> bool:
        layer = self.layer
        if layer.locked:
            return False
        selection = self.layer_selection_mask(layer)
        if selection is None or not np.any(selection):
            return False
        ys, xs = np.where(selection > 0)
        lx1, ly1, lx2, ly2 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
        patch = layer.pixels[ly1:ly2, lx1:lx2].copy()
        patch_mask = selection[ly1:ly2, lx1:lx2].astype(np.float32) / 255.0
        patch[:, :, 3] = np.clip(patch[:, :, 3].astype(np.float32) * patch_mask, 0, 255).astype(np.uint8)

        original_region = layer.pixels[ly1:ly2, lx1:lx2]
        keep_alpha = 1.0 - patch_mask
        original_region[:, :, 3] = np.clip(original_region[:, :, 3].astype(np.float32) * keep_alpha, 0, 255).astype(np.uint8)

        dest_x = int(x if x is not None else layer.x + lx1)
        dest_y = int(y if y is not None else layer.y + ly1)
        target_w = max(1, int(width or patch.shape[1]))
        target_h = max(1, int(height or patch.shape[0]))
        if (target_w, target_h) != (patch.shape[1], patch.shape[0]):
            patch = cv2.resize(patch, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        if flip_horizontal:
            patch = cv2.flip(patch, 1)
        if flip_vertical:
            patch = cv2.flip(patch, 0)
        if abs(float(angle)) > 0.001:
            center_x = dest_x + patch.shape[1] / 2.0
            center_y = dest_y + patch.shape[0] / 2.0
            patch = rotate_bound(patch, float(angle), cv2.INTER_CUBIC)
            dest_x = round(center_x - patch.shape[1] / 2.0)
            dest_y = round(center_y - patch.shape[0] / 2.0)

        old_x, old_y = layer.x, layer.y
        old_h, old_w = layer.pixels.shape[:2]
        new_x = min(old_x, dest_x)
        new_y = min(old_y, dest_y)
        new_right = max(old_x + old_w, dest_x + patch.shape[1])
        new_bottom = max(old_y + old_h, dest_y + patch.shape[0])
        new_pixels = blank_rgba(new_right - new_x, new_bottom - new_y, (0, 0, 0, 0))
        new_pixels[old_y - new_y : old_y - new_y + old_h, old_x - new_x : old_x - new_x + old_w] = layer.pixels
        alpha_blend_inplace(new_pixels, patch, dest_x - new_x, dest_y - new_y, 1.0)

        if layer.mask is not None:
            new_mask = np.zeros(new_pixels.shape[:2], dtype=np.uint8)
            paste_mask(new_mask, layer.mask, old_x - new_x, old_y - new_y)
            layer.mask = new_mask

        new_selection = np.zeros((self.height, self.width), dtype=np.uint8)
        transformed_alpha = np.where(patch[:, :, 3] > 0, 255, 0).astype(np.uint8)
        paste_mask(new_selection, transformed_alpha, dest_x, dest_y)
        self.selection_mask = new_selection if np.any(new_selection) else None
        layer.pixels = new_pixels
        layer.x = new_x
        layer.y = new_y
        self.dirty = True
        return True

    def perspective_transform_active_layer(self, corners: list[tuple[float, float]] | tuple[tuple[float, float], ...]) -> None:
        layer = self.layer
        if layer.locked:
            return
        if len(corners) != 4:
            raise ValueError("Perspective transform needs four destination corners.")
        h, w = layer.pixels.shape[:2]
        dst_doc = np.array(corners, dtype=np.float32)
        min_x = math.floor(float(dst_doc[:, 0].min()))
        min_y = math.floor(float(dst_doc[:, 1].min()))
        max_x = math.ceil(float(dst_doc[:, 0].max()))
        max_y = math.ceil(float(dst_doc[:, 1].max()))
        out_w = max(1, max_x - min_x)
        out_h = max(1, max_y - min_y)
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        dst = dst_doc - np.array([min_x, min_y], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)
        layer.pixels = cv2.warpPerspective(
            layer.pixels,
            matrix,
            (out_w, out_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        if layer.mask is not None:
            layer.mask = cv2.warpPerspective(
                layer.mask,
                matrix,
                (out_w, out_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        layer.x = min_x
        layer.y = min_y
        self.dirty = True

    def warp_active_layer(self, mode: str, amount: float = 0.35, wavelength: float = 96.0) -> None:
        layer = self.layer
        if layer.locked:
            return
        layer.pixels = warp_pixels(layer.pixels, mode, amount, wavelength, cv2.INTER_CUBIC)
        if layer.mask is not None:
            layer.mask = warp_pixels(layer.mask, mode, amount, wavelength, cv2.INTER_LINEAR)
        self.dirty = True


def normalized_box(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def shape_box_from_drag(
    start: tuple[int, int],
    end: tuple[int, int],
    shape: str,
    *,
    keep_proportions: bool = False,
    from_center: bool = False,
) -> tuple[int, int, int, int]:
    sx, sy = int(start[0]), int(start[1])
    ex, ey = int(end[0]), int(end[1])
    dx, dy = ex - sx, ey - sy
    if shape == "line" and keep_proportions and (dx or dy):
        length = math.hypot(dx, dy)
        angle = round(math.atan2(dy, dx) / (math.pi / 4.0)) * (math.pi / 4.0)
        dx, dy = round(math.cos(angle) * length), round(math.sin(angle) * length)
    elif shape != "line" and keep_proportions:
        size = max(abs(dx), abs(dy))
        dx = size if dx >= 0 else -size
        dy = size if dy >= 0 else -size
    if from_center:
        return sx - dx, sy - dy, sx + dx, sy + dy
    return sx, sy, sx + dx, sy + dy


def shape_geometry_from_drag(
    tool: str,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    shift: bool = False,
    alt: bool = False,
    sides: int = 5,
    inner_ratio: float = 0.5,
    custom_points: Any = None,
) -> dict[str, Any]:
    shape_by_tool = {
        "rect_shape": "rectangle",
        "ellipse_shape": "ellipse",
        "line_shape": "line",
        "bezier_shape": "bezier",
        "polygon_shape": "polygon",
        "star_shape": "star",
        "custom_shape": "custom",
    }
    shape = shape_by_tool.get(tool, tool.removesuffix("_shape"))
    raw_box = shape_box_from_drag(start, end, shape, keep_proportions=shift, from_center=alt)
    box = normalized_box(raw_box)
    geometry: dict[str, Any] = {"shape": shape, "box": box}
    if shape == "line":
        geometry["line"] = raw_box
    elif shape == "bezier":
        geometry["points"] = bezier_curve_points(None, box, 64)
    elif shape == "polygon":
        geometry["points"] = regular_polygon_points(box, max(3, int(sides)))
    elif shape == "star":
        geometry["points"] = star_points(box, max(3, int(sides)), inner_ratio)
    elif shape == "custom":
        geometry["points"] = custom_shape_points(custom_points, box)
    return geometry


def shape_drag_is_meaningful(geometry: dict[str, Any], minimum: int = 3) -> bool:
    x1, y1, x2, y2 = geometry["box"]
    if geometry.get("shape") in {"line", "bezier"}:
        return math.hypot(x2 - x1, y2 - y1) >= minimum
    return x2 - x1 >= minimum and y2 - y1 >= minimum


def layer_contains_point(layer: Layer, point: tuple[int, int], tolerance: int = 5) -> bool:
    """Hit-test the rendered object alpha, including thin lines and Bezier curves."""
    if not layer.visible or layer.kind == "adjustment" or layer.pixels.size == 0:
        return False
    local_x = int(point[0]) - int(layer.x)
    local_y = int(point[1]) - int(layer.y)
    height, width = layer.pixels.shape[:2]
    radius = max(0, int(tolerance))
    x1, x2 = max(0, local_x - radius), min(width, local_x + radius + 1)
    y1, y2 = max(0, local_y - radius), min(height, local_y + radius + 1)
    if x1 >= x2 or y1 >= y2:
        return False
    alpha = layer.pixels[y1:y2, x1:x2, 3]
    if layer.kind == "shape" and layer.shape_data is not None:
        kind = str(layer.shape_data.get("shape", "rectangle")).lower()
        if kind not in {"line", "bezier"} and 0 <= local_x < width and 0 <= local_y < height:
            return bool(layer.pixels[local_y, local_x, 3] > 8)
    return bool(np.any(alpha > 8))


def topmost_layer_at(
    document: Document,
    point: tuple[int, int],
    *,
    kinds: tuple[str, ...] = ("shape", "text"),
    tolerance: int = 5,
) -> int | None:
    for index in range(len(document.layers) - 1, -1, -1):
        layer = document.layers[index]
        if layer.kind in kinds and layer_contains_point(layer, point, tolerance):
            return index
    return None


def resize_box_from_handle(
    box: tuple[int, int, int, int],
    handle: str,
    point: tuple[int, int],
    *,
    keep_proportions: bool = False,
    from_center: bool = False,
    minimum: int = 2,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = normalized_box(box)
    px, py = int(point[0]), int(point[1])
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    original_ratio = max(1e-6, (x2 - x1) / max(1, y2 - y1))
    if "w" in handle:
        x1 = px
        if from_center:
            x2 = round(cx + (cx - px))
    if "e" in handle:
        x2 = px
        if from_center:
            x1 = round(cx - (px - cx))
    if "n" in handle:
        y1 = py
        if from_center:
            y2 = round(cy + (cy - py))
    if "s" in handle:
        y2 = py
        if from_center:
            y1 = round(cy - (py - cy))
    x1, y1, x2, y2 = normalized_box((x1, y1, x2, y2))
    if keep_proportions and len(handle) == 2:
        width, height = max(minimum, x2 - x1), max(minimum, y2 - y1)
        if width / height > original_ratio:
            height = max(minimum, round(width / original_ratio))
        else:
            width = max(minimum, round(height * original_ratio))
        if "w" in handle:
            x1 = x2 - width
        else:
            x2 = x1 + width
        if "n" in handle:
            y1 = y2 - height
        else:
            y2 = y1 + height
        if from_center:
            x1, x2 = round(cx - width / 2), round(cx + width / 2)
            y1, y2 = round(cy - height / 2), round(cy + height / 2)
    if x2 - x1 < minimum:
        x2 = x1 + minimum
    if y2 - y1 < minimum:
        y2 = y1 + minimum
    return int(x1), int(y1), int(x2), int(y2)


def selection_contour_points(mask: np.ndarray | None, threshold: int = 128) -> list[np.ndarray]:
    if mask is None or mask.ndim != 2 or not np.any(mask >= threshold):
        return []
    binary = (mask >= threshold).astype(np.uint8)
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return [contour[:, 0, :].copy() for contour in contours if len(contour) >= 2]


def union_rect(a: tuple[int, int, int, int] | None, b: tuple[int, int, int, int] | None) -> tuple[int, int, int, int] | None:
    if a is None:
        return b
    if b is None:
        return a
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def checker_background(width: int, height: int, tile: int = 16) -> np.ndarray:
    key = (width, height, tile)
    cached = _checker_cache.get(key)
    if cached is not None:
        return cached
    tile_arr = np.zeros((tile * 2, tile * 2, 4), dtype=np.uint8)
    tile_arr[:, :, 3] = 255
    tile_arr[:tile, :tile, :3] = 224
    tile_arr[tile:, tile:, :3] = 224
    tile_arr[:tile, tile:, :3] = 192
    tile_arr[tile:, :tile, :3] = 192
    reps_y = height // tile_arr.shape[0] + 1
    reps_x = width // tile_arr.shape[1] + 1
    bg = np.tile(tile_arr, (reps_y, reps_x, 1))[:height, :width].copy()
    if len(_checker_cache) > 8:
        _checker_cache.clear()
    _checker_cache[key] = bg
    return bg


def paste_mask(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    h, w = src.shape[:2]
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(dst.shape[1], int(x) + w), min(dst.shape[0], int(y) + h)
    if x1 >= x2 or y1 >= y2:
        return
    sx1, sy1 = x1 - x, y1 - y
    dst[y1:y2, x1:x2] = src[sy1 : sy1 + (y2 - y1), sx1 : sx1 + (x2 - x1)]


def shifted_mask(mask: np.ndarray, old_width: int, old_height: int, new_width: int, new_height: int, dx: int, dy: int) -> np.ndarray:
    out = np.zeros((new_height, new_width), dtype=np.uint8)
    x1, y1 = max(0, dx), max(0, dy)
    x2, y2 = min(new_width, dx + old_width), min(new_height, dy + old_height)
    if x1 >= x2 or y1 >= y2:
        return out
    sx1, sy1 = x1 - dx, y1 - dy
    out[y1:y2, x1:x2] = mask[sy1 : sy1 + (y2 - y1), sx1 : sx1 + (x2 - x1)]
    return out


def refine_selection_mask(mask: np.ndarray, smooth: int = 0, feather: int = 0, contrast: float = 1.0, shift: int = 0) -> np.ndarray:
    out = mask.copy()
    if shift > 0:
        kernel = np.ones((int(shift) * 2 + 1, int(shift) * 2 + 1), dtype=np.uint8)
        out = cv2.dilate(out, kernel)
    elif shift < 0:
        amount = abs(int(shift))
        kernel = np.ones((amount * 2 + 1, amount * 2 + 1), dtype=np.uint8)
        out = cv2.erode(out, kernel)
    if smooth > 0:
        radius = int(smooth)
        k = radius * 2 + 1
        out = cv2.GaussianBlur(out, (k, k), radius)
        out = np.where(out >= 128, 255, 0).astype(np.uint8)
    if feather > 0:
        radius = int(feather)
        k = radius * 2 + 1
        out = cv2.GaussianBlur(out, (k, k), radius)
    if abs(float(contrast) - 1.0) > 0.001:
        work = out.astype(np.float32)
        work = (work - 127.5) * max(0.0, float(contrast)) + 127.5
        out = np.clip(work, 0, 255).astype(np.uint8)
    return out


def refine_layer_mask(
    mask: np.ndarray,
    image: np.ndarray,
    smooth: int = 0,
    feather: int = 0,
    contrast: float = 1.0,
    shift: int = 0,
    edge_radius: int = 0,
    edge_strength: float = 0.0,
    confidence_threshold: int = 96,
) -> np.ndarray:
    out = refine_selection_mask(mask, max(0, int(smooth)), 0, 1.0, int(shift))
    radius = max(0, int(edge_radius))
    strength = float(np.clip(edge_strength, 0.0, 1.0))
    if radius > 0 and strength > 0.0 and np.any(out):
        out = correct_selection_edges(out, image, radius, strength, confidence_threshold)
    return refine_selection_mask(out, 0, max(0, int(feather)), max(0.0, float(contrast)), 0)


def cleanup_selection_edges(mask: np.ndarray, image: np.ndarray, radius: int = 3, strength: float = 0.7) -> np.ndarray:
    if mask is None:
        return np.zeros((0, 0), dtype=np.uint8)
    if not np.any(mask):
        return np.zeros_like(mask, dtype=np.uint8)
    radius = max(1, int(radius))
    strength = float(np.clip(strength, 0.0, 1.0))
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    soft = cv2.GaussianBlur(cleaned, (radius * 2 + 1, radius * 2 + 1), radius).astype(np.float32)

    gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
    gray = np.where(image[:, :, 3] > 0, gray, 0).astype(np.uint8)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 120)
    edge_band = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)))
    boundary = cv2.subtract(cv2.dilate(binary, kernel), cv2.erode(binary, kernel))
    preserve = (edge_band.astype(np.float32) / 255.0) * (boundary.astype(np.float32) / 255.0) * strength

    mixed = soft * (1.0 - preserve) + binary.astype(np.float32) * preserve
    mixed = (mixed - 127.5) * (1.0 + strength * 1.5) + 127.5
    return np.where(np.clip(mixed, 0, 255) >= 128, 255, 0).astype(np.uint8)


def selection_edge_confidence(mask: np.ndarray, image: np.ndarray, radius: int = 3) -> np.ndarray:
    if mask is None or mask.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    if not np.any(mask):
        return np.zeros_like(mask, dtype=np.uint8)
    radius = max(1, int(radius))
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    boundary = cv2.subtract(cv2.dilate(binary, kernel), cv2.erode(binary, kernel))
    if not np.any(boundary):
        return np.zeros_like(binary, dtype=np.uint8)

    gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
    if image.shape[2] > 3:
        gray = np.where(image[:, :, 3] > 0, gray, 0).astype(np.uint8)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 40, 130)
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    if float(magnitude.max()) > 0.0:
        magnitude = magnitude / float(magnitude.max())
    edge_band = cv2.dilate(edges, kernel).astype(np.float32) / 255.0
    confidence = np.maximum(edge_band, magnitude)
    confidence = cv2.GaussianBlur(confidence, (radius * 2 + 1, radius * 2 + 1), radius)
    return np.where(boundary > 0, np.clip(confidence * 255.0, 0, 255), 0).astype(np.uint8)


def correct_selection_edges(mask: np.ndarray, image: np.ndarray, radius: int = 3, strength: float = 0.65, threshold: int = 96) -> np.ndarray:
    if mask is None or mask.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    if not np.any(mask):
        return np.zeros_like(mask, dtype=np.uint8)
    radius = max(1, int(radius))
    strength = float(np.clip(strength, 0.0, 1.0))
    threshold = int(np.clip(threshold, 0, 255))
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    confidence = selection_edge_confidence(binary, image, radius)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    outer = cv2.dilate(binary, kernel)
    inner = cv2.erode(binary, kernel)
    boundary = cv2.subtract(outer, inner)
    if not np.any(boundary):
        return binary

    trusted = (confidence >= threshold) & (boundary > 0)
    weak = (confidence < threshold) & (boundary > 0)
    edge_locked = np.where(trusted, binary, 0).astype(np.uint8)
    candidate = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
    relaxed = cv2.GaussianBlur(candidate, (radius * 2 + 1, radius * 2 + 1), radius)
    relaxed = np.where(relaxed >= 128, 255, 0).astype(np.uint8)
    mixed = np.where(weak, relaxed, binary).astype(np.uint8)
    sharpened = np.where(edge_locked > 0, 255, mixed).astype(np.uint8)
    blend = binary.astype(np.float32) * (1.0 - strength) + sharpened.astype(np.float32) * strength
    return np.where(blend >= 128, 255, 0).astype(np.uint8)


def subject_selection_mask(pixels: np.ndarray) -> np.ndarray:
    if pixels.size == 0:
        return np.zeros(pixels.shape[:2], dtype=np.uint8)
    alpha = pixels[:, :, 3]
    visible = alpha > 0
    if not np.any(visible):
        return np.zeros(alpha.shape, dtype=np.uint8)
    coverage = float(np.count_nonzero(visible)) / float(visible.size)
    if coverage < 0.92:
        seed = visible.astype(np.uint8) * 255
    else:
        rgb = pixels[:, :, :3].astype(np.uint8)
        border = np.zeros(alpha.shape, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        border &= visible
        if np.count_nonzero(border) < 8:
            border = visible
        background = np.median(rgb[border].astype(np.float32), axis=0)
        diff = np.linalg.norm(rgb.astype(np.float32) - background, axis=2)
        diff = np.where(visible, diff, 0.0).astype(np.float32)
        diff_u8 = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        threshold, _ = cv2.threshold(diff_u8[visible], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        seed = np.where((diff_u8 >= max(12, int(threshold))) & visible, 255, 0).astype(np.uint8)

    h, w = seed.shape
    radius = max(1, min(h, w) // 80)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel)
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, kernel)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(seed, 8)
    if num <= 1:
        return seed

    yy, xx = np.indices(seed.shape)
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    diagonal = max(1.0, float(np.hypot(w, h)))
    best_label = 0
    best_score = -1.0
    for label in range(1, num):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < 4:
            continue
        component = labels == label
        component_cx = float(xx[component].mean())
        component_cy = float(yy[component].mean())
        center_bonus = 1.0 - min(1.0, float(np.hypot(component_cx - cx, component_cy - cy)) / diagonal)
        score = area * (0.65 + center_bonus)
        if score > best_score:
            best_score = score
            best_label = label
    if best_label == 0:
        return np.zeros_like(seed)
    subject = np.where(labels == best_label, 255, 0).astype(np.uint8)
    feather_radius = max(1, min(h, w) // 160)
    soft = cv2.GaussianBlur(subject, (feather_radius * 2 + 1, feather_radius * 2 + 1), feather_radius)
    return np.where(soft >= 96, 255, 0).astype(np.uint8)


def background_selection_mask(pixels: np.ndarray) -> np.ndarray:
    if pixels.size == 0:
        return np.zeros(pixels.shape[:2], dtype=np.uint8)
    alpha = pixels[:, :, 3]
    visible = alpha > 0
    if not np.any(visible):
        return np.full(alpha.shape, 255, dtype=np.uint8)
    coverage = float(np.count_nonzero(visible)) / float(visible.size)
    if coverage < 0.92:
        return border_connected_mask(~visible)

    rgb = pixels[:, :, :3].astype(np.uint8)
    border = np.zeros(alpha.shape, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    border &= visible
    if np.count_nonzero(border) < 8:
        border = visible
    background = np.median(rgb[border].astype(np.float32), axis=0)
    diff = np.linalg.norm(rgb.astype(np.float32) - background, axis=2)
    border_diff = diff[border]
    tolerance = max(18.0, float(np.percentile(border_diff, 75)) + 14.0)
    candidates = (diff <= tolerance) & visible
    mask = border_connected_mask(candidates)
    radius = max(1, min(mask.shape) // 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel).astype(np.uint8)


def sky_selection_mask(pixels: np.ndarray) -> np.ndarray:
    if pixels.size == 0:
        return np.zeros(pixels.shape[:2], dtype=np.uint8)
    alpha = pixels[:, :, 3]
    visible = alpha > 0
    if not np.any(visible):
        return np.zeros(alpha.shape, dtype=np.uint8)
    rgb = pixels[:, :, :3].astype(np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    h, w = alpha.shape
    yy = np.arange(h)[:, None]
    upper_weight = yy < max(1, int(h * 0.72))
    blue_sky = (hue >= 85) & (hue <= 132) & (saturation >= 24) & (value >= 70)
    pale_sky = (saturation < 55) & (value >= 172) & (rgb[:, :, 2] >= rgb[:, :, 0] - 8)
    candidates = (blue_sky | pale_sky) & upper_weight & visible
    connected = top_connected_mask(candidates)
    if not np.any(connected):
        return np.zeros(alpha.shape, dtype=np.uint8)
    radius = max(1, min(h, w) // 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, kernel)
    connected = cv2.morphologyEx(connected, cv2.MORPH_OPEN, kernel)
    return connected.astype(np.uint8)


def border_connected_mask(candidates: np.ndarray) -> np.ndarray:
    binary = candidates.astype(np.uint8)
    if binary.size == 0 or not np.any(binary):
        return np.zeros(binary.shape, dtype=np.uint8)
    num, labels, _, _ = cv2.connectedComponentsWithStats(binary, 8)
    border_labels = set(np.unique(labels[0, binary[0, :] > 0]).tolist())
    border_labels.update(np.unique(labels[-1, binary[-1, :] > 0]).tolist())
    border_labels.update(np.unique(labels[binary[:, 0] > 0, 0]).tolist())
    border_labels.update(np.unique(labels[binary[:, -1] > 0, -1]).tolist())
    border_labels.discard(0)
    if num <= 1 or not border_labels:
        return np.zeros(binary.shape, dtype=np.uint8)
    return np.where(np.isin(labels, list(border_labels)), 255, 0).astype(np.uint8)


def top_connected_mask(candidates: np.ndarray) -> np.ndarray:
    binary = candidates.astype(np.uint8)
    if binary.size == 0 or not np.any(binary):
        return np.zeros(binary.shape, dtype=np.uint8)
    num, labels, _, _ = cv2.connectedComponentsWithStats(binary, 8)
    top_labels = set(np.unique(labels[0, binary[0, :] > 0]).tolist())
    top_labels.discard(0)
    if num <= 1 or not top_labels:
        return np.zeros(binary.shape, dtype=np.uint8)
    return np.where(np.isin(labels, list(top_labels)), 255, 0).astype(np.uint8)


def effective_layer_mask(layer: Layer) -> np.ndarray | None:
    if layer.mask is None:
        return None
    mask = layer.mask
    radius = float(getattr(layer, "mask_feather", 0.0))
    if radius <= 0:
        return mask
    k = max(3, int(round(radius)) * 2 + 1)
    return cv2.GaussianBlur(mask, (k, k), radius).astype(np.uint8)


def layer_alpha_canvas(document: Document, layer: Layer, pixels: np.ndarray | None = None) -> np.ndarray:
    canvas = np.zeros((document.height, document.width), dtype=np.uint8)
    source = pixels if pixels is not None else render_layer_pixels(layer)
    alpha = source[:, :, 3].copy()
    if layer.mask is not None and layer.mask_enabled:
        mask = effective_layer_mask(layer)
        mask_alpha = ((1.0 - float(layer.mask_density)) + (mask.astype(np.float32) / 255.0) * float(layer.mask_density)).clip(0, 1)
        alpha = np.clip(alpha.astype(np.float32) * mask_alpha, 0, 255).astype(np.uint8)
    paste_mask(canvas, alpha, layer.x, layer.y)
    return canvas


def document_alpha_to_layer_mask(alpha_canvas: np.ndarray, layer: Layer) -> np.ndarray:
    mask = np.zeros(layer.pixels.shape[:2], dtype=np.uint8)
    x1, y1 = max(0, layer.x), max(0, layer.y)
    x2 = min(alpha_canvas.shape[1], layer.x + layer.pixels.shape[1])
    y2 = min(alpha_canvas.shape[0], layer.y + layer.pixels.shape[0])
    if x1 >= x2 or y1 >= y2:
        return mask
    lx1, ly1 = x1 - layer.x, y1 - layer.y
    mask[ly1 : ly1 + (y2 - y1), lx1 : lx1 + (x2 - x1)] = alpha_canvas[y1:y2, x1:x2]
    return mask


def render_layer_pixels(layer: Layer) -> np.ndarray:
    if not layer.filters:
        return layer.pixels
    return apply_filter_stack(layer.pixels, layer.filters)


def render_layer_effects(layer: Layer, pixels: np.ndarray | None = None) -> list[tuple[np.ndarray, int, int, float, str]]:
    effects: list[tuple[np.ndarray, int, int, float, str]] = []
    if not layer.effects:
        return effects
    source = pixels if pixels is not None else render_layer_pixels(layer)
    alpha = source[:, :, 3]
    if not np.any(alpha):
        return effects
    shadow = layer.effects.get("drop_shadow")
    if shadow and shadow.get("enabled", True):
        blur_radius = max(0, int(shadow.get("blur", 12)))
        offset_x = int(shadow.get("x", 10))
        offset_y = int(shadow.get("y", 10))
        opacity = float(shadow.get("opacity", 0.55))
        color = tuple(int(v) for v in shadow.get("color", [0, 0, 0, 255]))
        mask = effect_mask(alpha, blur_radius)
        effects.append((solid_from_alpha(mask, color), layer.x + offset_x, layer.y + offset_y, opacity, "Normal"))
    glow = layer.effects.get("outer_glow")
    if glow and glow.get("enabled", True):
        blur_radius = max(1, int(glow.get("blur", 18)))
        opacity = float(glow.get("opacity", 0.5))
        color = tuple(int(v) for v in glow.get("color", [255, 220, 80, 255]))
        mask = effect_mask(alpha, blur_radius)
        effects.append((solid_from_alpha(mask, color), layer.x, layer.y, opacity, "Screen"))
    stroke = layer.effects.get("stroke")
    if stroke and stroke.get("enabled", True):
        size = max(1, int(stroke.get("size", 4)))
        opacity = float(stroke.get("opacity", 1.0))
        color = tuple(int(v) for v in stroke.get("color", [255, 255, 255, 255]))
        kernel = np.ones((size * 2 + 1, size * 2 + 1), dtype=np.uint8)
        expanded = cv2.dilate((alpha > 0).astype(np.uint8) * 255, kernel)
        outline = np.where((expanded > 0) & (alpha == 0), 255, 0).astype(np.uint8)
        effects.append((solid_from_alpha(outline, color), layer.x, layer.y, opacity, "Normal"))
    return effects


def effect_mask(alpha: np.ndarray, blur_radius: int) -> np.ndarray:
    mask = alpha.copy()
    if blur_radius > 0:
        kernel = np.ones((max(1, blur_radius // 2) * 2 + 1, max(1, blur_radius // 2) * 2 + 1), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel)
        k = blur_radius * 2 + 1
        mask = cv2.GaussianBlur(mask, (k, k), blur_radius)
    return mask


def solid_from_alpha(alpha: np.ndarray, color: tuple[int, int, int, int]) -> np.ndarray:
    arr = np.zeros((alpha.shape[0], alpha.shape[1], 4), dtype=np.uint8)
    arr[:, :, :3] = color[:3]
    arr[:, :, 3] = np.clip(alpha.astype(np.float32) * (color[3] / 255.0), 0, 255).astype(np.uint8)
    return arr


def rotate_bound(arr: np.ndarray, angle: float, interpolation: int) -> np.ndarray:
    h, w = arr.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = max(1, int(h * sin + w * cos))
    new_h = max(1, int(h * cos + w * sin))
    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]
    border = 0 if arr.ndim == 2 else (0, 0, 0, 0)
    return cv2.warpAffine(arr, matrix, (new_w, new_h), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def render_smart_object(layer: Layer) -> np.ndarray:
    """Render a smart layer from its immutable source and stored transform."""
    if layer.smart_source is None:
        return layer.pixels
    source = layer.smart_source
    transform = (layer.smart_data or {}).get("transform") or {}
    target_w = max(1, int(transform.get("width", source.shape[1])))
    target_h = max(1, int(transform.get("height", source.shape[0])))
    shrinking = target_w < source.shape[1] or target_h < source.shape[0]
    rendered = cv2.resize(source, (target_w, target_h), interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC)
    if bool(transform.get("flip_horizontal", False)):
        rendered = cv2.flip(rendered, 1)
    if bool(transform.get("flip_vertical", False)):
        rendered = cv2.flip(rendered, 0)
    angle = float(transform.get("angle", 0.0)) % 360.0
    if abs(angle) > 0.001:
        rendered = rotate_bound(rendered, angle, cv2.INTER_CUBIC)
    layer.pixels = np.ascontiguousarray(rendered)
    return layer.pixels


def warp_pixels(arr: np.ndarray, mode: str, amount: float = 0.35, wavelength: float = 96.0, interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
    h, w = arr.shape[:2]
    if h <= 1 or w <= 1:
        return arr.copy()
    amount = float(np.clip(amount, -2.0, 2.0))
    wavelength = max(4.0, float(wavelength))
    yy, xx = np.indices((h, w), dtype=np.float32)
    src_x = xx.copy()
    src_y = yy.copy()
    mode = str(mode).lower().strip()
    if mode == "arc":
        nx = (xx / max(1.0, w - 1.0)) * 2.0 - 1.0
        src_y = yy - amount * h * 0.25 * (1.0 - nx * nx)
    elif mode == "arc_vertical":
        ny = (yy / max(1.0, h - 1.0)) * 2.0 - 1.0
        src_x = xx - amount * w * 0.25 * (1.0 - ny * ny)
    elif mode in {"bulge", "pinch"}:
        cx = (w - 1.0) / 2.0
        cy = (h - 1.0) / 2.0
        radius = max(1.0, min(w, h) / 2.0)
        dx = (xx - cx) / radius
        dy = (yy - cy) / radius
        r = np.sqrt(dx * dx + dy * dy)
        influence = np.clip(1.0 - r, 0.0, 1.0) ** 2
        direction = -1.0 if mode == "bulge" else 1.0
        factor = np.clip(1.0 + direction * amount * 0.75 * influence, 0.05, 4.0)
        src_x = cx + (xx - cx) * factor
        src_y = cy + (yy - cy) * factor
    elif mode == "wave_x":
        src_x = xx - np.sin(yy / wavelength * math.tau) * amount * w * 0.08
    elif mode == "wave_y":
        src_y = yy - np.sin(xx / wavelength * math.tau) * amount * h * 0.08
    else:
        return arr.copy()
    border = 0 if arr.ndim == 2 else (0, 0, 0, 0)
    return cv2.remap(
        arr,
        src_x.astype(np.float32),
        src_y.astype(np.float32),
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def alpha_blend(dst: np.ndarray, src: np.ndarray, x: int, y: int, opacity: float) -> np.ndarray:
    out = dst.copy()
    alpha_blend_inplace(out, src, x, y, opacity)
    return out


def blend_rgb(src: np.ndarray, dst: np.ndarray, mode: str) -> np.ndarray:
    mode = mode if mode in BLEND_MODES else "Normal"
    s = src.astype(np.float32)
    d = dst.astype(np.float32)
    if mode == "Multiply":
        return s * d / 255.0
    if mode == "Screen":
        return 255.0 - (255.0 - s) * (255.0 - d) / 255.0
    if mode == "Overlay":
        return np.where(d <= 127.5, 2.0 * s * d / 255.0, 255.0 - 2.0 * (255.0 - s) * (255.0 - d) / 255.0)
    if mode == "Soft Light":
        sn = s / 255.0
        dn = d / 255.0
        out = (1.0 - 2.0 * sn) * dn * dn + 2.0 * sn * dn
        return out * 255.0
    if mode == "Linear Light":
        return np.clip(d + 2.0 * s - 255.0, 0.0, 255.0)
    if mode == "Darken":
        return np.minimum(s, d)
    if mode == "Lighten":
        return np.maximum(s, d)
    if mode == "Difference":
        return np.abs(d - s)
    if mode in {"Color", "Luminosity"}:
        src_hls = cv2.cvtColor(np.clip(src, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HLS)
        dst_hls = cv2.cvtColor(np.clip(dst, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HLS)
        out_hls = dst_hls.copy()
        if mode == "Color":
            out_hls[:, :, 0] = src_hls[:, :, 0]
            out_hls[:, :, 2] = src_hls[:, :, 2]
        else:
            out_hls[:, :, 1] = src_hls[:, :, 1]
        return cv2.cvtColor(out_hls, cv2.COLOR_HLS2RGB).astype(np.float32)
    return s


def alpha_blend_inplace(dst: np.ndarray, src: np.ndarray, x: int, y: int, opacity: float, alpha_mask: np.ndarray | None = None, mask_density: float = 1.0, blend_mode: str = "Normal") -> None:
    h, w = src.shape[:2]
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(dst.shape[1], int(x) + w), min(dst.shape[0], int(y) + h)
    if x1 >= x2 or y1 >= y2:
        return
    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
    s = src[sy1:sy2, sx1:sx2].astype(np.float32)
    target = dst[y1:y2, x1:x2]
    d = target.astype(np.float32)
    sa = (s[:, :, 3:4] / 255.0) * float(opacity)
    if alpha_mask is not None:
        ma = alpha_mask[sy1:sy2, sx1:sx2].astype(np.float32) / 255.0
        sa *= (1.0 - float(mask_density)) + ma[:, :, None] * float(mask_density)
    da = d[:, :, 3:4] / 255.0
    oa = sa + da * (1.0 - sa)
    blended = blend_rgb(s[:, :, :3], d[:, :, :3], blend_mode)
    rgb = np.where(oa > 0, (blended * sa + d[:, :, :3] * da * (1.0 - sa)) / np.maximum(oa, 1e-6), 0)
    target[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    target[:, :, 3] = np.clip(oa[:, :, 0] * 255, 0, 255).astype(np.uint8)


def brush_mask(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    cached = _brush_mask_cache.get(radius)
    if cached is not None:
        return cached
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = (xx * xx + yy * yy) <= radius * radius
    if len(_brush_mask_cache) > 64:
        _brush_mask_cache.clear()
    _brush_mask_cache[radius] = mask
    return mask


def retouch_falloff_mask(radius: int, hardness: float = 0.5) -> np.ndarray:
    radius = max(1, int(radius))
    hardness = float(hardness)
    if hardness > 1.0:
        hardness /= 100.0
    hardness = float(np.clip(hardness, 0.0, 1.0))
    key = radius, int(round(hardness * 100))
    cached = _retouch_mask_cache.get(key)
    if cached is not None:
        return cached
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    distance = np.sqrt(xx * xx + yy * yy).astype(np.float32) / float(radius)
    solid_radius = hardness * 0.96
    falloff = np.clip((1.0 - distance) / max(0.04, 1.0 - solid_radius), 0.0, 1.0)
    if len(_retouch_mask_cache) > 128:
        _retouch_mask_cache.clear()
    _retouch_mask_cache[key] = falloff
    return falloff


def retouch_effect_halo(mode: str, strength: float) -> int:
    if mode == "blur":
        sigma = 0.65 + 1.85 * float(np.clip(strength, 0.0, 1.0))
        return max(3, int(math.ceil(sigma * 3.0)))
    if mode == "sharpen":
        return 4
    return 0


def retouch_effect_rgb(source: np.ndarray, mode: str, strength: float, tonal_range: str = "midtones") -> np.ndarray:
    rgb = source[:, :, :3].astype(np.float32)
    strength = float(np.clip(strength, 0.0, 1.0))
    if mode == "blur":
        sigma = 0.65 + 1.85 * strength
        alpha = source[:, :, 3].astype(np.float32) / 255.0
        weight = cv2.GaussianBlur(alpha, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
        premultiplied = cv2.GaussianBlur(rgb * alpha[:, :, None], (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
        return np.where(weight[:, :, None] > 1e-4, premultiplied / np.maximum(weight[:, :, None], 1e-4), rgb)
    if mode == "sharpen":
        blurred = cv2.GaussianBlur(rgb, (0, 0), 1.0, borderType=cv2.BORDER_REFLECT_101)
        detail = rgb - blurred
        detail_luma = np.abs(detail[:, :, 0] * 0.2126 + detail[:, :, 1] * 0.7152 + detail[:, :, 2] * 0.0722)
        detail[detail_luma < 2.0] = 0.0
        return np.clip(rgb + detail * 0.85, 0.0, 255.0)
    if mode not in {"dodge", "burn"}:
        return rgb
    hls = cv2.cvtColor(source[:, :, :3], cv2.COLOR_RGB2HLS).astype(np.float32)
    luminance = hls[:, :, 1] / 255.0
    range_name = str(tonal_range).lower()
    if range_name in {"shadows", "тени"}:
        tonal_weight = np.clip(1.0 - luminance, 0.0, 1.0)
    elif range_name in {"highlights", "света"}:
        tonal_weight = np.clip(luminance, 0.0, 1.0)
    else:
        tonal_weight = np.clip(1.0 - np.abs(luminance - 0.5) * 2.0, 0.08, 1.0)
    if mode == "dodge":
        luminance = luminance + (1.0 - luminance) * 0.38 * tonal_weight
    else:
        luminance = luminance - luminance * 0.38 * tonal_weight
    hls[:, :, 1] = np.clip(luminance * 255.0, 0.0, 255.0)
    return cv2.cvtColor(hls.astype(np.uint8), cv2.COLOR_HLS2RGB).astype(np.float32)


class RetouchStroke:
    def __init__(
        self,
        layer: Layer,
        mode: str,
        radius: int,
        hardness: float,
        strength: float,
        tonal_range: str = "midtones",
        selection_mask: np.ndarray | None = None,
        tile_size: int = 128,
    ) -> None:
        self.layer = layer
        self.mode = mode
        self.radius = max(1, int(radius))
        self.hardness = float(np.clip(hardness, 0.0, 1.0))
        self.strength = float(np.clip(strength, 0.0, 1.0))
        self.tonal_range = tonal_range
        self.selection_mask = selection_mask
        self.tile_size = max(32, int(tile_size))
        self.before_tiles: dict[tuple[int, int], tuple[tuple[int, int, int, int], np.ndarray]] = {}
        self.coverage_tiles: dict[tuple[int, int], np.ndarray] = {}

    def tile_keys(self, rect: tuple[int, int, int, int]):
        x1, y1, x2, y2 = rect
        if x1 >= x2 or y1 >= y2:
            return
        for ty in range(y1 // self.tile_size, (y2 - 1) // self.tile_size + 1):
            for tx in range(x1 // self.tile_size, (x2 - 1) // self.tile_size + 1):
                yield tx, ty

    def capture_before(self, rect: tuple[int, int, int, int]) -> None:
        height, width = self.layer.pixels.shape[:2]
        for key in self.tile_keys(rect):
            if key in self.before_tiles:
                continue
            tx, ty = key
            x1, y1 = tx * self.tile_size, ty * self.tile_size
            x2, y2 = min(width, x1 + self.tile_size), min(height, y1 + self.tile_size)
            tile_rect = x1, y1, x2, y2
            self.before_tiles[key] = tile_rect, self.layer.pixels[y1:y2, x1:x2].copy()

    def original_region(self, rect: tuple[int, int, int, int]) -> np.ndarray:
        self.capture_before(rect)
        x1, y1, x2, y2 = rect
        result = self.layer.pixels[y1:y2, x1:x2].copy()
        for key in self.tile_keys(rect):
            tile_rect, before = self.before_tiles[key]
            tx1, ty1, tx2, ty2 = tile_rect
            ix1, iy1 = max(x1, tx1), max(y1, ty1)
            ix2, iy2 = min(x2, tx2), min(y2, ty2)
            result[iy1 - y1 : iy2 - y1, ix1 - x1 : ix2 - x1] = before[iy1 - ty1 : iy2 - ty1, ix1 - tx1 : ix2 - tx1]
        return result

    def merge_coverage(self, rect: tuple[int, int, int, int], dab: np.ndarray) -> None:
        x1, y1, x2, y2 = rect
        for key in self.tile_keys(rect):
            tile_rect, _before = self.before_tiles[key]
            tx1, ty1, tx2, ty2 = tile_rect
            coverage = self.coverage_tiles.get(key)
            if coverage is None:
                coverage = np.zeros((ty2 - ty1, tx2 - tx1), dtype=np.float32)
                self.coverage_tiles[key] = coverage
            ix1, iy1 = max(x1, tx1), max(y1, ty1)
            ix2, iy2 = min(x2, tx2), min(y2, ty2)
            target = coverage[iy1 - ty1 : iy2 - ty1, ix1 - tx1 : ix2 - tx1]
            source = dab[iy1 - y1 : iy2 - y1, ix1 - x1 : ix2 - x1]
            np.maximum(target, source, out=target)

    def coverage_region(self, rect: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = rect
        result = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
        for key in self.tile_keys(rect):
            coverage = self.coverage_tiles.get(key)
            if coverage is None:
                continue
            tile_rect, _before = self.before_tiles[key]
            tx1, ty1, tx2, ty2 = tile_rect
            ix1, iy1 = max(x1, tx1), max(y1, ty1)
            ix2, iy2 = min(x2, tx2), min(y2, ty2)
            result[iy1 - y1 : iy2 - y1, ix1 - x1 : ix2 - x1] = coverage[iy1 - ty1 : iy2 - ty1, ix1 - tx1 : ix2 - tx1]
        return result

    @profiled("retouch.stroke_dab")
    def dab(self, x: int, y: int) -> tuple[int, int, int, int] | None:
        if self.layer.locked or self.strength <= 0.0:
            return None
        lx, ly = int(x) - self.layer.x, int(y) - self.layer.y
        height, width = self.layer.pixels.shape[:2]
        radius = self.radius
        x1, y1 = max(0, lx - radius), max(0, ly - radius)
        x2, y2 = min(width, lx + radius + 1), min(height, ly + radius + 1)
        if x1 >= x2 or y1 >= y2:
            return None
        full_mask = retouch_falloff_mask(radius, self.hardness)
        mx1, my1 = x1 - (lx - radius), y1 - (ly - radius)
        dab = full_mask[my1 : my1 + (y2 - y1), mx1 : mx1 + (x2 - x1)].copy()
        if self.selection_mask is not None:
            dab *= self.selection_mask[y1:y2, x1:x2].astype(np.float32) / 255.0
        if not np.any(dab > 0.0):
            return None
        halo = retouch_effect_halo(self.mode, self.strength)
        source_rect = max(0, x1 - halo), max(0, y1 - halo), min(width, x2 + halo), min(height, y2 + halo)
        self.capture_before(source_rect)
        self.merge_coverage((x1, y1, x2, y2), dab)
        source = self.original_region(source_rect)
        edited = retouch_effect_rgb(source, self.mode, self.strength, self.tonal_range)
        sx, sy = x1 - source_rect[0], y1 - source_rect[1]
        edited = edited[sy : sy + (y2 - y1), sx : sx + (x2 - x1)]
        original = self.original_region((x1, y1, x2, y2))
        mix = self.coverage_region((x1, y1, x2, y2)) * self.strength
        output = original.copy()
        output[:, :, :3] = np.clip(original[:, :, :3].astype(np.float32) * (1.0 - mix[:, :, None]) + edited * mix[:, :, None], 0, 255).astype(np.uint8)
        self.layer.pixels[y1:y2, x1:x2] = output
        return x1, y1, x2, y2


@profiled("stroke.brush_dab")
def draw_brush(layer: Layer, x: int, y: int, radius: int, color: tuple[int, int, int, int], opacity: float = 1.0, erase=False, selection_mask: np.ndarray | None = None) -> tuple[int, int, int, int] | None:
    if layer.locked:
        return None
    lx, ly = x - layer.x, y - layer.y
    if lx < -radius or ly < -radius or lx >= layer.pixels.shape[1] + radius or ly >= layer.pixels.shape[0] + radius:
        return None
    x1 = max(0, lx - radius)
    y1 = max(0, ly - radius)
    x2 = min(layer.pixels.shape[1], lx + radius + 1)
    y2 = min(layer.pixels.shape[0], ly + radius + 1)
    full_mask = brush_mask(radius)
    mx1 = x1 - (lx - radius)
    my1 = y1 - (ly - radius)
    mask = full_mask[my1 : my1 + (y2 - y1), mx1 : mx1 + (x2 - x1)]
    coverage = mask.astype(np.float32)
    if selection_mask is not None:
        coverage *= selection_mask[y1:y2, x1:x2].astype(np.float32) / 255.0
        if not np.any(coverage > 0.0):
            return None
    target = layer.pixels[y1:y2, x1:x2]
    if erase:
        target[:, :, 3] = np.clip(target[:, :, 3].astype(np.float32) * (1.0 - float(opacity) * coverage), 0, 255).astype(np.uint8)
    else:
        paint = np.array(color, dtype=np.float32)
        dst = target.astype(np.float32)
        original = dst.copy()
        sa = (paint[3] / 255.0) * float(opacity) * coverage
        da = dst[:, :, 3] / 255.0
        oa = sa + da * (1.0 - sa)
        dst[:, :, :3] = np.where(
            oa[:, :, None] > 0,
            (paint[:3] * sa[:, :, None] + dst[:, :, :3] * da[:, :, None] * (1.0 - sa[:, :, None])) / np.maximum(oa[:, :, None], 1e-6),
            0,
        )
        dst[:, :, 3] = oa * 255
        dst = np.where(coverage[:, :, None] > 0.0, dst, original)
        target[:] = np.clip(dst, 0, 255).astype(np.uint8)
    return x1, y1, x2, y2


@profiled("stroke.mask_dab")
def draw_mask_brush(layer: Layer, x: int, y: int, radius: int, value: int, opacity: float = 1.0, selection_mask: np.ndarray | None = None) -> tuple[int, int, int, int] | None:
    if layer.locked:
        return None
    if layer.mask is None:
        layer.mask = np.full(layer.pixels.shape[:2], 255, dtype=np.uint8)
        layer.mask_enabled = True
    lx, ly = x - layer.x, y - layer.y
    if lx < -radius or ly < -radius or lx >= layer.mask.shape[1] + radius or ly >= layer.mask.shape[0] + radius:
        return None
    x1 = max(0, lx - radius)
    y1 = max(0, ly - radius)
    x2 = min(layer.mask.shape[1], lx + radius + 1)
    y2 = min(layer.mask.shape[0], ly + radius + 1)
    full_mask = brush_mask(radius)
    mx1 = x1 - (lx - radius)
    my1 = y1 - (ly - radius)
    mask = full_mask[my1 : my1 + (y2 - y1), mx1 : mx1 + (x2 - x1)]
    coverage = mask.astype(np.float32)
    if selection_mask is not None:
        coverage *= selection_mask[y1:y2, x1:x2].astype(np.float32) / 255.0
        if not np.any(coverage > 0.0):
            return None
    target = layer.mask[y1:y2, x1:x2].astype(np.float32)
    mix = coverage * float(opacity)
    target = target * (1.0 - mix) + int(value) * mix
    layer.mask[y1:y2, x1:x2] = np.clip(target, 0, 255).astype(np.uint8)
    return x1, y1, x2, y2


@profiled("retouch.local_dab")
def local_retouch(
    layer: Layer,
    x: int,
    y: int,
    radius: int,
    mode: str,
    strength: float = 0.25,
    selection_mask: np.ndarray | None = None,
    hardness: float = 0.5,
    tonal_range: str = "midtones",
) -> tuple[int, int, int, int] | None:
    stroke = RetouchStroke(layer, mode, radius, hardness, strength, tonal_range, selection_mask)
    return stroke.dab(x, y)


@profiled("retouch.clone_heal_dab")
def clone_or_heal(
    layer: Layer,
    source_x: int,
    source_y: int,
    target_x: int,
    target_y: int,
    radius: int,
    opacity: float = 1.0,
    heal: bool = False,
    selection_mask: np.ndarray | None = None,
    hardness: float = 0.5,
    source_pixels: np.ndarray | None = None,
    source_origin: tuple[int, int] | None = None,
) -> tuple[int, int, int, int] | None:
    if layer.locked:
        return None
    radius = max(1, int(radius))
    source_image = layer.pixels if source_pixels is None else source_pixels
    source_offset = (layer.x, layer.y) if source_origin is None else source_origin
    sx, sy = int(source_x) - int(source_offset[0]), int(source_y) - int(source_offset[1])
    tx, ty = int(target_x) - layer.x, int(target_y) - layer.y
    target_h, target_w = layer.pixels.shape[:2]
    source_h, source_w = source_image.shape[:2]
    ox1 = max(-radius, -tx, -sx)
    oy1 = max(-radius, -ty, -sy)
    ox2 = min(radius + 1, target_w - tx, source_w - sx)
    oy2 = min(radius + 1, target_h - ty, source_h - sy)
    x1, y1, x2, y2 = tx + ox1, ty + oy1, tx + ox2, ty + oy2
    if x1 >= x2 or y1 >= y2:
        return None
    sx1 = sx + ox1
    sy1 = sy + oy1
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)
    full_mask = retouch_falloff_mask(radius, hardness)
    mx1 = ox1 + radius
    my1 = oy1 + radius
    mask = full_mask[my1 : my1 + (y2 - y1), mx1 : mx1 + (x2 - x1)].copy()
    if selection_mask is not None:
        mask *= selection_mask[y1:y2, x1:x2].astype(np.float32) / 255.0
        if not np.any(mask > 0.0):
            return None
    src = source_image[sy1:sy2, sx1:sx2].astype(np.float32)
    dst = layer.pixels[y1:y2, x1:x2].astype(np.float32)
    edited = src.copy()
    if heal:
        sigma = max(1.0, min(8.0, radius * 0.22))
        source_low = cv2.GaussianBlur(src[:, :, :3], (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
        target_low = cv2.GaussianBlur(dst[:, :, :3], (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
        edited[:, :, :3] = np.clip(target_low + (src[:, :, :3] - source_low), 0, 255)
        edited[:, :, 3] = dst[:, :, 3]
    mix = mask * float(np.clip(opacity, 0.0, 1.0))
    dst = dst * (1.0 - mix[:, :, None]) + edited * mix[:, :, None]
    layer.pixels[y1:y2, x1:x2] = np.clip(dst, 0, 255).astype(np.uint8)
    return x1, y1, x2, y2


@profiled("retouch.spot_heal_dab")
def spot_heal(
    layer: Layer,
    x: int,
    y: int,
    radius: int,
    strength: float = 1.0,
    selection_mask: np.ndarray | None = None,
    hardness: float = 0.45,
) -> tuple[int, int, int, int] | None:
    if layer.locked:
        return None
    radius = max(2, int(radius))
    lx, ly = int(x) - layer.x, int(y) - layer.y
    margin = max(5, radius)
    x1 = max(0, lx - radius - margin)
    y1 = max(0, ly - radius - margin)
    x2 = min(layer.pixels.shape[1], lx + radius + margin + 1)
    y2 = min(layer.pixels.shape[0], ly + radius + margin + 1)
    if x1 >= x2 or y1 >= y2:
        return None

    falloff_full = retouch_falloff_mask(radius, hardness)
    target_left, target_top = lx - radius, ly - radius
    fx1, fy1 = max(x1, target_left), max(y1, target_top)
    fx2, fy2 = min(x2, lx + radius + 1), min(y2, ly + radius + 1)
    falloff = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
    if fx1 < fx2 and fy1 < fy2:
        falloff[fy1 - y1 : fy2 - y1, fx1 - x1 : fx2 - x1] = falloff_full[
            fy1 - target_top : fy2 - target_top,
            fx1 - target_left : fx2 - target_left,
        ]
    if selection_mask is not None:
        falloff *= selection_mask[y1:y2, x1:x2].astype(np.float32) / 255.0
    target_mask = (falloff > 0.04).astype(np.uint8) * 255
    if not np.any(target_mask > 0):
        return None

    patch = layer.pixels[y1:y2, x1:x2].copy()
    inpaint_radius = max(2.0, min(12.0, radius * 0.55))
    healed_rgb = cv2.inpaint(patch[:, :, :3], target_mask, inpaint_radius, cv2.INPAINT_TELEA)
    feather = falloff * float(np.clip(strength, 0.0, 1.0))
    mixed = patch[:, :, :3].astype(np.float32) * (1.0 - feather[:, :, None]) + healed_rgb.astype(np.float32) * feather[:, :, None]
    layer.pixels[y1:y2, x1:x2, :3] = np.clip(mixed, 0, 255).astype(np.uint8)
    return x1, y1, x2, y2


def flood_fill(layer: Layer, x: int, y: int, color: tuple[int, int, int, int], tolerance: int, selection_mask: np.ndarray | None = None) -> None:
    if layer.locked:
        return
    lx, ly = x - layer.x, y - layer.y
    if lx < 0 or ly < 0 or lx >= layer.pixels.shape[1] or ly >= layer.pixels.shape[0]:
        return
    img = layer.pixels.copy()
    seed = img[ly, lx].astype(np.int16)
    diff = np.abs(img.astype(np.int16) - seed).max(axis=2)
    mask = (diff <= tolerance).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 4)
    label = labels[ly, lx]
    if label == 0 and mask[ly, lx] == 0:
        return
    region = labels == label
    if selection_mask is not None:
        coverage = (selection_mask.astype(np.float32) / 255.0) * region.astype(np.float32)
        target = layer.pixels.astype(np.float32)
        paint = np.array(color, dtype=np.float32)
        layer.pixels[:] = np.clip(target * (1.0 - coverage[:, :, None]) + paint * coverage[:, :, None], 0, 255).astype(np.uint8)
    else:
        layer.pixels[region] = np.array(color, dtype=np.uint8)


def apply_gradient(
    layer: Layer,
    vector: tuple[int, int, int, int],
    start: tuple[int, int, int, int],
    end: tuple[int, int, int, int],
    selection_mask: np.ndarray | None = None,
    kind: str = "linear",
    stops: list[Any] | None = None,
) -> None:
    if layer.locked:
        return
    x1, y1, x2, y2 = (int(value) for value in vector)
    gradient_stops = stops or [(0.0, start), (1.0, end)]
    height, width = layer.pixels.shape[:2]
    patch = GradientEngine.render(width, height, (x1, y1), (x2, y2), gradient_stops, kind, (layer.x, layer.y))
    if selection_mask is None:
        layer.pixels[:] = patch
    else:
        coverage = selection_mask.astype(np.float32) / 255.0
        target = layer.pixels.astype(np.float32)
        layer.pixels[:] = np.clip(
            target * (1.0 - coverage[:, :, None]) + patch.astype(np.float32) * coverage[:, :, None],
            0,
            255,
        ).astype(np.uint8)


def add_text(layer: Layer, x: int, y: int, text: str, color: tuple[int, int, int, int], size: int, selection_mask: np.ndarray | None = None) -> None:
    if layer.locked:
        return
    pil = rgba_array_to_pil(layer.pixels)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    draw.text((x - layer.x, y - layer.y), text, fill=color, font=font)
    rendered = pil_to_rgba_array(pil)
    if selection_mask is None:
        layer.pixels = rendered
    else:
        coverage = selection_mask.astype(np.float32) / 255.0
        layer.pixels[:] = np.clip(
            layer.pixels.astype(np.float32) * (1.0 - coverage[:, :, None]) + rendered.astype(np.float32) * coverage[:, :, None],
            0,
            255,
        ).astype(np.uint8)


def render_text_layer(layer: Layer) -> None:
    if layer.text_data is None:
        return
    layer.pixels[:] = 0
    pil = rgba_array_to_pil(layer.pixels)
    draw = ImageDraw.Draw(pil)
    data = layer.text_data
    bold = bool(data.get("bold", False))
    italic = bool(data.get("italic", False))
    underline = bool(data.get("underline", False))
    font = load_text_font(str(data.get("font_family", "arial.ttf")), int(data.get("size", 48)), bold, italic)
    color = tuple(int(v) for v in data.get("color", [255, 255, 255, 255]))
    x = int(data.get("x", 0))
    y = int(data.get("y", 0))
    size = int(data.get("size", 48))
    box_width = max(0, int(data.get("box_width", 0) or 0))
    spacing = max(0, int(data.get("line_spacing", max(2, size // 5))))
    tracking = int(data.get("tracking", 0))
    align = str(data.get("align", "left")).lower()
    path_mode = str(data.get("path_mode", "none")).lower()
    path_amount = int(data.get("path_amount", 0))
    baseline_shift = int(data.get("baseline_shift", 0))
    if path_mode == "bezier":
        draw_text_on_bezier_path(pil, data, font, color, tracking, bold, underline)
    else:
        lines = wrapped_text_lines(draw, str(data.get("text", "")), font, box_width, tracking)
        line_y = y
        for line in lines:
            bbox = draw.textbbox((0, 0), line or " ", font=font)
            line_width = text_line_width(draw, line or " ", font, tracking)
            dx = 0
            if box_width > 0 and align == "center":
                dx = max(0, (box_width - line_width) // 2)
            elif box_width > 0 and align == "right":
                dx = max(0, box_width - line_width)
            draw_text_styled(
                draw,
                (x + dx, line_y - baseline_shift),
                line,
                fill=color,
                font=font,
                tracking=tracking,
                bold=bold,
                underline=underline,
                path_mode=path_mode,
                path_amount=path_amount,
            )
            line_y += max(1, bbox[3] - bbox[1]) + spacing
    rendered = pil_to_rgba_array(pil)
    rotation = float(data.get("rotation", 0.0))
    flip_horizontal = bool(data.get("flip_horizontal", False))
    flip_vertical = bool(data.get("flip_vertical", False))
    if (abs(rotation) > 0.001 or flip_horizontal or flip_vertical) and np.any(rendered[:, :, 3]):
        ys, xs = np.where(rendered[:, :, 3] > 0)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
        patch = rendered[y1:y2, x1:x2]
        if flip_horizontal:
            patch = cv2.flip(patch, 1)
        if flip_vertical:
            patch = cv2.flip(patch, 0)
        if abs(rotation) > 0.001:
            patch = rotate_bound(patch, rotation, cv2.INTER_CUBIC)
        rotated = np.zeros_like(rendered)
        px = round((x1 + x2 - patch.shape[1]) / 2)
        py = round((y1 + y2 - patch.shape[0]) / 2)
        alpha_blend_inplace(rotated, patch, px, py, 1.0)
        rendered = rotated
    layer.pixels = rendered


def normalize_text_path_points(
    points: list[tuple[float, float]] | list[list[float]] | None,
    x: int,
    baseline_y: int,
    width: int,
) -> list[list[float]]:
    if isinstance(points, list) and len(points) == 4:
        normalized: list[list[float]] = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                break
            try:
                normalized.append([float(point[0]), float(point[1])])
            except (TypeError, ValueError):
                break
        if len(normalized) == 4:
            return normalized
    path_width = max(40.0, float(width))
    return [
        [float(x), float(baseline_y)],
        [float(x) + path_width / 3.0, float(baseline_y)],
        [float(x) + path_width * 2.0 / 3.0, float(baseline_y)],
        [float(x) + path_width, float(baseline_y)],
    ]


def normalize_text_path_range(start: float, end: float) -> tuple[float, float]:
    normalized_start = max(0.0, min(0.99, float(start)))
    normalized_end = max(0.01, min(1.0, float(end)))
    if normalized_end - normalized_start < 0.01:
        if normalized_start <= 0.99:
            normalized_end = min(1.0, normalized_start + 0.01)
        else:
            normalized_start = max(0.0, normalized_end - 0.01)
    return normalized_start, normalized_end


def text_path_samples(
    points: list[tuple[float, float]] | list[list[float]],
    steps: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = normalize_text_path_points(points, 0, 0, 100)
    control = np.asarray(normalized, dtype=np.float64)
    count = max(16, int(steps))
    t = np.linspace(0.0, 1.0, count + 1, dtype=np.float64)
    omt = 1.0 - t
    positions = (
        (omt ** 3)[:, None] * control[0]
        + (3.0 * omt * omt * t)[:, None] * control[1]
        + (3.0 * omt * t * t)[:, None] * control[2]
        + (t ** 3)[:, None] * control[3]
    )
    tangents = (
        (3.0 * omt * omt)[:, None] * (control[1] - control[0])
        + (6.0 * omt * t)[:, None] * (control[2] - control[1])
        + (3.0 * t * t)[:, None] * (control[3] - control[2])
    )
    lengths = np.linalg.norm(tangents, axis=1)
    invalid = lengths < 1e-8
    if np.any(invalid):
        differences = np.gradient(positions, axis=0)
        tangents[invalid] = differences[invalid]
        lengths = np.linalg.norm(tangents, axis=1)
    tangents /= np.maximum(lengths[:, None], 1e-8)
    segments = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segments)))
    return positions, tangents, cumulative


def text_path_point_at_distance(
    positions: np.ndarray,
    tangents: np.ndarray,
    cumulative: np.ndarray,
    distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(cumulative) < 2 or cumulative[-1] <= 1e-8:
        return positions[0].copy(), tangents[0].copy()
    target = max(0.0, min(float(cumulative[-1]), float(distance)))
    upper = min(len(cumulative) - 1, max(1, int(np.searchsorted(cumulative, target, side="right"))))
    lower = upper - 1
    span = max(1e-8, float(cumulative[upper] - cumulative[lower]))
    ratio = (target - float(cumulative[lower])) / span
    point = positions[lower] * (1.0 - ratio) + positions[upper] * ratio
    tangent = tangents[lower] * (1.0 - ratio) + tangents[upper] * ratio
    tangent /= max(1e-8, float(np.linalg.norm(tangent)))
    return point, tangent


def draw_text_on_bezier_path(
    image: Image.Image,
    data: dict[str, Any],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    tracking: int = 0,
    bold: bool = False,
    underline: bool = False,
) -> None:
    size = max(4, int(data.get("size", 48)))
    points = normalize_text_path_points(
        data.get("path_points"),
        int(data.get("x", 0)),
        int(data.get("y", 0)) + size,
        max(int(data.get("box_width", 0) or 0), size * 8),
    )
    sample_count = min(4096, max(256, int(cumulative_path_estimate(points) * 1.5)))
    positions, tangents, cumulative = text_path_samples(points, sample_count)
    path_reversed = bool(data.get("path_reverse", False))
    if path_reversed:
        positions = positions[::-1].copy()
        tangents = (-tangents[::-1]).copy()
        segments = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segments)))
    total_path = float(cumulative[-1])
    start_fraction = max(0.0, min(1.0, float(data.get("path_start", 0.0))))
    end_fraction = max(0.0, min(1.0, float(data.get("path_end", 1.0))))
    start_distance, end_distance = sorted((start_fraction * total_path, end_fraction * total_path))
    available = end_distance - start_distance
    if available <= 1.0:
        return
    text = " ".join(str(data.get("text", "")).splitlines())
    if not text:
        return
    measure = ImageDraw.Draw(image)
    advances: list[float] = []
    for char in text:
        bbox = measure.textbbox((0, 0), char or " ", font=font, stroke_width=1 if bold else 0)
        advances.append(float(max(1, bbox[2] - bbox[0]) + int(tracking)))
    total_text = max(0.0, sum(advances) - int(tracking))
    align = str(data.get("align", "left")).lower()
    cursor = start_distance
    if align == "center":
        cursor += max(0.0, (available - total_text) / 2.0)
    elif align == "right":
        cursor += max(0.0, available - total_text)
    baseline_shift = float(data.get("baseline_shift", 0))
    side = -1 if int(data.get("path_side", 1)) < 0 else 1
    first_distance = cursor
    last_distance = cursor
    for char, advance in zip(text, advances):
        center_distance = cursor + advance / 2.0
        if center_distance > end_distance:
            break
        point, tangent = text_path_point_at_distance(positions, tangents, cumulative, center_distance)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        if path_reversed:
            normal = -normal
        bbox = measure.textbbox((0, 0), char or " ", font=font, stroke_width=1 if bold else 0)
        glyph_width = max(1, bbox[2] - bbox[0])
        glyph_height = max(1, bbox[3] - bbox[1])
        padding = max(4, size // 4)
        glyph = Image.new("RGBA", (glyph_width + padding * 2, glyph_height + padding * 2), (0, 0, 0, 0))
        glyph_draw = ImageDraw.Draw(glyph)
        glyph_draw.text(
            (padding - bbox[0], padding - bbox[1]),
            char,
            fill=fill,
            font=font,
            stroke_width=1 if bold else 0,
            stroke_fill=fill,
        )
        angle = math.degrees(math.atan2(float(tangent[1]), float(tangent[0])))
        rotated = glyph.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=True)
        center = point - normal * side * (glyph_height / 2.0 + baseline_shift)
        left = round(float(center[0]) - rotated.width / 2.0)
        top = round(float(center[1]) - rotated.height / 2.0)
        image.paste(rotated, (left, top), rotated)
        cursor += advance
        last_distance = min(cursor, end_distance)
    if underline and last_distance > first_distance:
        path_draw = ImageDraw.Draw(image)
        samples = max(8, round((last_distance - first_distance) / 4.0))
        line_points: list[tuple[float, float]] = []
        for distance in np.linspace(first_distance, last_distance, samples):
            point, tangent = text_path_point_at_distance(positions, tangents, cumulative, float(distance))
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
            if path_reversed:
                normal = -normal
            underline_point = point + normal * side * (max(1.0, size * 0.08) - baseline_shift)
            line_points.append((float(underline_point[0]), float(underline_point[1])))
        path_draw.line(line_points, fill=fill, width=max(1, size // 16), joint="curve")


def cumulative_path_estimate(points: list[list[float]]) -> float:
    return max(16.0, sum(math.hypot(points[index + 1][0] - points[index][0], points[index + 1][1] - points[index][1]) for index in range(3)))


def wrapped_text_lines(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, box_width: int, tracking: int = 0) -> list[str]:
    if box_width <= 0:
        return text.splitlines() or [""]
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for word in paragraph.split(" "):
            candidate = word if not current else f"{current} {word}"
            if text_line_width(draw, candidate, font, tracking) <= box_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def text_line_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, tracking: int = 0) -> int:
    if not text:
        return 0
    bbox = draw.textbbox((0, 0), text, font=font)
    base_width = bbox[2] - bbox[0]
    return max(0, base_width + max(0, len(text) - 1) * int(tracking))


def draw_text_with_tracking(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int, int], font: ImageFont.ImageFont, tracking: int = 0) -> None:
    if tracking == 0 or len(text) <= 1:
        draw.text(xy, text, fill=fill, font=font)
        return
    x, y = xy
    for char in text:
        draw.text((x, y), char, fill=fill, font=font)
        bbox = draw.textbbox((0, 0), char, font=font)
        x += max(0, bbox[2] - bbox[0]) + int(tracking)


def draw_text_styled(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fill: tuple[int, int, int, int],
    font: ImageFont.ImageFont,
    tracking: int = 0,
    bold: bool = False,
    underline: bool = False,
    path_mode: str = "none",
    path_amount: int = 0,
) -> None:
    x, y = xy
    total_width = max(1, text_line_width(draw, text or " ", font, tracking))
    cursor = 0.0
    for char in text:
        bbox = draw.textbbox((0, 0), char or " ", font=font)
        char_width = max(1, bbox[2] - bbox[0])
        center = cursor + char_width / 2.0
        offset = 0.0
        if path_mode == "arc":
            normalized = center / total_width * 2.0 - 1.0
            offset = -float(path_amount) * (1.0 - normalized * normalized)
        elif path_mode == "wave":
            offset = float(path_amount) * math.sin(center / total_width * math.pi * 2.0)
        draw.text((x + cursor, y + offset), char, fill=fill, font=font, stroke_width=1 if bold else 0, stroke_fill=fill)
        cursor += char_width + int(tracking)
    if underline and text:
        underline_y = y + max(1, int(getattr(font, "size", 12) * 1.05))
        draw.line((x, underline_y, x + total_width, underline_y), fill=fill, width=max(1, int(getattr(font, "size", 12) / 16)))


def load_text_font(font_family: str, size: int, bold: bool = False, italic: bool = False) -> ImageFont.ImageFont:
    family = font_family.strip() or "arial.ttf"
    candidates: list[str] = []
    compact = family.lower().replace(" ", "").removesuffix(".ttf")
    if compact == "arial":
        candidates.append("arialbi.ttf" if bold and italic else "arialbd.ttf" if bold else "ariali.ttf" if italic else "arial.ttf")
    if not Path(family).suffix and (bold or italic):
        suffix = " Bold Italic" if bold and italic else " Bold" if bold else " Italic"
        candidates.extend([f"{family}{suffix}.ttf", f"{compact}{'bi' if bold and italic else 'bd' if bold else 'i'}.ttf"])
    candidates.append(family)
    if not Path(family).suffix:
        candidates.extend([f"{family}.ttf", f"{compact}.ttf"])
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_shape_layer(layer: Layer) -> None:
    if layer.shape_data is None:
        return
    layer.pixels[:] = 0
    data = layer.shape_data
    if str(data.get("shape", "rectangle")).lower() == "boolean":
        render_boolean_shape_layer(layer)
        return
    gradient = data.get("gradient")
    if isinstance(gradient, dict):
        height, width = layer.pixels.shape[:2]
        region = shape_render_region(data, width, height)
        if region is None:
            return
        x1, y1, x2, y2 = region
        local_data = translated_shape_data(data, -x1, -y1)
        mask = shape_data_to_mask(local_data, (y2 - y1, x2 - x1))
        start = tuple(float(value) for value in gradient.get("start", data.get("box", [0, 0])[:2]))
        end = tuple(float(value) for value in gradient.get("end", data.get("box", [0, 0, 1, 1])[2:4]))
        pixels = GradientEngine.render(
            x2 - x1,
            y2 - y1,
            (start[0] - x1, start[1] - y1),
            (end[0] - x1, end[1] - y1),
            gradient.get("stops"),
            str(gradient.get("type", "linear")),
        )
        opacity = float(np.clip(gradient.get("opacity", 1.0), 0.0, 1.0))
        pixels[:, :, 3] = np.clip(pixels[:, :, 3].astype(np.float32) * (mask.astype(np.float32) / 255.0) * opacity, 0, 255).astype(np.uint8)
        stroke = data.get("stroke")
        stroke_width = max(0, int(data.get("stroke_width", 0)))
        if stroke is not None and stroke_width > 0:
            stroke_color = np.array(stroke, dtype=np.uint8)
            kernel = np.ones((stroke_width * 2 + 1, stroke_width * 2 + 1), dtype=np.uint8)
            edge = (cv2.dilate(mask, kernel) > 0) & (cv2.erode(mask, kernel) == 0)
            pixels[edge] = stroke_color
        layer.pixels[y1:y2, x1:x2] = pixels
        return
    texture = data.get("texture")
    if isinstance(texture, dict):
        height, width = layer.pixels.shape[:2]
        region = shape_render_region(data, width, height)
        if region is None:
            return
        x1, y1, x2, y2 = region
        local_data = translated_shape_data(data, -x1, -y1)
        mask = shape_data_to_mask(local_data, (y2 - y1, x2 - x1))
        yy, xx = np.mgrid[y1:y2, x1:x2]
        size = max(2, int(texture.get("size", 18)))
        kind = str(texture.get("type", "checker"))
        if kind == "dots":
            cx = np.mod(xx, size) - size / 2.0
            cy = np.mod(yy, size) - size / 2.0
            selector = (cx * cx + cy * cy) <= (size * 0.24) ** 2
        elif kind == "stripes":
            selector = np.mod((xx + yy) // size, 2) == 0
        else:
            selector = np.mod(xx // size + yy // size, 2) == 0
        first = np.array(texture.get("color_a", data.get("fill", [255, 255, 255, 255])), dtype=np.uint8)
        second = np.array(texture.get("color_b", data.get("stroke", [0, 0, 0, 255]) or [0, 0, 0, 255]), dtype=np.uint8)
        pixels = np.where(selector[:, :, None], first, second).astype(np.uint8)
        pixels[:, :, 3] = np.clip(pixels[:, :, 3].astype(np.float32) * (mask.astype(np.float32) / 255.0), 0, 255).astype(np.uint8)
        stroke = data.get("stroke")
        stroke_width = max(0, int(data.get("stroke_width", 0)))
        if stroke is not None and stroke_width > 0:
            stroke_color = np.array(stroke, dtype=np.uint8)
            kernel = np.ones((stroke_width * 2 + 1, stroke_width * 2 + 1), dtype=np.uint8)
            edge = (cv2.dilate(mask, kernel) > 0) & (cv2.erode(mask, kernel) == 0)
            pixels[edge] = stroke_color
        layer.pixels[y1:y2, x1:x2] = pixels
        return
    pil = rgba_array_to_pil(layer.pixels)
    draw = ImageDraw.Draw(pil)
    box = tuple(int(v) for v in data.get("box", [0, 0, 1, 1]))
    fill = tuple(int(v) for v in data.get("fill", [255, 255, 255, 255]))
    stroke = data.get("stroke")
    outline = None if stroke is None else tuple(int(v) for v in stroke)
    stroke_width = max(0, int(data.get("stroke_width", 0)))
    shape = str(data.get("shape", "rectangle")).lower()
    if shape == "ellipse":
        draw.ellipse(box, fill=fill, outline=outline, width=stroke_width)
    elif shape == "line":
        draw.line((box[0], box[1], box[2], box[3]), fill=outline or fill, width=max(1, stroke_width or 1))
    elif shape == "bezier":
        points = bezier_curve_points(data.get("control_points"), box)
        draw.line(points, fill=outline or fill, width=max(1, stroke_width or 2), joint="curve")
    elif shape == "polygon":
        points = regular_polygon_points(box, max(3, int(data.get("sides", 5))))
        draw.polygon(points, fill=fill)
        if outline is not None and stroke_width > 0:
            draw.line(points + [points[0]], fill=outline, width=stroke_width)
    elif shape == "star":
        points = star_points(box, max(3, int(data.get("sides", 5))), float(data.get("inner_ratio", 0.5)))
        draw.polygon(points, fill=fill)
        if outline is not None and stroke_width > 0:
            draw.line(points + [points[0]], fill=outline, width=stroke_width)
    elif shape == "custom":
        points = custom_shape_points(data.get("custom_points"), box)
        draw.polygon(points, fill=fill)
        if outline is not None and stroke_width > 0 and points:
            draw.line(points + [points[0]], fill=outline, width=stroke_width)
    else:
        draw.rectangle(box, fill=fill, outline=outline, width=stroke_width)
    layer.pixels = pil_to_rgba_array(pil)


def shape_render_region(data: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int] | None:
    box = shape_data_bounds(data) or (0, 0, 1, 1)
    margin = max(1, int(data.get("stroke_width", 0)) * 2 + 1)
    x1 = max(0, box[0] - margin)
    y1 = max(0, box[1] - margin)
    x2 = min(width, box[2] + margin + 1)
    y2 = min(height, box[3] + margin + 1)
    return None if x1 >= x2 or y1 >= y2 else (x1, y1, x2, y2)


def translated_shape_data(data: dict[str, Any], dx: int, dy: int) -> dict[str, Any]:
    translated = json.loads(json.dumps(data))
    if str(data.get("shape", "rectangle")).lower() == "boolean":
        translated["children"] = [
            translated_shape_data(child, dx, dy) if isinstance(child, dict) else child
            for child in data.get("children", [])
        ]
    else:
        box = list(data.get("box", [0, 0, 1, 1]))
        translated["box"] = [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]
        points = data.get("control_points")
        if isinstance(points, list):
            translated["control_points"] = [[float(point[0]) + dx, float(point[1]) + dy] for point in points]
        gradient = data.get("gradient")
        if isinstance(gradient, dict):
            for key in ("start", "end"):
                point = gradient.get(key)
                if isinstance(point, list) and len(point) >= 2:
                    gradient[key] = [float(point[0]) + dx, float(point[1]) + dy]
    translated["box"] = list(shape_data_bounds(translated) or (0, 0, 1, 1))
    return translated


def shape_data_bounds(data: dict[str, Any]) -> tuple[int, int, int, int] | None:
    if str(data.get("shape", "rectangle")).lower() == "boolean":
        bounds = [
            shape_data_bounds(child)
            for child in data.get("children", [])
            if isinstance(child, dict) and bool(child.get("_enabled", True))
        ]
        bounds = [box for box in bounds if box is not None]
        if not bounds:
            return None
        return (
            min(box[0] for box in bounds), min(box[1] for box in bounds),
            max(box[2] for box in bounds), max(box[3] for box in bounds),
        )
    try:
        return normalized_box(tuple(int(round(value)) for value in data.get("box", [0, 0, 1, 1])))
    except (TypeError, ValueError):
        return None


def transform_shape_data_to_box(data: dict[str, Any], target: tuple[int, int, int, int]) -> dict[str, Any]:
    """Scale all editable geometry, including nested boolean operands, into target."""
    source = shape_data_bounds(data)
    if source is None:
        return json.loads(json.dumps(data))
    tx1, ty1, tx2, ty2 = normalized_box(target)
    sx1, sy1, sx2, sy2 = source
    scale_x = (tx2 - tx1) / max(1, sx2 - sx1)
    scale_y = (ty2 - ty1) / max(1, sy2 - sy1)

    def point(value: Any) -> list[float]:
        return [tx1 + (float(value[0]) - sx1) * scale_x, ty1 + (float(value[1]) - sy1) * scale_y]

    def transform(current: dict[str, Any]) -> dict[str, Any]:
        result = json.loads(json.dumps(current))
        if str(current.get("shape", "rectangle")).lower() == "boolean":
            result["children"] = [transform(child) if isinstance(child, dict) else child for child in current.get("children", [])]
        else:
            raw_box = current.get("box", [0, 0, 1, 1])
            first, second = point(raw_box[:2]), point(raw_box[2:4])
            result["box"] = [first[0], first[1], second[0], second[1]]
            points = current.get("control_points")
            if isinstance(points, list):
                result["control_points"] = [point(value) for value in points]
            gradient = result.get("gradient")
            if isinstance(gradient, dict):
                for key in ("start", "end"):
                    value = current.get("gradient", {}).get(key)
                    if isinstance(value, list) and len(value) >= 2:
                        gradient[key] = point(value)
        result["box"] = list(shape_data_bounds(result) or (tx1, ty1, tx2, ty2))
        return result

    transformed = transform(data)
    transformed["box"] = [tx1, ty1, tx2, ty2]
    return transformed


def render_boolean_shape_layer(layer: Layer) -> None:
    if layer.shape_data is None:
        return
    data = layer.shape_data
    mask = boolean_shape_mask(data, layer.pixels.shape[:2])
    fill = tuple(int(v) for v in data.get("fill", [255, 255, 255, 255]))
    out = np.zeros_like(layer.pixels)
    out[:, :, :3] = fill[:3]
    out[:, :, 3] = np.clip(mask.astype(np.float32) * (fill[3] / 255.0), 0, 255).astype(np.uint8)
    stroke = data.get("stroke")
    stroke_width = max(0, int(data.get("stroke_width", 0)))
    if stroke is not None and stroke_width > 0 and np.any(mask):
        stroke_color = tuple(int(v) for v in stroke)
        kernel = np.ones((stroke_width * 2 + 1, stroke_width * 2 + 1), dtype=np.uint8)
        outer = cv2.dilate(mask, kernel)
        inner = cv2.erode(mask, kernel)
        edge = ((outer > 0) & (inner == 0))
        out[edge, :3] = stroke_color[:3]
        out[edge, 3] = stroke_color[3]
    layer.pixels = out


def boolean_shape_mask(data: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    children = data.get("children", [])
    if not isinstance(children, list) or not children:
        return np.zeros(shape, dtype=np.uint8)
    masks = [
        shape_data_to_mask(child, shape)
        for child in children
        if isinstance(child, dict) and bool(child.get("_enabled", True))
    ]
    if not masks:
        return np.zeros(shape, dtype=np.uint8)
    mode = str(data.get("boolean_mode", "union")).lower()
    result = masks[0] > 0
    for mask in masks[1:]:
        other = mask > 0
        if mode == "subtract":
            result = result & ~other
        elif mode == "intersect":
            result = result & other
        elif mode == "xor":
            result = result ^ other
        else:
            result = result | other
    return (result.astype(np.uint8) * 255)


def shape_data_to_mask(data: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    if str(data.get("shape", "rectangle")).lower() == "boolean":
        return boolean_shape_mask(data, shape)
    pil = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(pil)
    box = tuple(int(v) for v in data.get("box", [0, 0, 1, 1]))
    stroke_width = max(1, int(data.get("stroke_width", 1)))
    kind = str(data.get("shape", "rectangle")).lower()
    if kind == "ellipse":
        draw.ellipse(box, fill=255)
    elif kind == "line":
        draw.line((box[0], box[1], box[2], box[3]), fill=255, width=stroke_width)
    elif kind == "bezier":
        draw.line(bezier_curve_points(data.get("control_points"), box), fill=255, width=max(1, stroke_width), joint="curve")
    elif kind == "polygon":
        draw.polygon(regular_polygon_points(box, max(3, int(data.get("sides", 5)))), fill=255)
    elif kind == "star":
        draw.polygon(star_points(box, max(3, int(data.get("sides", 5))), float(data.get("inner_ratio", 0.5))), fill=255)
    elif kind == "custom":
        draw.polygon(custom_shape_points(data.get("custom_points"), box), fill=255)
    else:
        draw.rectangle(box, fill=255)
    return np.array(pil, dtype=np.uint8)


def bezier_curve_points(raw_points: Any, box: tuple[int, int, int, int], steps: int = 64) -> list[tuple[float, float]]:
    if isinstance(raw_points, list) and len(raw_points) == 4:
        try:
            p0, p1, p2, p3 = [tuple(float(v) for v in point[:2]) for point in raw_points]
        except (TypeError, ValueError):
            p0 = p1 = p2 = p3 = None
    else:
        p0 = p1 = p2 = p3 = None
    if p0 is None:
        x1, y1, x2, y2 = normalized_box(box)
        p0, p1, p2, p3 = (x1, y2), (x1, y1), (x2, y1), (x2, y2)
    coords: list[tuple[float, float]] = []
    for i in range(max(2, steps) + 1):
        t = i / max(2, steps)
        mt = 1.0 - t
        x = mt**3 * p0[0] + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t**3 * p3[1]
        coords.append((x, y))
    return coords


def custom_shape_points(raw_points: Any, box: tuple[int, int, int, int]) -> list[tuple[float, float]]:
    if not isinstance(raw_points, list) or len(raw_points) < 3:
        raw_points = [[0.5, 0.0], [1.0, 0.5], [0.5, 1.0], [0.0, 0.5]]
    x1, y1, x2, y2 = normalized_box(box)
    width, height = x2 - x1, y2 - y1
    points: list[tuple[float, float]] = []
    for point in raw_points:
        try:
            px, py = float(point[0]), float(point[1])
        except (TypeError, ValueError, IndexError):
            continue
        points.append((x1 + np.clip(px, 0.0, 1.0) * width, y1 + np.clip(py, 0.0, 1.0) * height))
    return points if len(points) >= 3 else [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def regular_polygon_points(box: tuple[int, int, int, int], sides: int) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = normalized_box(box)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    rx, ry = abs(x2 - x1) / 2.0, abs(y2 - y1) / 2.0
    start = -math.pi / 2.0
    return [(cx + math.cos(start + i * 2.0 * math.pi / sides) * rx, cy + math.sin(start + i * 2.0 * math.pi / sides) * ry) for i in range(sides)]


def star_points(box: tuple[int, int, int, int], points: int, inner_ratio: float) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = normalized_box(box)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    rx, ry = abs(x2 - x1) / 2.0, abs(y2 - y1) / 2.0
    inner_ratio = float(np.clip(inner_ratio, 0.05, 0.95))
    start = -math.pi / 2.0
    coords: list[tuple[float, float]] = []
    for i in range(points * 2):
        scale = 1.0 if i % 2 == 0 else inner_ratio
        angle = start + i * math.pi / points
        coords.append((cx + math.cos(angle) * rx * scale, cy + math.sin(angle) * ry * scale))
    return coords


def adjust_brightness_contrast(arr: np.ndarray, brightness: int, contrast: float) -> np.ndarray:
    out = arr.copy().astype(np.float32)
    out[:, :, :3] = np.clip((out[:, :, :3] - 127.5) * contrast + 127.5 + brightness, 0, 255)
    return out.astype(np.uint8)


def adjust_saturation(arr: np.ndarray, saturation: float) -> np.ndarray:
    rgb = arr[:, :, :3]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    out = arr.copy()
    out[:, :, :3] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out


def adjust_vibrance(arr: np.ndarray, vibrance: float = 0.0, saturation: float = 1.0) -> np.ndarray:
    out = arr.copy()
    hsv = cv2.cvtColor(out[:, :, :3], cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[:, :, 1] / 255.0
    amount = float(np.clip(vibrance, -1.0, 1.0))
    if amount >= 0.0:
        sat = sat + (1.0 - sat) * (1.0 - sat) * amount
    else:
        sat = sat * (1.0 + amount)
    hsv[:, :, 1] = np.clip(sat * max(0.0, float(saturation)), 0.0, 1.0) * 255.0
    out[:, :, :3] = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out


def adjust_temperature_tint(arr: np.ndarray, temperature: float = 0.0, tint: float = 0.0) -> np.ndarray:
    out = arr.copy()
    rgb = out[:, :, :3].astype(np.float32)
    temperature = float(np.clip(temperature, -100.0, 100.0))
    tint = float(np.clip(tint, -100.0, 100.0))
    rgb[:, :, 0] += temperature * 0.72 + tint * 0.18
    rgb[:, :, 1] += tint * 0.52 - abs(temperature) * 0.06
    rgb[:, :, 2] -= temperature * 0.72 + tint * 0.18
    out[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return out


def adjust_hue_saturation(arr: np.ndarray, hue: int, saturation: float, lightness: int) -> np.ndarray:
    rgb = arr[:, :, :3]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + float(hue) / 2.0) % 180.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(saturation), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + float(lightness), 0, 255)
    out = arr.copy()
    out[:, :, :3] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out


def adjust_exposure(arr: np.ndarray, exposure: float, offset: float, gamma: float) -> np.ndarray:
    out = arr.copy().astype(np.float32)
    rgb = out[:, :, :3] / 255.0
    rgb = np.clip(rgb * (2.0 ** float(exposure)) + float(offset), 0, 1)
    rgb = np.power(rgb, 1.0 / max(0.01, float(gamma)))
    out[:, :, :3] = rgb * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def adjust_color_balance(arr: np.ndarray, red: int, green: int, blue: int) -> np.ndarray:
    out = arr.copy().astype(np.float32)
    shifts = np.array([red, green, blue], dtype=np.float32)
    out[:, :, :3] = np.clip(out[:, :, :3] + shifts, 0, 255)
    return out.astype(np.uint8)


def adjust_threshold(arr: np.ndarray, threshold: int) -> np.ndarray:
    out = arr.copy()
    gray = cv2.cvtColor(out[:, :, :3], cv2.COLOR_RGB2GRAY)
    bw = np.where(gray >= int(np.clip(threshold, 0, 255)), 255, 0).astype(np.uint8)
    out[:, :, :3] = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
    return out


def adjust_posterize(arr: np.ndarray, levels_count: int) -> np.ndarray:
    levels_count = int(np.clip(levels_count, 2, 64))
    out = arr.copy().astype(np.float32)
    rgb = out[:, :, :3] / 255.0
    out[:, :, :3] = np.round(rgb * (levels_count - 1)) * (255.0 / (levels_count - 1))
    return np.clip(out, 0, 255).astype(np.uint8)


def levels(arr: np.ndarray, black: int, white: int, gamma: float) -> np.ndarray:
    out = arr.copy().astype(np.float32)
    rgb = np.clip((out[:, :, :3] - black) / max(1, white - black), 0, 1)
    rgb = np.power(rgb, 1.0 / max(0.01, gamma)) * 255
    out[:, :, :3] = rgb
    return np.clip(out, 0, 255).astype(np.uint8)


def curves(arr: np.ndarray, shadows: int, midtones: int, highlights: int) -> np.ndarray:
    xs = np.array([0, 64, 128, 192, 255], dtype=np.float32)
    ys = np.array([0, shadows, midtones, highlights, 255], dtype=np.float32)
    lut = np.interp(np.arange(256), xs, ys).clip(0, 255).astype(np.uint8)
    out = arr.copy()
    out[:, :, :3] = lut[out[:, :, :3]]
    return out


def blur(arr: np.ndarray, radius: int) -> np.ndarray:
    k = max(1, radius * 2 + 1)
    out = arr.copy()
    out[:, :, :3] = cv2.GaussianBlur(out[:, :, :3], (k, k), radius)
    return out


def sharpen(arr: np.ndarray, amount: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(arr[:, :, :3], (0, 0), 1.2)
    out = arr.copy()
    out[:, :, :3] = np.clip(arr[:, :, :3].astype(np.float32) * (1 + amount) - blurred.astype(np.float32) * amount, 0, 255)
    return out.astype(np.uint8)


def add_noise(arr: np.ndarray, amount: float) -> np.ndarray:
    out = arr.copy().astype(np.float32)
    noise = np.random.normal(0, amount * 255, out[:, :, :3].shape)
    out[:, :, :3] = np.clip(out[:, :, :3] + noise, 0, 255)
    return out.astype(np.uint8)


def apply_filter_stack(arr: np.ndarray, filters: list[dict[str, Any]]) -> np.ndarray:
    out = arr.copy()
    for item in filters:
        if not bool(item.get("enabled", True)):
            continue
        before = out
        kind = str(item.get("type", "")).lower()
        if kind == "blur":
            filtered = blur(out, int(item.get("radius", 3)))
        elif kind == "sharpen":
            filtered = sharpen(out, float(item.get("amount", 1.0)))
        elif kind == "noise":
            filtered = deterministic_noise(out, float(item.get("amount", 0.03)), int(item.get("seed", 12345)))
        elif kind == "median":
            filtered = median_filter(out, int(item.get("size", 3)))
        elif kind == "edge":
            filtered = edge_filter(out, float(item.get("strength", 1.0)))
        elif kind == "emboss":
            filtered = emboss_filter(out, float(item.get("strength", 1.0)))
        else:
            continue
        opacity = float(np.clip(item.get("opacity", 1.0), 0.0, 1.0))
        blend_mode = str(item.get("blend_mode", "Normal"))
        blend_source = filtered
        if blend_mode != "Normal":
            blend_source = filtered.copy()
            blend_source[:, :, :3] = blend_rgb(filtered[:, :, :3].astype(np.float32), before[:, :, :3].astype(np.float32), blend_mode).clip(0, 255).astype(np.uint8)
        mask = filter_mask_from_item(item, out.shape[:2])
        if opacity >= 0.999:
            if mask is None:
                out = blend_source
            else:
                out = before.copy().astype(np.float32)
                out[:, :, :3] = before[:, :, :3].astype(np.float32) * (1.0 - mask[:, :, None]) + blend_source[:, :, :3].astype(np.float32) * mask[:, :, None]
                out = np.clip(out, 0, 255).astype(np.uint8)
        elif opacity <= 0.001:
            out = before
        else:
            if mask is not None:
                mask = mask * opacity
            else:
                mask = np.full(out.shape[:2], opacity, dtype=np.float32)
            out = before.copy().astype(np.float32)
            out[:, :, :3] = before[:, :, :3].astype(np.float32) * (1.0 - mask[:, :, None]) + blend_source[:, :, :3].astype(np.float32) * mask[:, :, None]
            out = np.clip(out, 0, 255).astype(np.uint8)
    return out


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
    return (mask.astype(np.float32) / 255.0).clip(0, 1)


def median_filter(arr: np.ndarray, size: int) -> np.ndarray:
    k = max(3, int(size) | 1)
    out = arr.copy()
    out[:, :, :3] = cv2.medianBlur(out[:, :, :3], k)
    return out


def deterministic_noise(arr: np.ndarray, amount: float, seed: int = 12345) -> np.ndarray:
    out = arr.copy().astype(np.float32)
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(0, float(amount) * 255, out[:, :, :3].shape)
    out[:, :, :3] = np.clip(out[:, :, :3] + noise, 0, 255)
    return out.astype(np.uint8)


def edge_filter(arr: np.ndarray, strength: float = 1.0) -> np.ndarray:
    gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB).astype(np.float32)
    out = arr.copy().astype(np.float32)
    mix = np.clip(float(strength), 0, 1)
    out[:, :, :3] = out[:, :, :3] * (1.0 - mix) + edge_rgb * mix
    return np.clip(out, 0, 255).astype(np.uint8)


def emboss_filter(arr: np.ndarray, strength: float = 1.0) -> np.ndarray:
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
    embossed = cv2.filter2D(arr[:, :, :3], -1, kernel) + 128
    out = arr.copy().astype(np.float32)
    mix = np.clip(float(strength), 0, 1)
    out[:, :, :3] = out[:, :, :3] * (1.0 - mix) + embossed.astype(np.float32) * mix
    return np.clip(out, 0, 255).astype(np.uint8)


def content_aware_fill(arr: np.ndarray, selection_mask: np.ndarray | None, radius: int = 3) -> np.ndarray:
    if selection_mask is None or not np.any(selection_mask):
        return arr.copy()
    mask = (selection_mask > 0).astype(np.uint8) * 255
    out = arr.copy()
    radius = max(1, int(radius))
    rgb = cv2.inpaint(out[:, :, :3], mask, radius, cv2.INPAINT_TELEA)
    alpha = cv2.inpaint(out[:, :, 3], mask, radius, cv2.INPAINT_TELEA)
    out[:, :, :3] = rgb
    out[:, :, 3] = np.where(mask > 0, np.maximum(alpha, out[:, :, 3]), out[:, :, 3]).astype(np.uint8)
    return out


def _expanded_mask(mask: np.ndarray, left: int, top: int, width: int, height: int) -> np.ndarray:
    result = np.zeros((height, width), dtype=np.uint8)
    result[top : top + mask.shape[0], left : left + mask.shape[1]] = mask
    return result


def generative_expand_pixels(
    arr: np.ndarray,
    left: int,
    top: int,
    right: int,
    bottom: int,
    method: str = "content-aware",
) -> np.ndarray:
    """Expand an image with deterministic local edge synthesis and preserve its center exactly."""
    margins = tuple(max(0, int(value)) for value in (top, bottom, left, right))
    if not any(margins):
        return arr.copy()
    mode = str(method).lower().strip()
    border_mode = cv2.BORDER_REPLICATE if mode in {"edge", "extend"} else cv2.BORDER_REFLECT_101
    expanded = cv2.copyMakeBorder(arr, *margins, borderType=border_mode)
    if mode in {"content-aware", "generative", "texture"}:
        outside = np.full(expanded.shape[:2], 255, dtype=np.uint8)
        outside[top : top + arr.shape[0], left : left + arr.shape[1]] = 0
        # Multi-scale local synthesis removes obvious mirrored repetitions while
        # retaining edge colour and texture. The original image is restored below.
        small = cv2.resize(expanded[:, :, :3], None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        texture = cv2.resize(small, (expanded.shape[1], expanded.shape[0]), interpolation=cv2.INTER_CUBIC)
        detail = expanded[:, :, :3].astype(np.float32) - cv2.GaussianBlur(expanded[:, :, :3], (0, 0), 3.0).astype(np.float32)
        synthesized = np.clip(texture.astype(np.float32) + detail * 0.65, 0, 255).astype(np.uint8)
        feather = cv2.GaussianBlur(outside, (0, 0), 6.0).astype(np.float32)[:, :, None] / 255.0
        expanded[:, :, :3] = np.clip(
            expanded[:, :, :3].astype(np.float32) * (1.0 - feather) + synthesized.astype(np.float32) * feather,
            0,
            255,
        ).astype(np.uint8)
        if np.any(arr[:, :, 3]):
            expanded[:, :, 3] = np.where(outside > 0, np.maximum(expanded[:, :, 3], 255), expanded[:, :, 3])
    expanded[top : top + arr.shape[0], left : left + arr.shape[1]] = arr
    return np.ascontiguousarray(expanded)


def frequency_separation(arr: np.ndarray, radius: float = 8.0, texture_strength: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    radius = max(0.5, float(radius))
    texture_strength = max(0.0, float(texture_strength))
    low = arr.copy()
    low_rgb = cv2.GaussianBlur(arr[:, :, :3], (0, 0), radius).astype(np.float32)
    low[:, :, :3] = np.clip(low_rgb, 0, 255).astype(np.uint8)
    detail = (arr[:, :, :3].astype(np.float32) - low_rgb) * texture_strength
    high = arr.copy()
    high[:, :, :3] = np.clip(127.5 + detail * 0.5, 0, 255).astype(np.uint8)
    return low, high


def portrait_cleanup(
    arr: np.ndarray,
    smoothing: float = 0.35,
    texture: float = 0.7,
    even_tone: float = 0.2,
    redness: float = 0.2,
) -> np.ndarray:
    smoothing = float(np.clip(smoothing, 0.0, 1.0))
    texture = float(np.clip(texture, 0.0, 1.5))
    even_tone = float(np.clip(even_tone, 0.0, 1.0))
    redness = float(np.clip(redness, 0.0, 1.0))
    if max(smoothing, even_tone, redness) <= 0.0:
        return arr.copy()

    rgb = arr[:, :, :3]
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    skin_binary = ((cr >= 132) & (cr <= 183) & (cb >= 76) & (cb <= 136) & (arr[:, :, 3] > 0)).astype(np.uint8) * 255
    if not np.any(skin_binary):
        return arr.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_binary = cv2.morphologyEx(skin_binary, cv2.MORPH_CLOSE, kernel)
    skin = cv2.GaussianBlur(skin_binary, (0, 0), 2.0).astype(np.float32) / 255.0
    skin *= skin_binary.astype(np.float32) / 255.0

    source = rgb.astype(np.float32)
    diameter = 9
    smooth = cv2.bilateralFilter(rgb, diameter, 30.0 + smoothing * 70.0, 18.0 + smoothing * 42.0).astype(np.float32)
    base = cv2.GaussianBlur(source, (0, 0), 1.2)
    detail = source - base
    smoothed = smooth + detail * texture
    cleaned = source * (1.0 - smoothing) + smoothed * smoothing

    if even_tone > 0.0:
        tone = cv2.GaussianBlur(cleaned, (0, 0), 5.0 + even_tone * 9.0)
        cleaned = cleaned * (1.0 - even_tone * 0.45) + tone * (even_tone * 0.45)

    if redness > 0.0:
        green_blue = (cleaned[:, :, 1] + cleaned[:, :, 2]) * 0.5
        excess = np.maximum(0.0, cleaned[:, :, 0] - green_blue)
        cleaned[:, :, 0] -= excess * redness * 0.65

    amount = skin[:, :, None]
    out = arr.copy()
    out[:, :, :3] = np.clip(source * (1.0 - amount) + cleaned * amount, 0, 255).astype(np.uint8)
    return out


def edge_aware_cleanup(arr: np.ndarray, selection_mask: np.ndarray | None, radius: int = 3, strength: float = 0.65) -> np.ndarray:
    if selection_mask is None or not np.any(selection_mask):
        return arr.copy()
    radius = max(1, int(radius))
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return arr.copy()
    mask = (selection_mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    outer = cv2.dilate(mask, kernel)
    inner = cv2.erode(mask, kernel)
    edge_band = cv2.subtract(outer, inner)
    if not np.any(edge_band):
        return arr.copy()
    feather = cv2.GaussianBlur(edge_band, (radius * 2 + 1, radius * 2 + 1), radius).astype(np.float32) / 255.0
    feather = np.clip(feather * strength, 0.0, 1.0)
    out = arr.copy().astype(np.float32)
    diameter = max(3, radius * 2 + 1)
    smooth_rgb = cv2.bilateralFilter(arr[:, :, :3], diameter, 24 + radius * 8, 12 + radius * 4).astype(np.float32)
    smooth_alpha = cv2.bilateralFilter(arr[:, :, 3], diameter, 24 + radius * 8, 12 + radius * 4).astype(np.float32)
    alpha = feather[:, :, None]
    out[:, :, :3] = out[:, :, :3] * (1.0 - alpha) + smooth_rgb * alpha
    out[:, :, 3] = out[:, :, 3] * (1.0 - feather) + smooth_alpha * feather
    return np.clip(out, 0, 255).astype(np.uint8)


def reduce_red_eye(arr: np.ndarray, selection_mask: np.ndarray | None = None, strength: float = 0.85) -> np.ndarray:
    out = arr.copy().astype(np.float32)
    rgb = out[:, :, :3]
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    red_mask = (r > 90) & (r > g * 1.35) & (r > b * 1.35) & (arr[:, :, 3] > 0)
    if selection_mask is not None:
        red_mask &= selection_mask > 0
    if not np.any(red_mask):
        return arr.copy()
    replacement = (g[red_mask] + b[red_mask]) * 0.5
    mix = np.clip(float(strength), 0, 1)
    r[red_mask] = r[red_mask] * (1.0 - mix) + replacement * mix
    return np.clip(out, 0, 255).astype(np.uint8)


def image_statistics(arr: np.ndarray) -> dict[str, Any]:
    rgb = arr[:, :, :3].astype(np.float32)
    alpha = arr[:, :, 3]
    stats: dict[str, Any] = {
        "width": int(arr.shape[1]),
        "height": int(arr.shape[0]),
        "opaque_pixels": int(np.count_nonzero(alpha)),
        "transparent_pixels": int(alpha.size - np.count_nonzero(alpha)),
        "channels": {},
        "histogram": {},
    }
    for index, name in enumerate(["red", "green", "blue"]):
        channel = rgb[:, :, index]
        stats["channels"][name] = {
            "min": float(channel.min()),
            "max": float(channel.max()),
            "mean": float(channel.mean()),
            "std": float(channel.std()),
        }
        hist, _ = np.histogram(channel, bins=16, range=(0, 255))
        stats["histogram"][name] = [int(v) for v in hist]
    stats["channels"]["alpha"] = {
        "min": int(alpha.min()),
        "max": int(alpha.max()),
        "mean": float(alpha.mean()),
        "std": float(alpha.std()),
    }
    return stats


def apply_adjustment_layer(out: np.ndarray, layer: Layer, clipping_mask: np.ndarray | None = None) -> None:
    if layer.adjustment is None:
        return
    kind = str(layer.adjustment.get("type", "")).lower()
    if kind == "brightness_contrast":
        adjusted = adjust_brightness_contrast(out, int(layer.adjustment.get("brightness", 0)), float(layer.adjustment.get("contrast", 1.0)))
    elif kind == "saturation":
        adjusted = adjust_saturation(out, float(layer.adjustment.get("saturation", 1.0)))
    elif kind == "vibrance":
        adjusted = adjust_vibrance(out, float(layer.adjustment.get("vibrance", 0.0)), float(layer.adjustment.get("saturation", 1.0)))
    elif kind == "temperature_tint":
        adjusted = adjust_temperature_tint(out, float(layer.adjustment.get("temperature", 0.0)), float(layer.adjustment.get("tint", 0.0)))
    elif kind == "hue_saturation":
        adjusted = adjust_hue_saturation(out, int(layer.adjustment.get("hue", 0)), float(layer.adjustment.get("saturation", 1.0)), int(layer.adjustment.get("lightness", 0)))
    elif kind == "exposure":
        adjusted = adjust_exposure(out, float(layer.adjustment.get("exposure", 0.0)), float(layer.adjustment.get("offset", 0.0)), float(layer.adjustment.get("gamma", 1.0)))
    elif kind == "color_balance":
        adjusted = adjust_color_balance(out, int(layer.adjustment.get("red", 0)), int(layer.adjustment.get("green", 0)), int(layer.adjustment.get("blue", 0)))
    elif kind == "threshold":
        adjusted = adjust_threshold(out, int(layer.adjustment.get("threshold", 128)))
    elif kind == "posterize":
        adjusted = adjust_posterize(out, int(layer.adjustment.get("levels", 6)))
    elif kind == "levels":
        adjusted = levels(out, int(layer.adjustment.get("black", 0)), int(layer.adjustment.get("white", 255)), float(layer.adjustment.get("gamma", 1.0)))
    elif kind == "curves":
        adjusted = curves(out, int(layer.adjustment.get("shadows", 64)), int(layer.adjustment.get("midtones", 128)), int(layer.adjustment.get("highlights", 192)))
    elif kind == "invert":
        adjusted = out.copy()
        adjusted[:, :, :3] = 255 - adjusted[:, :, :3]
    elif kind == "grayscale":
        adjusted = out.copy()
        gray = cv2.cvtColor(adjusted[:, :, :3], cv2.COLOR_RGB2GRAY)
        adjusted[:, :, :3] = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        return
    alpha = np.full(out.shape[:2], float(layer.opacity), dtype=np.float32)
    if layer.mask is not None and layer.mask_enabled:
        mask_canvas = np.zeros(out.shape[:2], dtype=np.uint8)
        paste_mask(mask_canvas, effective_layer_mask(layer), layer.x, layer.y)
        mask_alpha = ((1.0 - float(layer.mask_density)) + (mask_canvas.astype(np.float32) / 255.0) * float(layer.mask_density))
        alpha *= mask_alpha
    if clipping_mask is not None:
        alpha *= (clipping_mask.astype(np.float32) / 255.0).clip(0, 1)
    alpha = alpha[:, :, None].clip(0, 1)
    out[:, :, :3] = np.clip(adjusted[:, :, :3].astype(np.float32) * alpha + out[:, :, :3].astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)
