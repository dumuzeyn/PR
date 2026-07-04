from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import base64
import io
import json
import math
import uuid
import zipfile

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageDraw, ImageFont


_checker_cache: dict[tuple[int, int, int], np.ndarray] = {}
_brush_mask_cache: dict[int, np.ndarray] = {}
BLEND_MODES = [
    "Normal",
    "Multiply",
    "Screen",
    "Overlay",
    "Soft Light",
    "Darken",
    "Lighten",
    "Difference",
    "Color",
    "Luminosity",
]


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
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

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
            mask_density=self.mask_density,
            mask_feather=self.mask_feather,
            blend_mode=self.blend_mode,
            clipping=self.clipping,
            effects=json.loads(json.dumps(self.effects)),
            filters=json.loads(json.dumps(self.filters)),
            kind=self.kind,
            text_data=None if self.text_data is None else dict(self.text_data),
            shape_data=None if self.shape_data is None else dict(self.shape_data),
            adjustment=None if self.adjustment is None else dict(self.adjustment),
            smart_data=None if self.smart_data is None else json.loads(json.dumps(self.smart_data, ensure_ascii=False)),
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
        image = Image.open(path)
        arr = pil_to_rgba_array(image)
        h, w = arr.shape[:2]
        dpi = image.info.get("dpi", (300, 300))[0] if image.info.get("dpi") else 300
        doc = cls(width=w, height=h, dpi=dpi, metadata=image_metadata(image, path))
        doc.layers.append(Layer(Path(path).stem, arr))
        doc.path = str(path)
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
                    "mask_density": layer.mask_density,
                    "mask_feather": layer.mask_feather,
                    "blend_mode": layer.blend_mode,
                    "clipping": layer.clipping,
                    "effects": json.loads(json.dumps(layer.effects)),
                    "filters": json.loads(json.dumps(layer.filters)),
                    "kind": layer.kind,
                    "text_data": None if layer.text_data is None else dict(layer.text_data),
                    "shape_data": None if layer.shape_data is None else dict(layer.shape_data),
                    "adjustment": None if layer.adjustment is None else dict(layer.adjustment),
                    "smart_data": None if layer.smart_data is None else json.loads(json.dumps(layer.smart_data, ensure_ascii=False)),
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
                        "pixels": layer_path,
                    }
                )
                buf = io.BytesIO()
                rgba_array_to_pil(layer.pixels).save(buf, "PNG")
                zf.writestr(layer_path, buf.getvalue())
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
        out = checker_background(self.width, self.height).copy() if checker else blank_rgba(self.width, self.height, (0, 0, 0, 0))
        previous_alpha: np.ndarray | None = None
        for layer in self.layers:
            if layer.visible:
                if layer.kind == "adjustment" and layer.adjustment is not None:
                    apply_adjustment_layer(out, layer)
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
        img = rgba_array_to_pil(self.composite(checker=False))
        suffix = Path(path).suffix.lower()
        if suffix in [".jpg", ".jpeg"]:
            img.convert("RGB").save(path, quality=max(1, min(100, int(quality))), subsampling=0)
        elif suffix == ".webp":
            img.save(path, quality=max(1, min(100, int(quality))))
        else:
            img.save(path)
        self.dirty = False

    def add_layer(self, name="Layer", pixels: np.ndarray | None = None) -> None:
        if pixels is None:
            pixels = blank_rgba(self.width, self.height, (0, 0, 0, 0))
        self.layers.append(Layer(name, pixels))
        self.active_layer = len(self.layers) - 1
        self.dirty = True

    def place_image(self, path: str | Path, linked: bool = False) -> None:
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
            },
        )
        self.layers.append(layer)
        self.active_layer = len(self.layers) - 1
        embedded = list(self.metadata.get("embedded_images", []))
        embedded.append({"name": layer.name, "source_path": source_path, "size": [w, h], "linked": bool(linked)})
        self.metadata["embedded_images"] = embedded
        self.dirty = True

    def update_linked_layer(self) -> bool:
        layer = self.layer
        smart_data = layer.smart_data or {}
        source_path = smart_data.get("source_path")
        if not source_path or not Path(source_path).exists():
            return False
        image = Image.open(source_path)
        pixels = pil_to_rgba_array(image)
        source_h, source_w = pixels.shape[:2]
        target_h, target_w = layer.pixels.shape[:2]
        if (source_w, source_h) != (target_w, target_h):
            pixels = cv2.resize(pixels, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        layer.pixels = pixels
        layer.kind = "linked"
        layer.smart_data = {
            **smart_data,
            "linked": True,
            "source_path": str(Path(source_path).resolve()),
            "original_size": [source_w, source_h],
        }
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
    ) -> None:
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
            },
        )
        render_text_layer(layer)
        self.layers.append(layer)
        self.active_layer = len(self.layers) - 1
        self.dirty = True

    def add_shape_layer(
        self,
        shape: str,
        box: tuple[int, int, int, int],
        fill: tuple[int, int, int, int],
        stroke: tuple[int, int, int, int] | None = None,
        stroke_width: int = 0,
        sides: int = 5,
        inner_ratio: float = 0.5,
    ) -> None:
        layer = Layer(
            name=f"{shape.title()} shape",
            pixels=blank_rgba(self.width, self.height, (0, 0, 0, 0)),
            kind="shape",
            shape_data={
                "shape": shape,
                "box": [int(v) for v in normalized_box(box)],
                "fill": list(fill),
                "stroke": None if stroke is None else list(stroke),
                "stroke_width": int(stroke_width),
                "sides": int(sides),
                "inner_ratio": float(inner_ratio),
            },
        )
        render_shape_layer(layer)
        self.layers.append(layer)
        self.active_layer = len(self.layers) - 1
        self.dirty = True

    def edit_shape_layer(
        self,
        shape: str | None = None,
        fill: tuple[int, int, int, int] | None = None,
        stroke: tuple[int, int, int, int] | None = None,
        stroke_width: int | None = None,
        sides: int | None = None,
        inner_ratio: float | None = None,
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
        render_shape_layer(layer)
        self.dirty = True

    def edit_text_layer(
        self,
        text: str | None = None,
        size: int | None = None,
        color: tuple[int, int, int, int] | None = None,
        font_family: str | None = None,
        box_width: int | None = None,
        align: str | None = None,
        line_spacing: int | None = None,
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
        render_text_layer(layer)
        self.dirty = True

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
            layer.pixels = cv2.resize(layer.pixels, (new_w, new_h), interpolation=interpolation)
            if layer.mask is not None:
                layer.mask = cv2.resize(layer.mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            layer.x = round(layer.x * width / self.width)
            layer.y = round(layer.y * height / self.height)
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

    def set_rect_selection(self, box: tuple[int, int, int, int], mode: str = "replace") -> None:
        x1, y1, x2, y2 = normalized_box(box)
        x1, x2 = max(0, min(self.width, x1)), max(0, min(self.width, x2))
        y1, y2 = max(0, min(self.height, y1)), max(0, min(self.height, y2))
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        if x1 < x2 and y1 < y2:
            mask[y1:y2, x1:x2] = 255
        self.apply_selection_mask(mask, mode)

    def set_ellipse_selection(self, box: tuple[int, int, int, int], mode: str = "replace") -> None:
        x1, y1, x2, y2 = normalized_box(box)
        x1, x2 = max(0, min(self.width, x1)), max(0, min(self.width, x2))
        y1, y2 = max(0, min(self.height, y1)), max(0, min(self.height, y2))
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        if x1 < x2 and y1 < y2:
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        self.apply_selection_mask(mask, mode)

    def set_polygon_selection(self, points: list[tuple[int, int]], mode: str = "replace") -> None:
        if len(points) < 3:
            return
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        pts = np.array([[(max(0, min(self.width - 1, int(x))), max(0, min(self.height - 1, int(y)))) for x, y in points]], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255)
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

    def magic_wand_selection(self, layer: Layer, x: int, y: int, tolerance: int, mode: str = "replace") -> None:
        lx, ly = int(x) - layer.x, int(y) - layer.y
        if lx < 0 or ly < 0 or lx >= layer.pixels.shape[1] or ly >= layer.pixels.shape[0]:
            return
        seed = layer.pixels[ly, lx].astype(np.int16)
        diff = np.abs(layer.pixels.astype(np.int16) - seed).max(axis=2)
        candidates = (diff <= int(tolerance)).astype(np.uint8)
        _, labels, _, _ = cv2.connectedComponentsWithStats(candidates, 4)
        local = (labels == labels[ly, lx]).astype(np.uint8) * 255
        self.apply_selection_mask(self._layer_mask_to_document(layer, local), mode)

    def color_range_selection(self, layer: Layer, x: int, y: int, tolerance: int, mode: str = "replace") -> None:
        lx, ly = int(x) - layer.x, int(y) - layer.y
        if lx < 0 or ly < 0 or lx >= layer.pixels.shape[1] or ly >= layer.pixels.shape[0]:
            return
        seed = layer.pixels[ly, lx].astype(np.int16)
        diff = np.abs(layer.pixels.astype(np.int16) - seed).max(axis=2)
        local = (diff <= int(tolerance)).astype(np.uint8) * 255
        self.apply_selection_mask(self._layer_mask_to_document(layer, local), mode)

    def quick_selection_brush(self, layer: Layer, points: list[tuple[int, int]], radius: int, tolerance: int, mode: str = "replace") -> None:
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
            diff = np.abs(layer.pixels[:, :, :3].astype(np.float32) - seed).max(axis=2)
            candidates = ((diff <= tolerance) & (layer.pixels[:, :, 3] > 0)).astype(np.uint8)
            _, labels, _, _ = cv2.connectedComponentsWithStats(candidates, 4)
            label = labels[ly, lx]
            if label == 0 and candidates[ly, lx] == 0:
                continue
            component = (labels == label).astype(np.uint8) * 255
            brush_gate = np.zeros_like(component)
            cv2.circle(brush_gate, (lx, ly), radius * 3, 255, -1)
            local_union = np.maximum(local_union, np.where(brush_gate > 0, component, 0).astype(np.uint8))
        if np.any(local_union):
            self.apply_selection_mask(self._layer_mask_to_document(layer, local_union), mode)

    def select_opaque_pixels(self, layer: Layer, mode: str = "replace") -> None:
        local = (layer.pixels[:, :, 3] > 0).astype(np.uint8) * 255
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
            self.selection_mask = np.where(mask > 0, 0, self.selection_mask).astype(np.uint8)
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
        mask = self.selection_mask.copy()
        if shift > 0:
            kernel = np.ones((shift * 2 + 1, shift * 2 + 1), dtype=np.uint8)
            mask = cv2.dilate(mask, kernel)
        elif shift < 0:
            amount = abs(int(shift))
            kernel = np.ones((amount * 2 + 1, amount * 2 + 1), dtype=np.uint8)
            mask = cv2.erode(mask, kernel)
        if smooth > 0:
            k = int(smooth) * 2 + 1
            mask = cv2.GaussianBlur(mask, (k, k), smooth)
            mask = np.where(mask >= 128, 255, 0).astype(np.uint8)
        if feather > 0:
            k = int(feather) * 2 + 1
            mask = cv2.GaussianBlur(mask, (k, k), feather)
        if abs(float(contrast) - 1.0) > 0.001:
            work = mask.astype(np.float32)
            work = (work - 127.5) * max(0.0, float(contrast)) + 127.5
            mask = np.clip(work, 0, 255).astype(np.uint8)
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

    def add_hide_all_mask(self) -> None:
        layer = self.layer
        layer.mask = np.zeros(layer.pixels.shape[:2], dtype=np.uint8)
        layer.mask_enabled = True

    def add_mask_from_selection(self) -> None:
        layer = self.layer
        mask = self.layer_selection_mask(layer)
        if mask is None:
            self.add_reveal_all_mask()
        else:
            layer.mask = mask
            layer.mask_enabled = True

    def invert_active_mask(self) -> None:
        layer = self.layer
        if layer.mask is not None:
            layer.mask = 255 - layer.mask

    def toggle_active_mask(self) -> None:
        layer = self.layer
        if layer.mask is not None:
            layer.mask_enabled = not layer.mask_enabled

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

    def delete_active_mask(self) -> None:
        self.layer.mask = None

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


def normalized_box(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


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
    if selection_mask is not None:
        mask = mask & (selection_mask[y1:y2, x1:x2] > 0)
        if not np.any(mask):
            return None
    target = layer.pixels[y1:y2, x1:x2]
    if erase:
        target[mask, 3] = np.clip(target[mask, 3].astype(np.float32) * (1.0 - opacity), 0, 255)
    else:
        paint = np.array(color, dtype=np.float32)
        paint[3] *= opacity
        dst = target[mask].astype(np.float32)
        sa = paint[3] / 255.0
        da = dst[:, 3] / 255.0
        oa = sa + da * (1.0 - sa)
        dst[:, :3] = np.where(oa[:, None] > 0, (paint[:3] * sa + dst[:, :3] * da[:, None] * (1.0 - sa)) / np.maximum(oa[:, None], 1e-6), 0)
        dst[:, 3] = oa * 255
        target[mask] = np.clip(dst, 0, 255).astype(np.uint8)
    return x1, y1, x2, y2


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
    if selection_mask is not None:
        mask = mask & (selection_mask[y1:y2, x1:x2] > 0)
        if not np.any(mask):
            return None
    target = layer.mask[y1:y2, x1:x2].astype(np.float32)
    target[mask] = target[mask] * (1.0 - opacity) + int(value) * opacity
    layer.mask[y1:y2, x1:x2] = np.clip(target, 0, 255).astype(np.uint8)
    return x1, y1, x2, y2


def local_retouch(layer: Layer, x: int, y: int, radius: int, mode: str, opacity: float = 1.0, selection_mask: np.ndarray | None = None) -> tuple[int, int, int, int] | None:
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
    if selection_mask is not None:
        mask = mask & (selection_mask[y1:y2, x1:x2] > 0)
        if not np.any(mask):
            return None
    patch = layer.pixels[y1:y2, x1:x2].copy()
    edited = patch.copy().astype(np.float32)
    if mode == "blur":
        k = max(3, radius // 2 * 2 + 1)
        edited[:, :, :3] = cv2.GaussianBlur(patch[:, :, :3], (k, k), max(1, radius / 3))
    elif mode == "sharpen":
        blurred = cv2.GaussianBlur(patch[:, :, :3], (0, 0), 1.2)
        edited[:, :, :3] = np.clip(patch[:, :, :3].astype(np.float32) * 1.8 - blurred.astype(np.float32) * 0.8, 0, 255)
    elif mode == "dodge":
        edited[:, :, :3] = np.clip(patch[:, :, :3].astype(np.float32) + 45, 0, 255)
    elif mode == "burn":
        edited[:, :, :3] = np.clip(patch[:, :, :3].astype(np.float32) - 45, 0, 255)
    else:
        return None
    target = layer.pixels[y1:y2, x1:x2].astype(np.float32)
    mix = np.clip(float(opacity), 0, 1)
    target[mask] = target[mask] * (1.0 - mix) + edited[mask] * mix
    layer.pixels[y1:y2, x1:x2] = np.clip(target, 0, 255).astype(np.uint8)
    return x1, y1, x2, y2


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
) -> tuple[int, int, int, int] | None:
    if layer.locked:
        return None
    radius = max(1, int(radius))
    sx, sy = int(source_x) - layer.x, int(source_y) - layer.y
    tx, ty = int(target_x) - layer.x, int(target_y) - layer.y
    x1 = max(0, tx - radius)
    y1 = max(0, ty - radius)
    x2 = min(layer.pixels.shape[1], tx + radius + 1)
    y2 = min(layer.pixels.shape[0], ty + radius + 1)
    if x1 >= x2 or y1 >= y2:
        return None
    sx1 = sx + (x1 - tx)
    sy1 = sy + (y1 - ty)
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)
    if sx1 < 0 or sy1 < 0 or sx2 > layer.pixels.shape[1] or sy2 > layer.pixels.shape[0]:
        return None
    full_mask = brush_mask(radius)
    mx1 = x1 - (tx - radius)
    my1 = y1 - (ty - radius)
    mask = full_mask[my1 : my1 + (y2 - y1), mx1 : mx1 + (x2 - x1)]
    if selection_mask is not None:
        mask = mask & (selection_mask[y1:y2, x1:x2] > 0)
        if not np.any(mask):
            return None
    src = layer.pixels[sy1:sy2, sx1:sx2].astype(np.float32)
    dst = layer.pixels[y1:y2, x1:x2].astype(np.float32)
    edited = src.copy()
    if heal:
        src_mean = src[mask, :3].mean(axis=0) if np.any(mask) else src[:, :, :3].reshape(-1, 3).mean(axis=0)
        dst_mean = dst[mask, :3].mean(axis=0) if np.any(mask) else dst[:, :, :3].reshape(-1, 3).mean(axis=0)
        edited[:, :, :3] = np.clip(src[:, :, :3] - src_mean + dst_mean, 0, 255)
    mix = np.clip(float(opacity), 0, 1)
    dst[mask] = dst[mask] * (1.0 - mix) + edited[mask] * mix
    layer.pixels[y1:y2, x1:x2] = np.clip(dst, 0, 255).astype(np.uint8)
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
        region = region & (selection_mask > 0)
    layer.pixels[region] = np.array(color, dtype=np.uint8)


def apply_gradient(layer: Layer, box: tuple[int, int, int, int], start: tuple[int, int, int, int], end: tuple[int, int, int, int], selection_mask: np.ndarray | None = None) -> None:
    if layer.locked:
        return
    x1, y1, x2, y2 = normalized_box(box)
    x1 -= layer.x
    x2 -= layer.x
    y1 -= layer.y
    y2 -= layer.y
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(layer.pixels.shape[1], x2), min(layer.pixels.shape[0], y2)
    if x1 >= x2 or y1 >= y2:
        return
    width = x2 - x1
    t = np.linspace(0, 1, width, dtype=np.float32)
    grad = np.array(start, dtype=np.float32) * (1 - t[:, None]) + np.array(end, dtype=np.float32) * t[:, None]
    patch = np.tile(grad[None, :, :], (y2 - y1, 1, 1)).astype(np.uint8)
    if selection_mask is None:
        layer.pixels[y1:y2, x1:x2] = patch
    else:
        mask = selection_mask[y1:y2, x1:x2] > 0
        layer.pixels[y1:y2, x1:x2][mask] = patch[mask]


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
        mask = selection_mask > 0
        layer.pixels[mask] = rendered[mask]


def render_text_layer(layer: Layer) -> None:
    if layer.text_data is None:
        return
    layer.pixels[:] = 0
    pil = rgba_array_to_pil(layer.pixels)
    draw = ImageDraw.Draw(pil)
    data = layer.text_data
    font = load_text_font(str(data.get("font_family", "arial.ttf")), int(data.get("size", 48)))
    color = tuple(int(v) for v in data.get("color", [255, 255, 255, 255]))
    x = int(data.get("x", 0))
    y = int(data.get("y", 0))
    size = int(data.get("size", 48))
    box_width = max(0, int(data.get("box_width", 0) or 0))
    spacing = max(0, int(data.get("line_spacing", max(2, size // 5))))
    align = str(data.get("align", "left")).lower()
    lines = wrapped_text_lines(draw, str(data.get("text", "")), font, box_width)
    line_y = y
    for line in lines:
        bbox = draw.textbbox((0, 0), line or " ", font=font)
        line_width = bbox[2] - bbox[0]
        dx = 0
        if box_width > 0 and align == "center":
            dx = max(0, (box_width - line_width) // 2)
        elif box_width > 0 and align == "right":
            dx = max(0, box_width - line_width)
        draw.text((x + dx, line_y), line, fill=color, font=font)
        line_y += max(1, bbox[3] - bbox[1]) + spacing
    layer.pixels = pil_to_rgba_array(pil)


def wrapped_text_lines(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, box_width: int) -> list[str]:
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
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if bbox[2] - bbox[0] <= box_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def load_text_font(font_family: str, size: int) -> ImageFont.ImageFont:
    family = font_family.strip() or "arial.ttf"
    candidates = [family]
    if not Path(family).suffix:
        compact = family.lower().replace(" ", "")
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
    else:
        draw.rectangle(box, fill=fill, outline=outline, width=stroke_width)
    layer.pixels = pil_to_rgba_array(pil)


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
        kind = str(item.get("type", "")).lower()
        if kind == "blur":
            out = blur(out, int(item.get("radius", 3)))
        elif kind == "sharpen":
            out = sharpen(out, float(item.get("amount", 1.0)))
        elif kind == "noise":
            out = deterministic_noise(out, float(item.get("amount", 0.03)), int(item.get("seed", 12345)))
        elif kind == "median":
            out = median_filter(out, int(item.get("size", 3)))
        elif kind == "edge":
            out = edge_filter(out, float(item.get("strength", 1.0)))
        elif kind == "emboss":
            out = emboss_filter(out, float(item.get("strength", 1.0)))
    return out


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


def apply_adjustment_layer(out: np.ndarray, layer: Layer) -> None:
    if layer.adjustment is None:
        return
    kind = str(layer.adjustment.get("type", "")).lower()
    if kind == "brightness_contrast":
        adjusted = adjust_brightness_contrast(out, int(layer.adjustment.get("brightness", 0)), float(layer.adjustment.get("contrast", 1.0)))
    elif kind == "saturation":
        adjusted = adjust_saturation(out, float(layer.adjustment.get("saturation", 1.0)))
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
    alpha = alpha[:, :, None].clip(0, 1)
    out[:, :, :3] = np.clip(adjusted[:, :, :3].astype(np.float32) * alpha + out[:, :, :3].astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)
