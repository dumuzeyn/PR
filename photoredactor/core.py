from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import base64
import io
import json
import uuid
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


_checker_cache: dict[tuple[int, int, int], np.ndarray] = {}
_brush_mask_cache: dict[int, np.ndarray] = {}


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


@dataclass
class Layer:
    name: str
    pixels: np.ndarray
    x: int = 0
    y: int = 0
    opacity: float = 1.0
    visible: bool = True
    locked: bool = False
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

    @classmethod
    def new(cls, width: int = 1280, height: int = 900, background=(255, 255, 255, 255)) -> "Document":
        doc = cls(width=width, height=height, background=background)
        doc.layers.append(Layer("Background", blank_rgba(width, height, background)))
        return doc

    @classmethod
    def from_image(cls, path: str | Path) -> "Document":
        image = Image.open(path)
        arr = pil_to_rgba_array(image)
        h, w = arr.shape[:2]
        doc = cls(width=w, height=h, dpi=image.info.get("dpi", (300, 300))[0] if image.info.get("dpi") else 300)
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
            "layers": [
                {
                    "id": layer.id,
                    "name": layer.name,
                    "x": layer.x,
                    "y": layer.y,
                    "opacity": layer.opacity,
                    "visible": layer.visible,
                    "locked": layer.locked,
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
            "layers": [
                {
                    "name": layer.name,
                    "x": layer.x,
                    "y": layer.y,
                    "opacity": layer.opacity,
                    "visible": layer.visible,
                    "locked": layer.locked,
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
                    id=raw.get("id", uuid.uuid4().hex),
                )
            )
        doc.active_layer = min(int(data.get("active_layer", 0)), max(0, len(doc.layers) - 1))
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
            "layers": [],
        }
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
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
                        "pixels": layer_path,
                    }
                )
                buf = io.BytesIO()
                rgba_array_to_pil(layer.pixels).save(buf, "PNG")
                zf.writestr(layer_path, buf.getvalue())
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
                        id=raw.get("id", uuid.uuid4().hex),
                    )
                )
            doc.active_layer = min(int(manifest.get("active_layer", 0)), max(0, len(doc.layers) - 1))
        doc.path = str(path)
        return doc

    def composite(self, checker: bool = False) -> np.ndarray:
        out = checker_background(self.width, self.height).copy() if checker else blank_rgba(self.width, self.height, (0, 0, 0, 0))
        for layer in self.layers:
            if layer.visible:
                alpha_blend_inplace(out, layer.pixels, layer.x, layer.y, layer.opacity)
        return out

    def export_flat(self, path: str | Path) -> None:
        img = rgba_array_to_pil(self.composite(checker=False))
        suffix = Path(path).suffix.lower()
        if suffix in [".jpg", ".jpeg"]:
            img.convert("RGB").save(path, quality=95, subsampling=0)
        else:
            img.save(path)
        self.dirty = False

    def add_layer(self, name="Layer", pixels: np.ndarray | None = None) -> None:
        if pixels is None:
            pixels = blank_rgba(self.width, self.height, (0, 0, 0, 0))
        self.layers.append(Layer(name, pixels))
        self.active_layer = len(self.layers) - 1
        self.dirty = True

    def delete_active_layer(self) -> None:
        if len(self.layers) <= 1:
            return
        del self.layers[self.active_layer]
        self.active_layer = max(0, self.active_layer - 1)
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
        alpha_blend_inplace(merged, lower.pixels, lower.x - min_x, lower.y - min_y, lower.opacity)
        alpha_blend_inplace(merged, upper.pixels, upper.x - min_x, upper.y - min_y, upper.opacity)
        lower.pixels = merged
        lower.x = min_x
        lower.y = min_y
        lower.opacity = 1.0
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
            layer.x = round(layer.x * width / self.width)
            layer.y = round(layer.y * height / self.height)
        self.width, self.height = width, height
        self.dirty = True

    def resize_canvas(self, width: int, height: int, anchor="center") -> None:
        dx = (width - self.width) // 2 if anchor == "center" else 0
        dy = (height - self.height) // 2 if anchor == "center" else 0
        for layer in self.layers:
            layer.x += dx
            layer.y += dy
        self.width, self.height = width, height
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
            layer.pixels = canvas[y1:y2, x1:x2].copy()
            layer.x = 0
            layer.y = 0
        self.width, self.height = new_w, new_h
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


def alpha_blend(dst: np.ndarray, src: np.ndarray, x: int, y: int, opacity: float) -> np.ndarray:
    out = dst.copy()
    alpha_blend_inplace(out, src, x, y, opacity)
    return out


def alpha_blend_inplace(dst: np.ndarray, src: np.ndarray, x: int, y: int, opacity: float) -> None:
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
    da = d[:, :, 3:4] / 255.0
    oa = sa + da * (1.0 - sa)
    rgb = np.where(oa > 0, (s[:, :, :3] * sa + d[:, :, :3] * da * (1.0 - sa)) / np.maximum(oa, 1e-6), 0)
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


def draw_brush(layer: Layer, x: int, y: int, radius: int, color: tuple[int, int, int, int], opacity: float = 1.0, erase=False) -> tuple[int, int, int, int] | None:
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


def flood_fill(layer: Layer, x: int, y: int, color: tuple[int, int, int, int], tolerance: int) -> None:
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
    layer.pixels[region] = np.array(color, dtype=np.uint8)


def apply_gradient(layer: Layer, box: tuple[int, int, int, int], start: tuple[int, int, int, int], end: tuple[int, int, int, int]) -> None:
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
    layer.pixels[y1:y2, x1:x2] = np.tile(grad[None, :, :], (y2 - y1, 1, 1)).astype(np.uint8)


def add_text(layer: Layer, x: int, y: int, text: str, color: tuple[int, int, int, int], size: int) -> None:
    if layer.locked:
        return
    pil = rgba_array_to_pil(layer.pixels)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    draw.text((x - layer.x, y - layer.y), text, fill=color, font=font)
    layer.pixels = pil_to_rgba_array(pil)


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
