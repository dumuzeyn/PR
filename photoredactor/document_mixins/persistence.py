from __future__ import annotations

from ..core_shared import *
from ..layer import Layer
from ..geometry_ops import *
from ..selection_ops import *
from ..filter_ops import *
from ..render_ops import *
from ..retouch_ops import *
from ..text_ops import *
from ..shape_ops import *
from ..content_ops import *
from ..adjustment_ops import *


class PersistenceDocumentMixin:
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
                    "transform_data": None if layer.transform_data is None else json.loads(json.dumps(layer.transform_data)),
                    "transform_source": None if layer.transform_source is None else layer.transform_source.copy(),
                    "transform_mask_source": None if layer.transform_mask_source is None else layer.transform_mask_source.copy(),
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
                    transform_data=raw.get("transform_data"),
                    transform_source=None if raw.get("transform_source") is None else raw["transform_source"].copy(),
                    transform_mask_source=None if raw.get("transform_mask_source") is None else raw["transform_mask_source"].copy(),
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
                    "id": layer.id,
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
                    "transform_data": layer.transform_data,
                    "transform_source": None if layer.transform_source is None else encode_png(layer.transform_source),
                    "transform_mask_source": None if layer.transform_mask_source is None else encode_png(np.dstack([layer.transform_mask_source] * 4)),
                    "working_pixels": None if layer.working_pixels is None else encode_array(layer.working_pixels),
                    "working_model": layer.working_model,
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
                    transform_data=raw.get("transform_data"),
                    transform_source=None if raw.get("transform_source") is None else decode_png(raw["transform_source"]),
                    transform_mask_source=None if raw.get("transform_mask_source") is None else decode_png(raw["transform_mask_source"])[:, :, 0],
                    working_pixels=None if raw.get("working_pixels") is None else decode_array(raw["working_pixels"]),
                    working_model=raw.get("working_model", "RGBA"),
                    id=raw.get("id", uuid.uuid4().hex),
                )
            )
        doc.active_layer = min(int(data.get("active_layer", 0)), max(0, len(doc.layers) - 1))
        if data.get("selection"):
            doc.selection_mask = decode_png(data["selection"])[:, :, 0]
        doc.saved_selections = {name: decode_png(mask)[:, :, 0] for name, mask in data.get("saved_selections", {}).items()}
        doc.metadata = data.get("metadata", {})
        return doc
