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


class SmartObjectsDocumentMixin:
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
        if Path(path).suffix.lower() == ".prdx":
            return self.place_project(path, linked)
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

    def place_document(self, document: "Document", name: str = "Вложенный документ", source_path: str | Path | None = None, linked: bool = False) -> Layer:
        rendered = document.composite(False)
        height, width = rendered.shape[:2]
        path = None if source_path is None else str(Path(source_path).resolve())
        data: dict[str, Any] = {
            "linked": bool(linked and path),
            "content_type": "document",
            "nested_document": document.snapshot(),
            "source_path": path,
            "original_size": [width, height],
            "transform": {"width": width, "height": height, "angle": 0.0, "flip_horizontal": False, "flip_vertical": False},
        }
        if path and Path(path).exists():
            data["fingerprint"] = file_fingerprint(path)
        layer = Layer(
            name=name,
            pixels=rendered.copy(),
            kind="linked" if data["linked"] else "embedded",
            smart_data=data,
            smart_source=rendered.copy(),
        )
        self.layers.append(layer)
        self.active_layer = len(self.layers) - 1
        self.dirty = True
        return layer

    def place_project(self, path: str | Path, linked: bool = False) -> Layer:
        nested = type(self).open_project(path)
        return self.place_document(nested, Path(path).stem, path, linked)

    def convert_layers_to_smart_object(self, layer_ids: list[str] | set[str] | None = None) -> Layer | None:
        selected = set(layer_ids or [self.layer.id])
        indices = [index for index, layer in enumerate(self.layers) if layer.id in selected]
        if not indices:
            return None
        nested_layers: list[Layer] = []
        for index in indices:
            clone = self.layers[index].clone()
            clone.name = self.layers[index].name
            clone.id = self.layers[index].id
            nested_layers.append(clone)
        nested = type(self)(
            width=self.width,
            height=self.height,
            dpi=self.dpi,
            color_model=self.color_model,
            bit_depth=self.bit_depth,
            background=(0, 0, 0, 0),
            layers=nested_layers,
            active_layer=max(0, len(nested_layers) - 1),
            metadata={"source": "nested smart object"},
        )
        rendered = nested.composite(False)
        insert_at = min(indices)
        name = self.layers[max(indices)].name if len(indices) == 1 else f"Smart Object ({len(indices)} слоя)"
        for index in reversed(indices):
            del self.layers[index]
        height, width = rendered.shape[:2]
        smart = Layer(
            name=name,
            pixels=rendered.copy(),
            kind="embedded",
            smart_source=rendered.copy(),
            smart_data={
                "linked": False,
                "content_type": "document",
                "nested_document": nested.snapshot(),
                "original_size": [width, height],
                "transform": {"width": width, "height": height, "angle": 0.0, "flip_horizontal": False, "flip_vertical": False},
            },
        )
        self.layers.insert(insert_at, smart)
        self.active_layer = insert_at
        self.dirty = True
        return smart

    def active_smart_document(self) -> "Document" | None:
        layer = self.layer
        data = layer.smart_data or {}
        nested = data.get("nested_document")
        if layer.kind not in {"linked", "embedded"} or not isinstance(nested, dict):
            return None
        return type(self).restore(nested)

    @staticmethod
    def _prepare_smart_source_replacement(layer: Layer) -> dict[str, Any] | None:
        data = layer.transform_data
        if not data:
            return None
        base_x = float(data.get("base_x", layer.x))
        base_y = float(data.get("base_y", layer.y))
        saved = {
            "mode": str(data.get("mode", "perspective")),
            "points": [[float(point[0]) + base_x, float(point[1]) + base_y] for point in data.get("points", [])],
            "rows": int(data.get("rows", 4)),
            "columns": int(data.get("columns", 4)),
        }
        layer.x = int(data.get("render_base_x", base_x))
        layer.y = int(data.get("render_base_y", base_y))
        layer.mask = None if bool(data.get("mask_was_none", False)) else (None if layer.transform_mask_source is None else layer.transform_mask_source.copy())
        layer.transform_data = None
        layer.transform_source = None
        layer.transform_mask_source = None
        return saved

    def _restore_smart_advanced_transform(self, saved: dict[str, Any] | None) -> None:
        if saved and saved.get("points"):
            self.set_active_layer_advanced_transform(saved["mode"], saved["points"], saved["rows"], saved["columns"])

    def update_active_smart_document(self, document: "Document") -> bool:
        layer = self.layer
        if layer.kind not in {"linked", "embedded"}:
            return False
        advanced = self._prepare_smart_source_replacement(layer)
        rendered = document.composite(False)
        height, width = rendered.shape[:2]
        layer.smart_source = rendered.copy()
        layer.smart_data = {
            **(layer.smart_data or {}),
            "content_type": "document",
            "nested_document": document.snapshot(),
            "original_size": [width, height],
        }
        render_smart_object(layer)
        self._restore_smart_advanced_transform(advanced)
        layer.touch_pixels()
        self.dirty = True
        return True

    def linked_layer_status(self, layer: Layer | None = None) -> dict[str, Any]:
        layer = layer or self.layer
        data = layer.smart_data or {}
        path = data.get("source_path")
        if layer.kind != "linked" or not path:
            return {"status": "embedded", "path": path}
        if not Path(path).exists():
            return {"status": "missing", "path": path}
        saved = data.get("fingerprint") or {}
        current = file_fingerprint(path)
        modified = bool(saved) and (current.get("sha256") != saved.get("sha256"))
        return {"status": "modified" if modified else "current", "path": path, "saved": saved, "current": current}

    def update_linked_layer(self) -> bool:
        layer = self.layer
        smart_data = layer.smart_data or {}
        source_path = smart_data.get("source_path")
        if not source_path or not Path(source_path).exists():
            return False
        advanced = self._prepare_smart_source_replacement(layer)
        if Path(source_path).suffix.lower() == ".prdx" or smart_data.get("content_type") == "document":
            nested = type(self).open_project(source_path)
            pixels = nested.composite(False)
            nested_snapshot = nested.snapshot()
        else:
            image = Image.open(source_path)
            pixels = pil_to_rgba_array(image)
            nested_snapshot = None
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
        if nested_snapshot is not None:
            layer.smart_data["content_type"] = "document"
            layer.smart_data["nested_document"] = nested_snapshot
        render_smart_object(layer)
        self._restore_smart_advanced_transform(advanced)
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
        advanced = self._prepare_smart_source_replacement(layer)
        if Path(path).suffix.lower() == ".prdx":
            nested = type(self).open_project(path)
            pixels = nested.composite(False)
            nested_snapshot = nested.snapshot()
        else:
            pixels = pil_to_rgba_array(Image.open(path))
            nested_snapshot = None
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
        if nested_snapshot is not None:
            layer.smart_data["content_type"] = "document"
            layer.smart_data["nested_document"] = nested_snapshot
        else:
            layer.smart_data.pop("nested_document", None)
            layer.smart_data["content_type"] = "image"
        render_smart_object(layer)
        self._restore_smart_advanced_transform(advanced)
        layer.touch_pixels()
        self.dirty = True
        return True

    def resolve_linked_conflict(self, action: str, path: str | Path | None = None) -> bool:
        layer = self.layer
        if layer.kind != "linked":
            return False
        action = str(action).lower().strip()
        if action == "update":
            return self.update_linked_layer()
        if action == "embed":
            layer.kind = "embedded"
            layer.smart_data = {**(layer.smart_data or {}), "linked": False, "conflict_resolution": "kept_cached"}
            self.dirty = True
            return True
        if action == "relink" and path is not None:
            return self.relink_active_layer(path)
        return False

    def reset_active_smart_transform(self) -> bool:
        layer = self.layer
        if layer.kind not in {"linked", "embedded"} or layer.smart_source is None:
            return False
        center_x = layer.x + layer.pixels.shape[1] / 2.0
        center_y = layer.y + layer.pixels.shape[0] / 2.0
        advanced = layer.transform_data
        if advanced is not None:
            layer.mask = None if bool(advanced.get("mask_was_none", False)) else (None if layer.transform_mask_source is None else layer.transform_mask_source.copy())
            layer.transform_data = None
            layer.transform_source = None
            layer.transform_mask_source = None
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
