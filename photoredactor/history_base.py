from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any, Protocol

import numpy as np

from .core import Document, render_shape_layer, render_text_layer
from .performance import profiler

class Command(Protocol):
    label: str

    @property
    def memory_bytes(self) -> int:
        ...

    def undo(self, document: Document) -> None:
        ...

    def redo(self, document: Document) -> None:
        ...

@dataclass
class TilePatch:
    rect: tuple[int, int, int, int]
    before: np.ndarray
    after: np.ndarray

    @property
    def memory_bytes(self) -> int:
        return int(self.before.nbytes + self.after.nbytes)

@dataclass
class PixelTilePatchCommand:
    label: str
    layer_id: str
    patches: list[TilePatch]
    precision_patches: list[TilePatch] | None = None

    @property
    def memory_bytes(self) -> int:
        return sum(patch.memory_bytes for patch in self.patches) + sum(
            patch.memory_bytes for patch in (self.precision_patches or [])
        )

    @property
    def dirty_rects(self) -> list[tuple[int, int, int, int]]:
        return [patch.rect for patch in self.patches]

    def _apply(self, document: Document, use_after: bool) -> None:
        layer = document.get_layer(self.layer_id)
        if layer is None:
            return
        if self.precision_patches and layer.working_pixels is not None:
            working = layer.working_rgba()
            for patch in self.precision_patches:
                x1, y1, x2, y2 = patch.rect
                working[y1:y2, x1:x2] = patch.after if use_after else patch.before
            layer.set_working_rgba(working, layer.working_depth, layer.working_model)
        else:
            for patch in self.patches:
                x1, y1, x2, y2 = patch.rect
                layer.pixels[y1:y2, x1:x2] = patch.after if use_after else patch.before
            layer.touch_pixels()
        document.dirty = True

    def undo(self, document: Document) -> None:
        self._apply(document, False)

    def redo(self, document: Document) -> None:
        self._apply(document, True)

@dataclass
class MaskTilePatchCommand:
    label: str
    layer_id: str
    patches: list[TilePatch]

    @property
    def memory_bytes(self) -> int:
        return sum(patch.memory_bytes for patch in self.patches)

    @property
    def dirty_rects(self) -> list[tuple[int, int, int, int]]:
        return [patch.rect for patch in self.patches]

    def _apply(self, document: Document, use_after: bool) -> None:
        layer = document.get_layer(self.layer_id)
        if layer is None:
            return
        if layer.mask is None:
            layer.mask = np.full(layer.pixels.shape[:2], 255, dtype=np.uint8)
        for patch in self.patches:
            x1, y1, x2, y2 = patch.rect
            layer.mask[y1:y2, x1:x2] = patch.after if use_after else patch.before
        layer.touch_mask()
        document.dirty = True

    def undo(self, document: Document) -> None:
        self._apply(document, False)

    def redo(self, document: Document) -> None:
        self._apply(document, True)

@dataclass
class LayerPropertyCommand:
    label: str
    layer_id: str
    attribute: str
    before: Any
    after: Any

    @property
    def memory_bytes(self) -> int:
        return max(64, _value_memory(self.before) + _value_memory(self.after))

    def _apply(self, document: Document, value: Any) -> None:
        layer = document.get_layer(self.layer_id)
        if layer is None:
            return
        setattr(layer, self.attribute, copy.deepcopy(value))
        if self.attribute == "mask":
            layer.touch_mask()
        document.dirty = True

    def undo(self, document: Document) -> None:
        self._apply(document, self.before)

    def redo(self, document: Document) -> None:
        self._apply(document, self.after)

@dataclass
class LayerFieldsCommand:
    label: str
    layer_id: str
    before: dict[str, Any]
    after: dict[str, Any]

    @property
    def memory_bytes(self) -> int:
        return max(64, _value_memory(self.before) + _value_memory(self.after))

    def _apply(self, document: Document, values: dict[str, Any]) -> None:
        layer = document.get_layer(self.layer_id)
        if layer is None:
            return
        for attribute, value in values.items():
            setattr(layer, attribute, copy.deepcopy(value))
        if "pixels" in values:
            layer.touch_pixels()
        if "mask" in values or "mask_feather" in values:
            layer.touch_mask()
        document.dirty = True

    def undo(self, document: Document) -> None:
        self._apply(document, self.before)

    def redo(self, document: Document) -> None:
        self._apply(document, self.after)

@dataclass
class TextDataCommand:
    label: str
    layer_id: str
    before: dict[str, Any]
    after: dict[str, Any]
    before_name: str
    after_name: str

    @property
    def memory_bytes(self) -> int:
        return max(128, _value_memory(self.before) + _value_memory(self.after))

    def _apply(self, document: Document, data: dict[str, Any], name: str) -> None:
        layer = document.get_layer(self.layer_id)
        if layer is None:
            return
        layer.text_data = copy.deepcopy(data)
        layer.name = name
        render_text_layer(layer)
        layer.touch_pixels()
        document.dirty = True

    def undo(self, document: Document) -> None:
        self._apply(document, self.before, self.before_name)

    def redo(self, document: Document) -> None:
        self._apply(document, self.after, self.after_name)

@dataclass
class ShapeDataCommand:
    label: str
    layer_id: str
    before: dict[str, Any]
    after: dict[str, Any]
    before_name: str
    after_name: str

    @property
    def memory_bytes(self) -> int:
        return max(128, _value_memory(self.before) + _value_memory(self.after))

    def _apply(self, document: Document, data: dict[str, Any], name: str) -> None:
        layer = document.get_layer(self.layer_id)
        if layer is None:
            return
        layer.shape_data = copy.deepcopy(data)
        layer.name = name
        render_shape_layer(layer)
        layer.touch_pixels()
        document.dirty = True

    def undo(self, document: Document) -> None:
        self._apply(document, self.before, self.before_name)

    def redo(self, document: Document) -> None:
        self._apply(document, self.after, self.after_name)

@dataclass
class DocumentFieldsCommand:
    label: str
    before: dict[str, Any]
    after: dict[str, Any]

    @property
    def memory_bytes(self) -> int:
        return max(64, _value_memory(self.before) + _value_memory(self.after))

    def _apply(self, document: Document, values: dict[str, Any]) -> None:
        for attribute, value in values.items():
            setattr(document, attribute, copy.deepcopy(value))
        document.dirty = True

    def undo(self, document: Document) -> None:
        self._apply(document, self.before)

    def redo(self, document: Document) -> None:
        self._apply(document, self.after)

@dataclass
class LayerInsertCommand:
    label: str
    index: int
    layer: Any

    @property
    def memory_bytes(self) -> int:
        return _value_memory(self.layer.pixels) + _value_memory(self.layer.mask) + 512

    def undo(self, document: Document) -> None:
        document.layers = [item for item in document.layers if item.id != self.layer.id]
        document.active_layer = min(max(0, self.index - 1), max(0, len(document.layers) - 1))
        document.dirty = True

    def redo(self, document: Document) -> None:
        if document.get_layer(self.layer.id) is None:
            document.layers.insert(min(self.index, len(document.layers)), copy.deepcopy(self.layer))
        document.active_layer = min(self.index, len(document.layers) - 1)
        document.dirty = True

@dataclass
class LayerDeleteCommand:
    label: str
    index: int
    layer: Any

    @property
    def memory_bytes(self) -> int:
        return _value_memory(self.layer.pixels) + _value_memory(self.layer.mask) + 512

    def undo(self, document: Document) -> None:
        if document.get_layer(self.layer.id) is None:
            document.layers.insert(min(self.index, len(document.layers)), copy.deepcopy(self.layer))
        document.active_layer = min(self.index, len(document.layers) - 1)
        document.dirty = True

    def redo(self, document: Document) -> None:
        document.layers = [item for item in document.layers if item.id != self.layer.id]
        document.active_layer = min(max(0, self.index - 1), max(0, len(document.layers) - 1))
        document.dirty = True

@dataclass
class LayersDeleteCommand:
    label: str
    layers: list[tuple[int, Any]]
    active_layer_id: str | None = None

    @property
    def memory_bytes(self) -> int:
        return sum(_value_memory(layer.pixels) + _value_memory(layer.mask) + 512 for _, layer in self.layers)

    def undo(self, document: Document) -> None:
        for index, layer in sorted(self.layers, key=lambda item: item[0]):
            if document.get_layer(layer.id) is None:
                document.layers.insert(min(index, len(document.layers)), copy.deepcopy(layer))
        active = document.get_layer(self.active_layer_id) if self.active_layer_id else None
        document.active_layer = document.layers.index(active) if active is not None else min(document.active_layer, len(document.layers) - 1)
        document.dirty = True

    def redo(self, document: Document) -> None:
        deleted_ids = {layer.id for _, layer in self.layers}
        document.layers = [layer for layer in document.layers if layer.id not in deleted_ids]
        document.active_layer = min(document.active_layer, max(0, len(document.layers) - 1))
        document.dirty = True

@dataclass
class LayerReorderCommand:
    label: str
    layer_id: str
    before: int
    after: int

    @property
    def memory_bytes(self) -> int:
        return 96

    @staticmethod
    def _move(document: Document, layer_id: str, index: int) -> None:
        current = next((i for i, layer in enumerate(document.layers) if layer.id == layer_id), None)
        if current is None:
            return
        layer = document.layers.pop(current)
        index = max(0, min(index, len(document.layers)))
        document.layers.insert(index, layer)
        document.active_layer = index
        document.dirty = True

    def undo(self, document: Document) -> None:
        self._move(document, self.layer_id, self.before)

    def redo(self, document: Document) -> None:
        self._move(document, self.layer_id, self.after)

def _value_memory(value: Any) -> int:
    if value is None:
        return 0
    if hasattr(value, "nbytes"):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(_value_memory(key) + _value_memory(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return sum(_value_memory(item) for item in value)
    return len(str(value).encode("utf-8"))

__all__ = [name for name in globals() if not name.startswith("__")]
