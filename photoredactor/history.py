from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any, Protocol

import numpy as np

from .core import Document
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


@dataclass
class PixelPatchCommand:
    label: str
    layer_id: str
    rect: tuple[int, int, int, int]
    before: np.ndarray
    after: np.ndarray

    @property
    def memory_bytes(self) -> int:
        return int(self.before.nbytes + self.after.nbytes)

    @property
    def dirty_rects(self) -> list[tuple[int, int, int, int]]:
        return [self.rect]

    def undo(self, document: Document) -> None:
        layer = document.get_layer(self.layer_id)
        if layer is None:
            return
        x1, y1, x2, y2 = self.rect
        layer.pixels[y1:y2, x1:x2] = self.before
        layer.touch_pixels()
        document.dirty = True

    def redo(self, document: Document) -> None:
        layer = document.get_layer(self.layer_id)
        if layer is None:
            return
        x1, y1, x2, y2 = self.rect
        layer.pixels[y1:y2, x1:x2] = self.after
        layer.touch_pixels()
        document.dirty = True


@dataclass
class MaskPatchCommand:
    label: str
    layer_id: str
    rect: tuple[int, int, int, int]
    before: np.ndarray
    after: np.ndarray

    @property
    def memory_bytes(self) -> int:
        return int(self.before.nbytes + self.after.nbytes)

    @property
    def dirty_rects(self) -> list[tuple[int, int, int, int]]:
        return [self.rect]

    def undo(self, document: Document) -> None:
        layer = document.get_layer(self.layer_id)
        if layer is None or layer.mask is None:
            return
        x1, y1, x2, y2 = self.rect
        layer.mask[y1:y2, x1:x2] = self.before
        layer.touch_mask()
        document.dirty = True

    def redo(self, document: Document) -> None:
        layer = document.get_layer(self.layer_id)
        if layer is None:
            return
        if layer.mask is None:
            layer.mask = np.full(layer.pixels.shape[:2], 255, dtype=np.uint8)
        x1, y1, x2, y2 = self.rect
        layer.mask[y1:y2, x1:x2] = self.after
        layer.touch_mask()
        document.dirty = True


@dataclass
class LayerMoveCommand:
    label: str
    layer_id: str
    before: tuple[int, int]
    after: tuple[int, int]
    before_mask: np.ndarray | None = None
    after_mask: np.ndarray | None = None

    @property
    def memory_bytes(self) -> int:
        before_bytes = 0 if self.before_mask is None else int(self.before_mask.nbytes)
        after_bytes = 0 if self.after_mask is None else int(self.after_mask.nbytes)
        return 64 + before_bytes + after_bytes

    def undo(self, document: Document) -> None:
        layer = document.get_layer(self.layer_id)
        if layer is not None:
            layer.x, layer.y = self.before
            if self.before_mask is not None:
                layer.mask = self.before_mask.copy()
            document.dirty = True

    def redo(self, document: Document) -> None:
        layer = document.get_layer(self.layer_id)
        if layer is not None:
            layer.x, layer.y = self.after
            if self.after_mask is not None:
                layer.mask = self.after_mask.copy()
            document.dirty = True


@dataclass
class LayerOpacityCommand:
    label: str
    layer_id: str
    before: float
    after: float

    @property
    def memory_bytes(self) -> int:
        return 64

    def undo(self, document: Document) -> None:
        layer = document.get_layer(self.layer_id)
        if layer is not None:
            layer.opacity = self.before
            document.dirty = True

    def redo(self, document: Document) -> None:
        layer = document.get_layer(self.layer_id)
        if layer is not None:
            layer.opacity = self.after
            document.dirty = True


@dataclass
class LayerBlendModeCommand:
    label: str
    layer_id: str
    before: str
    after: str

    @property
    def memory_bytes(self) -> int:
        return 128

    def undo(self, document: Document) -> None:
        layer = document.get_layer(self.layer_id)
        if layer is not None:
            layer.blend_mode = self.before
            document.dirty = True

    def redo(self, document: Document) -> None:
        layer = document.get_layer(self.layer_id)
        if layer is not None:
            layer.blend_mode = self.after
            document.dirty = True


@dataclass
class DocumentStateCommand:
    label: str
    before: dict[str, Any]
    after: dict[str, Any]

    @property
    def memory_bytes(self) -> int:
        total = 0
        for state in (self.before, self.after):
            for layer in state.get("layers", []):
                pixels = layer.get("pixels")
                if hasattr(pixels, "nbytes"):
                    total += int(pixels.nbytes)
        return total

    def undo(self, document: Document) -> None:
        document.restore_raw_state(self.before)
        document.dirty = True

    def redo(self, document: Document) -> None:
        document.restore_raw_state(self.after)
        document.dirty = True


@dataclass
class SelectionMaskCommand:
    label: str
    before: np.ndarray | None
    after: np.ndarray | None

    @property
    def memory_bytes(self) -> int:
        total = 0
        if self.before is not None:
            total += int(self.before.nbytes)
        if self.after is not None:
            total += int(self.after.nbytes)
        return max(total, 1)

    def undo(self, document: Document) -> None:
        document.selection_mask = None if self.before is None else self.before.copy()
        document.dirty = True

    def redo(self, document: Document) -> None:
        document.selection_mask = None if self.after is None else self.after.copy()
        document.dirty = True


class History:
    def __init__(self, memory_limit_bytes: int = 512 * 1024 * 1024) -> None:
        self.memory_limit_bytes = memory_limit_bytes
        self.undo_stack: list[Command] = []
        self.redo_stack: list[Command] = []
        self.memory_bytes = 0
        self.last_command: Command | None = None

    def clear(self) -> None:
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.memory_bytes = 0
        self.last_command = None

    def push(self, command: Command) -> None:
        if command.memory_bytes <= 0:
            return
        self.undo_stack.append(command)
        self.memory_bytes += command.memory_bytes
        self.redo_stack.clear()
        while self.undo_stack and self.memory_bytes > self.memory_limit_bytes:
            dropped = self.undo_stack.pop(0)
            self.memory_bytes -= dropped.memory_bytes

    def undo(self, document: Document) -> str | None:
        if not self.undo_stack:
            return None
        command = self.undo_stack.pop()
        self.memory_bytes -= command.memory_bytes
        with profiler.measure("history.undo"):
            command.undo(document)
        self.redo_stack.append(command)
        self.last_command = command
        return command.label

    def redo(self, document: Document) -> str | None:
        if not self.redo_stack:
            return None
        command = self.redo_stack.pop()
        with profiler.measure("history.redo"):
            command.redo(document)
        self.undo_stack.append(command)
        self.memory_bytes += command.memory_bytes
        self.last_command = command
        return command.label
