from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .core import Document


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
class PixelPatchCommand:
    label: str
    layer_id: str
    rect: tuple[int, int, int, int]
    before: np.ndarray
    after: np.ndarray

    @property
    def memory_bytes(self) -> int:
        return int(self.before.nbytes + self.after.nbytes)

    def undo(self, document: Document) -> None:
        layer = document.get_layer(self.layer_id)
        if layer is None:
            return
        x1, y1, x2, y2 = self.rect
        layer.pixels[y1:y2, x1:x2] = self.before
        document.dirty = True

    def redo(self, document: Document) -> None:
        layer = document.get_layer(self.layer_id)
        if layer is None:
            return
        x1, y1, x2, y2 = self.rect
        layer.pixels[y1:y2, x1:x2] = self.after
        document.dirty = True


@dataclass
class LayerMoveCommand:
    label: str
    layer_id: str
    before: tuple[int, int]
    after: tuple[int, int]

    @property
    def memory_bytes(self) -> int:
        return 64

    def undo(self, document: Document) -> None:
        layer = document.get_layer(self.layer_id)
        if layer is not None:
            layer.x, layer.y = self.before
            document.dirty = True

    def redo(self, document: Document) -> None:
        layer = document.get_layer(self.layer_id)
        if layer is not None:
            layer.x, layer.y = self.after
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

    def clear(self) -> None:
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.memory_bytes = 0

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
        command.undo(document)
        self.redo_stack.append(command)
        return command.label

    def redo(self, document: Document) -> str | None:
        if not self.redo_stack:
            return None
        command = self.redo_stack.pop()
        command.redo(document)
        self.undo_stack.append(command)
        self.memory_bytes += command.memory_bytes
        return command.label
