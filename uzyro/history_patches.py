from __future__ import annotations

from dataclasses import field

from .history_base import *


@dataclass
class PixelPatchCommand:
    label: str
    layer_id: str
    rect: tuple[int, int, int, int]
    before: np.ndarray
    after: np.ndarray
    precision_before: np.ndarray | None = None
    precision_after: np.ndarray | None = None

    @property
    def memory_bytes(self) -> int:
        precision = 0
        if self.precision_before is not None:
            precision += int(self.precision_before.nbytes)
        if self.precision_after is not None:
            precision += int(self.precision_after.nbytes)
        return int(self.before.nbytes + self.after.nbytes + precision)

    def _apply(self, document: Document, display: np.ndarray, precise: np.ndarray | None) -> None:
        layer = document.get_layer(self.layer_id)
        if layer is None:
            return
        x1, y1, x2, y2 = self.rect
        if precise is not None and layer.working_pixels is not None:
            working = layer.working_rgba()
            working[y1:y2, x1:x2] = precise
            layer.set_working_rgba(working, layer.working_depth, layer.working_model)
        else:
            layer.pixels[y1:y2, x1:x2] = display
            layer.touch_pixels()
        document.dirty = True

    @property
    def dirty_rects(self) -> list[tuple[int, int, int, int]]:
        return [self.rect]

    def undo(self, document: Document) -> None:
        self._apply(document, self.before, self.precision_before)

    def redo(self, document: Document) -> None:
        self._apply(document, self.after, self.precision_after)

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
class LayerGroupMoveCommand:
    label: str
    before: dict[str, tuple[int, int]]
    after: dict[str, tuple[int, int]]
    before_masks: dict[str, np.ndarray | None] = field(default_factory=dict)
    after_masks: dict[str, np.ndarray | None] = field(default_factory=dict)

    @property
    def memory_bytes(self) -> int:
        arrays = [mask for mask in (*self.before_masks.values(), *self.after_masks.values()) if mask is not None]
        return 96 + len(self.before) * 48 + sum(int(mask.nbytes) for mask in arrays)

    @staticmethod
    def _apply(document: Document, positions: dict[str, tuple[int, int]], masks: dict[str, np.ndarray | None]) -> None:
        for layer_id, position in positions.items():
            layer = document.get_layer(layer_id)
            if layer is None:
                continue
            layer.x, layer.y = position
            if layer_id in masks and masks[layer_id] is not None:
                layer.mask = masks[layer_id].copy()
        document.dirty = True

    def undo(self, document: Document) -> None:
        self._apply(document, self.before, self.before_masks)

    def redo(self, document: Document) -> None:
        self._apply(document, self.after, self.after_masks)

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
class LayerVisibilityCommand:
    label: str
    layer_id: str
    before: bool
    after: bool

    @property
    def memory_bytes(self) -> int:
        return 64

    def undo(self, document: Document) -> None:
        layer = document.get_layer(self.layer_id)
        if layer is not None:
            layer.visible = self.before
            document.dirty = True

    def redo(self, document: Document) -> None:
        layer = document.get_layer(self.layer_id)
        if layer is not None:
            layer.visible = self.after
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

__all__ = [name for name in globals() if not name.startswith("__")]
