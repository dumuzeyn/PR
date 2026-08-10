from __future__ import annotations

from .history_base import *


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
