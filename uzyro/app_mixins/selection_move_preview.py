from __future__ import annotations

from ..app_shared import *


class SelectionMovePreviewMixin:
    def _initialize_move_preview_state(self) -> None:
        self._move_group_starts: dict[str, tuple[int, int]] = {}
        self._move_group_masks: dict[str, np.ndarray | None] = {}
        self._move_last_bounds: tuple[int, int, int, int] | None = None
        self._move_after_id: str | None = None
        self._move_pending_point: tuple[int, int] | None = None
        self._move_selection_start: tuple[int, int] | None = None
        self._move_selection_bounds: tuple[int, int, int, int] | None = None
        self._move_selection_delta = (0, 0)
        self._move_selection_preview_id: int | None = None
        self._move_selection_preview_image: ImageTk.PhotoImage | None = None
        self._move_selection_source_backup: tuple[str, tuple[int, int, int, int], np.ndarray, bool] | None = None

    def begin_move_selection_preview(self, bounds: tuple[int, int, int, int] | None) -> None:
        self.restore_move_selection_source(refresh=True)
        self.clear_move_selection_preview()
        if bounds is None:
            return
        x1, y1, x2, y2 = bounds
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0 or width * height > 4_000_000:
            return
        layer = self.doc.layer
        preview = np.zeros((height, width, 4), dtype=np.uint8)
        left, top = max(x1, layer.x), max(y1, layer.y)
        right = min(x2, layer.x + layer.pixels.shape[1])
        bottom = min(y2, layer.y + layer.pixels.shape[0])
        if left >= right or top >= bottom:
            return
        source = layer.pixels[top - layer.y:bottom - layer.y, left - layer.x:right - layer.x].copy()
        mask = self.doc.selection_mask[top:bottom, left:right].astype(np.float32) / 255.0
        source[:, :, 3] = np.rint(source[:, :, 3].astype(np.float32) * mask).astype(np.uint8)
        preview[top - y1:bottom - y1, left - x1:right - x1] = source
        image = Image.fromarray(preview, mode="RGBA")
        zoom = max(0.01, float(self.zoom.get()))
        size = max(1, round(width * zoom)), max(1, round(height * zoom))
        if size != image.size:
            image = image.resize(size, Image.Resampling.BILINEAR)
        self._move_selection_preview_image = ImageTk.PhotoImage(image)
        position = self.doc_to_canvas(x1, y1)
        self._move_selection_preview_id = self.canvas.create_image(*position, image=self._move_selection_preview_image, anchor=tk.NW)
        local_box = left - layer.x, top - layer.y, right - layer.x, bottom - layer.y
        source_box = layer.pixels[local_box[1]:local_box[3], local_box[0]:local_box[2]]
        self._move_selection_source_backup = (layer.id, local_box, source_box.copy(), bool(self.doc.dirty))
        source_box[:, :, 3] = np.rint(source_box[:, :, 3].astype(np.float32) * (1.0 - mask)).astype(np.uint8)
        self.request_canvas_refresh((left, top, right, bottom), layer, "pixels")
        self.clear_selection_overlay()

    def update_move_selection_preview(self, dx: int, dy: int) -> None:
        bounds = self._move_selection_bounds
        if bounds is None or self._move_selection_preview_id is None:
            return
        self.canvas.coords(self._move_selection_preview_id, *self.doc_to_canvas(bounds[0] + dx, bounds[1] + dy))
        self.canvas.tag_raise(self._move_selection_preview_id)

    def clear_move_selection_preview(self) -> None:
        if self._move_selection_preview_id is not None and hasattr(self, "canvas"):
            self.canvas.delete(self._move_selection_preview_id)
        self._move_selection_preview_id = None
        self._move_selection_preview_image = None

    def restore_move_selection_source(self, *, refresh: bool) -> None:
        backup = self._move_selection_source_backup
        self._move_selection_source_backup = None
        if backup is None:
            return
        layer = self.doc.get_layer(backup[0])
        if layer is None:
            return
        x1, y1, x2, y2 = backup[1]
        layer.pixels[y1:y2, x1:x2] = backup[2]
        self.doc.dirty = backup[3]
        if refresh:
            self.request_canvas_refresh((x1 + layer.x, y1 + layer.y, x2 + layer.x, y2 + layer.y), layer, "pixels")

    def cancel_move_selection_preview(self) -> None:
        self.restore_move_selection_source(refresh=True)
        self.clear_move_selection_preview()
        self._move_selection_start = None
        self._move_selection_bounds = None
        self._move_selection_delta = (0, 0)


__all__ = ["SelectionMovePreviewMixin"]
