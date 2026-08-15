from __future__ import annotations

from ..app_shared import *


class PatchInteractionMixin:
    def patch_selection_down(self, event) -> None:
        self.clear_patch_preview()
        self._patch_right_start = self.canvas_to_doc(event)
        self.drag_start = self._patch_right_start
        self.draw_selection(self._patch_right_start, self._patch_right_start)
        self.status_text("Проведите правой кнопкой, чтобы задать прямоугольник заплатки")

    def patch_selection_drag(self, event) -> str | None:
        start = getattr(self, "_patch_right_start", None)
        if self.tool.get() != "patch" or start is None:
            return None
        self.draw_selection(start, self.canvas_to_doc(event))
        return "break"

    def patch_selection_up(self, event) -> str | None:
        start = getattr(self, "_patch_right_start", None)
        if self.tool.get() != "patch" or start is None:
            return None
        end = self.canvas_to_doc(event)
        self._patch_right_start = None
        self.drag_start = None
        self.clear_drag_preview()
        if abs(end[0] - start[0]) < 2 or abs(end[1] - start[1]) < 2:
            return "break"
        box = (*start, *end)
        feather = int(self.selection_feather.get())
        self.run_selection_command("Область заплатки", lambda: self.doc.set_rect_selection(box, "replace", feather))
        self.status_text("Область заплатки готова. Перетащите её левой кнопкой на источник")
        return "break"

    def begin_patch_drag(self, point: tuple[int, int]) -> None:
        self.clear_patch_preview()
        if self.doc.layer.locked:
            self.status_text("Слой заблокирован")
            self.drag_start = None
            return
        if self.doc.selection_mask is None or not np.any(self.doc.selection_mask):
            self.status_text("Сначала создайте выделение для инструмента Заплатка")
            self.drag_start = None
            return
        x, y = point
        if x < 0 or y < 0 or x >= self.doc.width or y >= self.doc.height or self.doc.selection_mask[y, x] == 0:
            self.status_text("Начните перетаскивание внутри активного выделения")
            self.drag_start = None
            return
        self._patch_start_bounds = self.doc.selection_bounds()
        if bool(self.patch_sample_all_layers.get()):
            self._patch_sample_pixels = self.doc.composite(False).copy()
            self._patch_sample_origin = (0, 0)
        else:
            self._patch_sample_pixels = self.doc.layer.pixels.copy()
            self._patch_sample_origin = (self.doc.layer.x, self.doc.layer.y)
        self.draw_patch_preview(point)
        self.status_text("Перетащите выделение на область-источник")

    def patch_source_bounds_for_point(self, point: tuple[int, int]) -> tuple[int, int, int, int] | None:
        if self.drag_start is None or self._patch_start_bounds is None:
            return None
        dx, dy = point[0] - self.drag_start[0], point[1] - self.drag_start[1]
        x1, y1, x2, y2 = self._patch_start_bounds
        return x1 + dx, y1 + dy, x2 + dx, y2 + dy

    def patch_source_is_valid(self, bounds: tuple[int, int, int, int]) -> bool:
        pixels = self._patch_sample_pixels
        if pixels is None:
            return False
        ox, oy = self._patch_sample_origin
        x1, y1, x2, y2 = bounds
        return x1 >= ox and y1 >= oy and x2 <= ox + pixels.shape[1] and y2 <= oy + pixels.shape[0]

    def _patch_result(self, bounds: tuple[int, int, int, int]):
        selection = self.doc.layer_selection_mask(self.doc.layer)
        if selection is None or self._patch_sample_pixels is None:
            return None
        return build_patch_edit(
            self.doc.layer.pixels,
            (self.doc.layer.x, self.doc.layer.y),
            selection,
            bounds[0],
            bounds[1],
            source_pixels=self._patch_sample_pixels,
            source_origin=self._patch_sample_origin,
            structure=int(self.patch_structure.get()),
            color_adaptation=float(self.patch_color_adaptation.get()),
        )

    def draw_patch_preview(self, point: tuple[int, int], force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self._last_patch_preview_time < 1 / 30:
            return
        self._last_patch_preview_time = now
        bounds = self.patch_source_bounds_for_point(point)
        if bounds is None:
            return
        self._render_patch_bounds(bounds)

    def _render_patch_bounds(self, bounds: tuple[int, int, int, int]) -> None:
        valid = self.patch_source_is_valid(bounds)
        result = self._patch_result(bounds) if valid else None
        self._patch_pending_bounds = bounds if result is not None else None
        self._draw_patch_source_outline(bounds, valid)
        if result is None:
            self._clear_patch_image()
            return
        (x1, y1, x2, y2), pixels = result
        selection = self.doc.layer_selection_mask(self.doc.layer)[y1:y2, x1:x2]
        preview = pixels.copy()
        preview[:, :, 3] = np.rint(preview[:, :, 3].astype(np.float32) * selection.astype(np.float32) / 255.0).astype(np.uint8)
        image = Image.fromarray(preview, mode="RGBA")
        scale = float(self.zoom.get())
        if scale != 1.0:
            image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.BILINEAR)
        self._patch_preview_image = ImageTk.PhotoImage(image)
        cx, cy = self.doc_to_canvas(x1 + self.doc.layer.x, y1 + self.doc.layer.y)
        if self._patch_image_id is None:
            self._patch_image_id = self.canvas.create_image(cx, cy, image=self._patch_preview_image, anchor=tk.NW)
        else:
            self.canvas.itemconfigure(self._patch_image_id, image=self._patch_preview_image)
            self.canvas.coords(self._patch_image_id, cx, cy)
        self.canvas.tag_raise(self._patch_image_id)
        if self._patch_preview_id is not None:
            self.canvas.tag_raise(self._patch_preview_id)

    def patch_preview_settings_changed(self, *_args) -> None:
        if not hasattr(self, "canvas") or self._patch_pending_bounds is None:
            return
        if bool(self.patch_sample_all_layers.get()):
            self._patch_sample_pixels = self.doc.composite(False).copy()
            self._patch_sample_origin = (0, 0)
        else:
            self._patch_sample_pixels = self.doc.layer.pixels.copy()
            self._patch_sample_origin = (self.doc.layer.x, self.doc.layer.y)
        self._render_patch_bounds(self._patch_pending_bounds)

    def _draw_patch_source_outline(self, bounds: tuple[int, int, int, int], valid: bool) -> None:
        coords = [*self.doc_to_canvas(bounds[0], bounds[1]), *self.doc_to_canvas(bounds[2], bounds[3])]
        color = "#f2b84b" if valid else "#e25d5d"
        if self._patch_preview_id is None:
            self._patch_preview_id = self.canvas.create_rectangle(*coords, outline=color, dash=(6, 3), width=2)
        else:
            self.canvas.coords(self._patch_preview_id, *coords)
            self.canvas.itemconfigure(self._patch_preview_id, outline=color)
        self.canvas.tag_raise(self._patch_preview_id)

    def finish_patch_drag(self, point: tuple[int, int]) -> None:
        self.draw_patch_preview(point, force=True)
        if self._patch_pending_bounds is None:
            self.status_text("Источник заплатки должен полностью попадать в доступное изображение")
            self.clear_patch_preview()
            return
        self.status_text("Предпросмотр заплатки: Enter применяет, Escape отменяет")

    def apply_patch_preview(self) -> bool:
        bounds = self._patch_pending_bounds
        if bounds is None:
            return False
        source_pixels = self._patch_sample_pixels
        source_origin = self._patch_sample_origin
        structure = int(self.patch_structure.get())
        adaptation = float(self.patch_color_adaptation.get())
        layer = self.doc.layer
        selection = self.doc.layer_selection_mask(layer)
        if selection is None:
            return False
        tile_size = 128
        ys, xs = np.where(selection > 0)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
        before_working = layer.working_rgba() if layer.working_pixels is not None else None
        before_tiles = []
        for ty in range(y1 // tile_size, (y2 - 1) // tile_size + 1):
            for tx in range(x1 // tile_size, (x2 - 1) // tile_size + 1):
                px1, py1 = tx * tile_size, ty * tile_size
                px2, py2 = min(layer.pixels.shape[1], px1 + tile_size), min(layer.pixels.shape[0], py1 + tile_size)
                rect = px1, py1, px2, py2
                before_tiles.append((rect, layer.pixels[py1:py2, px1:px2].copy()))
        changed = self.doc.patch_active_selection(
            bounds[0], bounds[1], True, adaptation,
            structure=structure,
            source_pixels=source_pixels,
            source_origin=source_origin,
        )
        if not changed:
            self.clear_patch_preview()
            return False
        layer.touch_pixels()
        after_working = layer.working_rgba() if before_working is not None else None
        patches = []
        precision = [] if before_working is not None else None
        for rect, before in before_tiles:
            px1, py1, px2, py2 = rect
            after = layer.pixels[py1:py2, px1:px2].copy()
            if np.array_equal(before, after):
                continue
            patches.append(TilePatch(rect, before, after))
            if precision is not None and after_working is not None:
                precision.append(TilePatch(
                    rect,
                    before_working[py1:py2, px1:px2].copy(),
                    after_working[py1:py2, px1:px2].copy(),
                ))
        if patches:
            self.push_command(PixelTilePatchCommand("Применить заплатку", layer.id, patches, precision))
            self.request_canvas_refresh(self.local_to_document_rect((x1, y1, x2, y2), layer), layer, "pixels")
        self.clear_patch_preview()
        self.refresh_layers()
        return True

    def _clear_patch_image(self) -> None:
        if self._patch_image_id is not None:
            self.canvas.delete(self._patch_image_id)
            self._patch_image_id = None
        self._patch_preview_image = None

    def clear_patch_preview(self) -> None:
        if self._patch_preview_id is not None:
            self.canvas.delete(self._patch_preview_id)
            self._patch_preview_id = None
        self._clear_patch_image()
        self._patch_start_bounds = None
        self._patch_pending_bounds = None
        self._patch_sample_pixels = None
        self._last_patch_preview_time = 0.0
