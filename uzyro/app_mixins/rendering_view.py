from __future__ import annotations

from ..app_shared import *


class RenderingViewMixin:
    def invalidate_pixels(self, clear_layer_caches: bool = True) -> None:
        self._composite_dirty = True
        self._view_dirty = True
        self.render_engine.invalidate_full(self.doc, clear_layer_caches=clear_layer_caches)

    def invalidate_view(self) -> None:
        self._view_dirty = True

    def refresh_canvas(self) -> None:
        scale = self.zoom.get()
        reduced_view = (
            scale < 0.5
            and self.view_channel.get() == "RGB"
            and self.mask_preview.get() == MASK_PREVIEW_NORMAL
            and not color_settings(self.doc.metadata).get("soft_proof_enabled", False)
        )
        composite_changed = self._composite_dirty or self._composite_cache is None
        level = 0
        if reduced_view:
            display, level = self.render_engine.render_for_zoom(self.doc, scale, checker=True)
        elif self._composite_dirty or self._composite_cache is None:
            self._composite_cache = self.render_engine.render(self.doc, checker=True)
            self._composite_dirty = False
        if not reduced_view:
            display = self.apply_soft_proof_display(self._composite_cache)
            display = self._channel_display(display)
            display = self._mask_preview_display(display)
        pad_x = max(40, self.canvas.winfo_width() // 2)
        pad_y = max(40, self.canvas.winfo_height() // 2)
        self._canvas_origin = (pad_x, pad_y)
        view_signature = (
            round(scale, 6),
            self.doc.width,
            self.doc.height,
            self.view_channel.get(),
            self.mask_preview.get(),
            color_settings(self.doc.metadata).get("soft_proof_enabled", False),
            color_settings(self.doc.metadata).get("proof_profile_name", ""),
            self.doc.layer.id if self.doc.layers else None,
        )
        full_view = self._canvas_view_signature != view_signature or not self._canvas_tile_ids
        if scale < 0.5:
            self._update_canvas_mipmap(display, scale, pad_x, pad_y, level)
        else:
            changed_tiles = self.render_engine.last_changed_tiles if composite_changed and not full_view else set(self.render_engine.all_tiles(self.doc))
            self._update_canvas_tiles(display, changed_tiles, scale, pad_x, pad_y, full_view)
        scaled_width = max(1, round(self.doc.width * scale))
        scaled_height = max(1, round(self.doc.height * scale))
        self.canvas.configure(scrollregion=(0, 0, scaled_width + pad_x * 2, scaled_height + pad_y * 2))
        self._canvas_view_signature = view_signature
        self.zoom_label.configure(text=f"{round(scale * 100)}%")
        if hasattr(self, "status_zoom"):
            self.status_zoom.configure(text=f"{round(scale * 100)}%")
        self._last_render_time = time.perf_counter()
        self._view_dirty = False
        self.update_selection_overlay()
        self.update_object_bounds()
        self.update_grid_and_guides()
        if self.tool.get() == "quick_selection" and self._quick_points:
            self.update_quick_selection_preview(force=True)
        if self._last_pointer_event is not None:
            self.update_brush_preview(self._last_pointer_event)
        self.update_clone_source_marker()
        if self._crop_box is not None and self.tool.get() == "crop":
            self.draw_crop_overlay(self._crop_box)

    def _update_canvas_mipmap(self, display: np.ndarray, scale: float, pad_x: int, pad_y: int, level: int = 0) -> None:
        for item_id in self._canvas_tile_ids.values():
            self.canvas.delete(item_id)
        self._canvas_tile_ids.clear()
        self._canvas_tile_images.clear()
        if level == 0:
            key = (id(self.doc), self.render_engine.render_revision, self.view_channel.get(), self.mask_preview.get())
            reduced, level = self.render_engine.mipmaps.for_zoom(key, display, scale)
        else:
            reduced = display
        target = max(1, round(self.doc.width * scale)), max(1, round(self.doc.height * scale))
        image = rgba_array_to_pil(reduced)
        if image.size != target:
            image = image.resize(target, Image.Resampling.BILINEAR)
        self._preview_image = ImageTk.PhotoImage(image)
        if self._canvas_image_id is None:
            self._canvas_image_id = self.canvas.create_image(pad_x, pad_y, image=self._preview_image, anchor=tk.NW)
        else:
            self.canvas.itemconfigure(self._canvas_image_id, image=self._preview_image)
            self.canvas.coords(self._canvas_image_id, pad_x, pad_y)
        self.render_engine.profiler.count("canvas.mipmap_level", level)

    def _update_canvas_tiles(
        self,
        display: np.ndarray,
        changed_tiles: set[tuple[int, int]],
        scale: float,
        pad_x: int,
        pad_y: int,
        full_view: bool,
    ) -> None:
        if self._canvas_image_id is not None:
            self.canvas.delete(self._canvas_image_id)
            self._canvas_image_id = None
            self._preview_image = None
        if full_view:
            for item_id in self._canvas_tile_ids.values():
                self.canvas.delete(item_id)
            self._canvas_tile_ids.clear()
            self._canvas_tile_images.clear()
        resample = Image.Resampling.NEAREST if scale >= 4 else Image.Resampling.BILINEAR
        for tx, ty in sorted(changed_tiles):
            x1, y1, x2, y2 = self.render_engine.tile_rect(self.doc, tx, ty)
            with self.render_engine.profiler.measure("canvas.numpy_to_pil"):
                image = rgba_array_to_pil(display[y1:y2, x1:x2])
            left, top = round(x1 * scale), round(y1 * scale)
            right, bottom = round(x2 * scale), round(y2 * scale)
            size = max(1, right - left), max(1, bottom - top)
            if image.size != size:
                with self.render_engine.profiler.measure("canvas.resize_tile"):
                    image = image.resize(size, resample)
            with self.render_engine.profiler.measure("canvas.pil_to_imagetk"):
                photo = ImageTk.PhotoImage(image)
            key = tx, ty
            item_id = self._canvas_tile_ids.get(key)
            if item_id is None:
                item_id = self.canvas.create_image(pad_x + left, pad_y + top, image=photo, anchor=tk.NW)
                self._canvas_tile_ids[key] = item_id
            else:
                self.canvas.itemconfigure(item_id, image=photo)
                self.canvas.coords(item_id, pad_x + left, pad_y + top)
            self._canvas_tile_images[key] = photo

    def _channel_display(self, composite: np.ndarray) -> np.ndarray:
        channel = self.view_channel.get() if hasattr(self, "view_channel") else "RGB"
        if channel == "RGB":
            return composite
        source = self.render_engine.render(self.doc, checker=False)
        out = np.zeros_like(source)
        out[:, :, 3] = 255
        if channel == "Alpha":
            gray = source[:, :, 3]
        else:
            index = {"Red": 0, "Green": 1, "Blue": 2}.get(channel, 0)
            gray = source[:, :, index]
        out[:, :, :3] = gray[:, :, None]
        return out

    def _mask_preview_display(self, display: np.ndarray) -> np.ndarray:
        mode = self.mask_preview.get() if hasattr(self, "mask_preview") else MASK_PREVIEW_NORMAL
        if mode == MASK_PREVIEW_NORMAL:
            return display
        layer = self.doc.layer
        if layer.mask is None:
            return display
        mask_canvas = np.zeros((self.doc.height, self.doc.width), dtype=np.uint8)
        active_canvas = np.zeros((self.doc.height, self.doc.width), dtype=np.uint8)
        effective = effective_layer_mask(layer) if layer.mask_enabled else layer.mask
        paste_mask(mask_canvas, effective, layer.x, layer.y)
        paste_mask(active_canvas, np.full(layer.mask.shape, 255, dtype=np.uint8), layer.x, layer.y)
        if mode == MASK_PREVIEW_CHANNEL:
            gray = np.where(active_canvas > 0, mask_canvas, 72).astype(np.uint8)
            out = np.zeros((self.doc.height, self.doc.width, 4), dtype=np.uint8)
            out[:, :, :3] = gray[:, :, None]
            out[:, :, 3] = 255
            return out
        if mode == MASK_PREVIEW_OVERLAY:
            out = display.copy()
            hidden = ((255 - mask_canvas).astype(np.float32) / 255.0) * (active_canvas > 0)
            alpha = (hidden * 0.58)[:, :, None]
            red = np.zeros_like(out[:, :, :3])
            red[:, :, 0] = 255
            red[:, :, 1] = 36
            red[:, :, 2] = 68
            out[:, :, :3] = np.clip(out[:, :, :3].astype(np.float32) * (1.0 - alpha) + red.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
            out[:, :, 3] = 255
            return out
        return display

    def request_canvas_refresh(self, rect: tuple[int, int, int, int] | None = None, layer=None, kind: str = "pixels", preserve_layer_caches: bool = False) -> None:
        if rect is None:
            self.invalidate_pixels(clear_layer_caches=not preserve_layer_caches)
        else:
            self.render_engine.invalidate_region(self.doc, rect, layer, kind)
            self._composite_dirty = True
            self._view_dirty = True
        if self._render_after_id is not None:
            return
        elapsed_ms = (time.perf_counter() - self._last_render_time) * 1000
        delay = 0 if elapsed_ms >= 33 else int(33 - elapsed_ms)
        self._render_after_id = self.after(delay, self._run_scheduled_canvas_refresh)

    def _run_scheduled_canvas_refresh(self) -> None:
        self._render_after_id = None
        self.refresh_canvas()

    def refresh(self, preserve_render_cache: bool = False) -> None:
        self.invalidate_pixels(clear_layer_caches=not preserve_render_cache)
        if self._render_after_id is not None:
            self.after_cancel(self._render_after_id)
            self._render_after_id = None
        self.refresh_canvas()
        self.refresh_layers()
        self.refresh_properties()
        self.refresh_history_panel()
        self.info.configure(text=f"{self.doc.width} x {self.doc.height}px\nСлоев: {len(self.doc.layers)}\nАктивный: {self.doc.layer.name}")
        self.status_size.configure(text=f"{self.doc.width} x {self.doc.height}")

    def refresh_layers(self) -> None:
        known_ids = {layer.id for layer in self.doc.layers}
        selected_ids = set(getattr(self, "selected_layer_ids", set())) & known_ids
        if not selected_ids and self.doc.layers:
            selected_ids = {self.doc.layer.id}
        self.selected_layer_ids = selected_ids
        rows: list[str] = []
        for i, layer in enumerate(reversed(self.doc.layers)):
            marker = "👁" if layer.visible else "⊘"
            indicators: list[str] = []
            if layer.mask is not None:
                indicators.append("маска")
            if layer.locked:
                indicators.append("заблокирован")
            if layer.group_id:
                indicators.append("группа")
            if layer.kind == "linked":
                linked_status = self.doc.linked_layer_status(layer)["status"]
                if linked_status in {"missing", "modified"}:
                    indicators.append("связь потеряна" if linked_status == "missing" else "источник изменён")
            if layer.effects or layer.filters:
                indicators.append("эффекты")
            spot = assigned_spot_color(self.doc, layer.id)
            if spot is not None:
                indicators.append(f"плашечный: {spot.name}")
            suffix = f"  [{' '.join(indicators)}]" if indicators else ""
            rows.append(f"{marker}   {layer.name}{suffix}")
        existing = list(self.layer_list.get(0, tk.END))
        if len(existing) != len(rows):
            self.layer_list.delete(0, tk.END)
            for row in rows:
                self.layer_list.insert(tk.END, row)
        else:
            for index, row in enumerate(rows):
                if existing[index] != row:
                    self.layer_list.delete(index)
                    self.layer_list.insert(index, row)
        self.layer_list.selection_clear(0, tk.END)
        for row, layer in enumerate(reversed(self.doc.layers)):
            if layer.id in self.selected_layer_ids:
                self.layer_list.selection_set(row)
        self.layer_list.activate(len(self.doc.layers) - 1 - self.doc.active_layer)
        self.layer_opacity.set(self.doc.layer.opacity)
        self.blend_mode.set(self.doc.layer.blend_mode)
        self.refresh_layer_previews()

    def refresh_history_panel(self) -> None:
        if not hasattr(self, "history_list"):
            return
        labels = [command.label for command in self.history.undo_stack]
        existing = list(self.history_list.get(0, tk.END))
        if existing != labels:
            self.history_list.delete(0, tk.END)
            for label in labels:
                self.history_list.insert(tk.END, label)
            if labels:
                self.history_list.selection_set(tk.END)
                self.history_list.see(tk.END)


    def refresh_layer_previews(self) -> None:
        layer = self.doc.layer
        layer_key = (layer.id, layer.pixels_revision, id(layer.pixels), layer.pixels.shape)
        layer_preview = self._layer_thumbnail_cache.get(layer_key)
        if layer_preview is None:
            layer_preview = self.make_layer_thumbnail(layer.pixels)
            self._layer_thumbnail_cache[layer_key] = layer_preview
            self._trim_thumbnail_cache(self._layer_thumbnail_cache)
        self._layer_thumb_image = ImageTk.PhotoImage(layer_preview)
        self.layer_thumb.configure(image=self._layer_thumb_image)
        mask_key = (layer.id, layer.mask_revision, id(layer.mask), None if layer.mask is None else layer.mask.shape)
        mask_preview = self._mask_thumbnail_cache.get(mask_key)
        if mask_preview is None:
            mask_preview = self.make_mask_thumbnail(layer.mask)
            self._mask_thumbnail_cache[mask_key] = mask_preview
            self._trim_thumbnail_cache(self._mask_thumbnail_cache)
        self._mask_thumb_image = ImageTk.PhotoImage(mask_preview)
        self.mask_thumb.configure(image=self._mask_thumb_image)

    @staticmethod
    def _trim_thumbnail_cache(cache: dict[tuple[object, ...], Image.Image], limit: int = 128) -> None:
        while len(cache) > limit:
            cache.pop(next(iter(cache)))

    def make_layer_thumbnail(self, pixels: np.ndarray, size: int = 64) -> Image.Image:
        height, width = pixels.shape[:2]
        scale = min(1.0, size / max(1, width), size / max(1, height))
        preview_size = max(1, round(width * scale)), max(1, round(height * scale))
        preview = pixels if preview_size == (width, height) else cv2.resize(pixels, preview_size, interpolation=cv2.INTER_AREA)
        image = rgba_array_to_pil(preview)
        canvas = Image.new("RGBA", (size, size), (44, 46, 52, 255))
        x = (size - image.width) // 2
        y = (size - image.height) // 2
        canvas.alpha_composite(image, (x, y))
        return canvas

    def make_mask_thumbnail(self, mask: np.ndarray | None, size: int = 64) -> Image.Image:
        if mask is None:
            return Image.new("RGBA", (size, size), (72, 74, 82, 255))
        height, width = mask.shape[:2]
        scale = min(1.0, size / max(1, width), size / max(1, height))
        preview_size = max(1, round(width * scale)), max(1, round(height * scale))
        preview = mask if preview_size == (width, height) else cv2.resize(mask, preview_size, interpolation=cv2.INTER_AREA)
        image = Image.fromarray(preview.astype(np.uint8), "L")
        canvas = Image.new("L", (size, size), 72)
        x = (size - image.width) // 2
        y = (size - image.height) // 2
        canvas.paste(image, (x, y))
        return Image.merge("RGBA", (canvas, canvas, canvas, Image.new("L", (size, size), 255)))

    def status_text(self, text: str) -> None:
        if hasattr(self, "status"):
            self.status.configure(text=text)

    def document_copy(self) -> Document:
        doc = Document.new(1, 1)
        doc.restore_raw_state(self.doc.raw_state())
        return doc

    def run_background(self, label: str, worker, done=None, is_current=None) -> None:
        self.status_text(f"{label}...")
        future = self.executor.submit(worker)

        def complete() -> None:
            try:
                result = future.result()
            except Exception as exc:
                messagebox.showerror(label, str(exc))
                self.status_text(f"{label}: error")
                return
            if is_current is not None and not is_current():
                self.status_text(f"{label}: result discarded because the document changed")
                return
            if done:
                done(result)
            self.status_text(f"{label}: done")

        future.add_done_callback(lambda _future: self.after(0, complete))
