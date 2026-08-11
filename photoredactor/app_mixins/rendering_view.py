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
            marker = "V" if layer.visible else " "
            indicators: list[str] = []
            if layer.mask is not None:
                indicators.append("M")
            if layer.locked:
                indicators.append("L")
            if layer.kind == "linked":
                linked_status = self.doc.linked_layer_status(layer)["status"]
                if linked_status in {"missing", "modified"}:
                    indicators.append("?" if linked_status == "missing" else "!")
            if layer.effects or layer.filters:
                indicators.append("fx")
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

    def refresh_properties(self) -> None:
        if not hasattr(self, "object_properties") or not self.doc.layers:
            return
        panel = self.object_properties
        for child in panel.winfo_children():
            child.destroy()
        layer = self.doc.layer
        title = "Фигура" if layer.kind == "shape" else "Текст" if layer.kind == "text" else "Растровый слой"
        ttk.Label(panel, text=title, style="PanelTitle.TLabel").pack(anchor=tk.W, padx=10, pady=(3, 6))

        def numeric_row(label: str, initial: int | float, apply, start=-100000, end=100000, increment=1) -> None:
            row = ttk.Frame(panel)
            row.pack(fill=tk.X, padx=10, pady=2)
            ttk.Label(row, text=label, style="Secondary.TLabel").pack(side=tk.LEFT)
            variable = tk.DoubleVar(value=initial) if isinstance(initial, float) else tk.IntVar(value=initial)
            spin = ttk.Spinbox(row, textvariable=variable, from_=start, to=end, increment=increment, width=8)
            spin.pack(side=tk.RIGHT)
            commit = lambda _event=None, v=variable: apply(v.get())
            spin.bind("<Return>", commit)
            spin.bind("<FocusOut>", commit)

        def combo_row(label: str, initial: str, values: list[str], apply) -> None:
            row = ttk.Frame(panel)
            row.pack(fill=tk.X, padx=10, pady=2)
            ttk.Label(row, text=label, style="Secondary.TLabel").pack(side=tk.LEFT)
            variable = tk.StringVar(value=initial)
            combo = ttk.Combobox(row, textvariable=variable, values=values, state="readonly", width=14)
            combo.pack(side=tk.RIGHT)
            combo.bind("<<ComboboxSelected>>", lambda _event: apply(variable.get()))

        def color_row(label: str, color, command) -> None:
            row = ttk.Frame(panel)
            row.pack(fill=tk.X, padx=10, pady=2)
            ttk.Label(row, text=label, style="Secondary.TLabel").pack(side=tk.LEFT)
            swatch = tk.Button(
                row,
                command=command,
                background=self.color_hex(tuple(color)),
                activebackground=self.color_hex(tuple(color)),
                width=4,
                height=1,
                relief=tk.FLAT,
                borderwidth=1,
                highlightthickness=1,
                highlightbackground=TOKENS.BORDER,
                cursor="hand2",
            )
            swatch.pack(side=tk.RIGHT)

        bounds = self.object_document_bounds(layer)
        display_x = bounds[0] if bounds is not None else layer.x
        display_y = bounds[1] if bounds is not None else layer.y
        numeric_row("X", display_x, lambda value, y=display_y: self.set_object_position(int(value), y))
        numeric_row("Y", display_y, lambda value, x=display_x: self.set_object_position(x, int(value)))
        if layer.kind == "shape" and layer.shape_data is not None:
            box = shape_data_bounds(layer.shape_data) or (0, 0, 1, 1)
            numeric_row("Ширина", box[2] - box[0], lambda value: self.set_shape_size(int(value), None), 2, 100000)
            numeric_row("Высота", box[3] - box[1], lambda value: self.set_shape_size(None, int(value)), 2, 100000)
            numeric_row("Обводка", int(layer.shape_data.get("stroke_width", 0)), lambda value: self.set_shape_property("stroke_width", int(value)), 0, 100)
            kind = str(layer.shape_data.get("shape", "rectangle"))
            if kind in {"polygon", "star"}:
                numeric_row("Стороны" if kind == "polygon" else "Лучи", int(layer.shape_data.get("sides", 5)), lambda value: self.set_shape_property("sides", int(value)), 3, 64)
            if kind == "star":
                numeric_row("Внутренний радиус", float(layer.shape_data.get("inner_ratio", 0.5)), lambda value: self.set_shape_property("inner_ratio", float(value)), 0.05, 0.95, 0.05)
            color_row("Заливка", layer.shape_data.get("fill") or [0, 0, 0, 0], lambda: self.pick_shape_property_color("fill"))
            color_row("Обводка", layer.shape_data.get("stroke") or [0, 0, 0, 0], lambda: self.pick_shape_property_color("stroke"))
            if kind == "boolean":
                ttk.Button(panel, text="Редактировать операцию и контуры", command=self.edit_boolean_shape).pack(fill=tk.X, padx=10, pady=6)
        elif layer.kind == "text" and layer.text_data is not None:
            combo_row("Шрифт", str(layer.text_data.get("font_family", "Arial")), ["Arial", "Segoe UI", "Calibri", "Times New Roman", "Verdana", "Tahoma"], lambda value: self.set_text_property("font_family", value))
            numeric_row("Размер", int(layer.text_data.get("size", 48)), lambda value: self.set_text_property("size", int(value)), 4, 500)
            numeric_row("Интервал", int(layer.text_data.get("tracking", 0)), lambda value: self.set_text_property("tracking", int(value)), -100, 500)
            numeric_row("Межстрочный", int(layer.text_data.get("line_spacing", 10)), lambda value: self.set_text_property("line_spacing", int(value)), 0, 500)
            combo_row("Выравнивание", str(layer.text_data.get("align", "left")), ["left", "center", "right"], lambda value: self.set_text_property("align", value))
            color_row("Цвет", layer.text_data.get("color") or [255, 255, 255, 255], self.pick_text_property_color)
            ttk.Button(panel, text="Редактировать текст", command=self.edit_active_text_on_canvas).pack(fill=tk.X, padx=10, pady=6)
            ttk.Button(panel, text="Текст по контуру...", command=self.edit_text_path).pack(fill=tk.X, padx=10, pady=(0, 6))
        else:
            ttk.Button(panel, text="Фильтры слоя", command=self.edit_layer_filters).pack(fill=tk.X, padx=10, pady=5)

    def set_object_position(self, x: int, y: int) -> None:
        layer = self.doc.layer
        before = (layer.x, layer.y)
        bounds = self.object_document_bounds(layer)
        if bounds is None:
            after = (int(x), int(y))
        else:
            after = (layer.x + int(x) - bounds[0], layer.y + int(y) - bounds[1])
        if before == after:
            return
        layer.x, layer.y = after
        self.push_command(LayerMoveCommand("Переместить объект", layer.id, before, after))
        self.doc.dirty = True
        self.refresh()

    def set_shape_size(self, width: int | None, height: int | None) -> None:
        layer = self.doc.layer
        if layer.kind != "shape" or layer.shape_data is None:
            return
        before = copy.deepcopy(layer.shape_data)
        x1, y1, x2, y2 = shape_data_bounds(before) or (0, 0, 1, 1)
        target = (x1, y1, x1 + max(2, int(width if width is not None else x2 - x1)), y1 + max(2, int(height if height is not None else y2 - y1)))
        layer.shape_data = transform_shape_data_to_box(before, target)
        render_shape_layer(layer)
        layer.touch_pixels()
        self.push_command(ShapeDataCommand("Изменить размер фигуры", layer.id, before, copy.deepcopy(layer.shape_data), layer.name, layer.name))
        self.doc.dirty = True
        self.refresh()

    def set_shape_property(self, key: str, value) -> None:
        layer = self.doc.layer
        if layer.kind != "shape" or layer.shape_data is None or layer.shape_data.get(key) == value:
            return
        before = copy.deepcopy(layer.shape_data)
        layer.shape_data[key] = value
        render_shape_layer(layer)
        layer.touch_pixels()
        self.push_command(ShapeDataCommand("Изменить фигуру", layer.id, before, copy.deepcopy(layer.shape_data), layer.name, layer.name))
        self.doc.dirty = True
        self.refresh()

    def pick_shape_property_color(self, key: str) -> None:
        layer = self.doc.layer
        if layer.shape_data is None:
            return
        initial = layer.shape_data.get(key) or [255, 255, 255, 255]
        selected = colorchooser.askcolor(self.color_hex(tuple(initial)), parent=self)
        if selected[0] is not None:
            self.set_shape_property(key, (*[round(value) for value in selected[0]], 255))

    def pick_text_property_color(self) -> None:
        layer = self.doc.layer
        if layer.kind != "text" or layer.text_data is None:
            return
        initial = layer.text_data.get("color") or [255, 255, 255, 255]
        selected = colorchooser.askcolor(self.color_hex(tuple(initial)), parent=self)
        if selected[0] is not None:
            self.set_text_property("color", [*[round(value) for value in selected[0]], 255])

    def set_text_property(self, key: str, value) -> None:
        layer = self.doc.layer
        if layer.kind != "text" or layer.text_data is None or layer.text_data.get(key) == value:
            return
        before = copy.deepcopy(layer.text_data)
        layer.text_data[key] = value
        render_text_layer(layer)
        layer.touch_pixels()
        self.push_command(TextDataCommand("Изменить текст", layer.id, before, copy.deepcopy(layer.text_data)))
        self.doc.dirty = True
        self.refresh()

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
