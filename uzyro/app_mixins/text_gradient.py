from __future__ import annotations

from ..app_shared import *
from ..preview_ops import visible_document_rect


class TextGradientMixin:
    def update_drag_preview_item(self, kind: str, coords, **options) -> None:
        item_id = self._drag_preview_ids[0] if self._drag_preview_ids else None
        if item_id is not None and self.canvas.type(item_id) != kind:
            self.clear_drag_preview()
            item_id = None
        if item_id is None:
            creator = {
                "oval": self.canvas.create_oval,
                "line": self.canvas.create_line,
                "polygon": self.canvas.create_polygon,
                "rectangle": self.canvas.create_rectangle,
            }[kind]
            item_id = creator(*coords, **options)
            self._drag_preview_ids = [item_id]
        else:
            self.canvas.coords(item_id, *coords)
            self.canvas.itemconfigure(item_id, **options)

    def clear_drag_preview(self) -> None:
        for item_id in self._drag_preview_ids:
            self.canvas.delete(item_id)
        self._drag_preview_ids.clear()

    def current_gradient_kind(self) -> str:
        return {
            "Линейный": "linear",
            "Радиальный": "radial",
            "Отраженный": "reflected",
            "Ромб": "diamond",
            "Угловой": "angular",
        }.get(self.gradient_type.get(), "linear")

    def begin_text_editor(self, start: tuple[int, int], end: tuple[int, int], layer_id: str | None = None) -> None:
        self.cancel_text_edit()
        existing = self.doc.get_layer(layer_id) if layer_id else None
        if existing is not None and existing.text_data is not None:
            data = existing.text_data
            x, y = int(data.get("x", start[0])), int(data.get("y", start[1]))
            box_width = max(0, int(data.get("box_width", 0)))
            box_height = max(0, int(data.get("box_height", 0)))
            initial_text = str(data.get("text", ""))
            self._text_editor_before = copy.deepcopy(data)
            self._text_editor_layer_id = existing.id
            self.load_text_properties_from_layer(existing)
        else:
            x, y = int(start[0]), int(start[1])
            box_width = abs(int(end[0]) - x) if abs(int(end[0]) - x) >= 6 else 0
            box_height = abs(int(end[1]) - y) if box_width else 0
            initial_text = ""
            self._text_editor_before = None
            self._text_editor_layer_id = None
            self.text_box_width.set(box_width)
            self.text_box_height.set(box_height)
        self._text_editor_origin = (x, y)
        self._text_editor_box_width = box_width
        self._text_editor_box_height = box_height
        visible_width = box_width or min(520, max(260, self.doc.width - x))
        visible_height = max(72, box_height if box_width else int(self.text_size.get()) * 2)
        editor = tk.Text(
            self.canvas,
            wrap=tk.WORD if box_width else tk.NONE,
            width=max(8, round(visible_width / max(7, int(self.text_size.get()) * 0.55))),
            height=max(2, round(visible_height / max(14, int(self.text_size.get()) * 1.2))),
            undo=True,
            borderwidth=2,
            relief=tk.SOLID,
            padx=4,
            pady=3,
        )
        editor.insert("1.0", initial_text)
        cx, cy = self.doc_to_canvas(x, y)
        self._text_editor_window = self.canvas.create_window(cx, cy, window=editor, anchor=tk.NW)
        self._text_editor = editor
        self.update_text_editor_style()
        editor.focus_set()
        editor.mark_set(tk.INSERT, tk.END)
        self.status_text("Введите текст на холсте. Кнопка 'Готово' завершает редактирование.")

    def edit_active_text_on_canvas(self) -> None:
        layer = self.doc.layer
        if layer.kind != "text" or layer.text_data is None:
            self.status_text("Сначала выберите текстовый слой")
            return
        if layer.x or layer.y:
            layer.text_data["x"] = int(layer.text_data.get("x", 0)) + layer.x
            layer.text_data["y"] = int(layer.text_data.get("y", 0)) + layer.y
            layer.x = 0
            layer.y = 0
            render_text_layer(layer)
            layer.touch_pixels()
        x = int(layer.text_data.get("x", 0))
        y = int(layer.text_data.get("y", 0))
        width = max(240, int(layer.text_data.get("box_width", 0)))
        self.begin_text_editor((x, y), (x + width, y + int(layer.text_data.get("size", 48)) * 3), layer.id)

    def update_text_editor_style(self) -> None:
        editor = self._text_editor
        if editor is None:
            return
        styles = []
        if self.text_bold.get():
            styles.append("bold")
        if self.text_italic.get():
            styles.append("italic")
        font = tkfont.Font(
            family=self.text_font_family.get() or "Arial",
            size=max(8, round(int(self.text_size.get()) * float(self.zoom.get()) * 0.75)),
            weight="bold" if "bold" in styles else "normal",
            slant="italic" if "italic" in styles else "roman",
            underline=bool(self.text_underline.get()),
        )
        editor.configure(font=font, foreground=self.color_hex(self.foreground), insertbackground=self.color_hex(self.foreground))
        editor._uzyro_font = font
        editor.tag_configure("paragraph", justify=self.text_align.get(), spacing3=max(0, int(self.text_line_spacing.get())))
        editor.tag_add("paragraph", "1.0", tk.END)

    def finish_text_edit(self) -> None:
        editor = self._text_editor
        if editor is None:
            return
        text = editor.get("1.0", "end-1c")
        layer_id = self._text_editor_layer_id
        origin = self._text_editor_origin
        box_width = max(0, int(self.text_box_width.get() or self._text_editor_box_width))
        box_height = max(0, int(self.text_box_height.get() or self._text_editor_box_height)) if box_width else 0
        before = copy.deepcopy(self._text_editor_before)
        self._destroy_text_editor()
        if layer_id is not None:
            layer = self.doc.get_layer(layer_id)
            if layer is None or before is None:
                return
            self.doc.active_layer = self.doc.layers.index(layer)
            self.apply_text_values(layer, text, origin, box_width, box_height)
            after = copy.deepcopy(layer.text_data or {})
            if before != after:
                self.push_command(TextDataCommand("Edit text", layer.id, before, after, f"Text: {str(before.get('text', ''))[:24]}", layer.name))
        elif text:
            layer = self.doc.add_text_layer(
                text, origin[0], origin[1], self.foreground, int(self.text_size.get()),
                self.text_font_family.get(), box_width, self.text_align.get(),
                int(self.text_line_spacing.get()), int(self.text_tracking.get()),
                bool(self.text_bold.get()), bool(self.text_italic.get()), bool(self.text_underline.get()),
                rotation=float(self.text_rotation.get()),
                box_height=box_height, text_mode="paragraph" if box_width else "point",
            )
            self.selected_layer_ids = {layer.id}
            self.push_command(LayerInsertCommand("Text layer", self.doc.active_layer, copy.deepcopy(layer)))
        self.refresh()

    def cancel_text_edit(self) -> None:
        if self._text_editor is not None:
            self._destroy_text_editor()

    def _destroy_text_editor(self) -> None:
        if self._text_editor_window is not None:
            self.canvas.delete(self._text_editor_window)
        if self._text_editor is not None:
            self._text_editor.destroy()
        self._text_editor = None
        self._text_editor_window = None
        self._text_editor_layer_id = None
        self._text_editor_before = None

    def apply_text_values(self, layer: Layer, text: str, origin: tuple[int, int], box_width: int, box_height: int = 0) -> None:
        data = layer.text_data or {}
        data.update({
            "text": text,
            "x": int(origin[0]),
            "y": int(origin[1]),
            "color": list(self.foreground),
            "size": int(self.text_size.get()),
            "font_family": self.text_font_family.get(),
            "box_width": max(0, int(box_width)),
            "box_height": max(0, int(box_height)) if box_width else 0,
            "text_mode": "paragraph" if box_width else "point",
            "align": self.text_align.get(),
            "line_spacing": max(0, int(self.text_line_spacing.get())),
            "tracking": int(self.text_tracking.get()),
            "bold": bool(self.text_bold.get()),
            "italic": bool(self.text_italic.get()),
            "underline": bool(self.text_underline.get()),
            "rotation": float(self.text_rotation.get()),
        })
        layer.text_data = data
        layer.name = f"Text: {text[:24]}"
        render_text_layer(layer)
        layer.touch_pixels()
        self.doc.dirty = True

    def load_text_properties_from_layer(self, layer: Layer) -> None:
        if layer.text_data is None:
            return
        data = layer.text_data
        self._loading_text_properties = True
        try:
            self.text_font_family.set(str(data.get("font_family", "Arial")))
            self.text_size.set(int(data.get("size", 48)))
            self.text_bold.set(bool(data.get("bold", False)))
            self.text_italic.set(bool(data.get("italic", False)))
            self.text_underline.set(bool(data.get("underline", False)))
            self.text_align.set(str(data.get("align", "left")))
            self.text_line_spacing.set(int(data.get("line_spacing", 10)))
            self.text_tracking.set(int(data.get("tracking", 0)))
            self.text_rotation.set(float(data.get("rotation", 0.0)))
            self.text_box_width.set(int(data.get("box_width", 0)))
            self.text_box_height.set(int(data.get("box_height", 0)))
            color = data.get("color")
            if isinstance(color, list) and len(color) == 4:
                self.foreground = tuple(int(value) for value in color)
                self.refresh_color_control()
        finally:
            self._loading_text_properties = False

    def text_properties_changed(self, *_args) -> None:
        if self._loading_text_properties:
            return
        if self._text_editor is not None:
            self.update_text_editor_style()
            return
        if not self._editor_active or self.tool.get() != "text":
            return
        layer = self.doc.layer
        if layer.kind != "text" or layer.text_data is None:
            return
        if self._text_property_before is None:
            self._text_property_before = copy.deepcopy(layer.text_data)
        self.apply_text_values(
            layer,
            str(layer.text_data.get("text", "")),
            (int(layer.text_data.get("x", 0)), int(layer.text_data.get("y", 0))),
            int(self.text_box_width.get()),
            int(self.text_box_height.get()),
        )
        self.request_canvas_refresh()
        if self._text_property_after_id is not None:
            self.after_cancel(self._text_property_after_id)
        self._text_property_after_id = self.after(350, self.commit_text_property_history)

    def commit_text_property_history(self) -> None:
        self._text_property_after_id = None
        before = self._text_property_before
        self._text_property_before = None
        layer = self.doc.layer
        if before is None or layer.kind != "text" or layer.text_data is None:
            return
        after = copy.deepcopy(layer.text_data)
        if before != after:
            before_name = f"Text: {str(before.get('text', ''))[:24]}"
            self.push_command(TextDataCommand("Text properties", layer.id, before, after, before_name, layer.name))

    def current_gradient_stops(self) -> list[dict[str, object]]:
        if self.gradient_definition is not None:
            return copy.deepcopy(list(self.gradient_definition.get("stops", [])))
        stops: list[dict[str, object]] = [
            {"position": 0.0, "color": list(self.foreground)},
            {"position": 1.0, "color": list(self.background)},
        ]
        if self.gradient_mid_enabled.get():
            stops.append({
                "position": float(np.clip(self.gradient_mid_position.get(), 0.01, 0.99)),
                "color": list(self.gradient_mid_color),
            })
        return sorted(stops, key=lambda stop: float(stop["position"]))

    def current_gradient_definition(self) -> dict[str, object]:
        definition = copy.deepcopy(self.gradient_definition or {})
        definition["stops"] = self.current_gradient_stops()
        definition.setdefault("opacity_stops", [{"position": 0.0, "opacity": 1.0}, {"position": 1.0, "opacity": 1.0}])
        definition.setdefault("reverse", False)
        definition.setdefault("dither", False)
        definition.setdefault("transparency", True)
        definition.setdefault("interpolation_space", "srgb")
        definition.setdefault("noise", {"enabled": False, "roughness": 0.5, "color_model": "rgb", "seed": 0, "restrict_colors": False})
        return definition

    def gradient_render_options(self) -> dict[str, object]:
        definition = self.current_gradient_definition()
        return {
            "opacity_stops": definition["opacity_stops"],
            "reverse": definition["reverse"],
            "dither": definition["dither"],
            "transparency": definition["transparency"],
            "interpolation_space": definition["interpolation_space"],
            "noise": definition["noise"],
        }

    def pick_gradient_mid(self) -> None:
        color = colorchooser.askcolor(color=self.color_hex(self.gradient_mid_color), title="Средняя точка градиента")[0]
        if color:
            self.gradient_mid_color = tuple(map(int, color)) + (255,)
            self.gradient_mid_enabled.set(True)
            if hasattr(self, "tool_options_panel"):
                self.tool_options_panel.render()

    def update_gradient_preview(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        if start == end:
            return
        self._gradient_preview_pending = (start, end)
        if self._gradient_preview_after_id is not None:
            return
        elapsed = time.perf_counter() - self._last_gradient_preview_at
        delay = max(0, round((1 / 30 - elapsed) * 1000))
        if delay:
            self._gradient_preview_after_id = self.after(delay, self.render_pending_gradient_preview)
        else:
            self.render_pending_gradient_preview()

    def render_pending_gradient_preview(self) -> None:
        self._gradient_preview_after_id = None
        pending = self._gradient_preview_pending
        self._gradient_preview_pending = None
        if pending is None:
            return
        self._last_gradient_preview_at = time.perf_counter()
        start, end = pending
        if self.gradient_mode.get() == "Объект":
            self.update_gradient_object_preview(start, end)
            return
        x1, y1, x2, y2 = visible_document_rect(
            self.canvas, self._canvas_origin, self.zoom.get(), (self.doc.width, self.doc.height)
        )
        if x1 >= x2 or y1 >= y2:
            self.hide_gradient_preview_image()
            return
        scale = max(0.01, float(self.zoom.get()))
        width = max(1, round((x2 - x1) * scale))
        height = max(1, round((y2 - y1) * scale))
        pixels = GradientEngine.render(
            width,
            height,
            (start[0] * scale, start[1] * scale),
            (end[0] * scale, end[1] * scale),
            self.current_gradient_stops(),
            self.current_gradient_kind(),
            (x1 * scale, y1 * scale),
            **self.gradient_render_options(),
        )
        alpha = np.full((height, width), 190, dtype=np.uint8)
        if self.doc.selection_mask is not None:
            selection = self.doc.selection_mask[y1:y2, x1:x2]
            selection = cv2.resize(selection, (width, height), interpolation=cv2.INTER_LINEAR)
            alpha = np.minimum(alpha, selection)
        pixels[:, :, 3] = np.minimum(pixels[:, :, 3], alpha)
        image = Image.fromarray(pixels, "RGBA")
        self._gradient_preview_image = ImageTk.PhotoImage(image)
        x, y = self.doc_to_canvas(x1, y1)
        if self._gradient_preview_id is None:
            self._gradient_preview_id = self.canvas.create_image(x, y, image=self._gradient_preview_image, anchor=tk.NW)
        else:
            self.canvas.coords(self._gradient_preview_id, x, y)
            self.canvas.itemconfigure(self._gradient_preview_id, image=self._gradient_preview_image)
        self.canvas.tag_raise(self._gradient_preview_id)
        for item_id in self._drag_preview_ids:
            self.canvas.tag_raise(item_id)

    def update_gradient_object_preview(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        x1, x2 = sorted((int(start[0]), int(end[0])))
        y1, y2 = sorted((int(start[1]), int(end[1])))
        scale = max(0.01, float(self.zoom.get()))
        visible = visible_document_rect(
            self.canvas, self._canvas_origin, scale, (self.doc.width, self.doc.height)
        )
        clip_x1, clip_y1 = max(x1, visible[0]), max(y1, visible[1])
        clip_x2, clip_y2 = min(x2, visible[2]), min(y2, visible[3])
        if clip_x1 >= clip_x2 or clip_y1 >= clip_y2:
            self.hide_gradient_preview_image()
            return
        width = max(2, round((clip_x2 - clip_x1) * scale))
        height = max(2, round((clip_y2 - clip_y1) * scale))
        scaled_start = (start[0] * scale, start[1] * scale)
        scaled_end = (end[0] * scale, end[1] * scale)
        origin = (clip_x1 * scale, clip_y1 * scale)
        if self.gradient_object_fill.get() == "Текстура":
            yy, xx = np.mgrid[0:height, 0:width]
            xx = xx + round(clip_x1 * scale)
            yy = yy + round(clip_y1 * scale)
            size = max(2, round(18 * scale))
            texture_kind = self.gradient_texture.get()
            if texture_kind == "Точки":
                dx = np.mod(xx, size) - size / 2.0
                dy = np.mod(yy, size) - size / 2.0
                selector = (dx * dx + dy * dy) <= (size * 0.24) ** 2
            elif texture_kind == "Полосы":
                selector = np.mod((xx + yy) // size, 2) == 0
            else:
                selector = np.mod(xx // size + yy // size, 2) == 0
            pixels = np.where(selector[:, :, None], np.array(self.foreground), np.array(self.background)).astype(np.uint8)
        else:
            pixels = GradientEngine.render(
                width, height, scaled_start, scaled_end, self.current_gradient_stops(),
                self.current_gradient_kind(), origin, **self.gradient_render_options(),
            )
        mask_image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_image)
        box = (
            (x1 - clip_x1) * scale,
            (y1 - clip_y1) * scale,
            (x2 - clip_x1) * scale,
            (y2 - clip_y1) * scale,
        )
        shape = self.gradient_shape.get()
        if shape == "Эллипс":
            draw.ellipse(box, fill=220)
        elif shape == "Многоугольник":
            draw.polygon(regular_polygon_points(box, max(3, int(self.polygon_sides.get()))), fill=220)
        elif shape == "Звезда":
            draw.polygon(star_points(box, max(3, int(self.star_points_count.get())), float(self.star_inner_ratio.get())), fill=220)
        elif shape == "Произвольная":
            draw.polygon(custom_shape_points(CUSTOM_SHAPE_PRESETS.get(self.custom_shape_preset.get()), box), fill=220)
        else:
            draw.rectangle(box, fill=220)
        pixels[:, :, 3] = np.minimum(pixels[:, :, 3], np.array(mask_image, dtype=np.uint8))
        self._gradient_preview_image = ImageTk.PhotoImage(Image.fromarray(pixels, "RGBA"))
        x, y = self.doc_to_canvas(clip_x1, clip_y1)
        if self._gradient_preview_id is None:
            self._gradient_preview_id = self.canvas.create_image(x, y, image=self._gradient_preview_image, anchor=tk.NW)
        else:
            self.canvas.coords(self._gradient_preview_id, x, y)
            self.canvas.itemconfigure(self._gradient_preview_id, image=self._gradient_preview_image)
        self.canvas.tag_raise(self._gradient_preview_id)
        for item_id in self._drag_preview_ids:
            self.canvas.tag_raise(item_id)

    def hide_gradient_preview_image(self) -> None:
        if self._gradient_preview_id is not None:
            self.canvas.delete(self._gradient_preview_id)
            self._gradient_preview_id = None
        self._gradient_preview_image = None

    def clear_gradient_preview(self) -> None:
        if self._gradient_preview_after_id is not None:
            self.after_cancel(self._gradient_preview_after_id)
            self._gradient_preview_after_id = None
        self._gradient_preview_pending = None
        self.hide_gradient_preview_image()
        self._last_gradient_preview_at = 0.0

    def create_gradient_object(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        shape = {
            "Прямоугольник": "rectangle",
            "Эллипс": "ellipse",
            "Многоугольник": "polygon",
            "Звезда": "star",
            "Произвольная": "custom",
        }.get(self.gradient_shape.get(), "rectangle")
        gradient = {
            "type": self.current_gradient_kind(),
            "start": list(start),
            "end": list(end),
            "opacity": 1.0,
        }
        gradient.update(self.current_gradient_definition())
        texture = {
            "type": {"Шахматная": "checker", "Полосы": "stripes", "Точки": "dots"}.get(self.gradient_texture.get(), "checker"),
            "size": 18,
            "color_a": list(self.foreground),
            "color_b": list(self.background),
        }
        use_texture = self.gradient_object_fill.get() == "Текстура"
        layer = self.doc.add_shape_layer(
            shape,
            (*start, *end),
            self.foreground,
            self.background,
            int(self.shape_stroke_width.get()),
            int(self.polygon_sides.get() if shape == "polygon" else self.star_points_count.get()),
            float(self.star_inner_ratio.get()),
            custom_points=CUSTOM_SHAPE_PRESETS.get(self.custom_shape_preset.get()),
            gradient=None if use_texture else gradient,
            texture=texture if use_texture else None,
        )
        self.push_command(LayerInsertCommand("Gradient object", self.doc.active_layer, copy.deepcopy(layer)))
