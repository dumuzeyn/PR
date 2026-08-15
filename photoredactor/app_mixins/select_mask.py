from __future__ import annotations

from ..app_shared import *
from ..selection_ops import smart_radius_refine


class SelectMaskMixin:
    def select_and_mask_workspace(self) -> None:
        if self.doc.selection_mask is None:
            messagebox.showinfo("Выделить и маска", "Сначала создайте выделение.")
            return
        data = self.select_and_mask_dialog()
        if data is None:
            return
        refined_mask = np.asarray(data["mask"], dtype=np.uint8)
        output = data["output"]
        if output == "Маска слоя":
            layer = self.doc.layer
            local_mask = np.zeros(layer.pixels.shape[:2], dtype=np.uint8)
            x1, y1 = max(0, layer.x), max(0, layer.y)
            x2 = min(self.doc.width, layer.x + layer.pixels.shape[1])
            y2 = min(self.doc.height, layer.y + layer.pixels.shape[0])
            if x1 < x2 and y1 < y2:
                lx1, ly1 = x1 - layer.x, y1 - layer.y
                local_mask[ly1:ly1 + y2 - y1, lx1:lx1 + x2 - x1] = refined_mask[y1:y2, x1:x2]
            before_fields = {
                "mask": None if layer.mask is None else layer.mask.copy(),
                "mask_enabled": layer.mask_enabled,
                "mask_linked": layer.mask_linked,
            }
            layer.mask = local_mask
            layer.mask_enabled = True
            layer.mask_linked = True
            layer.touch_mask()
            after_fields = {"mask": layer.mask.copy(), "mask_enabled": True, "mask_linked": True}
            self.push_command(LayerFieldsCommand("Select and Mask: маска слоя", layer.id, before_fields, after_fields))
            self.doc.dirty = True
            self.paint_target.set("mask")
            self.mask_preview.set(MASK_PREVIEW_CHANNEL)
            self.selection_box = self.doc.selection_bounds()
            self.refresh()
            self.status_text("Select and Mask: маска слоя")
        elif output in {"Новый слой", "Новый слой с маской"}:
            source_layer = self.doc.layer
            local_mask = np.zeros(source_layer.pixels.shape[:2], dtype=np.uint8)
            x1, y1 = max(0, source_layer.x), max(0, source_layer.y)
            x2 = min(self.doc.width, source_layer.x + source_layer.pixels.shape[1])
            y2 = min(self.doc.height, source_layer.y + source_layer.pixels.shape[0])
            if x1 < x2 and y1 < y2:
                lx1, ly1 = x1 - source_layer.x, y1 - source_layer.y
                local_mask[ly1:ly1 + y2 - y1, lx1:lx1 + x2 - x1] = refined_mask[y1:y2, x1:x2]
            duplicate = source_layer.clone()
            duplicate.name = f"{source_layer.name} - выделено"
            if bool(data.get("decontaminate", False)):
                duplicate.pixels = decontaminate_edge_colors(duplicate.pixels, local_mask, float(data.get("decontaminate_strength", 0.5)))
                duplicate.kind = "raster"
                duplicate.text_data = None
                duplicate.shape_data = None
                duplicate.smart_data = None
                duplicate.smart_source = None
            if output == "Новый слой с маской":
                duplicate.mask = local_mask.copy()
                duplicate.mask_enabled = True
                duplicate.mask_linked = True
                duplicate.touch_mask()
            else:
                alpha = duplicate.pixels[:, :, 3].astype(np.float32) * (local_mask.astype(np.float32) / 255.0)
                duplicate.pixels[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
                duplicate.mask = None
                duplicate.touch_pixels()
            self.doc.layers.append(duplicate)
            self.doc.active_layer = len(self.doc.layers) - 1
            self.doc.dirty = True
            self.selected_layer_ids = {duplicate.id}
            self.push_command(LayerInsertCommand("Select and Mask: новый слой", self.doc.active_layer, copy.deepcopy(duplicate)))
            self.refresh()
            self.status_text(f"Select and Mask: {output.lower()}")
        else:
            self.run_selection_command("Select and Mask", lambda: setattr(self.doc, "selection_mask", refined_mask.copy()))
            self.refresh_canvas()

    def select_and_mask_dialog(self) -> dict[str, object] | None:
        source = self.doc.selection_mask.copy()
        composite = self.render_engine.render(self.doc, checker=False)
        dialog = tk.Toplevel(self)
        dialog.title("Выделить и маска")
        dialog.transient(self)
        dialog.grab_set()
        dialog.minsize(900, 620)

        smooth = tk.IntVar(value=2)
        feather = tk.IntVar(value=2)
        contrast = tk.DoubleVar(value=1.25)
        shift = tk.IntVar(value=0)
        output = tk.StringVar(value="Выделение")
        preview_mode = tk.StringVar(value=SELECT_MASK_PREVIEW_CUTOUT)
        brush_mode = tk.StringVar(value="Уточнить волосы")
        brush_size = tk.IntVar(value=36)
        edge_radius = tk.IntVar(value=7)
        edge_strength = tk.DoubleVar(value=0.8)
        smart_radius = tk.BooleanVar(value=True)
        decontaminate = tk.BooleanVar(value=False)
        decontaminate_strength = tk.DoubleVar(value=0.5)
        working = source.copy()
        stroke_history: list[np.ndarray] = []
        last_point: list[tuple[int, int] | None] = [None]
        result: dict[str, object] | None = None
        preview_after: list[str | None] = [None]

        header = ttk.Frame(dialog, padding=(12, 10, 12, 6))
        header.pack(fill=tk.X)
        ttk.Label(header, text="Выделить и маска", style="PanelTitle.TLabel").pack(side=tk.LEFT)
        stats = ttk.Label(header, text="", style="Secondary.TLabel")
        stats.pack(side=tk.RIGHT)

        body = ttk.PanedWindow(dialog, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=12)
        preview = tk.Canvas(body, width=570, height=540, background=TOKENS.WORKSPACE, highlightthickness=1, highlightbackground=TOKENS.BORDER, cursor="crosshair")
        controls = ttk.Frame(body, width=285, padding=(12, 2))
        body.add(preview, weight=1)
        body.add(controls, weight=0)

        footer = ttk.Frame(dialog, padding=(12, 8, 12, 12))
        footer.pack(fill=tk.X)
        ttk.Label(footer, text="Рисуйте по сложному краю прямо в предпросмотре", style="Secondary.TLabel").pack(side=tk.LEFT)

        transform = {"scale": 1.0, "ox": 0.0, "oy": 0.0, "size": 540}
        reduced_composite_cache: dict[tuple[int, int], np.ndarray] = {}
        mode_ids = {
            "Уточнить волосы": "refine",
            "Добавить": "add",
            "Вычесть": "subtract",
            "Сгладить кистью": "smooth",
        }

        def current_mask() -> np.ndarray:
            mask = refine_selection_mask(working, int(smooth.get()), int(feather.get()), float(contrast.get()), int(shift.get()))
            if bool(smart_radius.get()):
                mask = smart_radius_refine(mask, composite, int(edge_radius.get()), float(edge_strength.get()))
            return mask

        def preview_mask(reduced_size: tuple[int, int], reduced_composite: np.ndarray, scale: float) -> np.ndarray:
            if reduced_size == (self.doc.width, self.doc.height):
                return current_mask()
            reduced_working = cv2.resize(working, reduced_size, interpolation=cv2.INTER_AREA)
            mask = refine_selection_mask(
                reduced_working,
                round(int(smooth.get()) * scale),
                round(int(feather.get()) * scale),
                float(contrast.get()),
                round(int(shift.get()) * scale),
            )
            if bool(smart_radius.get()):
                radius = max(1, round(int(edge_radius.get()) * scale))
                mask = smart_radius_refine(mask, reduced_composite, radius, float(edge_strength.get()))
            return mask

        def update_preview(*_args) -> None:
            preview_after[0] = None
            preview.update_idletasks()
            available = max(300, min(preview.winfo_width(), preview.winfo_height()))
            size = min(720, available)
            scale = min(1.0, size / max(1, self.doc.width), size / max(1, self.doc.height))
            reduced_width = max(1, round(self.doc.width * scale))
            reduced_height = max(1, round(self.doc.height * scale))
            transform.update({"scale": scale, "ox": (preview.winfo_width() - reduced_width) / 2.0, "oy": (preview.winfo_height() - reduced_height) / 2.0, "size": size})
            reduced_size = (reduced_width, reduced_height)
            reduced_composite = reduced_composite_cache.get(reduced_size)
            if reduced_composite is None:
                reduced_composite = composite if reduced_size == (self.doc.width, self.doc.height) else cv2.resize(composite, reduced_size, interpolation=cv2.INTER_AREA)
                reduced_composite_cache[reduced_size] = reduced_composite
            try:
                reduced_mask = preview_mask(reduced_size, reduced_composite, scale)
            except (tk.TclError, ValueError):
                return
            canvas = self.render_select_mask_preview(reduced_composite, reduced_mask, preview_mode.get(), size)
            self._select_mask_preview_image = ImageTk.PhotoImage(canvas)
            preview.delete("preview")
            preview.create_image(preview.winfo_width() / 2.0, preview.winfo_height() / 2.0, image=self._select_mask_preview_image, tags="preview")
            preview.tag_lower("preview")
            area_scale = max(scale * scale, 1e-9)
            selected = round(np.count_nonzero(reduced_mask) / area_scale)
            soft = round(np.count_nonzero((reduced_mask > 0) & (reduced_mask < 255)) / area_scale)
            stats.configure(text=f"Выбрано: {selected} px  |  Полупрозрачных: {soft} px")

        def schedule_preview(*_args) -> None:
            if preview_after[0] is not None:
                try:
                    dialog.after_cancel(preview_after[0])
                except tk.TclError:
                    pass
            preview_after[0] = dialog.after(35, update_preview)

        def add_spin(parent: ttk.Frame, label: str, variable, from_: float, to: float, increment: float = 1.0) -> None:
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=3)
            ttk.Label(row, text=label).pack(side=tk.LEFT)
            ttk.Spinbox(row, textvariable=variable, from_=from_, to=to, increment=increment, width=9, command=schedule_preview).pack(side=tk.RIGHT)

        ttk.Label(controls, text="Кисть уточнения", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(2, 4))
        brush_box = ttk.Combobox(controls, textvariable=brush_mode, values=list(mode_ids), state="readonly")
        brush_box.pack(fill=tk.X, pady=(0, 5))
        add_spin(controls, "Размер", brush_size, 3, 300)
        add_spin(controls, "Радиус анализа", edge_radius, 1, 40)
        add_spin(controls, "Сила", edge_strength, 0.05, 1.0, 0.05)
        ttk.Checkbutton(controls, text="Умный радиус", variable=smart_radius, command=schedule_preview).pack(anchor=tk.W, pady=(2, 4))
        brush_buttons = ttk.Frame(controls)
        brush_buttons.pack(fill=tk.X, pady=(4, 8))

        def reset_brushes() -> None:
            nonlocal working
            if not np.array_equal(working, source):
                stroke_history.append(working.copy())
            working = source.copy()
            schedule_preview()

        def undo_brush() -> None:
            nonlocal working
            if stroke_history:
                working = stroke_history.pop()
                schedule_preview()

        ttk.Button(brush_buttons, text="Отменить штрих", command=undo_brush).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(brush_buttons, text="Сбросить", command=reset_brushes).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        ttk.Separator(controls).pack(fill=tk.X, pady=5)
        ttk.Label(controls, text="Глобальная обработка", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(4, 3))
        add_spin(controls, "Сглаживание", smooth, 0, 100)
        add_spin(controls, "Растушевка", feather, 0, 500)
        add_spin(controls, "Контраст", contrast, 0.0, 5.0, 0.05)
        add_spin(controls, "Сдвиг края", shift, -500, 500)

        ttk.Separator(controls).pack(fill=tk.X, pady=6)
        ttk.Label(controls, text="Просмотр", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(3, 2))
        preview_box = ttk.Combobox(controls, textvariable=preview_mode, values=SELECT_MASK_PREVIEW_MODES, state="readonly")
        preview_box.pack(fill=tk.X, pady=(0, 7))
        ttk.Label(controls, text="Результат", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(3, 2))
        output_box = ttk.Combobox(controls, textvariable=output, values=["Выделение", "Маска слоя", "Новый слой", "Новый слой с маской"], state="readonly")
        output_box.pack(fill=tk.X)
        decontaminate_check = ttk.Checkbutton(controls, text="Очистить цветную кайму", variable=decontaminate)
        decontaminate_check.pack(anchor=tk.W, pady=(7, 1))
        decontaminate_row = ttk.Frame(controls)
        decontaminate_row.pack(fill=tk.X, pady=3)
        ttk.Label(decontaminate_row, text="Сила очистки").pack(side=tk.LEFT)
        decontaminate_spin = ttk.Spinbox(decontaminate_row, textvariable=decontaminate_strength, from_=0.0, to=1.0, increment=0.05, width=9)
        decontaminate_spin.pack(side=tk.RIGHT)

        def update_output_controls(*_args) -> None:
            enabled = output.get() in {"Новый слой", "Новый слой с маской"}
            decontaminate_check.configure(state=tk.NORMAL if enabled else tk.DISABLED)
            decontaminate_spin.configure(state=tk.NORMAL if enabled else tk.DISABLED)
            if not enabled:
                decontaminate.set(False)

        def canvas_to_document(event) -> tuple[int, int] | None:
            scale = float(transform["scale"])
            source_width = self.doc.width * scale
            source_height = self.doc.height * scale
            left = (preview.winfo_width() - source_width) / 2.0
            top = (preview.winfo_height() - source_height) / 2.0
            x = round((event.x - left) / max(1e-8, scale))
            y = round((event.y - top) / max(1e-8, scale))
            if 0 <= x < self.doc.width and 0 <= y < self.doc.height:
                return x, y
            return None

        def apply_stroke(start: tuple[int, int], end: tuple[int, int]) -> None:
            nonlocal working
            stroke = np.zeros_like(working)
            width = max(1, int(brush_size.get()))
            cv2.line(stroke, start, end, 255, width, cv2.LINE_AA)
            cv2.circle(stroke, end, max(1, width // 2), 255, -1, cv2.LINE_AA)
            ys, xs = np.where(stroke > 0)
            if len(xs) == 0:
                return
            padding = max(width, int(edge_radius.get()) * 3) + 2
            x1, y1 = max(0, int(xs.min()) - padding), max(0, int(ys.min()) - padding)
            x2, y2 = min(self.doc.width, int(xs.max()) + padding + 1), min(self.doc.height, int(ys.max()) + padding + 1)
            working[y1:y2, x1:x2] = refine_selection_brush(
                working[y1:y2, x1:x2],
                composite[y1:y2, x1:x2],
                stroke[y1:y2, x1:x2],
                mode_ids.get(brush_mode.get(), "refine"),
                max(1, int(edge_radius.get())),
                float(edge_strength.get()),
            )
            schedule_preview()

        def brush_press(event) -> None:
            point = canvas_to_document(event)
            if point is None:
                return
            stroke_history.append(working.copy())
            last_point[0] = point
            apply_stroke(point, point)

        def brush_drag(event) -> None:
            point = canvas_to_document(event)
            if point is None or last_point[0] is None:
                return
            apply_stroke(last_point[0], point)
            last_point[0] = point

        def brush_release(_event) -> None:
            last_point[0] = None
            update_preview()

        def brush_cursor(event) -> None:
            preview.delete("brush-cursor")
            radius = max(2.0, float(brush_size.get()) * float(transform["scale"]) / 2.0)
            preview.create_oval(event.x - radius, event.y - radius, event.x + radius, event.y + radius, outline="#f4f4f4", width=1, tags="brush-cursor")

        def accept() -> None:
            nonlocal result
            result = {
                "smooth": max(0, int(smooth.get())),
                "feather": max(0, int(feather.get())),
                "contrast": max(0.0, float(contrast.get())),
                "shift": int(shift.get()),
                "output": output.get(),
                "mask": current_mask().copy(),
                "decontaminate": bool(decontaminate.get()),
                "decontaminate_strength": float(decontaminate_strength.get()),
            }
            dialog.destroy()

        def cancel() -> None:
            dialog.destroy()

        ttk.Button(footer, text="Применить", command=accept, style="Primary.TButton").pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(footer, text="Отмена", command=cancel).pack(side=tk.RIGHT)
        preview_box.bind("<<ComboboxSelected>>", update_preview)
        output_box.bind("<<ComboboxSelected>>", update_output_controls)
        preview.bind("<ButtonPress-1>", brush_press)
        preview.bind("<B1-Motion>", brush_drag)
        preview.bind("<ButtonRelease-1>", brush_release)
        preview.bind("<Motion>", brush_cursor)
        preview.bind("<Leave>", lambda _event: preview.delete("brush-cursor"))
        preview.bind("<Configure>", schedule_preview)
        for variable in [smooth, feather, contrast, shift, edge_radius, edge_strength, decontaminate_strength]:
            variable.trace_add("write", schedule_preview)
        dialog.protocol("WM_DELETE_WINDOW", cancel)
        self._select_mask_canvas = preview
        self._select_mask_brush_mode = brush_mode
        self._select_mask_brush_size = brush_size
        self._select_mask_output = output
        self._select_mask_decontaminate = decontaminate
        self._select_mask_apply_stroke = apply_stroke
        self._select_mask_working = lambda: working.copy()
        self._select_mask_accept = accept
        self.center_toplevel(dialog, 980, 700)
        update_output_controls()
        update_preview()
        dialog.wait_window()
        return result

    @staticmethod
    def mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
        if mask is None or not np.any(mask):
            return None
        ys, xs = np.where(mask > 0)
        return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)

    @staticmethod
    def render_select_mask_preview(composite: np.ndarray, mask: np.ndarray, mode: str, size: int = 160) -> Image.Image:
        preview_size = (size, size)
        height, width = composite.shape[:2]
        scale = min(1.0, size / max(1, width), size / max(1, height))
        reduced_size = max(1, round(width * scale)), max(1, round(height * scale))
        reduced = composite if reduced_size == (width, height) else cv2.resize(composite, reduced_size, interpolation=cv2.INTER_AREA)
        reduced_mask = mask if reduced_size == (width, height) else cv2.resize(mask, reduced_size, interpolation=cv2.INTER_NEAREST)
        mask_image = Image.fromarray(reduced_mask.astype(np.uint8), "L")
        source = rgba_array_to_pil(reduced)
        x = (size - source.width) // 2
        y = (size - source.height) // 2
        if mode == SELECT_MASK_PREVIEW_OVERLAY:
            canvas = Image.new("RGBA", preview_size, (44, 46, 52, 255))
            canvas.alpha_composite(source, (x, y))
            overlay = Image.new("RGBA", source.size, (255, 36, 68, 0))
            overlay.putalpha(mask_image.point(lambda value: int(value * 0.55)))
            canvas.alpha_composite(overlay, (x, y))
            return canvas
        if mode == SELECT_MASK_PREVIEW_ONION:
            canvas = Image.new("RGBA", preview_size, (44, 46, 52, 255))
            faded = source.copy()
            faded.putalpha(mask_image.point(lambda value: 72 + round(value * 0.72)))
            canvas.alpha_composite(faded, (x, y))
            return canvas
        if mode == SELECT_MASK_PREVIEW_MARCHING:
            canvas = Image.new("RGBA", preview_size, (44, 46, 52, 255))
            canvas.alpha_composite(source, (x, y))
            mask_array = np.asarray(mask_image, dtype=np.uint8)
            binary = np.where(mask_array >= 128, 255, 0).astype(np.uint8)
            contour = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)) > 0
            yy, xx = np.indices(contour.shape)
            ants = np.zeros((source.height, source.width, 4), dtype=np.uint8)
            ants[contour & (((xx + yy) // 4) % 2 == 0)] = (255, 255, 255, 255)
            ants[contour & (((xx + yy) // 4) % 2 == 1)] = (0, 0, 0, 255)
            canvas.alpha_composite(Image.fromarray(ants, "RGBA"), (x, y))
            return canvas
        if mode in {SELECT_MASK_PREVIEW_BLACK, SELECT_MASK_PREVIEW_WHITE}:
            background = (0, 0, 0, 255) if mode == SELECT_MASK_PREVIEW_BLACK else (255, 255, 255, 255)
            canvas = Image.new("RGBA", preview_size, background)
            cutout = source.copy()
            cutout.putalpha(mask_image)
            canvas.alpha_composite(cutout, (x, y))
            return canvas
        if mode == SELECT_MASK_PREVIEW_LAYERS:
            canvas = Image.new("RGBA", preview_size, (44, 46, 52, 255))
            canvas.alpha_composite(source, (x, y))
            dim = Image.new("RGBA", source.size, (0, 0, 0, 0))
            dim.putalpha(mask_image.point(lambda value: round((255 - value) * 0.62)))
            canvas.alpha_composite(dim, (x, y))
            return canvas
        if mode == SELECT_MASK_PREVIEW_CUTOUT:
            checker = Image.new("RGBA", preview_size, (44, 46, 52, 255))
            tile = 8
            for cy in range(0, size, tile):
                for cx in range(0, size, tile):
                    color = (90, 92, 98, 255) if ((cx // tile) + (cy // tile)) % 2 else (58, 60, 66, 255)
                    checker.paste(color, (cx, cy, min(cx + tile, size), min(cy + tile, size)))
            cutout = source.copy()
            cutout.putalpha(mask_image)
            checker.alpha_composite(cutout, (x, y))
            return checker
        if mode == SELECT_MASK_PREVIEW_EDGE_CONFIDENCE:
            confidence = selection_edge_confidence(mask, composite, 3)
            confidence_image = Image.fromarray(confidence.astype(np.uint8), "L")
            confidence_image.thumbnail(preview_size, Image.Resampling.NEAREST)
            canvas = Image.new("RGBA", preview_size, (44, 46, 52, 255))
            dimmed = source.copy()
            shade = Image.new("RGBA", dimmed.size, (0, 0, 0, 96))
            dimmed.alpha_composite(shade)
            canvas.alpha_composite(dimmed, (x, y))
            conf = np.array(confidence_image, dtype=np.uint8)
            overlay = np.zeros((confidence_image.height, confidence_image.width, 4), dtype=np.uint8)
            weak = (conf > 0) & (conf < 85)
            medium = (conf >= 85) & (conf < 170)
            strong = conf >= 170
            overlay[weak] = [255, 68, 68, 230]
            overlay[medium] = [255, 190, 64, 230]
            overlay[strong] = [64, 220, 110, 230]
            canvas.alpha_composite(Image.fromarray(overlay, "RGBA"), ((size - confidence_image.width) // 2, (size - confidence_image.height) // 2))
            return canvas
        canvas = Image.new("RGBA", preview_size, (44, 46, 52, 255))
        gray = Image.new("L", preview_size, 72)
        gray.paste(mask_image, ((size - mask_image.width) // 2, (size - mask_image.height) // 2))
        rgba = Image.merge("RGBA", (gray, gray, gray, Image.new("L", preview_size, 255)))
        canvas.alpha_composite(rgba)
        return canvas
