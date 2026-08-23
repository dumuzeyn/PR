from __future__ import annotations

from ..app_shared import *


class ContentAwareMixin:
    def filter_content_aware_fill(self) -> None:
        layer = self.doc.layer
        selection_mask = self.doc.layer_selection_mask(layer)
        if selection_mask is None or not np.any(selection_mask):
            messagebox.showinfo("Заливка с учетом содержимого", "Сначала создайте выделение на активном слое.")
            return
        if layer.locked:
            self.status_text("Слой заблокирован")
            return
        settings = self.content_aware_fill_dialog(selection_mask)
        if settings is None:
            return
        layer_id = layer.id
        generation = self._edit_generation
        pixels_revision = layer.pixels_revision
        before = layer.pixels.copy()
        ys, xs = np.where(selection_mask > 0)
        rect = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))

        def worker() -> np.ndarray:
            return content_aware_fill(
                before,
                selection_mask,
                int(settings["radius"]),
                np.asarray(settings["source_mask"], dtype=np.uint8),
                float(settings["color_adaptation"]),
                bool(settings["rotation_adaptation"]),
                bool(settings["scale_adaptation"]),
                int(settings["variant"]),
            )

        def done(after: np.ndarray) -> None:
            target = self.doc.get_layer(layer_id)
            if target is None:
                return
            x1, y1, x2, y2 = rect
            target.pixels[y1:y2, x1:x2] = after[y1:y2, x1:x2]
            target.touch_pixels()
            self.doc.dirty = True
            self.push_command(
                PixelPatchCommand(
                    "Заливка с учетом содержимого",
                    layer_id,
                    rect,
                    before[y1:y2, x1:x2].copy(),
                    after[y1:y2, x1:x2].copy(),
                )
            )
            self.invalidate_pixels()
            self.refresh()

        self.run_background(
            "Заливка с учетом содержимого",
            worker,
            done,
            lambda: self._edit_generation == generation
            and (target := self.doc.get_layer(layer_id)) is not None
            and target.pixels_revision == pixels_revision,
        )

    def content_aware_fill_dialog(self, selection_mask: np.ndarray) -> dict[str, object] | None:
        layer = self.doc.layer
        pixels = layer.pixels.copy()
        target = np.asarray(selection_mask, dtype=np.uint8)
        dialog = tk.Toplevel(self)
        dialog.title("Заливка с учетом содержимого")
        dialog.transient(self)
        dialog.grab_set()
        dialog.minsize(940, 640)
        result: dict[str, object] | None = None

        radius = tk.IntVar(value=6)
        color_adaptation = tk.DoubleVar(value=0.65)
        rotation_adaptation = tk.BooleanVar(value=True)
        scale_adaptation = tk.BooleanVar(value=True)
        brush_mode = tk.StringVar(value="Исключить")
        brush_size = tk.IntVar(value=48)
        variant = tk.IntVar(value=0)
        automatic_source = (pixels[:, :, 3] > 0) & ~(cv2.dilate((target > 0).astype(np.uint8), np.ones((13, 13), np.uint8)) > 0)
        source_mask = automatic_source.astype(np.uint8) * 255
        source_undo: list[np.ndarray] = []
        preview_variants: list[np.ndarray] = []
        preview_after: list[str | None] = [None]
        transform: dict[str, float] = {"scale": 1.0, "ox": 0.0, "oy": 0.0}

        header = ttk.Frame(dialog, padding=(12, 10, 12, 6))
        header.pack(fill=tk.X)
        ttk.Label(header, text="Заливка с учетом содержимого", style="PanelTitle.TLabel").pack(side=tk.LEFT)
        status = ttk.Label(header, style="Secondary.TLabel")
        status.pack(side=tk.RIGHT)
        body = ttk.PanedWindow(dialog, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=12)
        preview_panel = ttk.Frame(body)
        controls = ttk.Frame(body, width=250, padding=(12, 4))
        body.add(preview_panel, weight=1)
        body.add(controls, weight=0)

        canvases = ttk.Frame(preview_panel)
        canvases.pack(fill=tk.BOTH, expand=True)
        source_panel = ttk.Frame(canvases)
        result_panel = ttk.Frame(canvases)
        source_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        result_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
        ttk.Label(source_panel, text="Область-источник", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(0, 4))
        ttk.Label(result_panel, text="Предпросмотр результата", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(0, 4))
        source_canvas = tk.Canvas(source_panel, background=TOKENS.WORKSPACE_BG, highlightthickness=0, cursor="crosshair")
        result_canvas = tk.Canvas(result_panel, background=TOKENS.WORKSPACE_BG, highlightthickness=0)
        source_canvas.pack(fill=tk.BOTH, expand=True)
        result_canvas.pack(fill=tk.BOTH, expand=True)

        variants_bar = ttk.Frame(preview_panel, padding=(0, 8, 0, 0))
        variants_bar.pack(fill=tk.X)
        ttk.Label(variants_bar, text="Варианты:", style="PanelTitle.TLabel").pack(side=tk.LEFT, padx=(0, 8))
        variant_buttons: list[ttk.Button] = []

        ttk.Label(controls, text="Кисть источника", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(4, 3))
        mode_box = ttk.Combobox(controls, textvariable=brush_mode, values=["Добавить", "Исключить"], state="readonly")
        mode_box.pack(fill=tk.X)
        ttk.Label(controls, text="Размер кисти", style="Secondary.TLabel").pack(anchor=tk.W, pady=(8, 2))
        ttk.Scale(controls, variable=brush_size, from_=8, to=180).pack(fill=tk.X)
        source_actions = ttk.Frame(controls)
        source_actions.pack(fill=tk.X, pady=(8, 0))

        ttk.Label(controls, text="Поиск текстуры", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(18, 3))
        radius_value = ttk.Label(controls, style="Secondary.TLabel")
        radius_value.pack(anchor=tk.E)
        ttk.Scale(controls, variable=radius, from_=1, to=30).pack(fill=tk.X)
        ttk.Label(controls, text="Адаптация цвета", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(14, 3))
        color_value = ttk.Label(controls, style="Secondary.TLabel")
        color_value.pack(anchor=tk.E)
        ttk.Scale(controls, variable=color_adaptation, from_=0.0, to=1.0).pack(fill=tk.X)
        ttk.Checkbutton(controls, text="Подбирать поворот", variable=rotation_adaptation).pack(anchor=tk.W, pady=(14, 2))
        ttk.Checkbutton(controls, text="Подбирать масштаб", variable=scale_adaptation).pack(anchor=tk.W, pady=2)
        explanation = ttk.Label(
            controls,
            text="Зелёным отмечены пиксели, из которых разрешено брать текстуру. Красная область будет заменена.",
            wraplength=230,
            justify=tk.LEFT,
            style="Secondary.TLabel",
        )
        explanation.pack(fill=tk.X, pady=(18, 0))

        footer = ttk.Frame(dialog, padding=12)
        footer.pack(fill=tk.X)

        def preview_arrays() -> tuple[np.ndarray, np.ndarray, float]:
            max_side = 620
            scale = min(1.0, max_side / max(1, pixels.shape[0], pixels.shape[1]))
            size = (max(1, round(pixels.shape[1] * scale)), max(1, round(pixels.shape[0] * scale)))
            reduced_pixels = pixels if size == (pixels.shape[1], pixels.shape[0]) else cv2.resize(pixels, size, interpolation=cv2.INTER_AREA)
            reduced_target = target if size == (target.shape[1], target.shape[0]) else cv2.resize(target, size, interpolation=cv2.INTER_LINEAR)
            return reduced_pixels, reduced_target, scale

        def fitted_image(arr: np.ndarray, canvas: tk.Canvas) -> tuple[ImageTk.PhotoImage, float, float, float]:
            width = max(120, canvas.winfo_width())
            height = max(120, canvas.winfo_height())
            scale = min(width / max(1, arr.shape[1]), height / max(1, arr.shape[0]))
            size = (max(1, round(arr.shape[1] * scale)), max(1, round(arr.shape[0] * scale)))
            image = rgba_array_to_pil(arr)
            if image.size != size:
                image = image.resize(size, Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(image), scale, (width - size[0]) / 2.0, (height - size[1]) / 2.0

        def draw_previews() -> None:
            if not preview_variants:
                return
            reduced_pixels, reduced_target, reduction = preview_arrays()
            reduced_source = cv2.resize(source_mask, (reduced_pixels.shape[1], reduced_pixels.shape[0]), interpolation=cv2.INTER_NEAREST)
            source_display = reduced_pixels.copy().astype(np.float32)
            green = np.zeros_like(source_display)
            green[:, :, :] = (38, 210, 120, 255)
            source_alpha = (reduced_source.astype(np.float32) / 255.0 * 0.42)[:, :, None]
            source_display = source_display * (1.0 - source_alpha) + green * source_alpha
            red = np.zeros_like(source_display)
            red[:, :, :] = (235, 54, 72, 255)
            target_alpha = (reduced_target.astype(np.float32) / 255.0 * 0.52)[:, :, None]
            source_display = source_display * (1.0 - target_alpha) + red * target_alpha
            source_photo, canvas_scale, ox, oy = fitted_image(np.clip(source_display, 0, 255).astype(np.uint8), source_canvas)
            source_canvas.delete("all")
            source_canvas.create_image(ox, oy, image=source_photo, anchor=tk.NW)
            self._content_aware_source_photo = source_photo
            transform.update(scale=canvas_scale * reduction, ox=ox, oy=oy)

            current = preview_variants[min(max(0, variant.get()), len(preview_variants) - 1)]
            result_photo, _, result_ox, result_oy = fitted_image(current, result_canvas)
            result_canvas.delete("all")
            result_canvas.create_image(result_ox, result_oy, image=result_photo, anchor=tk.NW)
            self._content_aware_result_photo = result_photo
            for index, button in enumerate(variant_buttons):
                button.state(["disabled"] if index == variant.get() else ["!disabled"])

        def calculate_previews(*_args) -> None:
            preview_after[0] = None
            reduced_pixels, reduced_target, _ = preview_arrays()
            reduced_source = cv2.resize(source_mask, (reduced_pixels.shape[1], reduced_pixels.shape[0]), interpolation=cv2.INTER_NEAREST)
            preview_variants[:] = content_aware_fill_variants(
                reduced_pixels,
                reduced_target,
                reduced_source,
                int(radius.get()),
                float(color_adaptation.get()),
                bool(rotation_adaptation.get()),
                bool(scale_adaptation.get()),
                3,
            )
            radius_value.configure(text=f"{int(radius.get())} px")
            color_value.configure(text=f"{round(float(color_adaptation.get()) * 100)}%")
            status.configure(text=f"Источник: {np.count_nonzero(source_mask):,} px  |  Вариант {variant.get() + 1} из 3".replace(",", " "))
            draw_previews()

        def schedule_preview(*_args) -> None:
            if preview_after[0] is not None:
                try:
                    dialog.after_cancel(preview_after[0])
                except tk.TclError:
                    pass
            preview_after[0] = dialog.after(110, calculate_previews)

        def set_variant(index: int) -> None:
            variant.set(index)
            status.configure(text=f"Источник: {np.count_nonzero(source_mask):,} px  |  Вариант {index + 1} из 3".replace(",", " "))
            draw_previews()

        for index in range(3):
            button = ttk.Button(variants_bar, text=str(index + 1), width=5, command=lambda value=index: set_variant(value))
            button.pack(side=tk.LEFT, padx=3)
            variant_buttons.append(button)

        def reset_source() -> None:
            nonlocal source_mask
            source_undo.append(source_mask.copy())
            source_mask = automatic_source.astype(np.uint8) * 255
            schedule_preview()

        def undo_source() -> None:
            nonlocal source_mask
            if source_undo:
                source_mask = source_undo.pop()
                schedule_preview()

        ttk.Button(source_actions, text="Авто", command=reset_source).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        ttk.Button(source_actions, text="Отменить кисть", command=undo_source).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

        stroke_previous: list[tuple[int, int] | None] = [None]

        def source_point(event) -> tuple[int, int] | None:
            scale = max(0.0001, transform["scale"])
            x = round((event.x - transform["ox"]) / scale)
            y = round((event.y - transform["oy"]) / scale)
            if 0 <= x < pixels.shape[1] and 0 <= y < pixels.shape[0]:
                return x, y
            return None

        def paint_segment(start: tuple[int, int], end: tuple[int, int]) -> None:
            value = 255 if brush_mode.get() == "Добавить" else 0
            thickness = max(1, int(brush_size.get()))
            cv2.line(source_mask, start, end, value, thickness, cv2.LINE_AA)
            cv2.circle(source_mask, end, max(1, thickness // 2), value, -1, cv2.LINE_AA)
            source_mask[target > 0] = 0
            draw_previews()

        def brush_press(event) -> None:
            point = source_point(event)
            if point is None:
                return
            source_undo.append(source_mask.copy())
            stroke_previous[0] = point
            paint_segment(point, point)

        def brush_drag(event) -> None:
            point = source_point(event)
            if point is None or stroke_previous[0] is None:
                return
            paint_segment(stroke_previous[0], point)
            stroke_previous[0] = point

        def brush_release(_event=None) -> None:
            stroke_previous[0] = None
            schedule_preview()

        def accept() -> None:
            nonlocal result
            result = {
                "source_mask": source_mask.copy(),
                "radius": int(radius.get()),
                "color_adaptation": float(color_adaptation.get()),
                "rotation_adaptation": bool(rotation_adaptation.get()),
                "scale_adaptation": bool(scale_adaptation.get()),
                "variant": int(variant.get()),
            }
            dialog.destroy()

        ttk.Label(footer, text="Применяется только внутри текущего выделения", style="Secondary.TLabel").pack(side=tk.LEFT)
        ttk.Button(footer, text="Применить", command=accept, style="Primary.TButton").pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(footer, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT)
        source_canvas.bind("<ButtonPress-1>", brush_press)
        source_canvas.bind("<B1-Motion>", brush_drag)
        source_canvas.bind("<ButtonRelease-1>", brush_release)
        source_canvas.bind("<Configure>", lambda _event: draw_previews())
        result_canvas.bind("<Configure>", lambda _event: draw_previews())
        for variable in (radius, color_adaptation, rotation_adaptation, scale_adaptation):
            variable.trace_add("write", schedule_preview)
        self._content_aware_source_canvas = source_canvas
        self._content_aware_result_canvas = result_canvas
        self._content_aware_brush_mode = brush_mode
        self._content_aware_brush_size = brush_size
        self._content_aware_variant = variant
        self._content_aware_source_mask = lambda: source_mask.copy()
        self._content_aware_paint = paint_segment
        self._content_aware_accept = accept
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        self.center_toplevel(dialog, 1080, 700)
        calculate_previews()
        dialog.wait_window()
        return result

    def filter_edge_cleanup(self) -> None:
        layer = self.doc.layer
        selection_mask = self.doc.layer_selection_mask(layer)
        if selection_mask is None or not np.any(selection_mask):
            messagebox.showinfo("Очистка краев", "Сначала создайте выделение на активном слое.")
            return
        radius = simpledialog.askinteger("Очистка краев", "Радиус края:", initialvalue=3, minvalue=1, maxvalue=40)
        if radius is None:
            return
        strength = simpledialog.askfloat("Очистка краев", "Сила 0..1:", initialvalue=0.65, minvalue=0.0, maxvalue=1.0)
        if strength is not None:
            self.apply_to_layer("edge-aware cleanup", lambda arr: edge_aware_cleanup(arr, selection_mask, radius, strength))
