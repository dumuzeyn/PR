from __future__ import annotations

from ..app_shared import *


class TextVectorMixin:
    def edit_text_layer(self) -> None:
        layer = self.doc.layer
        if layer.kind != "text" or layer.text_data is None:
            messagebox.showinfo("Text layer", "Select a text layer first.")
            return
        self.refresh_properties()
        self.edit_active_text_on_canvas()

    def edit_text_path(self) -> None:
        if self._text_editor is not None:
            self.finish_text_edit()
        layer = self.doc.layer
        if layer.kind != "text" or layer.text_data is None:
            messagebox.showinfo("Текст по контуру", "Сначала выберите текстовый слой.")
            return
        before = copy.deepcopy(layer.text_data)
        working = copy.deepcopy(layer.text_data)
        size = max(4, int(working.get("size", 48)))
        points = normalize_text_path_points(
            working.get("path_points"),
            int(working.get("x", 0)),
            int(working.get("y", 0)) + size,
            max(int(working.get("box_width", 0) or 0), size * 8),
        )
        dialog = tk.Toplevel(self)
        dialog.title("Текст по контуру")
        dialog.transient(self)
        dialog.grab_set()
        dialog.minsize(840, 590)

        header = ttk.Frame(dialog, padding=(12, 10, 12, 6))
        header.pack(fill=tk.X)
        ttk.Label(header, text="Редактор контура текста", style="PanelTitle.TLabel").pack(side=tk.LEFT)
        ttk.Label(header, text="Перетаскивайте круги и квадратные направляющие", style="Secondary.TLabel").pack(side=tk.RIGHT)

        body = ttk.PanedWindow(dialog, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))
        preview = tk.Canvas(body, width=650, height=520, background=TOKENS.WORKSPACE, highlightthickness=1, highlightbackground=TOKENS.BORDER, cursor="crosshair")
        controls = ttk.Frame(body, width=235, padding=(12, 4))
        body.add(preview, weight=1)
        body.add(controls, weight=0)

        start_var = tk.DoubleVar(value=max(0.0, min(1.0, float(working.get("path_start", 0.0)))))
        end_var = tk.DoubleVar(value=max(0.0, min(1.0, float(working.get("path_end", 1.0)))))
        side_var = tk.IntVar(value=-1 if int(working.get("path_side", 1)) < 0 else 1)
        reverse_var = tk.BooleanVar(value=bool(working.get("path_reverse", False)))
        baseline_var = tk.IntVar(value=int(working.get("baseline_shift", 0)))
        active_point: list[int | None] = [None]
        transform = {"scale": 1.0, "ox": 0.0, "oy": 0.0}

        snapshot = self.document_copy()
        preview_layer = snapshot.get_layer(layer.id)
        if preview_layer is not None:
            preview_layer.visible = False
        base_image = rgba_array_to_pil(snapshot.composite(checker=True))
        scaled_base: dict[str, object] = {"size": None, "image": None}

        ttk.Label(controls, text="Участок контура", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(2, 5))
        start_label = ttk.Label(controls, text="Начало: 0%")
        start_label.pack(anchor=tk.W)
        start_scale = AccentScale(controls, variable=start_var, from_=0.0, to=1.0)
        start_scale.pack(fill=tk.X, pady=(0, 8))
        end_label = ttk.Label(controls, text="Конец: 100%")
        end_label.pack(anchor=tk.W)
        end_scale = AccentScale(controls, variable=end_var, from_=0.0, to=1.0)
        end_scale.pack(fill=tk.X, pady=(0, 12))

        ttk.Separator(controls).pack(fill=tk.X, pady=4)
        ttk.Label(controls, text="Сторона текста", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(7, 3))
        ttk.Radiobutton(controls, text="Над контуром", value=1, variable=side_var).pack(anchor=tk.W)
        ttk.Radiobutton(controls, text="Под контуром", value=-1, variable=side_var).pack(anchor=tk.W)
        ttk.Checkbutton(controls, text="Обратное направление", variable=reverse_var).pack(anchor=tk.W, pady=(6, 8))
        baseline_row = ttk.Frame(controls)
        baseline_row.pack(fill=tk.X, pady=4)
        ttk.Label(baseline_row, text="Смещение").pack(side=tk.LEFT)
        ttk.Spinbox(baseline_row, textvariable=baseline_var, from_=-500, to=500, width=8).pack(side=tk.RIGHT)

        ttk.Separator(controls).pack(fill=tk.X, pady=8)
        ttk.Label(controls, text="Форма", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(0, 4))
        preset_row = ttk.Frame(controls)
        preset_row.pack(fill=tk.X)

        footer = ttk.Frame(dialog, padding=(12, 0, 12, 12))
        footer.pack(fill=tk.X)
        ttk.Label(footer, text="Зелёная метка - начало, красная - конец текста", style="Secondary.TLabel").pack(side=tk.LEFT)

        def current_path_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            positions, tangents, cumulative = text_path_samples(points)
            if reverse_var.get():
                positions = positions[::-1].copy()
                tangents = (-tangents[::-1]).copy()
                segments = np.linalg.norm(np.diff(positions, axis=0), axis=1)
                cumulative = np.concatenate(([0.0], np.cumsum(segments)))
            return positions, tangents, cumulative

        def to_canvas(point: tuple[float, float] | list[float] | np.ndarray) -> tuple[float, float]:
            doc_x = float(point[0]) + layer.x
            doc_y = float(point[1]) + layer.y
            return transform["ox"] + doc_x * transform["scale"], transform["oy"] + doc_y * transform["scale"]

        def from_canvas(x: float, y: float) -> list[float]:
            return [
                (x - transform["ox"]) / max(1e-8, transform["scale"]) - layer.x,
                (y - transform["oy"]) / max(1e-8, transform["scale"]) - layer.y,
            ]

        def path_values() -> dict[str, object]:
            values = copy.deepcopy(working)
            values.update({
                "path_mode": "bezier",
                "path_points": [[float(point[0]), float(point[1])] for point in points],
                "path_start": float(start_var.get()),
                "path_end": float(end_var.get()),
                "path_side": int(side_var.get()),
                "path_reverse": bool(reverse_var.get()),
                "baseline_shift": int(baseline_var.get()),
            })
            return values

        def redraw(*_args) -> None:
            preview.update_idletasks()
            width = max(320, preview.winfo_width())
            height = max(300, preview.winfo_height())
            scale = min((width - 24) / max(1, self.doc.width), (height - 24) / max(1, self.doc.height))
            transform.update({"scale": scale, "ox": (width - self.doc.width * scale) / 2.0, "oy": (height - self.doc.height * scale) / 2.0})
            target_size = (max(1, round(self.doc.width * scale)), max(1, round(self.doc.height * scale)))
            if scaled_base["size"] != target_size:
                scaled_base["size"] = target_size
                scaled_base["image"] = base_image.resize(target_size, Image.Resampling.LANCZOS)
            frame = scaled_base["image"].copy()
            preview_data = path_values()
            preview_data["x"] = (float(preview_data.get("x", 0)) + layer.x) * scale
            preview_data["y"] = (float(preview_data.get("y", 0)) + layer.y) * scale
            preview_data["size"] = max(6, round(float(preview_data.get("size", 48)) * scale))
            preview_data["box_width"] = max(0, round(float(preview_data.get("box_width", 0) or 0) * scale))
            preview_data["tracking"] = round(float(preview_data.get("tracking", 0)) * scale)
            preview_data["baseline_shift"] = round(float(preview_data.get("baseline_shift", 0)) * scale)
            preview_data["path_points"] = [
                [(float(point[0]) + layer.x) * scale, (float(point[1]) + layer.y) * scale]
                for point in points
            ]
            temporary = Layer("Предпросмотр текста", np.zeros((target_size[1], target_size[0], 4), dtype=np.uint8), kind="text", text_data=preview_data)
            render_text_layer(temporary)
            text_image = rgba_array_to_pil(temporary.pixels)
            frame.paste(text_image, (0, 0), text_image)
            self._text_path_preview_image = ImageTk.PhotoImage(frame)
            preview.delete("all")
            preview.create_image(transform["ox"], transform["oy"], image=self._text_path_preview_image, anchor=tk.NW)
            canvas_points = [to_canvas(point) for point in points]
            preview.create_line(*canvas_points[0], *canvas_points[1], fill="#e8ad45", dash=(5, 4), width=2)
            preview.create_line(*canvas_points[2], *canvas_points[3], fill="#e8ad45", dash=(5, 4), width=2)
            positions, tangents, cumulative = current_path_geometry()
            curve = [to_canvas(point) for point in positions[::4]]
            preview.create_line(*[value for point in curve for value in point], fill="#33a6c8", width=3, smooth=True)
            total = float(cumulative[-1])
            for fraction, color, label in ((float(start_var.get()), "#4caf72", "НАЧАЛО"), (float(end_var.get()), "#e45b5b", "КОНЕЦ")):
                point, _tangent = text_path_point_at_distance(positions, tangents, cumulative, max(0.0, min(1.0, fraction)) * total)
                px, py = to_canvas(point)
                preview.create_polygon(px, py - 9, px + 8, py + 7, px - 8, py + 7, fill=color, outline="#111318")
                preview.create_text(px, py - 18, text=label, fill=color, font=("Segoe UI", 8, "bold"))
            for index, (px, py) in enumerate(canvas_points):
                color = "#33a6c8" if index in {0, 3} else "#e8ad45"
                if index in {0, 3}:
                    preview.create_oval(px - 8, py - 8, px + 8, py + 8, fill=color, outline="#111318", width=2)
                else:
                    preview.create_rectangle(px - 8, py - 8, px + 8, py + 8, fill=color, outline="#111318", width=2)
                preview.create_text(px, py + 18, text=f"P{index}", fill=color, font=("Segoe UI", 8, "bold"))
            start_label.configure(text=f"Начало: {round(float(start_var.get()) * 100)}%")
            end_label.configure(text=f"Конец: {round(float(end_var.get()) * 100)}%")

        def set_preset(kind: str) -> None:
            p0 = points[0]
            p3 = points[3]
            dx, dy = p3[0] - p0[0], p3[1] - p0[1]
            length = max(60.0, math.hypot(dx, dy))
            normal_x, normal_y = (-dy / length, dx / length) if length > 1e-8 else (0.0, 1.0)
            if kind == "line":
                points[1] = [p0[0] + dx / 3.0, p0[1] + dy / 3.0]
                points[2] = [p0[0] + dx * 2.0 / 3.0, p0[1] + dy * 2.0 / 3.0]
            elif kind == "arc":
                bend = length * 0.32
                points[1] = [p0[0] + dx / 3.0 - normal_x * bend, p0[1] + dy / 3.0 - normal_y * bend]
                points[2] = [p0[0] + dx * 2.0 / 3.0 - normal_x * bend, p0[1] + dy * 2.0 / 3.0 - normal_y * bend]
            else:
                bend = length * 0.38
                points[1] = [p0[0] + dx / 3.0 - normal_x * bend, p0[1] + dy / 3.0 - normal_y * bend]
                points[2] = [p0[0] + dx * 2.0 / 3.0 + normal_x * bend, p0[1] + dy * 2.0 / 3.0 + normal_y * bend]
            redraw()

        ttk.Button(preset_row, text="Прямая", command=lambda: set_preset("line")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(preset_row, text="Дуга", command=lambda: set_preset("arc")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(preset_row, text="S", command=lambda: set_preset("s")).pack(side=tk.LEFT, fill=tk.X, expand=True)

        def press(event) -> None:
            active_point[0] = None
            best = 15.0
            for index, point in enumerate(points):
                px, py = to_canvas(point)
                distance = math.hypot(event.x - px, event.y - py)
                if distance < best:
                    active_point[0], best = index, distance

        def drag(event) -> None:
            if active_point[0] is None:
                return
            points[int(active_point[0])] = from_canvas(event.x, event.y)
            redraw()

        def accept() -> None:
            start = float(start_var.get())
            end = float(end_var.get())
            if end - start < 0.01:
                messagebox.showerror("Текст по контуру", "Конец участка должен находиться правее начала минимум на 1%.", parent=dialog)
                return
            after = path_values()
            layer.text_data = copy.deepcopy(after)
            render_text_layer(layer)
            layer.touch_pixels()
            self.doc.dirty = True
            if before != after:
                self.push_command(TextDataCommand("Изменить контур текста", layer.id, before, copy.deepcopy(after), layer.name, layer.name))
            dialog.destroy()
            self.refresh()

        ttk.Button(footer, text="Применить", command=accept).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(footer, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT)
        preview.bind("<ButtonPress-1>", press)
        preview.bind("<B1-Motion>", drag)
        preview.bind("<Configure>", redraw)
        for variable in (start_var, end_var, side_var, reverse_var, baseline_var):
            variable.trace_add("write", redraw)
        ToolTip(preview, "Круги P0/P3 задают начало и конец контура. Квадраты P1/P2 меняют его изгиб.")
        self._text_path_canvas = preview
        self._text_path_start_var = start_var
        self._text_path_end_var = end_var
        self._text_path_side_var = side_var
        self._text_path_reverse_var = reverse_var
        self._text_path_points = points
        self._text_path_accept = accept
        self.center_toplevel(dialog, 940, 680)
        redraw()
        dialog.wait_window()

    def transform_text_box(self) -> None:
        layer = self.doc.layer
        if layer.kind != "text" or layer.text_data is None or not np.any(layer.pixels[:, :, 3]):
            messagebox.showinfo("Текстовый блок", "Выберите непустой текстовый слой.")
            return
        ys, xs = np.where(layer.pixels[:, :, 3] > 0)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)

        class TextPreview:
            pass

        preview = TextPreview()
        preview.x, preview.y = x1, y1
        preview.pixels = layer.pixels[y1:y2, x1:x2].copy()
        data = self.free_transform_dialog(preview)
        if data is None:
            return
        self.run_document_command(
            "Transform text box",
            lambda: self.doc.transform_active_text_box(
                int(data["x"]), int(data["y"]), int(data["width"]), int(data["height"]),
                float(data["angle"]), bool(data["flip_horizontal"]), bool(data["flip_vertical"]),
            ),
        )
        self.refresh()
