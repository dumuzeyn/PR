from __future__ import annotations

from ..app_shared import *


class TransformWorkspaceMixin:
    def transform_workspace_dialog(self, layer: Layer, initial_mode: str = "Свободная") -> dict[str, object] | None:
        dialog = tk.Toplevel(self)
        dialog.title("Трансформация")
        dialog.transient(self)
        dialog.grab_set()
        dialog.minsize(980, 650)
        result: dict[str, object] | None = None
        rows = columns = 4

        stored = layer.transform_data or {}
        if layer.transform_source is not None and stored:
            source = layer.transform_source.copy()
            source_x = int(stored.get("base_x", layer.x))
            source_y = int(stored.get("base_y", layer.y))
        else:
            visible = layer.pixels[:, :, 3] > 0
            if np.any(visible):
                ys, xs = np.where(visible)
                x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
            else:
                x1, y1, x2, y2 = 0, 0, layer.pixels.shape[1], layer.pixels.shape[0]
            source = layer.pixels[y1:y2, x1:x2].copy()
            source_x, source_y = layer.x + x1, layer.y + y1

        mode = tk.StringVar(value=initial_mode if initial_mode in {"Свободная", "Перспектива", "Сетка"} else "Свободная")
        x_var = tk.DoubleVar(value=float(source_x))
        y_var = tk.DoubleVar(value=float(source_y))
        width_var = tk.DoubleVar(value=float(source.shape[1]))
        height_var = tk.DoubleVar(value=float(source.shape[0]))
        angle_var = tk.DoubleVar(value=0.0)
        keep_ratio = tk.BooleanVar(value=True)
        flip_horizontal = tk.BooleanVar(value=False)
        flip_vertical = tk.BooleanVar(value=False)
        show_grid = tk.BooleanVar(value=True)
        node_x = tk.DoubleVar(value=float(source_x))
        node_y = tk.DoubleVar(value=float(source_y))
        preset = tk.StringVar(value="Без деформации")
        selected_node = tk.IntVar(value=0)
        loading = [False]
        preview_transform = {"scale": 1.0, "ox": 0.0, "oy": 0.0}
        drag_state: dict[str, object] = {"kind": None, "last": None}

        def rectangle_points() -> list[list[float]]:
            x, y = float(x_var.get()), float(y_var.get())
            width, height = max(1.0, float(width_var.get())), max(1.0, float(height_var.get()))
            return [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]

        def regular_mesh() -> list[list[float]]:
            x, y = float(x_var.get()), float(y_var.get())
            width, height = max(1.0, float(width_var.get())), max(1.0, float(height_var.get()))
            return [[x + width * column / (columns - 1), y + height * row / (rows - 1)] for row in range(rows) for column in range(columns)]

        stored_mode = str(stored.get("mode", ""))
        stored_points = stored.get("points")
        if isinstance(stored_points, list) and stored_mode in {"perspective", "mesh"}:
            base_x = float(stored.get("base_x", source_x))
            base_y = float(stored.get("base_y", source_y))
            points = [[float(point[0]) + base_x, float(point[1]) + base_y] for point in stored_points]
            expected = 4 if stored_mode == "perspective" else rows * columns
            if len(points) != expected:
                points = rectangle_points() if stored_mode == "perspective" else regular_mesh()
        else:
            points = rectangle_points() if mode.get() == "Перспектива" else regular_mesh()

        header = ttk.Frame(dialog, padding=(12, 10, 12, 6))
        header.pack(fill=tk.X)
        ttk.Label(header, text="Трансформация слоя", style="PanelTitle.TLabel").pack(side=tk.LEFT)
        status = ttk.Label(header, text=f"{layer.name}  |  {layer.kind}", style="Secondary.TLabel")
        status.pack(side=tk.RIGHT)
        body = ttk.PanedWindow(dialog, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=12)
        canvas = tk.Canvas(body, background="#202226", highlightthickness=0, cursor="crosshair")
        controls = ttk.Frame(body, width=260, padding=(12, 4))
        body.add(canvas, weight=1)
        body.add(controls, weight=0)
        footer = ttk.Frame(dialog, padding=12)
        footer.pack(fill=tk.X)

        ttk.Label(controls, text="Режим", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(4, 3))
        mode_box = ttk.Combobox(controls, textvariable=mode, values=["Свободная", "Перспектива", "Сетка"], state="readonly")
        mode_box.pack(fill=tk.X)
        ttk.Checkbutton(controls, text="Показывать сетку", variable=show_grid).pack(anchor=tk.W, pady=(10, 0))

        free_controls = ttk.Frame(controls)
        node_controls = ttk.Frame(controls)
        mesh_controls = ttk.Frame(controls)
        free_controls.pack(fill=tk.X, pady=(14, 0))
        fields: list[ttk.Spinbox] = []
        for label, variable in (("X", x_var), ("Y", y_var), ("Ширина", width_var), ("Высота", height_var), ("Поворот", angle_var)):
            row = ttk.Frame(free_controls)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=10).pack(side=tk.LEFT)
            field = ttk.Spinbox(row, textvariable=variable, from_=-100000, to=100000, increment=1, width=12)
            field.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            fields.append(field)
        ttk.Checkbutton(free_controls, text="Сохранять пропорции", variable=keep_ratio).pack(anchor=tk.W, pady=(8, 2))
        ttk.Checkbutton(free_controls, text="Отразить по горизонтали", variable=flip_horizontal).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(free_controls, text="Отразить по вертикали", variable=flip_vertical).pack(anchor=tk.W, pady=2)

        ttk.Label(node_controls, text="Выбранный узел", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(0, 5))
        for label, variable in (("X", node_x), ("Y", node_y)):
            row = ttk.Frame(node_controls)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=10).pack(side=tk.LEFT)
            ttk.Spinbox(row, textvariable=variable, from_=-100000, to=100000, increment=1, width=12).pack(side=tk.RIGHT, fill=tk.X, expand=True)

        ttk.Label(mesh_controls, text="Форма сетки", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(0, 4))
        preset_box = ttk.Combobox(mesh_controls, textvariable=preset, values=["Без деформации", "Арка", "Выпуклость", "Волна"], state="readonly")
        preset_box.pack(fill=tk.X)

        def current_free_values() -> tuple[float, float, float, float, float]:
            try:
                return float(x_var.get()), float(y_var.get()), max(1.0, float(width_var.get())), max(1.0, float(height_var.get())), float(angle_var.get())
            except (tk.TclError, ValueError):
                return float(source_x), float(source_y), float(source.shape[1]), float(source.shape[0]), 0.0

        def transformed_preview() -> tuple[np.ndarray, float, float]:
            selected_mode = mode.get()
            if selected_mode == "Свободная":
                x, y, width, height, angle = current_free_values()
                image = rgba_array_to_pil(source).resize((max(1, round(width)), max(1, round(height))), Image.Resampling.BICUBIC)
                if flip_horizontal.get():
                    image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if flip_vertical.get():
                    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                if abs(angle) > 0.001:
                    image = image.rotate(-angle, expand=True, resample=Image.Resampling.BICUBIC)
                return np.asarray(image, dtype=np.uint8), x + (width - image.width) / 2.0, y + (height - image.height) / 2.0
            if selected_mode == "Перспектива":
                output, offset = perspective_warp_pixels(source, points, cv2.INTER_CUBIC)
            else:
                output, offset = mesh_warp_pixels(source, points, rows, columns, cv2.INTER_CUBIC)
            return output, float(offset[0]), float(offset[1])

        def doc_to_preview(point: tuple[float, float] | list[float]) -> tuple[float, float]:
            return preview_transform["ox"] + float(point[0]) * preview_transform["scale"], preview_transform["oy"] + float(point[1]) * preview_transform["scale"]

        def preview_to_doc(x: float, y: float) -> tuple[float, float]:
            scale = max(1e-8, preview_transform["scale"])
            return (x - preview_transform["ox"]) / scale, (y - preview_transform["oy"]) / scale

        def free_handle_points() -> dict[str, tuple[float, float]]:
            x, y, width, height, angle = current_free_values()
            raw = {
                "nw": (x, y), "n": (x + width / 2.0, y), "ne": (x + width, y),
                "e": (x + width, y + height / 2.0), "se": (x + width, y + height),
                "s": (x + width / 2.0, y + height), "sw": (x, y + height), "w": (x, y + height / 2.0),
            }
            center = np.array([x + width / 2.0, y + height / 2.0])
            radians = math.radians(angle)
            matrix = np.array([[math.cos(radians), -math.sin(radians)], [math.sin(radians), math.cos(radians)]])
            rotated = {name: tuple((np.array(point) - center) @ matrix.T + center) for name, point in raw.items()}
            rotation = (np.array([x + width / 2.0, y - max(24.0, height * 0.12)]) - center) @ matrix.T + center
            rotated["rotate"] = tuple(rotation)
            return rotated

        def redraw(*_args) -> None:
            try:
                output, output_x, output_y = transformed_preview()
            except (ValueError, cv2.error, tk.TclError):
                return
            canvas_width, canvas_height = max(320, canvas.winfo_width()), max(300, canvas.winfo_height())
            bounds_x = [0.0, float(self.doc.width), output_x, output_x + output.shape[1]]
            bounds_y = [0.0, float(self.doc.height), output_y, output_y + output.shape[0]]
            active_points = free_handle_points().values() if mode.get() == "Свободная" else points
            for point in active_points:
                bounds_x.append(float(point[0])); bounds_y.append(float(point[1]))
            min_x, max_x = min(bounds_x), max(bounds_x)
            min_y, max_y = min(bounds_y), max(bounds_y)
            scale = min((canvas_width - 48) / max(1.0, max_x - min_x), (canvas_height - 48) / max(1.0, max_y - min_y))
            preview_transform.update(scale=scale, ox=24 - min_x * scale, oy=24 - min_y * scale)
            canvas.delete("all")
            doc_a = doc_to_preview((0, 0)); doc_b = doc_to_preview((self.doc.width, self.doc.height))
            canvas.create_rectangle(*doc_a, *doc_b, fill="#2c2f35", outline="#666c76")
            photo_image = rgba_array_to_pil(output)
            if abs(scale - 1.0) > 0.001:
                photo_image = photo_image.resize((max(1, round(output.shape[1] * scale)), max(1, round(output.shape[0] * scale))), Image.Resampling.LANCZOS)
            self._transform_workspace_photo = ImageTk.PhotoImage(photo_image)
            image_x, image_y = doc_to_preview((output_x, output_y))
            canvas.create_image(image_x, image_y, image=self._transform_workspace_photo, anchor=tk.NW)
            if mode.get() == "Свободная":
                handles = free_handle_points()
                ordered = [handles[name] for name in ("nw", "ne", "se", "sw", "nw")]
                canvas.create_line(*(coordinate for point in ordered for coordinate in doc_to_preview(point)), fill=TOKENS.ACCENT, width=2)
                top = doc_to_preview(handles["n"]); rotation = doc_to_preview(handles["rotate"])
                canvas.create_line(*top, *rotation, fill=TOKENS.ACCENT, width=1)
                for name, point in handles.items():
                    px, py = doc_to_preview(point)
                    radius_px = 6 if name == "rotate" else 5
                    canvas.create_oval(px - radius_px, py - radius_px, px + radius_px, py + radius_px, fill="#ffffff", outline=TOKENS.ACCENT, width=2, tags=("transform-node", name))
            elif mode.get() == "Перспектива":
                ordered = [*points, points[0]]
                canvas.create_line(*(coordinate for point in ordered for coordinate in doc_to_preview(point)), fill=TOKENS.ACCENT, width=2)
                for index, point in enumerate(points):
                    px, py = doc_to_preview(point)
                    canvas.create_oval(px - 6, py - 6, px + 6, py + 6, fill="#ffffff" if index != selected_node.get() else TOKENS.ACCENT, outline=TOKENS.ACCENT, width=2)
            else:
                if show_grid.get():
                    for row in range(rows):
                        line = points[row * columns:(row + 1) * columns]
                        canvas.create_line(*(coordinate for point in line for coordinate in doc_to_preview(point)), fill=TOKENS.ACCENT, width=1)
                    for column in range(columns):
                        line = [points[row * columns + column] for row in range(rows)]
                        canvas.create_line(*(coordinate for point in line for coordinate in doc_to_preview(point)), fill=TOKENS.ACCENT, width=1)
                for index, point in enumerate(points):
                    px, py = doc_to_preview(point)
                    canvas.create_oval(px - 5, py - 5, px + 5, py + 5, fill="#ffffff" if index != selected_node.get() else TOKENS.ACCENT, outline=TOKENS.ACCENT, width=1)
            status.configure(text=f"{layer.name}  |  {mode.get()}  |  {output.shape[1]} x {output.shape[0]} px")

        def load_selected_node() -> None:
            if mode.get() == "Свободная" or not points:
                return
            index = min(max(0, selected_node.get()), len(points) - 1)
            loading[0] = True
            node_x.set(round(points[index][0], 2)); node_y.set(round(points[index][1], 2))
            loading[0] = False

        def node_fields_changed(*_args) -> None:
            if loading[0] or mode.get() == "Свободная" or not points:
                return
            try:
                index = min(max(0, selected_node.get()), len(points) - 1)
                points[index] = [float(node_x.get()), float(node_y.get())]
            except (tk.TclError, ValueError):
                return
            redraw()

        def change_mode(*_args) -> None:
            nonlocal points
            free_controls.pack_forget(); node_controls.pack_forget(); mesh_controls.pack_forget()
            if mode.get() == "Свободная":
                free_controls.pack(fill=tk.X, pady=(14, 0))
            else:
                node_controls.pack(fill=tk.X, pady=(14, 0))
                if mode.get() == "Перспектива":
                    if len(points) != 4:
                        points = rectangle_points()
                else:
                    mesh_controls.pack(fill=tk.X, pady=(14, 0))
                    if len(points) != rows * columns:
                        points = regular_mesh()
                selected_node.set(0)
                load_selected_node()
            redraw()

        def apply_mesh_preset(*_args) -> None:
            nonlocal points
            points = regular_mesh()
            x, y, width, height, _ = current_free_values()
            selected = preset.get()
            for index, point in enumerate(points):
                row, column = divmod(index, columns)
                nx = column / (columns - 1)
                ny = row / (rows - 1)
                if selected == "Арка":
                    point[1] -= math.sin(nx * math.pi) * height * 0.2
                elif selected == "Выпуклость":
                    dx, dy = nx - 0.5, ny - 0.5
                    factor = 1.0 + 0.32 * max(0.0, 1.0 - math.hypot(dx, dy) * 1.6)
                    point[0] = x + width * (0.5 + dx * factor)
                    point[1] = y + height * (0.5 + dy * factor)
                elif selected == "Волна":
                    point[1] += math.sin(nx * math.tau) * height * 0.12
            load_selected_node(); redraw()

        def nearest_node(event) -> int | None:
            best, distance = None, 14.0
            for index, point in enumerate(points):
                px, py = doc_to_preview(point)
                current = math.hypot(event.x - px, event.y - py)
                if current < distance:
                    best, distance = index, current
            return best

        def press(event) -> None:
            doc_point = preview_to_doc(event.x, event.y)
            if mode.get() == "Свободная":
                handles = free_handle_points()
                best, distance = None, 14.0
                for name, point in handles.items():
                    px, py = doc_to_preview(point)
                    current = math.hypot(event.x - px, event.y - py)
                    if current < distance:
                        best, distance = name, current
                if best is None:
                    x, y, width, height, _ = current_free_values()
                    if x <= doc_point[0] <= x + width and y <= doc_point[1] <= y + height:
                        best = "move"
                drag_state.update(kind=best, last=doc_point)
            else:
                index = nearest_node(event)
                if index is not None:
                    selected_node.set(index); load_selected_node(); redraw()
                drag_state.update(kind=index, last=doc_point)

        def drag(event) -> None:
            current = preview_to_doc(event.x, event.y)
            kind = drag_state.get("kind")
            last = drag_state.get("last")
            if kind is None or last is None:
                return
            if mode.get() != "Свободная":
                points[int(kind)] = [current[0], current[1]]
                drag_state["last"] = current
                load_selected_node(); redraw(); return
            x, y, width, height, angle = current_free_values()
            if kind == "move":
                x_var.set(x + current[0] - last[0]); y_var.set(y + current[1] - last[1])
            elif kind == "rotate":
                center_x, center_y = x + width / 2.0, y + height / 2.0
                angle_var.set(math.degrees(math.atan2(current[1] - center_y, current[0] - center_x)) + 90.0)
            else:
                box = resize_box_from_handle((round(x), round(y), round(x + width), round(y + height)), str(kind), (round(current[0]), round(current[1])), keep_proportions=keep_ratio.get())
                x_var.set(box[0]); y_var.set(box[1]); width_var.set(box[2] - box[0]); height_var.set(box[3] - box[1])
            drag_state["last"] = current

        def accept() -> None:
            nonlocal result
            selected_mode = mode.get()
            if selected_mode == "Свободная":
                x, y, width, height, angle = current_free_values()
                result = {
                    "mode": selected_mode, "x": round(x), "y": round(y), "width": round(width), "height": round(height),
                    "angle": angle, "flip_horizontal": flip_horizontal.get(), "flip_vertical": flip_vertical.get(),
                }
            else:
                result = {"mode": selected_mode, "points": [list(point) for point in points], "rows": rows, "columns": columns}
            dialog.destroy()

        def reset() -> None:
            nonlocal points
            x_var.set(source_x); y_var.set(source_y); width_var.set(source.shape[1]); height_var.set(source.shape[0]); angle_var.set(0.0)
            flip_horizontal.set(False); flip_vertical.set(False); preset.set("Без деформации")
            points = rectangle_points() if mode.get() == "Перспектива" else regular_mesh()
            selected_node.set(0); load_selected_node(); redraw()

        def remove_saved_transform() -> None:
            nonlocal result
            result = {"mode": "Сбросить"}
            dialog.destroy()

        ttk.Button(footer, text="Исходные узлы", command=reset).pack(side=tk.LEFT)
        if stored:
            ttk.Button(footer, text="Снять трансформацию", command=remove_saved_transform).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(footer, text="Применить", command=accept, style="Primary.TButton").pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(footer, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT)
        canvas.bind("<Configure>", redraw)
        canvas.bind("<ButtonPress-1>", press)
        canvas.bind("<B1-Motion>", drag)
        canvas.bind("<ButtonRelease-1>", lambda _event: drag_state.update(kind=None, last=None))
        mode.trace_add("write", change_mode)
        show_grid.trace_add("write", redraw)
        preset.trace_add("write", apply_mesh_preset)
        node_x.trace_add("write", node_fields_changed); node_y.trace_add("write", node_fields_changed)
        for variable in (x_var, y_var, width_var, height_var, angle_var, flip_horizontal, flip_vertical):
            variable.trace_add("write", redraw)
        self._transform_workspace_canvas = canvas
        self._transform_workspace_mode = mode
        self._transform_workspace_points = lambda: [list(point) for point in points]
        self._transform_workspace_selected_node = selected_node
        self._transform_workspace_node_x = node_x
        self._transform_workspace_node_y = node_y
        self._transform_workspace_accept = accept
        self._transform_workspace_reset = reset
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        self.center_toplevel(dialog, 1080, 720)
        change_mode()
        dialog.wait_window()
        return result

    def transform_selected_pixels(self) -> None:
        layer = self.doc.layer
        if layer.locked:
            self.status_text("Слой заблокирован")
            return
        selection = self.doc.layer_selection_mask(layer)
        if selection is None or not np.any(selection):
            messagebox.showinfo("Трансформация", "Сначала создайте выделение на активном слое.")
            return
        ys, xs = np.where(selection > 0)
        lx1, ly1, lx2, ly2 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
        patch = layer.pixels[ly1:ly2, lx1:lx2].copy()
        patch[:, :, 3] = np.clip(patch[:, :, 3].astype(np.float32) * (selection[ly1:ly2, lx1:lx2].astype(np.float32) / 255.0), 0, 255).astype(np.uint8)
        preview = Layer("Выделенные пиксели", patch, x=layer.x + lx1, y=layer.y + ly1)
        data = self.transform_workspace_dialog(preview, "Свободная")
        if data is None:
            return
        def edit() -> None:
            if data["mode"] == "Свободная":
                self.doc.transform_selected_pixels(
                    int(data["x"]), int(data["y"]), int(data["width"]), int(data["height"]),
                    float(data["angle"]), bool(data["flip_horizontal"]), bool(data["flip_vertical"]),
                )
            else:
                self.doc.transform_selected_pixels_advanced(
                    "perspective" if data["mode"] == "Перспектива" else "mesh",
                    data["points"], int(data.get("rows", 4)), int(data.get("columns", 4)),
                )
        self.run_document_command("Трансформация выделенных пикселей", edit)
        self.refresh()

    def perspective_transform_layer(self) -> None:
        layer = self.doc.layer
        if layer.locked:
            self.status_text("Слой заблокирован")
            return
        data = self.transform_workspace_dialog(layer, "Перспектива")
        if data is None:
            return
        self.run_document_command("Перспективная трансформация", lambda: self.apply_transform_workspace_data(data))
        self.refresh()

    def warp_layer(self) -> None:
        layer = self.doc.layer
        if layer.locked:
            self.status_text("Слой заблокирован")
            return
        data = self.transform_workspace_dialog(layer, "Сетка")
        if data is None:
            return
        self.run_document_command("Деформация слоя", lambda: self.apply_transform_workspace_data(data))
        self.refresh()

    def warp_layer_dialog(self, layer) -> dict[str, object] | None:
        dialog = tk.Toplevel(self)
        dialog.title("Деформация слоя")
        dialog.transient(self)
        dialog.resizable(False, False)
        dialog.grab_set()
        result: dict[str, object] | None = None
        mode = tk.StringVar(value="arc")
        amount = tk.DoubleVar(value=0.35)
        wavelength = tk.DoubleVar(value=96.0)
        preview = ttk.Label(dialog)
        preview.grid(row=0, column=0, rowspan=5, padx=12, pady=12)
        ttk.Label(dialog, text="Режим").grid(row=0, column=1, sticky="w", pady=(12, 2))
        mode_box = ttk.Combobox(dialog, textvariable=mode, values=["arc", "arc_vertical", "bulge", "pinch", "wave_x", "wave_y"], state="readonly", width=18)
        mode_box.grid(row=0, column=2, sticky="ew", padx=(8, 12), pady=(12, 2))
        ttk.Label(dialog, text="Сила").grid(row=1, column=1, sticky="w")
        ttk.Scale(dialog, from_=-1.0, to=1.0, variable=amount, orient=tk.HORIZONTAL).grid(row=1, column=2, sticky="ew", padx=(8, 12))
        ttk.Label(dialog, text="Длина волны").grid(row=2, column=1, sticky="w")
        ttk.Scale(dialog, from_=8, to=512, variable=wavelength, orient=tk.HORIZONTAL).grid(row=2, column=2, sticky="ew", padx=(8, 12))
        values = ttk.Label(dialog, text="")
        values.grid(row=3, column=1, columnspan=2, sticky="w", pady=(6, 0))
        source = rgba_array_to_pil(layer.pixels)
        source.thumbnail((220, 220), Image.Resampling.LANCZOS)
        source_array = np.array(source.convert("RGBA"), dtype=np.uint8)

        def update_preview(*_args) -> None:
            try:
                shown = warp_pixels(source_array, mode.get(), float(amount.get()), float(wavelength.get()), cv2.INTER_CUBIC)
            except (tk.TclError, ValueError):
                return
            canvas = Image.new("RGBA", (220, 220), (44, 46, 52, 255))
            image = rgba_array_to_pil(shown)
            canvas.alpha_composite(image, ((220 - image.width) // 2, (220 - image.height) // 2))
            self._warp_preview_image = ImageTk.PhotoImage(canvas)
            preview.configure(image=self._warp_preview_image)
            values.configure(text=f"Сила: {amount.get():.2f}   Волна: {wavelength.get():.0f}")

        buttons = ttk.Frame(dialog)
        buttons.grid(row=4, column=1, columnspan=2, sticky="e", padx=12, pady=12)

        def accept() -> None:
            nonlocal result
            result = {"mode": mode.get(), "amount": float(amount.get()), "wavelength": float(wavelength.get())}
            dialog.destroy()

        ttk.Button(buttons, text="ОК", command=accept).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT)
        mode_box.bind("<<ComboboxSelected>>", update_preview)
        amount.trace_add("write", update_preview)
        wavelength.trace_add("write", update_preview)
        update_preview()
        dialog.wait_window()
        return result
