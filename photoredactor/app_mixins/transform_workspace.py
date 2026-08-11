from __future__ import annotations

from ..app_shared import *
from ..preview_ops import preview_scale_for_bounds, transform_preview_pixels
from ..warp_grid import insert_grid_line, normalized_grid_positions, regular_grid_points, remove_grid_line
class TransformWorkspaceMixin:
    def transform_workspace_dialog(self, layer: Layer, initial_mode: str = "Свободная") -> dict[str, object] | None:
        dialog = tk.Toplevel(self)
        dialog.title("Трансформация")
        dialog.transient(self)
        dialog.grab_set()
        dialog.minsize(980, 650)
        result: dict[str, object] | None = None
        stored = layer.transform_data or {}
        rows = max(2, int(stored.get("rows", 4)))
        columns = max(2, int(stored.get("columns", 4)))
        row_positions = normalized_grid_positions(stored.get("row_positions"), rows)
        column_positions = normalized_grid_positions(stored.get("column_positions"), columns)
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
        selected_line: list[tuple[str, int] | None] = [None]
        loading = [False]
        preview_transform = {"scale": 1.0, "ox": 0.0, "oy": 0.0}
        drag_state: dict[str, object] = {"kind": None, "last": None}
        preview_sources: dict[float, np.ndarray] = {}
        redraw_after_id = [None]

        def rectangle_points() -> list[list[float]]:
            x, y = float(x_var.get()), float(y_var.get())
            width, height = max(1.0, float(width_var.get())), max(1.0, float(height_var.get()))
            return [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]

        def regular_mesh() -> list[list[float]]:
            x, y = float(x_var.get()), float(y_var.get())
            width, height = max(1.0, float(width_var.get())), max(1.0, float(height_var.get()))
            return regular_grid_points((x, y, width, height), row_positions, column_positions)

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
        topology = tk.StringVar(value=f"{rows} x {columns}" if rows == columns and rows in {3, 4, 5} else "Пользовательская")
        ttk.Label(mesh_controls, text="Плотность", style="Secondary.TLabel").pack(anchor=tk.W, pady=(9, 3))
        topology_box = ttk.Combobox(mesh_controls, textvariable=topology, values=["3 x 3", "4 x 4", "5 x 5", "Пользовательская"], state="readonly")
        topology_box.pack(fill=tk.X)
        split_row = ttk.Frame(mesh_controls); split_row.pack(fill=tk.X, pady=(8, 3))
        ttk.Button(split_row, text="Разделить ↔", command=lambda: split_grid("row")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(split_row, text="Разделить ↕", command=lambda: split_grid("column")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        ttk.Button(mesh_controls, text="Разделить крест-накрест", command=lambda: split_grid("cross")).pack(fill=tk.X, pady=3)
        ttk.Button(mesh_controls, text="Удалить выбранную линию", command=lambda: delete_grid_line()).pack(fill=tk.X, pady=3)
        grid_status = ttk.Label(mesh_controls, text="", style="Secondary.TLabel", wraplength=230)
        grid_status.pack(fill=tk.X, pady=(5, 0))

        def current_free_values() -> tuple[float, float, float, float, float]:
            try:
                return float(x_var.get()), float(y_var.get()), max(1.0, float(width_var.get())), max(1.0, float(height_var.get())), float(angle_var.get())
            except (tk.TclError, ValueError):
                return float(source_x), float(source_y), float(source.shape[1]), float(source.shape[0]), 0.0

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
            redraw_after_id[0] = None
            canvas_width, canvas_height = max(320, canvas.winfo_width()), max(300, canvas.winfo_height())
            active_points = free_handle_points().values() if mode.get() == "Свободная" else points
            bounds_x = [0.0, float(self.doc.width), *(float(point[0]) for point in active_points)]
            bounds_y = [0.0, float(self.doc.height), *(float(point[1]) for point in active_points)]
            min_x, max_x = min(bounds_x), max(bounds_x)
            min_y, max_y = min(bounds_y), max(bounds_y)
            scale = preview_scale_for_bounds((canvas_width, canvas_height), (min_x, min_y, max_x, max_y))
            preview_transform.update(scale=scale, ox=24 - min_x * scale, oy=24 - min_y * scale)
            try:
                output, output_x, output_y, pixel_scale = transform_preview_pixels(
                    source, preview_sources, mode.get(), min(1.0, scale), current_free_values(), points,
                    rows, columns, flip_horizontal.get(), flip_vertical.get(),
                    row_positions, column_positions,
                )
            except (ValueError, cv2.error, tk.TclError):
                return
            canvas.delete("all")
            doc_a = doc_to_preview((0, 0)); doc_b = doc_to_preview((self.doc.width, self.doc.height))
            canvas.create_rectangle(*doc_a, *doc_b, fill="#2c2f35", outline="#666c76")
            photo_image = rgba_array_to_pil(output)
            display_scale = scale / pixel_scale
            if abs(display_scale - 1.0) > 0.001:
                photo_image = photo_image.resize((max(1, round(output.shape[1] * display_scale)), max(1, round(output.shape[0] * display_scale))), Image.Resampling.BILINEAR)
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
                        active = selected_line[0] == ("row", row)
                        canvas.create_line(*(coordinate for point in line for coordinate in doc_to_preview(point)), fill=TOKENS.DANGER if active else TOKENS.ACCENT, width=3 if active else 1)
                    for column in range(columns):
                        line = [points[row * columns + column] for row in range(rows)]
                        active = selected_line[0] == ("column", column)
                        canvas.create_line(*(coordinate for point in line for coordinate in doc_to_preview(point)), fill=TOKENS.DANGER if active else TOKENS.ACCENT, width=3 if active else 1)
                for index, point in enumerate(points):
                    px, py = doc_to_preview(point)
                    canvas.create_oval(px - 5, py - 5, px + 5, py + 5, fill="#ffffff" if index != selected_node.get() else TOKENS.ACCENT, outline=TOKENS.ACCENT, width=1)
            status.configure(text=f"{layer.name}  |  {mode.get()}  |  {round(output.shape[1] / pixel_scale)} x {round(output.shape[0] / pixel_scale)} px")
            grid_status.configure(text=f"Сетка {rows} x {columns}. Кликните внутреннюю линию, чтобы удалить её.")

        def schedule_redraw(*_args) -> None:
            if redraw_after_id[0] is None:
                redraw_after_id[0] = dialog.after(33, redraw)

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
            schedule_redraw()

        def change_mode(*_args) -> None:
            nonlocal points
            selected_line[0] = None
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
            schedule_redraw()

        def apply_mesh_preset(*_args) -> None:
            nonlocal points
            points = regular_mesh()
            x, y, width, height, _ = current_free_values()
            selected = preset.get()
            for index, point in enumerate(points):
                row, column = divmod(index, columns)
                nx = column_positions[column]
                ny = row_positions[row]
                if selected == "Арка":
                    point[1] -= math.sin(nx * math.pi) * height * 0.2
                elif selected == "Выпуклость":
                    dx, dy = nx - 0.5, ny - 0.5
                    factor = 1.0 + 0.32 * max(0.0, 1.0 - math.hypot(dx, dy) * 1.6)
                    point[0] = x + width * (0.5 + dx * factor)
                    point[1] = y + height * (0.5 + dy * factor)
                elif selected == "Волна":
                    point[1] += math.sin(nx * math.tau) * height * 0.12
            load_selected_node(); schedule_redraw()

        def apply_topology(*_args) -> None:
            nonlocal points, rows, columns, row_positions, column_positions
            value = topology.get()
            if value == "Пользовательская":
                return
            size = int(value.split()[0])
            rows = columns = size
            row_positions = normalized_grid_positions(None, rows)
            column_positions = normalized_grid_positions(None, columns)
            points = regular_mesh()
            selected_node.set(0); selected_line[0] = None
            load_selected_node(); schedule_redraw()

        def split_grid(axis: str) -> None:
            nonlocal points, rows, columns, row_positions, column_positions
            index = min(max(0, selected_node.get()), len(points) - 1)
            row, column = divmod(index, columns)
            axes = ("row", "column") if axis == "cross" else (axis,)
            for current_axis in axes:
                interval = min(row, rows - 2) if current_axis == "row" else min(column, columns - 2)
                points, rows, columns, row_positions, column_positions, selected = insert_grid_line(
                    points, rows, columns, row_positions, column_positions, current_axis, interval
                )
                selected_node.set(selected)
                selected_line[0] = (current_axis, interval + 1)
            topology.set("Пользовательская")
            load_selected_node(); schedule_redraw()

        def delete_grid_line() -> None:
            nonlocal points, rows, columns, row_positions, column_positions
            if selected_line[0] is None:
                grid_status.configure(text="Сначала кликните внутреннюю линию сетки.")
                return
            axis, index = selected_line[0]
            try:
                points, rows, columns, row_positions, column_positions, selected = remove_grid_line(
                    points, rows, columns, row_positions, column_positions, axis, index
                )
            except ValueError:
                grid_status.configure(text="Крайнюю линию удалить нельзя.")
                return
            selected_node.set(selected); selected_line[0] = None
            topology.set("Пользовательская")
            load_selected_node(); schedule_redraw()

        def nearest_node(event) -> int | None:
            best, distance = None, 14.0
            for index, point in enumerate(points):
                px, py = doc_to_preview(point)
                current = math.hypot(event.x - px, event.y - py)
                if current < distance:
                    best, distance = index, current
            return best

        def nearest_grid_line(event) -> tuple[str, int] | None:
            best: tuple[str, int] | None = None
            best_distance = 10.0
            candidates = [
                ("row", index, points[index * columns:(index + 1) * columns]) for index in range(1, rows - 1)
            ] + [
                ("column", index, [points[row * columns + index] for row in range(rows)]) for index in range(1, columns - 1)
            ]
            target = np.array([event.x, event.y], dtype=np.float64)
            for axis, index, line in candidates:
                shown = [np.array(doc_to_preview(point), dtype=np.float64) for point in line]
                for start, end in zip(shown, shown[1:]):
                    segment = end - start
                    ratio = float(np.clip(np.dot(target - start, segment) / max(1e-8, np.dot(segment, segment)), 0.0, 1.0))
                    distance = float(np.linalg.norm(target - (start + segment * ratio)))
                    if distance < best_distance:
                        best, best_distance = (axis, index), distance
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
                    selected_line[0] = None
                    selected_node.set(index); load_selected_node(); schedule_redraw()
                    drag_state.update(kind=index, last=doc_point)
                elif mode.get() == "Сетка":
                    selected_line[0] = nearest_grid_line(event)
                    drag_state.update(kind=None, last=doc_point); schedule_redraw()

        def drag(event) -> None:
            nonlocal points
            current = preview_to_doc(event.x, event.y)
            kind = drag_state.get("kind")
            last = drag_state.get("last")
            if kind is None or last is None:
                return
            if mode.get() != "Свободная":
                points[int(kind)] = [current[0], current[1]]
                drag_state["last"] = current
                load_selected_node(); schedule_redraw(); return
            x, y, width, height, angle = current_free_values()
            shift = bool(event.state & 0x0001)
            control = bool(event.state & 0x0004)
            alt = bool(event.state & 0x0008)
            if kind == "move":
                dx, dy = current[0] - last[0], current[1] - last[1]
                if shift:
                    dx, dy = (dx, 0.0) if abs(dx) >= abs(dy) else (0.0, dy)
                x_var.set(x + dx); y_var.set(y + dy)
            elif kind == "rotate":
                center_x, center_y = x + width / 2.0, y + height / 2.0
                new_angle = math.degrees(math.atan2(current[1] - center_y, current[0] - center_x)) + 90.0
                angle_var.set(round(new_angle / 15.0) * 15.0 if shift else new_angle)
            elif control and str(kind) in {"nw", "ne", "se", "sw"}:
                handles = free_handle_points()
                points = [list(handles[name]) for name in ("nw", "ne", "se", "sw")]
                node_index = {"nw": 0, "ne": 1, "se": 2, "sw": 3}[str(kind)]
                points[node_index] = [current[0], current[1]]
                selected_node.set(node_index)
                drag_state["kind"] = node_index
                mode.set("Перспектива")
            else:
                box = resize_box_from_handle(
                    (round(x), round(y), round(x + width), round(y + height)), str(kind),
                    (round(current[0]), round(current[1])),
                    keep_proportions=bool(keep_ratio.get()) != shift, from_center=alt,
                )
                x_var.set(box[0]); y_var.set(box[1]); width_var.set(box[2] - box[0]); height_var.set(box[3] - box[1])
            drag_state["last"] = current

        def update_cursor(event) -> None:
            if mode.get() != "Свободная":
                canvas.configure(cursor="crosshair")
                return
            best, distance = None, 13.0
            for name, point in free_handle_points().items():
                px, py = doc_to_preview(point)
                current = math.hypot(event.x - px, event.y - py)
                if current < distance:
                    best, distance = name, current
            cursor = "exchange" if best == "rotate" else "sizing" if best else "fleur"
            canvas.configure(cursor=cursor)

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
                result = {
                    "mode": selected_mode, "points": [list(point) for point in points], "rows": rows, "columns": columns,
                    "row_positions": list(row_positions), "column_positions": list(column_positions),
                }
            close_dialog()

        def reset() -> None:
            nonlocal points, row_positions, column_positions
            x_var.set(source_x); y_var.set(source_y); width_var.set(source.shape[1]); height_var.set(source.shape[0]); angle_var.set(0.0)
            flip_horizontal.set(False); flip_vertical.set(False); preset.set("Без деформации")
            row_positions = normalized_grid_positions(None, rows); column_positions = normalized_grid_positions(None, columns)
            points = rectangle_points() if mode.get() == "Перспектива" else regular_mesh()
            selected_node.set(0); load_selected_node(); schedule_redraw()

        def remove_saved_transform() -> None:
            nonlocal result
            result = {"mode": "Сбросить"}
            close_dialog()

        def close_dialog() -> None:
            if redraw_after_id[0] is not None:
                dialog.after_cancel(redraw_after_id[0])
                redraw_after_id[0] = None
            dialog.destroy()

        ttk.Button(footer, text="Исходные узлы", command=reset).pack(side=tk.LEFT)
        if stored:
            ttk.Button(footer, text="Снять трансформацию", command=remove_saved_transform).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(footer, text="Применить", command=accept, style="Primary.TButton").pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(footer, text="Отмена", command=close_dialog).pack(side=tk.RIGHT)
        canvas.bind("<Configure>", schedule_redraw)
        canvas.bind("<ButtonPress-1>", press)
        canvas.bind("<B1-Motion>", drag)
        canvas.bind("<ButtonRelease-1>", lambda _event: drag_state.update(kind=None, last=None))
        canvas.bind("<Motion>", update_cursor)
        mode.trace_add("write", change_mode)
        show_grid.trace_add("write", schedule_redraw)
        preset.trace_add("write", apply_mesh_preset)
        topology.trace_add("write", apply_topology)
        node_x.trace_add("write", node_fields_changed); node_y.trace_add("write", node_fields_changed)
        for variable in (x_var, y_var, width_var, height_var, angle_var, flip_horizontal, flip_vertical):
            variable.trace_add("write", schedule_redraw)
        self._transform_workspace_canvas = canvas; self._transform_workspace_mode = mode
        self._transform_workspace_points = lambda: [list(point) for point in points]
        self._transform_workspace_selected_node = selected_node; self._transform_workspace_node_x = node_x
        self._transform_workspace_node_y = node_y
        self._transform_workspace_topology = topology; self._transform_workspace_grid_shape = lambda: (rows, columns)
        self._transform_workspace_grid_axes = lambda: (list(row_positions), list(column_positions)); self._transform_workspace_split_grid = split_grid
        self._transform_workspace_delete_grid_line = delete_grid_line; self._transform_workspace_select_grid_line = lambda axis, index: selected_line.__setitem__(0, (axis, index))
        self._transform_workspace_accept = accept; self._transform_workspace_reset = reset
        dialog.protocol("WM_DELETE_WINDOW", close_dialog)
        dialog.bind("<Return>", lambda _event: accept())
        dialog.bind("<Escape>", lambda _event: close_dialog())
        self.center_toplevel(dialog, 1080, 720)
        change_mode()
        dialog.wait_window()
        return result
