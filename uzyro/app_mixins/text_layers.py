from __future__ import annotations

from ..app_shared import *


class TextLayersMixin:
    def show_text_window(self, title: str, text: str) -> None:
        window = tk.Toplevel(self)
        window.title(title)
        window.geometry("720x520")
        frame = ttk.Frame(window)
        frame.pack(fill=tk.BOTH, expand=True)
        area = tk.Text(frame, wrap=tk.WORD)
        scroll = ttk.Scrollbar(frame, command=area.yview)
        area.configure(yscrollcommand=scroll.set)
        area.insert("1.0", text)
        area.configure(state=tk.DISABLED)
        area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def text_layer_dialog(self, title: str, initial: dict) -> dict | None:
        window = tk.Toplevel(self)
        window.title(title)
        window.geometry("640x580")
        window.transient(self)
        window.grab_set()
        result: dict | None = None

        frame = ttk.Frame(window, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        text_widget = tk.Text(frame, height=8, wrap=tk.WORD)
        text_widget.insert("1.0", str(initial.get("text", "")))
        text_widget.grid(row=0, column=0, columnspan=4, sticky="nsew", pady=(0, 8))

        size_var = tk.IntVar(value=int(initial.get("size", 48)))
        width_var = tk.IntVar(value=int(initial.get("box_width", 0) or 0))
        spacing_var = tk.IntVar(value=int(initial.get("line_spacing", max(2, size_var.get() // 5))))
        tracking_var = tk.IntVar(value=int(initial.get("tracking", 0)))
        align_var = tk.StringVar(value=str(initial.get("align", "left")))
        font_var = tk.StringVar(value=str(initial.get("font_family", "Arial")))
        bold_var = tk.BooleanVar(value=bool(initial.get("bold", False)))
        italic_var = tk.BooleanVar(value=bool(initial.get("italic", False)))
        underline_var = tk.BooleanVar(value=bool(initial.get("underline", False)))
        path_labels = {
            "Без контура": "none",
            "Дуга": "arc",
            "Волна": "wave",
            "Редактируемый контур": "bezier",
        }
        initial_path_mode = str(initial.get("path_mode", "none"))
        path_mode_var = tk.StringVar(value=next((label for label, mode in path_labels.items() if mode == initial_path_mode), "Без контура"))
        path_amount_var = tk.IntVar(value=int(initial.get("path_amount", 0)))
        baseline_var = tk.IntVar(value=int(initial.get("baseline_shift", 0)))
        rotation_var = tk.DoubleVar(value=float(initial.get("rotation", 0.0)))

        families = sorted(set(tkfont.families()))
        ttk.Label(frame, text="Шрифт").grid(row=1, column=0, sticky=tk.W)
        ttk.Combobox(frame, textvariable=font_var, values=families, state="normal").grid(row=1, column=1, columnspan=3, sticky="ew", pady=2)
        ttk.Label(frame, text="Размер").grid(row=2, column=0, sticky=tk.W)
        ttk.Spinbox(frame, from_=4, to=500, textvariable=size_var, width=8).grid(row=2, column=1, sticky="ew", pady=2)
        ttk.Label(frame, text="Ширина блока").grid(row=2, column=2, sticky=tk.W, padx=(10, 0))
        ttk.Spinbox(frame, from_=0, to=100000, textvariable=width_var, width=10).grid(row=2, column=3, sticky="ew", pady=2)
        ttk.Label(frame, text="Выравнивание").grid(row=3, column=0, sticky=tk.W)
        ttk.Combobox(frame, textvariable=align_var, values=["left", "center", "right"], state="readonly").grid(row=3, column=1, sticky="ew", pady=2)
        ttk.Label(frame, text="Межстрочный интервал").grid(row=3, column=2, sticky=tk.W, padx=(10, 0))
        ttk.Spinbox(frame, from_=0, to=500, textvariable=spacing_var, width=10).grid(row=3, column=3, sticky="ew", pady=2)
        ttk.Label(frame, text="Трекинг").grid(row=4, column=0, sticky=tk.W)
        ttk.Spinbox(frame, from_=-50, to=500, textvariable=tracking_var, width=8).grid(row=4, column=1, sticky="ew", pady=2)
        styles = ttk.Frame(frame)
        styles.grid(row=4, column=2, columnspan=2, sticky="w", padx=(10, 0))
        ttk.Checkbutton(styles, text="Жирный", variable=bold_var).pack(side=tk.LEFT)
        ttk.Checkbutton(styles, text="Курсив", variable=italic_var).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Checkbutton(styles, text="Подчеркнуть", variable=underline_var).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(frame, text="Текст по контуру").grid(row=5, column=0, sticky=tk.W)
        ttk.Combobox(frame, textvariable=path_mode_var, values=list(path_labels), state="readonly").grid(row=5, column=1, sticky="ew", pady=2)
        ttk.Label(frame, text="Изгиб").grid(row=5, column=2, sticky=tk.W, padx=(10, 0))
        ttk.Spinbox(frame, from_=-500, to=500, textvariable=path_amount_var, width=10).grid(row=5, column=3, sticky="ew", pady=2)
        ttk.Label(frame, text="Базовая линия").grid(row=6, column=0, sticky=tk.W)
        ttk.Spinbox(frame, from_=-500, to=500, textvariable=baseline_var, width=8).grid(row=6, column=1, sticky="ew", pady=2)
        ttk.Label(frame, text="Поворот").grid(row=6, column=2, sticky=tk.W, padx=(10, 0))
        ttk.Spinbox(frame, from_=-360, to=360, increment=1, textvariable=rotation_var, width=10).grid(row=6, column=3, sticky="ew", pady=2)

        buttons = ttk.Frame(frame)
        buttons.grid(row=7, column=0, columnspan=4, sticky="e", pady=(12, 0))

        def safe_int(var, default: int, minimum: int = 0) -> int:
            try:
                return max(minimum, int(var.get()))
            except (tk.TclError, ValueError):
                return default

        def accept() -> None:
            nonlocal result
            text = text_widget.get("1.0", "end-1c")
            size = safe_int(size_var, 48, 4)
            result = {
                "text": text,
                "size": size,
                "font_family": font_var.get().strip() or "Arial",
                "box_width": safe_int(width_var, 0, 0),
                "align": align_var.get() if align_var.get() in {"left", "center", "right"} else "left",
                "line_spacing": safe_int(spacing_var, max(2, size // 5), 0),
                "tracking": safe_int(tracking_var, 0, -50),
                "bold": bold_var.get(),
                "italic": italic_var.get(),
                "underline": underline_var.get(),
                "path_mode": path_labels.get(path_mode_var.get(), "none"),
                "path_amount": safe_int(path_amount_var, 0, -500),
                "path_points": copy.deepcopy(initial.get("path_points")),
                "path_start": float(initial.get("path_start", 0.0)),
                "path_end": float(initial.get("path_end", 1.0)),
                "path_side": int(initial.get("path_side", 1)),
                "path_reverse": bool(initial.get("path_reverse", False)),
                "baseline_shift": safe_int(baseline_var, 0, -500),
                "rotation": float(rotation_var.get()),
            }
            window.destroy()

        ttk.Button(buttons, text="Применить", command=accept).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Отмена", command=window.destroy).pack(side=tk.RIGHT)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(3, weight=1)
        frame.rowconfigure(0, weight=1)
        text_widget.focus_set()
        self.wait_window(window)
        return result

    def export_layers(self) -> None:
        dst = filedialog.askdirectory(title="Export layers folder")
        if not dst:
            return
        snapshot = self.document_copy()

        def worker():
            out_dir = Path(dst)
            out_dir.mkdir(parents=True, exist_ok=True)
            count = 0
            for i, layer in enumerate(snapshot.layers):
                safe = "".join(ch if ch.isalnum() or ch in " ._-" else "_" for ch in layer.name).strip() or "Layer"
                path = out_dir / f"{i:03d}_{safe}.png"
                rgba_array_to_pil(layer.pixels).save(path)
                count += 1
            return count

        self.run_background("Export layers", worker, lambda count: messagebox.showinfo("Export layers", f"Exported {count} layers."))

    def new_layer(self) -> None:
        self.doc.add_layer(f"Layer {len(self.doc.layers) + 1}")
        self.push_command(LayerInsertCommand("New layer", self.doc.active_layer, copy.deepcopy(self.doc.layer)))
        self.refresh()

    def duplicate_layer(self) -> None:
        self.doc.duplicate_active_layer()
        self.push_command(LayerInsertCommand("Duplicate layer", self.doc.active_layer, copy.deepcopy(self.doc.layer)))
        self.refresh()

    def delete_layer(self) -> None:
        if len(self.doc.layers) <= 1:
            return
        selected = set(getattr(self, "selected_layer_ids", set()))
        if len(selected) > 1:
            active_id = self.doc.layer.id
            deletions = [(index, copy.deepcopy(layer)) for index, layer in enumerate(self.doc.layers) if layer.id in selected]
            if len(deletions) >= len(self.doc.layers):
                deletions = deletions[1:]
            deleted_ids = {layer.id for _, layer in deletions}
            self.doc.layers = [layer for layer in self.doc.layers if layer.id not in deleted_ids]
            self.doc.active_layer = min(self.doc.active_layer, len(self.doc.layers) - 1)
            self.doc.dirty = True
            self.selected_layer_ids = {self.doc.layer.id}
            self.push_command(LayersDeleteCommand("Delete layers", deletions, active_id))
            self.refresh()
            return
        index = self.doc.active_layer
        deleted = copy.deepcopy(self.doc.layer)
        self.doc.delete_active_layer()
        self.selected_layer_ids = {self.doc.layer.id}
        self.push_command(LayerDeleteCommand("Delete layer", index, deleted))
        self.refresh()

    def rename_layer(self) -> None:
        name = simpledialog.askstring("Rename layer", "Name:", initialvalue=self.doc.layer.name)
        if not name:
            return

        self.set_layer_property("Rename layer", "name", name, affects_canvas=False)

    def move_layer(self, delta: int) -> None:
        i = self.doc.active_layer
        j = i + delta
        if 0 <= j < len(self.doc.layers):
            layer_id = self.doc.layer.id
            self.doc.layers[i], self.doc.layers[j] = self.doc.layers[j], self.doc.layers[i]
            self.doc.active_layer = j
            self.doc.dirty = True
            self.push_command(LayerReorderCommand("Layer reorder", layer_id, i, j))
            self.refresh()

    def move_layer_to(self, target: int) -> None:
        if not self.doc.layers:
            return
        before = self.doc.active_layer
        target = max(0, min(int(target), len(self.doc.layers) - 1))
        if target == before:
            return
        layer = self.doc.layers.pop(before)
        self.doc.layers.insert(target, layer)
        self.doc.active_layer = target
        self.doc.dirty = True
        self.push_command(LayerReorderCommand("Изменить порядок объекта", layer.id, before, target))
        self.refresh()

    def free_transform_layer(self) -> None:
        layer = self.doc.layer
        if layer.locked:
            self.status_text("Слой заблокирован")
            return
        data = self.transform_workspace_dialog(layer, "Свободная")
        if data is None:
            return
        self.run_document_command("Трансформация слоя", lambda: self.apply_transform_workspace_data(data))
        self.refresh()

    def apply_transform_workspace_data(self, data: dict[str, object]) -> None:
        layer = self.doc.layer
        mode = str(data.get("mode", "Свободная"))
        if mode == "Сбросить":
            self.doc.reset_active_layer_advanced_transform()
            return
        if mode == "Перспектива":
            self.doc.set_active_layer_advanced_transform("perspective", data["points"])
            return
        if mode == "Сетка":
            self.doc.set_active_layer_advanced_transform(
                "mesh", data["points"], int(data.get("rows", 4)), int(data.get("columns", 4)),
                data.get("row_positions"), data.get("column_positions"),
            )
            return
        if layer.transform_data is not None:
            self.doc.reset_active_layer_advanced_transform()
            layer = self.doc.layer
        x, y = int(data["x"]), int(data["y"])
        width, height = max(1, int(data["width"])), max(1, int(data["height"]))
        angle = float(data["angle"])
        flip_horizontal = bool(data["flip_horizontal"])
        flip_vertical = bool(data["flip_vertical"])
        if layer.kind == "text" and layer.text_data is not None:
            self.doc.transform_active_text_box(x - layer.x, y - layer.y, width, height, angle, flip_horizontal, flip_vertical)
            layer.touch_pixels()
            return
        if layer.kind == "shape" and layer.shape_data is not None:
            layer.shape_data = transform_shape_data_to_box(layer.shape_data, (x - layer.x, y - layer.y, x - layer.x + width, y - layer.y + height))
            render_shape_layer(layer)
            if abs(angle) > 0.001 or flip_horizontal or flip_vertical:
                corners = np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], dtype=np.float64)
                if flip_horizontal:
                    corners = corners[[1, 0, 3, 2]]
                if flip_vertical:
                    corners = corners[[3, 2, 1, 0]]
                if abs(angle) > 0.001:
                    center = np.array([x + width / 2.0, y + height / 2.0])
                    radians = math.radians(angle)
                    matrix = np.array([[math.cos(radians), -math.sin(radians)], [math.sin(radians), math.cos(radians)]])
                    corners = (corners - center) @ matrix.T + center
                self.doc.set_active_layer_advanced_transform("perspective", corners.tolist())
            layer.touch_pixels()
            self.doc.dirty = True
            return
        self.doc.transform_active_layer(x, y, width, height, angle, flip_horizontal, flip_vertical)
        layer.touch_pixels()
