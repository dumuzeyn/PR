from __future__ import annotations

from ..app_shared import *


class BooleanShapesMixin:
    def boolean_shape_editor(self, initial: dict[str, object], title: str) -> dict[str, object] | None:
        dialog = tk.Toplevel(self)
        dialog.title(title)
        dialog.geometry("920x650")
        dialog.minsize(820, 580)
        dialog.transient(self)
        dialog.grab_set()
        working = copy.deepcopy(initial)
        result: dict[str, object] | None = None
        mode_labels = {
            "Объединение": "union",
            "Вычитание": "subtract",
            "Пересечение": "intersect",
            "Исключение": "xor",
        }
        shape_labels = {
            "rectangle": "Прямоугольник", "ellipse": "Эллипс", "line": "Линия", "bezier": "Кривая Безье",
            "polygon": "Многоугольник", "star": "Звезда", "custom": "Своя фигура", "boolean": "Булева фигура",
        }
        mode_name = next((label for label, value in mode_labels.items() if value == working.get("boolean_mode")), "Объединение")
        mode_var = tk.StringVar(value=mode_name)

        top = ttk.Frame(dialog)
        top.pack(fill=tk.X, padx=12, pady=(12, 8))
        ttk.Label(top, text="Операция", style="PanelTitle.TLabel").pack(side=tk.LEFT, padx=(0, 10))
        for label in mode_labels:
            ttk.Radiobutton(top, text=label, value=label, variable=mode_var).pack(side=tk.LEFT, padx=3)

        body = ttk.PanedWindow(dialog, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=12)
        left = ttk.Frame(body, width=260)
        right = ttk.Frame(body)
        body.add(left, weight=0)
        body.add(right, weight=1)

        ttk.Label(left, text="Исходные контуры", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(0, 5))
        contour_list = tk.Listbox(left, exportselection=False, activestyle="dotbox", height=14)
        contour_list.pack(fill=tk.BOTH, expand=True)
        list_buttons = ttk.Frame(left)
        list_buttons.pack(fill=tk.X, pady=6)

        preview = tk.Canvas(right, height=330, background="#252a30", highlightthickness=1, highlightbackground=TOKENS.BORDER)
        preview.pack(fill=tk.BOTH, expand=True, padx=(10, 0))
        editor = ttk.LabelFrame(right, text="Параметры выбранного контура")
        editor.pack(fill=tk.X, padx=(10, 0), pady=(8, 0))

        name_var = tk.StringVar()
        enabled_var = tk.BooleanVar(value=True)
        kind_var = tk.StringVar()
        x_var = tk.IntVar()
        y_var = tk.IntVar()
        width_var = tk.IntVar(value=1)
        height_var = tk.IntVar(value=1)
        sides_var = tk.IntVar(value=5)
        ratio_var = tk.DoubleVar(value=0.5)
        loading = [False]
        preview_state: dict[str, float] = {"scale": 1.0, "ox": 0.0, "oy": 0.0}
        preview_drag: dict[str, object] = {"index": None, "last": (0, 0)}

        fields = [
            ("Имя", lambda parent: ttk.Entry(parent, textvariable=name_var, width=22)),
            ("Тип", lambda parent: ttk.Label(parent, textvariable=kind_var, style="Secondary.TLabel")),
            ("X", lambda parent: ttk.Spinbox(parent, textvariable=x_var, from_=-100000, to=100000, width=10)),
            ("Y", lambda parent: ttk.Spinbox(parent, textvariable=y_var, from_=-100000, to=100000, width=10)),
            ("Ширина", lambda parent: ttk.Spinbox(parent, textvariable=width_var, from_=1, to=100000, width=10)),
            ("Высота", lambda parent: ttk.Spinbox(parent, textvariable=height_var, from_=1, to=100000, width=10)),
            ("Стороны/лучи", lambda parent: ttk.Spinbox(parent, textvariable=sides_var, from_=3, to=64, width=10)),
            ("Внутр. радиус", lambda parent: ttk.Spinbox(parent, textvariable=ratio_var, from_=0.05, to=0.95, increment=0.05, width=10)),
        ]
        field_widgets: list[tk.Widget] = []
        for index, (label, create_widget) in enumerate(fields):
            row, column = divmod(index, 4)
            slot = ttk.Frame(editor)
            slot.grid(row=row, column=column, sticky="ew", padx=6, pady=5)
            ttk.Label(slot, text=label, style="Secondary.TLabel").pack(anchor=tk.W)
            widget = create_widget(slot)
            widget.pack(fill=tk.X)
            field_widgets.append(widget)
            editor.columnconfigure(column, weight=1)
        ttk.Checkbutton(editor, text="Участвует в операции", variable=enabled_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=6, pady=5)

        def children() -> list[dict[str, object]]:
            raw = working.setdefault("children", [])
            return raw if isinstance(raw, list) else []

        def selected_index() -> int | None:
            selected = contour_list.curselection()
            return int(selected[0]) if selected else None

        def update_bounds() -> None:
            working["box"] = list(shape_data_bounds(working) or (0, 0, 1, 1))

        def preview_image() -> None:
            working["boolean_mode"] = mode_labels[mode_var.get()]
            update_bounds()
            pixels = np.zeros((self.doc.height, self.doc.width, 4), dtype=np.uint8)
            temporary = Layer("preview", pixels, kind="shape", shape_data=copy.deepcopy(working))
            render_shape_layer(temporary)
            pil = rgba_array_to_pil(temporary.pixels)
            checker = Image.new("RGBA", pil.size, (224, 226, 230, 255))
            checker_draw = ImageDraw.Draw(checker)
            cell = max(6, min(pil.size) // 24)
            for yy in range(0, pil.height, cell):
                for xx in range(0, pil.width, cell):
                    if (xx // cell + yy // cell) % 2:
                        checker_draw.rectangle((xx, yy, xx + cell, yy + cell), fill=(198, 202, 208, 255))
            checker.alpha_composite(pil)
            preview.update_idletasks()
            available = (max(120, preview.winfo_width() - 20), max(120, preview.winfo_height() - 20))
            checker.thumbnail(available, Image.Resampling.LANCZOS)
            self._boolean_preview_image = ImageTk.PhotoImage(checker)
            preview.delete("all")
            ox = (preview.winfo_width() - checker.width) / 2
            oy = (preview.winfo_height() - checker.height) / 2
            scale = min(checker.width / max(1, self.doc.width), checker.height / max(1, self.doc.height))
            preview_state.update({"scale": scale, "ox": ox, "oy": oy})
            preview.create_image(ox, oy, image=self._boolean_preview_image, anchor=tk.NW)
            index = selected_index()
            if index is not None and 0 <= index < len(children()):
                box = shape_data_bounds(children()[index])
                if box is not None:
                    preview.create_rectangle(
                        ox + box[0] * scale, oy + box[1] * scale,
                        ox + box[2] * scale, oy + box[3] * scale,
                        outline="#1677b8", width=2, dash=(5, 3),
                    )

        def refresh_list(select: int | None = None) -> None:
            contour_list.delete(0, tk.END)
            for number, child in enumerate(children(), 1):
                kind = str(child.get("shape", "rectangle"))
                raw_name = str(child.get("_name") or "")
                name = shape_labels.get(kind, kind) if not raw_name or raw_name.lower().endswith(" shape") else raw_name
                state = "" if bool(child.get("_enabled", True)) else " (выключен)"
                contour_list.insert(tk.END, f"{number}. {name} | {shape_labels.get(kind, kind)}{state}")
            if children():
                index = max(0, min(len(children()) - 1, 0 if select is None else select))
                contour_list.selection_set(index)
                contour_list.activate(index)
                contour_list.see(index)
            load_selected()

        def load_selected(_event=None) -> None:
            index = selected_index()
            if index is None or not (0 <= index < len(children())):
                return
            child = children()[index]
            box = shape_data_bounds(child) or (0, 0, 1, 1)
            loading[0] = True
            kind = str(child.get("shape", "rectangle"))
            raw_name = str(child.get("_name") or "")
            name_var.set(shape_labels.get(kind, kind) if not raw_name or raw_name.lower().endswith(" shape") else raw_name)
            enabled_var.set(bool(child.get("_enabled", True)))
            kind_var.set(shape_labels.get(kind, kind))
            x_var.set(box[0]); y_var.set(box[1])
            width_var.set(max(1, box[2] - box[0])); height_var.set(max(1, box[3] - box[1]))
            sides_var.set(max(3, int(child.get("sides", 5))))
            ratio_var.set(float(child.get("inner_ratio", 0.5)))
            loading[0] = False
            preview_image()

        def commit_selected(_event=None) -> None:
            if loading[0]:
                return
            index = selected_index()
            if index is None or not (0 <= index < len(children())):
                return
            try:
                x, y = int(x_var.get()), int(y_var.get())
                width, height = max(1, int(width_var.get())), max(1, int(height_var.get()))
                sides = max(3, min(64, int(sides_var.get())))
                ratio = float(np.clip(float(ratio_var.get()), 0.05, 0.95))
            except (tk.TclError, ValueError):
                return
            child = transform_shape_data_to_box(children()[index], (x, y, x + width, y + height))
            child["_name"] = name_var.get().strip() or f"Контур {index + 1}"
            child["_enabled"] = bool(enabled_var.get())
            child["sides"] = sides
            child["inner_ratio"] = ratio
            children()[index] = child
            update_bounds()
            refresh_list(index)

        def move_child(delta: int) -> None:
            commit_selected()
            index = selected_index()
            if index is None:
                return
            target = max(0, min(len(children()) - 1, index + delta))
            if target != index:
                item = children().pop(index)
                children().insert(target, item)
                refresh_list(target)

        def duplicate_child() -> None:
            commit_selected()
            index = selected_index()
            if index is None:
                return
            clone = copy.deepcopy(children()[index])
            box = shape_data_bounds(clone) or (0, 0, 1, 1)
            clone = transform_shape_data_to_box(clone, (box[0] + 12, box[1] + 12, box[2] + 12, box[3] + 12))
            clone["_name"] = f"{clone.get('_name', 'Контур')} копия"
            children().insert(index + 1, clone)
            refresh_list(index + 1)

        def delete_child() -> None:
            index = selected_index()
            if index is None or len(children()) <= 1:
                return
            children().pop(index)
            refresh_list(min(index, len(children()) - 1))

        def begin_preview_drag(event) -> None:
            index = selected_index()
            if index is None:
                return
            box = shape_data_bounds(children()[index])
            scale = preview_state["scale"]
            if box is None or scale <= 0:
                return
            doc_x = (event.x - preview_state["ox"]) / scale
            doc_y = (event.y - preview_state["oy"]) / scale
            if box[0] <= doc_x <= box[2] and box[1] <= doc_y <= box[3]:
                preview_drag["index"] = index
                preview_drag["last"] = (event.x, event.y)

        def drag_preview_contour(event) -> None:
            index = preview_drag.get("index")
            if index is None or not (0 <= int(index) < len(children())):
                return
            scale = max(0.0001, preview_state["scale"])
            last_x, last_y = preview_drag["last"]
            dx, dy = round((event.x - last_x) / scale), round((event.y - last_y) / scale)
            if not dx and not dy:
                return
            child = children()[int(index)]
            box = shape_data_bounds(child)
            if box is None:
                return
            children()[int(index)] = transform_shape_data_to_box(child, (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy))
            preview_drag["last"] = (event.x, event.y)
            update_bounds()
            loading[0] = True
            x_var.set(box[0] + dx); y_var.set(box[1] + dy)
            loading[0] = False
            preview_image()

        def finish_preview_drag(_event=None) -> None:
            index = preview_drag.get("index")
            preview_drag["index"] = None
            if index is not None:
                refresh_list(int(index))

        ttk.Button(list_buttons, text="Вверх", command=lambda: move_child(-1)).pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(list_buttons, text="Вниз", command=lambda: move_child(1)).pack(side=tk.LEFT, padx=3)
        ttk.Button(list_buttons, text="Копия", command=duplicate_child).pack(side=tk.LEFT, padx=3)
        ttk.Button(list_buttons, text="Удалить", command=delete_child).pack(side=tk.LEFT, padx=3)

        for widget in field_widgets:
            widget.bind("<Return>", commit_selected)
            widget.bind("<FocusOut>", commit_selected)
            widget.bind("<<Increment>>", lambda _event: dialog.after_idle(commit_selected))
            widget.bind("<<Decrement>>", lambda _event: dialog.after_idle(commit_selected))
        enabled_var.trace_add("write", lambda *_args: commit_selected())
        mode_var.trace_add("write", lambda *_args: preview_image())
        contour_list.bind("<<ListboxSelect>>", load_selected)
        preview.bind("<Configure>", lambda _event: preview_image())
        preview.bind("<ButtonPress-1>", begin_preview_drag)
        preview.bind("<B1-Motion>", drag_preview_contour)
        preview.bind("<ButtonRelease-1>", finish_preview_drag)

        def accept() -> None:
            nonlocal result
            commit_selected()
            working["boolean_mode"] = mode_labels[mode_var.get()]
            update_bounds()
            result = copy.deepcopy(working)
            dialog.destroy()

        buttons = ttk.Frame(dialog)
        buttons.pack(fill=tk.X, padx=12, pady=12)
        ttk.Button(buttons, text="Применить", command=accept).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT, padx=(0, 8))
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        self._boolean_editor_dialog = dialog
        self._boolean_editor_list = contour_list
        self._boolean_editor_preview = preview
        refresh_list(0)
        self.center_toplevel(dialog, 920, 650)
        dialog.wait_window()
        return result
