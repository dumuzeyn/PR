from __future__ import annotations

from ..app_shared import *


class ObjectPropertiesMixin:
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
            spin.configure(command=commit)

        def combo_row(label: str, initial: str, values: list[str], apply) -> None:
            row = ttk.Frame(panel)
            row.pack(fill=tk.X, padx=10, pady=2)
            ttk.Label(row, text=label, style="Secondary.TLabel").pack(side=tk.LEFT)
            variable = tk.StringVar(value=initial)
            combo = ttk.Combobox(row, textvariable=variable, values=values, state="readonly", width=14)
            combo.pack(side=tk.RIGHT)
            combo.bind("<<ComboboxSelected>>", lambda _event: apply(variable.get()))

        def boolean_row(label: str, initial: bool, apply) -> None:
            variable = tk.BooleanVar(value=bool(initial))
            control = ttk.Checkbutton(panel, text=label, variable=variable, command=lambda: apply(bool(variable.get())))
            control.pack(anchor=tk.W, padx=10, pady=2)

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
        ttk.Label(panel, text="Трансформация", style="PanelTitle.TLabel").pack(anchor=tk.W, padx=10, pady=(7, 3))
        numeric_row("X", display_x, lambda value, y=display_y: self.set_object_position(int(value), y))
        numeric_row("Y", display_y, lambda value, x=display_x: self.set_object_position(x, int(value)))
        if layer.kind == "shape" and layer.shape_data is not None:
            box = shape_data_bounds(layer.shape_data) or (0, 0, 1, 1)
            numeric_row("Ширина", box[2] - box[0], lambda value: self.set_shape_size(int(value), None), 2, 100000)
            numeric_row("Высота", box[3] - box[1], lambda value: self.set_shape_size(None, int(value)), 2, 100000)
            numeric_row("Поворот", float(layer.shape_data.get("rotation", 0.0)), lambda value: self.set_shape_property("rotation", float(value)), -360.0, 360.0, 1.0)
            ttk.Label(panel, text="Заливка", style="PanelTitle.TLabel").pack(anchor=tk.W, padx=10, pady=(7, 3))
            color_row("Цвет", layer.shape_data.get("fill") or [0, 0, 0, 0], lambda: self.pick_shape_property_color("fill"))
            numeric_row("Непрозрачность, %", round(float(layer.shape_data.get("fill_opacity", 1.0)) * 100), lambda value: self.set_shape_property("fill_opacity", float(value) / 100.0), 0, 100)
            ttk.Label(panel, text="Граница", style="PanelTitle.TLabel").pack(anchor=tk.W, padx=10, pady=(7, 3))
            boolean_row("Показывать границу", bool(layer.shape_data.get("stroke_enabled", int(layer.shape_data.get("stroke_width", 0)) > 0)), lambda value: self.set_shape_property("stroke_enabled", value))
            color_row("Цвет", layer.shape_data.get("stroke") or [0, 0, 0, 0], lambda: self.pick_shape_property_color("stroke"))
            numeric_row("Толщина обводки", int(layer.shape_data.get("stroke_width", 0)), lambda value: self.set_shape_property("stroke_width", int(value)), 0, 100)
            numeric_row("Непрозрачность, %", round(float(layer.shape_data.get("stroke_opacity", 1.0)) * 100), lambda value: self.set_shape_property("stroke_opacity", float(value) / 100.0), 0, 100)
            combo_row("Обводка", str(layer.shape_data.get("stroke_alignment", "center")), ["inside", "center", "outside"], lambda value: self.set_shape_property("stroke_alignment", value))
            combo_row("Концы линии", str(layer.shape_data.get("stroke_cap", "round")), ["butt", "round", "square"], lambda value: self.set_shape_property("stroke_cap", value))
            combo_row("Соединения", str(layer.shape_data.get("stroke_join", "round")), ["miter", "round", "bevel"], lambda value: self.set_shape_property("stroke_join", value))
            numeric_row("Предел острого угла", float(layer.shape_data.get("miter_limit", 4.0)), lambda value: self.set_shape_property("miter_limit", float(value)), 1.0, 20.0, 0.5)
            dash_patterns = {
                "Сплошная": [], "Точки": [2.0, 2.0], "Штрихи": [8.0, 4.0],
                "Штрих-точка": [8.0, 3.0, 2.0, 3.0],
            }
            current_dash = next((name for name, pattern in dash_patterns.items() if pattern == layer.shape_data.get("dash_pattern", [])), "Сплошная")
            combo_row("Штриховка", current_dash, list(dash_patterns), lambda value: self.set_shape_property("dash_pattern", dash_patterns[value]))
            numeric_row("Смещение штрихов", float(layer.shape_data.get("dash_offset", 0.0)), lambda value: self.set_shape_property("dash_offset", float(value)), -1000.0, 1000.0, 1.0)
            numeric_row("Непрозрачность объекта, %", round(float(layer.shape_data.get("opacity", 1.0)) * 100), lambda value: self.set_shape_property("opacity", float(value) / 100.0), 0, 100)
            kind = str(layer.shape_data.get("shape", "rectangle"))
            if kind in {"polygon", "star"}:
                numeric_row("Стороны" if kind == "polygon" else "Лучи", int(layer.shape_data.get("sides", 5)), lambda value: self.set_shape_property("sides", int(value)), 3, 64)
            if kind == "star":
                numeric_row("Внутренний радиус", float(layer.shape_data.get("inner_ratio", 0.5)), lambda value: self.set_shape_property("inner_ratio", float(value)), 0.05, 0.95, 0.05)
            if kind == "rectangle":
                numeric_row("Радиус углов", int(layer.shape_data.get("corner_radius", 0)), lambda value: self.set_shape_property("corner_radius", int(value)), 0, 10000)
            if isinstance(layer.shape_data.get("gradient"), dict):
                gradient = layer.shape_data["gradient"]
                ttk.Label(panel, text="Градиент", style="PanelTitle.TLabel").pack(anchor=tk.W, padx=10, pady=(7, 3))
                combo_row("Тип", str(gradient.get("type", "linear")), list(GradientEngine.TYPES), lambda value: self.set_shape_gradient_property("type", value))
                numeric_row("Угол", float(gradient.get("angle", 0.0)), lambda value: self.set_shape_gradient_property("angle", float(value)), -360.0, 360.0, 1.0)
                numeric_row("Масштаб", float(gradient.get("scale", 1.0)), lambda value: self.set_shape_gradient_property("scale", float(value)), 0.01, 10.0, 0.05)
            ttk.Button(panel, text="Редактор градиента...", command=self.edit_shape_gradient).pack(fill=tk.X, padx=10, pady=6)
            if kind == "boolean":
                ttk.Button(panel, text="Редактировать операцию и контуры", command=self.edit_boolean_shape).pack(fill=tk.X, padx=10, pady=6)
        elif layer.kind == "text" and layer.text_data is not None:
            ttk.Label(panel, text="Символ", style="PanelTitle.TLabel").pack(anchor=tk.W, padx=10, pady=(7, 3))
            combo_row("Шрифт", str(layer.text_data.get("font_family", "Arial")), ["Arial", "Segoe UI", "Calibri", "Times New Roman", "Verdana", "Tahoma"], lambda value: self.set_text_property("font_family", value))
            style_name = "Жирный курсив" if layer.text_data.get("bold") and layer.text_data.get("italic") else "Жирное" if layer.text_data.get("bold") else "Курсив" if layer.text_data.get("italic") else "Обычное"
            combo_row("Начертание", style_name, ["Обычное", "Жирное", "Курсив", "Жирный курсив"], self.set_text_style)
            numeric_row("Размер", int(layer.text_data.get("size", 48)), lambda value: self.set_text_property("size", int(value)), 4, 500)
            numeric_row("Трекинг", int(layer.text_data.get("tracking", 0)), lambda value: self.set_text_property("tracking", int(value)), -100, 500)
            ttk.Checkbutton(panel, text="Кернинг", variable=tk.BooleanVar(value=bool(layer.text_data.get("kerning_enabled", True))), command=lambda: self.set_text_property("kerning_enabled", not bool(layer.text_data.get("kerning_enabled", True)))).pack(anchor=tk.W, padx=10, pady=2)
            ttk.Checkbutton(panel, text="Стандартные лигатуры", variable=tk.BooleanVar(value=bool(layer.text_data.get("standard_ligatures", True))), command=lambda: self.set_text_property("standard_ligatures", not bool(layer.text_data.get("standard_ligatures", True)))).pack(anchor=tk.W, padx=10, pady=2)
            ttk.Checkbutton(panel, text="Дополнительные лигатуры", variable=tk.BooleanVar(value=bool(layer.text_data.get("discretionary_ligatures", False))), command=lambda: self.set_text_property("discretionary_ligatures", not bool(layer.text_data.get("discretionary_ligatures", False)))).pack(anchor=tk.W, padx=10, pady=2)
            numeric_row("Стилистический набор", int(layer.text_data.get("stylistic_set", 0)), lambda value: self.set_text_property("stylistic_set", int(value)), 0, 20)
            combo_row("Направление", str(layer.text_data.get("direction", "auto")), ["auto", "ltr", "rtl"], lambda value: self.set_text_property("direction", value))
            numeric_row("Масштаб по горизонтали, %", int(layer.text_data.get("horizontal_scale", 100)), lambda value: self.set_text_property("horizontal_scale", int(value)), 1, 1000)
            numeric_row("Масштаб по вертикали, %", int(layer.text_data.get("vertical_scale", 100)), lambda value: self.set_text_property("vertical_scale", int(value)), 1, 1000)
            numeric_row("Сдвиг базовой линии", int(layer.text_data.get("baseline_shift", 0)), lambda value: self.set_text_property("baseline_shift", int(value)), -500, 500)
            style_row = ttk.Frame(panel)
            style_row.pack(fill=tk.X, padx=10, pady=3)
            for label, key in (("Жирный", "bold"), ("Курсив", "italic"), ("Подчеркнуть", "underline"), ("Зачеркнуть", "strike_through")):
                ttk.Checkbutton(style_row, text=label, variable=tk.BooleanVar(value=bool(layer.text_data.get(key, False))), command=lambda k=key: self.set_text_property(k, not bool(layer.text_data.get(k, False)))).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Checkbutton(panel, text="Вертикальный текст", variable=tk.BooleanVar(value=bool(layer.text_data.get("vertical", False))), command=lambda: self.set_text_property("vertical", not bool(layer.text_data.get("vertical", False)))).pack(anchor=tk.W, padx=10, pady=2)
            ttk.Label(panel, text="Абзац", style="PanelTitle.TLabel").pack(anchor=tk.W, padx=10, pady=(7, 3))
            combo_row("Режим текста", str(layer.text_data.get("text_mode", "paragraph" if int(layer.text_data.get("box_width", 0)) > 0 else "point")), ["point", "paragraph"], lambda value: self.set_text_property("text_mode", value))
            numeric_row("Ширина блока", int(layer.text_data.get("box_width", 0)), lambda value: self.set_text_property("box_width", int(value)), 0, 100000)
            numeric_row("Высота блока", int(layer.text_data.get("box_height", 0)), lambda value: self.set_text_property("box_height", int(value)), 0, 100000)
            numeric_row("Интерлиньяж", int(layer.text_data.get("line_spacing", 10)), lambda value: self.set_text_property("line_spacing", int(value)), 0, 500)
            combo_row("Выравнивание", str(layer.text_data.get("align", "left")), ["left", "center", "right", "justify"], lambda value: self.set_text_property("align", value))
            numeric_row("Отступ слева", int(layer.text_data.get("indent_left", 0)), lambda value: self.set_text_property("indent_left", int(value)), 0, 10000)
            numeric_row("Отступ справа", int(layer.text_data.get("indent_right", 0)), lambda value: self.set_text_property("indent_right", int(value)), 0, 10000)
            numeric_row("Первая строка", int(layer.text_data.get("first_line_indent", 0)), lambda value: self.set_text_property("first_line_indent", int(value)), -10000, 10000)
            numeric_row("Интервал до", int(layer.text_data.get("spacing_before", 0)), lambda value: self.set_text_property("spacing_before", int(value)), 0, 10000)
            numeric_row("Интервал после", int(layer.text_data.get("spacing_after", 0)), lambda value: self.set_text_property("spacing_after", int(value)), 0, 10000)
            color_row("Цвет", layer.text_data.get("color") or [255, 255, 255, 255], self.pick_text_property_color)
            ttk.Button(panel, text="Редактировать текст", command=self.edit_active_text_on_canvas).pack(fill=tk.X, padx=10, pady=6)
            ttk.Button(panel, text="Текст по контуру...", command=self.edit_text_path).pack(fill=tk.X, padx=10, pady=(0, 6))
        else:
            ttk.Button(panel, text="Фильтры слоя", command=self.edit_layer_filters).pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(panel, text="Объект", style="PanelTitle.TLabel").pack(anchor=tk.W, padx=10, pady=(7, 3))
        boolean_row("Видимый", bool(layer.visible), lambda value: self.set_layer_property("Видимость объекта", "visible", value))
        boolean_row("Заблокирован", bool(layer.locked), lambda value: self.set_layer_property("Блокировка объекта", "locked", value, affects_canvas=False))

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

    def set_shape_gradient_property(self, key: str, value) -> None:
        layer = self.doc.layer
        if layer.kind != "shape" or layer.shape_data is None or not isinstance(layer.shape_data.get("gradient"), dict):
            return
        gradient = copy.deepcopy(layer.shape_data["gradient"])
        if gradient.get(key) == value:
            return
        gradient[key] = value
        self.set_shape_property("gradient", gradient)

    def edit_shape_gradient(self) -> None:
        layer = self.doc.layer
        if layer.kind != "shape" or layer.shape_data is None:
            return
        box = shape_data_bounds(layer.shape_data) or (0, 0, 1, 1)
        initial = copy.deepcopy(layer.shape_data.get("gradient"))
        if not isinstance(initial, dict):
            initial = self.current_gradient_definition()
            initial.update({"type": "linear", "start": [box[0], box[1]], "end": [box[2], box[3]], "angle": 0.0, "scale": 1.0})

        def apply(definition: dict[str, object]) -> None:
            for key in ("type", "start", "end", "angle", "scale", "opacity"):
                if key in initial:
                    definition.setdefault(key, initial[key])
            self.set_shape_property("gradient", definition)

        self.gradient_editor_dialog(initial, apply)

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

    def set_text_style(self, value: str) -> None:
        layer = self.doc.layer
        if layer.kind != "text" or layer.text_data is None:
            return
        before = copy.deepcopy(layer.text_data)
        layer.text_data["bold"] = value in {"Жирное", "Жирный курсив"}
        layer.text_data["italic"] = value in {"Курсив", "Жирный курсив"}
        if layer.text_data == before:
            return
        render_text_layer(layer)
        layer.touch_pixels()
        self.push_command(TextDataCommand("Изменить начертание", layer.id, before, copy.deepcopy(layer.text_data)))
        self.doc.dirty = True
        self.refresh()
