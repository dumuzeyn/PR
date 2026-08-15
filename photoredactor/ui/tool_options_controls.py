from __future__ import annotations

from .tool_options_shared import *


OPTION_HINTS = {
    "Размер": "Диаметр отпечатка инструмента в пикселях.",
    "Жёсткость": "Резкость края отпечатка: 100% даёт чёткий край, меньшие значения делают его мягче.",
    "Непрозрачность": "Максимальная заметность результата одного штриха.",
    "Поток": "Сколько краски наносится каждым отпечатком во время непрерывного штриха.",
    "Интервал": "Расстояние между отпечатками кисти. 0% создаёт непрерывную линию.",
    "Сглаживание": "Стабилизирует траекторию указателя и уменьшает дрожание линии.",
    "Сила": "Интенсивность воздействия инструмента на изображение.",
    "Экспозиция": "Сила осветления или затемнения за один проход.",
    "Допуск": "Насколько цвет может отличаться от выбранного и всё ещё попадать в область.",
    "Допуск области": "Допустимое отличие цвета внутри связанной области градиентной заливки.",
    "Структура": "Насколько заплатка сохраняет детали и текстуру источника.",
    "Цветовая адаптация": "Насколько цвет источника подстраивается под место назначения.",
    "Диффузия": "Скорость смешивания восстановленной текстуры с окружением.",
    "Толщина": "Толщина контура создаваемой фигуры.",
    "Внутренний радиус": "Отношение внутреннего радиуса звезды к внешнему.",
}


class ToolOptionsControlsMixin:
    def _add_brush_engine_options(self, tool: str) -> None:
        self._add_scale("Поток", self.brush_flow, 0.01, 1.0, "%", percent=True)
        self._add_scale("Интервал", self.brush_spacing, 0.0, 1.0, "%", percent=True)
        self._add_scale("Сглаживание", self.brush_smoothing, 0.0, 1.0, "%", percent=True)
        if tool == "brush":
            ttk.Label(self.body, text="Режим наложения").pack(anchor=tk.W, padx=8, pady=(6, 0))
            ttk.Combobox(
                self.body,
                textvariable=self.brush_blend_mode,
                values=list(BRUSH_BLEND_MODES),
                state="readonly",
            ).pack(fill=tk.X, padx=8)
        pressure = ttk.LabelFrame(self.body, text="Перо")
        pressure.pack(fill=tk.X, padx=8, pady=(8, 0))
        ttk.Checkbutton(pressure, text="Нажим изменяет размер", variable=self.pressure_size).pack(anchor=tk.W)
        ttk.Checkbutton(pressure, text="Нажим изменяет непрозрачность", variable=self.pressure_opacity).pack(anchor=tk.W)
        ttk.Checkbutton(pressure, text="Нажим изменяет поток", variable=self.pressure_flow).pack(anchor=tk.W)
        ttk.Button(self.body, text="Движок кисти...", command=self.open_brush_engine_panel).pack(fill=tk.X, padx=8, pady=(8, 0))

    def _add_brush_presets(self) -> None:
        ttk.Label(self.body, text="Пресет кисти").pack(anchor=tk.W, padx=8, pady=(8, 0))
        box = ttk.Combobox(self.body, textvariable=self.brush_preset, values=list(self.brush_presets), state="readonly")
        box.pack(fill=tk.X, padx=8)
        box.bind("<<ComboboxSelected>>", lambda _event: self.apply_brush_preset())
        ttk.Button(self.body, text="Добавить свою кисть...", command=self.import_custom_brush).pack(fill=tk.X, padx=8, pady=(5, 0))
        self._brush_preset_editor(self.body, compact=False)

    def _compact_brush_presets(self, parent: ttk.Frame) -> None:
        menu_button = ttk.Menubutton(parent, text="Перо и пресеты")
        menu = tk.Menu(menu_button, tearoff=False)
        menu.add_checkbutton(label="Нажим → размер", variable=self.pressure_size)
        menu.add_checkbutton(label="Нажим → непрозрачность", variable=self.pressure_opacity)
        menu.add_checkbutton(label="Нажим → поток", variable=self.pressure_flow)
        menu.add_separator()
        for name in self.brush_presets:
            menu.add_radiobutton(
                label=name,
                value=name,
                variable=self.brush_preset,
                command=self.apply_brush_preset,
            )
        menu.add_separator()
        menu.add_command(label="Добавить кисть из изображения...", command=self.import_custom_brush)
        menu.add_command(label="Сохранить новый пресет", command=lambda: self.save_brush_preset(True))
        menu.add_command(label="Удалить выбранный", command=self.delete_brush_preset)
        menu.add_command(label="Восстановить стандартные", command=self.reset_brush_presets)
        menu_button.configure(menu=menu)
        menu_button.pack(side=tk.LEFT, padx=(6, 2), pady=4)
        if self.tooltip_factory is not None:
            self.tooltip_factory(menu_button, "Настройки графического пера и управление пресетами кисти")

    def _brush_preset_editor(self, parent: ttk.Frame, *, compact: bool) -> None:
        row = parent if compact else ttk.Frame(parent)
        if not compact:
            row.pack(fill=tk.X, padx=8, pady=(4, 8))
        entry = ttk.Entry(row, textvariable=self.brush_preset_name, width=12)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=not compact, padx=(3 if compact else 0, 2))
        for text, command, hint in (
            ("+", self.save_brush_preset, "Сохранить текущие параметры как пресет"),
            ("−", self.delete_brush_preset, "Удалить выбранный пользовательский пресет"),
            ("↺", self.reset_brush_presets, "Восстановить стандартные пресеты"),
        ):
            button = ttk.Button(row, text=text, command=command, width=2)
            button.pack(side=tk.LEFT, padx=1)
            if self.tooltip_factory is not None:
                self.tooltip_factory(button, hint)

    def _compact_spin(self, parent: ttk.Frame, label: str, variable: tk.Variable, start: int, end: int, width: int) -> None:
        caption = ttk.Label(parent, text=label, style="Topbar.TLabel")
        caption.pack(side=tk.LEFT, padx=(5, 2))
        control = ttk.Spinbox(parent, textvariable=variable, from_=start, to=end, width=width)
        control.pack(side=tk.LEFT, padx=(0, 4), pady=4)
        self._attach_option_hint((caption, control), label)

    def _compact_percent(self, parent: ttk.Frame, label: str, variable: tk.Variable) -> None:
        caption = ttk.Label(parent, text=label, style="Topbar.TLabel")
        caption.pack(side=tk.LEFT, padx=(5, 2))
        value = ttk.Label(parent, width=5, style="Topbar.TLabel")
        value.pack(side=tk.LEFT)
        scale = ttk.Scale(parent, variable=variable, from_=0.0, to=1.0, length=90, command=lambda raw: value.configure(text=f"{round(float(raw) * 100)}%"))
        scale.pack(side=tk.LEFT, padx=(0, 4))
        value.configure(text=f"{round(float(variable.get()) * 100)}%")
        self._attach_option_hint((caption, value, scale), label)

    def _compact_combo(self, parent: ttk.Frame, label: str, variable: tk.Variable, values: list[str], width: int, readonly: bool = True) -> None:
        caption = ttk.Label(parent, text=label, style="Topbar.TLabel")
        caption.pack(side=tk.LEFT, padx=(5, 2))
        control = ttk.Combobox(parent, textvariable=variable, values=values, width=width, state="readonly" if readonly else "normal")
        control.pack(side=tk.LEFT, padx=(0, 4), pady=4)
        self._attach_option_hint((caption, control), label)

    def _attach_option_hint(self, widgets, label: str) -> None:
        if self.tooltip_factory is None:
            return
        hint = OPTION_HINTS.get(label) or next((text for key, text in OPTION_HINTS.items() if label.startswith(key)), None)
        if hint:
            for widget in widgets:
                self.tooltip_factory(widget, hint)

    def _compact_single_color(self, parent: ttk.Frame, command, index: int, tooltip: str) -> None:
        colors = self.color_provider() if self.color_provider is not None else ((0, 0, 0, 255), (255, 255, 255, 255))
        color = colors[index]
        hex_color = "#%02x%02x%02x" % tuple(int(value) for value in color[:3])
        swatch = tk.Button(parent, command=command, background=hex_color, activebackground=hex_color, width=3, height=1, relief=tk.FLAT, borderwidth=1, highlightthickness=1, highlightbackground=TOKENS.BORDER, cursor="hand2")
        swatch.pack(side=tk.LEFT, padx=4, pady=4)
        if self.tooltip_factory is not None:
            self.tooltip_factory(swatch, tooltip)

    def _compact_color_pair(self, parent: ttk.Frame, foreground_name: str, background_name: str) -> None:
        colors = self.color_provider() if self.color_provider is not None else ((0, 0, 0, 255), (255, 255, 255, 255))
        surface = tk.Canvas(parent, width=43, height=30, highlightthickness=0, borderwidth=0, background=TOKENS.SURFACE, cursor="hand2")
        surface.pack(side=tk.LEFT, padx=5, pady=3)
        background = "#%02x%02x%02x" % tuple(int(value) for value in colors[1][:3])
        foreground = "#%02x%02x%02x" % tuple(int(value) for value in colors[0][:3])
        surface.create_rectangle(17, 9, 40, 28, fill=background, outline=TOKENS.BORDER, tags="background")
        surface.create_rectangle(2, 2, 25, 21, fill=foreground, outline=TOKENS.TEXT_SECONDARY, tags="foreground")
        surface.tag_bind("foreground", "<Button-1>", lambda _event: self.pick_foreground())
        surface.tag_bind("background", "<Button-1>", lambda _event: self.pick_background())
        if self.tooltip_factory is not None:
            self.tooltip_factory(surface, f"{foreground_name} / {background_name}\nНажмите на нужный образец")

    def _add_scale(
        self,
        label: str,
        variable: tk.Variable,
        start: float,
        end: float,
        unit: str = "",
        *,
        integer: bool = False,
        percent: bool = False,
    ) -> None:
        row = ttk.Frame(self.body)
        row.pack(fill=tk.X, padx=8, pady=(6, 0))
        caption = ttk.Label(row, text=label)
        caption.pack(side=tk.LEFT)
        value_label = ttk.Label(row, width=8, anchor=tk.E)
        value_label.pack(side=tk.RIGHT)

        def show(value=None) -> None:
            number = float(variable.get() if value is None else value)
            if percent:
                text = f"{round(number * 100)}%"
            elif integer:
                text = f"{round(number)} {unit}".strip()
            else:
                text = f"{number:.2f} {unit}".strip()
            value_label.configure(text=text)

        scale = ttk.Scale(self.body, from_=start, to=end, variable=variable, orient=tk.HORIZONTAL, command=show)
        scale.pack(fill=tk.X, padx=8)
        self._attach_option_hint((caption, value_label, scale), label)
        show()

    def _add_spinbox(self, label: str, variable: tk.Variable, start: int, end: int) -> None:
        row = ttk.Frame(self.body)
        row.pack(fill=tk.X, padx=8, pady=(6, 0))
        caption = ttk.Label(row, text=label)
        caption.pack(side=tk.LEFT)
        control = ttk.Spinbox(row, textvariable=variable, from_=start, to=end, increment=1, width=7)
        control.pack(side=tk.RIGHT)
        self._attach_option_hint((caption, control), label)

    def _add_tonal_range(self) -> None:
        ttk.Label(self.body, text="Диапазон").pack(anchor=tk.W, padx=8, pady=(6, 0))
        ttk.Combobox(
            self.body,
            textvariable=self.tonal_range,
            values=["Тени", "Средние тона", "Света"],
            state="readonly",
        ).pack(fill=tk.X, padx=8)

    def _add_clone_source_options(self) -> None:
        ttk.Checkbutton(self.body, text="Выровненный источник", variable=self.clone_aligned).pack(anchor=tk.W, padx=8, pady=(6, 2))
        ttk.Label(self.body, text="Образец").pack(anchor=tk.W, padx=8)
        ttk.Combobox(
            self.body,
            textvariable=self.clone_sampling,
            values=["Текущий слой", "Текущий и ниже", "Все видимые"],
            state="readonly",
        ).pack(fill=tk.X, padx=8)
        ttk.Label(self.body, text="Alt + левый клик выбирает источник.", wraplength=220).pack(fill=tk.X, padx=8, pady=(4, 8))

    def _add_source_transform_options(self) -> None:
        ttk.Checkbutton(self.body, text="Выровненный источник", variable=self.clone_aligned).pack(anchor=tk.W, padx=8, pady=(6, 2))
        ttk.Label(self.body, text="Образец").pack(anchor=tk.W, padx=8)
        ttk.Combobox(
            self.body,
            textvariable=self.clone_sampling,
            values=["Текущий слой", "Текущий и ниже", "Все видимые"],
            state="readonly",
        ).pack(fill=tk.X, padx=8)
        coordinates = ttk.LabelFrame(self.body, text="Источник")
        coordinates.pack(fill=tk.X, padx=8, pady=(8, 0))
        self._source_pair(coordinates, "X", self.clone_source_x, "Y", self.clone_source_y, -50000, 50000)
        self._source_pair(coordinates, "Смещение X", self.clone_offset_x, "Y", self.clone_offset_y, -50000, 50000)
        transform = ttk.LabelFrame(self.body, text="Преобразование")
        transform.pack(fill=tk.X, padx=8, pady=(8, 0))
        self._source_pair(transform, "Масштаб X, %", self.clone_scale_x, "Y, %", self.clone_scale_y, 5, 2000)
        self._source_pair(transform, "Поворот", self.clone_rotation, "", None, -180, 180)
        flip = ttk.Frame(transform)
        flip.pack(fill=tk.X, padx=4, pady=3)
        ttk.Checkbutton(flip, text="Отразить X", variable=self.clone_flip_horizontal).pack(side=tk.LEFT)
        ttk.Checkbutton(flip, text="Отразить Y", variable=self.clone_flip_vertical).pack(side=tk.LEFT, padx=(8, 0))
        overlay = ttk.LabelFrame(self.body, text="Наложение источника")
        overlay.pack(fill=tk.X, padx=8, pady=(8, 0))
        ttk.Checkbutton(overlay, text="Показывать", variable=self.clone_overlay_visible).pack(anchor=tk.W, padx=4)
        self._add_scale("Прозрачность наложения", self.clone_overlay_opacity, 0.05, 1.0, "%", percent=True)
        ttk.Label(self.body, text="Alt + левый клик выбирает источник.", wraplength=220).pack(fill=tk.X, padx=8, pady=(4, 8))

    @staticmethod
    def _source_pair(parent, first_label, first_var, second_label, second_var, start, end) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=4, pady=3)
        ttk.Label(row, text=first_label).pack(side=tk.LEFT)
        ttk.Spinbox(row, textvariable=first_var, from_=start, to=end, width=7).pack(side=tk.LEFT, padx=(4, 8))
        if second_var is not None:
            ttk.Label(row, text=second_label).pack(side=tk.LEFT)
            ttk.Spinbox(row, textvariable=second_var, from_=start, to=end, width=7).pack(side=tk.LEFT, padx=4)

    def _add_patch_options(self) -> None:
        self._add_scale("Структура", self.patch_structure, 1, 7, integer=True)
        self._add_scale("Цветовая адаптация", self.patch_color_adaptation, 0, 10, integer=True)
        ttk.Checkbutton(self.body, text="Образец со всех слоёв", variable=self.patch_sample_all_layers).pack(anchor=tk.W, padx=8, pady=6)
        ttk.Button(self.body, text="Применить", command=self.apply_patch_preview).pack(fill=tk.X, padx=8, pady=(2, 4))
        ttk.Label(
            self.body,
            text="Перетащите выделение к источнику. Enter применяет предпросмотр, Escape отменяет.",
            wraplength=220,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=8, pady=(2, 8))

    def _add_gradient_options(self) -> None:
        ttk.Label(self.body, text="Режим").pack(anchor=tk.W, padx=8, pady=(6, 0))
        ttk.Combobox(self.body, textvariable=self.gradient_mode, values=["Заливка", "Объект"], state="readonly").pack(fill=tk.X, padx=8)
        if self.gradient_mode.get() == "Объект":
            ttk.Label(self.body, text="Фигура").pack(anchor=tk.W, padx=8, pady=(6, 0))
            ttk.Combobox(
                self.body,
                textvariable=self.gradient_shape,
                values=["Прямоугольник", "Эллипс", "Многоугольник", "Звезда", "Произвольная"],
                state="readonly",
            ).pack(fill=tk.X, padx=8)
            ttk.Label(self.body, text="Заливка").pack(anchor=tk.W, padx=8, pady=(6, 0))
            ttk.Combobox(self.body, textvariable=self.gradient_object_fill, values=["Градиент", "Текстура"], state="readonly").pack(fill=tk.X, padx=8)
        uses_gradient = self.gradient_mode.get() == "Заливка" or self.gradient_object_fill.get() == "Градиент"
        if uses_gradient:
            ttk.Label(self.body, text="Тип").pack(anchor=tk.W, padx=8, pady=(6, 0))
            ttk.Combobox(
                self.body,
                textvariable=self.gradient_type,
                values=["Линейный", "Радиальный", "Отраженный", "Ромб", "Угловой"],
                state="readonly",
            ).pack(fill=tk.X, padx=8)
            if self.gradient_mode.get() == "Заливка":
                self._add_spinbox("Допуск области", self.tolerance, 0, 128)
            ttk.Button(self.body, text="Редактор градиента...", command=self.open_gradient_editor).pack(fill=tk.X, padx=8, pady=(8, 2))
        else:
            ttk.Label(self.body, text="Текстура").pack(anchor=tk.W, padx=8, pady=(6, 0))
            ttk.Combobox(self.body, textvariable=self.gradient_texture, values=["Шахматная", "Полосы", "Точки"], state="readonly").pack(fill=tk.X, padx=8)
        ttk.Label(self.body, text="Проведите от цвета A к цвету B. Escape отменяет предпросмотр.", wraplength=220).pack(fill=tk.X, padx=8, pady=(4, 8))

    def _add_crop_options(self) -> None:
        ttk.Label(self.body, text="Соотношение сторон").pack(anchor=tk.W, padx=8, pady=(6, 0))
        ttk.Combobox(
            self.body,
            textvariable=self.crop_aspect,
            values=["Свободно", "1:1", "4:3", "3:2", "16:9", "Исходное", "Свое"],
            state="readonly",
        ).pack(fill=tk.X, padx=8)
        self._add_spinbox("Ширина", self.crop_custom_width, 1, 50000)
        self._add_spinbox("Высота", self.crop_custom_height, 1, 50000)
        ttk.Label(
            self.body,
            text="Протяните рамку. Enter или двойной клик применяет, Escape отменяет.",
            wraplength=220,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=8, pady=(6, 8))

    def _add_text_options(self) -> None:
        ttk.Label(self.body, text="Шрифт").pack(anchor=tk.W, padx=8, pady=(6, 0))
        ttk.Combobox(
            self.body,
            textvariable=self.text_font_family,
            values=["Arial", "Segoe UI", "Calibri", "Times New Roman", "Verdana", "Tahoma"],
        ).pack(fill=tk.X, padx=8)
        self._add_spinbox("Размер", self.text_size, 4, 500)
        style_row = ttk.Frame(self.body)
        style_row.pack(fill=tk.X, padx=8, pady=(6, 0))
        ttk.Checkbutton(style_row, text="B", variable=self.text_bold, style="Toolbutton").pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Checkbutton(style_row, text="I", variable=self.text_italic, style="Toolbutton").pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Checkbutton(style_row, text="U", variable=self.text_underline, style="Toolbutton").pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Label(self.body, text="Выравнивание").pack(anchor=tk.W, padx=8, pady=(6, 0))
        ttk.Combobox(self.body, textvariable=self.text_align, values=["left", "center", "right"], state="readonly").pack(fill=tk.X, padx=8)
        self._add_spinbox("Интервал строк", self.text_line_spacing, 0, 500)
        self._add_spinbox("Трекинг", self.text_tracking, -100, 500)
        self._add_spinbox("Ширина блока", self.text_box_width, 0, 50000)
        self._add_scale("Поворот", self.text_rotation, -180, 180, "°")
        ttk.Button(self.body, text="Редактировать активный текст", command=self.edit_active_text).pack(fill=tk.X, padx=8, pady=(8, 2))
        ttk.Button(self.body, text="Текст по контуру...", command=self.edit_text_path).pack(fill=tk.X, padx=8, pady=2)
        ttk.Button(self.body, text="Готово", command=self.finish_text_edit).pack(fill=tk.X, padx=8, pady=(2, 2))

    def _add_shape_options(self, tool: str) -> None:
        center = ttk.Checkbutton(self.body, text="Строить из центра", variable=self.shape_from_center)
        center.pack(anchor=tk.W, padx=8, pady=(0, 6))
        if self.tooltip_factory is not None:
            self.tooltip_factory(center, "Выключено: первая точка является углом фигуры. Включено: первая точка является центром. Alt временно меняет режим.")
        if tool in {"line_shape", "bezier_shape"}:
            ttk.Button(self.body, text="Цвет линии", command=self.pick_foreground).pack(fill=tk.X, padx=8, pady=(0, 6))
        else:
            row = ttk.Frame(self.body)
            row.pack(fill=tk.X, padx=8, pady=(0, 6))
            ttk.Button(row, text="Заливка", command=self.pick_foreground).pack(side=tk.LEFT, fill=tk.X, expand=True)
            ttk.Button(row, text="Обводка", command=self.pick_background).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        self._add_spinbox("Толщина, px", self.shape_stroke_width, 0 if tool not in {"line_shape", "bezier_shape"} else 1, 100)
        if tool == "polygon_shape":
            self._add_spinbox("Стороны", self.polygon_sides, 3, 64)
        elif tool == "star_shape":
            self._add_spinbox("Лучи", self.star_points, 3, 64)
            self._add_scale("Внутренний радиус", self.star_inner_ratio, 0.05, 0.95, "%", percent=True)
        elif tool == "custom_shape":
            ttk.Label(self.body, text="Фигура").pack(anchor=tk.W, padx=8, pady=(6, 0))
            ttk.Combobox(
                self.body,
                textvariable=self.custom_shape_preset,
                values=self.custom_shape_presets,
                state="readonly",
            ).pack(fill=tk.X, padx=8)

    def _add_color_buttons(self) -> None:
        row = ttk.Frame(self.body)
        row.pack(fill=tk.X, padx=8, pady=(0, 6))
        self._compact_color_pair(row, "Основной цвет", "Дополнительный цвет")

    def _add_selection_mode(self) -> None:
        ttk.Label(self.body, text="Режим выделения").pack(anchor=tk.W, padx=8, pady=(6, 2))
        row = ttk.Frame(self.body)
        row.pack(fill=tk.X, padx=8)
        definitions = [
            ("Н", "replace", "Создать новое выделение."),
            ("+", "add", "Добавить область к текущему выделению."),
            ("−", "subtract", "Вычесть область из текущего выделения."),
            ("∩", "intersect", "Оставить пересечение с текущим выделением."),
        ]
        for label, value, description in definitions:
            button = ttk.Radiobutton(row, text=label, value=value, variable=self.selection_mode, style="Toolbutton", width=3)
            button.pack(side=tk.LEFT, fill=tk.X, expand=True)
            if self.tooltip_factory is not None:
                self.tooltip_factory(button, description)

    def _add_paint_target(self) -> None:
        ttk.Label(self.body, text="Куда рисовать").pack(anchor=tk.W, padx=8, pady=(6, 0))
        box = ttk.Combobox(self.body, textvariable=self.paint_target, values=["pixels", "mask"], state="readonly")
        box.pack(fill=tk.X, padx=8)
        box.bind("<<ComboboxSelected>>", lambda _event: self.set_paint_target())
        if self.tooltip_factory is not None:
            self.tooltip_factory(box, "pixels рисует по слою, mask рисует по маске активного слоя.")

    def _add_retouch_preset(self) -> None:
        ttk.Label(self.body, text="Пресет ретуши").pack(anchor=tk.W, padx=8, pady=(6, 0))
        box = ttk.Combobox(self.body, textvariable=self.retouch_preset, values=list(self.retouch_presets), state="readonly")
        box.pack(fill=tk.X, padx=8, pady=(0, 4))
        box.bind("<<ComboboxSelected>>", lambda _event: self.apply_retouch_preset())
        ttk.Button(self.body, text="Применить пресет", command=self.apply_retouch_preset).pack(fill=tk.X, padx=8, pady=(0, 8))
