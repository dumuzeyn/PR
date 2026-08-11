from __future__ import annotations

from .tool_options_shared import *


class ToolOptionsBaseMixin:
    def __init__(
        self,
        parent: tk.Widget,
        *,
        tool_var: tk.StringVar,
        definitions: list[tuple[str, str, str]],
        brush_size: tk.IntVar,
        opacity: tk.DoubleVar,
        hardness: tk.DoubleVar,
        brush_flow: tk.DoubleVar,
        brush_spacing: tk.DoubleVar,
        brush_smoothing: tk.DoubleVar,
        brush_blend_mode: tk.StringVar,
        pressure_size: tk.BooleanVar,
        pressure_opacity: tk.BooleanVar,
        pressure_flow: tk.BooleanVar,
        brush_preset: tk.StringVar,
        brush_preset_name: tk.StringVar,
        brush_presets: dict,
        apply_brush_preset,
        save_brush_preset,
        delete_brush_preset,
        reset_brush_presets,
        retouch_strength: tk.DoubleVar,
        exposure: tk.DoubleVar,
        tonal_range: tk.StringVar,
        tolerance: tk.IntVar,
        color_range_sample_hex: tk.StringVar,
        selection_mode: tk.StringVar,
        quick_smooth: tk.IntVar,
        quick_edge_radius: tk.IntVar,
        quick_edge_strength: tk.DoubleVar,
        paint_target: tk.StringVar,
        retouch_preset: tk.StringVar,
        retouch_presets: dict,
        pick_foreground,
        pick_background,
        set_paint_target,
        apply_retouch_preset,
        shape_stroke_width: tk.IntVar,
        polygon_sides: tk.IntVar,
        star_points: tk.IntVar,
        star_inner_ratio: tk.DoubleVar,
        custom_shape_preset: tk.StringVar,
        custom_shape_presets: list[str],
        selection_feather: tk.IntVar,
        selection_antialias: tk.BooleanVar,
        magic_contiguous: tk.BooleanVar,
        magic_sample_all_layers: tk.BooleanVar,
        color_range_sample_all_layers: tk.BooleanVar,
        open_color_range_dialog: Callable[[], None],
        clone_aligned: tk.BooleanVar,
        clone_sampling: tk.StringVar,
        clone_source_x: tk.IntVar,
        clone_source_y: tk.IntVar,
        clone_offset_x: tk.IntVar,
        clone_offset_y: tk.IntVar,
        clone_scale_x: tk.DoubleVar,
        clone_scale_y: tk.DoubleVar,
        clone_rotation: tk.DoubleVar,
        clone_flip_horizontal: tk.BooleanVar,
        clone_flip_vertical: tk.BooleanVar,
        clone_overlay_visible: tk.BooleanVar,
        clone_overlay_opacity: tk.DoubleVar,
        spot_healing_mode: tk.StringVar,
        patch_structure: tk.IntVar,
        patch_color_adaptation: tk.DoubleVar,
        patch_sample_all_layers: tk.BooleanVar,
        apply_patch_preview,
        open_clone_source_panel,
        gradient_type: tk.StringVar,
        gradient_mode: tk.StringVar,
        gradient_shape: tk.StringVar,
        gradient_object_fill: tk.StringVar,
        gradient_texture: tk.StringVar,
        gradient_mid_enabled: tk.BooleanVar,
        gradient_mid_position: tk.DoubleVar,
        pick_gradient_mid,
        crop_aspect: tk.StringVar,
        crop_custom_width: tk.IntVar,
        crop_custom_height: tk.IntVar,
        text_font_family: tk.StringVar,
        text_size: tk.IntVar,
        text_bold: tk.BooleanVar,
        text_italic: tk.BooleanVar,
        text_underline: tk.BooleanVar,
        text_align: tk.StringVar,
        text_line_spacing: tk.IntVar,
        text_tracking: tk.IntVar,
        text_rotation: tk.DoubleVar,
        text_box_width: tk.IntVar,
        finish_text_edit,
        edit_active_text,
        edit_text_path,
        tooltip_factory=None,
        compact: bool = False,
        auto_select: tk.BooleanVar | None = None,
        color_provider=None,
    ) -> None:
        super().__init__(parent)
        self.tool_var = tool_var
        self.definitions = definitions
        self.label_by_id = {value: label for label, value, _description in definitions}
        self.description_by_id = {value: description for label, value, description in definitions}
        self.brush_size = brush_size
        self.opacity = opacity
        self.hardness = hardness
        self.brush_flow = brush_flow
        self.brush_spacing = brush_spacing
        self.brush_smoothing = brush_smoothing
        self.brush_blend_mode = brush_blend_mode
        self.pressure_size = pressure_size
        self.pressure_opacity = pressure_opacity
        self.pressure_flow = pressure_flow
        self.brush_preset = brush_preset
        self.brush_preset_name = brush_preset_name
        self.brush_presets = brush_presets
        self.apply_brush_preset = apply_brush_preset
        self.save_brush_preset = save_brush_preset
        self.delete_brush_preset = delete_brush_preset
        self.reset_brush_presets = reset_brush_presets
        self.retouch_strength = retouch_strength
        self.exposure = exposure
        self.tonal_range = tonal_range
        self.tolerance = tolerance
        self.color_range_sample_hex = color_range_sample_hex
        self.selection_mode = selection_mode
        self.quick_smooth = quick_smooth
        self.quick_edge_radius = quick_edge_radius
        self.quick_edge_strength = quick_edge_strength
        self.paint_target = paint_target
        self.retouch_preset = retouch_preset
        self.retouch_presets = retouch_presets
        self.pick_foreground = pick_foreground
        self.pick_background = pick_background
        self.set_paint_target = set_paint_target
        self.apply_retouch_preset = apply_retouch_preset
        self.shape_stroke_width = shape_stroke_width
        self.polygon_sides = polygon_sides
        self.star_points = star_points
        self.star_inner_ratio = star_inner_ratio
        self.custom_shape_preset = custom_shape_preset
        self.custom_shape_presets = custom_shape_presets
        self.selection_feather = selection_feather
        self.selection_antialias = selection_antialias
        self.magic_contiguous = magic_contiguous
        self.magic_sample_all_layers = magic_sample_all_layers
        self.color_range_sample_all_layers = color_range_sample_all_layers
        self.open_color_range_dialog = open_color_range_dialog
        self.clone_aligned = clone_aligned
        self.clone_sampling = clone_sampling
        self.clone_source_x = clone_source_x
        self.clone_source_y = clone_source_y
        self.clone_offset_x = clone_offset_x
        self.clone_offset_y = clone_offset_y
        self.clone_scale_x = clone_scale_x
        self.clone_scale_y = clone_scale_y
        self.clone_rotation = clone_rotation
        self.clone_flip_horizontal = clone_flip_horizontal
        self.clone_flip_vertical = clone_flip_vertical
        self.clone_overlay_visible = clone_overlay_visible
        self.clone_overlay_opacity = clone_overlay_opacity
        self.spot_healing_mode = spot_healing_mode
        self.patch_structure = patch_structure
        self.patch_color_adaptation = patch_color_adaptation
        self.patch_sample_all_layers = patch_sample_all_layers
        self.apply_patch_preview = apply_patch_preview
        self.open_clone_source_panel = open_clone_source_panel
        self.gradient_type = gradient_type
        self.gradient_mode = gradient_mode
        self.gradient_shape = gradient_shape
        self.gradient_object_fill = gradient_object_fill
        self.gradient_texture = gradient_texture
        self.gradient_mid_enabled = gradient_mid_enabled
        self.gradient_mid_position = gradient_mid_position
        self.pick_gradient_mid = pick_gradient_mid
        self.crop_aspect = crop_aspect
        self.crop_custom_width = crop_custom_width
        self.crop_custom_height = crop_custom_height
        self.text_font_family = text_font_family
        self.text_size = text_size
        self.text_bold = text_bold
        self.text_italic = text_italic
        self.text_underline = text_underline
        self.text_align = text_align
        self.text_line_spacing = text_line_spacing
        self.text_tracking = text_tracking
        self.text_rotation = text_rotation
        self.text_box_width = text_box_width
        self.finish_text_edit = finish_text_edit
        self.edit_active_text = edit_active_text
        self.edit_text_path = edit_text_path
        self.tooltip_factory = tooltip_factory
        self.compact = compact
        self.auto_select = auto_select
        self.color_provider = color_provider
        self._advanced_visible = False
        self._gradient_render_after_id: str | None = None
        self.gradient_mode.trace_add("write", self._gradient_layout_changed)
        self.gradient_object_fill.trace_add("write", self._gradient_layout_changed)
        self.title = ttk.Label(self, text="Параметры инструмента", style="ToolOptionsTitle.TLabel" if compact else "TLabel")
        self.title.pack(side=tk.LEFT if compact else tk.TOP, anchor=tk.W, padx=(4, 8), pady=4 if compact else 7)
        if compact:
            ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5), pady=7)
        self.body = ttk.Frame(self)
        self.body.pack(side=tk.LEFT if compact else tk.TOP, fill=tk.BOTH, expand=True)
        self.render()

    def _gradient_layout_changed(self, *_args) -> None:
        if self.tool_var.get() != "gradient" or self._gradient_render_after_id is not None:
            return
        self._gradient_render_after_id = self.after_idle(self._render_gradient_layout)

    def _render_gradient_layout(self) -> None:
        self._gradient_render_after_id = None
        if self.winfo_exists() and self.tool_var.get() == "gradient":
            self.render()

    def render(self) -> None:
        for child in self.body.winfo_children():
            child.destroy()
        tool = self.tool_var.get()
        label = self.label_by_id.get(tool, tool)
        self.title.configure(text=f"Параметры: {label}" if self.compact else label)
        if self.compact:
            self._render_compact(tool)
            return
        description = self.description_by_id.get(tool)
        if description:
            note = ttk.Label(self.body, text=description, wraplength=220, justify=tk.LEFT)
            note.pack(fill=tk.X, padx=8, pady=(0, 8))

        shown = False
        if tool in COLOR_TOOLS:
            self._add_color_buttons()
            shown = True
        if tool in {"brush", "eraser"}:
            self._add_scale("Размер", self.brush_size, 1, 220, "px", integer=True)
            self._add_scale("Жёсткость", self.hardness, 0.0, 1.0, "%", percent=True)
            self._add_scale("Непрозрачность", self.opacity, 0.01, 1.0, "%", percent=True)
            self._add_brush_engine_options(tool)
            shown = True
        elif tool in {"blur_tool", "sharpen_tool"}:
            self._add_scale("Размер", self.brush_size, 1, 220, "px", integer=True)
            self._add_scale("Жёсткость", self.hardness, 0.0, 1.0, "%", percent=True)
            self._add_scale("Непрозрачность", self.retouch_strength, 0.01, 1.0, "%", percent=True)
            self._add_brush_engine_options(tool)
            shown = True
        elif tool in {"dodge", "burn"}:
            self._add_scale("Размер", self.brush_size, 1, 220, "px", integer=True)
            self._add_scale("Жёсткость", self.hardness, 0.0, 1.0, "%", percent=True)
            self._add_scale("Экспозиция", self.exposure, 0.01, 1.0, "%", percent=True)
            self._add_tonal_range()
            shown = True
        elif tool == "clone":
            self._add_scale("Размер", self.brush_size, 1, 220, "px", integer=True)
            self._add_scale("Жёсткость", self.hardness, 0.0, 1.0, "%", percent=True)
            self._add_scale("Непрозрачность", self.opacity, 0.01, 1.0, "%", percent=True)
            self._add_source_transform_options()
            self._add_brush_engine_options(tool)
            shown = True
        elif tool == "healing":
            self._add_scale("Размер", self.brush_size, 1, 220, "px", integer=True)
            self._add_scale("Жёсткость", self.hardness, 0.0, 1.0, "%", percent=True)
            self._add_scale("Сила", self.retouch_strength, 0.01, 1.0, "%", percent=True)
            self._add_source_transform_options()
            self._add_brush_engine_options(tool)
            shown = True
        elif tool == "spot_healing":
            ttk.Label(
                self.body,
                text="Проведите кистью по дефекту. Источник выбирается автоматически.",
                wraplength=220,
                justify=tk.LEFT,
            ).pack(fill=tk.X, padx=8, pady=(0, 6))
            self._add_scale("Размер", self.brush_size, 1, 220, "px", integer=True)
            self._add_scale("Жёсткость", self.hardness, 0.0, 1.0, "%", percent=True)
            self._add_scale("Непрозрачность", self.retouch_strength, 0.01, 1.0, "%", percent=True)
            ttk.Label(self.body, text="Режим").pack(anchor=tk.W, padx=8, pady=(6, 0))
            ttk.Combobox(
                self.body,
                textvariable=self.spot_healing_mode,
                values=["Соответствие окружению", "С учётом содержимого"],
                state="readonly",
            ).pack(fill=tk.X, padx=8)
            self._add_brush_engine_options(tool)
            shown = True
        elif tool == "patch":
            self._add_patch_options()
            shown = True
        elif tool == "gradient":
            self._add_gradient_options()
            shown = True
        elif tool == "text":
            self._add_text_options()
            shown = True
        elif tool == "crop":
            self._add_crop_options()
            shown = True
        elif tool == "quick_selection":
            self._add_scale("Размер", self.brush_size, 1, 220, "px", integer=True)
            shown = True
        if tool in {"brush", "eraser"}:
            self._add_paint_target()
            shown = True
        if tool in TOLERANCE_TOOLS:
            self._add_scale("Допуск", self.tolerance, 0, 128, integer=True)
            shown = True
        if tool == "color_range":
            ttk.Label(self.body, text="Цвет образца").pack(anchor=tk.W, padx=8, pady=(6, 0))
            sample = tk.Label(self.body, textvariable=self.color_range_sample_hex, background=self.color_range_sample_hex.get(), relief=tk.SOLID, borderwidth=1)
            sample.pack(fill=tk.X, padx=8, pady=(2, 4))
            ttk.Checkbutton(self.body, text="Образец со всех слоёв", variable=self.color_range_sample_all_layers).pack(anchor=tk.W, padx=8, pady=(3, 0))
            ttk.Button(self.body, text="Диапазон с предпросмотром...", command=self.open_color_range_dialog).pack(fill=tk.X, padx=8, pady=(6, 4))
            ttk.Label(self.body, text="Shift добавляет образец, Ctrl вычитает его.", wraplength=220, justify=tk.LEFT).pack(fill=tk.X, padx=8, pady=(0, 6))
        if tool == "quick_selection":
            self._add_scale("Сглаживание края", self.quick_smooth, 0, 12, "px", integer=True)
            self._add_scale("Радиус анализа", self.quick_edge_radius, 0, 12, "px", integer=True)
            self._add_scale("Привязка к краю", self.quick_edge_strength, 0.0, 1.0, "%", percent=True)
            shown = True
        if tool in RETOUCH_TOOLS:
            self._add_retouch_preset()
            shown = True
        if tool in SELECTION_TOOLS:
            self._add_selection_mode()
            if tool in {"select", "ellipse_select", "lasso", "magnetic_lasso", "polygon_lasso"}:
                self._add_spinbox("Растушёвка, px", self.selection_feather, 0, 250)
                ttk.Checkbutton(self.body, text="Сглаживание", variable=self.selection_antialias).pack(anchor=tk.W, padx=8, pady=(6, 0))
            if tool == "magic_wand":
                ttk.Checkbutton(self.body, text="Смежные пиксели", variable=self.magic_contiguous).pack(anchor=tk.W, padx=8, pady=(6, 0))
                ttk.Checkbutton(self.body, text="Образец со всех слоёв", variable=self.magic_sample_all_layers).pack(anchor=tk.W, padx=8, pady=(4, 0))
                ttk.Checkbutton(self.body, text="Сглаживание", variable=self.selection_antialias).pack(anchor=tk.W, padx=8, pady=(4, 0))
            text = ttk.Label(self.body, text="Shift добавляет к выделению, Ctrl вычитает, Shift+Ctrl пересекает.", wraplength=220, justify=tk.LEFT)
            text.pack(fill=tk.X, padx=8, pady=(4, 8))
            shown = True
        if tool in SHAPE_TOOLS:
            self._add_shape_options(tool)
            shown = True
        if tool == "hand":
            text = ttk.Label(self.body, text="Перетаскивайте холст мышью. Также работают пробел + левая кнопка и средняя кнопка мыши.", wraplength=220, justify=tk.LEFT)
            text.pack(fill=tk.X, padx=8, pady=(4, 8))
            shown = True
        if not shown:
            ttk.Label(self.body, text="Для этого инструмента нет дополнительных параметров.", wraplength=220, justify=tk.LEFT).pack(fill=tk.X, padx=8, pady=4)

    def _render_compact(self, tool: str) -> None:
        primary = ttk.Frame(self.body)
        primary.pack(side=tk.LEFT, fill=tk.X, expand=True)
        if tool == "move" and self.auto_select is not None:
            ttk.Checkbutton(primary, text="Автовыбор", variable=self.auto_select).pack(side=tk.LEFT, padx=4)
        elif tool in BRUSH_TOOLS:
            self._compact_spin(primary, "Размер", self.brush_size, 1, 220, 5)
            variable = self.opacity if tool in {"brush", "eraser", "clone"} else self.exposure if tool in {"dodge", "burn"} else self.retouch_strength
            self._compact_percent(primary, "Непрозрачность" if tool in {"brush", "eraser", "clone", "healing", "spot_healing"} else "Сила", variable)
            if tool in BRUSH_TOOLS:
                self._compact_percent(primary, "Поток", self.brush_flow)
            if tool == "spot_healing":
                self._compact_combo(primary, "Режим", self.spot_healing_mode, ["Соответствие окружению", "С учётом содержимого"], 22)
            if tool in {"brush", "eraser"}:
                self._compact_single_color(primary, self.pick_foreground, 0, "Основной цвет")
        elif tool in SHAPE_TOOLS:
            self._compact_color_pair(primary, "Заливка", "Обводка")
            self._compact_spin(primary, "Толщина", self.shape_stroke_width, 0, 100, 4)
            if tool == "polygon_shape":
                self._compact_spin(primary, "Стороны", self.polygon_sides, 3, 64, 4)
            elif tool == "star_shape":
                self._compact_spin(primary, "Лучи", self.star_points, 3, 64, 4)
        elif tool == "gradient":
            self._compact_combo(primary, "Режим", self.gradient_mode, ["Заливка", "Объект"], 9)
            if self.gradient_mode.get() == "Объект":
                self._compact_combo(primary, "Фигура", self.gradient_shape, ["Прямоугольник", "Эллипс", "Многоугольник", "Звезда", "Произвольная"], 12)
                self._compact_combo(primary, "Заливка", self.gradient_object_fill, ["Градиент", "Текстура"], 9)
            if self.gradient_mode.get() == "Заливка" or self.gradient_object_fill.get() == "Градиент":
                self._compact_combo(primary, "Тип", self.gradient_type, ["Линейный", "Радиальный", "Отраженный", "Ромб", "Угловой"], 12)
            else:
                self._compact_combo(primary, "Текстура", self.gradient_texture, ["Шахматная", "Полосы", "Точки"], 10)
            self._compact_color_pair(primary, "Цвет 1", "Цвет 2")
        elif tool == "text":
            self._compact_combo(primary, "Шрифт", self.text_font_family, ["Arial", "Segoe UI", "Calibri", "Times New Roman", "Verdana", "Tahoma"], 15, readonly=False)
            self._compact_spin(primary, "Размер", self.text_size, 4, 500, 5)
            self._compact_single_color(primary, self.pick_foreground, 0, "Цвет текста")
            ttk.Checkbutton(primary, text="B", variable=self.text_bold, style="Toolbutton").pack(side=tk.LEFT, padx=2)
            ttk.Checkbutton(primary, text="I", variable=self.text_italic, style="Toolbutton").pack(side=tk.LEFT, padx=2)
        elif tool == "crop":
            self._compact_combo(primary, "Формат", self.crop_aspect, ["Свободно", "1:1", "4:3", "3:2", "16:9", "Исходное", "Свое"], 10)
        elif tool in SELECTION_TOOLS:
            self._compact_combo(primary, "Режим", self.selection_mode, ["replace", "add", "subtract", "intersect"], 10)
            if tool in TOLERANCE_TOOLS:
                self._compact_spin(primary, "Допуск", self.tolerance, 0, 128, 4)
            if tool == "quick_selection":
                self._compact_spin(primary, "Размер", self.brush_size, 1, 220, 5)
            elif tool == "magic_wand":
                ttk.Checkbutton(primary, text="Смежные", variable=self.magic_contiguous).pack(side=tk.LEFT, padx=3)
                ttk.Checkbutton(primary, text="Все слои", variable=self.magic_sample_all_layers).pack(side=tk.LEFT, padx=3)
            elif tool == "color_range":
                ttk.Checkbutton(primary, text="Все слои", variable=self.color_range_sample_all_layers).pack(side=tk.LEFT, padx=3)
                ttk.Button(primary, text="Предпросмотр...", command=self.open_color_range_dialog).pack(side=tk.LEFT, padx=3)
        elif tool == "fill":
            self._compact_single_color(primary, self.pick_foreground, 0, "Цвет заливки")
            self._compact_spin(primary, "Допуск", self.tolerance, 0, 128, 4)
        elif tool == "patch":
            self._compact_spin(primary, "Структура", self.patch_structure, 1, 7, 3)
            self._compact_spin(primary, "Цвет", self.patch_color_adaptation, 0, 10, 3)
            ttk.Checkbutton(primary, text="Все слои", variable=self.patch_sample_all_layers).pack(side=tk.LEFT, padx=4)
            ttk.Button(primary, text="Применить", command=self.apply_patch_preview).pack(side=tk.LEFT, padx=4)

        gradient_has_stops = tool == "gradient" and (self.gradient_mode.get() == "Заливка" or self.gradient_object_fill.get() == "Градиент")
        if tool in BRUSH_TOOLS or tool in {"text", "crop", "quick_selection", "star_shape", "custom_shape", "patch"} or gradient_has_stops:
            ttk.Button(primary, text="Дополнительно", command=self._toggle_compact_advanced).pack(side=tk.LEFT, padx=(8, 3))
        if self._advanced_visible:
            self._render_compact_advanced(primary, tool)

    def _toggle_compact_advanced(self) -> None:
        self._advanced_visible = not self._advanced_visible
        self.render()

    def _render_compact_advanced(self, parent: ttk.Frame, tool: str) -> None:
        ttk.Separator(parent, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=7, pady=4)
        if tool in BRUSH_TOOLS:
            self._compact_percent(parent, "Жёсткость", self.hardness)
        if tool in BRUSH_TOOLS:
            self._compact_percent(parent, "Интервал", self.brush_spacing)
            self._compact_percent(parent, "Сглаживание", self.brush_smoothing)
            if tool == "brush":
                self._compact_combo(parent, "Режим", self.brush_blend_mode, list(BRUSH_BLEND_MODES), 10)
            self._compact_brush_presets(parent)
        if tool in {"clone", "healing"}:
            ttk.Checkbutton(parent, text="Выровненный", variable=self.clone_aligned).pack(side=tk.LEFT, padx=3)
            ttk.Button(parent, text="Источник...", command=self.open_clone_source_panel).pack(side=tk.LEFT, padx=3)
        elif tool == "gradient" and (self.gradient_mode.get() == "Заливка" or self.gradient_object_fill.get() == "Градиент"):
            ttk.Checkbutton(parent, text="Средняя точка", variable=self.gradient_mid_enabled, command=lambda: self.after_idle(self.render)).pack(side=tk.LEFT, padx=3)
            if self.gradient_mid_enabled.get():
                self._compact_percent(parent, "Позиция", self.gradient_mid_position)
                ttk.Button(parent, text="Цвет", command=self.pick_gradient_mid).pack(side=tk.LEFT, padx=3)
        elif tool == "text":
            self._compact_combo(parent, "Выравнивание", self.text_align, ["left", "center", "right"], 9)
            self._compact_spin(parent, "Интервал", self.text_tracking, -100, 500, 5)
            ttk.Button(parent, text="Контур...", command=self.edit_text_path).pack(side=tk.LEFT, padx=3)
        elif tool == "crop" and self.crop_aspect.get() == "Свое":
            self._compact_spin(parent, "Ширина", self.crop_custom_width, 1, 50000, 7)
            self._compact_spin(parent, "Высота", self.crop_custom_height, 1, 50000, 7)
        elif tool == "quick_selection":
            self._compact_spin(parent, "Сглаживание", self.quick_smooth, 0, 12, 4)
        elif tool == "star_shape":
            self._compact_percent(parent, "Внутренний радиус", self.star_inner_ratio)
