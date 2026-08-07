from __future__ import annotations

import tkinter as tk
from tkinter import ttk


BRUSH_TOOLS = {"brush", "eraser", "blur_tool", "sharpen_tool", "dodge", "burn", "clone", "healing", "spot_healing"}
RETOUCH_TOOLS = {"blur_tool", "sharpen_tool", "dodge", "burn", "clone", "healing", "spot_healing"}
TOLERANCE_TOOLS = {"fill", "magic_wand", "color_range", "quick_selection"}
COLOR_TOOLS = {"brush", "fill", "gradient", "text"}
SHAPE_TOOLS = {"rect_shape", "ellipse_shape", "line_shape", "bezier_shape", "polygon_shape", "star_shape", "custom_shape"}
SELECTION_TOOLS = {"select", "ellipse_select", "lasso", "magnetic_lasso", "polygon_lasso", "quick_selection", "magic_wand", "color_range"}


class ToolOptionsPanel(ttk.Frame):
    def __init__(
        self,
        parent: tk.Widget,
        *,
        tool_var: tk.StringVar,
        definitions: list[tuple[str, str, str]],
        brush_size: tk.IntVar,
        opacity: tk.DoubleVar,
        hardness: tk.DoubleVar,
        retouch_strength: tk.DoubleVar,
        exposure: tk.DoubleVar,
        tonal_range: tk.StringVar,
        tolerance: tk.IntVar,
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
        tooltip_factory=None,
    ) -> None:
        super().__init__(parent)
        self.tool_var = tool_var
        self.definitions = definitions
        self.label_by_id = {value: label for label, value, _description in definitions}
        self.description_by_id = {value: description for label, value, description in definitions}
        self.brush_size = brush_size
        self.opacity = opacity
        self.hardness = hardness
        self.retouch_strength = retouch_strength
        self.exposure = exposure
        self.tonal_range = tonal_range
        self.tolerance = tolerance
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
        self.tooltip_factory = tooltip_factory
        self.title = ttk.Label(self, text="Параметры инструмента")
        self.title.pack(anchor=tk.W, padx=8, pady=(8, 4))
        self.body = ttk.Frame(self)
        self.body.pack(fill=tk.BOTH, expand=True)
        self.render()

    def render(self) -> None:
        for child in self.body.winfo_children():
            child.destroy()
        tool = self.tool_var.get()
        label = self.label_by_id.get(tool, tool)
        self.title.configure(text=f"Параметры: {label}")
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
            self._add_scale("Непрозрачность", self.opacity, 0.01, 1.0, "%", percent=True)
            shown = True
        elif tool in {"blur_tool", "sharpen_tool"}:
            self._add_scale("Размер", self.brush_size, 1, 220, "px", integer=True)
            self._add_scale("Жёсткость", self.hardness, 0.0, 1.0, "%", percent=True)
            self._add_scale("Сила", self.retouch_strength, 0.01, 1.0, "%", percent=True)
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
            shown = True
        elif tool in {"healing", "spot_healing"}:
            self._add_scale("Размер", self.brush_size, 1, 220, "px", integer=True)
            self._add_scale("Жёсткость", self.hardness, 0.0, 1.0, "%", percent=True)
            self._add_scale("Сила", self.retouch_strength, 0.01, 1.0, "%", percent=True)
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
        ttk.Label(row, text=label).pack(side=tk.LEFT)
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
        show()

    def _add_spinbox(self, label: str, variable: tk.Variable, start: int, end: int) -> None:
        row = ttk.Frame(self.body)
        row.pack(fill=tk.X, padx=8, pady=(6, 0))
        ttk.Label(row, text=label).pack(side=tk.LEFT)
        ttk.Spinbox(row, textvariable=variable, from_=start, to=end, increment=1, width=7).pack(side=tk.RIGHT)

    def _add_tonal_range(self) -> None:
        ttk.Label(self.body, text="Диапазон").pack(anchor=tk.W, padx=8, pady=(6, 0))
        ttk.Combobox(
            self.body,
            textvariable=self.tonal_range,
            values=["Тени", "Средние тона", "Света"],
            state="readonly",
        ).pack(fill=tk.X, padx=8)

    def _add_shape_options(self, tool: str) -> None:
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
        fg = ttk.Button(row, text="Передний цвет", command=self.pick_foreground)
        fg.pack(side=tk.LEFT, fill=tk.X, expand=True)
        bg = ttk.Button(row, text="Фон", command=self.pick_background)
        bg.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        if self.tooltip_factory is not None:
            self.tooltip_factory(fg, "Цвет кисти, заливки, текста и фигур.")
            self.tooltip_factory(bg, "Второй цвет для градиента и обводки фигур.")

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
