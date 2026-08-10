from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import json
import math
import os
from pathlib import Path
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import colorchooser, filedialog, messagebox, simpledialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageGrab, ImageSequence, ImageTk

from .core import (
    Document,
    Layer,
    BLEND_MODES,
    GradientEngine,
    SourceAnchor,
    add_noise,
    adjust_brightness_contrast,
    adjust_color_balance,
    adjust_exposure,
    adjust_hue_saturation,
    adjust_posterize,
    adjust_saturation,
    adjust_temperature_tint,
    adjust_threshold,
    adjust_vibrance,
    automatic_selection_mask,
    apply_gradient,
    apply_filter_stack,
    bezier_curve_points,
    custom_shape_points,
    blur,
    curves,
    content_aware_fill,
    clone_or_heal,
    draw_mask_brush,
    draw_brush,
    encode_png,
    edge_aware_cleanup,
    flood_fill,
    frequency_separation,
    generative_expand_pixels,
    image_statistics,
    effective_layer_mask,
    levels,
    local_retouch,
    paste_mask,
    portrait_cleanup,
    refine_selection_mask,
    refine_selection_brush,
    decontaminate_edge_colors,
    rgba_array_to_pil,
    reduce_red_eye,
    regular_polygon_points,
    render_shape_layer,
    render_text_layer,
    normalize_text_path_points,
    text_path_samples,
    text_path_point_at_distance,
    RetouchStroke,
    RAW_EXTENSIONS,
    selection_edge_confidence,
    selection_contour_points,
    shape_drag_is_meaningful,
    shape_data_bounds,
    shape_geometry_from_drag,
    layer_contains_point,
    resize_box_from_handle,
    star_points,
    topmost_layer_at,
    transform_shape_data_to_box,
    union_rect,
    warp_pixels,
    sharpen,
    spot_heal,
)
from .history import (
    DocumentStateCommand,
    DocumentFieldsCommand,
    History,
    LayerBlendModeCommand,
    LayerDeleteCommand,
    LayersDeleteCommand,
    LayerFieldsCommand,
    LayerInsertCommand,
    LayerMoveCommand,
    LayerOpacityCommand,
    LayerVisibilityCommand,
    LayerPropertyCommand,
    LayerReorderCommand,
    MaskPatchCommand,
    MaskTilePatchCommand,
    PixelPatchCommand,
    PixelTilePatchCommand,
    SelectionMaskCommand,
    ShapeDataCommand,
    TextDataCommand,
    TilePatch,
)
from .rendering import RenderEngine
from .automation import ActionRecorder, ActionRunner
from .color_management import COLOR_MODELS, BIT_DEPTHS, color_settings
from .plugins import PluginRegistry
from .ui.tool_options import ToolOptionsPanel
from .ui.tool_palette import ToolPalette, ToolPaletteDialog, normalize_tool_order, normalize_visible_tools
from .ui.icons import SHORTCUTS, action_icon
from .ui.shortcuts import TOOL_SHORTCUT_GROUPS, accelerator, command_for_event, event_key
from .ui.scrollable_frame import ScrollableFrame
from .ui.theme import TOKENS, configure_theme


TOOL_DEMO_DIR = Path(__file__).resolve().parent / "assets" / "tool_demos"


def tool_demo_path(tool_id: str) -> Path:
    return TOOL_DEMO_DIR / f"{tool_id}.gif"


class ToolTip:
    def __init__(self, widget: tk.Widget, text: str, delay: int = 450, *, demo: str | None = None) -> None:
        self.widget = widget
        self.text = text
        self.delay = delay
        self.demo = demo
        self._after_id: str | None = None
        self._animation_id: str | None = None
        self._tip: tk.Toplevel | None = None
        self._demo_label: tk.Label | None = None
        self._frames: list[ImageTk.PhotoImage] = []
        self._durations: list[int] = []
        self._frame_index = 0
        widget.bind("<Enter>", self.schedule, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<ButtonPress>", self.hide, add="+")

    def schedule(self, _event=None) -> None:
        self.cancel()
        self._after_id = self.widget.after(self.delay, self.show)

    def cancel(self) -> None:
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def show(self) -> None:
        if self._tip is not None:
            return
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        try:
            self._tip.wm_attributes("-topmost", True)
        except tk.TclError:
            pass
        body = tk.Frame(
            self._tip,
            background=TOKENS.SURFACE_HOVER,
            highlightbackground=TOKENS.BORDER,
            highlightthickness=1,
            padx=8,
            pady=7,
        )
        body.pack()
        label = tk.Label(
            body,
            text=self.text,
            justify=tk.LEFT,
            background=TOKENS.SURFACE_HOVER,
            foreground=TOKENS.TEXT_PRIMARY,
            borderwidth=0,
            padx=0,
            pady=0,
            wraplength=288,
        )
        label.pack(anchor=tk.W)
        self._load_demo_frames()
        if self._frames:
            self._demo_label = tk.Label(body, borderwidth=0, background=TOKENS.SURFACE_HOVER)
            self._demo_label.pack(pady=(7, 0))
            self._frame_index = 0
            self._animate()

        self._tip.update_idletasks()
        tip_width = self._tip.winfo_reqwidth()
        tip_height = self._tip.winfo_reqheight()
        x = self.widget.winfo_rootx() + self.widget.winfo_width() + 10
        y = self.widget.winfo_rooty()
        screen_width = self.widget.winfo_screenwidth()
        screen_height = self.widget.winfo_screenheight()
        if x + tip_width > screen_width - 8:
            x = self.widget.winfo_rootx() - tip_width - 10
        x = max(8, min(x, screen_width - tip_width - 8))
        y = max(8, min(y, screen_height - tip_height - 8))
        self._tip.wm_geometry(f"+{x}+{y}")

    def _load_demo_frames(self) -> None:
        if self._frames or not self.demo:
            return
        path = tool_demo_path(self.demo)
        try:
            with Image.open(path) as animation:
                for frame in ImageSequence.Iterator(animation):
                    rendered = frame.convert("RGB")
                    self._frames.append(ImageTk.PhotoImage(rendered, master=self.widget))
                    self._durations.append(max(60, min(250, int(frame.info.get("duration", 110)))))
        except (OSError, tk.TclError):
            self._frames.clear()
            self._durations.clear()

    def _animate(self) -> None:
        if self._tip is None or self._demo_label is None or not self._frames:
            return
        index = self._frame_index % len(self._frames)
        self._demo_label.configure(image=self._frames[index])
        self._frame_index = (index + 1) % len(self._frames)
        self._animation_id = self.widget.after(self._durations[index], self._animate)

    def hide(self, _event=None) -> None:
        self.cancel()
        if self._animation_id is not None:
            try:
                self.widget.after_cancel(self._animation_id)
            except tk.TclError:
                pass
            self._animation_id = None
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None
        self._demo_label = None


TOOL_DEFINITIONS = [
    ("Рука", "hand", "Перемещает холст по экрану без изменения изображения."),
    ("Перемещение", "move", "Двигает активный слой мышью."),
    ("Кисть", "brush", "Рисует выбранным передним цветом по пикселям или маске."),
    ("Ластик", "eraser", "Стирает пиксели слоя или закрашивает маску черным."),
    ("Размытие", "blur_tool", "Локально смягчает участок активного слоя."),
    ("Резкость", "sharpen_tool", "Локально усиливает контраст деталей."),
    ("Осветлитель", "dodge", "Осветляет область под кистью."),
    ("Затемнитель", "burn", "Затемняет область под кистью."),
    ("Штамп", "clone", "Копирует пиксели из источника. Alt+клик задает источник."),
    ("Лечение", "healing", "Копирует источник и подгоняет цвет под место назначения."),
    ("Точечное восстановление", "spot_healing", "Автоматически удаляет мелкие дефекты без выбора источника."),
    ("Заплатка", "patch", "Перетащите активное выделение на область-источник для ретуши."),
    ("Заливка", "fill", "Заполняет связанную область выбранным цветом."),
    ("Градиент", "gradient", "Растягивает переход от переднего цвета к фоновому."),
    ("Текст", "text", "Создает редактируемый текстовый слой."),
    ("Пипетка", "eyedropper", "Берет цвет из изображения в передний цвет."),
    ("Прямоуг. фигура", "rect_shape", "Создает редактируемый прямоугольник."),
    ("Эллипс", "ellipse_shape", "Создает редактируемый эллипс."),
    ("Линия", "line_shape", "Создает редактируемую линию."),
    ("Кривая Безье", "bezier_shape", "Создает редактируемую кубическую кривую Безье."),
    ("Многоугольник", "polygon_shape", "Создает редактируемый многоугольник."),
    ("Звезда", "star_shape", "Создает редактируемую звезду."),
    ("Своя фигура", "custom_shape", "Создает фигуру из библиотеки пользовательских контуров."),
    ("Прямоуг. выделение", "select", "Выделяет прямоугольную область."),
    ("Эллипс выделение", "ellipse_select", "Выделяет овальную область."),
    ("Лассо", "lasso", "Рисует свободный контур выделения от руки."),
    ("Магнитное лассо", "magnetic_lasso", "Привязывает контур выделения к ближайшим сильным краям."),
    ("Полигон", "polygon_lasso", "Строит выделение кликами по вершинам. Двойной клик завершает."),
    ("Быстрое выделение", "quick_selection", "Добавляет похожие соседние области кистью."),
    ("Волшебная палочка", "magic_wand", "Выделяет связанную область похожего цвета."),
    ("Цветовой диапазон", "color_range", "Выделяет похожий цвет по всему активному слою."),
    ("Кадрирование", "crop", "Задает область обрезки документа."),
]

RETOUCH_PRESETS = {
    "Мягкая ретушь": {"brush_size": 18, "opacity": 0.22, "tolerance": 18, "tool": "healing"},
    "Средняя ретушь": {"brush_size": 34, "opacity": 0.45, "tolerance": 28, "tool": "healing"},
    "Сильная ретушь": {"brush_size": 58, "opacity": 0.72, "tolerance": 44, "tool": "healing"},
    "Детальная ретушь": {"brush_size": 9, "opacity": 0.34, "tolerance": 12, "tool": "healing"},
    "Мелкие дефекты": {"brush_size": 6, "opacity": 0.95, "tolerance": 12, "tool": "spot_healing"},
    "Точечные пятна": {"brush_size": 14, "opacity": 0.9, "tolerance": 20, "tool": "spot_healing"},
}

TOOL_SETTINGS_DEFAULTS = {
    "brush": {"size": 28, "opacity": 1.0},
    "eraser": {"size": 28, "opacity": 1.0},
    "blur_tool": {"size": 32, "hardness": 0.5, "strength": 0.25},
    "sharpen_tool": {"size": 28, "hardness": 0.5, "strength": 0.2},
    "dodge": {"size": 32, "hardness": 0.45, "exposure": 0.15, "range": "Средние тона"},
    "burn": {"size": 32, "hardness": 0.45, "exposure": 0.15, "range": "Средние тона"},
    "clone": {"size": 28, "hardness": 0.5, "opacity": 0.8},
    "healing": {"size": 26, "hardness": 0.45, "strength": 0.65},
    "spot_healing": {"size": 10, "hardness": 0.4, "strength": 0.65},
}

CUSTOM_SHAPE_PRESETS = {
    "Ромб": [(0.5, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 0.5)],
    "Стрелка": [(0.0, 0.35), (0.62, 0.35), (0.62, 0.08), (1.0, 0.5), (0.62, 0.92), (0.62, 0.65), (0.0, 0.65)],
    "Сердце": [(0.5, 1.0), (0.08, 0.58), (0.0, 0.28), (0.16, 0.05), (0.38, 0.08), (0.5, 0.25), (0.62, 0.08), (0.84, 0.05), (1.0, 0.28), (0.92, 0.58)],
    "Выноска": [(0.0, 0.0), (1.0, 0.0), (1.0, 0.72), (0.62, 0.72), (0.45, 1.0), (0.42, 0.72), (0.0, 0.72)],
}

DOCUMENT_PRESETS = [
    {"name": "Full HD", "description": "Экран и видео", "width": 1920, "height": 1080, "dpi": 72, "background": "Белый"},
    {"name": "4K UHD", "description": "Видео высокого разрешения", "width": 3840, "height": 2160, "dpi": 72, "background": "Черный"},
    {"name": "Квадрат 1080", "description": "Публикация в соцсетях", "width": 1080, "height": 1080, "dpi": 72, "background": "Белый"},
    {"name": "Портрет 4:5", "description": "Вертикальная публикация", "width": 1080, "height": 1350, "dpi": 72, "background": "Белый"},
    {"name": "История", "description": "Экран смартфона 9:16", "width": 1080, "height": 1920, "dpi": 72, "background": "Белый"},
    {"name": "A4", "description": "Печать 210 x 297 мм", "width": 2480, "height": 3508, "dpi": 300, "background": "Белый"},
    {"name": "A3", "description": "Печать 297 x 420 мм", "width": 3508, "height": 4961, "dpi": 300, "background": "Белый"},
    {"name": "Фото 10 x 15", "description": "Печать фотографии", "width": 1200, "height": 1800, "dpi": 300, "background": "Белый"},
    {"name": "Иконка", "description": "Прозрачный квадрат", "width": 1024, "height": 1024, "dpi": 72, "background": "Прозрачный"},
    {"name": "Свой размер", "description": "Ручная настройка", "width": 1280, "height": 900, "dpi": 72, "background": "Белый"},
]

DOCUMENT_BACKGROUNDS = {
    "Белый": (255, 255, 255, 255),
    "Черный": (0, 0, 0, 255),
    "Прозрачный": (0, 0, 0, 0),
}

MASK_PREVIEW_NORMAL = "Обычный"
MASK_PREVIEW_OVERLAY = "Красное перекрытие"
MASK_PREVIEW_CHANNEL = "Черно-белая маска"
MASK_PREVIEW_MODES = [MASK_PREVIEW_NORMAL, MASK_PREVIEW_OVERLAY, MASK_PREVIEW_CHANNEL]

SELECT_MASK_PREVIEW_CHANNEL = "Канал маски"
SELECT_MASK_PREVIEW_OVERLAY = "Красное перекрытие"
SELECT_MASK_PREVIEW_CUTOUT = "Вырез на прозрачности"
SELECT_MASK_PREVIEW_EDGE_CONFIDENCE = "Уверенность края"
SELECT_MASK_PREVIEW_MODES = [SELECT_MASK_PREVIEW_CHANNEL, SELECT_MASK_PREVIEW_OVERLAY, SELECT_MASK_PREVIEW_CUTOUT, SELECT_MASK_PREVIEW_EDGE_CONFIDENCE]

FILTER_TYPES = ["blur", "sharpen", "noise", "median", "edge", "emboss"]
FILTER_LABELS = {
    "blur": "Размытие",
    "sharpen": "Резкость",
    "noise": "Шум",
    "median": "Медиана",
    "edge": "Края",
    "emboss": "Тиснение",
}

FILTER_STACK_PRESETS = {
    "Портрет мягко": [
        {"type": "median", "size": 3, "opacity": 0.28, "blend_mode": "Normal"},
        {"type": "blur", "radius": 2, "opacity": 0.18, "blend_mode": "Soft Light"},
        {"type": "sharpen", "amount": 0.7, "opacity": 0.35, "blend_mode": "Normal"},
    ],
    "Детальная резкость": [
        {"type": "sharpen", "amount": 1.8, "opacity": 0.75, "blend_mode": "Normal"},
        {"type": "edge", "strength": 0.28, "opacity": 0.22, "blend_mode": "Overlay"},
    ],
    "Чистый скан": [
        {"type": "median", "size": 3, "opacity": 0.8, "blend_mode": "Normal"},
        {"type": "sharpen", "amount": 0.9, "opacity": 0.55, "blend_mode": "Normal"},
    ],
    "Графичные края": [
        {"type": "edge", "strength": 0.75, "opacity": 0.6, "blend_mode": "Multiply"},
        {"type": "emboss", "strength": 0.35, "opacity": 0.25, "blend_mode": "Overlay"},
    ],
}

ADJUSTMENT_TYPES = [
    "brightness_contrast",
    "saturation",
    "vibrance",
    "temperature_tint",
    "hue_saturation",
    "exposure",
    "color_balance",
    "levels",
    "curves",
    "threshold",
    "posterize",
    "invert",
    "grayscale",
]
ADJUSTMENT_LABELS = {
    "brightness_contrast": "\u042f\u0440\u043a\u043e\u0441\u0442\u044c/\u041a\u043e\u043d\u0442\u0440\u0430\u0441\u0442",
    "saturation": "\u041d\u0430\u0441\u044b\u0449\u0435\u043d\u043d\u043e\u0441\u0442\u044c",
    "vibrance": "Вибрация",
    "temperature_tint": "Температура/Оттенок",
    "hue_saturation": "\u0422\u043e\u043d/\u041d\u0430\u0441\u044b\u0449\u0435\u043d\u043d\u043e\u0441\u0442\u044c",
    "exposure": "\u042d\u043a\u0441\u043f\u043e\u0437\u0438\u0446\u0438\u044f",
    "color_balance": "\u0426\u0432\u0435\u0442\u043e\u0432\u043e\u0439 \u0431\u0430\u043b\u0430\u043d\u0441",
    "levels": "\u0423\u0440\u043e\u0432\u043d\u0438",
    "curves": "\u041a\u0440\u0438\u0432\u044b\u0435",
    "threshold": "\u041f\u043e\u0440\u043e\u0433",
    "posterize": "\u041f\u043e\u0441\u0442\u0435\u0440\u0438\u0437\u0430\u0446\u0438\u044f",
    "invert": "\u0418\u043d\u0432\u0435\u0440\u0441\u0438\u044f",
    "grayscale": "\u0427\u0435\u0440\u043d\u043e-\u0431\u0435\u043b\u043e\u0435",
}

ADJUSTMENT_PRESETS = {
    "Яркий портрет": {"type": "brightness_contrast", "brightness": 12, "contrast": 1.18},
    "Теплый тон": {"type": "color_balance", "red": 12, "green": 4, "blue": -10},
    "Кино-контраст": {"type": "curves", "shadows": 48, "midtones": 128, "highlights": 210},
    "Мягкая экспозиция": {"type": "exposure", "exposure": 0.25, "offset": 0.0, "gamma": 1.08},
    "Холодные тени": {"type": "color_balance", "red": -8, "green": 0, "blue": 14},
    "Чистый черно-белый": {"type": "grayscale"},
}


ADJUSTMENT_PRESETS.update(
    {
        "Живые цвета": {"type": "vibrance", "vibrance": 0.42, "saturation": 1.05},
        "Золотой час": {"type": "temperature_tint", "temperature": 26, "tint": 5},
        "Холодный свет": {"type": "temperature_tint", "temperature": -22, "tint": -3},
    }
)


class PhotoRedactorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.theme = TOKENS
        self.ui_style = configure_theme(self)
        self.configure(background=TOKENS.BACKGROUND)
        self.withdraw()
        self.title("PhotoRedactor - редактор изображений")
        self.geometry("1440x920")
        self.minsize(1000, 640)

        self.doc = Document.new()
        self.history = History()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.app_data_dir = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "PhotoRedactor"
        self.recovery_path = self.app_data_dir / "recovery.prdx"
        self.settings_path = self.app_data_dir / "settings.json"
        self.recent_files: list[str] = []
        self.action_recorder = ActionRecorder()
        self.action_runner = ActionRunner()
        self.plugin_registry = PluginRegistry()
        self.plugin_registry.discover()
        for name, callback in self.plugin_registry.action_commands.items():
            self.action_runner.register(name, callback)
        self._edit_generation = 0
        self.adjustment_presets = {name: dict(value) for name, value in ADJUSTMENT_PRESETS.items()}
        self.tool = tk.StringVar(value="brush")
        self.auto_select = tk.BooleanVar(value=True)
        self.tool_order = [value for _label, value, _description in TOOL_DEFINITIONS]
        self.visible_tools = list(self.tool_order)
        self.tool_pane_position = 360
        self.custom_canvas_width = 1280
        self.custom_canvas_height = 900
        self.custom_canvas_dpi = 72
        self.custom_canvas_background = "Белый"
        self.paint_target = tk.StringVar(value="pixels")
        self.selection_mode = tk.StringVar(value="replace")
        self.retouch_preset = tk.StringVar(value="Средняя ретушь")
        self.zoom = tk.DoubleVar(value=1.0)
        self.brush_size = tk.IntVar(value=28)
        self.opacity = tk.DoubleVar(value=1.0)
        self.hardness = tk.DoubleVar(value=0.5)
        self.retouch_strength = tk.DoubleVar(value=0.25)
        self.exposure = tk.DoubleVar(value=0.15)
        self.tonal_range = tk.StringVar(value="Средние тона")
        self.tool_settings = copy.deepcopy(TOOL_SETTINGS_DEFAULTS)
        self._active_settings_tool = "brush"
        self.shape_stroke_width = tk.IntVar(value=2)
        self.polygon_sides = tk.IntVar(value=5)
        self.star_points_count = tk.IntVar(value=5)
        self.star_inner_ratio = tk.DoubleVar(value=0.5)
        self.custom_shape_preset = tk.StringVar(value=next(iter(CUSTOM_SHAPE_PRESETS)))
        self.selection_feather = tk.IntVar(value=0)
        self.selection_antialias = tk.BooleanVar(value=True)
        self.magic_contiguous = tk.BooleanVar(value=True)
        self.tolerance = tk.IntVar(value=24)
        self.color_range_sample_hex = tk.StringVar(value="#000000")
        self.clone_aligned = tk.BooleanVar(value=True)
        self.clone_sampling = tk.StringVar(value="Текущий слой")
        self.gradient_type = tk.StringVar(value="Линейный")
        self.gradient_mode = tk.StringVar(value="Заливка")
        self.gradient_shape = tk.StringVar(value="Прямоугольник")
        self.gradient_object_fill = tk.StringVar(value="Градиент")
        self.gradient_texture = tk.StringVar(value="Шахматная")
        self.gradient_mid_enabled = tk.BooleanVar(value=False)
        self.gradient_mid_position = tk.DoubleVar(value=0.5)
        self.gradient_mid_color = (255, 90, 80, 255)
        self.crop_aspect = tk.StringVar(value="Свободно")
        self.crop_custom_width = tk.IntVar(value=1920)
        self.crop_custom_height = tk.IntVar(value=1080)
        self.text_font_family = tk.StringVar(value="Arial")
        self.text_size = tk.IntVar(value=48)
        self.text_bold = tk.BooleanVar(value=False)
        self.text_italic = tk.BooleanVar(value=False)
        self.text_underline = tk.BooleanVar(value=False)
        self.text_align = tk.StringVar(value="left")
        self.text_line_spacing = tk.IntVar(value=10)
        self.text_tracking = tk.IntVar(value=0)
        self.text_rotation = tk.DoubleVar(value=0.0)
        self.text_box_width = tk.IntVar(value=0)
        self.quick_smooth = tk.IntVar(value=1)
        self.quick_edge_radius = tk.IntVar(value=2)
        self.quick_edge_strength = tk.DoubleVar(value=0.5)
        self.grid_visible = tk.BooleanVar(value=False)
        self.grid_spacing = tk.IntVar(value=64)
        self.view_channel = tk.StringVar(value="RGB")
        self.mask_preview = tk.StringVar(value=MASK_PREVIEW_NORMAL)
        self.foreground = (30, 120, 255, 255)
        self.background = (255, 255, 255, 255)
        self.drag_start: tuple[int, int] | None = None
        self.last_point: tuple[int, int] | None = None
        self._move_layer_id: str | None = None
        self._move_start: tuple[int, int] | None = None
        self._move_start_mask: np.ndarray | None = None
        self._move_last_bounds: tuple[int, int, int, int] | None = None
        self._object_resize_handle: str | None = None
        self._object_resize_before: dict | None = None
        self._object_resize_layer_id: str | None = None
        self._object_resize_rendered_bounds: tuple[int, int, int, int] | None = None
        self._last_object_resize_render = 0.0
        self._object_bounds_ids: list[int] = []
        self._stroke_layer_id: str | None = None
        self._stroke_kind = "pixels"
        self._stroke_rect: tuple[int, int, int, int] | None = None
        self._stroke_before: np.ndarray | None = None
        self._stroke_tiles: dict[tuple[int, int], tuple[tuple[int, int, int, int], np.ndarray]] = {}
        self._stroke_selection_mask: np.ndarray | None = None
        self._retouch_stroke: RetouchStroke | None = None
        self._opacity_layer_id: str | None = None
        self._opacity_before: float | None = None
        self._space_down = False
        self._panning = False
        self.selection_id: int | None = None
        self._selection_overlay_ids: list[int] = []
        self._selection_contours: list[np.ndarray] = []
        self._selection_contour_signature: tuple[int, int, int] | None = None
        self._selection_dash_phase = 0
        self._selection_animation_id: str | None = None
        self._drag_preview_ids: list[int] = []
        self._gradient_preview_id: int | None = None
        self._gradient_preview_image: ImageTk.PhotoImage | None = None
        self._last_gradient_preview_at = 0.0
        self._crop_box: tuple[int, int, int, int] | None = None
        self._crop_overlay_ids: list[int] = []
        self._crop_drag_handle: str | None = None
        self._crop_drag_origin_box: tuple[int, int, int, int] | None = None
        self._text_editor: tk.Text | None = None
        self._text_editor_window: int | None = None
        self._text_editor_layer_id: str | None = None
        self._text_editor_origin = (0, 0)
        self._text_editor_box_width = 0
        self._text_editor_before: dict[str, object] | None = None
        self._text_property_after_id: str | None = None
        self._text_property_before: dict[str, object] | None = None
        self._loading_text_properties = False
        self._shape_drag_options: dict | None = None
        self.selection_box: tuple[int, int, int, int] | None = None
        self._lasso_points: list[tuple[int, int]] = []
        self._polygon_points: list[tuple[int, int]] = []
        self._polygon_ids: list[int] = []
        self._magnetic_edges: np.ndarray | None = None
        self._quick_points: list[tuple[int, int]] = []
        self._quick_mode = "replace"
        self._quick_preview_mask: np.ndarray | None = None
        self._quick_preview_processed = 0
        self._quick_base_selection: np.ndarray | None = None
        self._quick_preview_id: int | None = None
        self._quick_preview_image: ImageTk.PhotoImage | None = None
        self._last_quick_preview_time = 0.0
        self._brush_preview_ids: list[int] = []
        self._last_pointer_event = None
        self._clone_source: tuple[int, int] | None = None
        self._clone_anchor_target: tuple[int, int] | None = None
        self._clone_anchor_source: tuple[int, int] | None = None
        self._source_anchor = SourceAnchor()
        self._clone_sample_pixels: np.ndarray | None = None
        self._clone_sample_origin = (0, 0)
        self._clone_source_marker_ids: list[int] = []
        self.selected_layer_ids: set[str] = {self.doc.layer.id}
        self._pixel_clipboard: np.ndarray | None = None
        self._pixel_clipboard_origin = (0, 0)
        self._patch_start_bounds: tuple[int, int, int, int] | None = None
        self._patch_preview_id: int | None = None
        self._guide_doc_lines: list[tuple[str, int]] = []
        self._overlay_ids: list[int] = []
        self._preview_image: ImageTk.PhotoImage | None = None
        self._layer_thumb_image: ImageTk.PhotoImage | None = None
        self._mask_thumb_image: ImageTk.PhotoImage | None = None
        self._select_mask_preview_image: ImageTk.PhotoImage | None = None
        self._mask_edge_preview_image: ImageTk.PhotoImage | None = None
        self._filter_stack_preview_image: ImageTk.PhotoImage | None = None
        self._adjustment_preview_image: ImageTk.PhotoImage | None = None
        self._transform_preview_image: ImageTk.PhotoImage | None = None
        self._warp_preview_image: ImageTk.PhotoImage | None = None
        self._bezier_preview_image: ImageTk.PhotoImage | None = None
        self._text_path_preview_image: ImageTk.PhotoImage | None = None
        self._frequency_preview_images: list[ImageTk.PhotoImage] = []
        self._portrait_preview_images: list[ImageTk.PhotoImage] = []
        self._startup_clipboard_preview: ImageTk.PhotoImage | None = None
        self._new_document_preview: ImageTk.PhotoImage | None = None
        self.startup_frame: tk.Frame | None = None
        self._editor_active = False
        self._canvas_image_id: int | None = None
        self._canvas_tile_ids: dict[tuple[int, int], int] = {}
        self._canvas_tile_images: dict[tuple[int, int], ImageTk.PhotoImage] = {}
        self._canvas_view_signature: tuple[object, ...] | None = None
        self._layer_thumbnail_cache: dict[tuple[object, ...], Image.Image] = {}
        self._mask_thumbnail_cache: dict[tuple[object, ...], Image.Image] = {}
        self._canvas_origin = (0, 0)
        self._render_after_id: str | None = None
        self._initial_fit_after_id: str | None = None
        self._performing_initial_fit = False
        self._last_render_time = 0.0
        self._composite_cache = None
        self._composite_dirty = True
        self._view_dirty = True
        self.render_engine = RenderEngine(tile_size=256)

        self._build_ui()
        self.tool.trace_add("write", self.tool_changed)
        self.brush_size.trace_add("write", self.brush_size_changed)
        self.tolerance.trace_add("write", self.quick_preview_settings_changed)
        self.quick_smooth.trace_add("write", self.quick_preview_settings_changed)
        self.quick_edge_radius.trace_add("write", self.quick_preview_settings_changed)
        self.quick_edge_strength.trace_add("write", self.quick_preview_settings_changed)
        self.selection_mode.trace_add("write", self.selection_mode_changed)
        for variable in (
            self.text_font_family, self.text_size, self.text_bold, self.text_italic,
            self.text_underline, self.text_align, self.text_line_spacing,
            self.text_tracking, self.text_rotation, self.text_box_width,
        ):
            variable.trace_add("write", self.text_properties_changed)
        self.load_settings()
        self.refresh_recent_menu()
        self.refresh()
        self.show_start_screen()
        self.deiconify()
        self.schedule_autosave()

    def destroy(self) -> None:
        if self._selection_animation_id is not None:
            try:
                self.after_cancel(self._selection_animation_id)
            except tk.TclError:
                pass
            self._selection_animation_id = None
        if self._initial_fit_after_id is not None:
            try:
                self.after_cancel(self._initial_fit_after_id)
            except tk.TclError:
                pass
            self._initial_fit_after_id = None
        self.autosave_recovery()
        self.save_settings()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.render_engine.scratch.close()
        super().destroy()

    def load_settings(self) -> None:
        try:
            if self.settings_path.exists():
                data = json.loads(self.settings_path.read_text(encoding="utf-8"))
                self.recent_files = [str(path) for path in data.get("recent_files", []) if Path(path).exists()]
                self.tool_order = normalize_tool_order(data.get("tool_order"), TOOL_DEFINITIONS)
                self.visible_tools = normalize_visible_tools(data.get("visible_tools"), self.tool_order)
                if int(data.get("tool_schema_version", 1)) < 2 and "spot_healing" not in self.visible_tools:
                    self.visible_tools.append("spot_healing")
                self.tool_pane_position = int(data.get("tool_pane_position", self.tool_pane_position))
                saved_tool_settings = data.get("tool_settings", {})
                if isinstance(saved_tool_settings, dict):
                    for tool_id, defaults in TOOL_SETTINGS_DEFAULTS.items():
                        saved = saved_tool_settings.get(tool_id)
                        if isinstance(saved, dict):
                            self.tool_settings[tool_id] = {**defaults, **{key: saved[key] for key in defaults if key in saved}}
                shape_settings = data.get("shape_settings", {})
                if isinstance(shape_settings, dict):
                    self.shape_stroke_width.set(max(0, min(100, int(shape_settings.get("stroke_width", 2)))))
                    self.polygon_sides.set(max(3, min(64, int(shape_settings.get("polygon_sides", 5)))))
                    self.star_points_count.set(max(3, min(64, int(shape_settings.get("star_points", 5)))))
                    self.star_inner_ratio.set(float(np.clip(shape_settings.get("star_inner_ratio", 0.5), 0.05, 0.95)))
                    preset = str(shape_settings.get("custom_shape_preset", self.custom_shape_preset.get()))
                    if preset in CUSTOM_SHAPE_PRESETS:
                        self.custom_shape_preset.set(preset)
                custom_canvas = data.get("custom_canvas", {})
                self.custom_canvas_width = max(1, min(50000, int(custom_canvas.get("width", self.custom_canvas_width))))
                self.custom_canvas_height = max(1, min(50000, int(custom_canvas.get("height", self.custom_canvas_height))))
                self.custom_canvas_dpi = max(1, min(2400, int(custom_canvas.get("dpi", self.custom_canvas_dpi))))
                saved_background = str(custom_canvas.get("background", self.custom_canvas_background))
                if saved_background in DOCUMENT_BACKGROUNDS:
                    self.custom_canvas_background = saved_background
                if self.tool.get() not in self.tool_order:
                    self.tool.set(self.visible_tools[0])
                self.load_active_tool_settings(self.tool.get())
                if hasattr(self, "tool_palette"):
                    self.tool_palette.set_configuration(self.tool_order, self.visible_tools)
                if hasattr(self, "tool_split"):
                    self.after_idle(self.apply_tool_pane_position)
                self.refresh_tool_menu()
        except Exception:
            self.recent_files = []
            self.tool_order = normalize_tool_order(None, TOOL_DEFINITIONS)
            self.visible_tools = normalize_visible_tools(None, self.tool_order)

    def save_settings(self) -> None:
        try:
            self.app_data_dir.mkdir(parents=True, exist_ok=True)
            self.capture_tool_pane_position()
            self.save_active_tool_settings()
            self.settings_path.write_text(
                json.dumps(
                    {
                        "recent_files": self.recent_files[:12],
                        "tool_order": self.tool_order,
                        "visible_tools": self.visible_tools,
                        "tool_schema_version": 2,
                        "tool_pane_position": self.tool_pane_position,
                        "tool_settings": self.tool_settings,
                        "shape_settings": {
                            "stroke_width": int(self.shape_stroke_width.get()),
                            "polygon_sides": int(self.polygon_sides.get()),
                            "star_points": int(self.star_points_count.get()),
                            "star_inner_ratio": float(self.star_inner_ratio.get()),
                            "custom_shape_preset": self.custom_shape_preset.get(),
                        },
                        "custom_canvas": {
                            "width": self.custom_canvas_width,
                            "height": self.custom_canvas_height,
                            "dpi": self.custom_canvas_dpi,
                            "background": self.custom_canvas_background,
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    def add_recent_file(self, path: str) -> None:
        resolved = str(Path(path))
        self.recent_files = [item for item in self.recent_files if item.lower() != resolved.lower()]
        self.recent_files.insert(0, resolved)
        self.recent_files = self.recent_files[:12]
        self.save_settings()
        self.refresh_recent_menu()

    def refresh_recent_menu(self) -> None:
        if not hasattr(self, "recent_menu"):
            return
        self.recent_menu.delete(0, tk.END)
        if not self.recent_files:
            self.recent_menu.add_command(label="Нет недавних файлов", state=tk.DISABLED)
            self.refresh_startup_recent()
            return
        for path in self.recent_files:
            label = Path(path).name
            self.recent_menu.add_command(label=label, command=lambda p=path: self.open_path(p))
        self.recent_menu.add_separator()
        self.recent_menu.add_command(label="Очистить список", command=self.clear_recent_files)
        self.refresh_startup_recent()

    def clear_recent_files(self) -> None:
        self.recent_files.clear()
        self.save_settings()
        self.refresh_recent_menu()

    @staticmethod
    def read_clipboard_image() -> Image.Image | None:
        try:
            value = ImageGrab.grabclipboard()
            if isinstance(value, Image.Image):
                return value.convert("RGBA")
            if isinstance(value, list):
                for path in value:
                    suffix = Path(path).suffix.lower()
                    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
                        with Image.open(path) as image:
                            return image.convert("RGBA")
        except (OSError, ValueError):
            pass
        return None

    def show_start_screen(self) -> None:
        self._editor_active = False
        self.editor_root.pack_forget()
        self.status_frame.pack_forget()
        self.config(menu="")
        self.minsize(820, 560)
        self.title("PhotoRedactor")
        self.restore_centered_window(1060, 700)
        if self.startup_frame is not None and self.startup_frame.winfo_exists():
            self.startup_frame.destroy()

        clipboard_image = self.read_clipboard_image()
        self._startup_clipboard_image = clipboard_image
        frame = tk.Frame(self, background=TOKENS.SURFACE)
        frame.pack(fill=tk.BOTH, expand=True)
        self.startup_frame = frame

        header = tk.Frame(frame, background=TOKENS.BACKGROUND, height=82)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text="PhotoRedactor", font=("Segoe UI Semibold", 22), foreground=TOKENS.TEXT_PRIMARY, background=TOKENS.BACKGROUND).pack(anchor=tk.W, padx=36, pady=(14, 0))
        tk.Label(header, text="Стартовый экран", font=("Segoe UI", 9), foreground=TOKENS.TEXT_SECONDARY, background=TOKENS.BACKGROUND).pack(anchor=tk.W, padx=38)

        content = tk.Frame(frame, background=TOKENS.SURFACE)
        content.pack(fill=tk.BOTH, expand=True, padx=36, pady=26)
        content.columnconfigure(0, minsize=300)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(1, weight=1)

        tk.Label(content, text="Начать", font=("Segoe UI Semibold", 14), foreground=TOKENS.TEXT_PRIMARY, background=TOKENS.SURFACE).grid(row=0, column=0, sticky="w", pady=(0, 10))
        actions = tk.Frame(content, background=TOKENS.SURFACE)
        actions.grid(row=1, column=0, sticky="new", padx=(0, 34))

        new_button = tk.Button(
            actions,
            text="Создать новый холст",
            command=self.new_document,
            anchor="w",
            font=("Segoe UI Semibold", 11),
            background=TOKENS.ACCENT,
            foreground="white",
            activebackground=TOKENS.ACCENT_HOVER,
            activeforeground="white",
            relief=tk.FLAT,
            padx=18,
            pady=13,
            cursor="hand2",
        )
        new_button.pack(fill=tk.X, pady=(0, 8))
        open_button = tk.Button(
            actions,
            text="Открыть изображение или проект",
            command=self.open_file,
            anchor="w",
            font=("Segoe UI", 10),
            background=TOKENS.SURFACE_HOVER,
            foreground=TOKENS.TEXT_PRIMARY,
            activebackground=TOKENS.SURFACE_SELECTED,
            activeforeground=TOKENS.TEXT_PRIMARY,
            relief=tk.FLAT,
            padx=18,
            pady=12,
            cursor="hand2",
        )
        open_button.pack(fill=tk.X, pady=(0, 8))
        ToolTip(new_button, "Открывает выбор размера, формата и фона нового холста.")
        ToolTip(open_button, "Открывает PNG, JPEG, WebP, BMP, TIFF и проекты PhotoRedactor.")

        if clipboard_image is not None:
            clipboard_box = tk.Frame(actions, background=TOKENS.SURFACE_HOVER, padx=12, pady=12)
            clipboard_box.pack(fill=tk.X, pady=(12, 8))
            preview = clipboard_image.copy()
            preview.thumbnail((72, 72), Image.Resampling.LANCZOS)
            preview_canvas = Image.new("RGBA", (72, 72), (228, 230, 232, 255))
            preview_canvas.alpha_composite(preview, ((72 - preview.width) // 2, (72 - preview.height) // 2))
            self._startup_clipboard_preview = ImageTk.PhotoImage(preview_canvas)
            tk.Label(clipboard_box, image=self._startup_clipboard_preview, background=TOKENS.SURFACE_HOVER).pack(side=tk.LEFT, padx=(0, 12))
            clipboard_text = tk.Frame(clipboard_box, background=TOKENS.SURFACE_HOVER)
            clipboard_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            tk.Label(clipboard_text, text="Изображение в буфере", font=("Segoe UI Semibold", 10), foreground=TOKENS.TEXT_PRIMARY, background=TOKENS.SURFACE_HOVER).pack(anchor=tk.W)
            tk.Label(clipboard_text, text=f"{clipboard_image.width} x {clipboard_image.height} px", foreground=TOKENS.TEXT_SECONDARY, background=TOKENS.SURFACE_HOVER).pack(anchor=tk.W)
            paste_button = ttk.Button(clipboard_text, text="Открыть как новый холст", command=self.create_from_clipboard)
            paste_button.pack(anchor=tk.W, pady=(7, 0))
            ToolTip(paste_button, "Создает холст точно по размеру изображения в буфере.")

        if self.recovery_path.exists():
            ttk.Button(actions, text="Восстановить последнюю сессию", command=self.open_recovery).pack(fill=tk.X, pady=(12, 0))

        recent_header = tk.Frame(content, background=TOKENS.SURFACE)
        recent_header.grid(row=0, column=1, sticky="ew", pady=(0, 12))
        tk.Label(recent_header, text="Недавние файлы", font=("Segoe UI Semibold", 14), foreground=TOKENS.TEXT_PRIMARY, background=TOKENS.SURFACE).pack(side=tk.LEFT)
        ttk.Button(recent_header, text="Очистить", command=self.clear_recent_files).pack(side=tk.RIGHT)

        recent_area = tk.Frame(content, background=TOKENS.SURFACE)
        recent_area.grid(row=1, column=1, sticky="nsew")
        recent_area.columnconfigure(0, weight=1)
        recent_area.rowconfigure(1, weight=1)
        tk.Label(recent_area, text="Имя файла  |  Тип  |  Изменен", anchor="w", padx=12, pady=8, foreground=TOKENS.TEXT_SECONDARY, background=TOKENS.SURFACE_HOVER).grid(row=0, column=0, sticky="ew")
        self.startup_recent_list = tk.Listbox(
            recent_area,
            exportselection=False,
            activestyle="none",
            background=TOKENS.SURFACE,
            foreground=TOKENS.TEXT_PRIMARY,
            selectbackground=TOKENS.SURFACE_SELECTED,
            selectforeground=TOKENS.TEXT_PRIMARY,
            highlightthickness=0,
            borderwidth=0,
            font=("Segoe UI", 10),
        )
        recent_scroll = ttk.Scrollbar(recent_area, orient=tk.VERTICAL, command=self.startup_recent_list.yview)
        self.startup_recent_list.configure(yscrollcommand=recent_scroll.set)
        self.startup_recent_list.grid(row=1, column=0, sticky="nsew")
        recent_scroll.grid(row=1, column=1, sticky="ns")
        self.startup_recent_list.bind("<Double-Button-1>", lambda _event: self.open_startup_recent())
        self.startup_recent_list.bind("<Return>", lambda _event: self.open_startup_recent())
        ttk.Button(recent_area, text="Открыть выбранный", command=self.open_startup_recent).grid(row=2, column=0, sticky="e", pady=(10, 0))
        self.refresh_startup_recent()

    def restore_centered_window(self, preferred_width: int, preferred_height: int) -> None:
        try:
            self.state("normal")
        except tk.TclError:
            try:
                self.attributes("-zoomed", False)
            except tk.TclError:
                pass
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        width = min(preferred_width, max(820, screen_width - 120))
        height = min(preferred_height, max(560, screen_height - 120))
        x = max(0, (screen_width - width) // 2)
        y = max(0, (screen_height - height) // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    @staticmethod
    def center_toplevel(window: tk.Toplevel, preferred_width: int, preferred_height: int) -> None:
        window.update_idletasks()
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        width = min(max(preferred_width, window.winfo_reqwidth()), max(640, screen_width - 80))
        height = min(max(preferred_height, window.winfo_reqheight()), max(520, screen_height - 80))
        x = max(0, (screen_width - width) // 2)
        y = max(0, (screen_height - height) // 2)
        window.geometry(f"{width}x{height}+{x}+{y}")
        window.update_idletasks()

    def refresh_startup_recent(self) -> None:
        if not hasattr(self, "startup_recent_list") or not self.startup_recent_list.winfo_exists():
            return
        recent_list = self.startup_recent_list
        recent_list.delete(0, tk.END)
        self._startup_recent_paths: list[str] = []
        for path in self.recent_files:
            item = Path(path)
            if not item.exists():
                continue
            kind = "Проект" if item.suffix.lower() == ".prdx" else "Изображение"
            modified = time.strftime("%d.%m.%Y %H:%M", time.localtime(item.stat().st_mtime))
            recent_list.insert(tk.END, f"  {item.name}  |  {kind}  |  {modified}")
            self._startup_recent_paths.append(path)
        if self._startup_recent_paths:
            recent_list.selection_set(0)
        else:
            recent_list.insert(tk.END, "  Недавних файлов пока нет")

    def open_startup_recent(self) -> None:
        selection = self.startup_recent_list.curselection() if hasattr(self, "startup_recent_list") else ()
        if not selection:
            return
        try:
            path = self._startup_recent_paths[int(selection[0])]
        except IndexError:
            return
        self.open_path(path)

    def show_editor(self) -> None:
        if self.startup_frame is not None and self.startup_frame.winfo_exists():
            self.startup_frame.destroy()
        self.startup_frame = None
        self.config(menu=self.editor_menu)
        self.geometry("1440x920")
        self.minsize(1000, 640)
        document_name = Path(self.doc.path).name if self.doc.path else "Новый документ"
        self.title(f"{document_name} - PhotoRedactor")
        self.editor_root.pack(fill=tk.BOTH, expand=True)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self._editor_active = True
        self.refresh_recent_menu()
        self.refresh()
        self.maximize_window()
        if self._initial_fit_after_id is not None:
            self.after_cancel(self._initial_fit_after_id)
        self._initial_fit_after_id = self.after(80, self.finish_initial_fit)

    def finish_initial_fit(self) -> None:
        self._initial_fit_after_id = None
        if self._editor_active and self.winfo_exists():
            self.update_idletasks()
            self._performing_initial_fit = True
            try:
                self.fit_to_screen()
            finally:
                self._performing_initial_fit = False
            self._initial_fit_after_id = self.after(160, self.finish_initial_center)

    def finish_initial_center(self) -> None:
        self._initial_fit_after_id = None
        if self._editor_active and self.winfo_exists():
            self.update_idletasks()
            self.center_canvas_on_doc(self.doc.width / 2, self.doc.height / 2)

    def maximize_window(self) -> None:
        try:
            self.state("zoomed")
        except tk.TclError:
            try:
                self.attributes("-zoomed", True)
            except tk.TclError:
                pass

    def _build_ui(self) -> None:
        self._build_menu()
        self.editor_root = ttk.Frame(self, style="App.TFrame")
        self.editor_root.pack(fill=tk.BOTH, expand=True)
        options_bar = ttk.Frame(self.editor_root, style="Topbar.TFrame", height=46)
        options_bar.pack(fill=tk.X)
        options_bar.pack_propagate(False)
        self._build_tool_options(options_bar)
        ttk.Separator(self.editor_root).pack(fill=tk.X)
        root = ttk.PanedWindow(self.editor_root, orient=tk.HORIZONTAL)
        root.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(root, width=188, style="Panel.TFrame")
        center = ttk.Frame(root, style="Workspace.TFrame")
        right = ttk.Frame(root, width=292, style="Panel.TFrame")
        root.add(left, weight=0)
        root.add(center, weight=1)
        root.add(right, weight=0)

        self._build_tools(left)
        self._build_canvas(center)
        self._build_panels(right)

        self.status_frame = ttk.Frame(self, style="Status.TFrame")
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status = ttk.Label(self.status_frame, text="", anchor=tk.W, style="Status.TLabel")
        self.status.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.status_coords = ttk.Label(self.status_frame, text="", style="Status.TLabel", width=13, anchor=tk.E)
        self.status_coords.pack(side=tk.RIGHT)
        self.status_zoom = ttk.Label(self.status_frame, text="100%", style="Status.TLabel", width=7, anchor=tk.E)
        self.status_zoom.pack(side=tk.RIGHT)
        self.zoom_label = self.status_zoom
        self.status_size = ttk.Label(self.status_frame, text="", style="Status.TLabel", width=14, anchor=tk.E)
        self.status_size.pack(side=tk.RIGHT)
        self._build_shortcuts()
        self.bind_all("<KeyPress-space>", self.space_down)
        self.bind_all("<KeyRelease-space>", self.space_up)
        for sequence in ("<Left>", "<Right>", "<Up>", "<Down>", "<Shift-Left>", "<Shift-Right>", "<Shift-Up>", "<Shift-Down>"):
            self.bind_all(sequence, self.nudge_selected_object)

    def _build_shortcuts(self) -> None:
        bindings = {
            "<Delete>": self.shortcut_delete,
            "<Escape>": self.cancel_incomplete_interaction,
            "<Return>": self.shortcut_enter,
        }
        for sequence, callback in bindings.items():
            self.bind_all(sequence, callback)
        self.bind_all("<Control-KeyPress>", self.shortcut_control_key)
        self.bind_all("<KeyPress>", self.shortcut_plain_key)

    def shortcut_control_key(self, event):
        callbacks = {
            "undo": self.shortcut_undo, "redo": self.shortcut_redo, "save": self.shortcut_save,
            "save_as": self.shortcut_save_as, "open": self.shortcut_open, "new_document": self.shortcut_new,
            "select_all": self.shortcut_select_all, "deselect": self.shortcut_deselect,
            "invert_selection": self.shortcut_invert_selection, "copy": self.shortcut_copy,
            "cut": self.shortcut_cut, "paste": self.shortcut_paste, "new_layer": self.shortcut_new_layer,
            "duplicate_layer": self.shortcut_duplicate_layer, "merge_down": self.shortcut_merge_down,
            "flatten": self.shortcut_flatten, "free_transform": self.shortcut_free_transform,
            "fit_to_screen": self.shortcut_fit_to_screen, "actual_size": self.shortcut_actual_size,
        }
        callback = callbacks.get(command_for_event(event))
        return callback(event) if callback is not None else None

    def shortcut_plain_key(self, event):
        if int(getattr(event, "state", 0)) & 0x0004:
            return None
        key = event_key(event)
        if key == "x":
            return self.shortcut_swap_colors(event)
        if key == "d":
            return self.shortcut_reset_colors(event)
        if key == "+":
            return self.shortcut_zoom_in(event)
        if key == "-":
            return self.shortcut_zoom_out(event)
        if key in {"[", "]"}:
            return self.shortcut_brush_size(1 if key == "]" else -1)
        tools = TOOL_SHORTCUT_GROUPS.get(key)
        if tools and self.shortcut_context() == "canvas" and self._editor_active:
            current = self.tool.get()
            target = tools[(tools.index(current) + 1) % len(tools)] if current in tools else tools[0]
            return self.shortcut_tool(target)
        return None

    def shortcut_tool(self, tool: str):
        if self.shortcut_context() == "canvas" and self._editor_active:
            self.select_tool(tool)
            return "break"
        return None

    @staticmethod
    def _widget_is_descendant(widget, parent) -> bool:
        current = widget
        while current is not None:
            if current is parent:
                return True
            current = getattr(current, "master", None)
        return False

    def shortcut_context(self) -> str:
        focus = self.focus_get()
        if focus is not None and isinstance(focus, (tk.Text, tk.Entry, ttk.Entry, ttk.Spinbox)):
            return "text"
        if hasattr(self, "layer_list") and focus is not None and self._widget_is_descendant(focus, self.layer_list):
            return "layers"
        grabbed = self.grab_current()
        if grabbed is not None and grabbed is not self:
            return "modal"
        return "canvas"

    def shortcut_undo(self, _event=None):
        if self.shortcut_context() == "text":
            return None
        if self._editor_active:
            self.undo()
        return "break"

    def shortcut_redo(self, _event=None):
        if self.shortcut_context() == "text":
            return None
        if self._editor_active:
            self.redo()
        return "break"

    def shortcut_save(self, _event=None):
        if self._editor_active:
            self.save()
        return "break"

    def shortcut_save_as(self, _event=None):
        if self._editor_active:
            self.save_as_project()
        return "break"

    def shortcut_open(self, _event=None):
        self.open_file()
        return "break"

    def shortcut_new(self, _event=None):
        self.new_document()
        return "break"

    def shortcut_select_all(self, _event=None):
        context = self.shortcut_context()
        if context == "text":
            focus = self.focus_get()
            if isinstance(focus, tk.Text):
                focus.tag_add(tk.SEL, "1.0", "end-1c")
                focus.mark_set(tk.INSERT, "1.0")
            elif focus is not None:
                focus.selection_range(0, tk.END)
            return "break"
        if context == "layers":
            self.layer_list.selection_set(0, tk.END)
            self.layer_selected(None)
            return "break"
        if context == "canvas" and self._editor_active:
            self.select_all()
            return "break"
        return None

    def shortcut_deselect(self, _event=None):
        if self.shortcut_context() == "canvas" and self._editor_active:
            self.clear_selection()
            return "break"
        return None

    def shortcut_invert_selection(self, _event=None):
        if self.shortcut_context() == "canvas" and self._editor_active:
            self.invert_selection()
            return "break"
        return None

    def shortcut_delete(self, _event=None):
        context = self.shortcut_context()
        if context == "text":
            return None
        if context == "layers":
            self.delete_layer()
        elif context == "canvas":
            self.delete_selected_pixels()
        return "break"

    def shortcut_copy(self, _event=None):
        if self.shortcut_context() == "text":
            return None
        self.copy_pixels()
        return "break"

    def shortcut_cut(self, _event=None):
        if self.shortcut_context() == "text":
            return None
        self.copy_pixels()
        self.delete_selected_pixels()
        return "break"

    def shortcut_paste(self, _event=None):
        if self.shortcut_context() == "text":
            return None
        self.paste_pixels()
        return "break"

    def shortcut_enter(self, _event=None):
        if self.shortcut_context() == "text":
            return None
        if self.tool.get() == "crop":
            self.apply_crop_overlay()
            return "break"
        return None

    def shortcut_zoom_in(self, _event=None):
        if self._editor_active and self.shortcut_context() != "text":
            self.set_zoom(self.zoom.get() * 1.25)
            return "break"
        return None

    def shortcut_zoom_out(self, _event=None):
        if self._editor_active and self.shortcut_context() != "text":
            self.set_zoom(self.zoom.get() / 1.25)
            return "break"
        return None

    def shortcut_brush_size(self, direction: int):
        sized_tools = {"brush", "eraser", "blur_tool", "sharpen_tool", "dodge", "burn", "clone", "healing", "spot_healing", "quick_selection"}
        if self.shortcut_context() != "canvas" or not self._editor_active or self.tool.get() not in sized_tools:
            return None
        current = int(self.brush_size.get())
        step = max(1, current // 10)
        updated = max(1, min(220, current + step * int(direction)))
        self.brush_size.set(updated)
        self.status_text(f"Размер инструмента: {updated} px")
        return "break"

    def shortcut_new_layer(self, _event=None):
        if self.shortcut_context() != "text" and self._editor_active:
            self.new_layer()
            return "break"
        return None

    def shortcut_duplicate_layer(self, _event=None):
        if self.shortcut_context() != "text" and self._editor_active:
            self.duplicate_layer()
            return "break"
        return None

    def shortcut_merge_down(self, _event=None):
        if self.shortcut_context() != "text" and self._editor_active:
            self.merge_down()
            return "break"
        return None

    def shortcut_flatten(self, _event=None):
        if self.shortcut_context() != "text" and self._editor_active:
            self.flatten()
            return "break"
        return None

    def shortcut_free_transform(self, _event=None):
        if self.shortcut_context() == "canvas" and self._editor_active:
            self.free_transform_layer()
            return "break"
        return None

    def shortcut_fit_to_screen(self, _event=None):
        if self.shortcut_context() == "canvas" and self._editor_active:
            self.fit_to_screen()
            return "break"
        return None

    def shortcut_actual_size(self, _event=None):
        if self.shortcut_context() == "canvas" and self._editor_active:
            self.set_zoom(1.0)
            return "break"
        return None

    def nudge_selected_object(self, event):
        if self.shortcut_context() != "canvas" or self.tool.get() != "move" or self.doc.layer.id not in self.selected_layer_ids:
            return None
        layer = self.doc.layer
        if layer.kind not in {"shape", "text"} or layer.locked:
            return None
        step = 10 if event.state & 0x0001 else 1
        dx = -step if event.keysym == "Left" else step if event.keysym == "Right" else 0
        dy = -step if event.keysym == "Up" else step if event.keysym == "Down" else 0
        before = (layer.x, layer.y)
        layer.x += dx
        layer.y += dy
        self.doc.dirty = True
        self.push_command(LayerMoveCommand("Сдвинуть объект", layer.id, before, (layer.x, layer.y)))
        self.refresh()
        return "break"

    def shortcut_swap_colors(self, _event=None):
        if self.shortcut_context() == "canvas":
            self.swap_colors()
            return "break"
        return None

    def shortcut_reset_colors(self, _event=None):
        if self.shortcut_context() == "canvas":
            self.reset_colors()
            return "break"
        return None

    def cancel_incomplete_interaction(self, _event=None) -> str:
        if self._text_editor is not None:
            self.cancel_text_edit()
            self.status_text("Редактирование текста отменено")
            return "break"
        if hasattr(self, "canvas"):
            self.clear_drag_preview()
            self.clear_lasso_overlay()
            self.clear_quick_selection_preview()
            self.clear_gradient_preview()
        self.drag_start = None
        self.last_point = None
        self._shape_drag_options = None
        self._crop_box = None
        self._crop_drag_handle = None
        self._crop_drag_origin_box = None
        self._clone_anchor_target = None
        self._clone_anchor_source = None
        if hasattr(self, "canvas"):
            self.update_selection_overlay()
        return "break"

    def _build_menu(self) -> None:
        menu = tk.Menu(self)
        self.editor_menu = menu
        self.config(menu=menu)

        file_menu = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Файл", menu=file_menu)
        self.file_menu = file_menu
        file_menu.add_command(label="Новый холст", command=self.new_document, accelerator=accelerator("new_document"))
        file_menu.add_command(label="Открыть изображение/проект", command=self.open_file, accelerator=accelerator("open"))
        self.recent_menu = tk.Menu(file_menu, tearoff=False)
        file_menu.add_cascade(label="Недавние файлы", menu=self.recent_menu)
        file_menu.add_command(label="Поместить встроенное", command=self.place_embedded)
        file_menu.add_command(label="Поместить связанное", command=self.place_linked)
        file_menu.add_command(label="Загрузить файлы как слои", command=self.load_files_as_layers)
        file_menu.add_separator()
        file_menu.add_command(label="Сохранить проект", command=self.save, accelerator=accelerator("save"))
        file_menu.add_command(label="Сохранить проект как", command=self.save_as_project, accelerator=accelerator("save_as"))
        file_menu.add_command(label="Экспорт изображения", command=self.export_image)
        file_menu.add_command(label="Экспорт слоев", command=self.export_layers)
        file_menu.add_separator()
        file_menu.add_command(label="Открыть восстановление", command=self.open_recovery)
        file_menu.add_command(label="Очистить восстановление", command=self.clear_recovery)
        file_menu.add_separator()
        file_menu.add_command(label="Пакетный размер/конвертация", command=self.batch_process)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.destroy)

        edit = tk.Menu(menu, tearoff=False)
        self.edit_menu = edit
        menu.add_cascade(label="Правка", menu=edit)
        edit.add_command(label="Отменить", command=self.undo, accelerator=accelerator("undo"))
        edit.add_command(label="Повторить", command=self.redo, accelerator=accelerator("redo"))
        edit.add_separator()
        edit.add_command(label="Вырезать", command=self.shortcut_cut, accelerator=accelerator("cut"))
        edit.add_command(label="Копировать", command=self.shortcut_copy, accelerator=accelerator("copy"))
        edit.add_command(label="Вставить", command=self.shortcut_paste, accelerator=accelerator("paste"))
        edit.add_command(label="Удалить выбранные пиксели", command=self.shortcut_delete, accelerator="Delete")
        edit.add_separator()
        edit.add_command(label="Снять выделение", command=self.clear_selection, accelerator=accelerator("deselect"))

        self.tools_menu = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Инструменты", menu=self.tools_menu)
        self.refresh_tool_menu()

        select = tk.Menu(menu, tearoff=False)
        self.select_menu = select
        menu.add_cascade(label="Выделение", menu=select)
        select.add_command(label="Выделить все", command=self.select_all, accelerator=accelerator("select_all"))
        select.add_command(label="Инвертировать выделение", command=self.invert_selection, accelerator=accelerator("invert_selection"))
        select.add_command(label="Снять выделение", command=self.clear_selection, accelerator=accelerator("deselect"))
        select.add_separator()
        select.add_command(label="Выделить непрозрачные пиксели", command=self.select_opaque_pixels)
        select.add_command(label="Выделить объект", command=self.select_subject)
        select.add_command(label="Выделить фон", command=self.select_background)
        select.add_command(label="Выделить небо", command=self.select_sky)
        select.add_command(label="Автоматическое выделение...", command=self.automatic_selection_workspace)
        select.add_command(label="Одна строка", command=self.single_row_selection)
        select.add_command(label="Один столбец", command=self.single_column_selection)
        select.add_separator()
        select.add_command(label="Растушевка", command=self.feather_selection)
        select.add_command(label="Сгладить", command=self.smooth_selection)
        select.add_command(label="Расширить", command=self.grow_selection)
        select.add_command(label="Сжать", command=self.shrink_selection)
        select.add_command(label="Граница", command=self.border_selection)
        select.add_command(label="Уточнить край", command=self.refine_selection)
        select.add_command(label="Умная очистка края", command=self.cleanup_selection_edges)
        select.add_command(label="Коррекция края по уверенности", command=self.correct_selection_edges)
        select.add_command(label="Выделить и маска", command=self.select_and_mask_workspace)
        select.add_separator()
        select.add_command(label="Сохранить выделение", command=self.save_selection)
        select.add_command(label="Загрузить выделение", command=self.load_selection)
        select.add_command(label="Удалить сохраненное выделение", command=self.delete_saved_selection)

        image = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Изображение", menu=image)
        image.add_command(label="Размер изображения", command=self.resize_image)
        image.add_command(label="Размер холста", command=self.resize_canvas)
        image.add_command(label="Генеративное расширение холста", command=self.generative_expand_dialog)
        image.add_command(label="Обрезать по выделению", command=self.crop_to_selection)
        image.add_command(label="Обрезать прозрачные пиксели", command=self.trim_transparent)
        image.add_command(label="Показать все слои", command=self.reveal_all)
        image.add_separator()
        image.add_command(label="Повернуть на 90 по часовой", command=lambda: self.rotate(90))
        image.add_command(label="Повернуть на 180", command=lambda: self.rotate(180))
        image.add_command(label="Отразить горизонтально", command=lambda: self.flip(horizontal=True))
        image.add_command(label="Отразить вертикально", command=lambda: self.flip(horizontal=False))

        color_menu = tk.Menu(image, tearoff=False)
        image.add_separator()
        image.add_cascade(label="Управление цветом", menu=color_menu)
        color_menu.add_command(label="Назначить ICC-профиль", command=self.assign_icc_profile)
        color_menu.add_command(label="Преобразовать в ICC-профиль", command=self.convert_icc_profile)
        model_menu = tk.Menu(color_menu, tearoff=False)
        color_menu.add_cascade(label="Цветовая модель", menu=model_menu)
        for model in COLOR_MODELS:
            model_menu.add_command(label=model, command=lambda value=model: self.change_color_model(value))
        depth_menu = tk.Menu(color_menu, tearoff=False)
        color_menu.add_cascade(label="Глубина каналов", menu=depth_menu)
        for depth in BIT_DEPTHS:
            depth_menu.add_command(label=f"{depth} бит", command=lambda value=depth: self.change_bit_depth(value))

        layer = tk.Menu(menu, tearoff=False)
        self.layer_menu = layer
        menu.add_cascade(label="Слой", menu=layer)
        layer.add_command(label="Новый слой", command=self.new_layer, accelerator=accelerator("new_layer"))
        layer.add_command(label="Дублировать слой", command=self.duplicate_layer, accelerator=accelerator("duplicate_layer"))
        layer.add_command(label="Удалить слой", command=self.delete_layer)
        layer.add_command(label="Переименовать слой", command=self.rename_layer)
        layer.add_command(label="Заблокировать/разблокировать", command=self.toggle_layer_lock)
        layer.add_command(label="Редактировать текстовый слой", command=self.edit_text_layer)
        layer.add_command(label="Редактировать контур текста", command=self.edit_text_path)
        layer.add_command(label="Трансформировать текстовый блок", command=self.transform_text_box)
        layer.add_command(label="Редактировать фигуру", command=self.edit_shape_layer)
        layer.add_command(label="Редактировать точки Безье", command=self.edit_bezier_points)
        layer.add_command(label="Булева операция фигур", command=self.boolean_shape_layers)
        layer.add_command(label="\u0420\u0435\u0434\u0430\u043a\u0442\u0438\u0440\u043e\u0432\u0430\u0442\u044c \u043a\u043e\u0440\u0440\u0435\u043a\u0442\u0438\u0440\u0443\u044e\u0449\u0438\u0439 \u0441\u043b\u043e\u0439", command=self.edit_adjustment_layer)
        layer.add_separator()
        layer.add_command(label="Поднять выше", command=lambda: self.move_layer(1))
        layer.add_command(label="Опустить ниже", command=lambda: self.move_layer(-1))
        layer.add_command(label="Свободная трансформация", command=self.free_transform_layer, accelerator=accelerator("free_transform"))
        layer.add_command(label="Трансформировать выделенные пиксели", command=self.transform_selected_pixels)
        layer.add_command(label="Перспективная трансформация", command=self.perspective_transform_layer)
        layer.add_command(label="Деформация слоя", command=self.warp_layer)
        layer.add_command(label="Обновить связанный слой", command=self.update_linked_layer)
        layer.add_command(label="Перелинковать слой", command=self.relink_layer)
        smart_menu = tk.Menu(layer, tearoff=False)
        layer.add_cascade(label="Smart Object", menu=smart_menu)
        smart_menu.add_command(label="Показать статус связи", command=self.show_linked_layer_status)
        smart_menu.add_command(label="Заменить содержимое", command=self.replace_smart_contents)
        smart_menu.add_command(label="Преобразовать во встроенный", command=self.convert_smart_to_embedded)
        smart_menu.add_command(label="Сбросить трансформацию", command=self.reset_smart_transform)
        layer.add_command(label="Переключить обтравочную маску", command=self.toggle_clipping_mask)
        layer.add_command(label="Стили слоя", command=self.edit_layer_styles)
        layer.add_command(label="Фильтры слоя", command=self.edit_layer_filters)
        layer.add_command(label="Очистить фильтры слоя", command=self.clear_layer_filters)
        layer.add_command(label="Объединить с нижним", command=self.merge_down, accelerator=accelerator("merge_down"))
        layer.add_command(label="Свести изображение", command=self.flatten, accelerator=accelerator("flatten"))
        layer.add_separator()
        layer.add_command(label="Редактировать маску как канал", command=self.edit_active_mask_channel)
        layer.add_command(label="Добавить маску из выделения", command=self.add_mask_from_selection)
        layer.add_command(label="Добавить белую маску", command=self.add_reveal_all_mask)
        layer.add_command(label="Добавить черную маску", command=self.add_hide_all_mask)
        layer.add_command(label="Инвертировать маску", command=self.invert_layer_mask)
        layer.add_command(label="Включить/выключить маску", command=self.toggle_layer_mask)
        layer.add_command(label="Связать/отвязать маску", command=self.toggle_layer_mask_link)
        layer.add_command(label="Плотность маски", command=self.set_mask_density)
        layer.add_command(label="Растушевка маски", command=self.set_mask_feather)
        layer.add_command(label="Уточнить край маски", command=self.refine_layer_mask)
        layer.add_command(label="Применить маску", command=self.apply_layer_mask)
        layer.add_command(label="Удалить маску", command=self.delete_layer_mask)

        adj = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Коррекция", menu=adj)
        adj.add_command(label="Яркость/контраст", command=self.adjust_brightness_contrast)
        adj.add_command(label="Насыщенность", command=self.adjust_saturation)
        adj.add_command(label="Тон/Насыщенность", command=self.adjust_hue_saturation)
        adj.add_command(label="Экспозиция", command=self.adjust_exposure)
        adj.add_command(label="Цветовой баланс", command=self.adjust_color_balance)
        adj.add_command(label="Уровни", command=self.adjust_levels)
        adj.add_command(label="Кривые", command=self.adjust_curves)
        adj.add_command(label="Порог", command=self.adjust_threshold)
        adj.add_command(label="Постеризация", command=self.adjust_posterize)
        adj.add_command(label="Инверсия", command=self.adjust_invert)
        adj.add_command(label="Черно-белое", command=self.adjust_grayscale)
        adj.add_separator()
        adj.add_command(label="Добавить корректирующий слой", command=self.add_adjustment_layer)
        adj.add_command(label="\u0420\u0435\u0434\u0430\u043a\u0442\u0438\u0440\u043e\u0432\u0430\u0442\u044c \u043a\u043e\u0440\u0440\u0435\u043a\u0442\u0438\u0440\u0443\u044e\u0449\u0438\u0439 \u0441\u043b\u043e\u0439", command=self.edit_adjustment_layer)

        filters = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Фильтр", menu=filters)
        filters.add_command(label="Размытие по Гауссу", command=self.filter_blur)
        filters.add_command(label="Резкость", command=self.filter_sharpen)
        filters.add_command(label="Шум", command=self.filter_noise)
        filters.add_separator()
        filters.add_command(label="Заливка с учетом содержимого", command=self.filter_content_aware_fill)
        filters.add_command(label="Очистка краев выделения", command=self.filter_edge_cleanup)
        filters.add_command(label="Удаление красных глаз", command=self.filter_red_eye)
        filters.add_command(label="Заплатка из источника", command=self.filter_patch_selection)
        self.plugin_filters_menu = tk.Menu(filters, tearoff=False, postcommand=self.refresh_plugin_filter_menu)
        filters.add_separator()
        filters.add_cascade(label="Плагины", menu=self.plugin_filters_menu)

        retouch = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Ретушь", menu=retouch)
        retouch.add_command(label="Частотное разложение", command=self.frequency_separation_layers)
        retouch.add_command(label="Портретная обработка", command=self.portrait_cleanup_layer)
        retouch.add_separator()
        retouch.add_command(label="Выбрать точечное восстановление", command=lambda: self.tool.set("spot_healing"))
        retouch.add_command(label="Удаление красных глаз", command=self.filter_red_eye)
        retouch.add_command(label="Заплатка из источника", command=self.filter_patch_selection)

        analysis = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Анализ", menu=analysis)
        analysis.add_command(label="Статистика изображения", command=self.show_image_statistics)
        analysis.add_command(label="Гистограмма", command=self.show_histogram)
        analysis.add_command(label="Метаданные / EXIF", command=self.show_metadata)
        analysis.add_command(label="Редактировать метаданные", command=self.edit_metadata)
        analysis.add_command(label="Состояние кэша и GPU", command=self.show_cache_status)

        actions = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Действия", menu=actions)
        actions.add_command(label="Начать запись", command=self.start_action_recording)
        actions.add_command(label="Остановить запись", command=self.stop_action_recording)
        actions.add_command(label="Сохранить запись", command=self.save_action_recording)
        actions.add_command(label="Очистить запись", command=self.clear_action_recording)
        actions.add_separator()
        actions.add_command(label="Выполнить действие...", command=self.run_action_file)
        actions.add_command(label="Пакетно выполнить действие...", command=self.batch_action_file)
        actions.add_separator()
        actions.add_command(label="Перезагрузить плагины", command=self.reload_plugins)
        actions.add_command(label="Ошибки плагинов", command=self.show_plugin_errors)

        view = tk.Menu(menu, tearoff=False)
        self.view_menu = view
        menu.add_cascade(label="Вид", menu=view)
        view.add_command(label="Увеличить", command=lambda: self.set_zoom(self.zoom.get() * 1.25), accelerator="+")
        view.add_command(label="Уменьшить", command=lambda: self.set_zoom(self.zoom.get() / 1.25), accelerator="-")
        view.add_command(label="100%", command=lambda: self.set_zoom(1.0), accelerator=accelerator("actual_size"))
        view.add_command(label="По размеру окна", command=self.fit_to_screen, accelerator=accelerator("fit_to_screen"))
        view.add_separator()
        channel = tk.Menu(view, tearoff=False)
        view.add_cascade(label="Канал", menu=channel)
        for label, name in [("RGB", "RGB"), ("Красный", "Red"), ("Зеленый", "Green"), ("Синий", "Blue"), ("Альфа", "Alpha")]:
            channel.add_radiobutton(label=label, value=name, variable=self.view_channel, command=self.set_view_channel)
        mask_view = tk.Menu(view, tearoff=False)
        view.add_cascade(label="Просмотр маски", menu=mask_view)
        for label in MASK_PREVIEW_MODES:
            mask_view.add_radiobutton(label=label, value=label, variable=self.mask_preview, command=self.set_mask_preview)
        view.add_separator()
        view.add_checkbutton(label="Сетка", variable=self.grid_visible, command=self.refresh_canvas)
        view.add_command(label="Шаг сетки", command=self.set_grid_spacing)
        view.add_command(label="Добавить горизонтальную направляющую", command=self.add_horizontal_guide)
        view.add_command(label="Добавить вертикальную направляющую", command=self.add_vertical_guide)
        view.add_command(label="Очистить направляющие", command=self.clear_guides)

        view.add_separator()
        view.add_command(label="Настроить панель инструментов...", command=self.configure_tool_palette)

    def _build_tools(self, parent: ttk.Frame) -> None:
        parent.configure(width=188)
        parent.pack_propagate(False)
        self.tool_palette = ToolPalette(
            parent,
            definitions=TOOL_DEFINITIONS,
            tool_var=self.tool,
            order=self.tool_order,
            visible=self.visible_tools,
            select_tool=self.select_tool,
            configure_tools=self.configure_tool_palette,
            tooltip_factory=ToolTip,
        )
        self.tool_palette.pack(fill=tk.BOTH, expand=True)

    def _build_tool_options(self, parent: ttk.Frame) -> None:
        self.tool_options_panel = ToolOptionsPanel(
            parent,
            tool_var=self.tool,
            definitions=TOOL_DEFINITIONS,
            brush_size=self.brush_size,
            opacity=self.opacity,
            hardness=self.hardness,
            retouch_strength=self.retouch_strength,
            exposure=self.exposure,
            tonal_range=self.tonal_range,
            tolerance=self.tolerance,
            color_range_sample_hex=self.color_range_sample_hex,
            selection_mode=self.selection_mode,
            quick_smooth=self.quick_smooth,
            quick_edge_radius=self.quick_edge_radius,
            quick_edge_strength=self.quick_edge_strength,
            paint_target=self.paint_target,
            retouch_preset=self.retouch_preset,
            retouch_presets=RETOUCH_PRESETS,
            pick_foreground=self.pick_foreground,
            pick_background=self.pick_background,
            set_paint_target=self.set_paint_target,
            apply_retouch_preset=self.apply_retouch_preset,
            shape_stroke_width=self.shape_stroke_width,
            polygon_sides=self.polygon_sides,
            star_points=self.star_points_count,
            star_inner_ratio=self.star_inner_ratio,
            custom_shape_preset=self.custom_shape_preset,
            custom_shape_presets=list(CUSTOM_SHAPE_PRESETS),
            selection_feather=self.selection_feather,
            selection_antialias=self.selection_antialias,
            magic_contiguous=self.magic_contiguous,
            clone_aligned=self.clone_aligned,
            clone_sampling=self.clone_sampling,
            gradient_type=self.gradient_type,
            gradient_mode=self.gradient_mode,
            gradient_shape=self.gradient_shape,
            gradient_object_fill=self.gradient_object_fill,
            gradient_texture=self.gradient_texture,
            gradient_mid_enabled=self.gradient_mid_enabled,
            gradient_mid_position=self.gradient_mid_position,
            pick_gradient_mid=self.pick_gradient_mid,
            crop_aspect=self.crop_aspect,
            crop_custom_width=self.crop_custom_width,
            crop_custom_height=self.crop_custom_height,
            text_font_family=self.text_font_family,
            text_size=self.text_size,
            text_bold=self.text_bold,
            text_italic=self.text_italic,
            text_underline=self.text_underline,
            text_align=self.text_align,
            text_line_spacing=self.text_line_spacing,
            text_tracking=self.text_tracking,
            text_rotation=self.text_rotation,
            text_box_width=self.text_box_width,
            finish_text_edit=self.finish_text_edit,
            edit_active_text=self.edit_active_text_on_canvas,
            edit_text_path=self.edit_text_path,
            tooltip_factory=ToolTip,
            compact=True,
            auto_select=self.auto_select,
            color_provider=lambda: (self.foreground, self.background),
        )
        self.tool_options_panel.pack(fill=tk.BOTH, expand=True)

    def _build_color_control(self, parent: ttk.Frame) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=(3, 7))
        self.color_control_canvas = tk.Canvas(frame, width=46, height=40, highlightthickness=0, background=TOKENS.SURFACE)
        self.color_control_canvas.pack(anchor=tk.CENTER)
        self.color_control_canvas.bind("<Button-1>", self.color_control_click)
        ToolTip(self.color_control_canvas, "Основной и дополнительный цвета\nX - поменять местами, D - сбросить")
        self.refresh_color_control()

    def refresh_color_control(self) -> None:
        if hasattr(self, "tool_options_panel"):
            self.tool_options_panel.render()
        if not hasattr(self, "color_control_canvas"):
            return
        canvas = self.color_control_canvas
        canvas.delete("all")
        canvas.create_rectangle(19, 14, 43, 37, fill=self.color_hex(self.background), outline=TOKENS.BORDER, width=1, tags="background")
        canvas.create_rectangle(3, 2, 27, 25, fill=self.color_hex(self.foreground), outline=TOKENS.TEXT_SECONDARY, width=1, tags="foreground")

    def color_control_click(self, event) -> None:
        if event.x >= 19 and event.y >= 14:
            self.pick_background()
        else:
            self.pick_foreground()

    def swap_colors(self) -> None:
        self.foreground, self.background = self.background, self.foreground
        self.refresh_color_control()

    def reset_colors(self) -> None:
        self.foreground = (0, 0, 0, 255)
        self.background = (255, 255, 255, 255)
        self.refresh_color_control()

    def select_tool(self, value: str) -> None:
        if value in self.tool_order:
            self.tool.set(value)

    def save_active_tool_settings(self) -> None:
        settings = self.tool_settings.get(self._active_settings_tool)
        if settings is None:
            return
        if "size" in settings:
            settings["size"] = int(self.brush_size.get())
        if "opacity" in settings:
            settings["opacity"] = float(self.opacity.get())
        if "hardness" in settings:
            settings["hardness"] = float(self.hardness.get())
        if "strength" in settings:
            settings["strength"] = float(self.retouch_strength.get())
        if "exposure" in settings:
            settings["exposure"] = float(self.exposure.get())
        if "range" in settings:
            settings["range"] = self.tonal_range.get()

    def load_active_tool_settings(self, tool: str) -> None:
        settings = self.tool_settings.get(tool)
        self._active_settings_tool = tool
        if settings is None:
            return
        if "size" in settings:
            self.brush_size.set(int(settings["size"]))
        if "opacity" in settings:
            self.opacity.set(float(settings["opacity"]))
        if "hardness" in settings:
            self.hardness.set(float(settings["hardness"]))
        if "strength" in settings:
            self.retouch_strength.set(float(settings["strength"]))
        if "exposure" in settings:
            self.exposure.set(float(settings["exposure"]))
        if "range" in settings:
            self.tonal_range.set(str(settings["range"]))

    def tool_changed(self, *_args) -> None:
        new_tool = self.tool.get()
        if self._text_editor is not None and new_tool != "text":
            self.finish_text_edit()
        if new_tool != self._active_settings_tool:
            self.save_active_tool_settings()
            self.load_active_tool_settings(new_tool)
        if new_tool == "text" and self.doc.layer.kind == "text":
            self.load_text_properties_from_layer(self.doc.layer)
        self.cancel_incomplete_interaction()
        if hasattr(self, "tool_options_panel"):
            self.tool_options_panel.render()
        self.refresh_tool_menu()
        if self.tool.get() not in self.brush_preview_tools():
            self.clear_brush_preview()
        elif self._last_pointer_event is not None:
            self.update_brush_preview(self._last_pointer_event)
        if self.tool.get() != "quick_selection":
            self._quick_points.clear()
            self.clear_quick_selection_preview()
        hint = {
            "move": "Кликните объект для выбора и перетащите его",
            "clone": "Alt+клик - выбрать источник",
            "healing": "Alt+клик - выбрать источник",
            "crop": "Потяните область кадрирования; Enter - применить",
            "text": "Потяните для создания текстовой области",
            "hand": "Перетаскивайте холст",
        }.get(new_tool, self.tool_label(new_tool))
        self.status_text(hint)
        cursor = "crosshair" if new_tool.endswith("_shape") or new_tool in {"crop", "select", "ellipse_select"} else "xterm" if new_tool == "text" else "fleur" if new_tool in {"move", "hand"} else "crosshair" if new_tool in self.brush_preview_tools() else "arrow"
        if hasattr(self, "canvas"):
            self.canvas.configure(cursor=cursor)
        self.update_clone_source_marker()

    def tool_label(self, value: str) -> str:
        for label, tool_id, _description in TOOL_DEFINITIONS:
            if tool_id == value:
                return label
        return value

    def refresh_tool_menu(self) -> None:
        if not hasattr(self, "tools_menu"):
            return
        self.tools_menu.delete(0, tk.END)
        by_id = {value: (label, description) for label, value, description in TOOL_DEFINITIONS}
        for value in self.tool_order:
            if value not in by_id:
                continue
            label, _description = by_id[value]
            self.tools_menu.add_radiobutton(
                label=label,
                value=value,
                variable=self.tool,
                command=lambda v=value: self.select_tool(v),
                accelerator=SHORTCUTS.get(value, ""),
            )
        self.tools_menu.add_separator()
        self.tools_menu.add_command(label="Настроить панель инструментов...", command=self.configure_tool_palette)

    def configure_tool_palette(self) -> None:
        dialog = ToolPaletteDialog(self, definitions=TOOL_DEFINITIONS, order=self.tool_order, visible=self.visible_tools)
        self.wait_window(dialog)
        if dialog.result is None:
            return
        order, visible = dialog.result
        self.tool_order = normalize_tool_order(order, TOOL_DEFINITIONS)
        self.visible_tools = normalize_visible_tools(visible, self.tool_order)
        if self.tool.get() not in self.visible_tools:
            self.tool.set(self.visible_tools[0])
        self.tool_palette.set_configuration(self.tool_order, self.visible_tools)
        self.refresh_tool_menu()
        self.save_settings()

    def capture_tool_pane_position(self) -> None:
        if not hasattr(self, "tool_split"):
            return
        try:
            self.tool_pane_position = int(self.tool_split.sashpos(0))
        except Exception:
            pass

    def apply_tool_pane_position(self) -> None:
        if not hasattr(self, "tool_split"):
            return
        try:
            height = max(1, self.tool_split.winfo_height())
            position = max(160, min(int(self.tool_pane_position), max(160, height - 180)))
            self.tool_split.sashpos(0, position)
        except Exception:
            pass

    def _build_canvas(self, parent: ttk.Frame) -> None:
        frame = ttk.Frame(parent, style="Workspace.TFrame")
        frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(frame, bg=TOKENS.WORKSPACE, highlightthickness=0, borderwidth=0)
        xbar = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        ybar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self.canvas.bind("<ButtonPress-1>", self.pointer_down)
        self.canvas.bind("<Alt-Button-1>", self.clone_source_click)
        self.canvas.bind("<B1-Motion>", self.pointer_drag)
        self.canvas.bind("<ButtonRelease-1>", self.pointer_up)
        self.canvas.bind("<ButtonPress-3>", self.selection_right_click)
        self.canvas.bind("<Double-Button-1>", self.pointer_double_click)
        self.canvas.bind("<ButtonPress-2>", self.pan_down)
        self.canvas.bind("<B2-Motion>", self.pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self.pan_up)
        self.canvas.bind("<Motion>", self.pointer_motion)
        self.canvas.bind("<Leave>", self.pointer_leave)
        self.canvas.bind("<FocusOut>", self.canvas_focus_lost)
        self.canvas.bind("<MouseWheel>", self.mouse_wheel)

    def selection_right_click(self, event) -> str | None:
        selection_tools = {
            "select", "ellipse_select", "lasso", "magnetic_lasso", "polygon_lasso",
            "quick_selection", "magic_wand", "color_range",
        }
        mask = self.doc.selection_mask
        if self.tool.get() not in selection_tools or mask is None:
            return None
        x, y = self.canvas_to_doc(event)
        inside = 0 <= x < self.doc.width and 0 <= y < self.doc.height and bool(mask[y, x] > 0)
        if not inside:
            self.run_selection_command("Снять выделение", self.doc.clear_selection)
        return "break"

    def canvas_focus_lost(self, _event) -> None:
        if self.drag_start is not None and self.tool.get().endswith("_shape"):
            self.cancel_incomplete_interaction()

    def apply_retouch_preset(self) -> None:
        preset = RETOUCH_PRESETS.get(self.retouch_preset.get())
        if preset is None:
            return
        preset_tool = str(preset.get("tool", "healing"))
        if preset_tool in {"blur_tool", "sharpen_tool", "dodge", "burn", "clone", "healing", "spot_healing"}:
            self.tool.set(preset_tool)
        self.brush_size.set(int(preset["brush_size"]))
        if preset_tool == "clone":
            self.opacity.set(float(preset["opacity"]))
        elif preset_tool in {"dodge", "burn"}:
            self.exposure.set(float(preset["opacity"]))
        else:
            self.retouch_strength.set(float(preset["opacity"]))
        self.tolerance.set(int(preset["tolerance"]))
        self.status_text(f"Пресет ретуши: {self.retouch_preset.get()}")

    def _build_panels(self, parent: ttk.Frame) -> None:
        self.right_tabs = ttk.Notebook(parent)
        self.right_tabs.pack(fill=tk.BOTH, expand=True)
        layers_tab = ttk.Frame(self.right_tabs, style="Panel.TFrame")
        properties_tab = ttk.Frame(self.right_tabs, style="Panel.TFrame")
        history_tab = ttk.Frame(self.right_tabs, style="Panel.TFrame")
        self.right_tabs.add(layers_tab, text="Слои")
        self.right_tabs.add(properties_tab, text="Свойства")
        self.right_tabs.add(history_tab, text="История")

        self.layer_list = tk.Listbox(
            layers_tab,
            height=16,
            exportselection=False,
            selectmode=tk.EXTENDED,
            activestyle="none",
            background=TOKENS.SURFACE,
            foreground=TOKENS.TEXT_PRIMARY,
            selectbackground=TOKENS.SURFACE_SELECTED,
            selectforeground=TOKENS.TEXT_PRIMARY,
            highlightthickness=0,
            borderwidth=0,
            font=("Segoe UI", 10),
        )
        self.layer_list.pack(fill=tk.BOTH, expand=True, padx=6, pady=(6, 0))
        self.layer_list.bind("<Button-1>", self.layer_list_click)
        self.layer_list.bind("<<ListboxSelect>>", self.layer_selected)
        buttons = ttk.Frame(layers_tab)
        buttons.pack(fill=tk.X, padx=6, pady=6)
        self._panel_icons = {
            name: action_icon(self, name, color=TOKENS.DANGER if name == "delete" else TOKENS.TEXT_PRIMARY)
            for name in ("add", "delete", "duplicate", "up", "down")
        }
        add = ttk.Button(buttons, image=self._panel_icons["add"], width=3, command=self.new_layer)
        delete = ttk.Button(buttons, image=self._panel_icons["delete"], width=3, command=self.delete_layer, style="Danger.TButton")
        duplicate = ttk.Button(buttons, image=self._panel_icons["duplicate"], width=3, command=self.duplicate_layer)
        up = ttk.Button(buttons, image=self._panel_icons["up"], width=3, command=lambda: self.move_layer(1))
        down = ttk.Button(buttons, image=self._panel_icons["down"], width=3, command=lambda: self.move_layer(-1))
        for button in (add, delete, duplicate, up, down):
            button.pack(side=tk.LEFT, padx=(0, 3))
        ToolTip(add, "Новый слой")
        ToolTip(delete, "Удалить выбранные слои")
        ToolTip(duplicate, "Дублировать слой")
        ToolTip(up, "Поднять слой")
        ToolTip(down, "Опустить слой")

        thumbs = ttk.Frame(layers_tab)
        thumbs.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.layer_thumb = ttk.Label(thumbs)
        self.layer_thumb.pack(side=tk.LEFT)
        self.mask_thumb = ttk.Label(thumbs)
        self.mask_thumb.pack(side=tk.LEFT, padx=(8, 0))
        self.layer_thumb.bind("<Button-1>", self.edit_pixels_channel)
        self.mask_thumb.bind("<Button-1>", self.edit_mask_channel)
        ToolTip(self.layer_thumb, "Пиксели активного слоя")
        ToolTip(self.mask_thumb, "Маска активного слоя")

        properties = ScrollableFrame(properties_tab, height=500)
        properties.pack(fill=tk.BOTH, expand=True)
        property_root = properties.content
        ttk.Label(property_root, text="Слой", style="PanelTitle.TLabel").pack(anchor=tk.W, padx=10, pady=(10, 4))
        ttk.Label(property_root, text="Непрозрачность", style="Secondary.TLabel").pack(anchor=tk.W, padx=10)
        self.layer_opacity = tk.DoubleVar(value=1.0)
        self.layer_opacity_scale = ttk.Scale(property_root, from_=0.0, to=1.0, variable=self.layer_opacity, command=self.change_layer_opacity)
        self.layer_opacity_scale.pack(fill=tk.X, padx=10)
        self.layer_opacity_scale.bind("<ButtonPress-1>", self.begin_layer_opacity_change)
        self.layer_opacity_scale.bind("<ButtonRelease-1>", self.end_layer_opacity_change)
        ttk.Label(property_root, text="Режим наложения", style="Secondary.TLabel").pack(anchor=tk.W, padx=10, pady=(7, 0))
        self.blend_mode = tk.StringVar(value="Normal")
        self.blend_mode_box = ttk.Combobox(property_root, textvariable=self.blend_mode, values=BLEND_MODES, state="readonly")
        self.blend_mode_box.pack(fill=tk.X, padx=10)
        self.blend_mode_box.bind("<<ComboboxSelected>>", self.change_blend_mode)
        common = ttk.Frame(property_root)
        common.pack(fill=tk.X, padx=10, pady=7)
        ttk.Button(common, text="Видимость", command=self.toggle_layer_visible).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(common, text="Блокировка", command=self.toggle_layer_lock).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        ttk.Label(property_root, text="Просмотр маски", style="Secondary.TLabel").pack(anchor=tk.W, padx=10)
        self.mask_preview_box = ttk.Combobox(property_root, textvariable=self.mask_preview, values=MASK_PREVIEW_MODES, state="readonly")
        self.mask_preview_box.pack(fill=tk.X, padx=10, pady=(0, 7))
        self.mask_preview_box.bind("<<ComboboxSelected>>", lambda _event: self.set_mask_preview())
        ttk.Separator(property_root).pack(fill=tk.X, padx=10, pady=7)
        self.object_properties = ttk.Frame(property_root)
        self.object_properties.pack(fill=tk.X)

        self.history_list = tk.Listbox(
            history_tab,
            activestyle="none",
            background=TOKENS.SURFACE,
            foreground=TOKENS.TEXT_PRIMARY,
            selectbackground=TOKENS.SURFACE_SELECTED,
            highlightthickness=0,
            borderwidth=0,
        )
        self.history_list.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        history_actions = ttk.Frame(history_tab)
        history_actions.pack(fill=tk.X, padx=6, pady=(0, 6))
        ttk.Button(history_actions, text="Отменить", command=self.undo).pack(side=tk.LEFT)
        ttk.Button(history_actions, text="Повторить", command=self.redo).pack(side=tk.LEFT, padx=4)
        self.info = ttk.Label(property_root, text="", justify=tk.LEFT, style="Secondary.TLabel")
        self.info.pack(anchor=tk.W, padx=10, pady=8)

    def push_command(self, command) -> None:
        self.history.push(command)
        self._edit_generation += 1
        self.record_action(command.label)
        self.status_text(command.label)
        self.refresh_history_panel()

    def run_document_command(self, label: str, fn) -> None:
        before = self.doc.raw_state()
        fn()
        after = self.doc.raw_state()
        command = self.compact_document_command(label, before, after)
        if command is not None:
            self.history.push(command)
        self._edit_generation += 1
        self.record_action(label)
        self.status_text(label)
        self.refresh_history_panel()

    @staticmethod
    def state_value_equal(left, right) -> bool:
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and left.shape == right.shape and np.array_equal(left, right)
        if isinstance(left, dict) and isinstance(right, dict):
            return left.keys() == right.keys() and all(PhotoRedactorApp.state_value_equal(left[key], right[key]) for key in left)
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            return len(left) == len(right) and all(PhotoRedactorApp.state_value_equal(a, b) for a, b in zip(left, right))
        return left == right

    def compact_document_command(self, label: str, before: dict, after: dict):
        before_layers = before.get("layers", [])
        after_layers = after.get("layers", [])
        same_layer_order = [layer.get("id") for layer in before_layers] == [layer.get("id") for layer in after_layers]
        document_keys = (set(before) | set(after)) - {"layers"}
        changed_document = {
            key: (before.get(key), after.get(key))
            for key in document_keys
            if not self.state_value_equal(before.get(key), after.get(key))
        }
        changed_layers: list[tuple[str, dict, dict]] = []
        if same_layer_order:
            for old_layer, new_layer in zip(before_layers, after_layers):
                changed_fields = {
                    key
                    for key in set(old_layer) | set(new_layer)
                    if not self.state_value_equal(old_layer.get(key), new_layer.get(key))
                }
                if changed_fields:
                    changed_layers.append(
                        (
                            str(old_layer["id"]),
                            {key: old_layer.get(key) for key in changed_fields},
                            {key: new_layer.get(key) for key in changed_fields},
                        )
                    )
        if same_layer_order and len(changed_layers) == 1 and not changed_document:
            layer_id, old_fields, new_fields = changed_layers[0]
            return LayerFieldsCommand(label, layer_id, old_fields, new_fields)
        if same_layer_order and not changed_layers and changed_document:
            return DocumentFieldsCommand(
                label,
                {key: value[0] for key, value in changed_document.items()},
                {key: value[1] for key, value in changed_document.items()},
            )
        if same_layer_order and not changed_layers and not changed_document:
            return None
        return DocumentStateCommand(label, before, after)

    def run_selection_command(self, label: str, fn) -> None:
        before = None if self.doc.selection_mask is None else self.doc.selection_mask.copy()
        fn()
        after = None if self.doc.selection_mask is None else self.doc.selection_mask.copy()
        self.history.push(SelectionMaskCommand(label, before, after))
        self._edit_generation += 1
        self.record_action(label)
        self.selection_box = self.doc.selection_bounds()
        self._selection_contour_signature = None
        self.update_selection_overlay()
        self.status_text(label)

    def record_action(self, label: str) -> None:
        if not self.action_recorder.recording:
            return
        normalized = label.lower()
        command = ""
        params: dict[str, object] = {}
        if "resize image" in normalized or "размер изображения" in normalized:
            command, params = "resize_image", {"width": self.doc.width, "height": self.doc.height}
        elif "resize canvas" in normalized or "размер холста" in normalized:
            command, params = "resize_canvas", {"width": self.doc.width, "height": self.doc.height, "anchor": "center"}
        elif "flatten" in normalized or "свести" in normalized:
            command = "flatten"
        elif "rotate" in normalized or "повернуть" in normalized:
            command, params = "rotate", {"angle": 180 if "180" in normalized else 90}
        elif "flip" in normalized or "отразить" in normalized:
            command, params = "flip", {"axis": "vertical" if "vertical" in normalized or "вертик" in normalized else "horizontal"}
        elif "bit depth" in normalized or "глубина" in normalized:
            command, params = "set_bit_depth", {"bit_depth": self.doc.bit_depth}
        elif "color model" in normalized or "цветовая модель" in normalized:
            command, params = "set_color_model", {"color_model": self.doc.color_model}
        elif self.doc.layer.filters:
            command, params = "filter_stack", {"filters": copy.deepcopy(self.doc.layer.filters)}
        if command:
            self.action_recorder.record(command, params, label)

    def start_action_recording(self) -> None:
        self.action_recorder.start()
        self.status_text("Запись действия начата")

    def stop_action_recording(self) -> None:
        self.action_recorder.stop()
        self.status_text(f"Запись остановлена: {len(self.action_recorder.steps)} шагов")

    def clear_action_recording(self) -> None:
        self.action_recorder.steps.clear()
        self.status_text("Запись действия очищена")

    def save_action_recording(self) -> None:
        if not self.action_recorder.steps:
            messagebox.showinfo("Действия", "Нет записанных исполняемых шагов.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Action JSON", "*.json")])
        if not path:
            return
        self.action_recorder.save(path)
        self.status_text(f"Действие сохранено: {path}")

    def run_action_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Действие PhotoRedactor", "*.json"), ("Все файлы", "*.*")])
        if not path:
            return
        try:
            self.run_document_command("Выполнить действие", lambda: self.action_runner.run(self.doc, path))
            self.refresh()
        except Exception as exc:
            messagebox.showerror("Действия", str(exc))

    def batch_action_file(self) -> None:
        action = filedialog.askopenfilename(filetypes=[("Действие PhotoRedactor", "*.json")])
        if not action:
            return
        sources = filedialog.askopenfilenames(filetypes=[("Изображения", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff")])
        if not sources:
            return
        destination = filedialog.askdirectory(title="Папка результата")
        if not destination:
            return
        self.run_background(
            "Пакетное действие",
            lambda: self.action_runner.batch(action, list(sources), destination),
            lambda results: messagebox.showinfo("Действия", f"Обработано файлов: {len(results)}"),
        )

    def schedule_autosave(self) -> None:
        self.after(60000, self.autosave_tick)

    def autosave_tick(self) -> None:
        self.autosave_recovery()
        self.schedule_autosave()

    def autosave_recovery(self) -> None:
        if not getattr(self, "doc", None) or not self.doc.dirty:
            return
        try:
            self.recovery_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot = self.document_copy()
            snapshot.save_project(self.recovery_path)
            self.doc.dirty = True
        except Exception:
            pass

    def check_recovery_file(self) -> None:
        if self.recovery_path.exists() and messagebox.askyesno("Recovery", "A recovery file was found. Open it?"):
            self.open_recovery()

    def open_recovery(self) -> None:
        if not self.recovery_path.exists():
            messagebox.showinfo("Восстановление", "Файл восстановления не найден.")
            return
        self.doc = Document.open_project(self.recovery_path)
        self.history.clear()
        self.selection_box = self.doc.selection_bounds()

    def run_pixel_delta_command(self, label: str, fn) -> tuple[int, int, int, int] | None:
        layer = self.doc.layer
        if layer.locked or layer.kind == "adjustment":
            return None
        layer_id = layer.id
        before = layer.pixels.copy()
        fn()
        target = self.doc.get_layer(layer_id)
        if target is None or target.pixels.shape != before.shape:
            return None
        changed = np.any(target.pixels != before, axis=2)
        if not np.any(changed):
            return None
        ys, xs = np.where(changed)
        rect = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        x1, y1, x2, y2 = rect
        self.push_command(PixelPatchCommand(label, layer_id, rect, before[y1:y2, x1:x2].copy(), target.pixels[y1:y2, x1:x2].copy()))
        self.doc.dirty = True
        self.request_canvas_refresh(self.local_to_document_rect(rect, target), target, "pixels")
        self.refresh_layers()
        return rect

    def set_layer_property(self, label: str, attribute: str, value, affects_canvas: bool = True, preserve_render_cache: bool = True) -> bool:
        layer = self.doc.layer
        before = copy.deepcopy(getattr(layer, attribute))
        after = copy.deepcopy(value)
        if before == after:
            return False
        setattr(layer, attribute, after)
        if attribute == "mask":
            layer.touch_mask()
        self.doc.dirty = True
        self.push_command(LayerPropertyCommand(label, layer.id, attribute, before, after))
        if affects_canvas:
            self.refresh(preserve_render_cache=preserve_render_cache)
        else:
            self.refresh_layers()
        return True
        self.show_editor()
        self.status_text(f"Открыто восстановление: {self.recovery_path}")

    def clear_recovery(self) -> None:
        if self.recovery_path.exists():
            self.recovery_path.unlink()
        self.status_text("Recovery cleared")

    def undo(self) -> None:
        label = self.history.undo(self.doc)
        if label:
            self._edit_generation += 1
            self.refresh_history_command(self.history.last_command)
            self.status_text(f"Undo: {label}")

    def redo(self) -> None:
        label = self.history.redo(self.doc)
        if label:
            self._edit_generation += 1
            self.refresh_history_command(self.history.last_command)
            self.status_text(f"Redo: {label}")

    def refresh_history_command(self, command) -> None:
        if isinstance(command, (PixelPatchCommand, PixelTilePatchCommand, MaskPatchCommand, MaskTilePatchCommand)):
            layer = self.doc.get_layer(command.layer_id)
            if layer is not None:
                kind = "mask" if isinstance(command, (MaskPatchCommand, MaskTilePatchCommand)) else "pixels"
                for rect in command.dirty_rects:
                    self.render_engine.invalidate_region(self.doc, self.local_to_document_rect(rect, layer), layer, kind)
                self._composite_dirty = True
                self._view_dirty = True
                self.refresh_canvas()
                self.refresh_layers()
                self.info.configure(text=f"{self.doc.width} x {self.doc.height}px\nРЎР»РѕРµРІ: {len(self.doc.layers)}\nРђРєС‚РёРІРЅС‹Р№: {self.doc.layer.name}")
                return
        if isinstance(command, LayerPropertyCommand):
            if command.attribute in {"name", "locked", "mask_linked"}:
                self.refresh_layers()
                return
            self.refresh(preserve_render_cache=command.attribute not in {"filters", "effects", "mask", "mask_feather"})
            return
        if isinstance(command, (LayerOpacityCommand, LayerBlendModeCommand, LayerVisibilityCommand)):
            self.refresh(preserve_render_cache=True)
            return
        if isinstance(command, LayerFieldsCommand):
            changed = set(command.before) | set(command.after)
            self.refresh(preserve_render_cache=not bool(changed & {"pixels", "filters", "effects", "mask", "mask_feather"}))
            return
        self.refresh()

    def invalidate_pixels(self, clear_layer_caches: bool = True) -> None:
        self._composite_dirty = True
        self._view_dirty = True
        self.render_engine.invalidate_full(self.doc, clear_layer_caches=clear_layer_caches)

    def invalidate_view(self) -> None:
        self._view_dirty = True

    def refresh_canvas(self) -> None:
        composite_changed = self._composite_dirty or self._composite_cache is None
        if self._composite_dirty or self._composite_cache is None:
            self._composite_cache = self.render_engine.render(self.doc, checker=True)
            self._composite_dirty = False
        display = self._channel_display(self._composite_cache)
        display = self._mask_preview_display(display)
        scale = self.zoom.get()
        pad_x = max(40, self.canvas.winfo_width() // 2)
        pad_y = max(40, self.canvas.winfo_height() // 2)
        self._canvas_origin = (pad_x, pad_y)
        view_signature = (
            round(scale, 6),
            self.doc.width,
            self.doc.height,
            self.view_channel.get(),
            self.mask_preview.get(),
            self.doc.layer.id if self.doc.layers else None,
        )
        full_view = self._canvas_view_signature != view_signature or not self._canvas_tile_ids
        if scale < 0.5:
            self._update_canvas_mipmap(display, scale, pad_x, pad_y)
        else:
            changed_tiles = self.render_engine.last_changed_tiles if composite_changed and not full_view else set(self.render_engine.all_tiles(self.doc))
            self._update_canvas_tiles(display, changed_tiles, scale, pad_x, pad_y, full_view)
        scaled_width = max(1, round(self.doc.width * scale))
        scaled_height = max(1, round(self.doc.height * scale))
        self.canvas.configure(scrollregion=(0, 0, scaled_width + pad_x * 2, scaled_height + pad_y * 2))
        self._canvas_view_signature = view_signature
        self.zoom_label.configure(text=f"{round(scale * 100)}%")
        if hasattr(self, "status_zoom"):
            self.status_zoom.configure(text=f"{round(scale * 100)}%")
        self._last_render_time = time.perf_counter()
        self._view_dirty = False
        self.update_selection_overlay()
        self.update_object_bounds()
        self.update_grid_and_guides()
        if self.tool.get() == "quick_selection" and self._quick_points:
            self.update_quick_selection_preview(force=True)
        if self._last_pointer_event is not None:
            self.update_brush_preview(self._last_pointer_event)
        self.update_clone_source_marker()
        if self._crop_box is not None and self.tool.get() == "crop":
            self.draw_crop_overlay(self._crop_box)

    def _update_canvas_mipmap(self, display: np.ndarray, scale: float, pad_x: int, pad_y: int) -> None:
        for item_id in self._canvas_tile_ids.values():
            self.canvas.delete(item_id)
        self._canvas_tile_ids.clear()
        self._canvas_tile_images.clear()
        key = (
            id(self.doc),
            self.render_engine.render_revision,
            self.view_channel.get(),
            self.mask_preview.get(),
        )
        reduced, level = self.render_engine.mipmaps.for_zoom(key, display, scale)
        target = max(1, round(self.doc.width * scale)), max(1, round(self.doc.height * scale))
        image = rgba_array_to_pil(reduced)
        if image.size != target:
            image = image.resize(target, Image.Resampling.BILINEAR)
        self._preview_image = ImageTk.PhotoImage(image)
        if self._canvas_image_id is None:
            self._canvas_image_id = self.canvas.create_image(pad_x, pad_y, image=self._preview_image, anchor=tk.NW)
        else:
            self.canvas.itemconfigure(self._canvas_image_id, image=self._preview_image)
            self.canvas.coords(self._canvas_image_id, pad_x, pad_y)
        self.render_engine.profiler.count("canvas.mipmap_level", level)

    def _update_canvas_tiles(
        self,
        display: np.ndarray,
        changed_tiles: set[tuple[int, int]],
        scale: float,
        pad_x: int,
        pad_y: int,
        full_view: bool,
    ) -> None:
        if self._canvas_image_id is not None:
            self.canvas.delete(self._canvas_image_id)
            self._canvas_image_id = None
            self._preview_image = None
        if full_view:
            for item_id in self._canvas_tile_ids.values():
                self.canvas.delete(item_id)
            self._canvas_tile_ids.clear()
            self._canvas_tile_images.clear()
        resample = Image.Resampling.NEAREST if scale >= 4 else Image.Resampling.BILINEAR
        for tx, ty in sorted(changed_tiles):
            x1, y1, x2, y2 = self.render_engine.tile_rect(self.doc, tx, ty)
            with self.render_engine.profiler.measure("canvas.numpy_to_pil"):
                image = rgba_array_to_pil(display[y1:y2, x1:x2])
            left, top = round(x1 * scale), round(y1 * scale)
            right, bottom = round(x2 * scale), round(y2 * scale)
            size = max(1, right - left), max(1, bottom - top)
            if image.size != size:
                with self.render_engine.profiler.measure("canvas.resize_tile"):
                    image = image.resize(size, resample)
            with self.render_engine.profiler.measure("canvas.pil_to_imagetk"):
                photo = ImageTk.PhotoImage(image)
            key = tx, ty
            item_id = self._canvas_tile_ids.get(key)
            if item_id is None:
                item_id = self.canvas.create_image(pad_x + left, pad_y + top, image=photo, anchor=tk.NW)
                self._canvas_tile_ids[key] = item_id
            else:
                self.canvas.itemconfigure(item_id, image=photo)
                self.canvas.coords(item_id, pad_x + left, pad_y + top)
            self._canvas_tile_images[key] = photo

    def _channel_display(self, composite: np.ndarray) -> np.ndarray:
        channel = self.view_channel.get() if hasattr(self, "view_channel") else "RGB"
        if channel == "RGB":
            return composite
        source = self.render_engine.render(self.doc, checker=False)
        out = np.zeros_like(source)
        out[:, :, 3] = 255
        if channel == "Alpha":
            gray = source[:, :, 3]
        else:
            index = {"Red": 0, "Green": 1, "Blue": 2}.get(channel, 0)
            gray = source[:, :, index]
        out[:, :, :3] = gray[:, :, None]
        return out

    def _mask_preview_display(self, display: np.ndarray) -> np.ndarray:
        mode = self.mask_preview.get() if hasattr(self, "mask_preview") else MASK_PREVIEW_NORMAL
        if mode == MASK_PREVIEW_NORMAL:
            return display
        layer = self.doc.layer
        if layer.mask is None:
            return display
        mask_canvas = np.zeros((self.doc.height, self.doc.width), dtype=np.uint8)
        active_canvas = np.zeros((self.doc.height, self.doc.width), dtype=np.uint8)
        effective = effective_layer_mask(layer) if layer.mask_enabled else layer.mask
        paste_mask(mask_canvas, effective, layer.x, layer.y)
        paste_mask(active_canvas, np.full(layer.mask.shape, 255, dtype=np.uint8), layer.x, layer.y)
        if mode == MASK_PREVIEW_CHANNEL:
            gray = np.where(active_canvas > 0, mask_canvas, 72).astype(np.uint8)
            out = np.zeros((self.doc.height, self.doc.width, 4), dtype=np.uint8)
            out[:, :, :3] = gray[:, :, None]
            out[:, :, 3] = 255
            return out
        if mode == MASK_PREVIEW_OVERLAY:
            out = display.copy()
            hidden = ((255 - mask_canvas).astype(np.float32) / 255.0) * (active_canvas > 0)
            alpha = (hidden * 0.58)[:, :, None]
            red = np.zeros_like(out[:, :, :3])
            red[:, :, 0] = 255
            red[:, :, 1] = 36
            red[:, :, 2] = 68
            out[:, :, :3] = np.clip(out[:, :, :3].astype(np.float32) * (1.0 - alpha) + red.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
            out[:, :, 3] = 255
            return out
        return display

    def request_canvas_refresh(self, rect: tuple[int, int, int, int] | None = None, layer=None, kind: str = "pixels", preserve_layer_caches: bool = False) -> None:
        if rect is None:
            self.invalidate_pixels(clear_layer_caches=not preserve_layer_caches)
        else:
            self.render_engine.invalidate_region(self.doc, rect, layer, kind)
            self._composite_dirty = True
            self._view_dirty = True
        if self._render_after_id is not None:
            return
        elapsed_ms = (time.perf_counter() - self._last_render_time) * 1000
        delay = 0 if elapsed_ms >= 33 else int(33 - elapsed_ms)
        self._render_after_id = self.after(delay, self._run_scheduled_canvas_refresh)

    def _run_scheduled_canvas_refresh(self) -> None:
        self._render_after_id = None
        self.refresh_canvas()

    def refresh(self, preserve_render_cache: bool = False) -> None:
        self.invalidate_pixels(clear_layer_caches=not preserve_render_cache)
        if self._render_after_id is not None:
            self.after_cancel(self._render_after_id)
            self._render_after_id = None
        self.refresh_canvas()
        self.refresh_layers()
        self.refresh_properties()
        self.refresh_history_panel()
        self.info.configure(text=f"{self.doc.width} x {self.doc.height}px\nСлоев: {len(self.doc.layers)}\nАктивный: {self.doc.layer.name}")
        self.status_size.configure(text=f"{self.doc.width} x {self.doc.height}")

    def refresh_layers(self) -> None:
        known_ids = {layer.id for layer in self.doc.layers}
        selected_ids = set(getattr(self, "selected_layer_ids", set())) & known_ids
        if not selected_ids and self.doc.layers:
            selected_ids = {self.doc.layer.id}
        self.selected_layer_ids = selected_ids
        rows: list[str] = []
        for i, layer in enumerate(reversed(self.doc.layers)):
            marker = "V" if layer.visible else " "
            indicators: list[str] = []
            if layer.mask is not None:
                indicators.append("M")
            if layer.locked:
                indicators.append("L")
            if layer.kind == "linked":
                linked_status = self.doc.linked_layer_status(layer)["status"]
                if linked_status in {"missing", "modified"}:
                    indicators.append("?" if linked_status == "missing" else "!")
            if layer.effects or layer.filters:
                indicators.append("fx")
            suffix = f"  [{' '.join(indicators)}]" if indicators else ""
            rows.append(f"{marker}   {layer.name}{suffix}")
        existing = list(self.layer_list.get(0, tk.END))
        if len(existing) != len(rows):
            self.layer_list.delete(0, tk.END)
            for row in rows:
                self.layer_list.insert(tk.END, row)
        else:
            for index, row in enumerate(rows):
                if existing[index] != row:
                    self.layer_list.delete(index)
                    self.layer_list.insert(index, row)
        self.layer_list.selection_clear(0, tk.END)
        for row, layer in enumerate(reversed(self.doc.layers)):
            if layer.id in self.selected_layer_ids:
                self.layer_list.selection_set(row)
        self.layer_list.activate(len(self.doc.layers) - 1 - self.doc.active_layer)
        self.layer_opacity.set(self.doc.layer.opacity)
        self.blend_mode.set(self.doc.layer.blend_mode)
        self.refresh_layer_previews()

    def refresh_history_panel(self) -> None:
        if not hasattr(self, "history_list"):
            return
        labels = [command.label for command in self.history.undo_stack]
        existing = list(self.history_list.get(0, tk.END))
        if existing != labels:
            self.history_list.delete(0, tk.END)
            for label in labels:
                self.history_list.insert(tk.END, label)
            if labels:
                self.history_list.selection_set(tk.END)
                self.history_list.see(tk.END)

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

        def combo_row(label: str, initial: str, values: list[str], apply) -> None:
            row = ttk.Frame(panel)
            row.pack(fill=tk.X, padx=10, pady=2)
            ttk.Label(row, text=label, style="Secondary.TLabel").pack(side=tk.LEFT)
            variable = tk.StringVar(value=initial)
            combo = ttk.Combobox(row, textvariable=variable, values=values, state="readonly", width=14)
            combo.pack(side=tk.RIGHT)
            combo.bind("<<ComboboxSelected>>", lambda _event: apply(variable.get()))

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
        numeric_row("X", display_x, lambda value, y=display_y: self.set_object_position(int(value), y))
        numeric_row("Y", display_y, lambda value, x=display_x: self.set_object_position(x, int(value)))
        if layer.kind == "shape" and layer.shape_data is not None:
            box = shape_data_bounds(layer.shape_data) or (0, 0, 1, 1)
            numeric_row("Ширина", box[2] - box[0], lambda value: self.set_shape_size(int(value), None), 2, 100000)
            numeric_row("Высота", box[3] - box[1], lambda value: self.set_shape_size(None, int(value)), 2, 100000)
            numeric_row("Обводка", int(layer.shape_data.get("stroke_width", 0)), lambda value: self.set_shape_property("stroke_width", int(value)), 0, 100)
            kind = str(layer.shape_data.get("shape", "rectangle"))
            if kind in {"polygon", "star"}:
                numeric_row("Стороны" if kind == "polygon" else "Лучи", int(layer.shape_data.get("sides", 5)), lambda value: self.set_shape_property("sides", int(value)), 3, 64)
            if kind == "star":
                numeric_row("Внутренний радиус", float(layer.shape_data.get("inner_ratio", 0.5)), lambda value: self.set_shape_property("inner_ratio", float(value)), 0.05, 0.95, 0.05)
            color_row("Заливка", layer.shape_data.get("fill") or [0, 0, 0, 0], lambda: self.pick_shape_property_color("fill"))
            color_row("Обводка", layer.shape_data.get("stroke") or [0, 0, 0, 0], lambda: self.pick_shape_property_color("stroke"))
            if kind == "boolean":
                ttk.Button(panel, text="Редактировать операцию и контуры", command=self.edit_boolean_shape).pack(fill=tk.X, padx=10, pady=6)
        elif layer.kind == "text" and layer.text_data is not None:
            combo_row("Шрифт", str(layer.text_data.get("font_family", "Arial")), ["Arial", "Segoe UI", "Calibri", "Times New Roman", "Verdana", "Tahoma"], lambda value: self.set_text_property("font_family", value))
            numeric_row("Размер", int(layer.text_data.get("size", 48)), lambda value: self.set_text_property("size", int(value)), 4, 500)
            numeric_row("Интервал", int(layer.text_data.get("tracking", 0)), lambda value: self.set_text_property("tracking", int(value)), -100, 500)
            numeric_row("Межстрочный", int(layer.text_data.get("line_spacing", 10)), lambda value: self.set_text_property("line_spacing", int(value)), 0, 500)
            combo_row("Выравнивание", str(layer.text_data.get("align", "left")), ["left", "center", "right"], lambda value: self.set_text_property("align", value))
            color_row("Цвет", layer.text_data.get("color") or [255, 255, 255, 255], self.pick_text_property_color)
            ttk.Button(panel, text="Редактировать текст", command=self.edit_active_text_on_canvas).pack(fill=tk.X, padx=10, pady=6)
            ttk.Button(panel, text="Текст по контуру...", command=self.edit_text_path).pack(fill=tk.X, padx=10, pady=(0, 6))
        else:
            ttk.Button(panel, text="Фильтры слоя", command=self.edit_layer_filters).pack(fill=tk.X, padx=10, pady=5)

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

    def refresh_layer_previews(self) -> None:
        layer = self.doc.layer
        layer_key = (layer.id, layer.pixels_revision, id(layer.pixels), layer.pixels.shape)
        layer_preview = self._layer_thumbnail_cache.get(layer_key)
        if layer_preview is None:
            layer_preview = self.make_layer_thumbnail(layer.pixels)
            self._layer_thumbnail_cache[layer_key] = layer_preview
            self._trim_thumbnail_cache(self._layer_thumbnail_cache)
        self._layer_thumb_image = ImageTk.PhotoImage(layer_preview)
        self.layer_thumb.configure(image=self._layer_thumb_image)
        mask_key = (layer.id, layer.mask_revision, id(layer.mask), None if layer.mask is None else layer.mask.shape)
        mask_preview = self._mask_thumbnail_cache.get(mask_key)
        if mask_preview is None:
            mask_preview = self.make_mask_thumbnail(layer.mask)
            self._mask_thumbnail_cache[mask_key] = mask_preview
            self._trim_thumbnail_cache(self._mask_thumbnail_cache)
        self._mask_thumb_image = ImageTk.PhotoImage(mask_preview)
        self.mask_thumb.configure(image=self._mask_thumb_image)

    @staticmethod
    def _trim_thumbnail_cache(cache: dict[tuple[object, ...], Image.Image], limit: int = 128) -> None:
        while len(cache) > limit:
            cache.pop(next(iter(cache)))

    def make_layer_thumbnail(self, pixels: np.ndarray, size: int = 64) -> Image.Image:
        height, width = pixels.shape[:2]
        scale = min(1.0, size / max(1, width), size / max(1, height))
        preview_size = max(1, round(width * scale)), max(1, round(height * scale))
        preview = pixels if preview_size == (width, height) else cv2.resize(pixels, preview_size, interpolation=cv2.INTER_AREA)
        image = rgba_array_to_pil(preview)
        canvas = Image.new("RGBA", (size, size), (44, 46, 52, 255))
        x = (size - image.width) // 2
        y = (size - image.height) // 2
        canvas.alpha_composite(image, (x, y))
        return canvas

    def make_mask_thumbnail(self, mask: np.ndarray | None, size: int = 64) -> Image.Image:
        if mask is None:
            return Image.new("RGBA", (size, size), (72, 74, 82, 255))
        height, width = mask.shape[:2]
        scale = min(1.0, size / max(1, width), size / max(1, height))
        preview_size = max(1, round(width * scale)), max(1, round(height * scale))
        preview = mask if preview_size == (width, height) else cv2.resize(mask, preview_size, interpolation=cv2.INTER_AREA)
        image = Image.fromarray(preview.astype(np.uint8), "L")
        canvas = Image.new("L", (size, size), 72)
        x = (size - image.width) // 2
        y = (size - image.height) // 2
        canvas.paste(image, (x, y))
        return Image.merge("RGBA", (canvas, canvas, canvas, Image.new("L", (size, size), 255)))

    def status_text(self, text: str) -> None:
        if hasattr(self, "status"):
            self.status.configure(text=text)

    def document_copy(self) -> Document:
        doc = Document.new(1, 1)
        doc.restore_raw_state(self.doc.raw_state())
        return doc

    def run_background(self, label: str, worker, done=None, is_current=None) -> None:
        self.status_text(f"{label}...")
        future = self.executor.submit(worker)

        def complete() -> None:
            try:
                result = future.result()
            except Exception as exc:
                messagebox.showerror(label, str(exc))
                self.status_text(f"{label}: error")
                return
            if is_current is not None and not is_current():
                self.status_text(f"{label}: result discarded because the document changed")
                return
            if done:
                done(result)
            self.status_text(f"{label}: done")

        future.add_done_callback(lambda _future: self.after(0, complete))

    def canvas_to_doc(self, event) -> tuple[int, int]:
        ox, oy = self._canvas_origin
        x = int((self.canvas.canvasx(event.x) - ox) / self.zoom.get())
        y = int((self.canvas.canvasy(event.y) - oy) / self.zoom.get())
        return x, y

    def doc_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        ox, oy = self._canvas_origin
        scale = self.zoom.get()
        return ox + x * scale, oy + y * scale

    def selection_mode_from_event(self, event) -> str:
        shift = bool(event.state & 0x0001)
        ctrl = bool(event.state & 0x0004)
        if shift and ctrl:
            return "intersect"
        if shift:
            return "add"
        if ctrl:
            return "subtract"
        mode = self.selection_mode.get()
        return mode if mode in {"replace", "add", "subtract", "intersect"} else "replace"

    @staticmethod
    def brush_preview_tools() -> set[str]:
        return {"brush", "eraser", "blur_tool", "sharpen_tool", "dodge", "burn", "clone", "healing", "spot_healing", "quick_selection"}

    def pointer_motion(self, event) -> None:
        self._last_pointer_event = event
        point = self.canvas_to_doc(event)
        if hasattr(self, "status_coords"):
            self.status_coords.configure(text=f"{point[0]}, {point[1]}")
        if self.tool.get() == "move" and not self._panning:
            if self._move_layer_id is not None:
                self.canvas.configure(cursor="fleur")
            else:
                handle = self.object_handle_at(point)
                cursor_by_handle = {
                    "nw": "size_nw_se", "se": "size_nw_se", "ne": "size_ne_sw", "sw": "size_ne_sw",
                    "n": "sb_v_double_arrow", "s": "sb_v_double_arrow", "e": "sb_h_double_arrow", "w": "sb_h_double_arrow",
                }
                if handle:
                    self.canvas.configure(cursor=cursor_by_handle[handle])
                else:
                    hit = topmost_layer_at(self.doc, point, tolerance=max(2, round(5 / max(self.zoom.get(), 0.01))))
                    self.canvas.configure(cursor="fleur" if hit is not None else "arrow")
        if not self._panning:
            self.update_brush_preview(event)
        if self.tool.get() == "polygon_lasso" and self._polygon_points:
            self.draw_polygon_lasso(self.canvas_to_doc(event))

    def pointer_leave(self, _event) -> None:
        self._last_pointer_event = None
        if hasattr(self, "status_coords"):
            self.status_coords.configure(text="")
        self.clear_brush_preview()

    def object_document_bounds(self, layer: Layer | None = None) -> tuple[int, int, int, int] | None:
        layer = layer or self.doc.layer
        if layer.kind == "shape" and layer.shape_data is not None:
            x1, y1, x2, y2 = shape_data_bounds(layer.shape_data) or (0, 0, 1, 1)
            return x1 + layer.x, y1 + layer.y, x2 + layer.x, y2 + layer.y
        if layer.kind == "text" and layer.pixels.size:
            ys, xs = np.where(layer.pixels[:, :, 3] > 8)
            if len(xs):
                return int(xs.min()) + layer.x, int(ys.min()) + layer.y, int(xs.max() + 1) + layer.x, int(ys.max() + 1) + layer.y
        return None

    def layer_render_bounds(self, layer: Layer) -> tuple[int, int, int, int]:
        object_bounds = self.object_document_bounds(layer)
        if object_bounds is not None:
            return object_bounds
        height, width = layer.pixels.shape[:2]
        return layer.x, layer.y, layer.x + width, layer.y + height

    def object_handle_points(self, bounds: tuple[int, int, int, int]) -> dict[str, tuple[float, float]]:
        x1, y1, x2, y2 = bounds
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return {"nw": (x1, y1), "n": (cx, y1), "ne": (x2, y1), "e": (x2, cy), "se": (x2, y2), "s": (cx, y2), "sw": (x1, y2), "w": (x1, cy)}

    def object_handle_at(self, point: tuple[int, int]) -> str | None:
        if self.tool.get() != "move" or self.doc.layer.kind != "shape" or self.doc.layer.id not in self.selected_layer_ids:
            return None
        bounds = self.object_document_bounds()
        if bounds is None:
            return None
        tolerance = max(3.0, 7.0 / max(self.zoom.get(), 0.01))
        for name, (x, y) in self.object_handle_points(bounds).items():
            if abs(point[0] - x) <= tolerance and abs(point[1] - y) <= tolerance:
                return name
        return None

    def update_object_bounds(self) -> None:
        if not hasattr(self, "canvas"):
            return
        layer = self.doc.layer if self.doc.layers else None
        bounds = None if layer is None or layer.id not in self.selected_layer_ids or layer.kind not in {"shape", "text"} else self.object_document_bounds(layer)
        if bounds is None:
            for item_id in self._object_bounds_ids:
                self.canvas.delete(item_id)
            self._object_bounds_ids.clear()
            return
        x1, y1 = self.doc_to_canvas(bounds[0], bounds[1])
        x2, y2 = self.doc_to_canvas(bounds[2], bounds[3])
        points = self.object_handle_points((x1, y1, x2, y2))
        dpi_scale = max(1.0, float(self.tk.call("tk", "scaling")))
        radius = max(3, round(3 * dpi_scale))
        if len(self._object_bounds_ids) != 9:
            for item_id in self._object_bounds_ids:
                self.canvas.delete(item_id)
            outline = self.canvas.create_rectangle(x1, y1, x2, y2, outline=TOKENS.ACCENT, width=1, dash=(4, 3))
            handles = [self.canvas.create_rectangle(0, 0, 0, 0, fill=TOKENS.ACCENT, outline=TOKENS.TEXT_PRIMARY, width=1) for _ in points]
            self._object_bounds_ids = [outline, *handles]
        self.canvas.coords(self._object_bounds_ids[0], x1, y1, x2, y2)
        for item_id, (_name, (x, y)) in zip(self._object_bounds_ids[1:], points.items()):
            self.canvas.coords(item_id, x - radius, y - radius, x + radius, y + radius)
            self.canvas.tag_raise(item_id)
        self.canvas.tag_raise(self._object_bounds_ids[0])

    def select_object_at(self, point: tuple[int, int], add: bool = False) -> Layer | None:
        index = topmost_layer_at(self.doc, point, tolerance=max(2, round(5 / max(self.zoom.get(), 0.01))))
        if index is None:
            if not add:
                self.selected_layer_ids.clear()
                self.refresh_layers()
                self.update_object_bounds()
            return None
        layer = self.doc.layers[index]
        self.doc.active_layer = index
        if add:
            self.selected_layer_ids.add(layer.id)
        else:
            self.selected_layer_ids = {layer.id}
        self.refresh_layers()
        self.refresh_properties()
        self.update_object_bounds()
        return layer

    def begin_object_resize(self, handle: str) -> None:
        layer = self.doc.layer
        if layer.kind != "shape" or layer.shape_data is None or layer.locked:
            return
        self._object_resize_handle = handle
        self._object_resize_layer_id = layer.id
        self._object_resize_before = copy.deepcopy(layer.shape_data)
        self._object_resize_rendered_bounds = self.object_document_bounds(layer)
        self._last_object_resize_render = 0.0

    def resize_selected_object_live(self, point: tuple[int, int], state: int) -> None:
        layer = self.doc.get_layer(self._object_resize_layer_id or "")
        if layer is None or layer.shape_data is None or self._object_resize_before is None or self._object_resize_handle is None:
            return
        old_box = shape_data_bounds(self._object_resize_before) or (0, 0, 1, 1)
        document_box = tuple(value + (layer.x if index % 2 == 0 else layer.y) for index, value in enumerate(old_box))
        resized = resize_box_from_handle(
            document_box,
            self._object_resize_handle,
            point,
            keep_proportions=bool(state & 0x0001),
            from_center=bool(state & 0x0008),
        )
        layer.shape_data = transform_shape_data_to_box(
            self._object_resize_before,
            (resized[0] - layer.x, resized[1] - layer.y, resized[2] - layer.x, resized[3] - layer.y),
        )
        self.doc.dirty = True
        self.update_object_bounds()
        now = time.perf_counter()
        if now - self._last_object_resize_render >= 1 / 30:
            old_bounds = self._object_resize_rendered_bounds or document_box
            render_shape_layer(layer)
            new_bounds = self.object_document_bounds(layer) or resized
            self.request_canvas_refresh(union_rect(old_bounds, new_bounds), layer, "pixels")
            self._object_resize_rendered_bounds = new_bounds
            self._last_object_resize_render = now

    def finish_object_resize(self) -> None:
        layer = self.doc.get_layer(self._object_resize_layer_id or "")
        if layer is not None and layer.shape_data is not None and self._object_resize_before is not None and layer.shape_data != self._object_resize_before:
            old_bounds = self._object_resize_rendered_bounds or self.object_document_bounds(layer)
            render_shape_layer(layer)
            new_bounds = self.object_document_bounds(layer)
            if old_bounds is not None and new_bounds is not None:
                self.request_canvas_refresh(union_rect(old_bounds, new_bounds), layer, "pixels")
            self.push_command(ShapeDataCommand("Изменить размер фигуры", layer.id, self._object_resize_before, copy.deepcopy(layer.shape_data), layer.name, layer.name))
        self._object_resize_handle = None
        self._object_resize_layer_id = None
        self._object_resize_before = None
        self._object_resize_rendered_bounds = None
        self._last_object_resize_render = 0.0
        self.refresh_properties()

    def brush_size_changed(self, *_args) -> None:
        if self._last_pointer_event is not None:
            self.update_brush_preview(self._last_pointer_event)
        self.update_clone_source_marker()
        self.quick_preview_settings_changed()

    def quick_preview_settings_changed(self, *_args) -> None:
        if self.tool.get() == "quick_selection" and self._quick_points:
            self.reset_quick_preview_cache()
            self.update_quick_selection_preview(force=True)

    def selection_mode_changed(self, *_args) -> None:
        if self._last_pointer_event is not None and self.tool.get() in {
            "select",
            "ellipse_select",
            "lasso",
            "magnetic_lasso",
            "polygon_lasso",
            "quick_selection",
            "magic_wand",
            "color_range",
        }:
            self.update_brush_preview(self._last_pointer_event)

    def update_brush_preview(self, event) -> None:
        tool = self.tool.get()
        if tool not in self.brush_preview_tools():
            self.clear_brush_preview()
            return
        point = self.canvas_to_doc(event)
        if point[0] < 0 or point[1] < 0 or point[0] >= self.doc.width or point[1] >= self.doc.height:
            self.clear_brush_preview()
            return
        cx, cy = self.doc_to_canvas(point[0] + 0.5, point[1] + 0.5)
        radius = max(1.0, float(self.brush_size.get()) * float(self.zoom.get()))
        mode = self.selection_mode_from_event(event) if tool == "quick_selection" else "replace"
        color_by_mode = {"replace": "#50e3ff", "add": "#59f28a", "subtract": "#ff6262", "intersect": "#ffd166"}
        label_by_mode = {"replace": "new", "add": "+", "subtract": "-", "intersect": "x"}
        outline = color_by_mode.get(mode, "#50e3ff")
        coords = [cx - radius, cy - radius, cx + radius, cy + radius]
        if not self._brush_preview_ids:
            fill_id = self.canvas.create_oval(*coords, outline="", fill=outline, stipple="gray25")
            ring_id = self.canvas.create_oval(*coords, outline=outline, width=2)
            cross_h = self.canvas.create_line(cx - 6, cy, cx + 6, cy, fill=outline, width=1)
            cross_v = self.canvas.create_line(cx, cy - 6, cx, cy + 6, fill=outline, width=1)
            text_id = self.canvas.create_text(cx, cy + radius + 12, text=label_by_mode.get(mode, ""), fill=outline, font=("Segoe UI", 9, "bold"))
            self._brush_preview_ids = [fill_id, ring_id, cross_h, cross_v, text_id]
        else:
            fill_id, ring_id, cross_h, cross_v, text_id = self._brush_preview_ids
            self.canvas.coords(fill_id, *coords)
            self.canvas.coords(ring_id, *coords)
            self.canvas.coords(cross_h, cx - 6, cy, cx + 6, cy)
            self.canvas.coords(cross_v, cx, cy - 6, cx, cy + 6)
            self.canvas.coords(text_id, cx, cy + radius + 12)
        fill_id, ring_id, cross_h, cross_v, text_id = self._brush_preview_ids
        self.canvas.itemconfigure(fill_id, fill=outline)
        self.canvas.itemconfigure(ring_id, outline=outline)
        self.canvas.itemconfigure(cross_h, fill=outline)
        self.canvas.itemconfigure(cross_v, fill=outline)
        self.canvas.itemconfigure(text_id, text=label_by_mode.get(mode, ""), fill=outline, state=tk.NORMAL if tool == "quick_selection" else tk.HIDDEN)
        for item_id in self._brush_preview_ids:
            self.canvas.tag_raise(item_id)

    def clear_brush_preview(self) -> None:
        for item_id in self._brush_preview_ids:
            self.canvas.delete(item_id)
        self._brush_preview_ids.clear()

    def update_quick_selection_preview(self, force: bool = False) -> None:
        if self.tool.get() != "quick_selection" or not self._quick_points:
            self.clear_quick_selection_preview()
            return
        now = time.perf_counter()
        if not force and now - self._last_quick_preview_time < 0.075:
            return
        self._last_quick_preview_time = now
        new_points = self._quick_points[self._quick_preview_processed :]
        if new_points:
            with self.render_engine.profiler.measure("selection.quick_preview"):
                partial = self.doc._quick_selection_mask(
                    self.doc.layer,
                    new_points,
                    max(2, int(self.brush_size.get())),
                    int(self.tolerance.get()),
                    int(self.quick_smooth.get()),
                    int(self.quick_edge_radius.get()),
                    float(self.quick_edge_strength.get()),
                )
            self._quick_preview_mask = partial if self._quick_preview_mask is None else np.maximum(self._quick_preview_mask, partial)
            self._quick_preview_processed = len(self._quick_points)
        mask = self._quick_preview_mask
        current = self._quick_base_selection
        if mask is not None and current is not None:
            if self._quick_mode == "add":
                mask = np.maximum(current, mask)
            elif self._quick_mode == "subtract":
                mask = np.clip(current.astype(np.float32) * (1.0 - mask.astype(np.float32) / 255.0), 0, 255).astype(np.uint8)
            elif self._quick_mode == "intersect":
                mask = np.minimum(current, mask)
        if mask is None:
            mask = current
        if mask is None or not np.any(mask):
            self.clear_quick_selection_preview()
            return
        bounds = self.mask_bounds(mask)
        if bounds is None:
            self.clear_quick_selection_preview()
            return
        x1, y1, x2, y2 = bounds
        mask_image = Image.fromarray(mask[y1:y2, x1:x2], mode="L")
        scale = self.zoom.get()
        if scale != 1.0:
            mask_image = mask_image.resize(
                (max(1, round((x2 - x1) * scale)), max(1, round((y2 - y1) * scale))),
                Image.Resampling.NEAREST,
            )
        color_by_mode = {
            "replace": (45, 205, 255),
            "add": (55, 225, 120),
            "subtract": (255, 80, 80),
            "intersect": (255, 195, 60),
        }
        color = color_by_mode.get(self._quick_mode, color_by_mode["replace"])
        alpha = mask_image.point(lambda value: 78 if value else 0)
        overlay = Image.new("RGBA", mask_image.size, (*color, 0))
        overlay.putalpha(alpha)
        self._quick_preview_image = ImageTk.PhotoImage(overlay)
        ox, oy = self._canvas_origin
        preview_x, preview_y = ox + round(x1 * scale), oy + round(y1 * scale)
        if self._quick_preview_id is None:
            self._quick_preview_id = self.canvas.create_image(preview_x, preview_y, image=self._quick_preview_image, anchor=tk.NW)
        else:
            self.canvas.itemconfigure(self._quick_preview_id, image=self._quick_preview_image)
            self.canvas.coords(self._quick_preview_id, preview_x, preview_y)
        self.canvas.tag_raise(self._quick_preview_id)
        for item_id in self._brush_preview_ids:
            self.canvas.tag_raise(item_id)

    def clear_quick_selection_preview(self) -> None:
        if self._quick_preview_id is not None:
            self.canvas.delete(self._quick_preview_id)
            self._quick_preview_id = None
        self._quick_preview_image = None
        self.reset_quick_preview_cache()

    def reset_quick_preview_cache(self) -> None:
        self._quick_preview_mask = None
        self._quick_preview_processed = 0

    def space_down(self, _event) -> None:
        self._space_down = True
        self.canvas.configure(cursor="fleur")

    def space_up(self, _event) -> None:
        self._space_down = False
        if not self._panning:
            self.canvas.configure(cursor="")

    def pan_down(self, event) -> None:
        self._panning = True
        self.canvas.scan_mark(event.x, event.y)
        self.canvas.configure(cursor="fleur")

    def pan_drag(self, event) -> None:
        if self._panning:
            self.canvas.scan_dragto(event.x, event.y, gain=1)
            if self._last_pointer_event is not None:
                self.update_brush_preview(self._last_pointer_event)

    def pan_up(self, _event) -> None:
        self._panning = False
        self.canvas.configure(cursor="fleur" if self._space_down else "")

    def pointer_down(self, event) -> None:
        if self.tool.get() == "hand" or self._space_down:
            self.pan_down(event)
            return
        self.canvas.focus_set()
        point = self.canvas_to_doc(event)
        self.drag_start = point
        self.last_point = point
        tool = self.tool.get()
        if tool == "crop" and self._crop_box is not None:
            handle = self.crop_handle_at(point)
            if handle is not None:
                self._crop_drag_handle = handle
                self._crop_drag_origin_box = self._crop_box
                self.status_text("Перетащите маркер кадрирования")
            else:
                x1, y1, x2, y2 = self._crop_box
                if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
                    self.drag_start = None
                    return
        if tool.endswith("_shape"):
            self._shape_drag_options = self.current_shape_options(tool)
            self.clear_selection_overlay()
        if tool in ["brush", "eraser", "blur_tool", "sharpen_tool", "dodge", "burn", "clone", "healing", "spot_healing"]:
            if tool in ["clone", "healing"]:
                if event.state & 0x0008:
                    self.set_clone_source(point)
                    return
                if self._source_anchor.point is None:
                    self.status_text("Сначала задайте источник: Alt + левый клик")
                    self.drag_start = None
                    return
                self._source_anchor.aligned = bool(self.clone_aligned.get())
                self._source_anchor.sampling = self.clone_sampling.get()
                self._source_anchor.begin_stroke(point)
                self._clone_anchor_target = self._source_anchor.stroke_target
                self._clone_anchor_source = self._source_anchor.stroke_source
                self.prepare_clone_sample()
            kind = "mask" if tool in ["brush", "eraser"] and self.paint_target.get() == "mask" else "pixels"
            self.begin_stroke(kind)
            self.paint_at(point)
        elif tool == "fill":
            self.run_pixel_delta_command("Fill", lambda: flood_fill(self.doc.layer, point[0], point[1], self.foreground, int(self.tolerance.get()), self.doc.layer_selection_mask(self.doc.layer)))
        elif tool == "magic_wand":
            mode = self.selection_mode_from_event(event)
            contiguous = bool(self.magic_contiguous.get())
            self.run_selection_command("Magic wand selection", lambda: self.doc.magic_wand_selection(self.doc.layer, point[0], point[1], int(self.tolerance.get()), mode, contiguous))
        elif tool == "color_range":
            mode = self.selection_mode_from_event(event)
            lx, ly = point[0] - self.doc.layer.x, point[1] - self.doc.layer.y
            if 0 <= lx < self.doc.layer.pixels.shape[1] and 0 <= ly < self.doc.layer.pixels.shape[0]:
                sample = tuple(int(value) for value in self.doc.layer.pixels[ly, lx])
                self.color_range_sample_hex.set(self.color_hex(sample).upper())
                if hasattr(self, "tool_options_panel"):
                    self.tool_options_panel.render()
            self.run_selection_command("Color range selection", lambda: self.doc.color_range_selection(self.doc.layer, point[0], point[1], int(self.tolerance.get()), mode))
            self.status_text(f"Диапазон цвета {self.color_range_sample_hex.get()}, допуск {int(self.tolerance.get())}")
        elif tool == "quick_selection":
            self._quick_points = [point]
            self._quick_mode = self.selection_mode_from_event(event)
            self._quick_base_selection = None if self.doc.selection_mask is None else self.doc.selection_mask.copy()
            self.reset_quick_preview_cache()
            self.update_quick_selection_preview(force=True)
            self.status_text("Кисть быстрого выделения")
        elif tool == "patch":
            self.begin_patch_drag(point)
        elif tool == "eyedropper":
            self.pick_color_from_document(point)
        elif tool == "text":
            if self._text_editor is not None:
                self.finish_text_edit()
            self.status_text("Клик создает строку текста, перетаскивание создает текстовый блок")
        elif tool == "move":
            handle = self.object_handle_at(point)
            if handle is not None:
                self.begin_object_resize(handle)
                return
            layer = self.select_object_at(point, add=bool(event.state & 0x0001)) if self.auto_select.get() else self.doc.layer
            if layer is None and self.doc.layers and self.doc.layer.kind not in {"shape", "text", "adjustment"}:
                candidate = self.doc.layer
                if layer_contains_point(candidate, point, 0):
                    layer = candidate
            if layer is None or not layer_contains_point(layer, point, max(2, round(5 / max(self.zoom.get(), 0.01)))):
                self.drag_start = None
                self.status_text("На этом месте нет редактируемого объекта")
                return
            if layer.locked:
                self.status_text("Объект выбран, но слой заблокирован")
                self.drag_start = None
                return
            self._move_layer_id = layer.id
            self._move_start = (layer.x, layer.y)
            self._move_start_mask = None if layer.mask is None else layer.mask.copy()
            self._move_last_bounds = self.layer_render_bounds(layer)
        elif tool == "polygon_lasso":
            self._polygon_points.append(point)
            self.draw_polygon_lasso()
        elif tool == "lasso":
            self._lasso_points = [point]
        elif tool == "magnetic_lasso":
            self._magnetic_edges = self.doc.magnetic_edge_map(self.render_engine.render(self.doc, False))
            snapped = self.doc.snap_point_to_edge(point, self._magnetic_edges, max(8, int(self.tolerance.get())))
            self._lasso_points = [snapped]
            self.status_text(f"Магнитное лассо: {snapped[0]}, {snapped[1]}")

    def pointer_drag(self, event) -> None:
        self._last_pointer_event = event
        self.update_brush_preview(event)
        if self._panning:
            self.pan_drag(event)
            return
        point = self.canvas_to_doc(event)
        tool = self.tool.get()
        if tool in ["brush", "eraser", "blur_tool", "sharpen_tool", "dodge", "burn", "clone", "healing", "spot_healing"]:
            self.paint_line(self.last_point or point, point)
            self.last_point = point
        elif tool == "move" and self._object_resize_handle is not None:
            self.resize_selected_object_live(point, event.state)
        elif tool == "move" and self.drag_start:
            dx, dy = point[0] - self.drag_start[0], point[1] - self.drag_start[1]
            if dx or dy:
                layer = self.doc.get_layer(self._move_layer_id or "")
                if layer is not None:
                    old_bounds = self._move_last_bounds or self.layer_render_bounds(layer)
                    self.doc.move_active_layer(dx, dy)
                    new_bounds = self.layer_render_bounds(layer)
                    dirty = union_rect(old_bounds, new_bounds)
                    self._move_last_bounds = new_bounds
                    self.drag_start = point
                    self.request_canvas_refresh(dirty, layer, "transform")
                    self.update_object_bounds()
        elif tool in ["select", "ellipse_select", "crop", "gradient", "text", "rect_shape", "ellipse_shape", "line_shape", "bezier_shape", "polygon_shape", "star_shape", "custom_shape"]:
            self.draw_selection(self.drag_start, point, event.state)
            if tool == "gradient" and self.drag_start:
                self.update_gradient_preview(self.drag_start, point)
        elif tool == "lasso" and self.drag_start:
            self._lasso_points.append(point)
            self.draw_lasso()
        elif tool == "magnetic_lasso" and self.drag_start:
            snapped = self.magnetic_lasso_point(point)
            if not self._lasso_points or (snapped[0] - self._lasso_points[-1][0]) ** 2 + (snapped[1] - self._lasso_points[-1][1]) ** 2 >= 4:
                self._lasso_points.append(snapped)
                self.draw_lasso()
        elif tool == "quick_selection" and self.drag_start:
            spacing = max(1, int(self.brush_size.get()) // 2)
            previous = self._quick_points[-1]
            if (point[0] - previous[0]) ** 2 + (point[1] - previous[1]) ** 2 >= spacing ** 2:
                self._quick_points.append(point)
                self.update_quick_selection_preview()
            self.status_text(f"Точек быстрого выделения: {len(self._quick_points)}")
        elif tool == "patch" and self.drag_start:
            self.draw_patch_preview(point)

    def pointer_up(self, event) -> None:
        if self._panning:
            self.pan_up(event)
            return
        point = self.canvas_to_doc(event)
        tool = self.tool.get()
        if tool in ["brush", "eraser", "blur_tool", "sharpen_tool", "dodge", "burn", "clone", "healing", "spot_healing"]:
            self.end_stroke(f"{tool.title()} stroke")
            self._source_anchor.end_stroke()
            self._clone_anchor_target = None
            self._clone_anchor_source = None
            self._clone_sample_pixels = None
        elif tool == "move":
            if self._object_resize_handle is not None:
                self.finish_object_resize()
            else:
                self.end_move_layer()
        elif tool == "gradient" and self.drag_start:
            if self.gradient_mode.get() == "Объект":
                self.create_gradient_object(self.drag_start, point)
                self.refresh()
            else:
                kind = self.current_gradient_kind()
                stops = self.current_gradient_stops()
                self.run_pixel_delta_command(
                    "Gradient",
                    lambda: apply_gradient(
                        self.doc.layer,
                        (*self.drag_start, *point),
                        self.foreground,
                        self.background,
                        self.doc.layer_selection_mask(self.doc.layer),
                        kind,
                        stops,
                    ),
                )
            self.clear_gradient_preview()
        elif tool == "text" and self.drag_start:
            self.begin_text_editor(self.drag_start, point)
            self.clear_crop_overlay()
        elif tool in ["rect_shape", "ellipse_shape", "line_shape", "bezier_shape", "polygon_shape", "star_shape", "custom_shape"] and self.drag_start:
            geometry = self.shape_geometry_for_drag(tool, self.drag_start, point, event.state)
            if shape_drag_is_meaningful(geometry):
                self.create_shape_from_drag(tool, geometry)
                self.refresh()
            else:
                self.update_selection_overlay()
        elif tool in ["select", "ellipse_select", "crop"] and self.drag_start:
            self.selection_box = (*self.drag_start, *point)
            if tool == "select":
                mode = self.selection_mode_from_event(event)
                feather = int(self.selection_feather.get())
                self.run_selection_command("Rectangular selection", lambda: self.doc.set_rect_selection(self.selection_box, mode, feather))
            elif tool == "ellipse_select":
                mode = self.selection_mode_from_event(event)
                feather = int(self.selection_feather.get())
                antialias = bool(self.selection_antialias.get())
                self.run_selection_command("Elliptical selection", lambda: self.doc.set_ellipse_selection(self.selection_box, mode, feather, antialias))
            elif tool == "crop":
                if self._crop_drag_handle is None:
                    self._crop_box = self.crop_box_for_drag(self.drag_start, point)
                self.draw_crop_overlay(self._crop_box)
                self.status_text("Кадрирование готово: Enter или двойной клик применяет, Escape отменяет")
            self.draw_selection(self.drag_start, point, event.state)
        elif tool == "lasso" and len(self._lasso_points) >= 3:
            mode = self.selection_mode_from_event(event)
            points = list(self._lasso_points)
            feather = int(self.selection_feather.get())
            antialias = bool(self.selection_antialias.get())
            self.run_selection_command("Lasso selection", lambda: self.doc.set_polygon_selection(points, mode, feather, antialias))
            self.clear_lasso_overlay()
        elif tool == "magnetic_lasso":
            if self.drag_start:
                self._lasso_points.append(self.magnetic_lasso_point(point))
            if len(self._lasso_points) >= 3:
                mode = self.selection_mode_from_event(event)
                points = list(self._lasso_points)
                feather = int(self.selection_feather.get())
                antialias = bool(self.selection_antialias.get())
                self.run_selection_command("Magnetic lasso selection", lambda: self.doc.set_polygon_selection(points, mode, feather, antialias))
            self.clear_lasso_overlay()
            self._magnetic_edges = None
        elif tool == "quick_selection" and self._quick_points:
            points = list(self._quick_points)
            mode = self._quick_mode
            radius = max(2, int(self.brush_size.get()))
            tolerance = int(self.tolerance.get())
            smooth = int(self.quick_smooth.get())
            edge_radius = int(self.quick_edge_radius.get())
            edge_strength = float(self.quick_edge_strength.get())
            self._quick_points.clear()
            self.clear_quick_selection_preview()
            self.run_selection_command(
                "Quick selection",
                lambda: self.doc.quick_selection_brush(
                    self.doc.layer,
                    points,
                    radius,
                    tolerance,
                    mode,
                    smooth,
                    edge_radius,
                    edge_strength,
                ),
            )
        elif tool == "patch" and self.drag_start:
            self.finish_patch_drag(point)
        if tool != "crop":
            self.clear_drag_preview()
        self.drag_start = None
        self.last_point = None
        self._shape_drag_options = None
        self._crop_drag_handle = None
        self._crop_drag_origin_box = None

    def pointer_double_click(self, event) -> None:
        if self.tool.get() == "move":
            point = self.canvas_to_doc(event)
            layer = self.select_object_at(point)
            if layer is not None and layer.kind == "text":
                self.edit_active_text_on_canvas()
            return
        if self.tool.get() == "crop" and self._crop_box is not None:
            point = self.canvas_to_doc(event)
            x1, y1, x2, y2 = self._crop_box
            if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
                self.apply_crop_overlay()
            return
        if self.tool.get() == "polygon_lasso" and len(self._polygon_points) >= 3:
            mode = self.selection_mode_from_event(event)
            points = list(self._polygon_points)
            feather = int(self.selection_feather.get())
            antialias = bool(self.selection_antialias.get())
            self.run_selection_command("Polygon lasso selection", lambda: self.doc.set_polygon_selection(points, mode, feather, antialias))
            self.clear_lasso_overlay()

    def begin_stroke(self, kind: str = "pixels") -> None:
        self._edit_generation += 1
        self._stroke_layer_id = self.doc.layer.id
        self._stroke_kind = kind
        if kind == "mask" and self.doc.layer.mask is None:
            self.doc.layer.mask = np.full(self.doc.layer.pixels.shape[:2], 255, dtype=np.uint8)
            self.doc.layer.mask_enabled = True
        self._stroke_rect = None
        self._stroke_before = None
        self._stroke_tiles = {}
        self._stroke_selection_mask = self.doc.layer_selection_mask(self.doc.layer)
        tool = self.tool.get()
        if kind == "pixels" and tool in {"blur_tool", "sharpen_tool", "dodge", "burn"}:
            mode = "blur" if tool == "blur_tool" else "sharpen" if tool == "sharpen_tool" else tool
            strength = float(self.exposure.get()) if tool in {"dodge", "burn"} else float(self.retouch_strength.get())
            self._retouch_stroke = RetouchStroke(
                self.doc.layer,
                mode,
                int(self.brush_size.get()),
                float(self.hardness.get()),
                strength,
                self.tonal_range.get(),
                self._stroke_selection_mask,
            )
        else:
            self._retouch_stroke = None

    def brush_local_rect(self, point: tuple[int, int]) -> tuple[int, int, int, int] | None:
        layer = self.doc.layer
        radius = int(self.brush_size.get())
        if self.tool.get() == "spot_healing":
            radius *= 2
        lx, ly = point[0] - layer.x, point[1] - layer.y
        if lx < -radius or ly < -radius or lx >= layer.pixels.shape[1] + radius or ly >= layer.pixels.shape[0] + radius:
            return None
        return (
            max(0, lx - radius),
            max(0, ly - radius),
            min(layer.pixels.shape[1], lx + radius + 1),
            min(layer.pixels.shape[0], ly + radius + 1),
        )

    def capture_stroke_before(self, rect: tuple[int, int, int, int] | None) -> None:
        if rect is None:
            return
        if self._retouch_stroke is not None:
            return
        layer = self.doc.layer
        target = layer.mask if self._stroke_kind == "mask" else layer.pixels
        if target is None:
            return
        x1, y1, x2, y2 = rect
        tile_size = 128
        self._stroke_rect = union_rect(self._stroke_rect, rect)
        for ty in range(y1 // tile_size, (y2 - 1) // tile_size + 1):
            for tx in range(x1 // tile_size, (x2 - 1) // tile_size + 1):
                key = tx, ty
                if key in self._stroke_tiles:
                    continue
                px1, py1 = tx * tile_size, ty * tile_size
                px2, py2 = min(target.shape[1], px1 + tile_size), min(target.shape[0], py1 + tile_size)
                tile_rect = px1, py1, px2, py2
                self._stroke_tiles[key] = tile_rect, target[py1:py2, px1:px2].copy()

    def end_stroke(self, label: str) -> None:
        if self._retouch_stroke is not None:
            self._stroke_tiles = self._retouch_stroke.before_tiles
        if self._stroke_layer_id and self._stroke_tiles:
            layer = self.doc.get_layer(self._stroke_layer_id)
            if layer is not None:
                target = layer.mask if self._stroke_kind == "mask" else layer.pixels
                patches: list[TilePatch] = []
                if target is not None:
                    for rect, before in self._stroke_tiles.values():
                        x1, y1, x2, y2 = rect
                        after = target[y1:y2, x1:x2].copy()
                        if not np.array_equal(before, after):
                            patches.append(TilePatch(rect, before, after))
                if self._stroke_kind == "mask" and layer.mask is not None:
                    self.push_command(MaskTilePatchCommand(label, self._stroke_layer_id, patches))
                elif self._stroke_kind == "pixels":
                    self.push_command(PixelTilePatchCommand(label, self._stroke_layer_id, patches))
        self._stroke_layer_id = None
        self._stroke_kind = "pixels"
        self._stroke_rect = None
        self._stroke_before = None
        self._stroke_tiles = {}
        self._stroke_selection_mask = None
        self._retouch_stroke = None
        if self._editor_active:
            self.refresh_layer_previews()

    def end_move_layer(self) -> None:
        if self._move_layer_id and self._move_start:
            layer = self.doc.get_layer(self._move_layer_id)
            if layer is not None:
                end = (layer.x, layer.y)
                if end != self._move_start:
                    after_mask = None if layer.mask is None else layer.mask.copy()
                    self.push_command(LayerMoveCommand("Move layer", self._move_layer_id, self._move_start, end, self._move_start_mask, after_mask))
        self._move_layer_id = None
        self._move_start = None
        self._move_start_mask = None
        self._move_last_bounds = None

    def paint_at(self, point: tuple[int, int]) -> None:
        self.capture_stroke_before(self.brush_local_rect(point))
        tool = self.tool.get()
        selection_mask = self._stroke_selection_mask
        changed = None
        if tool == "spot_healing":
            changed = spot_heal(
                self.doc.layer,
                point[0],
                point[1],
                int(self.brush_size.get()),
                float(self.retouch_strength.get()),
                selection_mask,
                float(self.hardness.get()),
            )
        elif tool in ["clone", "healing"]:
            source = self.clone_source_for_point(point)
            if source is not None:
                amount = float(self.opacity.get()) if tool == "clone" else float(self.retouch_strength.get())
                changed = clone_or_heal(
                    self.doc.layer,
                    source[0],
                    source[1],
                    point[0],
                    point[1],
                    int(self.brush_size.get()),
                    amount,
                    tool == "healing",
                    selection_mask,
                    float(self.hardness.get()),
                    self._clone_sample_pixels,
                    self._clone_sample_origin,
                )
        elif self._stroke_kind == "mask":
            changed = draw_mask_brush(self.doc.layer, point[0], point[1], int(self.brush_size.get()), 0 if tool == "eraser" else 255, float(self.opacity.get()), selection_mask)
        elif tool in ["blur_tool", "sharpen_tool", "dodge", "burn"]:
            if self._retouch_stroke is not None:
                changed = self._retouch_stroke.dab(point[0], point[1])
        else:
            changed = draw_brush(
                self.doc.layer,
                point[0],
                point[1],
                int(self.brush_size.get()),
                self.foreground,
                float(self.opacity.get()),
                tool == "eraser",
                selection_mask,
            )
        self.doc.dirty = True
        if changed is not None:
            rect = self.local_to_document_rect(changed, self.doc.layer)
            self.request_canvas_refresh(rect, self.doc.layer, self._stroke_kind)
        elif tool in {"clone", "healing"}:
            self.status_text("Источник и цель не пересекают доступные пиксели")

    def paint_line(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        radius = max(1, int(self.brush_size.get()))
        distance = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
        tool = self.tool.get()
        spacing = radius * (0.9 if tool == "spot_healing" else 0.45)
        steps = max(1, int(np.ceil(distance / max(1.0, spacing))))
        selection_mask = self._stroke_selection_mask
        opacity = float(self.opacity.get())
        changed_rect = None
        for i in range(1, steps + 1):
            t = i / steps
            x = round(start[0] * (1 - t) + end[0] * t)
            y = round(start[1] * (1 - t) + end[1] * t)
            self.capture_stroke_before(self.brush_local_rect((x, y)))
            changed = None
            if tool == "spot_healing":
                changed = spot_heal(self.doc.layer, x, y, radius, float(self.retouch_strength.get()), selection_mask, float(self.hardness.get()))
            elif tool in ["clone", "healing"]:
                source = self.clone_source_for_point((x, y))
                if source is not None:
                    amount = opacity if tool == "clone" else float(self.retouch_strength.get())
                    changed = clone_or_heal(
                        self.doc.layer, source[0], source[1], x, y, radius, amount,
                        tool == "healing", selection_mask, float(self.hardness.get()),
                        self._clone_sample_pixels, self._clone_sample_origin,
                    )
            elif self._stroke_kind == "mask":
                changed = draw_mask_brush(self.doc.layer, x, y, radius, 0 if tool == "eraser" else 255, opacity, selection_mask)
            elif tool in ["blur_tool", "sharpen_tool", "dodge", "burn"]:
                if self._retouch_stroke is not None:
                    changed = self._retouch_stroke.dab(x, y)
            else:
                changed = draw_brush(
                    self.doc.layer,
                    x,
                    y,
                    radius,
                    self.foreground,
                    opacity,
                    tool == "eraser",
                    selection_mask,
                )
            changed_rect = union_rect(changed_rect, changed)
        self.doc.dirty = True
        if changed_rect is not None:
            rect = self.local_to_document_rect(changed_rect, self.doc.layer)
            self.request_canvas_refresh(rect, self.doc.layer, self._stroke_kind)
        elif tool in {"clone", "healing"}:
            self.status_text("Источник и цель не пересекают доступные пиксели")

    @staticmethod
    def local_to_document_rect(rect: tuple[int, int, int, int], layer) -> tuple[int, int, int, int]:
        return rect[0] + layer.x, rect[1] + layer.y, rect[2] + layer.x, rect[3] + layer.y

    def clone_source_for_point(self, point: tuple[int, int]) -> tuple[int, int] | None:
        return self._source_anchor.source_for(point)

    def clone_source_click(self, event) -> str:
        if self.tool.get() not in {"clone", "healing"}:
            return "break"
        self.canvas.focus_set()
        self.set_clone_source(self.canvas_to_doc(event))
        return "break"

    def set_clone_source(self, point: tuple[int, int]) -> None:
        if point[0] < 0 or point[1] < 0 or point[0] >= self.doc.width or point[1] >= self.doc.height:
            self.status_text("Источник должен находиться внутри холста")
            return
        self._source_anchor.set_source(point)
        self._clone_source = point
        self.drag_start = None
        self.last_point = None
        self.update_clone_source_marker()
        self.status_text(f"Источник выбран: {point[0]}, {point[1]}. Теперь проведите кистью по цели.")

    def prepare_clone_sample(self) -> None:
        mode = self.clone_sampling.get()
        if mode == "Текущий слой":
            self._clone_sample_pixels = self.doc.layer.pixels.copy()
            self._clone_sample_origin = (self.doc.layer.x, self.doc.layer.y)
            return
        temporary = copy.copy(self.doc)
        if mode == "Текущий и ниже":
            temporary.layers = list(self.doc.layers[: self.doc.active_layer + 1])
            temporary.active_layer = len(temporary.layers) - 1
        else:
            temporary.layers = list(self.doc.layers)
        self._clone_sample_pixels = temporary.composite(False).copy()
        self._clone_sample_origin = (0, 0)

    def update_clone_source_marker(self) -> None:
        for item_id in self._clone_source_marker_ids:
            self.canvas.delete(item_id)
        self._clone_source_marker_ids.clear()
        if self._source_anchor.point is None or self.tool.get() not in {"clone", "healing"}:
            return
        cx, cy = self.doc_to_canvas(*self._source_anchor.point)
        radius = max(4.0, float(self.brush_size.get()) * float(self.zoom.get()))
        self._clone_source_marker_ids = [
            self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, outline="#ffb000", dash=(5, 3), width=2),
            self.canvas.create_line(cx - 7, cy, cx + 7, cy, fill="#ffb000", width=2),
            self.canvas.create_line(cx, cy - 7, cx, cy + 7, fill="#ffb000", width=2),
        ]
        for item_id in self._clone_source_marker_ids:
            self.canvas.tag_raise(item_id)

    def begin_patch_drag(self, point: tuple[int, int]) -> None:
        if self.doc.layer.locked:
            self.status_text("Слой заблокирован")
            self.drag_start = None
            return
        if self.doc.selection_mask is None or not np.any(self.doc.selection_mask):
            self.status_text("Сначала создайте выделение для инструмента Заплатка")
            self.drag_start = None
            return
        x, y = point
        if x < 0 or y < 0 or x >= self.doc.width or y >= self.doc.height or self.doc.selection_mask[y, x] == 0:
            self.status_text("Начните перетаскивание внутри активного выделения")
            self.drag_start = None
            return
        self._patch_start_bounds = self.doc.selection_bounds()
        self.draw_patch_preview(point)
        self.status_text("Перетащите выделение на область-источник")

    def patch_source_bounds_for_point(self, point: tuple[int, int]) -> tuple[int, int, int, int] | None:
        if self.drag_start is None or self._patch_start_bounds is None:
            return None
        dx = point[0] - self.drag_start[0]
        dy = point[1] - self.drag_start[1]
        x1, y1, x2, y2 = self._patch_start_bounds
        return x1 + dx, y1 + dy, x2 + dx, y2 + dy

    def patch_source_in_active_layer(self, bounds: tuple[int, int, int, int]) -> bool:
        layer = self.doc.layer
        x1, y1, x2, y2 = bounds
        return x1 >= layer.x and y1 >= layer.y and x2 <= layer.x + layer.pixels.shape[1] and y2 <= layer.y + layer.pixels.shape[0]

    def draw_patch_preview(self, point: tuple[int, int]) -> None:
        bounds = self.patch_source_bounds_for_point(point)
        if bounds is None:
            return
        x1, y1 = self.doc_to_canvas(bounds[0], bounds[1])
        x2, y2 = self.doc_to_canvas(bounds[2], bounds[3])
        coords = [x1, y1, x2, y2]
        valid = self.patch_source_in_active_layer(bounds)
        color = "#ffb000" if valid else "#ff4a4a"
        if self._patch_preview_id is None:
            self._patch_preview_id = self.canvas.create_rectangle(*coords, outline=color, dash=(6, 3), width=2)
        else:
            self.canvas.coords(self._patch_preview_id, *coords)
            self.canvas.itemconfigure(self._patch_preview_id, outline=color)
        self.canvas.tag_raise(self._patch_preview_id)

    def clear_patch_preview(self) -> None:
        if self._patch_preview_id is not None:
            self.canvas.delete(self._patch_preview_id)
            self._patch_preview_id = None
        self._patch_start_bounds = None

    def finish_patch_drag(self, point: tuple[int, int]) -> None:
        bounds = self.patch_source_bounds_for_point(point)
        if bounds is None:
            self.clear_patch_preview()
            return
        if not self.patch_source_in_active_layer(bounds):
            self.status_text("Источник заплатки должен полностью попадать в активный слой")
            self.clear_patch_preview()
            return
        source_x, source_y = bounds[0], bounds[1]
        self.run_document_command("Интерактивная заплатка", lambda: self.doc.patch_active_selection(source_x, source_y, True))
        self.clear_patch_preview()
        self.refresh()

    def current_shape_options(self, tool: str) -> dict:
        custom_name = self.custom_shape_preset.get()
        custom_points = CUSTOM_SHAPE_PRESETS.get(custom_name, next(iter(CUSTOM_SHAPE_PRESETS.values())))
        line_shape = tool in {"line_shape", "bezier_shape"}
        return {
            "fill": self.foreground,
            "stroke": self.foreground if line_shape else self.background,
            "stroke_width": max(1 if line_shape else 0, int(self.shape_stroke_width.get())),
            "sides": max(3, min(64, int(self.polygon_sides.get() if tool == "polygon_shape" else self.star_points_count.get()))),
            "inner_ratio": float(np.clip(self.star_inner_ratio.get(), 0.05, 0.95)),
            "custom_points": custom_points,
        }

    def shape_geometry_for_drag(self, tool: str, start: tuple[int, int], end: tuple[int, int], state: int = 0) -> dict:
        options = self._shape_drag_options or self.current_shape_options(tool)
        return shape_geometry_from_drag(
            tool,
            start,
            end,
            shift=bool(state & 0x0001),
            alt=bool(state & 0x0008),
            sides=int(options["sides"]),
            inner_ratio=float(options["inner_ratio"]),
            custom_points=options["custom_points"],
        )

    @staticmethod
    def color_hex(color: tuple[int, int, int, int]) -> str:
        return "#{:02x}{:02x}{:02x}".format(*color[:3])

    def draw_selection(self, start: tuple[int, int] | None, end: tuple[int, int], state: int = 0) -> None:
        if not start:
            return
        tool = self.tool.get()
        if tool == "crop":
            if self._crop_drag_handle is not None and self._crop_drag_origin_box is not None:
                self._crop_box = self.resize_crop_box(self._crop_drag_origin_box, self._crop_drag_handle, end)
            else:
                self._crop_box = self.crop_box_for_drag(start, end)
            self.draw_crop_overlay(self._crop_box)
            return
        is_shape = tool.endswith("_shape")
        if is_shape:
            shape_options = self._shape_drag_options or self.current_shape_options(tool)
            geometry = self.shape_geometry_for_drag(tool, start, end, state)
            shape = str(geometry["shape"])
            fill = self.color_hex(shape_options["fill"])
            outline = self.color_hex(shape_options["stroke"])
            stroke_width = int(shape_options["stroke_width"])
            width = max(1, round(stroke_width * self.zoom.get()))
            visible_outline = outline if stroke_width > 0 else ""
            x1, y1, x2, y2 = geometry["box"]
            canvas_box = (*self.doc_to_canvas(x1, y1), *self.doc_to_canvas(x2, y2))
            if shape == "ellipse":
                self.update_drag_preview_item("oval", canvas_box, fill=fill, outline=visible_outline, width=width)
            elif shape == "line":
                line = geometry["line"]
                coords = (*self.doc_to_canvas(line[0], line[1]), *self.doc_to_canvas(line[2], line[3]))
                self.update_drag_preview_item("line", coords, fill=outline, width=width)
            elif shape == "bezier":
                curve = [value for point in geometry["points"] for value in self.doc_to_canvas(point[0], point[1])]
                self.update_drag_preview_item("line", curve, fill=outline, width=width, smooth=True)
            elif shape in {"polygon", "star", "custom"}:
                polygon = [value for point in geometry["points"] for value in self.doc_to_canvas(point[0], point[1])]
                self.update_drag_preview_item("polygon", polygon, fill=fill, outline=visible_outline, width=width)
            else:
                self.update_drag_preview_item("rectangle", canvas_box, fill=fill, outline=visible_outline, width=width)
        else:
            x1, y1 = self.doc_to_canvas(start[0], start[1])
            x2, y2 = self.doc_to_canvas(end[0], end[1])
            coords = [x1, y1, x2, y2]
            if tool == "ellipse_select":
                self.update_drag_preview_item("oval", coords, outline="#50e3ff", dash=(5, 4), width=2)
            elif tool == "gradient":
                self.update_drag_preview_item("line", coords, fill="#50e3ff", width=2, arrow=tk.LAST)
            else:
                self.update_drag_preview_item("rectangle", coords, outline="#50e3ff", dash=(5, 4), width=2)
        for item_id in self._drag_preview_ids:
            self.canvas.tag_raise(item_id)

    def update_drag_preview_item(self, kind: str, coords, **options) -> None:
        item_id = self._drag_preview_ids[0] if self._drag_preview_ids else None
        if item_id is not None and self.canvas.type(item_id) != kind:
            self.clear_drag_preview()
            item_id = None
        if item_id is None:
            creator = {
                "oval": self.canvas.create_oval,
                "line": self.canvas.create_line,
                "polygon": self.canvas.create_polygon,
                "rectangle": self.canvas.create_rectangle,
            }[kind]
            item_id = creator(*coords, **options)
            self._drag_preview_ids = [item_id]
        else:
            self.canvas.coords(item_id, *coords)
            self.canvas.itemconfigure(item_id, **options)

    def clear_drag_preview(self) -> None:
        for item_id in self._drag_preview_ids:
            self.canvas.delete(item_id)
        self._drag_preview_ids.clear()

    def current_gradient_kind(self) -> str:
        return {
            "Линейный": "linear",
            "Радиальный": "radial",
            "Отраженный": "reflected",
            "Ромб": "diamond",
            "Угловой": "angular",
        }.get(self.gradient_type.get(), "linear")

    def begin_text_editor(self, start: tuple[int, int], end: tuple[int, int], layer_id: str | None = None) -> None:
        self.cancel_text_edit()
        existing = self.doc.get_layer(layer_id) if layer_id else None
        if existing is not None and existing.text_data is not None:
            data = existing.text_data
            x, y = int(data.get("x", start[0])), int(data.get("y", start[1]))
            box_width = max(0, int(data.get("box_width", 0)))
            initial_text = str(data.get("text", ""))
            self._text_editor_before = copy.deepcopy(data)
            self._text_editor_layer_id = existing.id
            self.load_text_properties_from_layer(existing)
        else:
            x, y = int(start[0]), int(start[1])
            box_width = abs(int(end[0]) - x) if abs(int(end[0]) - x) >= 6 else 0
            initial_text = ""
            self._text_editor_before = None
            self._text_editor_layer_id = None
            self.text_box_width.set(box_width)
        self._text_editor_origin = (x, y)
        self._text_editor_box_width = box_width
        visible_width = box_width or min(520, max(260, self.doc.width - x))
        visible_height = max(72, abs(int(end[1]) - int(start[1])) if box_width else int(self.text_size.get()) * 2)
        editor = tk.Text(
            self.canvas,
            wrap=tk.WORD if box_width else tk.NONE,
            width=max(8, round(visible_width / max(7, int(self.text_size.get()) * 0.55))),
            height=max(2, round(visible_height / max(14, int(self.text_size.get()) * 1.2))),
            undo=True,
            borderwidth=2,
            relief=tk.SOLID,
            padx=4,
            pady=3,
        )
        editor.insert("1.0", initial_text)
        cx, cy = self.doc_to_canvas(x, y)
        self._text_editor_window = self.canvas.create_window(cx, cy, window=editor, anchor=tk.NW)
        self._text_editor = editor
        self.update_text_editor_style()
        editor.focus_set()
        editor.mark_set(tk.INSERT, tk.END)
        self.status_text("Введите текст на холсте. Кнопка 'Готово' завершает редактирование.")

    def edit_active_text_on_canvas(self) -> None:
        layer = self.doc.layer
        if layer.kind != "text" or layer.text_data is None:
            self.status_text("Сначала выберите текстовый слой")
            return
        if layer.x or layer.y:
            layer.text_data["x"] = int(layer.text_data.get("x", 0)) + layer.x
            layer.text_data["y"] = int(layer.text_data.get("y", 0)) + layer.y
            layer.x = 0
            layer.y = 0
            render_text_layer(layer)
            layer.touch_pixels()
        x = int(layer.text_data.get("x", 0))
        y = int(layer.text_data.get("y", 0))
        width = max(240, int(layer.text_data.get("box_width", 0)))
        self.begin_text_editor((x, y), (x + width, y + int(layer.text_data.get("size", 48)) * 3), layer.id)

    def update_text_editor_style(self) -> None:
        editor = self._text_editor
        if editor is None:
            return
        styles = []
        if self.text_bold.get():
            styles.append("bold")
        if self.text_italic.get():
            styles.append("italic")
        font = tkfont.Font(
            family=self.text_font_family.get() or "Arial",
            size=max(8, round(int(self.text_size.get()) * float(self.zoom.get()) * 0.75)),
            weight="bold" if "bold" in styles else "normal",
            slant="italic" if "italic" in styles else "roman",
            underline=bool(self.text_underline.get()),
        )
        editor.configure(font=font, foreground=self.color_hex(self.foreground), insertbackground=self.color_hex(self.foreground))
        editor._photoredactor_font = font
        editor.tag_configure("paragraph", justify=self.text_align.get(), spacing3=max(0, int(self.text_line_spacing.get())))
        editor.tag_add("paragraph", "1.0", tk.END)

    def finish_text_edit(self) -> None:
        editor = self._text_editor
        if editor is None:
            return
        text = editor.get("1.0", "end-1c")
        layer_id = self._text_editor_layer_id
        origin = self._text_editor_origin
        box_width = max(0, int(self.text_box_width.get() or self._text_editor_box_width))
        before = copy.deepcopy(self._text_editor_before)
        self._destroy_text_editor()
        if layer_id is not None:
            layer = self.doc.get_layer(layer_id)
            if layer is None or before is None:
                return
            self.doc.active_layer = self.doc.layers.index(layer)
            self.apply_text_values(layer, text, origin, box_width)
            after = copy.deepcopy(layer.text_data or {})
            if before != after:
                self.push_command(TextDataCommand("Edit text", layer.id, before, after, f"Text: {str(before.get('text', ''))[:24]}", layer.name))
        elif text:
            layer = self.doc.add_text_layer(
                text, origin[0], origin[1], self.foreground, int(self.text_size.get()),
                self.text_font_family.get(), box_width, self.text_align.get(),
                int(self.text_line_spacing.get()), int(self.text_tracking.get()),
                bool(self.text_bold.get()), bool(self.text_italic.get()), bool(self.text_underline.get()),
                rotation=float(self.text_rotation.get()),
            )
            self.selected_layer_ids = {layer.id}
            self.push_command(LayerInsertCommand("Text layer", self.doc.active_layer, copy.deepcopy(layer)))
        self.refresh()

    def cancel_text_edit(self) -> None:
        if self._text_editor is not None:
            self._destroy_text_editor()

    def _destroy_text_editor(self) -> None:
        if self._text_editor_window is not None:
            self.canvas.delete(self._text_editor_window)
        if self._text_editor is not None:
            self._text_editor.destroy()
        self._text_editor = None
        self._text_editor_window = None
        self._text_editor_layer_id = None
        self._text_editor_before = None

    def apply_text_values(self, layer: Layer, text: str, origin: tuple[int, int], box_width: int) -> None:
        data = layer.text_data or {}
        data.update({
            "text": text,
            "x": int(origin[0]),
            "y": int(origin[1]),
            "color": list(self.foreground),
            "size": int(self.text_size.get()),
            "font_family": self.text_font_family.get(),
            "box_width": max(0, int(box_width)),
            "align": self.text_align.get(),
            "line_spacing": max(0, int(self.text_line_spacing.get())),
            "tracking": int(self.text_tracking.get()),
            "bold": bool(self.text_bold.get()),
            "italic": bool(self.text_italic.get()),
            "underline": bool(self.text_underline.get()),
            "rotation": float(self.text_rotation.get()),
        })
        layer.text_data = data
        layer.name = f"Text: {text[:24]}"
        render_text_layer(layer)
        layer.touch_pixels()
        self.doc.dirty = True

    def load_text_properties_from_layer(self, layer: Layer) -> None:
        if layer.text_data is None:
            return
        data = layer.text_data
        self._loading_text_properties = True
        try:
            self.text_font_family.set(str(data.get("font_family", "Arial")))
            self.text_size.set(int(data.get("size", 48)))
            self.text_bold.set(bool(data.get("bold", False)))
            self.text_italic.set(bool(data.get("italic", False)))
            self.text_underline.set(bool(data.get("underline", False)))
            self.text_align.set(str(data.get("align", "left")))
            self.text_line_spacing.set(int(data.get("line_spacing", 10)))
            self.text_tracking.set(int(data.get("tracking", 0)))
            self.text_rotation.set(float(data.get("rotation", 0.0)))
            self.text_box_width.set(int(data.get("box_width", 0)))
            color = data.get("color")
            if isinstance(color, list) and len(color) == 4:
                self.foreground = tuple(int(value) for value in color)
                self.refresh_color_control()
        finally:
            self._loading_text_properties = False

    def text_properties_changed(self, *_args) -> None:
        if self._loading_text_properties:
            return
        if self._text_editor is not None:
            self.update_text_editor_style()
            return
        if not self._editor_active or self.tool.get() != "text":
            return
        layer = self.doc.layer
        if layer.kind != "text" or layer.text_data is None:
            return
        if self._text_property_before is None:
            self._text_property_before = copy.deepcopy(layer.text_data)
        self.apply_text_values(
            layer,
            str(layer.text_data.get("text", "")),
            (int(layer.text_data.get("x", 0)), int(layer.text_data.get("y", 0))),
            int(self.text_box_width.get()),
        )
        self.request_canvas_refresh()
        if self._text_property_after_id is not None:
            self.after_cancel(self._text_property_after_id)
        self._text_property_after_id = self.after(350, self.commit_text_property_history)

    def commit_text_property_history(self) -> None:
        self._text_property_after_id = None
        before = self._text_property_before
        self._text_property_before = None
        layer = self.doc.layer
        if before is None or layer.kind != "text" or layer.text_data is None:
            return
        after = copy.deepcopy(layer.text_data)
        if before != after:
            before_name = f"Text: {str(before.get('text', ''))[:24]}"
            self.push_command(TextDataCommand("Text properties", layer.id, before, after, before_name, layer.name))

    def current_gradient_stops(self) -> list[dict[str, object]]:
        stops: list[dict[str, object]] = [
            {"position": 0.0, "color": list(self.foreground)},
            {"position": 1.0, "color": list(self.background)},
        ]
        if self.gradient_mid_enabled.get():
            stops.append({
                "position": float(np.clip(self.gradient_mid_position.get(), 0.01, 0.99)),
                "color": list(self.gradient_mid_color),
            })
        return sorted(stops, key=lambda stop: float(stop["position"]))

    def pick_gradient_mid(self) -> None:
        color = colorchooser.askcolor(color=self.color_hex(self.gradient_mid_color), title="Средняя точка градиента")[0]
        if color:
            self.gradient_mid_color = tuple(map(int, color)) + (255,)
            self.gradient_mid_enabled.set(True)
            if hasattr(self, "tool_options_panel"):
                self.tool_options_panel.render()

    def update_gradient_preview(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        if start == end:
            return
        if self.gradient_mode.get() == "Объект":
            now = time.perf_counter()
            if now - self._last_gradient_preview_at < 1 / 30:
                return
            self._last_gradient_preview_at = now
            self.update_gradient_object_preview(start, end)
            return
        pixels = GradientEngine.render(
            self.doc.width,
            self.doc.height,
            start,
            end,
            self.current_gradient_stops(),
            self.current_gradient_kind(),
        )
        alpha = np.full((self.doc.height, self.doc.width), 190, dtype=np.uint8)
        if self.doc.selection_mask is not None:
            alpha = np.minimum(alpha, self.doc.selection_mask)
        pixels[:, :, 3] = np.minimum(pixels[:, :, 3], alpha)
        image = Image.fromarray(pixels, "RGBA")
        scale = float(self.zoom.get())
        if scale != 1.0:
            image = image.resize(
                (max(1, round(self.doc.width * scale)), max(1, round(self.doc.height * scale))),
                Image.Resampling.BILINEAR,
            )
        self._gradient_preview_image = ImageTk.PhotoImage(image)
        x, y = self.doc_to_canvas(0, 0)
        if self._gradient_preview_id is None:
            self._gradient_preview_id = self.canvas.create_image(x, y, image=self._gradient_preview_image, anchor=tk.NW)
        else:
            self.canvas.coords(self._gradient_preview_id, x, y)
            self.canvas.itemconfigure(self._gradient_preview_id, image=self._gradient_preview_image)
        self.canvas.tag_raise(self._gradient_preview_id)
        for item_id in self._drag_preview_ids:
            self.canvas.tag_raise(item_id)

    def update_gradient_object_preview(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        x1, x2 = sorted((int(start[0]), int(end[0])))
        y1, y2 = sorted((int(start[1]), int(end[1])))
        scale = max(0.01, float(self.zoom.get()))
        width = max(2, round((x2 - x1) * scale))
        height = max(2, round((y2 - y1) * scale))
        local_start = ((start[0] - x1) * scale, (start[1] - y1) * scale)
        local_end = ((end[0] - x1) * scale, (end[1] - y1) * scale)
        if self.gradient_object_fill.get() == "Текстура":
            yy, xx = np.mgrid[0:height, 0:width]
            size = max(2, round(18 * scale))
            texture_kind = self.gradient_texture.get()
            if texture_kind == "Точки":
                dx = np.mod(xx, size) - size / 2.0
                dy = np.mod(yy, size) - size / 2.0
                selector = (dx * dx + dy * dy) <= (size * 0.24) ** 2
            elif texture_kind == "Полосы":
                selector = np.mod((xx + yy) // size, 2) == 0
            else:
                selector = np.mod(xx // size + yy // size, 2) == 0
            pixels = np.where(selector[:, :, None], np.array(self.foreground), np.array(self.background)).astype(np.uint8)
        else:
            pixels = GradientEngine.render(width, height, local_start, local_end, self.current_gradient_stops(), self.current_gradient_kind())
        mask_image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_image)
        box = (0, 0, width - 1, height - 1)
        shape = self.gradient_shape.get()
        if shape == "Эллипс":
            draw.ellipse(box, fill=220)
        elif shape == "Многоугольник":
            draw.polygon(regular_polygon_points(box, max(3, int(self.polygon_sides.get()))), fill=220)
        elif shape == "Звезда":
            draw.polygon(star_points(box, max(3, int(self.star_points_count.get())), float(self.star_inner_ratio.get())), fill=220)
        elif shape == "Произвольная":
            draw.polygon(custom_shape_points(CUSTOM_SHAPE_PRESETS.get(self.custom_shape_preset.get()), box), fill=220)
        else:
            draw.rectangle(box, fill=220)
        pixels[:, :, 3] = np.minimum(pixels[:, :, 3], np.array(mask_image, dtype=np.uint8))
        self._gradient_preview_image = ImageTk.PhotoImage(Image.fromarray(pixels, "RGBA"))
        x, y = self.doc_to_canvas(x1, y1)
        if self._gradient_preview_id is None:
            self._gradient_preview_id = self.canvas.create_image(x, y, image=self._gradient_preview_image, anchor=tk.NW)
        else:
            self.canvas.coords(self._gradient_preview_id, x, y)
            self.canvas.itemconfigure(self._gradient_preview_id, image=self._gradient_preview_image)
        self.canvas.tag_raise(self._gradient_preview_id)
        for item_id in self._drag_preview_ids:
            self.canvas.tag_raise(item_id)

    def clear_gradient_preview(self) -> None:
        if self._gradient_preview_id is not None:
            self.canvas.delete(self._gradient_preview_id)
            self._gradient_preview_id = None
        self._gradient_preview_image = None
        self._last_gradient_preview_at = 0.0

    def create_gradient_object(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        shape = {
            "Прямоугольник": "rectangle",
            "Эллипс": "ellipse",
            "Многоугольник": "polygon",
            "Звезда": "star",
            "Произвольная": "custom",
        }.get(self.gradient_shape.get(), "rectangle")
        gradient = {
            "type": self.current_gradient_kind(),
            "start": list(start),
            "end": list(end),
            "stops": self.current_gradient_stops(),
            "opacity": 1.0,
        }
        texture = {
            "type": {"Шахматная": "checker", "Полосы": "stripes", "Точки": "dots"}.get(self.gradient_texture.get(), "checker"),
            "size": 18,
            "color_a": list(self.foreground),
            "color_b": list(self.background),
        }
        use_texture = self.gradient_object_fill.get() == "Текстура"
        layer = self.doc.add_shape_layer(
            shape,
            (*start, *end),
            self.foreground,
            self.background,
            int(self.shape_stroke_width.get()),
            int(self.polygon_sides.get() if shape == "polygon" else self.star_points_count.get()),
            float(self.star_inner_ratio.get()),
            custom_points=CUSTOM_SHAPE_PRESETS.get(self.custom_shape_preset.get()),
            gradient=None if use_texture else gradient,
            texture=texture if use_texture else None,
        )
        self.push_command(LayerInsertCommand("Gradient object", self.doc.active_layer, copy.deepcopy(layer)))

    def crop_box_for_drag(self, start: tuple[int, int], end: tuple[int, int]) -> tuple[int, int, int, int]:
        sx, sy = start
        ex, ey = end
        ratios = {
            "1:1": 1.0,
            "4:3": 4.0 / 3.0,
            "3:2": 3.0 / 2.0,
            "16:9": 16.0 / 9.0,
            "Исходное": self.doc.width / max(1, self.doc.height),
            "Свое": max(1, int(self.crop_custom_width.get())) / max(1, int(self.crop_custom_height.get())),
        }
        ratio = ratios.get(self.crop_aspect.get())
        if ratio is not None:
            dx, dy = ex - sx, ey - sy
            sign_x = -1 if dx < 0 else 1
            sign_y = -1 if dy < 0 else 1
            width, height = abs(dx), abs(dy)
            if height == 0 or width / max(1, height) > ratio:
                height = round(width / ratio)
            else:
                width = round(height * ratio)
            ex, ey = sx + sign_x * width, sy + sign_y * height
        x1, x2 = sorted((max(0, min(self.doc.width, sx)), max(0, min(self.doc.width, ex))))
        y1, y2 = sorted((max(0, min(self.doc.height, sy)), max(0, min(self.doc.height, ey))))
        return int(x1), int(y1), int(x2), int(y2)

    def crop_handle_at(self, point: tuple[int, int]) -> str | None:
        if self._crop_box is None:
            return None
        x1, y1, x2, y2 = self._crop_box
        handles = {
            "nw": (x1, y1), "n": ((x1 + x2) / 2, y1), "ne": (x2, y1),
            "w": (x1, (y1 + y2) / 2), "e": (x2, (y1 + y2) / 2),
            "sw": (x1, y2), "s": ((x1 + x2) / 2, y2), "se": (x2, y2),
        }
        threshold = max(3.0, 9.0 / max(0.05, float(self.zoom.get())))
        nearest = None
        best = threshold
        for name, (hx, hy) in handles.items():
            distance = math.hypot(point[0] - hx, point[1] - hy)
            if distance <= best:
                nearest, best = name, distance
        return nearest

    def resize_crop_box(
        self,
        box: tuple[int, int, int, int],
        handle: str,
        point: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        px = max(0, min(self.doc.width, int(point[0])))
        py = max(0, min(self.doc.height, int(point[1])))
        if "w" in handle:
            x1 = px
        if "e" in handle:
            x2 = px
        if "n" in handle:
            y1 = py
        if "s" in handle:
            y2 = py
        if len(handle) == 2 and self.crop_aspect.get() != "Свободно":
            anchor_x = x2 if "w" in handle else x1
            anchor_y = y2 if "n" in handle else y1
            return self.crop_box_for_drag((anchor_x, anchor_y), (px, py))
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        return int(left), int(top), int(right), int(bottom)

    def draw_crop_overlay(self, box: tuple[int, int, int, int]) -> None:
        self.clear_crop_overlay()
        x1, y1, x2, y2 = box
        left, top = self.doc_to_canvas(0, 0)
        right, bottom = self.doc_to_canvas(self.doc.width, self.doc.height)
        cx1, cy1 = self.doc_to_canvas(x1, y1)
        cx2, cy2 = self.doc_to_canvas(x2, y2)
        shade = {"fill": "#111111", "outline": "", "stipple": "gray50"}
        self._crop_overlay_ids.extend([
            self.canvas.create_rectangle(left, top, right, cy1, **shade),
            self.canvas.create_rectangle(left, cy2, right, bottom, **shade),
            self.canvas.create_rectangle(left, cy1, cx1, cy2, **shade),
            self.canvas.create_rectangle(cx2, cy1, right, cy2, **shade),
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="#f5f5f5", width=2),
        ])
        for fraction in (1 / 3, 2 / 3):
            gx = cx1 + (cx2 - cx1) * fraction
            gy = cy1 + (cy2 - cy1) * fraction
            self._crop_overlay_ids.append(self.canvas.create_line(gx, cy1, gx, cy2, fill="#e0e0e0", dash=(4, 4)))
            self._crop_overlay_ids.append(self.canvas.create_line(cx1, gy, cx2, gy, fill="#e0e0e0", dash=(4, 4)))
        handles = [
            (cx1, cy1), ((cx1 + cx2) / 2, cy1), (cx2, cy1),
            (cx1, (cy1 + cy2) / 2), (cx2, (cy1 + cy2) / 2),
            (cx1, cy2), ((cx1 + cx2) / 2, cy2), (cx2, cy2),
        ]
        for hx, hy in handles:
            self._crop_overlay_ids.append(self.canvas.create_rectangle(hx - 4, hy - 4, hx + 4, hy + 4, fill="#ffffff", outline="#1976d2"))
        for item_id in self._crop_overlay_ids:
            self.canvas.tag_raise(item_id)

    def clear_crop_overlay(self) -> None:
        if not hasattr(self, "canvas"):
            return
        for item_id in self._crop_overlay_ids:
            self.canvas.delete(item_id)
        self._crop_overlay_ids.clear()

    def apply_crop_overlay(self) -> None:
        if self._crop_box is None:
            self.status_text("Сначала протяните рамку кадрирования")
            return
        x1, y1, x2, y2 = self._crop_box
        if x2 - x1 < 2 or y2 - y1 < 2:
            self.status_text("Область кадрирования слишком мала")
            return
        self.run_document_command("Crop", lambda: self.doc.crop(self._crop_box))
        self.doc.clear_selection()
        self.selection_box = None
        self._crop_box = None
        self.clear_crop_overlay()
        self.refresh()

    def draw_lasso(self) -> None:
        self.delete_lasso_overlay()
        if len(self._lasso_points) < 2:
            return
        coords = [coord for point in self._lasso_points for xy in [self.doc_to_canvas(point[0], point[1])] for coord in xy]
        self._polygon_ids.append(self.canvas.create_line(*coords, fill="#50e3ff", dash=(4, 3), width=2, smooth=True))

    def draw_polygon_lasso(self, hover: tuple[int, int] | None = None) -> None:
        self.delete_lasso_overlay()
        preview_points = list(self._polygon_points)
        if hover is not None and preview_points:
            preview_points.append(hover)
        if len(preview_points) >= 2:
            coords = [coord for point in preview_points for xy in [self.doc_to_canvas(point[0], point[1])] for coord in xy]
            self._polygon_ids.append(self.canvas.create_line(*coords, fill="#50e3ff", dash=(4, 3), width=2))
        for x, y in self._polygon_points:
            cx, cy = self.doc_to_canvas(x, y)
            self._polygon_ids.append(self.canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill="#50e3ff", outline=""))

    def clear_lasso_overlay(self) -> None:
        self.delete_lasso_overlay()
        self._lasso_points.clear()
        self._polygon_points.clear()
        self._magnetic_edges = None

    def delete_lasso_overlay(self) -> None:
        for item_id in self._polygon_ids:
            self.canvas.delete(item_id)
        self._polygon_ids.clear()

    def magnetic_lasso_point(self, point: tuple[int, int]) -> tuple[int, int]:
        if self._magnetic_edges is None:
            self._magnetic_edges = self.doc.magnetic_edge_map(self.render_engine.render(self.doc, False))
        return self.doc.snap_point_to_edge(point, self._magnetic_edges, max(8, int(self.tolerance.get())))

    def clear_selection_overlay(self) -> None:
        if self.selection_id is not None:
            self.canvas.delete(self.selection_id)
            self.selection_id = None
        for item_id in self._selection_overlay_ids:
            self.canvas.delete(item_id)
        self._selection_overlay_ids.clear()
        if self._selection_animation_id is not None:
            try:
                self.after_cancel(self._selection_animation_id)
            except tk.TclError:
                pass
            self._selection_animation_id = None

    def selection_contours(self) -> list[np.ndarray]:
        mask = self.doc.selection_mask
        if mask is None or not np.any(mask >= 128):
            self._selection_contours = []
            self._selection_contour_signature = None
            return []
        signature = (id(mask), mask.shape[0], mask.shape[1])
        if signature != self._selection_contour_signature:
            self._selection_contours = selection_contour_points(mask)
            self._selection_contour_signature = signature
        return self._selection_contours

    def update_selection_overlay(self) -> None:
        self.clear_selection_overlay()
        contours = self.selection_contours()
        if not contours:
            return
        epsilon = max(0.5, 0.8 / max(0.05, float(self.zoom.get())))
        for contour in contours:
            simplified = cv2.approxPolyDP(contour.reshape(-1, 1, 2), epsilon, True)[:, 0, :]
            if len(simplified) < 2:
                continue
            closed = np.vstack([simplified, simplified[0]])
            coords = [value for x, y in closed for value in self.doc_to_canvas(float(x), float(y))]
            dark = self.canvas.create_line(*coords, fill="#111111", width=2, dash=(5, 5), dashoffset=self._selection_dash_phase)
            light = self.canvas.create_line(*coords, fill="#f4f4f4", width=1, dash=(5, 5), dashoffset=self._selection_dash_phase + 5)
            self._selection_overlay_ids.extend([dark, light])
        for item_id in self._selection_overlay_ids:
            self.canvas.tag_raise(item_id)
        if self._selection_overlay_ids:
            self._selection_animation_id = self.after(240, self.animate_selection_overlay)
        self.update_grid_and_guides()

    def animate_selection_overlay(self) -> None:
        self._selection_animation_id = None
        if not self._selection_overlay_ids:
            return
        self._selection_dash_phase = (self._selection_dash_phase + 1) % 10
        for index, item_id in enumerate(self._selection_overlay_ids):
            try:
                self.canvas.itemconfigure(item_id, dashoffset=self._selection_dash_phase + (5 if index % 2 else 0))
            except tk.TclError:
                return
        self._selection_animation_id = self.after(240, self.animate_selection_overlay)

    def update_grid_and_guides(self) -> None:
        for item_id in self._overlay_ids:
            self.canvas.delete(item_id)
        self._overlay_ids.clear()
        ox, oy = self._canvas_origin
        right, bottom = self.doc_to_canvas(self.doc.width, self.doc.height)
        if self.grid_visible.get():
            spacing = max(4, int(self.grid_spacing.get()))
            for x in range(spacing, self.doc.width, spacing):
                cx, _ = self.doc_to_canvas(x, 0)
                self._overlay_ids.append(self.canvas.create_line(cx, oy, cx, bottom, fill="#3b3f48", dash=(2, 6)))
            for y in range(spacing, self.doc.height, spacing):
                _, cy = self.doc_to_canvas(0, y)
                self._overlay_ids.append(self.canvas.create_line(ox, cy, right, cy, fill="#3b3f48", dash=(2, 6)))
        for orientation, value in self._guide_doc_lines:
            if orientation == "h":
                _, cy = self.doc_to_canvas(0, value)
                self._overlay_ids.append(self.canvas.create_line(ox, cy, right, cy, fill="#ff4fd8", width=1))
            else:
                cx, _ = self.doc_to_canvas(value, 0)
                self._overlay_ids.append(self.canvas.create_line(cx, oy, cx, bottom, fill="#ff4fd8", width=1))
        for item_id in self._overlay_ids:
            self.canvas.tag_raise(item_id)

    def create_shape_from_drag(self, tool: str, geometry: dict) -> None:
        options = self._shape_drag_options or self.current_shape_options(tool)
        shape = str(geometry["shape"])
        box = tuple(int(v) for v in geometry.get("line", geometry["box"]))
        layer = self.doc.add_shape_layer(
            shape,
            box,
            options["fill"],
            options["stroke"],
            int(options["stroke_width"]),
            int(options["sides"]),
            float(options["inner_ratio"]),
            custom_points=options["custom_points"],
        )
        self.selected_layer_ids = {layer.id}
        self.push_command(LayerInsertCommand("Shape layer", self.doc.active_layer, copy.deepcopy(layer)))

    def run_shape_data_command(self, label: str, edit) -> None:
        layer = self.doc.layer
        if layer.kind != "shape" or layer.shape_data is None:
            return
        before = copy.deepcopy(layer.shape_data)
        before_name = layer.name
        edit()
        after = copy.deepcopy(layer.shape_data or {})
        if before != after or before_name != layer.name:
            self.push_command(ShapeDataCommand(label, layer.id, before, after, before_name, layer.name))

    def clear_selection(self) -> None:
        self.run_selection_command("Clear selection", self.doc.clear_selection)

    def copy_pixels(self) -> None:
        if not self._editor_active:
            return
        layer = self.doc.layer
        selection = self.doc.layer_selection_mask(layer)
        if selection is None:
            self._pixel_clipboard = layer.pixels.copy()
            self._pixel_clipboard_origin = (layer.x, layer.y)
        else:
            ys, xs = np.where(selection > 0)
            if len(xs) == 0:
                return
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            pixels = layer.pixels[y1:y2, x1:x2].copy()
            coverage = selection[y1:y2, x1:x2].astype(np.float32) / 255.0
            pixels[:, :, 3] = np.clip(pixels[:, :, 3].astype(np.float32) * coverage, 0, 255).astype(np.uint8)
            self._pixel_clipboard = pixels
            self._pixel_clipboard_origin = (layer.x + x1, layer.y + y1)
        self.status_text("Пиксели скопированы")

    def paste_pixels(self) -> None:
        if self._pixel_clipboard is None:
            self.status_text("Буфер пикселей пуст")
            return
        layer = Layer(
            "Вставленные пиксели",
            self._pixel_clipboard.copy(),
            x=int(self._pixel_clipboard_origin[0]),
            y=int(self._pixel_clipboard_origin[1]),
        )
        self.doc.layers.append(layer)
        self.doc.active_layer = len(self.doc.layers) - 1
        self.doc.dirty = True
        self.selected_layer_ids = {layer.id}
        self.push_command(LayerInsertCommand("Paste pixels", self.doc.active_layer, copy.deepcopy(layer)))
        self.refresh()

    def delete_selected_pixels(self) -> None:
        if not self._editor_active or self.doc.selection_mask is None:
            self.status_text("Удаление пикселей: сначала создайте выделение")
            return
        layer = self.doc.layer
        if layer.locked:
            self.status_text("Слой заблокирован")
            return
        selection = self.doc.layer_selection_mask(layer)
        if selection is None or not np.any(selection):
            self.status_text("Выделение не пересекает активный слой")
            return

        def erase() -> None:
            coverage = selection.astype(np.float32) / 255.0
            layer.pixels[:, :, 3] = np.clip(
                layer.pixels[:, :, 3].astype(np.float32) * (1.0 - coverage),
                0,
                255,
            ).astype(np.uint8)
            layer.touch_pixels()

        self.run_pixel_delta_command("Delete selected pixels", erase)

    def select_all(self) -> None:
        self.run_selection_command("Select all", self.doc.select_all)

    def invert_selection(self) -> None:
        self.run_selection_command("Invert selection", self.doc.invert_selection)

    def feather_selection(self) -> None:
        radius = simpledialog.askinteger("Feather", "Radius px:", initialvalue=8, minvalue=1, maxvalue=500)
        if radius:
            self.run_selection_command("Feather selection", lambda: self.doc.feather_selection(radius))

    def grow_selection(self) -> None:
        pixels = simpledialog.askinteger("Grow", "Pixels:", initialvalue=8, minvalue=1, maxvalue=500)
        if pixels:
            self.run_selection_command("Grow selection", lambda: self.doc.grow_selection(pixels))

    def shrink_selection(self) -> None:
        pixels = simpledialog.askinteger("Shrink", "Pixels:", initialvalue=8, minvalue=1, maxvalue=500)
        if pixels:
            self.run_selection_command("Shrink selection", lambda: self.doc.shrink_selection(pixels))

    def smooth_selection(self) -> None:
        radius = simpledialog.askinteger("Smooth", "Radius px:", initialvalue=4, minvalue=1, maxvalue=500)
        if radius:
            self.run_selection_command("Smooth selection", lambda: self.doc.smooth_selection(radius))

    def border_selection(self) -> None:
        width = simpledialog.askinteger("Border", "Width px:", initialvalue=8, minvalue=1, maxvalue=500)
        if width:
            self.run_selection_command("Border selection", lambda: self.doc.border_selection(width))

    def refine_selection(self) -> None:
        raw = simpledialog.askstring("Refine selection", "smooth,feather,contrast,shift:", initialvalue="2,2,1.25,0")
        if not raw:
            return
        try:
            parts = [part.strip() for part in raw.split(",")]
            if len(parts) != 4:
                raise ValueError
            smooth = max(0, int(float(parts[0])))
            feather = max(0, int(float(parts[1])))
            contrast = max(0.0, float(parts[2]))
            shift = int(float(parts[3]))
        except ValueError:
            messagebox.showerror("Refine selection", "Use: smooth,feather,contrast,shift")
            return
        self.run_selection_command("Refine selection", lambda: self.doc.refine_selection(smooth, feather, contrast, shift))

    def cleanup_selection_edges(self) -> None:
        if self.doc.selection_mask is None:
            messagebox.showinfo("Умная очистка края", "Сначала создайте выделение.")
            return
        radius = simpledialog.askinteger("Умная очистка края", "Радиус края:", initialvalue=3, minvalue=1, maxvalue=40)
        if radius is None:
            return
        strength = simpledialog.askfloat("Умная очистка края", "Сила 0..1:", initialvalue=0.7, minvalue=0.0, maxvalue=1.0)
        if strength is None:
            return
        self.run_selection_command("Умная очистка края", lambda: self.doc.cleanup_selection_edges(radius, strength))

    def correct_selection_edges(self) -> None:
        if self.doc.selection_mask is None:
            messagebox.showinfo("Коррекция края", "Сначала создайте выделение.")
            return
        radius = simpledialog.askinteger("Коррекция края", "Радиус анализа:", initialvalue=3, minvalue=1, maxvalue=40)
        if radius is None:
            return
        strength = simpledialog.askfloat("Коррекция края", "Сила 0..1:", initialvalue=0.65, minvalue=0.0, maxvalue=1.0)
        if strength is None:
            return
        threshold = simpledialog.askinteger("Коррекция края", "Порог уверенности 0..255:", initialvalue=96, minvalue=0, maxvalue=255)
        if threshold is None:
            return
        self.run_selection_command("Коррекция края по уверенности", lambda: self.doc.correct_selection_edges(radius, strength, threshold))

    def select_and_mask_workspace(self) -> None:
        if self.doc.selection_mask is None:
            messagebox.showinfo("Выделить и маска", "Сначала создайте выделение.")
            return
        data = self.select_and_mask_dialog()
        if data is None:
            return
        refined_mask = np.asarray(data["mask"], dtype=np.uint8)
        output = data["output"]
        if output == "Маска слоя":
            layer = self.doc.layer
            local_mask = np.zeros(layer.pixels.shape[:2], dtype=np.uint8)
            x1, y1 = max(0, layer.x), max(0, layer.y)
            x2 = min(self.doc.width, layer.x + layer.pixels.shape[1])
            y2 = min(self.doc.height, layer.y + layer.pixels.shape[0])
            if x1 < x2 and y1 < y2:
                lx1, ly1 = x1 - layer.x, y1 - layer.y
                local_mask[ly1:ly1 + y2 - y1, lx1:lx1 + x2 - x1] = refined_mask[y1:y2, x1:x2]
            before_fields = {
                "mask": None if layer.mask is None else layer.mask.copy(),
                "mask_enabled": layer.mask_enabled,
                "mask_linked": layer.mask_linked,
            }
            layer.mask = local_mask
            layer.mask_enabled = True
            layer.mask_linked = True
            layer.touch_mask()
            after_fields = {"mask": layer.mask.copy(), "mask_enabled": True, "mask_linked": True}
            self.push_command(LayerFieldsCommand("Select and Mask: маска слоя", layer.id, before_fields, after_fields))
            self.doc.dirty = True
            self.paint_target.set("mask")
            self.mask_preview.set(MASK_PREVIEW_CHANNEL)
            self.selection_box = self.doc.selection_bounds()
            self.refresh()
            self.status_text("Select and Mask: маска слоя")
        elif output in {"Новый слой", "Новый слой с маской"}:
            source_layer = self.doc.layer
            local_mask = np.zeros(source_layer.pixels.shape[:2], dtype=np.uint8)
            x1, y1 = max(0, source_layer.x), max(0, source_layer.y)
            x2 = min(self.doc.width, source_layer.x + source_layer.pixels.shape[1])
            y2 = min(self.doc.height, source_layer.y + source_layer.pixels.shape[0])
            if x1 < x2 and y1 < y2:
                lx1, ly1 = x1 - source_layer.x, y1 - source_layer.y
                local_mask[ly1:ly1 + y2 - y1, lx1:lx1 + x2 - x1] = refined_mask[y1:y2, x1:x2]
            duplicate = source_layer.clone()
            duplicate.name = f"{source_layer.name} - выделено"
            if bool(data.get("decontaminate", False)):
                duplicate.pixels = decontaminate_edge_colors(duplicate.pixels, local_mask, float(data.get("decontaminate_strength", 0.5)))
                duplicate.kind = "raster"
                duplicate.text_data = None
                duplicate.shape_data = None
                duplicate.smart_data = None
                duplicate.smart_source = None
            if output == "Новый слой с маской":
                duplicate.mask = local_mask.copy()
                duplicate.mask_enabled = True
                duplicate.mask_linked = True
                duplicate.touch_mask()
            else:
                alpha = duplicate.pixels[:, :, 3].astype(np.float32) * (local_mask.astype(np.float32) / 255.0)
                duplicate.pixels[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
                duplicate.mask = None
                duplicate.touch_pixels()
            self.doc.layers.append(duplicate)
            self.doc.active_layer = len(self.doc.layers) - 1
            self.doc.dirty = True
            self.selected_layer_ids = {duplicate.id}
            self.push_command(LayerInsertCommand("Select and Mask: новый слой", self.doc.active_layer, copy.deepcopy(duplicate)))
            self.refresh()
            self.status_text(f"Select and Mask: {output.lower()}")
        else:
            self.run_selection_command("Select and Mask", lambda: setattr(self.doc, "selection_mask", refined_mask.copy()))
            self.refresh_canvas()

    def select_and_mask_dialog(self) -> dict[str, object] | None:
        source = self.doc.selection_mask.copy()
        composite = self.render_engine.render(self.doc, checker=False)
        dialog = tk.Toplevel(self)
        dialog.title("Выделить и маска")
        dialog.transient(self)
        dialog.grab_set()
        dialog.minsize(900, 620)

        smooth = tk.IntVar(value=2)
        feather = tk.IntVar(value=2)
        contrast = tk.DoubleVar(value=1.25)
        shift = tk.IntVar(value=0)
        output = tk.StringVar(value="Выделение")
        preview_mode = tk.StringVar(value=SELECT_MASK_PREVIEW_CUTOUT)
        brush_mode = tk.StringVar(value="Уточнить волосы")
        brush_size = tk.IntVar(value=36)
        edge_radius = tk.IntVar(value=7)
        edge_strength = tk.DoubleVar(value=0.8)
        decontaminate = tk.BooleanVar(value=False)
        decontaminate_strength = tk.DoubleVar(value=0.5)
        working = source.copy()
        stroke_history: list[np.ndarray] = []
        last_point: list[tuple[int, int] | None] = [None]
        result: dict[str, object] | None = None

        header = ttk.Frame(dialog, padding=(12, 10, 12, 6))
        header.pack(fill=tk.X)
        ttk.Label(header, text="Выделить и маска", style="PanelTitle.TLabel").pack(side=tk.LEFT)
        stats = ttk.Label(header, text="", style="Secondary.TLabel")
        stats.pack(side=tk.RIGHT)

        body = ttk.PanedWindow(dialog, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=12)
        preview = tk.Canvas(body, width=570, height=540, background=TOKENS.WORKSPACE, highlightthickness=1, highlightbackground=TOKENS.BORDER, cursor="crosshair")
        controls = ttk.Frame(body, width=285, padding=(12, 2))
        body.add(preview, weight=1)
        body.add(controls, weight=0)

        footer = ttk.Frame(dialog, padding=(12, 8, 12, 12))
        footer.pack(fill=tk.X)
        ttk.Label(footer, text="Рисуйте по сложному краю прямо в предпросмотре", style="Secondary.TLabel").pack(side=tk.LEFT)

        transform = {"scale": 1.0, "ox": 0.0, "oy": 0.0, "size": 540}
        reduced_composite_cache: dict[tuple[int, int], np.ndarray] = {}
        mode_ids = {
            "Уточнить волосы": "refine",
            "Добавить": "add",
            "Вычесть": "subtract",
            "Сгладить кистью": "smooth",
        }

        def current_mask() -> np.ndarray:
            return refine_selection_mask(working, int(smooth.get()), int(feather.get()), float(contrast.get()), int(shift.get()))

        def update_preview(*_args) -> None:
            try:
                mask = current_mask()
            except (tk.TclError, ValueError):
                return
            preview.update_idletasks()
            available = max(300, min(preview.winfo_width(), preview.winfo_height()))
            size = min(720, available)
            scale = min(1.0, size / max(1, self.doc.width), size / max(1, self.doc.height))
            reduced_width = max(1, round(self.doc.width * scale))
            reduced_height = max(1, round(self.doc.height * scale))
            transform.update({"scale": scale, "ox": (preview.winfo_width() - reduced_width) / 2.0, "oy": (preview.winfo_height() - reduced_height) / 2.0, "size": size})
            reduced_size = (reduced_width, reduced_height)
            reduced_composite = reduced_composite_cache.get(reduced_size)
            if reduced_composite is None:
                reduced_composite = composite if reduced_size == (self.doc.width, self.doc.height) else cv2.resize(composite, reduced_size, interpolation=cv2.INTER_AREA)
                reduced_composite_cache[reduced_size] = reduced_composite
            reduced_mask = mask if reduced_size == (self.doc.width, self.doc.height) else cv2.resize(mask, reduced_size, interpolation=cv2.INTER_AREA)
            canvas = self.render_select_mask_preview(reduced_composite, reduced_mask, preview_mode.get(), size)
            self._select_mask_preview_image = ImageTk.PhotoImage(canvas)
            preview.delete("preview")
            preview.create_image(preview.winfo_width() / 2.0, preview.winfo_height() / 2.0, image=self._select_mask_preview_image, tags="preview")
            preview.tag_lower("preview")
            selected = int(np.count_nonzero(mask))
            soft = int(np.count_nonzero((mask > 0) & (mask < 255)))
            stats.configure(text=f"Выбрано: {selected} px  |  Полупрозрачных: {soft} px")

        def add_spin(parent: ttk.Frame, label: str, variable, from_: float, to: float, increment: float = 1.0) -> None:
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=3)
            ttk.Label(row, text=label).pack(side=tk.LEFT)
            ttk.Spinbox(row, textvariable=variable, from_=from_, to=to, increment=increment, width=9, command=update_preview).pack(side=tk.RIGHT)

        ttk.Label(controls, text="Кисть уточнения", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(2, 4))
        brush_box = ttk.Combobox(controls, textvariable=brush_mode, values=list(mode_ids), state="readonly")
        brush_box.pack(fill=tk.X, pady=(0, 5))
        add_spin(controls, "Размер", brush_size, 3, 300)
        add_spin(controls, "Радиус анализа", edge_radius, 1, 40)
        add_spin(controls, "Сила", edge_strength, 0.05, 1.0, 0.05)
        brush_buttons = ttk.Frame(controls)
        brush_buttons.pack(fill=tk.X, pady=(4, 8))

        def reset_brushes() -> None:
            nonlocal working
            if not np.array_equal(working, source):
                stroke_history.append(working.copy())
            working = source.copy()
            update_preview()

        def undo_brush() -> None:
            nonlocal working
            if stroke_history:
                working = stroke_history.pop()
                update_preview()

        ttk.Button(brush_buttons, text="Отменить штрих", command=undo_brush).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(brush_buttons, text="Сбросить", command=reset_brushes).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        ttk.Separator(controls).pack(fill=tk.X, pady=5)
        ttk.Label(controls, text="Глобальная обработка", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(4, 3))
        add_spin(controls, "Сглаживание", smooth, 0, 100)
        add_spin(controls, "Растушевка", feather, 0, 500)
        add_spin(controls, "Контраст", contrast, 0.0, 5.0, 0.05)
        add_spin(controls, "Сдвиг края", shift, -500, 500)

        ttk.Separator(controls).pack(fill=tk.X, pady=6)
        ttk.Label(controls, text="Просмотр", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(3, 2))
        preview_box = ttk.Combobox(controls, textvariable=preview_mode, values=SELECT_MASK_PREVIEW_MODES, state="readonly")
        preview_box.pack(fill=tk.X, pady=(0, 7))
        ttk.Label(controls, text="Результат", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(3, 2))
        output_box = ttk.Combobox(controls, textvariable=output, values=["Выделение", "Маска слоя", "Новый слой", "Новый слой с маской"], state="readonly")
        output_box.pack(fill=tk.X)
        decontaminate_check = ttk.Checkbutton(controls, text="Очистить цветную кайму", variable=decontaminate)
        decontaminate_check.pack(anchor=tk.W, pady=(7, 1))
        decontaminate_row = ttk.Frame(controls)
        decontaminate_row.pack(fill=tk.X, pady=3)
        ttk.Label(decontaminate_row, text="Сила очистки").pack(side=tk.LEFT)
        decontaminate_spin = ttk.Spinbox(decontaminate_row, textvariable=decontaminate_strength, from_=0.0, to=1.0, increment=0.05, width=9)
        decontaminate_spin.pack(side=tk.RIGHT)

        def update_output_controls(*_args) -> None:
            enabled = output.get() in {"Новый слой", "Новый слой с маской"}
            decontaminate_check.configure(state=tk.NORMAL if enabled else tk.DISABLED)
            decontaminate_spin.configure(state=tk.NORMAL if enabled else tk.DISABLED)
            if not enabled:
                decontaminate.set(False)

        def canvas_to_document(event) -> tuple[int, int] | None:
            scale = float(transform["scale"])
            source_width = self.doc.width * scale
            source_height = self.doc.height * scale
            left = (preview.winfo_width() - source_width) / 2.0
            top = (preview.winfo_height() - source_height) / 2.0
            x = round((event.x - left) / max(1e-8, scale))
            y = round((event.y - top) / max(1e-8, scale))
            if 0 <= x < self.doc.width and 0 <= y < self.doc.height:
                return x, y
            return None

        def apply_stroke(start: tuple[int, int], end: tuple[int, int]) -> None:
            nonlocal working
            stroke = np.zeros_like(working)
            width = max(1, int(brush_size.get()))
            cv2.line(stroke, start, end, 255, width, cv2.LINE_AA)
            cv2.circle(stroke, end, max(1, width // 2), 255, -1, cv2.LINE_AA)
            ys, xs = np.where(stroke > 0)
            if len(xs) == 0:
                return
            padding = max(width, int(edge_radius.get()) * 3) + 2
            x1, y1 = max(0, int(xs.min()) - padding), max(0, int(ys.min()) - padding)
            x2, y2 = min(self.doc.width, int(xs.max()) + padding + 1), min(self.doc.height, int(ys.max()) + padding + 1)
            working[y1:y2, x1:x2] = refine_selection_brush(
                working[y1:y2, x1:x2],
                composite[y1:y2, x1:x2],
                stroke[y1:y2, x1:x2],
                mode_ids.get(brush_mode.get(), "refine"),
                max(1, int(edge_radius.get())),
                float(edge_strength.get()),
            )
            update_preview()

        def brush_press(event) -> None:
            point = canvas_to_document(event)
            if point is None:
                return
            stroke_history.append(working.copy())
            last_point[0] = point
            apply_stroke(point, point)

        def brush_drag(event) -> None:
            point = canvas_to_document(event)
            if point is None or last_point[0] is None:
                return
            apply_stroke(last_point[0], point)
            last_point[0] = point

        def brush_release(_event) -> None:
            last_point[0] = None

        def brush_cursor(event) -> None:
            preview.delete("brush-cursor")
            radius = max(2.0, float(brush_size.get()) * float(transform["scale"]) / 2.0)
            preview.create_oval(event.x - radius, event.y - radius, event.x + radius, event.y + radius, outline="#f4f4f4", width=1, tags="brush-cursor")

        def accept() -> None:
            nonlocal result
            result = {
                "smooth": max(0, int(smooth.get())),
                "feather": max(0, int(feather.get())),
                "contrast": max(0.0, float(contrast.get())),
                "shift": int(shift.get()),
                "output": output.get(),
                "mask": current_mask().copy(),
                "decontaminate": bool(decontaminate.get()),
                "decontaminate_strength": float(decontaminate_strength.get()),
            }
            dialog.destroy()

        def cancel() -> None:
            dialog.destroy()

        ttk.Button(footer, text="Применить", command=accept, style="Primary.TButton").pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(footer, text="Отмена", command=cancel).pack(side=tk.RIGHT)
        preview_box.bind("<<ComboboxSelected>>", update_preview)
        output_box.bind("<<ComboboxSelected>>", update_output_controls)
        preview.bind("<ButtonPress-1>", brush_press)
        preview.bind("<B1-Motion>", brush_drag)
        preview.bind("<ButtonRelease-1>", brush_release)
        preview.bind("<Motion>", brush_cursor)
        preview.bind("<Leave>", lambda _event: preview.delete("brush-cursor"))
        preview.bind("<Configure>", update_preview)
        for variable in [smooth, feather, contrast, shift, decontaminate_strength]:
            variable.trace_add("write", update_preview)
        dialog.protocol("WM_DELETE_WINDOW", cancel)
        self._select_mask_canvas = preview
        self._select_mask_brush_mode = brush_mode
        self._select_mask_brush_size = brush_size
        self._select_mask_output = output
        self._select_mask_decontaminate = decontaminate
        self._select_mask_apply_stroke = apply_stroke
        self._select_mask_working = lambda: working.copy()
        self._select_mask_accept = accept
        self.center_toplevel(dialog, 980, 700)
        update_output_controls()
        update_preview()
        dialog.wait_window()
        return result

    @staticmethod
    def mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
        if mask is None or not np.any(mask):
            return None
        ys, xs = np.where(mask > 0)
        return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)

    @staticmethod
    def render_select_mask_preview(composite: np.ndarray, mask: np.ndarray, mode: str, size: int = 160) -> Image.Image:
        preview_size = (size, size)
        height, width = composite.shape[:2]
        scale = min(1.0, size / max(1, width), size / max(1, height))
        reduced_size = max(1, round(width * scale)), max(1, round(height * scale))
        reduced = composite if reduced_size == (width, height) else cv2.resize(composite, reduced_size, interpolation=cv2.INTER_AREA)
        reduced_mask = mask if reduced_size == (width, height) else cv2.resize(mask, reduced_size, interpolation=cv2.INTER_NEAREST)
        mask_image = Image.fromarray(reduced_mask.astype(np.uint8), "L")
        source = rgba_array_to_pil(reduced)
        x = (size - source.width) // 2
        y = (size - source.height) // 2
        if mode == SELECT_MASK_PREVIEW_OVERLAY:
            canvas = Image.new("RGBA", preview_size, (44, 46, 52, 255))
            canvas.alpha_composite(source, (x, y))
            overlay = Image.new("RGBA", source.size, (255, 36, 68, 0))
            overlay.putalpha(mask_image.point(lambda value: int(value * 0.55)))
            canvas.alpha_composite(overlay, (x, y))
            return canvas
        if mode == SELECT_MASK_PREVIEW_CUTOUT:
            checker = Image.new("RGBA", preview_size, (44, 46, 52, 255))
            tile = 8
            for cy in range(0, size, tile):
                for cx in range(0, size, tile):
                    color = (90, 92, 98, 255) if ((cx // tile) + (cy // tile)) % 2 else (58, 60, 66, 255)
                    checker.paste(color, (cx, cy, min(cx + tile, size), min(cy + tile, size)))
            cutout = source.copy()
            cutout.putalpha(mask_image)
            checker.alpha_composite(cutout, (x, y))
            return checker
        if mode == SELECT_MASK_PREVIEW_EDGE_CONFIDENCE:
            confidence = selection_edge_confidence(mask, composite, 3)
            confidence_image = Image.fromarray(confidence.astype(np.uint8), "L")
            confidence_image.thumbnail(preview_size, Image.Resampling.NEAREST)
            canvas = Image.new("RGBA", preview_size, (44, 46, 52, 255))
            dimmed = source.copy()
            shade = Image.new("RGBA", dimmed.size, (0, 0, 0, 96))
            dimmed.alpha_composite(shade)
            canvas.alpha_composite(dimmed, (x, y))
            conf = np.array(confidence_image, dtype=np.uint8)
            overlay = np.zeros((confidence_image.height, confidence_image.width, 4), dtype=np.uint8)
            weak = (conf > 0) & (conf < 85)
            medium = (conf >= 85) & (conf < 170)
            strong = conf >= 170
            overlay[weak] = [255, 68, 68, 230]
            overlay[medium] = [255, 190, 64, 230]
            overlay[strong] = [64, 220, 110, 230]
            canvas.alpha_composite(Image.fromarray(overlay, "RGBA"), ((size - confidence_image.width) // 2, (size - confidence_image.height) // 2))
            return canvas
        canvas = Image.new("RGBA", preview_size, (44, 46, 52, 255))
        gray = Image.new("L", preview_size, 72)
        gray.paste(mask_image, ((size - mask_image.width) // 2, (size - mask_image.height) // 2))
        rgba = Image.merge("RGBA", (gray, gray, gray, Image.new("L", preview_size, 255)))
        canvas.alpha_composite(rgba)
        return canvas

    def select_opaque_pixels(self) -> None:
        self.run_selection_command("Select opaque pixels", lambda: self.doc.select_opaque_pixels(self.doc.layer))

    def automatic_selection_workspace(self) -> None:
        data = self.automatic_selection_dialog()
        if data is None:
            return
        mask = np.asarray(data["mask"], dtype=np.uint8)
        self.run_selection_command("Автоматическое выделение", lambda: setattr(self.doc, "selection_mask", mask.copy()))
        self.refresh_canvas()
        if data["output"] == "Уточнить в «Выделить и маска»":
            self.select_and_mask_workspace()

    def automatic_selection_dialog(self) -> dict[str, object] | None:
        layer = self.doc.layer
        if layer.kind == "adjustment" or layer.pixels.size == 0:
            messagebox.showinfo("Автоматическое выделение", "Активный слой не содержит изображения.")
            return None
        dialog = tk.Toplevel(self)
        dialog.title("Автоматическое выделение")
        dialog.transient(self)
        dialog.grab_set()
        dialog.minsize(820, 590)
        target = tk.StringVar(value="Объект")
        sensitivity = tk.DoubleVar(value=0.55)
        preview_mode = tk.StringVar(value=SELECT_MASK_PREVIEW_OVERLAY)
        output = tk.StringVar(value="Уточнить в «Выделить и маска»")
        result: dict[str, object] | None = None
        composite = self.render_engine.render(self.doc, checker=False)

        header = ttk.Frame(dialog, padding=(12, 10, 12, 6))
        header.pack(fill=tk.X)
        ttk.Label(header, text="Автоматическое выделение", style="PanelTitle.TLabel").pack(side=tk.LEFT)
        quality = ttk.Label(header, text="", style="Secondary.TLabel")
        quality.pack(side=tk.RIGHT)
        body = ttk.PanedWindow(dialog, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=12)
        preview = ttk.Label(body, anchor=tk.CENTER)
        controls = ttk.Frame(body, width=250, padding=(12, 4))
        body.add(preview, weight=1)
        body.add(controls, weight=0)
        footer = ttk.Frame(dialog, padding=12)
        footer.pack(fill=tk.X)

        ttk.Label(controls, text="Что выделить", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(4, 3))
        target_box = ttk.Combobox(controls, textvariable=target, values=["Объект", "Фон", "Небо"], state="readonly")
        target_box.pack(fill=tk.X)
        ttk.Label(controls, text="Чувствительность", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(14, 3))
        sensitivity_value = ttk.Label(controls, text="55%", style="Secondary.TLabel")
        sensitivity_value.pack(anchor=tk.E)
        ttk.Scale(controls, variable=sensitivity, from_=0.0, to=1.0).pack(fill=tk.X)
        ttk.Label(controls, text="Просмотр", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(14, 3))
        mode_box = ttk.Combobox(controls, textvariable=preview_mode, values=SELECT_MASK_PREVIEW_MODES, state="readonly")
        mode_box.pack(fill=tk.X)
        ttk.Label(controls, text="После выбора", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(14, 3))
        ttk.Combobox(controls, textvariable=output, values=["Применить выделение", "Уточнить в «Выделить и маска»"], state="readonly").pack(fill=tk.X)
        description = ttk.Label(controls, wraplength=230, justify=tk.LEFT, style="Secondary.TLabel")
        description.pack(fill=tk.X, pady=(16, 0))

        latest_mask: list[np.ndarray | None] = [None]
        latest_signature: list[tuple[str, float] | None] = [None]
        preview_after: list[str | None] = [None]

        def calculate_mask() -> np.ndarray:
            local = automatic_selection_mask(layer.pixels, target.get(), float(sensitivity.get()))
            return self.doc._layer_mask_to_document(layer, local)

        def update_preview(*_args) -> None:
            preview_after[0] = None
            try:
                mask = calculate_mask()
            except (tk.TclError, ValueError):
                return
            latest_mask[0] = mask
            latest_signature[0] = (target.get(), round(float(sensitivity.get()), 4))
            size = 520
            image = self.render_select_mask_preview(composite, mask, preview_mode.get(), size)
            self._automatic_selection_preview_image = ImageTk.PhotoImage(image)
            preview.configure(image=self._automatic_selection_preview_image)
            sensitivity_value.configure(text=f"{round(float(sensitivity.get()) * 100)}%")
            selected = np.count_nonzero(mask >= 128)
            soft = np.count_nonzero((mask > 0) & (mask < 255))
            quality.configure(text=f"Выбрано: {selected} px  |  Мягкий край: {soft} px")
            notes = {
                "Объект": "Ищет отличающиеся от краёв изображения области и сохраняет мелкие связанные детали.",
                "Фон": "Строит мягкое дополнение к найденному объекту, включая прозрачные края.",
                "Небо": "Анализирует связанные с верхним краем синие и светлые области.",
            }
            description.configure(text=notes[target.get()])

        def schedule_preview(*_args) -> None:
            if preview_after[0] is not None:
                try:
                    dialog.after_cancel(preview_after[0])
                except tk.TclError:
                    pass
            preview_after[0] = dialog.after(90, update_preview)

        def accept() -> None:
            nonlocal result
            signature = (target.get(), round(float(sensitivity.get()), 4))
            mask = latest_mask[0] if latest_mask[0] is not None and latest_signature[0] == signature else calculate_mask()
            result = {"mask": mask.copy(), "target": target.get(), "sensitivity": float(sensitivity.get()), "output": output.get()}
            dialog.destroy()

        ttk.Label(footer, text="Результат можно сразу доработать кистями сложного края", style="Secondary.TLabel").pack(side=tk.LEFT)
        ttk.Button(footer, text="Применить", command=accept, style="Primary.TButton").pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(footer, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT)
        for variable in (target, sensitivity, preview_mode):
            variable.trace_add("write", schedule_preview)
        self._automatic_selection_target = target
        self._automatic_selection_sensitivity = sensitivity
        self._automatic_selection_output = output
        self._automatic_selection_accept = accept
        self._automatic_selection_preview = preview
        self.center_toplevel(dialog, 900, 650)
        update_preview()
        dialog.wait_window()
        return result

    def select_subject(self) -> None:
        self.run_selection_command("Выделить объект", lambda: self.doc.select_subject(self.doc.layer))

    def select_background(self) -> None:
        self.run_selection_command("Выделить фон", lambda: self.doc.select_background(self.doc.layer))

    def select_sky(self) -> None:
        self.run_selection_command("Выделить небо", lambda: self.doc.select_sky(self.doc.layer))

    def single_row_selection(self) -> None:
        y = simpledialog.askinteger("Single row", "Y coordinate:", initialvalue=self.doc.height // 2, minvalue=0, maxvalue=max(0, self.doc.height - 1))
        if y is not None:
            self.run_selection_command("Single row selection", lambda: self.doc.set_single_row_selection(y))

    def single_column_selection(self) -> None:
        x = simpledialog.askinteger("Single column", "X coordinate:", initialvalue=self.doc.width // 2, minvalue=0, maxvalue=max(0, self.doc.width - 1))
        if x is not None:
            self.run_selection_command("Single column selection", lambda: self.doc.set_single_column_selection(x))

    def save_selection(self) -> None:
        if self.doc.selection_mask is None:
            messagebox.showinfo("Save selection", "There is no active selection.")
            return
        default = f"Selection {len(self.doc.saved_selections) + 1}"
        name = simpledialog.askstring("Save selection", "Name:", initialvalue=default)
        if name:
            self.run_document_command("Save selection", lambda: self.doc.save_selection(name))
            self.refresh()

    def load_selection(self) -> None:
        if not self.doc.saved_selections:
            messagebox.showinfo("Load selection", "No saved selections.")
            return
        names = ", ".join(self.doc.saved_selections)
        name = simpledialog.askstring("Load selection", f"Name:\n{names}")
        if name:
            self.run_selection_command("Load selection", lambda: self.doc.load_selection(name))

    def delete_saved_selection(self) -> None:
        if not self.doc.saved_selections:
            messagebox.showinfo("Delete selection", "No saved selections.")
            return
        names = ", ".join(self.doc.saved_selections)
        name = simpledialog.askstring("Delete selection", f"Name:\n{names}")
        if name:
            self.run_document_command("Delete saved selection", lambda: self.doc.delete_saved_selection(name))
            self.refresh()

    def new_document(self) -> None:
        clipboard_image = self.read_clipboard_image()
        settings = self.new_document_dialog(clipboard_image)
        if settings is None:
            return
        self.create_document_from_settings(settings, clipboard_image)

    def available_document_presets(self, clipboard_image: Image.Image | None = None) -> list[dict[str, object]]:
        presets = [dict(item) for item in DOCUMENT_PRESETS]
        for preset in presets:
            if preset["name"] == "Свой размер":
                preset.update(
                    width=self.custom_canvas_width,
                    height=self.custom_canvas_height,
                    dpi=self.custom_canvas_dpi,
                    background=self.custom_canvas_background,
                )
                break
        if clipboard_image is not None:
            presets.insert(
                0,
                {
                    "name": "Из буфера обмена",
                    "description": "Размер скопированного изображения",
                    "width": clipboard_image.width,
                    "height": clipboard_image.height,
                    "dpi": 72,
                    "background": "Прозрачный",
                    "clipboard": True,
                },
            )
        return presets

    def new_document_dialog(self, clipboard_image: Image.Image | None = None) -> dict[str, object] | None:
        presets = self.available_document_presets(clipboard_image)
        dialog = tk.Toplevel(self)
        dialog.title("Новый холст")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)
        result: dict[str, object] | None = None

        width = tk.IntVar(value=int(presets[0]["width"]))
        height = tk.IntVar(value=int(presets[0]["height"]))
        dpi = tk.IntVar(value=int(presets[0]["dpi"]))
        background = tk.StringVar(value=str(presets[0]["background"]))
        include_clipboard = tk.BooleanVar(value=bool(presets[0].get("clipboard", False)))

        body = ttk.Frame(dialog)
        body.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)
        preset_panel = ttk.Frame(body, width=320)
        preset_panel.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 16))
        ttk.Label(preset_panel, text="Форматы", font=("Segoe UI Semibold", 12)).pack(anchor=tk.W, pady=(0, 6))
        preset_list_area = ttk.Frame(preset_panel)
        preset_list_area.pack(fill=tk.BOTH, expand=True)
        preset_row_height = 52
        preset_list_width = 306
        preset_list = tk.Canvas(
            preset_list_area,
            width=preset_list_width,
            height=572,
            background="#ffffff",
            highlightbackground="#c8cdd3",
            highlightthickness=1,
            cursor="hand2",
            takefocus=True,
        )
        preset_scroll = ttk.Scrollbar(preset_list_area, orient=tk.VERTICAL, command=preset_list.yview)
        preset_list.configure(yscrollcommand=preset_scroll.set)
        preset_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        preset_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._new_document_preset_canvas = preset_list
        preset_items: list[tuple[int, int, int]] = []
        for index, preset in enumerate(presets):
            y1 = index * preset_row_height
            tag = f"preset-{index}"
            rectangle = preset_list.create_rectangle(
                0,
                y1,
                preset_list_width,
                y1 + preset_row_height,
                fill="#ffffff",
                outline="#e0e3e7",
                tags=(tag, "preset"),
            )
            name_item = preset_list.create_text(
                12,
                y1 + 16,
                text=str(preset["name"]),
                anchor=tk.W,
                fill="#15191e",
                font=("Segoe UI Semibold", 10),
                tags=(tag, "preset"),
            )
            size_item = preset_list.create_text(
                12,
                y1 + 36,
                text=f"Размер холста: {preset['width']} x {preset['height']} px",
                anchor=tk.W,
                fill="#626a73",
                font=("Segoe UI", 9),
                tags=(tag, "preset"),
            )
            preset_items.append((rectangle, name_item, size_item))
            preset_list.tag_bind(tag, "<Button-1>", lambda _event, selected=index: select_preset(selected))
        preset_list.configure(scrollregion=(0, 0, preset_list_width, len(presets) * preset_row_height))

        right_panel = ttk.Frame(body)
        right_panel.grid(row=0, column=1, rowspan=2, sticky="new")
        preview_panel = ttk.Frame(right_panel)
        preview_panel.pack(fill=tk.X)
        ttk.Label(preview_panel, text="Предпросмотр", font=("Segoe UI Semibold", 12)).pack(anchor=tk.W, pady=(0, 6))
        current_size_label = ttk.Label(preview_panel, text="", font=("Segoe UI Semibold", 14))
        current_size_label.pack(anchor=tk.W, pady=(0, 7))
        self._new_document_size_label = current_size_label
        preview = tk.Canvas(preview_panel, width=330, height=210, background="#25282d", highlightthickness=0)
        preview.pack()
        preset_description = ttk.Label(preview_panel, text="", wraplength=330, justify=tk.LEFT)
        preset_description.pack(fill=tk.X, pady=(7, 0))

        settings = ttk.LabelFrame(right_panel, text="Параметры")
        settings.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(settings, text="Ширина, px").grid(row=0, column=0, sticky="w", padx=10, pady=(10, 4))
        width_entry = ttk.Spinbox(settings, textvariable=width, from_=1, to=50000, width=12)
        width_entry.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=(10, 4))
        ttk.Label(settings, text="Высота, px").grid(row=1, column=0, sticky="w", padx=10, pady=4)
        height_entry = ttk.Spinbox(settings, textvariable=height, from_=1, to=50000, width=12)
        height_entry.grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=4)
        swap_button = ttk.Button(settings, text="Поменять ориентацию")
        swap_button.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=4)
        ttk.Label(settings, text="DPI").grid(row=3, column=0, sticky="w", padx=10, pady=4)
        ttk.Spinbox(settings, textvariable=dpi, from_=1, to=2400, width=12).grid(row=3, column=1, sticky="ew", padx=(0, 10), pady=4)
        ttk.Label(settings, text="Фон").grid(row=4, column=0, sticky="w", padx=10, pady=4)
        ttk.Combobox(settings, textvariable=background, values=list(DOCUMENT_BACKGROUNDS), state="readonly", width=16).grid(row=4, column=1, sticky="ew", padx=(0, 10), pady=4)
        clipboard_check = ttk.Checkbutton(settings, text="Добавить изображение из буфера", variable=include_clipboard)
        if clipboard_image is not None:
            clipboard_check.grid(row=5, column=0, columnspan=2, sticky="w", padx=10, pady=(5, 10))
        settings.columnconfigure(1, weight=1)

        def safe_dimension(variable: tk.IntVar, fallback: int) -> int:
            try:
                return max(1, min(50000, int(variable.get())))
            except (tk.TclError, ValueError):
                return fallback

        def update_preview(*_args) -> None:
            canvas_width = safe_dimension(width, 1280)
            canvas_height = safe_dimension(height, 900)
            current_size_label.configure(text=f"{canvas_width} x {canvas_height} px")
            preview.delete("all")
            margin = 16
            scale = min((330 - margin * 2) / canvas_width, (210 - margin * 2) / canvas_height)
            shown_width = max(1, round(canvas_width * scale))
            shown_height = max(1, round(canvas_height * scale))
            x1 = (330 - shown_width) // 2
            y1 = (210 - shown_height) // 2
            x2, y2 = x1 + shown_width, y1 + shown_height
            if background.get() == "Прозрачный":
                tile = max(5, min(12, shown_width // 12 if shown_width else 5))
                for py in range(y1, y2, tile):
                    for px in range(x1, x2, tile):
                        color = "#d4d6d8" if ((px - x1) // tile + (py - y1) // tile) % 2 == 0 else "#ffffff"
                        preview.create_rectangle(px, py, min(px + tile, x2), min(py + tile, y2), fill=color, outline="")
            else:
                fill = "#ffffff" if background.get() == "Белый" else "#050505"
                preview.create_rectangle(x1, y1, x2, y2, fill=fill, outline="")
            if clipboard_image is not None and include_clipboard.get():
                image = clipboard_image.copy()
                image.thumbnail((shown_width, shown_height), Image.Resampling.LANCZOS)
                self._new_document_preview = ImageTk.PhotoImage(image)
                preview.create_image((x1 + x2) // 2, (y1 + y2) // 2, image=self._new_document_preview)
            preview.create_rectangle(x1, y1, x2, y2, outline="#7d8794", width=1)

        selected_preset_index = 0

        def paint_preset_selection() -> None:
            for index, (rectangle, name_item, size_item) in enumerate(preset_items):
                selected = index == selected_preset_index
                preset_list.itemconfigure(rectangle, fill="#1976d2" if selected else "#ffffff")
                preset_list.itemconfigure(name_item, fill="#ffffff" if selected else "#15191e")
                preset_list.itemconfigure(size_item, fill="#dcecff" if selected else "#626a73")

        def select_preset(index: int) -> None:
            nonlocal selected_preset_index
            selected_preset_index = max(0, min(len(presets) - 1, int(index)))
            preset = presets[selected_preset_index]
            width.set(int(preset["width"]))
            height.set(int(preset["height"]))
            dpi.set(int(preset["dpi"]))
            background.set(str(preset["background"]))
            include_clipboard.set(bool(preset.get("clipboard", False)))
            preset_description.configure(text=str(preset["description"]))
            paint_preset_selection()
            row_top = selected_preset_index * preset_row_height
            row_bottom = row_top + preset_row_height
            visible_top = preset_list.canvasy(0)
            visible_bottom = visible_top + preset_list.winfo_height()
            total_height = max(1, len(presets) * preset_row_height)
            if preset_list.winfo_height() > preset_row_height:
                if row_top < visible_top:
                    preset_list.yview_moveto(row_top / total_height)
                elif row_bottom > visible_bottom:
                    preset_list.yview_moveto(max(0.0, (row_bottom - preset_list.winfo_height()) / total_height))
            update_preview()

        def move_preset_selection(delta: int) -> str:
            select_preset(selected_preset_index + delta)
            return "break"

        def scroll_presets(event) -> str:
            preset_list.yview_scroll(-1 if event.delta > 0 else 1, "units")
            return "break"

        def swap_orientation() -> None:
            old_width = safe_dimension(width, 1280)
            old_height = safe_dimension(height, 900)
            width.set(old_height)
            height.set(old_width)
            update_preview()

        def accept() -> None:
            nonlocal result
            try:
                canvas_width = int(width.get())
                canvas_height = int(height.get())
                canvas_dpi = int(dpi.get())
            except (tk.TclError, ValueError):
                messagebox.showerror("Новый холст", "Укажите целые числа для размера и DPI.", parent=dialog)
                return
            if not (1 <= canvas_width <= 50000 and 1 <= canvas_height <= 50000 and 1 <= canvas_dpi <= 2400):
                messagebox.showerror("Новый холст", "Размер должен быть от 1 до 50000 px, DPI - от 1 до 2400.", parent=dialog)
                return
            result = {
                "width": canvas_width,
                "height": canvas_height,
                "dpi": canvas_dpi,
                "background": DOCUMENT_BACKGROUNDS.get(background.get(), DOCUMENT_BACKGROUNDS["Белый"]),
                "include_clipboard": bool(include_clipboard.get() and clipboard_image is not None),
            }
            selected_preset = presets[selected_preset_index]
            uses_custom_size = selected_preset["name"] == "Свой размер"
            uses_custom_size = uses_custom_size or canvas_width != int(selected_preset["width"]) or canvas_height != int(selected_preset["height"])
            if uses_custom_size:
                self.remember_custom_canvas(canvas_width, canvas_height, canvas_dpi, background.get())
            dialog.destroy()

        def accept_preset(index: int) -> str:
            select_preset(index)
            accept()
            return "break"

        swap_button.configure(command=swap_orientation)
        for index in range(len(presets)):
            preset_list.tag_bind(
                f"preset-{index}",
                "<Double-Button-1>",
                lambda _event, selected=index: accept_preset(selected),
            )
        self._new_document_accept_preset = accept_preset
        preset_list.bind("<Up>", lambda _event: move_preset_selection(-1))
        preset_list.bind("<Down>", lambda _event: move_preset_selection(1))
        preset_list.bind("<MouseWheel>", scroll_presets)
        for variable in (width, height, background, include_clipboard):
            variable.trace_add("write", update_preview)
        buttons = ttk.Frame(dialog)
        buttons.pack(fill=tk.X, padx=14, pady=(0, 14))
        ttk.Button(buttons, text="Создать", command=accept).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT)
        ToolTip(preset_list, "Выберите готовый формат, затем при необходимости измените его параметры.")
        select_preset(0)
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        self.center_toplevel(dialog, 820, 680)
        preset_list.yview_moveto(0.0)
        dialog.wait_window()
        return result

    def remember_custom_canvas(self, width: int, height: int, dpi: int, background: str) -> None:
        self.custom_canvas_width = max(1, min(50000, int(width)))
        self.custom_canvas_height = max(1, min(50000, int(height)))
        self.custom_canvas_dpi = max(1, min(2400, int(dpi)))
        self.custom_canvas_background = background if background in DOCUMENT_BACKGROUNDS else "Белый"
        self.save_settings()

    def create_document_from_settings(self, settings: dict[str, object], clipboard_image: Image.Image | None = None) -> None:
        width = int(settings["width"])
        height = int(settings["height"])
        dpi = int(settings["dpi"])
        background = tuple(settings["background"])
        include_clipboard = bool(settings.get("include_clipboard", False) and clipboard_image is not None)
        self.doc = Document.new(width, height, background)
        self._edit_generation += 1
        self.doc.dpi = dpi
        self.doc.metadata = {"source": "clipboard" if include_clipboard else "new document", "preset_size": [width, height]}
        if include_clipboard and clipboard_image is not None:
            canvas = Image.new("RGBA", (width, height), background)
            image = clipboard_image.convert("RGBA").copy()
            if image.width > width or image.height > height:
                image.thumbnail((width, height), Image.Resampling.LANCZOS)
            canvas.alpha_composite(image, ((width - image.width) // 2, (height - image.height) // 2))
            self.doc.layer.pixels = np.array(canvas, dtype=np.uint8)
            self.doc.layer.name = "Из буфера обмена"
        self.history.clear()
        self.selection_box = None
        self.show_editor()

    def create_from_clipboard(self) -> None:
        image = self.read_clipboard_image() or getattr(self, "_startup_clipboard_image", None)
        if image is None:
            messagebox.showinfo("Буфер обмена", "В буфере больше нет изображения.")
            return
        self.create_document_from_settings(
            {"width": image.width, "height": image.height, "dpi": 72, "background": DOCUMENT_BACKGROUNDS["Прозрачный"], "include_clipboard": True},
            image,
        )

    def new_from_preset(self) -> None:
        self.new_document()

    def open_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Проекты, изображения и RAW", "*.prdx *.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.dng *.arw *.cr2 *.cr3 *.nef *.nrw *.orf *.raf *.rw2 *.pef *.raw"), ("Все файлы", "*.*")])
        if not path:
            return
        self.open_path(path)

    def open_path(self, path: str) -> None:
        if not Path(path).exists():
            messagebox.showerror("Открытие", f"Файл не найден:\n{path}")
            self.recent_files = [item for item in self.recent_files if item.lower() != path.lower()]
            self.refresh_recent_menu()
            return
        try:
            document = Document.open_project(path) if path.lower().endswith(".prdx") else Document.from_image(path)
        except Exception as exc:
            messagebox.showerror("Открытие", f"Не удалось открыть файл:\n{exc}")
            return
        self.doc = document
        self._edit_generation += 1
        self.history.clear()
        self.selection_box = self.doc.selection_bounds()
        self.add_recent_file(path)
        self.show_editor()

    def place_embedded(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"), ("All", "*.*")])
        if not path:
            return
        self.run_document_command("Place embedded", lambda: self.doc.place_image(path))
        self.add_recent_file(path)
        self.refresh()

    def place_linked(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"), ("All", "*.*")])
        if not path:
            return
        self.run_document_command("Place linked", lambda: self.doc.place_image(path, linked=True))
        self.add_recent_file(path)
        self.refresh()

    def update_linked_layer(self) -> None:
        layer = self.doc.layer
        source_path = (layer.smart_data or {}).get("source_path")
        if layer.kind != "linked" or not source_path:
            messagebox.showinfo("Linked layer", "The active layer is not linked to an external file.")
            return
        if not Path(source_path).exists():
            messagebox.showerror("Linked layer", f"Linked source file not found:\n{source_path}")
            return
        self.run_document_command("Update linked layer", self.doc.update_linked_layer)
        self.refresh()

    def show_linked_layer_status(self) -> None:
        status = self.doc.linked_layer_status()
        labels = {
            "embedded": "Объект встроен в проект",
            "current": "Связанный файл актуален",
            "modified": "Связанный файл изменён вне редактора",
            "missing": "Связанный файл не найден",
        }
        messagebox.showinfo("Smart Object", f"{labels.get(status['status'], status['status'])}\n\n{status.get('path') or ''}")

    def replace_smart_contents(self) -> None:
        if self.doc.layer.kind not in {"linked", "embedded"}:
            messagebox.showinfo("Smart Object", "Активный слой не является Smart Object.")
            return
        path = filedialog.askopenfilename(filetypes=[("Изображения", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"), ("Все файлы", "*.*")])
        if not path:
            return
        linked = messagebox.askyesno("Smart Object", "Оставить содержимое связанным с внешним файлом?")
        self.run_document_command("Заменить содержимое Smart Object", lambda: self.doc.replace_active_smart_contents(path, linked))
        self.refresh()

    def convert_smart_to_embedded(self) -> None:
        if self.doc.layer.kind not in {"linked", "embedded"}:
            messagebox.showinfo("Smart Object", "Активный слой не является Smart Object.")
            return
        self.run_document_command("Преобразовать Smart Object во встроенный", self.doc.convert_active_smart_to_embedded)
        self.refresh()

    def reset_smart_transform(self) -> None:
        self.run_document_command("Сбросить трансформацию Smart Object", self.doc.reset_active_smart_transform)
        self.refresh()

    def relink_layer(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"), ("All", "*.*")])
        if not path:
            return
        self.run_document_command("Relink layer", lambda: self.doc.relink_active_layer(path))
        self.add_recent_file(path)
        self.refresh()

    def load_files_as_layers(self) -> None:
        paths = filedialog.askopenfilenames(filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"), ("All", "*.*")])
        if not paths:
            return

        def edit():
            for path in paths:
                self.doc.place_image(path)

        self.run_document_command("Load files as layers", edit)
        for path in paths:
            self.add_recent_file(path)
        self.refresh()

    def save(self) -> None:
        if self.doc.path and self.doc.path.lower().endswith(".prdx"):
            self.save_project_async(self.doc.path)
        else:
            self.save_as_project()

    def save_as_project(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".prdx", filetypes=[("PhotoRedactor project", "*.prdx")])
        if path:
            self.save_project_async(path)

    def save_project_async(self, path: str) -> None:
        snapshot = self.document_copy()
        generation = self._edit_generation

        def worker():
            snapshot.save_project(path)
            return path

        def done(saved_path):
            self.doc.path = saved_path
            if self._edit_generation == generation:
                self.doc.dirty = False
            self.add_recent_file(saved_path)

        self.run_background("Save project", worker, done)

    def export_image(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("WebP", "*.webp"), ("TIFF", "*.tiff"), ("BMP", "*.bmp")])
        if path:
            suffix = Path(path).suffix.lower()
            quality = 95
            if suffix in {".jpg", ".jpeg", ".webp"}:
                quality = simpledialog.askinteger("Export quality", "Quality 1..100:", initialvalue=95, minvalue=1, maxvalue=100) or 95
            snapshot = self.document_copy()

            def worker():
                snapshot.export_flat(path, quality)
                return path

            self.run_background("Export image", worker)

    def pick_foreground(self) -> None:
        color = colorchooser.askcolor(color=self.color_hex(self.foreground), title="Основной цвет")[0]
        if color:
            self.foreground = tuple(map(int, color)) + (255,)
            self.refresh_color_control()
            self.text_properties_changed()

    def pick_background(self) -> None:
        color = colorchooser.askcolor(color=self.color_hex(self.background), title="Дополнительный цвет")[0]
        if color:
            self.background = tuple(map(int, color)) + (255,)
            self.refresh_color_control()

    def pick_color_from_document(self, point: tuple[int, int]) -> None:
        x, y = point
        if x < 0 or y < 0 or x >= self.doc.width or y >= self.doc.height:
            return
        rgba = self.render_engine.render(self.doc, False)[y, x]
        self.foreground = tuple(int(v) for v in rgba)
        self.refresh_color_control()
        self.status_text(f"Основной цвет: {self.color_hex(self.foreground).upper()}")

    def show_image_statistics(self) -> None:
        stats = image_statistics(self.render_engine.render(self.doc, False))
        text = [
            f"Size: {stats['width']} x {stats['height']}",
            f"Opaque pixels: {stats['opaque_pixels']}",
            f"Transparent pixels: {stats['transparent_pixels']}",
            "",
        ]
        for name, values in stats["channels"].items():
            text.append(f"{name}: min {values['min']:.2f}, max {values['max']:.2f}, mean {values['mean']:.2f}, std {values['std']:.2f}")
        messagebox.showinfo("Image statistics", "\n".join(text))

    def show_histogram(self) -> None:
        stats = image_statistics(self.render_engine.render(self.doc, False))
        lines = []
        for channel, values in stats["histogram"].items():
            lines.append(channel.upper())
            peak = max(values) or 1
            for i, value in enumerate(values):
                start = i * 16
                end = start + 15
                bar = "#" * max(1, round(value / peak * 32)) if value else ""
                lines.append(f"{start:03d}-{end:03d}: {bar} {value}")
            lines.append("")
        self.show_text_window("Histogram", "\n".join(lines))

    def show_metadata(self) -> None:
        text = json.dumps(self.doc.metadata or {}, ensure_ascii=False, indent=2)
        self.show_text_window("Metadata / EXIF", text if text != "{}" else "No metadata.")

    def edit_metadata(self) -> None:
        window = tk.Toplevel(self)
        window.title("Редактор метаданных")
        window.geometry("720x500")
        window.transient(self)
        working = copy.deepcopy(self.doc.metadata or {})
        tree = ttk.Treeview(window, columns=("value",), show="tree headings")
        tree.heading("#0", text="Поле")
        tree.heading("value", text="Значение")
        tree.column("#0", width=230)
        tree.column("value", width=450)
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 4))

        def refill() -> None:
            tree.delete(*tree.get_children())
            for key, value in sorted(working.items(), key=lambda item: str(item[0]).lower()):
                tree.insert("", tk.END, iid=str(key), text=str(key), values=(json.dumps(value, ensure_ascii=False),))

        def set_value() -> None:
            selected = tree.selection()
            old_key = selected[0] if selected else ""
            key = simpledialog.askstring("Метаданные", "Название поля:", initialvalue=old_key, parent=window)
            if not key:
                return
            initial = json.dumps(working.get(old_key, ""), ensure_ascii=False)
            raw = simpledialog.askstring("Метаданные", "Значение:", initialvalue=initial, parent=window)
            if raw is None:
                return
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = raw
            if old_key and old_key != key:
                working.pop(old_key, None)
            working[key] = value
            refill()

        def remove_value() -> None:
            for key in tree.selection():
                working.pop(key, None)
            refill()

        buttons = ttk.Frame(window)
        buttons.pack(fill=tk.X, padx=10, pady=(4, 10))
        ttk.Button(buttons, text="Добавить / изменить", command=set_value).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Удалить", command=remove_value).pack(side=tk.LEFT, padx=6)

        def apply() -> None:
            self.run_document_command("Редактировать метаданные", lambda: setattr(self.doc, "metadata", copy.deepcopy(working)))
            self.doc.dirty = True
            window.destroy()
            self.refresh()

        ttk.Button(buttons, text="Применить", command=apply).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Отмена", command=window.destroy).pack(side=tk.RIGHT, padx=6)
        refill()

    def show_cache_status(self) -> None:
        status = self.render_engine.cache_status()
        gpu = status["gpu"]
        text = (
            f"Кэш в памяти: {status['memory_bytes'] / 1024 / 1024:.1f} МБ\n"
            f"Объектов в памяти: {status['memory_items']}\n"
            f"Объектов на scratch-диске: {status['disk_items']}\n"
            f"GPU доступен: {'да' if gpu['available'] else 'нет'}\n"
            f"GPU включен: {'да' if gpu['enabled'] else 'нет'}\n"
            f"Устройств: {gpu['devices']}"
        )
        messagebox.showinfo("Большие документы", text)

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

    def free_transform_layer(self) -> None:
        layer = self.doc.layer
        if layer.locked:
            self.status_text("Слой заблокирован")
            return
        if layer.kind == "text" and layer.text_data is not None:
            self.transform_text_box()
            return
        data = self.free_transform_dialog(layer)
        if data is None:
            return
        self.run_document_command(
            "Free transform",
            lambda: self.doc.transform_active_layer(
                int(data["x"]), int(data["y"]), int(data["width"]), int(data["height"]),
                float(data["angle"]), bool(data["flip_horizontal"]), bool(data["flip_vertical"]),
            ),
        )
        self.refresh()

    def free_transform_dialog(self, layer) -> dict[str, object] | None:
        dialog = tk.Toplevel(self)
        dialog.title("Свободная трансформация")
        dialog.transient(self)
        dialog.resizable(False, False)
        dialog.grab_set()
        result: dict[str, object] | None = None
        x_var = tk.IntVar(value=int(layer.x))
        y_var = tk.IntVar(value=int(layer.y))
        width_var = tk.IntVar(value=int(layer.pixels.shape[1]))
        height_var = tk.IntVar(value=int(layer.pixels.shape[0]))
        angle_var = tk.DoubleVar(value=0.0)
        flip_h_var = tk.BooleanVar(value=False)
        flip_v_var = tk.BooleanVar(value=False)
        keep_ratio = tk.BooleanVar(value=True)
        original_ratio = layer.pixels.shape[1] / max(1, layer.pixels.shape[0])

        canvas = tk.Canvas(dialog, width=500, height=340, background="#22252b", highlightthickness=0, cursor="crosshair")
        canvas.grid(row=0, column=0, rowspan=9, padx=12, pady=12)
        source = rgba_array_to_pil(layer.pixels)
        handle_positions: dict[str, tuple[float, float]] = {}
        drag_state: dict[str, object] = {"handle": None, "last": (0, 0)}

        def safe_values() -> tuple[int, int, int, int, float]:
            try:
                return int(x_var.get()), int(y_var.get()), max(1, int(width_var.get())), max(1, int(height_var.get())), float(angle_var.get())
            except (tk.TclError, ValueError):
                return layer.x, layer.y, layer.pixels.shape[1], layer.pixels.shape[0], 0.0

        def preview_geometry() -> tuple[float, float, float]:
            scale = min(460 / max(1, self.doc.width), 300 / max(1, self.doc.height))
            return scale, (500 - self.doc.width * scale) / 2, (340 - self.doc.height * scale) / 2

        def redraw(*_args) -> None:
            x, y, width, height, angle = safe_values()
            scale, ox, oy = preview_geometry()
            canvas.delete("all")
            canvas.create_rectangle(ox, oy, ox + self.doc.width * scale, oy + self.doc.height * scale, fill="#343840", outline="#707680")
            preview = source.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BILINEAR)
            if flip_h_var.get():
                preview = preview.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if flip_v_var.get():
                preview = preview.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            if abs(angle) > 0.001:
                preview = preview.rotate(-angle, expand=True, resample=Image.Resampling.BICUBIC)
            self._transform_preview_image = ImageTk.PhotoImage(preview)
            canvas.create_image(ox + (x + width / 2) * scale, oy + (y + height / 2) * scale, image=self._transform_preview_image)
            x1, y1 = ox + x * scale, oy + y * scale
            x2, y2 = ox + (x + width) * scale, oy + (y + height) * scale
            canvas.create_rectangle(x1, y1, x2, y2, outline="#50e3ff", width=2, dash=(5, 3))
            positions = {
                "nw": (x1, y1), "n": ((x1 + x2) / 2, y1), "ne": (x2, y1), "e": (x2, (y1 + y2) / 2),
                "se": (x2, y2), "s": ((x1 + x2) / 2, y2), "sw": (x1, y2), "w": (x1, (y1 + y2) / 2),
            }
            handle_positions.clear(); handle_positions.update(positions)
            for hx, hy in positions.values():
                canvas.create_rectangle(hx - 5, hy - 5, hx + 5, hy + 5, fill="#f7f9fb", outline="#167d96")

        def press(event) -> None:
            nearest = None
            distance = 12.0
            for name, (hx, hy) in handle_positions.items():
                current = ((event.x - hx) ** 2 + (event.y - hy) ** 2) ** 0.5
                if current < distance:
                    nearest, distance = name, current
            if nearest is None:
                x, y, width, height, _angle = safe_values()
                scale, ox, oy = preview_geometry()
                if ox + x * scale <= event.x <= ox + (x + width) * scale and oy + y * scale <= event.y <= oy + (y + height) * scale:
                    nearest = "move"
            drag_state["handle"] = nearest
            drag_state["last"] = (event.x, event.y)

        def drag(event) -> None:
            handle = drag_state.get("handle")
            if not handle:
                return
            last_x, last_y = drag_state["last"]
            scale, _ox, _oy = preview_geometry()
            dx, dy = (event.x - last_x) / scale, (event.y - last_y) / scale
            x, y, width, height, _angle = safe_values()
            if handle == "move":
                x += round(dx); y += round(dy)
            else:
                if "w" in str(handle): x += round(dx); width -= round(dx)
                if "e" in str(handle): width += round(dx)
                if "n" in str(handle): y += round(dy); height -= round(dy)
                if "s" in str(handle): height += round(dy)
                width, height = max(1, width), max(1, height)
                if keep_ratio.get() and str(handle) in {"nw", "ne", "se", "sw"}:
                    if abs(dx) >= abs(dy): height = max(1, round(width / original_ratio))
                    else: width = max(1, round(height * original_ratio))
            x_var.set(x); y_var.set(y); width_var.set(width); height_var.set(height)
            drag_state["last"] = (event.x, event.y)

        for row, (label, variable) in enumerate([("X", x_var), ("Y", y_var), ("Ширина", width_var), ("Высота", height_var), ("Поворот", angle_var)]):
            ttk.Label(dialog, text=label).grid(row=row, column=1, sticky="w", padx=(0, 8), pady=(12 if row == 0 else 4, 0))
            ttk.Spinbox(dialog, textvariable=variable, from_=-100000 if row < 2 else 1, to=100000, width=12).grid(row=row, column=2, sticky="ew", padx=(0, 12), pady=(12 if row == 0 else 4, 0))
        ttk.Checkbutton(dialog, text="Сохранять пропорции", variable=keep_ratio).grid(row=5, column=1, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(dialog, text="Отразить по горизонтали", variable=flip_h_var, command=redraw).grid(row=6, column=1, columnspan=2, sticky="w")
        ttk.Checkbutton(dialog, text="Отразить по вертикали", variable=flip_v_var, command=redraw).grid(row=7, column=1, columnspan=2, sticky="w")
        buttons = ttk.Frame(dialog)
        buttons.grid(row=8, column=1, columnspan=2, sticky="e", padx=12, pady=12)

        def accept() -> None:
            nonlocal result
            x, y, width, height, angle = safe_values()
            result = {"x": x, "y": y, "width": width, "height": height, "angle": angle, "flip_horizontal": flip_h_var.get(), "flip_vertical": flip_v_var.get()}
            dialog.destroy()

        ttk.Button(buttons, text="ОК", command=accept).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT)
        canvas.bind("<ButtonPress-1>", press)
        canvas.bind("<B1-Motion>", drag)
        for variable in [x_var, y_var, width_var, height_var, angle_var]: variable.trace_add("write", redraw)
        redraw()
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
        initial = f"{layer.x + lx1},{layer.y + ly1},{lx2 - lx1},{ly2 - ly1},0,false,false"
        raw = simpledialog.askstring("Трансформация выделенных пикселей", "x,y,width,height,rotation,flipH,flipV:", initialvalue=initial)
        if not raw:
            return
        try:
            parts = [part.strip() for part in raw.split(",")]
            if len(parts) != 7:
                raise ValueError
            x, y, width, height = [int(float(value)) for value in parts[:4]]
            angle = float(parts[4])
            flip_h = parts[5].lower() in {"1", "true", "yes", "y", "да", "д"}
            flip_v = parts[6].lower() in {"1", "true", "yes", "y", "да", "д"}
        except ValueError:
            messagebox.showerror("Трансформация выделенных пикселей", "Используйте: x,y,width,height,rotation,flipH,flipV")
            return
        self.run_document_command(
            "Transform selected pixels",
            lambda: self.doc.transform_selected_pixels(x, y, width, height, angle, flip_h, flip_v),
        )
        self.refresh()

    def perspective_transform_layer(self) -> None:
        layer = self.doc.layer
        if layer.locked:
            self.status_text("Слой заблокирован")
            return
        w, h = layer.pixels.shape[1], layer.pixels.shape[0]
        initial = f"{layer.x},{layer.y},{layer.x + w},{layer.y},{layer.x + w},{layer.y + h},{layer.x},{layer.y + h}"
        raw = simpledialog.askstring(
            "Perspective transform",
            "TLx,TLy,TRx,TRy,BRx,BRy,BLx,BLy:",
            initialvalue=initial,
        )
        if not raw:
            return
        try:
            values = [float(part.strip()) for part in raw.split(",")]
            if len(values) != 8:
                raise ValueError
            corners = [(values[i], values[i + 1]) for i in range(0, 8, 2)]
        except ValueError:
            messagebox.showerror("Perspective transform", "Use: TLx,TLy,TRx,TRy,BRx,BRy,BLx,BLy")
            return
        self.run_document_command("Perspective transform", lambda: self.doc.perspective_transform_active_layer(corners))
        self.refresh()

    def warp_layer(self) -> None:
        layer = self.doc.layer
        if layer.locked:
            self.status_text("Слой заблокирован")
            return
        data = self.warp_layer_dialog(layer)
        if data is None:
            return
        self.run_document_command("Warp layer", lambda: self.doc.warp_active_layer(str(data["mode"]), float(data["amount"]), float(data["wavelength"])))
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

    def toggle_clipping_mask(self) -> None:
        if self.doc.active_layer <= 0:
            messagebox.showinfo("Clipping mask", "The bottom layer cannot be clipped.")
            return
        self.set_layer_property("Toggle clipping mask", "clipping", not self.doc.layer.clipping)

    def edit_layer_styles(self) -> None:
        layer = self.doc.layer
        initial = self.effects_to_text(layer.effects)
        raw = simpledialog.askstring(
            "Layer styles",
            "Use semicolon-separated styles:\nstroke,size,opacity\nshadow,x,y,blur,opacity\nglow,blur,opacity\nempty clears styles",
            initialvalue=initial,
        )
        if raw is None:
            return
        try:
            effects = self.parse_effects(raw)
        except ValueError:
            messagebox.showerror("Layer styles", "Invalid style string.")
            return
        self.set_layer_property("Layer styles", "effects", effects, preserve_render_cache=False)

    def edit_layer_filters(self) -> None:
        layer = self.doc.layer
        filters = self.layer_filters_dialog(layer.filters, layer.pixels, self.doc.layer_selection_mask(layer))
        if filters is None:
            return
        self.set_layer_property("Layer filters", "filters", filters, preserve_render_cache=False)

    def clear_layer_filters(self) -> None:
        self.set_layer_property("Clear layer filters", "filters", [], preserve_render_cache=False)

    def layer_filters_dialog(self, initial_filters: list[dict], pixels: np.ndarray, selection_mask: np.ndarray | None = None) -> list[dict] | None:
        filters = []
        for item in initial_filters:
            normalized = self.normalize_filter_item(item)
            if normalized is not None:
                filters.append(normalized)
        result: list[dict] | None = None
        updating_controls = False
        dialog = tk.Toplevel(self)
        dialog.title("Фильтры слоя")
        dialog.transient(self)
        dialog.resizable(False, False)
        dialog.grab_set()

        preview = ttk.Label(dialog)
        preview.grid(row=0, column=0, rowspan=11, padx=12, pady=12, sticky="n")
        listbox = tk.Listbox(dialog, height=8, width=26, exportselection=False)
        listbox.grid(row=0, column=1, rowspan=6, padx=(0, 8), pady=12, sticky="ns")

        controls = ttk.Frame(dialog)
        controls.grid(row=0, column=2, padx=(0, 12), pady=12, sticky="new")
        ttk.Label(controls, text="Пресет").grid(row=0, column=0, sticky="w")
        filter_preset = tk.StringVar(value=next(iter(FILTER_STACK_PRESETS)))
        preset_box = ttk.Combobox(controls, textvariable=filter_preset, values=list(FILTER_STACK_PRESETS), state="readonly", width=14)
        preset_box.grid(row=0, column=1, sticky="ew", pady=(0, 6))
        ttk.Button(controls, text="Применить", command=lambda: apply_preset()).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        ttk.Label(controls, text="Тип").grid(row=2, column=0, sticky="w")
        filter_type = tk.StringVar(value=FILTER_TYPES[0])
        type_box = ttk.Combobox(controls, textvariable=filter_type, values=FILTER_TYPES, state="readonly", width=14)
        type_box.grid(row=2, column=1, sticky="ew", pady=(0, 6))
        ttk.Label(controls, text="Параметр").grid(row=3, column=0, sticky="w")
        value_var = tk.DoubleVar(value=3.0)
        value_spin = ttk.Spinbox(controls, textvariable=value_var, from_=0.0, to=500.0, increment=1.0, width=12)
        value_spin.grid(row=3, column=1, sticky="ew", pady=(0, 8))
        enabled_var = tk.BooleanVar(value=True)
        enabled_check = ttk.Checkbutton(controls, text="Включен", variable=enabled_var)
        enabled_check.grid(row=4, column=0, columnspan=2, sticky="w", pady=(0, 6))
        ttk.Label(controls, text="Opacity").grid(row=5, column=0, sticky="w")
        opacity_var = tk.DoubleVar(value=100.0)
        opacity_spin = ttk.Spinbox(controls, textvariable=opacity_var, from_=0.0, to=100.0, increment=5.0, width=12)
        opacity_spin.grid(row=5, column=1, sticky="ew", pady=(0, 8))
        ttk.Label(controls, text="Режим").grid(row=6, column=0, sticky="w")
        filter_blend_mode = tk.StringVar(value="Normal")
        filter_blend_box = ttk.Combobox(controls, textvariable=filter_blend_mode, values=BLEND_MODES, state="readonly", width=14)
        filter_blend_box.grid(row=6, column=1, sticky="ew", pady=(0, 8))
        hint = ttk.Label(controls, text="", wraplength=190, justify=tk.LEFT)
        hint.grid(row=7, column=0, columnspan=2, sticky="w", pady=(0, 8))

        buttons = ttk.Frame(dialog)
        buttons.grid(row=6, column=1, columnspan=2, sticky="ew", padx=(0, 12), pady=(0, 8))
        mask_buttons = ttk.Frame(dialog)
        mask_buttons.grid(row=7, column=1, columnspan=2, sticky="ew", padx=(0, 12), pady=(0, 8))
        preset_file_buttons = ttk.Frame(dialog)
        preset_file_buttons.grid(row=8, column=1, columnspan=2, sticky="ew", padx=(0, 12), pady=(0, 8))
        bottom = ttk.Frame(dialog)
        bottom.grid(row=9, column=1, columnspan=2, sticky="e", padx=(0, 12), pady=(0, 12))

        def selected_index() -> int | None:
            selection = listbox.curselection()
            return None if not selection else int(selection[0])

        def item_text(item: dict) -> str:
            kind = str(item.get("type", "")).lower()
            label = FILTER_LABELS.get(kind, kind)
            value = self.filter_primary_value(item)
            enabled = bool(item.get("enabled", True))
            opacity = int(round(float(item.get("opacity", 1.0)) * 100))
            blend_mode = str(item.get("blend_mode", "Normal"))
            prefix = "" if enabled else "выкл. "
            mask = ", маска" if item.get("mask") else ""
            blend = "" if blend_mode == "Normal" else f", {blend_mode}"
            return f"{prefix}{label}: {value:g}, {opacity}%{blend}{mask}"

        def refresh_list(select_index: int | None = None) -> None:
            listbox.delete(0, tk.END)
            for item in filters:
                listbox.insert(tk.END, item_text(item))
            if filters:
                index = 0 if select_index is None else max(0, min(select_index, len(filters) - 1))
                listbox.selection_set(index)
                load_selected()
            update_preview()

        def update_value_controls(kind: str) -> None:
            if kind == "blur":
                value_spin.configure(from_=1, to=500, increment=1)
                hint.configure(text="Радиус размытия в пикселях.")
            elif kind == "sharpen":
                value_spin.configure(from_=0, to=10, increment=0.1)
                hint.configure(text="Сила повышения резкости.")
            elif kind == "noise":
                value_spin.configure(from_=0, to=1, increment=0.01)
                hint.configure(text="Количество детерминированного шума 0..1.")
            elif kind == "median":
                value_spin.configure(from_=3, to=101, increment=2)
                hint.configure(text="Нечетный размер медианного окна.")
            else:
                value_spin.configure(from_=0, to=1, increment=0.05)
                hint.configure(text="Сила смешивания эффекта 0..1.")

        def load_selected(_event=None) -> None:
            nonlocal updating_controls
            index = selected_index()
            if index is None:
                update_value_controls(filter_type.get())
                return
            updating_controls = True
            item = filters[index]
            kind = str(item.get("type", "blur")).lower()
            filter_type.set(kind if kind in FILTER_TYPES else "blur")
            update_value_controls(filter_type.get())
            value_var.set(self.filter_primary_value(item))
            enabled_var.set(bool(item.get("enabled", True)))
            opacity_var.set(float(item.get("opacity", 1.0)) * 100.0)
            blend_mode = str(item.get("blend_mode", "Normal"))
            filter_blend_mode.set(blend_mode if blend_mode in BLEND_MODES else "Normal")
            updating_controls = False

        def current_item(original: dict | None = None) -> dict:
            metadata = dict(original or {})
            metadata["enabled"] = bool(enabled_var.get())
            metadata["opacity"] = float(opacity_var.get()) / 100.0
            metadata["blend_mode"] = filter_blend_mode.get()
            return self.make_filter_item(filter_type.get(), value_var.get(), metadata)

        def apply_current(_event=None) -> None:
            if updating_controls:
                return
            index = selected_index()
            if index is None:
                return
            filters[index] = current_item(filters[index])
            refresh_list(index)

        def add_filter() -> None:
            filters.append(current_item())
            refresh_list(len(filters) - 1)

        def remove_filter() -> None:
            index = selected_index()
            if index is None:
                return
            del filters[index]
            refresh_list(min(index, len(filters) - 1) if filters else None)

        def move_filter(delta: int) -> None:
            index = selected_index()
            if index is None:
                return
            target = index + delta
            if target < 0 or target >= len(filters):
                return
            filters[index], filters[target] = filters[target], filters[index]
            refresh_list(target)

        def apply_preset() -> None:
            preset = FILTER_STACK_PRESETS.get(filter_preset.get())
            if preset is None:
                return
            filters.clear()
            for item in preset:
                normalized = self.normalize_filter_item(item)
                if normalized is not None:
                    filters.append(normalized)
            refresh_list(0 if filters else None)

        def load_preset_file() -> None:
            path = filedialog.askopenfilename(filetypes=[("PhotoRedactor filter preset", "*.json"), ("JSON", "*.json")])
            if not path:
                return
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
                raw_filters = data.get("filters") if isinstance(data, dict) else data
                if not isinstance(raw_filters, list):
                    raise ValueError
                loaded = []
                for item in raw_filters:
                    if isinstance(item, dict):
                        normalized = self.normalize_filter_item(item)
                        if normalized is not None:
                            loaded.append(normalized)
                if not loaded:
                    raise ValueError
            except Exception:
                messagebox.showerror("Пресет фильтров", "Не удалось загрузить пресет фильтров.")
                return
            filters.clear()
            filters.extend(loaded)
            refresh_list(0)

        def save_preset_file() -> None:
            apply_current()
            normalized_filters = []
            for item in filters:
                normalized = self.normalize_filter_item(item)
                if normalized is not None:
                    normalized_filters.append(normalized)
            if not normalized_filters:
                messagebox.showinfo("Пресет фильтров", "Добавьте хотя бы один фильтр перед сохранением пресета.")
                return
            path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("PhotoRedactor filter preset", "*.json"), ("JSON", "*.json")])
            if not path:
                return
            name = Path(path).stem
            data = {"format": "PhotoRedactor filter preset", "version": 1, "name": name, "filters": normalized_filters}
            try:
                Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as exc:
                messagebox.showerror("Пресет фильтров", str(exc))
                return
            self.status_text(f"Пресет фильтров сохранен: {name}")

        def set_filter_mask_from_selection() -> None:
            index = selected_index()
            if index is None:
                return
            if selection_mask is None or not np.any(selection_mask):
                messagebox.showinfo("Маска фильтра", "Создайте выделение на активном слое перед добавлением маски фильтра.")
                return
            filters[index] = current_item(filters[index])
            filters[index]["mask"] = encode_png(np.dstack([selection_mask] * 4))
            refresh_list(index)

        def clear_filter_mask() -> None:
            index = selected_index()
            if index is None:
                return
            filters[index] = current_item(filters[index])
            filters[index].pop("mask", None)
            refresh_list(index)

        preview_scale = min(1.0, 180 / max(1, pixels.shape[1]), 180 / max(1, pixels.shape[0]))
        preview_size = max(1, round(pixels.shape[1] * preview_scale)), max(1, round(pixels.shape[0] * preview_scale))
        preview_source = pixels.copy() if preview_size == (pixels.shape[1], pixels.shape[0]) else cv2.resize(pixels, preview_size, interpolation=cv2.INTER_AREA)

        def update_preview() -> None:
            shown = apply_filter_stack(preview_source, filters) if filters else preview_source
            image = rgba_array_to_pil(shown)
            canvas = Image.new("RGBA", (180, 180), (44, 46, 52, 255))
            canvas.alpha_composite(image, ((180 - image.width) // 2, (180 - image.height) // 2))
            self._filter_stack_preview_image = ImageTk.PhotoImage(canvas)
            preview.configure(image=self._filter_stack_preview_image)

        def accept() -> None:
            nonlocal result
            apply_current()
            result = []
            for item in filters:
                normalized = self.normalize_filter_item(item)
                if normalized is not None:
                    result.append(normalized)
            dialog.destroy()

        def cancel() -> None:
            dialog.destroy()

        ttk.Button(buttons, text="Добавить", command=add_filter).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(buttons, text="Удалить", command=remove_filter).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(buttons, text="Вверх", command=lambda: move_filter(-1)).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(buttons, text="Вниз", command=lambda: move_filter(1)).pack(side=tk.LEFT)
        ttk.Button(mask_buttons, text="Маска из выделения", command=set_filter_mask_from_selection).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(mask_buttons, text="Удалить маску", command=clear_filter_mask).pack(side=tk.LEFT)
        ttk.Button(preset_file_buttons, text="Загрузить пресет", command=load_preset_file).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(preset_file_buttons, text="Сохранить пресет", command=save_preset_file).pack(side=tk.LEFT)
        ttk.Button(bottom, text="ОК", command=accept).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(bottom, text="Отмена", command=cancel).pack(side=tk.RIGHT)
        listbox.bind("<<ListboxSelect>>", load_selected)
        type_box.bind("<<ComboboxSelected>>", lambda _event: (update_value_controls(filter_type.get()), apply_current()))
        value_spin.bind("<KeyRelease>", apply_current)
        value_spin.bind("<FocusOut>", apply_current)
        opacity_spin.bind("<KeyRelease>", apply_current)
        opacity_spin.bind("<FocusOut>", apply_current)
        filter_blend_box.bind("<<ComboboxSelected>>", apply_current)
        value_var.trace_add("write", lambda *_args: apply_current())
        enabled_var.trace_add("write", lambda *_args: apply_current())
        opacity_var.trace_add("write", lambda *_args: apply_current())
        filter_blend_mode.trace_add("write", lambda *_args: apply_current())
        dialog.protocol("WM_DELETE_WINDOW", cancel)
        refresh_list(0 if filters else None)
        if not filters:
            update_value_controls(filter_type.get())
            update_preview()
        dialog.wait_window()
        return result

    def normalize_filter_item(self, item: dict) -> dict | None:
        kind = str(item.get("type", "")).lower()
        if kind not in FILTER_TYPES:
            return None
        return self.make_filter_item(kind, self.filter_primary_value(item), item)

    def make_filter_item(self, kind: str, value: float, original: dict | None = None) -> dict:
        original = original or {}
        if kind == "blur":
            item = {"type": "blur", "radius": max(1, int(float(value)))}
        elif kind == "sharpen":
            item = {"type": "sharpen", "amount": max(0.0, float(value))}
        elif kind == "noise":
            item = {"type": "noise", "amount": max(0.0, min(1.0, float(value))), "seed": int(original.get("seed", 12345))}
        elif kind == "median":
            item = {"type": "median", "size": max(3, int(float(value)) | 1)}
        elif kind == "edge":
            item = {"type": "edge", "strength": max(0.0, min(1.0, float(value)))}
        else:
            item = {"type": "emboss", "strength": max(0.0, min(1.0, float(value)))}
        item["enabled"] = bool(original.get("enabled", True))
        item["opacity"] = max(0.0, min(1.0, float(original.get("opacity", 1.0))))
        blend_mode = str(original.get("blend_mode", "Normal"))
        item["blend_mode"] = blend_mode if blend_mode in BLEND_MODES else "Normal"
        if isinstance(original.get("mask"), str) and original.get("mask"):
            item["mask"] = original["mask"]
        return item

    @staticmethod
    def filter_primary_value(item: dict) -> float:
        kind = str(item.get("type", "")).lower()
        if kind == "blur":
            return float(item.get("radius", 3))
        if kind == "sharpen":
            return float(item.get("amount", 1.0))
        if kind == "noise":
            return float(item.get("amount", 0.03))
        if kind == "median":
            return float(item.get("size", 3))
        if kind in {"edge", "emboss"}:
            return float(item.get("strength", 1.0))
        return 1.0

    def effects_to_text(self, effects: dict) -> str:
        parts = []
        stroke = effects.get("stroke")
        if stroke:
            parts.append(f"stroke,{stroke.get('size', 4)},{stroke.get('opacity', 1.0)}")
        shadow = effects.get("drop_shadow")
        if shadow:
            parts.append(f"shadow,{shadow.get('x', 10)},{shadow.get('y', 10)},{shadow.get('blur', 12)},{shadow.get('opacity', 0.55)}")
        glow = effects.get("outer_glow")
        if glow:
            parts.append(f"glow,{glow.get('blur', 18)},{glow.get('opacity', 0.5)}")
        return ";".join(parts)

    def parse_effects(self, raw: str) -> dict:
        effects = {}
        if not raw.strip():
            return effects
        for chunk in raw.split(";"):
            parts = [part.strip() for part in chunk.split(",") if part.strip()]
            if not parts:
                continue
            kind = parts[0].lower()
            if kind == "stroke" and len(parts) == 3:
                effects["stroke"] = {"enabled": True, "size": int(float(parts[1])), "opacity": float(parts[2]), "color": list(self.background)}
            elif kind == "shadow" and len(parts) == 5:
                effects["drop_shadow"] = {"enabled": True, "x": int(float(parts[1])), "y": int(float(parts[2])), "blur": int(float(parts[3])), "opacity": float(parts[4]), "color": [0, 0, 0, 255]}
            elif kind == "glow" and len(parts) == 3:
                effects["outer_glow"] = {"enabled": True, "blur": int(float(parts[1])), "opacity": float(parts[2]), "color": list(self.foreground)}
            else:
                raise ValueError
        return effects

    def filters_to_text(self, filters: list[dict]) -> str:
        parts = []
        for item in filters:
            kind = str(item.get("type", "")).lower()
            if kind == "blur":
                parts.append(self.filter_text_chunk("blur", item.get("radius", 3), item))
            elif kind == "sharpen":
                parts.append(self.filter_text_chunk("sharpen", item.get("amount", 1.0), item))
            elif kind == "noise":
                parts.append(self.filter_text_chunk("noise", item.get("amount", 0.03), item))
            elif kind == "median":
                parts.append(self.filter_text_chunk("median", item.get("size", 3), item))
            elif kind == "edge":
                parts.append(self.filter_text_chunk("edge", item.get("strength", 1.0), item))
            elif kind == "emboss":
                parts.append(self.filter_text_chunk("emboss", item.get("strength", 1.0), item))
        return ";".join(parts)

    @staticmethod
    def filter_text_chunk(kind: str, value: float, item: dict) -> str:
        opacity = float(item.get("opacity", 1.0))
        blend_mode = str(item.get("blend_mode", "Normal"))
        if abs(opacity - 1.0) <= 0.001 and blend_mode == "Normal":
            return f"{kind},{value}"
        return f"{kind},{value},{opacity},{blend_mode}"

    def parse_filters(self, raw: str) -> list[dict]:
        filters: list[dict] = []
        if not raw.strip():
            return filters
        for chunk in raw.split(";"):
            parts = [part.strip() for part in chunk.split(",") if part.strip()]
            if not parts:
                continue
            kind = parts[0].lower()
            if len(parts) not in {2, 4}:
                raise ValueError
            metadata: dict[str, object] = {}
            if len(parts) == 4:
                metadata["opacity"] = max(0.0, min(1.0, float(parts[2])))
                metadata["blend_mode"] = parts[3] if parts[3] in BLEND_MODES else "Normal"
            if kind == "blur":
                filters.append(self.make_filter_item("blur", max(1, int(float(parts[1]))), metadata))
            elif kind == "sharpen":
                filters.append(self.make_filter_item("sharpen", max(0.0, float(parts[1])), metadata))
            elif kind == "noise":
                filters.append(self.make_filter_item("noise", max(0.0, min(1.0, float(parts[1]))), metadata))
            elif kind == "median":
                filters.append(self.make_filter_item("median", max(3, int(float(parts[1])) | 1), metadata))
            elif kind == "edge":
                filters.append(self.make_filter_item("edge", max(0.0, min(1.0, float(parts[1]))), metadata))
            elif kind == "emboss":
                filters.append(self.make_filter_item("emboss", max(0.0, min(1.0, float(parts[1]))), metadata))
            else:
                raise ValueError
        return filters

    def merge_down(self) -> None:
        self.run_document_command("Merge down", self.doc.merge_down)
        self.refresh()

    def flatten(self) -> None:
        self.run_document_command("Flatten", self.doc.flatten)
        self.refresh()

    def add_mask_from_selection(self) -> None:
        self.run_document_command("Add mask from selection", self.doc.add_mask_from_selection)
        self.refresh()

    def add_reveal_all_mask(self) -> None:
        self.run_document_command("Add reveal-all mask", self.doc.add_reveal_all_mask)
        self.refresh()

    def add_hide_all_mask(self) -> None:
        self.run_document_command("Add hide-all mask", self.doc.add_hide_all_mask)
        self.refresh()

    def invert_layer_mask(self) -> None:
        self.run_document_command("Invert mask", self.doc.invert_active_mask)
        self.refresh()

    def toggle_layer_mask(self) -> None:
        self.set_layer_property("Toggle mask", "mask_enabled", not self.doc.layer.mask_enabled)

    def toggle_layer_mask_link(self) -> None:
        if self.doc.layer.mask is None:
            messagebox.showinfo("Маска слоя", "У активного слоя нет маски.")
            return
        self.set_layer_property("Toggle mask link", "mask_linked", not self.doc.layer.mask_linked, affects_canvas=False)

    def set_mask_density(self) -> None:
        layer = self.doc.layer
        if layer.mask is None:
            messagebox.showinfo("Mask density", "Active layer has no mask.")
            return
        value = simpledialog.askfloat("Mask density", "Density 0..1:", initialvalue=float(layer.mask_density), minvalue=0.0, maxvalue=1.0)
        if value is not None:
            self.set_layer_property("Mask density", "mask_density", float(value))

    def set_mask_feather(self) -> None:
        layer = self.doc.layer
        if layer.mask is None:
            messagebox.showinfo("Mask feather", "Active layer has no mask.")
            return
        value = simpledialog.askfloat("Mask feather", "Radius px:", initialvalue=float(layer.mask_feather), minvalue=0.0, maxvalue=500.0)
        if value is not None:
            self.set_layer_property("Mask feather", "mask_feather", float(value), preserve_render_cache=False)

    def refine_layer_mask(self) -> None:
        if self.doc.layer.mask is None:
            messagebox.showinfo("Уточнить край маски", "У активного слоя нет маски.")
            return
        data = self.refine_layer_mask_dialog()
        if data is None:
            return
        self.run_document_command(
            "Уточнить край маски",
            lambda: self.doc.refine_active_mask(
                int(data["smooth"]),
                int(data["feather"]),
                float(data["contrast"]),
                int(data["shift"]),
                int(data["edge_radius"]),
                float(data["edge_strength"]),
                int(data["confidence_threshold"]),
            ),
        )
        self.refresh()

    def refine_layer_mask_dialog(self) -> dict[str, object] | None:
        layer = self.doc.layer
        dialog = tk.Toplevel(self)
        dialog.title("Уточнить край маски")
        dialog.transient(self)
        dialog.resizable(False, False)
        dialog.grab_set()

        smooth = tk.IntVar(value=2)
        feather = tk.IntVar(value=1)
        contrast = tk.DoubleVar(value=1.15)
        shift = tk.IntVar(value=0)
        edge_radius = tk.IntVar(value=3)
        edge_strength = tk.DoubleVar(value=0.65)
        confidence_threshold = tk.IntVar(value=96)
        preview_mode = tk.StringVar(value=SELECT_MASK_PREVIEW_CHANNEL)
        result: dict[str, object] | None = None

        preview = ttk.Label(dialog)
        preview.grid(row=0, column=0, rowspan=9, padx=12, pady=12, sticky="n")
        stats = ttk.Label(dialog, text="", justify=tk.LEFT)
        stats.grid(row=9, column=0, padx=12, pady=(0, 12), sticky="w")

        def values() -> dict[str, object]:
            return {
                "smooth": max(0, int(smooth.get())),
                "feather": max(0, int(feather.get())),
                "contrast": max(0.0, float(contrast.get())),
                "shift": int(shift.get()),
                "edge_radius": max(0, int(edge_radius.get())),
                "edge_strength": float(np.clip(edge_strength.get(), 0.0, 1.0)),
                "confidence_threshold": int(np.clip(confidence_threshold.get(), 0, 255)),
            }

        def update_preview(*_args) -> None:
            try:
                current = values()
            except (tk.TclError, ValueError):
                return
            mask = self.doc.preview_active_mask_refinement(**current)
            if mask is None:
                return
            canvas = self.render_select_mask_preview(layer.pixels, mask, preview_mode.get(), 180)
            self._mask_edge_preview_image = ImageTk.PhotoImage(canvas)
            preview.configure(image=self._mask_edge_preview_image)
            active = int(np.count_nonzero(mask))
            stats.configure(text=f"Активных пикселей: {active}\nГраницы: {self.mask_bounds(mask) or '-'}")

        def add_spin(row: int, label: str, variable, from_: float, to: float, increment: float = 1.0) -> None:
            ttk.Label(dialog, text=label).grid(row=row, column=1, sticky="w", padx=(0, 12), pady=(8, 0))
            spin = ttk.Spinbox(dialog, textvariable=variable, from_=from_, to=to, increment=increment, width=10, command=update_preview)
            spin.grid(row=row, column=2, sticky="ew", padx=(0, 12), pady=(8, 0))

        add_spin(0, "Сглаживание", smooth, 0, 100)
        add_spin(1, "Растушёвка", feather, 0, 500)
        add_spin(2, "Контраст", contrast, 0.0, 5.0, 0.05)
        add_spin(3, "Сдвиг края", shift, -500, 500)
        add_spin(4, "Радиус анализа", edge_radius, 0, 40)
        add_spin(5, "Сила привязки", edge_strength, 0.0, 1.0, 0.05)
        add_spin(6, "Порог уверенности", confidence_threshold, 0, 255)
        ttk.Label(dialog, text="Просмотр").grid(row=7, column=1, sticky="w", padx=(0, 12), pady=(8, 0))
        preview_box = ttk.Combobox(dialog, textvariable=preview_mode, values=SELECT_MASK_PREVIEW_MODES, state="readonly", width=18)
        preview_box.grid(row=7, column=2, sticky="ew", padx=(0, 12), pady=(8, 0))

        buttons = ttk.Frame(dialog)
        buttons.grid(row=8, column=1, columnspan=2, sticky="e", padx=12, pady=12)

        def accept() -> None:
            nonlocal result
            try:
                result = values()
            except (tk.TclError, ValueError):
                return
            dialog.destroy()

        ttk.Button(buttons, text="ОК", command=accept).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT)
        preview_box.bind("<<ComboboxSelected>>", update_preview)
        for variable in [smooth, feather, contrast, shift, edge_radius, edge_strength, confidence_threshold]:
            variable.trace_add("write", update_preview)
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        update_preview()
        dialog.wait_window()
        return result

    def apply_layer_mask(self) -> None:
        self.run_document_command("Apply mask", self.doc.apply_active_mask)
        self.refresh()

    def delete_layer_mask(self) -> None:
        self.run_document_command("Delete mask", self.doc.delete_active_mask)
        self.refresh()

    def layer_selected(self, _event) -> None:
        sel = self.layer_list.curselection()
        if sel:
            self.selected_layer_ids = {self.doc.layers[len(self.doc.layers) - 1 - row].id for row in sel}
            active_row = int(self.layer_list.index(tk.ACTIVE))
            active_index = len(self.doc.layers) - 1 - active_row
            if not (0 <= active_index < len(self.doc.layers)) or self.doc.layers[active_index].id not in self.selected_layer_ids:
                active_index = len(self.doc.layers) - 1 - sel[0]
            self.doc.active_layer = active_index
            if self.tool.get() == "text" and self.doc.layer.kind == "text":
                self.load_text_properties_from_layer(self.doc.layer)
            self.layer_opacity.set(self.doc.layer.opacity)
            self.blend_mode.set(self.doc.layer.blend_mode)
            self.refresh_layer_previews()
            self.refresh_properties()
            self.update_object_bounds()
            self.info.configure(text=f"{self.doc.width} x {self.doc.height}px\nСлоев: {len(self.doc.layers)}\nВыбрано: {len(self.selected_layer_ids)}")

    def layer_list_click(self, event) -> str | None:
        if int(event.x) > 28 or not self.doc.layers:
            return None
        row = int(self.layer_list.nearest(event.y))
        bounds = self.layer_list.bbox(row)
        if bounds is None or event.y < bounds[1] or event.y >= bounds[1] + bounds[3]:
            return "break"
        layer_index = len(self.doc.layers) - 1 - row
        if layer_index < 0 or layer_index >= len(self.doc.layers):
            return "break"
        active_index = self.doc.active_layer
        layer = self.doc.layers[layer_index]
        before = bool(layer.visible)
        layer.visible = not before
        self.doc.dirty = True
        self.push_command(LayerVisibilityCommand("Видимость слоя", layer.id, before, layer.visible))
        self.doc.active_layer = active_index
        self.refresh(preserve_render_cache=True)
        return "break"

    def begin_layer_opacity_change(self, _event) -> None:
        self._opacity_layer_id = self.doc.layer.id
        self._opacity_before = self.doc.layer.opacity

    def change_layer_opacity(self, _value) -> None:
        self.doc.layer.opacity = float(self.layer_opacity.get())
        self.doc.dirty = True
        self.request_canvas_refresh(preserve_layer_caches=True)

    def end_layer_opacity_change(self, _event) -> None:
        if self._opacity_layer_id is not None and self._opacity_before is not None:
            layer = self.doc.get_layer(self._opacity_layer_id)
            if layer is not None and layer.opacity != self._opacity_before:
                self.push_command(LayerOpacityCommand("Layer opacity", self._opacity_layer_id, self._opacity_before, layer.opacity))
        self._opacity_layer_id = None
        self._opacity_before = None

    def change_blend_mode(self, _event) -> None:
        layer = self.doc.layer
        before = layer.blend_mode
        after = self.blend_mode.get()
        if before == after:
            return
        layer.blend_mode = after
        self.doc.dirty = True
        self.push_command(LayerBlendModeCommand("Layer blend mode", layer.id, before, after))
        self.refresh(preserve_render_cache=True)

    def toggle_layer_visible(self) -> None:
        layer = self.doc.layer
        before = bool(layer.visible)
        layer.visible = not before
        self.doc.dirty = True
        self.push_command(LayerVisibilityCommand("Видимость слоя", layer.id, before, layer.visible))
        self.refresh(preserve_render_cache=True)

    def toggle_layer_lock(self) -> None:
        self.set_layer_property("Toggle layer lock", "locked", not self.doc.layer.locked, affects_canvas=False)

    def edit_text_layer(self) -> None:
        layer = self.doc.layer
        if layer.kind != "text" or layer.text_data is None:
            messagebox.showinfo("Text layer", "Select a text layer first.")
            return
        data = self.text_layer_dialog("Edit text layer", layer.text_data)
        if data is None:
            return
        self.run_document_command(
            "Edit text layer",
            lambda: self.doc.edit_text_layer(
                text=data["text"],
                size=data["size"],
                color=self.foreground,
                font_family=data["font_family"],
                box_width=data["box_width"],
                align=data["align"],
                line_spacing=data["line_spacing"],
                tracking=data["tracking"],
                bold=data["bold"],
                italic=data["italic"],
                underline=data["underline"],
                path_mode=data["path_mode"],
                path_amount=data["path_amount"],
                path_points=data["path_points"],
                path_start=data["path_start"],
                path_end=data["path_end"],
                path_side=data["path_side"],
                path_reverse=data["path_reverse"],
                baseline_shift=data["baseline_shift"],
                rotation=data["rotation"],
            ),
        )
        self.refresh()

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
        start_scale = ttk.Scale(controls, variable=start_var, from_=0.0, to=1.0)
        start_scale.pack(fill=tk.X, pady=(0, 8))
        end_label = ttk.Label(controls, text="Конец: 100%")
        end_label.pack(anchor=tk.W)
        end_scale = ttk.Scale(controls, variable=end_var, from_=0.0, to=1.0)
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

    def edit_shape_layer(self) -> None:
        layer = self.doc.layer
        if layer.kind != "shape" or layer.shape_data is None:
            messagebox.showinfo("Shape layer", "Select a shape layer first.")
            return
        control_points = layer.shape_data.get("control_points")
        suffix = ""
        if isinstance(control_points, list) and len(control_points) == 4:
            suffix = "," + ",".join(str(v) for point in control_points for v in point[:2])
        initial = f"{layer.shape_data.get('shape', 'rectangle')},{layer.shape_data.get('stroke_width', 0)},{layer.shape_data.get('sides', 5)},{layer.shape_data.get('inner_ratio', 0.5)}{suffix}"
        raw = simpledialog.askstring("Shape layer", "shape,stroke_width,sides,inner_ratio[,p0x,p0y,p1x,p1y,p2x,p2y,p3x,p3y]:", initialvalue=initial)
        if raw is None:
            return
        try:
            parts = [part.strip() for part in raw.split(",")]
            if len(parts) not in {4, 12}:
                raise ValueError
            shape = parts[0].lower()
            if shape not in {"rectangle", "ellipse", "line", "bezier", "polygon", "star", "custom"}:
                raise ValueError
            stroke_width = max(0, int(float(parts[1])))
            sides = max(3, int(float(parts[2])))
            inner_ratio = max(0.05, min(0.95, float(parts[3])))
            new_control_points = None
            if len(parts) == 12:
                values = [float(value) for value in parts[4:]]
                new_control_points = [(values[i], values[i + 1]) for i in range(0, 8, 2)]
        except ValueError:
            messagebox.showerror("Shape layer", "Use: shape,stroke_width,sides,inner_ratio[,p0x,p0y,p1x,p1y,p2x,p2y,p3x,p3y]")
            return
        self.run_shape_data_command(
            "Edit shape layer",
            lambda: self.doc.edit_shape_layer(shape=shape, fill=self.foreground, stroke=self.background, stroke_width=stroke_width, sides=sides, inner_ratio=inner_ratio, control_points=new_control_points),
        )
        self.refresh()

    def edit_bezier_points(self) -> None:
        layer = self.doc.layer
        if layer.kind != "shape" or layer.shape_data is None or str(layer.shape_data.get("shape")) != "bezier":
            messagebox.showinfo("Точки Безье", "Выберите слой с кривой Безье.")
            return
        raw_points = layer.shape_data.get("control_points")
        if not isinstance(raw_points, list) or len(raw_points) != 4:
            x1, y1, x2, y2 = [float(v) for v in layer.shape_data.get("box", [0, 0, 1, 1])]
            raw_points = [[x1, y2], [x1, y1], [x2, y1], [x2, y2]]
        points = [[float(point[0]), float(point[1])] for point in raw_points]
        dialog = tk.Toplevel(self)
        dialog.title("Редактор точек Безье")
        dialog.transient(self)
        dialog.resizable(False, False)
        dialog.grab_set()
        canvas = tk.Canvas(dialog, width=620, height=440, background="#22252b", highlightthickness=0, cursor="crosshair")
        canvas.pack(padx=12, pady=(12, 6))
        buttons = ttk.Frame(dialog)
        buttons.pack(fill=tk.X, padx=12, pady=(0, 12))
        active_point: list[int | None] = [None]
        scale = min(580 / max(1, self.doc.width), 400 / max(1, self.doc.height))
        ox, oy = (620 - self.doc.width * scale) / 2, (440 - self.doc.height * scale) / 2
        background = rgba_array_to_pil(self.render_engine.render(self.doc, checker=True)).resize((max(1, round(self.doc.width * scale)), max(1, round(self.doc.height * scale))), Image.Resampling.BILINEAR)
        self._bezier_preview_image = ImageTk.PhotoImage(background)

        def to_canvas(point: list[float] | tuple[float, float]) -> tuple[float, float]:
            return ox + float(point[0]) * scale, oy + float(point[1]) * scale

        def redraw() -> None:
            canvas.delete("all")
            canvas.create_image(ox, oy, image=self._bezier_preview_image, anchor=tk.NW)
            p = [to_canvas(point) for point in points]
            canvas.create_line(*p[0], *p[1], fill="#ffd166", dash=(4, 3), width=1)
            canvas.create_line(*p[2], *p[3], fill="#ffd166", dash=(4, 3), width=1)
            curve = bezier_curve_points(points, tuple(int(v) for v in layer.shape_data.get("box", [0, 0, 1, 1])), 96)
            coords = [value for point in curve for value in to_canvas(point)]
            canvas.create_line(*coords, fill="#50e3ff", width=3, smooth=True)
            for index, (px, py) in enumerate(p):
                color = "#50e3ff" if index in {0, 3} else "#ffd166"
                if index in {0, 3}:
                    canvas.create_oval(px - 8, py - 8, px + 8, py + 8, fill=color, outline="#111318", width=2)
                else:
                    canvas.create_rectangle(px - 8, py - 8, px + 8, py + 8, fill=color, outline="#111318", width=2)
                label = ("P0 начало", "P1 ручка", "P2 ручка", "P3 конец")[index]
                canvas.create_text(px, py - 17, text=label, fill=color, font=("Segoe UI", 9, "bold"))

        def press(event) -> None:
            active_point[0] = None
            best = 14.0
            for index, point in enumerate(points):
                px, py = to_canvas(point)
                distance = ((event.x - px) ** 2 + (event.y - py) ** 2) ** 0.5
                if distance < best:
                    active_point[0], best = index, distance

        def drag(event) -> None:
            if active_point[0] is None:
                return
            index = int(active_point[0])
            points[index] = [(event.x - ox) / scale, (event.y - oy) / scale]
            redraw()

        def accept() -> None:
            self.run_shape_data_command("Edit Bezier points", lambda: self.doc.edit_shape_layer(control_points=[tuple(point) for point in points]))
            dialog.destroy()
            self.refresh()

        ttk.Label(buttons, text="Круги - концы, квадраты - управляющие ручки.").pack(side=tk.LEFT)
        ttk.Button(buttons, text="ОК", command=accept).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT)
        ToolTip(canvas, "Перетаскивайте конечные P0/P3 и управляющие P1/P2.")
        canvas.bind("<ButtonPress-1>", press)
        canvas.bind("<B1-Motion>", drag)
        redraw()
        dialog.wait_window()

    def boolean_shape_layers(self) -> None:
        initial = self.doc.boolean_shape_data_with_lower("union")
        if initial is None:
            messagebox.showinfo("Булева операция фигур", "Выберите незаблокированную фигуру, расположенную прямо над другой фигурой.")
            return
        edited = self.boolean_shape_editor(initial, "Создать булеву фигуру")
        if edited is None:
            return
        mode = str(edited.get("boolean_mode", "union"))
        self.run_document_command("Создать булеву фигуру", lambda: self.doc.boolean_active_shape_with_lower(mode, edited))
        self.selected_layer_ids = {self.doc.layer.id}
        self.refresh()

    def edit_boolean_shape(self) -> None:
        layer = self.doc.layer
        if layer.kind != "shape" or layer.shape_data is None or str(layer.shape_data.get("shape")) != "boolean":
            return
        edited = self.boolean_shape_editor(layer.shape_data, "Редактировать булеву фигуру")
        if edited is None or edited == layer.shape_data:
            return

        def apply() -> None:
            layer.shape_data = copy.deepcopy(edited)
            layer.name = f"Булева фигура: {edited.get('boolean_mode', 'union')}"
            render_shape_layer(layer)
            layer.touch_pixels()
            self.doc.dirty = True

        self.run_shape_data_command("Редактировать булеву фигуру", apply)
        self.refresh()

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

    def resize_image(self) -> None:
        width = simpledialog.askinteger("Resize image", "Width px:", initialvalue=self.doc.width, minvalue=1, maxvalue=100000)
        height = simpledialog.askinteger("Resize image", "Height px:", initialvalue=self.doc.height, minvalue=1, maxvalue=100000)
        if width and height:
            self.run_document_command("Resize image", lambda: self.doc.resize_image(width, height))
            self.refresh()

    def resize_canvas(self) -> None:
        width = simpledialog.askinteger("Resize canvas", "Width px:", initialvalue=self.doc.width, minvalue=1, maxvalue=100000)
        height = simpledialog.askinteger("Resize canvas", "Height px:", initialvalue=self.doc.height, minvalue=1, maxvalue=100000)
        if width and height:
            self.run_document_command("Resize canvas", lambda: self.doc.resize_canvas(width, height))
            self.refresh()

    def generative_expand_dialog(self) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("Генеративное расширение холста")
        dialog.geometry("560x500")
        dialog.transient(self)
        dialog.grab_set()
        values = {name: tk.IntVar(value=max(64, size // 8)) for name, size in (
            ("Слева", self.doc.width), ("Сверху", self.doc.height), ("Справа", self.doc.width), ("Снизу", self.doc.height)
        )}
        method = tk.StringVar(value="content-aware")
        controls = ttk.Frame(dialog, padding=12)
        controls.pack(fill=tk.X)
        for column, (name, variable) in enumerate(values.items()):
            ttk.Label(controls, text=name).grid(row=0, column=column, padx=4, sticky="w")
            ttk.Spinbox(controls, from_=0, to=100000, textvariable=variable, width=10).grid(row=1, column=column, padx=4)
        ttk.Label(controls, text="Заполнение").grid(row=2, column=0, sticky="w", pady=(12, 2))
        ttk.Combobox(
            controls,
            textvariable=method,
            values=("content-aware", "mirror", "edge"),
            state="readonly",
        ).grid(row=3, column=0, columnspan=4, sticky="ew", padx=4)
        preview = ttk.Label(dialog, anchor=tk.CENTER)
        preview.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        def update_preview(*_args) -> None:
            try:
                source = self.render_engine.render(self.doc, False)
                scale = min(1.0, 360 / max(source.shape[0], source.shape[1]))
                small = cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                document_scale = small.shape[1] / self.doc.width
                margins = [max(0, round(values[name].get() * document_scale)) for name in ("Слева", "Сверху", "Справа", "Снизу")]
                result = generative_expand_pixels(small, margins[0], margins[1], margins[2], margins[3], method.get())
                image = rgba_array_to_pil(result)
                image.thumbnail((500, 320), Image.Resampling.LANCZOS)
                self._generative_preview_image = ImageTk.PhotoImage(image)
                preview.configure(image=self._generative_preview_image)
            except (tk.TclError, ValueError):
                pass

        for variable in values.values():
            variable.trace_add("write", update_preview)
        method.trace_add("write", update_preview)

        buttons = ttk.Frame(dialog, padding=(12, 0, 12, 12))
        buttons.pack(fill=tk.X)

        def apply() -> None:
            margins = [values[name].get() for name in ("Слева", "Сверху", "Справа", "Снизу")]
            if not any(value > 0 for value in margins):
                messagebox.showinfo("Расширение", "Укажите размер расширения хотя бы с одной стороны.", parent=dialog)
                return
            dialog.destroy()
            self.run_document_command(
                "Генеративное расширение",
                lambda: self.doc.generative_expand(*margins, method.get()),
            )
            self.refresh()

        ttk.Button(buttons, text="Расширить", command=apply).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT, padx=6)
        update_preview()

    def change_bit_depth(self, bit_depth: int) -> None:
        self.run_document_command(f"Глубина каналов {bit_depth} бит", lambda: self.doc.set_bit_depth(bit_depth))
        self.refresh()

    def change_color_model(self, color_model: str) -> None:
        self.run_document_command(f"Цветовая модель {color_model}", lambda: self.doc.set_color_model(color_model))
        self.refresh()

    def _choose_icc_profile(self, title: str) -> str | None:
        use_srgb = messagebox.askyesnocancel(title, "Использовать стандартный профиль sRGB?\n\nДа: sRGB\nНет: выбрать ICC-файл")
        if use_srgb is None:
            return None
        if use_srgb:
            return "sRGB"
        return filedialog.askopenfilename(title=title, filetypes=[("ICC-профили", "*.icc *.icm"), ("Все файлы", "*.*")]) or None

    def assign_icc_profile(self) -> None:
        profile = self._choose_icc_profile("Назначить ICC-профиль")
        if profile:
            self.run_document_command("Назначить ICC-профиль", lambda: self.doc.assign_color_profile(profile))
            self.refresh()

    def convert_icc_profile(self) -> None:
        profile = self._choose_icc_profile("Преобразовать в ICC-профиль")
        if not profile:
            return
        try:
            self.run_document_command("Преобразовать ICC-профиль", lambda: self.doc.convert_color_profile(profile))
            self.refresh()
        except Exception as exc:
            messagebox.showerror("Управление цветом", str(exc))

    def crop_to_selection(self) -> None:
        crop_box = self.doc.selection_bounds() or self.selection_box
        if not crop_box:
            messagebox.showinfo("Crop", "Create a rectangular selection first.")
            return
        self.run_document_command("Crop", lambda: self.doc.crop(crop_box))
        self.doc.clear_selection()
        self.selection_box = None
        self.update_selection_overlay()
        self.refresh()

    def trim_transparent(self) -> None:
        self.run_document_command("Trim transparent pixels", self.doc.trim_transparent)
        self.selection_box = self.doc.selection_bounds()
        self.refresh()

    def reveal_all(self) -> None:
        self.run_document_command("Reveal all layers", self.doc.reveal_all)
        self.selection_box = self.doc.selection_bounds()
        self.refresh()

    def rotate(self, angle: int) -> None:
        def edit():
            old_w, old_h = self.doc.width, self.doc.height
            for layer in self.doc.layers:
                lx, ly, lw, lh = layer.x, layer.y, layer.pixels.shape[1], layer.pixels.shape[0]
                if layer.kind in {"linked", "embedded"} and layer.smart_data is not None:
                    transform = dict(layer.smart_data.get("transform") or {})
                    transform["angle"] = float(transform.get("angle", 0.0)) + angle
                    layer.smart_data = {**layer.smart_data, "transform": transform}
                if angle == 90:
                    layer.pixels = cv2.rotate(layer.pixels, cv2.ROTATE_90_CLOCKWISE)
                    if layer.mask is not None:
                        layer.mask = cv2.rotate(layer.mask, cv2.ROTATE_90_CLOCKWISE)
                    layer.x = old_h - (ly + lh)
                    layer.y = lx
                elif angle == 180:
                    layer.pixels = cv2.rotate(layer.pixels, cv2.ROTATE_180)
                    if layer.mask is not None:
                        layer.mask = cv2.rotate(layer.mask, cv2.ROTATE_180)
                    layer.x = old_w - (lx + lw)
                    layer.y = old_h - (ly + lh)
                layer.touch_pixels()
            if self.doc.selection_mask is not None:
                if angle == 90:
                    self.doc.selection_mask = cv2.rotate(self.doc.selection_mask, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    self.doc.selection_mask = cv2.rotate(self.doc.selection_mask, cv2.ROTATE_180)
            for name, mask in list(self.doc.saved_selections.items()):
                if angle == 90:
                    self.doc.saved_selections[name] = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    self.doc.saved_selections[name] = cv2.rotate(mask, cv2.ROTATE_180)
            if angle in [90, 270]:
                self.doc.width, self.doc.height = self.doc.height, self.doc.width
            self.doc.dirty = True

        self.run_document_command(f"Rotate {angle}", edit)
        self.refresh()

    def flip(self, horizontal: bool) -> None:
        def edit():
            code = 1 if horizontal else 0
            for layer in self.doc.layers:
                layer.pixels = cv2.flip(layer.pixels, code)
                if layer.kind in {"linked", "embedded"} and layer.smart_data is not None:
                    transform = dict(layer.smart_data.get("transform") or {})
                    key = "flip_horizontal" if horizontal else "flip_vertical"
                    transform[key] = not bool(transform.get(key, False))
                    layer.smart_data = {**layer.smart_data, "transform": transform}
                if layer.mask is not None:
                    layer.mask = cv2.flip(layer.mask, code)
                if horizontal:
                    layer.x = self.doc.width - (layer.x + layer.pixels.shape[1])
                else:
                    layer.y = self.doc.height - (layer.y + layer.pixels.shape[0])
                layer.touch_pixels()
            if self.doc.selection_mask is not None:
                self.doc.selection_mask = cv2.flip(self.doc.selection_mask, code)
            for name, mask in list(self.doc.saved_selections.items()):
                self.doc.saved_selections[name] = cv2.flip(mask, code)
            self.doc.dirty = True

        self.run_document_command("Flip horizontal" if horizontal else "Flip vertical", edit)
        self.refresh()

    def apply_to_layer(self, label: str, fn) -> None:
        layer = self.doc.layer
        if layer.locked:
            self.status_text("Слой заблокирован")
            return
        layer_id = layer.id
        generation = self._edit_generation
        pixels_revision = layer.pixels_revision
        before = layer.pixels.copy()
        selection_mask = self.doc.layer_selection_mask(layer)
        rect = (0, 0, before.shape[1], before.shape[0])

        def worker():
            after = fn(before.copy())
            if selection_mask is not None:
                alpha = (selection_mask.astype(np.float32) / 255.0)[:, :, None]
                after = (after.astype(np.float32) * alpha + before.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
            return after

        def done(after):
            target = self.doc.get_layer(layer_id)
            if target is None:
                return
            target.pixels = after
            target.touch_pixels()
            self.doc.dirty = True
            self.push_command(PixelPatchCommand(label, layer_id, rect, before, after.copy()))
            self.invalidate_pixels()
            self.refresh()

        self.run_background(
            label,
            worker,
            done,
            lambda: self._edit_generation == generation
            and (target := self.doc.get_layer(layer_id)) is not None
            and target.pixels_revision == pixels_revision,
        )

    def adjust_brightness_contrast(self) -> None:
        b = simpledialog.askinteger("Brightness", "Brightness -255..255:", initialvalue=0, minvalue=-255, maxvalue=255)
        c = simpledialog.askfloat("Contrast", "Contrast multiplier:", initialvalue=1.1, minvalue=0.0, maxvalue=10.0)
        if b is not None and c is not None:
            self.apply_to_layer("brightness/contrast", lambda arr: adjust_brightness_contrast(arr, b, c))

    def adjust_saturation(self) -> None:
        s = simpledialog.askfloat("Saturation", "Saturation multiplier:", initialvalue=1.2, minvalue=0.0, maxvalue=10.0)
        if s is not None:
            self.apply_to_layer("saturation", lambda arr: adjust_saturation(arr, s))

    def adjust_hue_saturation(self) -> None:
        hue = simpledialog.askinteger("Hue/Saturation", "Hue shift -180..180:", initialvalue=0, minvalue=-180, maxvalue=180)
        saturation = simpledialog.askfloat("Hue/Saturation", "Saturation multiplier:", initialvalue=1.0, minvalue=0.0, maxvalue=10.0)
        lightness = simpledialog.askinteger("Hue/Saturation", "Lightness -255..255:", initialvalue=0, minvalue=-255, maxvalue=255)
        if hue is not None and saturation is not None and lightness is not None:
            self.apply_to_layer("hue/saturation", lambda arr: adjust_hue_saturation(arr, hue, saturation, lightness))

    def adjust_exposure(self) -> None:
        exposure = simpledialog.askfloat("Exposure", "Exposure stops -5..5:", initialvalue=0.0, minvalue=-5.0, maxvalue=5.0)
        offset = simpledialog.askfloat("Exposure", "Offset -1..1:", initialvalue=0.0, minvalue=-1.0, maxvalue=1.0)
        gamma = simpledialog.askfloat("Exposure", "Gamma:", initialvalue=1.0, minvalue=0.01, maxvalue=10.0)
        if exposure is not None and offset is not None and gamma is not None:
            self.apply_to_layer("exposure", lambda arr: adjust_exposure(arr, exposure, offset, gamma))

    def adjust_color_balance(self) -> None:
        red = simpledialog.askinteger("Color balance", "Red shift -255..255:", initialvalue=0, minvalue=-255, maxvalue=255)
        green = simpledialog.askinteger("Color balance", "Green shift -255..255:", initialvalue=0, minvalue=-255, maxvalue=255)
        blue = simpledialog.askinteger("Color balance", "Blue shift -255..255:", initialvalue=0, minvalue=-255, maxvalue=255)
        if red is not None and green is not None and blue is not None:
            self.apply_to_layer("color balance", lambda arr: adjust_color_balance(arr, red, green, blue))

    def adjust_levels(self) -> None:
        black = simpledialog.askinteger("Levels", "Black point:", initialvalue=0, minvalue=0, maxvalue=254)
        white = simpledialog.askinteger("Levels", "White point:", initialvalue=255, minvalue=1, maxvalue=255)
        gamma = simpledialog.askfloat("Levels", "Gamma:", initialvalue=1.0, minvalue=0.01, maxvalue=10.0)
        if black is not None and white is not None and gamma is not None:
            self.apply_to_layer("levels", lambda arr: levels(arr, black, white, gamma))

    def adjust_curves(self) -> None:
        shadows = simpledialog.askinteger("Curves", "Shadows output:", initialvalue=64, minvalue=0, maxvalue=255)
        midtones = simpledialog.askinteger("Curves", "Midtones output:", initialvalue=128, minvalue=0, maxvalue=255)
        highlights = simpledialog.askinteger("Curves", "Highlights output:", initialvalue=192, minvalue=0, maxvalue=255)
        if shadows is not None and midtones is not None and highlights is not None:
            self.apply_to_layer("curves", lambda arr: curves(arr, shadows, midtones, highlights))

    def adjust_threshold(self) -> None:
        threshold = simpledialog.askinteger("Threshold", "Threshold 0..255:", initialvalue=128, minvalue=0, maxvalue=255)
        if threshold is not None:
            self.apply_to_layer("threshold", lambda arr: adjust_threshold(arr, threshold))

    def adjust_posterize(self) -> None:
        levels_count = simpledialog.askinteger("Posterize", "Levels 2..64:", initialvalue=6, minvalue=2, maxvalue=64)
        if levels_count is not None:
            self.apply_to_layer("posterize", lambda arr: adjust_posterize(arr, levels_count))

    def adjust_invert(self) -> None:
        self.apply_to_layer("invert", lambda arr: self._invert(arr))

    def adjust_grayscale(self) -> None:
        self.apply_to_layer("grayscale", lambda arr: self._grayscale(arr))


    def add_adjustment_layer(self) -> None:
        data = self.adjustment_layer_dialog()
        if data is None:
            return
        adjustment = data["adjustment"]
        name = data["name"]
        self.run_document_command(f"{name} adjustment layer", lambda: self.doc.add_adjustment_layer(name, adjustment))
        self.refresh()

    def edit_adjustment_layer(self) -> None:
        layer = self.doc.layer
        if layer.kind != "adjustment" or layer.adjustment is None:
            messagebox.showinfo("Adjustment layer", "Select an adjustment layer first.")
            return
        data = self.adjustment_layer_dialog(layer.adjustment)
        if data is None:
            return
        adjustment = data["adjustment"]
        name = data["name"]

        def edit() -> None:
            active = self.doc.layer
            active.adjustment = dict(adjustment)
            active.name = str(name)
            self.doc.dirty = True

        self.run_document_command("Edit adjustment layer", edit)
        self.refresh()

    def adjustment_layer_dialog(self, initial: dict | None = None) -> dict | None:
        initial = dict(initial or {"type": "brightness_contrast", "brightness": 0, "contrast": 1.1})
        result: dict | None = None
        dialog = tk.Toplevel(self)
        dialog.title("\u041a\u043e\u0440\u0440\u0435\u043a\u0442\u0438\u0440\u0443\u044e\u0449\u0438\u0439 \u0441\u043b\u043e\u0439")
        dialog.transient(self)
        dialog.resizable(False, False)
        dialog.grab_set()

        source = self.render_engine.render(self.doc, checker=False)
        adjustment_type = tk.StringVar(value=str(initial.get("type", "brightness_contrast")))
        values = [tk.DoubleVar(value=0.0), tk.DoubleVar(value=0.0), tk.DoubleVar(value=0.0)]
        labels: list[ttk.Label] = []
        spins: list[ttk.Spinbox] = []
        updating = False

        preview = ttk.Label(dialog)
        preview.grid(row=0, column=0, rowspan=8, padx=12, pady=12, sticky="n")
        ttk.Label(dialog, text="Пресет").grid(row=0, column=1, sticky="w", padx=(0, 12), pady=(12, 4))
        adjustment_preset = tk.StringVar(value=next(iter(self.adjustment_presets)))
        preset_box = ttk.Combobox(dialog, textvariable=adjustment_preset, values=list(self.adjustment_presets), state="readonly", width=22)
        preset_box.grid(row=0, column=2, sticky="ew", padx=(0, 12), pady=(12, 4))
        ttk.Button(dialog, text="Применить пресет", command=lambda: apply_adjustment_preset()).grid(row=1, column=1, columnspan=2, sticky="ew", padx=(0, 12), pady=(0, 8))
        ttk.Label(dialog, text="\u0422\u0438\u043f").grid(row=2, column=1, sticky="w", padx=(0, 12), pady=(4, 4))
        type_box = ttk.Combobox(dialog, textvariable=adjustment_type, values=ADJUSTMENT_TYPES, state="readonly", width=22)
        type_box.grid(row=2, column=2, sticky="ew", padx=(0, 12), pady=(4, 4))
        hint = ttk.Label(dialog, text="", wraplength=240, justify=tk.LEFT)
        hint.grid(row=6, column=1, columnspan=2, sticky="w", padx=(0, 12), pady=(8, 0))

        for index in range(3):
            label = ttk.Label(dialog, text="")
            spin = ttk.Spinbox(dialog, textvariable=values[index], from_=0, to=255, increment=1, width=12)
            label.grid(row=index + 3, column=1, sticky="w", padx=(0, 12), pady=4)
            spin.grid(row=index + 3, column=2, sticky="ew", padx=(0, 12), pady=4)
            labels.append(label)
            spins.append(spin)

        buttons = ttk.Frame(dialog)
        buttons.grid(row=7, column=1, columnspan=2, sticky="e", padx=12, pady=12)

        def current_adjustment() -> dict:
            return self.make_adjustment_item(adjustment_type.get(), [value.get() for value in values])

        def set_values_for_kind(kind: str, adjustment: dict | None = None) -> None:
            nonlocal updating
            updating = True
            adjustment = adjustment or {}
            specs = self.adjustment_specs(kind, adjustment)
            for index, (label_text, default, from_value, to_value, increment) in enumerate(specs):
                labels[index].configure(text=label_text)
                spins[index].configure(from_=from_value, to=to_value, increment=increment)
                values[index].set(default)
                labels[index].grid()
                spins[index].grid()
            for index in range(len(specs), 3):
                labels[index].grid_remove()
                spins[index].grid_remove()
            hint.configure(text=self.adjustment_hint(kind))
            updating = False

        preview_scale = min(1.0, 180 / max(1, source.shape[1]), 180 / max(1, source.shape[0]))
        preview_size = max(1, round(source.shape[1] * preview_scale)), max(1, round(source.shape[0] * preview_scale))
        adjustment_preview_source = source.copy() if preview_size == (source.shape[1], source.shape[0]) else cv2.resize(source, preview_size, interpolation=cv2.INTER_AREA)

        def update_preview(*_args) -> None:
            if updating:
                return
            shown = self.apply_adjustment_preview(adjustment_preview_source, current_adjustment())
            image = rgba_array_to_pil(shown)
            canvas = Image.new("RGBA", (180, 180), (44, 46, 52, 255))
            canvas.alpha_composite(image, ((180 - image.width) // 2, (180 - image.height) // 2))
            self._adjustment_preview_image = ImageTk.PhotoImage(canvas)
            preview.configure(image=self._adjustment_preview_image)

        def type_changed(_event=None) -> None:
            set_values_for_kind(adjustment_type.get())
            update_preview()

        def apply_adjustment_preset() -> None:
            preset = self.adjustment_presets.get(adjustment_preset.get())
            if preset is None:
                return
            kind = str(preset.get("type", "brightness_contrast"))
            if kind not in ADJUSTMENT_TYPES:
                return
            adjustment_type.set(kind)
            set_values_for_kind(kind, preset)
            update_preview()

        def import_adjustment_presets() -> None:
            path = filedialog.askopenfilename(filetypes=[("PhotoRedactor presets", "*.json"), ("JSON", "*.json")], parent=dialog)
            if not path:
                return
            try:
                payload = json.loads(Path(path).read_text(encoding="utf-8"))
                presets = payload.get("presets", payload) if isinstance(payload, dict) else {}
                added = 0
                for name, preset in presets.items():
                    if isinstance(name, str) and isinstance(preset, dict) and str(preset.get("type", "")) in ADJUSTMENT_TYPES:
                        self.adjustment_presets[name] = dict(preset)
                        added += 1
                if added == 0:
                    raise ValueError("Файл не содержит поддерживаемых пресетов.")
                preset_box.configure(values=list(self.adjustment_presets))
                adjustment_preset.set(next(reversed(self.adjustment_presets)))
                apply_adjustment_preset()
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                messagebox.showerror("Импорт пресетов", str(exc), parent=dialog)

        def export_adjustment_presets() -> None:
            name = simpledialog.askstring("Сохранить пресет", "Название пресета:", initialvalue=self.adjustment_name(current_adjustment()), parent=dialog)
            if not name:
                return
            self.adjustment_presets[name] = current_adjustment()
            preset_box.configure(values=list(self.adjustment_presets))
            adjustment_preset.set(name)
            path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("PhotoRedactor presets", "*.json"), ("JSON", "*.json")], parent=dialog)
            if path:
                payload = {"format": "PhotoRedactor adjustment presets", "version": 1, "presets": self.adjustment_presets}
                try:
                    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                except OSError as exc:
                    messagebox.showerror("Экспорт пресетов", str(exc), parent=dialog)

        def accept() -> None:
            nonlocal result
            adjustment = current_adjustment()
            result = {"adjustment": adjustment, "name": self.adjustment_name(adjustment)}
            dialog.destroy()

        def cancel() -> None:
            dialog.destroy()

        type_box.bind("<<ComboboxSelected>>", type_changed)
        for value in values:
            value.trace_add("write", update_preview)
        ttk.Button(buttons, text="\u041e\u041a", command=accept).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="\u041e\u0442\u043c\u0435\u043d\u0430", command=cancel).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Экспорт", command=export_adjustment_presets).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(buttons, text="Импорт", command=import_adjustment_presets).pack(side=tk.LEFT)
        dialog.protocol("WM_DELETE_WINDOW", cancel)
        set_values_for_kind(adjustment_type.get(), initial)
        update_preview()
        dialog.wait_window()
        return result

    @staticmethod
    def adjustment_specs(kind: str, adjustment: dict) -> list[tuple[str, float, float, float, float]]:
        if kind == "brightness_contrast":
            return [("\u042f\u0440\u043a\u043e\u0441\u0442\u044c", float(adjustment.get("brightness", 0)), -255, 255, 1), ("\u041a\u043e\u043d\u0442\u0440\u0430\u0441\u0442", float(adjustment.get("contrast", 1.0)), 0, 5, 0.05)]
        if kind == "saturation":
            return [("\u041d\u0430\u0441\u044b\u0449\u0435\u043d\u043d\u043e\u0441\u0442\u044c", float(adjustment.get("saturation", 1.0)), 0, 5, 0.05)]
        if kind == "vibrance":
            return [("Вибрация", float(adjustment.get("vibrance", 0.0)), -1, 1, 0.05), ("Насыщенность", float(adjustment.get("saturation", 1.0)), 0, 3, 0.05)]
        if kind == "temperature_tint":
            return [("Температура", float(adjustment.get("temperature", 0.0)), -100, 100, 1), ("Оттенок", float(adjustment.get("tint", 0.0)), -100, 100, 1)]
        if kind == "hue_saturation":
            return [("\u0422\u043e\u043d", float(adjustment.get("hue", 0)), -180, 180, 1), ("\u041d\u0430\u0441\u044b\u0449\u0435\u043d\u043d\u043e\u0441\u0442\u044c", float(adjustment.get("saturation", 1.0)), 0, 5, 0.05), ("\u0421\u0432\u0435\u0442\u043b\u043e\u0442\u0430", float(adjustment.get("lightness", 0)), -255, 255, 1)]
        if kind == "exposure":
            return [("\u042d\u043a\u0441\u043f\u043e\u0437\u0438\u0446\u0438\u044f", float(adjustment.get("exposure", 0.0)), -5, 5, 0.05), ("\u0421\u0434\u0432\u0438\u0433", float(adjustment.get("offset", 0.0)), -1, 1, 0.01), ("\u0413\u0430\u043c\u043c\u0430", float(adjustment.get("gamma", 1.0)), 0.01, 10, 0.05)]
        if kind == "color_balance":
            return [("\u041a\u0440\u0430\u0441\u043d\u044b\u0439", float(adjustment.get("red", 0)), -255, 255, 1), ("\u0417\u0435\u043b\u0435\u043d\u044b\u0439", float(adjustment.get("green", 0)), -255, 255, 1), ("\u0421\u0438\u043d\u0438\u0439", float(adjustment.get("blue", 0)), -255, 255, 1)]
        if kind == "levels":
            return [("\u0427\u0435\u0440\u043d\u0430\u044f \u0442\u043e\u0447\u043a\u0430", float(adjustment.get("black", 0)), 0, 254, 1), ("\u0411\u0435\u043b\u0430\u044f \u0442\u043e\u0447\u043a\u0430", float(adjustment.get("white", 255)), 1, 255, 1), ("\u0413\u0430\u043c\u043c\u0430", float(adjustment.get("gamma", 1.0)), 0.01, 10, 0.05)]
        if kind == "curves":
            return [("\u0422\u0435\u043d\u0438", float(adjustment.get("shadows", 64)), 0, 255, 1), ("\u0421\u0440\u0435\u0434\u043d\u0438\u0435", float(adjustment.get("midtones", 128)), 0, 255, 1), ("\u0421\u0432\u0435\u0442\u0430", float(adjustment.get("highlights", 192)), 0, 255, 1)]
        if kind == "threshold":
            return [("\u041f\u043e\u0440\u043e\u0433", float(adjustment.get("threshold", 128)), 0, 255, 1)]
        if kind == "posterize":
            return [("\u0423\u0440\u043e\u0432\u043d\u0438", float(adjustment.get("levels", 6)), 2, 64, 1)]
        return []

    @staticmethod
    def adjustment_hint(kind: str) -> str:
        if kind in {"invert", "grayscale"}:
            return "\u042d\u0442\u043e\u0442 \u0442\u0438\u043f \u043d\u0435 \u0442\u0440\u0435\u0431\u0443\u0435\u0442 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u043e\u0432."
        return "\u0418\u0437\u043c\u0435\u043d\u0435\u043d\u0438\u044f \u043f\u043e\u043a\u0430\u0437\u044b\u0432\u0430\u044e\u0442\u0441\u044f \u0432 \u043f\u0440\u0435\u0434\u043f\u0440\u043e\u0441\u043c\u043e\u0442\u0440\u0435."

    @staticmethod
    def make_adjustment_item(kind: str, values: list[float]) -> dict:
        if kind == "brightness_contrast":
            return {"type": kind, "brightness": int(values[0]), "contrast": float(values[1])}
        if kind == "saturation":
            return {"type": kind, "saturation": float(values[0])}
        if kind == "vibrance":
            return {"type": kind, "vibrance": float(values[0]), "saturation": float(values[1])}
        if kind == "temperature_tint":
            return {"type": kind, "temperature": float(values[0]), "tint": float(values[1])}
        if kind == "hue_saturation":
            return {"type": kind, "hue": int(values[0]), "saturation": float(values[1]), "lightness": int(values[2])}
        if kind == "exposure":
            return {"type": kind, "exposure": float(values[0]), "offset": float(values[1]), "gamma": max(0.01, float(values[2]))}
        if kind == "color_balance":
            return {"type": kind, "red": int(values[0]), "green": int(values[1]), "blue": int(values[2])}
        if kind == "levels":
            black = int(values[0])
            white = max(black + 1, int(values[1]))
            return {"type": kind, "black": black, "white": min(255, white), "gamma": max(0.01, float(values[2]))}
        if kind == "curves":
            return {"type": kind, "shadows": int(values[0]), "midtones": int(values[1]), "highlights": int(values[2])}
        if kind == "threshold":
            return {"type": kind, "threshold": int(values[0])}
        if kind == "posterize":
            return {"type": kind, "levels": int(values[0])}
        if kind in {"invert", "grayscale"}:
            return {"type": kind}
        return {"type": "brightness_contrast", "brightness": 0, "contrast": 1.0}

    @staticmethod
    def adjustment_name(adjustment: dict) -> str:
        kind = str(adjustment.get("type", "brightness_contrast")).lower()
        return ADJUSTMENT_LABELS.get(kind, "Adjustment")

    def apply_adjustment_preview(self, arr: np.ndarray, adjustment: dict) -> np.ndarray:
        kind = str(adjustment.get("type", "")).lower()
        if kind == "brightness_contrast":
            return adjust_brightness_contrast(arr, int(adjustment.get("brightness", 0)), float(adjustment.get("contrast", 1.0)))
        if kind == "saturation":
            return adjust_saturation(arr, float(adjustment.get("saturation", 1.0)))
        if kind == "vibrance":
            return adjust_vibrance(arr, float(adjustment.get("vibrance", 0.0)), float(adjustment.get("saturation", 1.0)))
        if kind == "temperature_tint":
            return adjust_temperature_tint(arr, float(adjustment.get("temperature", 0.0)), float(adjustment.get("tint", 0.0)))
        if kind == "hue_saturation":
            return adjust_hue_saturation(arr, int(adjustment.get("hue", 0)), float(adjustment.get("saturation", 1.0)), int(adjustment.get("lightness", 0)))
        if kind == "exposure":
            return adjust_exposure(arr, float(adjustment.get("exposure", 0.0)), float(adjustment.get("offset", 0.0)), float(adjustment.get("gamma", 1.0)))
        if kind == "color_balance":
            return adjust_color_balance(arr, int(adjustment.get("red", 0)), int(adjustment.get("green", 0)), int(adjustment.get("blue", 0)))
        if kind == "levels":
            return levels(arr, int(adjustment.get("black", 0)), int(adjustment.get("white", 255)), float(adjustment.get("gamma", 1.0)))
        if kind == "curves":
            return curves(arr, int(adjustment.get("shadows", 64)), int(adjustment.get("midtones", 128)), int(adjustment.get("highlights", 192)))
        if kind == "threshold":
            return adjust_threshold(arr, int(adjustment.get("threshold", 128)))
        if kind == "posterize":
            return adjust_posterize(arr, int(adjustment.get("levels", 6)))
        if kind == "invert":
            return self._invert(arr)
        if kind == "grayscale":
            return self._grayscale(arr)
        return arr.copy()

    @staticmethod
    def _invert(arr):
        out = arr.copy()
        out[:, :, :3] = 255 - out[:, :, :3]
        return out

    @staticmethod
    def _grayscale(arr):
        out = arr.copy()
        gray = cv2.cvtColor(out[:, :, :3], cv2.COLOR_RGB2GRAY)
        out[:, :, :3] = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return out

    def frequency_separation_layers(self) -> None:
        layer = self.doc.layer
        if layer.locked or layer.kind == "adjustment":
            messagebox.showinfo("Частотное разложение", "Выберите незаблокированный слой с изображением.")
            return
        settings = self.frequency_separation_dialog(layer.pixels)
        if settings is None:
            return
        self.run_document_command(
            "Частотное разложение",
            lambda: self.doc.frequency_separate_active(settings["radius"], settings["texture_strength"]),
        )
        self.refresh()

    def frequency_separation_dialog(self, source: np.ndarray) -> dict[str, float] | None:
        dialog = tk.Toplevel(self)
        dialog.title("Частотное разложение")
        dialog.transient(self)
        dialog.resizable(False, False)
        dialog.grab_set()
        result: dict[str, float] | None = None
        radius = tk.DoubleVar(value=8.0)
        texture_strength = tk.DoubleVar(value=1.0)

        preview_row = ttk.Frame(dialog)
        preview_row.pack(fill=tk.X, padx=12, pady=(12, 8))
        preview_labels: list[ttk.Label] = []
        for title in ["Цвет и тон", "Текстура", "Результат"]:
            column = ttk.Frame(preview_row)
            column.pack(side=tk.LEFT, padx=4)
            ttk.Label(column, text=title).pack(pady=(0, 4))
            label = ttk.Label(column)
            label.pack()
            preview_labels.append(label)

        controls = ttk.Frame(dialog)
        controls.pack(fill=tk.X, padx=16)
        radius_value = ttk.Label(controls, width=8)
        texture_value = ttk.Label(controls, width=8)
        ttk.Label(controls, text="Радиус размытия").grid(row=0, column=0, sticky="w")
        ttk.Scale(controls, from_=0.5, to=40.0, variable=radius, orient=tk.HORIZONTAL).grid(row=0, column=1, sticky="ew", padx=8)
        radius_value.grid(row=0, column=2, sticky="e")
        ttk.Label(controls, text="Сила текстуры").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Scale(controls, from_=0.0, to=2.0, variable=texture_strength, orient=tk.HORIZONTAL).grid(row=1, column=1, sticky="ew", padx=8, pady=(8, 0))
        texture_value.grid(row=1, column=2, sticky="e", pady=(8, 0))
        controls.columnconfigure(1, weight=1)

        thumb_image = rgba_array_to_pil(source)
        thumb_image.thumbnail((190, 190), Image.Resampling.LANCZOS)
        thumb = np.array(thumb_image.convert("RGBA"), dtype=np.uint8)

        def photo_for(arr: np.ndarray) -> ImageTk.PhotoImage:
            image = rgba_array_to_pil(arr)
            canvas = Image.new("RGBA", (190, 190), (44, 46, 52, 255))
            canvas.alpha_composite(image, ((190 - image.width) // 2, (190 - image.height) // 2))
            return ImageTk.PhotoImage(canvas)

        def update_preview(*_args) -> None:
            low, high = frequency_separation(thumb, radius.get(), texture_strength.get())
            recombined = low.copy()
            recombined[:, :, :3] = np.clip(
                low[:, :, :3].astype(np.float32) + high[:, :, :3].astype(np.float32) * 2.0 - 255.0,
                0,
                255,
            ).astype(np.uint8)
            self._frequency_preview_images = [photo_for(low), photo_for(high), photo_for(recombined)]
            for label, image in zip(preview_labels, self._frequency_preview_images):
                label.configure(image=image)
            radius_value.configure(text=f"{radius.get():.1f} px")
            texture_value.configure(text=f"{texture_strength.get():.2f}")

        def accept() -> None:
            nonlocal result
            result = {"radius": float(radius.get()), "texture_strength": float(texture_strength.get())}
            dialog.destroy()

        buttons = ttk.Frame(dialog)
        buttons.pack(fill=tk.X, padx=12, pady=12)
        ttk.Button(buttons, text="ОК", command=accept).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT)
        ToolTip(preview_labels[0], "На этом слое удобно выравнивать цвет и тон кожи.")
        ToolTip(preview_labels[1], "Этот слой сохраняет поры, волосы и мелкие детали.")
        radius.trace_add("write", update_preview)
        texture_strength.trace_add("write", update_preview)
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        update_preview()
        dialog.wait_window()
        return result

    def portrait_cleanup_layer(self) -> None:
        layer = self.doc.layer
        if layer.locked or layer.kind == "adjustment":
            messagebox.showinfo("Портретная обработка", "Выберите незаблокированный слой с портретом.")
            return
        settings = self.portrait_cleanup_dialog(layer.pixels)
        if settings is None:
            return
        self.apply_to_layer(
            "Портретная обработка",
            lambda arr: portrait_cleanup(arr, settings["smoothing"], settings["texture"], settings["even_tone"], settings["redness"]),
        )

    def portrait_cleanup_dialog(self, source: np.ndarray) -> dict[str, float] | None:
        dialog = tk.Toplevel(self)
        dialog.title("Портретная обработка")
        dialog.transient(self)
        dialog.resizable(False, False)
        dialog.grab_set()
        result: dict[str, float] | None = None
        variables = {
            "smoothing": tk.DoubleVar(value=0.35),
            "texture": tk.DoubleVar(value=0.7),
            "even_tone": tk.DoubleVar(value=0.2),
            "redness": tk.DoubleVar(value=0.2),
        }

        preview_row = ttk.Frame(dialog)
        preview_row.pack(fill=tk.X, padx=12, pady=(12, 8))
        preview_labels: list[ttk.Label] = []
        for title in ["До", "После"]:
            column = ttk.Frame(preview_row)
            column.pack(side=tk.LEFT, padx=5)
            ttk.Label(column, text=title).pack(pady=(0, 4))
            label = ttk.Label(column)
            label.pack()
            preview_labels.append(label)

        thumb_image = rgba_array_to_pil(source)
        thumb_image.thumbnail((250, 220), Image.Resampling.LANCZOS)
        thumb = np.array(thumb_image.convert("RGBA"), dtype=np.uint8)

        def photo_for(arr: np.ndarray) -> ImageTk.PhotoImage:
            image = rgba_array_to_pil(arr)
            canvas = Image.new("RGBA", (250, 220), (44, 46, 52, 255))
            canvas.alpha_composite(image, ((250 - image.width) // 2, (220 - image.height) // 2))
            return ImageTk.PhotoImage(canvas)

        controls = ttk.Frame(dialog)
        controls.pack(fill=tk.X, padx=16)
        value_labels: dict[str, ttk.Label] = {}
        specs = [
            ("smoothing", "Сглаживание кожи", 0.0, 1.0),
            ("texture", "Сохранение текстуры", 0.0, 1.5),
            ("even_tone", "Выравнивание тона", 0.0, 1.0),
            ("redness", "Уменьшение покраснений", 0.0, 1.0),
        ]
        for row, (key, title, start, end) in enumerate(specs):
            ttk.Label(controls, text=title).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Scale(controls, from_=start, to=end, variable=variables[key], orient=tk.HORIZONTAL).grid(row=row, column=1, sticky="ew", padx=8, pady=3)
            value_labels[key] = ttk.Label(controls, width=7)
            value_labels[key].grid(row=row, column=2, sticky="e")
        controls.columnconfigure(1, weight=1)

        original_photo = photo_for(thumb)

        def update_preview(*_args) -> None:
            cleaned = portrait_cleanup(
                thumb,
                variables["smoothing"].get(),
                variables["texture"].get(),
                variables["even_tone"].get(),
                variables["redness"].get(),
            )
            self._portrait_preview_images = [original_photo, photo_for(cleaned)]
            for label, image in zip(preview_labels, self._portrait_preview_images):
                label.configure(image=image)
            for key, label in value_labels.items():
                label.configure(text=f"{variables[key].get():.2f}")

        def accept() -> None:
            nonlocal result
            result = {key: float(variable.get()) for key, variable in variables.items()}
            dialog.destroy()

        buttons = ttk.Frame(dialog)
        buttons.pack(fill=tk.X, padx=12, pady=12)
        ttk.Button(buttons, text="ОК", command=accept).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT)
        ToolTip(preview_labels[1], "Обработка автоматически ограничивается тонами кожи.")
        for variable in variables.values():
            variable.trace_add("write", update_preview)
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        update_preview()
        dialog.wait_window()
        return result

    def filter_blur(self) -> None:
        r = simpledialog.askinteger("Gaussian blur", "Radius:", initialvalue=3, minvalue=1, maxvalue=200)
        if r:
            self.apply_to_layer("blur", lambda arr: blur(arr, r))

    def filter_sharpen(self) -> None:
        a = simpledialog.askfloat("Sharpen", "Amount:", initialvalue=1.0, minvalue=0.0, maxvalue=10.0)
        if a is not None:
            self.apply_to_layer("sharpen", lambda arr: sharpen(arr, a))

    def filter_noise(self) -> None:
        a = simpledialog.askfloat("Noise", "Amount 0..1:", initialvalue=0.04, minvalue=0.0, maxvalue=1.0)
        if a is not None:
            self.apply_to_layer("noise", lambda arr: add_noise(arr, a))

    def filter_content_aware_fill(self) -> None:
        layer = self.doc.layer
        selection_mask = self.doc.layer_selection_mask(layer)
        if selection_mask is None or not np.any(selection_mask):
            messagebox.showinfo("Заливка с учетом содержимого", "Сначала создайте выделение на активном слое.")
            return
        radius = simpledialog.askinteger("Заливка с учетом содержимого", "Радиус поиска:", initialvalue=3, minvalue=1, maxvalue=30)
        if radius:
            self.apply_to_layer("Заливка с учетом содержимого", lambda arr: content_aware_fill(arr, selection_mask, radius))

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

    def filter_red_eye(self) -> None:
        selection_mask = self.doc.layer_selection_mask(self.doc.layer)
        strength = simpledialog.askfloat("Удаление красных глаз", "Сила 0..1:", initialvalue=0.85, minvalue=0.0, maxvalue=1.0)
        if strength is not None:
            self.apply_to_layer("Удаление красных глаз", lambda arr: reduce_red_eye(arr, selection_mask, strength))

    def filter_patch_selection(self) -> None:
        if self.doc.selection_mask is None:
            messagebox.showinfo("Заплатка", "Сначала создайте выделение.")
            return
        x = simpledialog.askinteger("Заплатка", "X левого верхнего угла источника:", initialvalue=0, minvalue=-100000, maxvalue=100000)
        if x is None:
            return
        y = simpledialog.askinteger("Заплатка", "Y левого верхнего угла источника:", initialvalue=0, minvalue=-100000, maxvalue=100000)
        if y is None:
            return
        heal = messagebox.askyesno("Заплатка", "Подогнать цвет источника под место назначения?")
        self.run_document_command("Заплатка", lambda: self.doc.patch_active_selection(x, y, heal))
        self.refresh()

    def refresh_plugin_filter_menu(self) -> None:
        self.plugin_filters_menu.delete(0, tk.END)
        if not self.plugin_registry.filters:
            self.plugin_filters_menu.add_command(label="Нет доступных фильтров", state=tk.DISABLED)
            return
        for name, plugin in sorted(self.plugin_registry.filters.items()):
            self.plugin_filters_menu.add_command(label=name, command=lambda value=name: self.apply_plugin_filter(value))

    def apply_plugin_filter(self, name: str) -> None:
        raw = simpledialog.askstring("Фильтр-плагин", "Параметры JSON:", initialvalue="{}")
        if raw is None:
            return
        try:
            params = json.loads(raw)
            if not isinstance(params, dict):
                raise ValueError("Параметры должны быть JSON-объектом")
            self.apply_to_layer(name, lambda pixels: self.plugin_registry.apply_filter(name, pixels, params))
        except Exception as exc:
            messagebox.showerror("Фильтр-плагин", str(exc))

    def reload_plugins(self) -> None:
        count = self.plugin_registry.discover()
        for name, callback in self.plugin_registry.action_commands.items():
            if name not in self.action_runner.commands:
                self.action_runner.register(name, callback)
        self.status_text(f"Плагины перезагружены: {count}")
        if self.plugin_registry.errors:
            self.show_plugin_errors()

    def show_plugin_errors(self) -> None:
        if not self.plugin_registry.errors:
            messagebox.showinfo("Плагины", "Ошибок загрузки нет.")
            return
        self.show_text_window("Ошибки плагинов", "\n".join(self.plugin_registry.errors))

    def set_view_channel(self) -> None:
        self.invalidate_view()
        self.refresh_canvas()

    def set_mask_preview(self) -> None:
        self.invalidate_view()
        self.refresh_canvas()

    def set_paint_target(self) -> None:
        if self.paint_target.get() == "mask":
            self.prepare_mask_editing(create_if_missing=True)
        else:
            self.mask_preview.set(MASK_PREVIEW_NORMAL)
            self.set_mask_preview()
            self.status_text("Рисование по пикселям активного слоя")

    def edit_pixels_channel(self, _event=None) -> None:
        self.paint_target.set("pixels")
        self.mask_preview.set(MASK_PREVIEW_NORMAL)
        self.set_mask_preview()
        self.status_text("Рисование по пикселям активного слоя")

    def edit_mask_channel(self, _event=None) -> None:
        self.prepare_mask_editing(create_if_missing=True)

    def edit_active_mask_channel(self) -> None:
        self.prepare_mask_editing(create_if_missing=True)

    def prepare_mask_editing(self, create_if_missing: bool) -> None:
        layer = self.doc.layer
        if layer.mask is None:
            if not create_if_missing:
                self.mask_preview.set(MASK_PREVIEW_NORMAL)
                self.set_mask_preview()
                self.status_text("У активного слоя нет маски")
                return
            self.run_document_command("Add reveal-all mask", self.doc.add_reveal_all_mask)
        self.paint_target.set("mask")
        self.mask_preview.set(MASK_PREVIEW_CHANNEL)
        self.refresh()
        self.status_text("Рисование по маске активного слоя")

    def set_zoom(self, value: float) -> None:
        old_zoom = max(0.0001, float(self.zoom.get()))
        ox, oy = self._canvas_origin
        center_x = self.canvas.canvasx(max(1, self.canvas.winfo_width()) / 2)
        center_y = self.canvas.canvasy(max(1, self.canvas.winfo_height()) / 2)
        doc_center_x = (center_x - ox) / old_zoom
        doc_center_y = (center_y - oy) / old_zoom
        self.zoom.set(max(0.05, min(16.0, value)))
        self.invalidate_view()
        self.refresh_canvas()
        self.center_canvas_on_doc(doc_center_x, doc_center_y)

    def center_canvas_on_doc(self, doc_x: float, doc_y: float) -> None:
        if self._initial_fit_after_id is not None and not self._performing_initial_fit:
            try:
                self.after_cancel(self._initial_fit_after_id)
            except tk.TclError:
                pass
            self._initial_fit_after_id = None
        raw_region = str(self.canvas.cget("scrollregion")).split()
        if len(raw_region) != 4:
            return
        try:
            region = tuple(float(value) for value in raw_region)
        except ValueError:
            return
        target_x, target_y = self.doc_to_canvas(doc_x, doc_y)
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        scroll_w = max(1.0, region[2] - region[0])
        scroll_h = max(1.0, region[3] - region[1])
        left = target_x - width / 2.0
        top = target_y - height / 2.0
        self.canvas.xview_moveto(max(0.0, min(1.0, (left - region[0]) / scroll_w)))
        self.canvas.yview_moveto(max(0.0, min(1.0, (top - region[1]) / scroll_h)))

    def fit_to_screen(self) -> None:
        self.update_idletasks()
        w = max(1, self.canvas.winfo_width() - 20)
        h = max(1, self.canvas.winfo_height() - 20)
        self.set_zoom(min(w / self.doc.width, h / self.doc.height))
        self.center_canvas_on_doc(self.doc.width / 2, self.doc.height / 2)

    def set_grid_spacing(self) -> None:
        spacing = simpledialog.askinteger("Grid", "Spacing px:", initialvalue=int(self.grid_spacing.get()), minvalue=4, maxvalue=5000)
        if spacing:
            self.grid_spacing.set(spacing)
            self.refresh_canvas()

    def add_horizontal_guide(self) -> None:
        y = simpledialog.askinteger("Horizontal guide", "Y coordinate:", initialvalue=self.doc.height // 2, minvalue=0, maxvalue=max(0, self.doc.height))
        if y is not None:
            self._guide_doc_lines.append(("h", y))
            self.refresh_canvas()

    def add_vertical_guide(self) -> None:
        x = simpledialog.askinteger("Vertical guide", "X coordinate:", initialvalue=self.doc.width // 2, minvalue=0, maxvalue=max(0, self.doc.width))
        if x is not None:
            self._guide_doc_lines.append(("v", x))
            self.refresh_canvas()

    def clear_guides(self) -> None:
        self._guide_doc_lines.clear()
        self.refresh_canvas()

    def mouse_wheel(self, event) -> None:
        if event.state & 0x0004:
            self.set_zoom(self.zoom.get() * (1.1 if event.delta > 0 else 0.9))
        elif event.state & 0x0001:
            self.canvas.xview_scroll(-1 if event.delta > 0 else 1, "units")
        else:
            self.canvas.yview_scroll(-3 if event.delta > 0 else 3, "units")

    def batch_process(self) -> None:
        src = filedialog.askdirectory(title="Source folder")
        if not src:
            return
        dst = filedialog.askdirectory(title="Destination folder")
        if not dst:
            return
        width = simpledialog.askinteger("Batch", "Max width px, empty for original:", initialvalue=1920, minvalue=1, maxvalue=50000)

        def worker():
            exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"} | RAW_EXTENSIONS
            count = 0
            for path in Path(src).rglob("*"):
                if path.suffix.lower() not in exts:
                    continue
                doc = Document.from_image(path)
                if width and doc.width > width:
                    doc.resize_image(width, max(1, round(doc.height * width / doc.width)))
                out = Path(dst) / f"{path.stem}.png"
                doc.export_flat(out)
                count += 1
            return count

        self.run_background("Batch", worker, lambda count: messagebox.showinfo("Batch", f"Processed {count} files."))


def enable_high_dpi() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass


def main() -> None:
    enable_high_dpi()
    app = PhotoRedactorApp()
    app.mainloop()
