from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

def _configure_source_tk_runtime() -> None:
    if os.name != "nt" or getattr(sys, "frozen", False):
        return
    root = Path(sys.base_prefix) / "tcl"
    for variable, pattern, marker in (
        ("TCL_LIBRARY", "tcl*", "init.tcl"),
        ("TK_LIBRARY", "tk*", "tk.tcl"),
    ):
        if variable in os.environ:
            continue
        candidate = next((path for path in sorted(root.glob(pattern)) if (path / marker).is_file()), None)
        if candidate is not None:
            os.environ[variable] = str(candidate)

_configure_source_tk_runtime()
import tkinter as tk
import tkinter.font as tkfont
from tkinter import colorchooser, filedialog, messagebox, simpledialog, ttk
import uuid

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageGrab, ImageSequence, ImageTk

from .core import (
    Document,
    Layer,
    BLEND_MODES,
    GradientEngine,
    SourceAnchor,
    SourceTransform,
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
    apply_adjustment,
    apply_gradient,
    apply_filter_stack,
    bezier_curve_points,
    BrushPathSampler,
    BrushSettings,
    CloneHealingStroke,
    custom_shape_points,
    blur,
    curves,
    content_aware_fill,
    content_aware_fill_variants,
    clone_or_heal,
    draw_mask_brush,
    draw_brush,
    decode_png,
    encode_png,
    edge_aware_cleanup,
    flood_fill,
    contiguous_color_region,
    frequency_separation,
    generative_expand_pixels,
    image_statistics,
    effective_layer_mask,
    levels,
    local_retouch,
    paste_mask,
    PixelBrushStroke,
    MaskBrushStroke,
    portrait_cleanup,
    pointer_input,
    pointer_pressure,
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
    TabletSample,
    retouch_falloff_mask,
    RAW_EXTENSIONS,
    selection_edge_confidence,
    selection_contour_points,
    shape_drag_is_meaningful,
    shape_data_bounds,
    shape_geometry_from_drag,
    layer_contains_point,
    mesh_warp_pixels,
    perspective_warp_pixels,
    resize_box_from_handle,
    star_points,
    topmost_layer_at,
    transform_shape_data_to_box,
    union_rect,
    warp_pixels,
    sharpen,
    spot_heal,
    sample_source_patch,
    build_patch_edit,
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
    GeneratedExpandCommand,
    LayerMoveCommand,
    LayerGroupMoveCommand,
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
    ObjectStatesCommand,
    TextDataCommand,
    TilePatch,
)
from .document_manager import DocumentManager, DocumentSession
from .object_hit_testing import ObjectHit, hit_test_document, hit_test_stack, layers_inside_box
from .rendering import RenderEngine
from .gpu_acceleration import acceleration_metrics, acceleration_status, benchmark_acceleration, calibrate_acceleration, reset_acceleration_metrics
from .large_document import gpu_status
from .interactive_performance import benchmark_interactive_paths
from .generative_api import GeneratedVariant, GenerativeAPIError, MAX_SEED, validate_outpaint_dimensions, variant_seeds
from .psd_compat import PSDCompatibilityError, export_psd
from .automation import ActionRecorder, ActionRunner, ActionStep, load_action
from .batch_queue import BatchQueue
from .color_management import COLOR_MODELS, BIT_DEPTHS, color_settings, display_rgba, profile_details, soft_proof_rgba, gamut_warning_mask
from .print_pipeline import document_source_profile, proof_document, print_preflight, export_cmyk_tiff, export_color_separations
from .spot_colors import SpotColor, assign_spot_color, assigned_spot_color, document_spot_colors, lab_to_srgb, load_library, replace_document_spot_colors, save_library
from .windows_print import print_document
from .plugins import PluginRegistry
from .ui.tool_options import ToolOptionsPanel
from .ui.tool_palette import ToolPalette, ToolPaletteDialog, normalize_tool_order, normalize_visible_tools
from .ui.icons import SHORTCUTS, action_icon
from .ui.shortcuts import TOOL_SHORTCUT_GROUPS, accelerator, command_for_event, event_alt_down, event_key
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
    ("Перемещение", "move", "Двигает активный слой мышью."),
    ("Прямоуг. выделение", "select", "Выделяет прямоугольную область."),
    ("Эллипс выделение", "ellipse_select", "Выделяет овальную область."),
    ("Лассо", "lasso", "Рисует свободный контур выделения от руки."),
    ("Полигон", "polygon_lasso", "Строит выделение кликами по вершинам. Двойной клик завершает."),
    ("Магнитное лассо", "magnetic_lasso", "Привязывает контур выделения к ближайшим сильным краям."),
    ("Быстрое выделение", "quick_selection", "Добавляет похожие соседние области кистью."),
    ("Волшебная палочка", "magic_wand", "Выделяет связанную область похожего цвета."),
    ("Цветовой диапазон", "color_range", "Выделяет похожий цвет по всему активному слою."),
    ("Кадрирование", "crop", "Задает область обрезки документа."),
    ("Пипетка", "eyedropper", "Берет цвет из изображения в передний цвет."),
    ("Точечное восстановление", "spot_healing", "Автоматически удаляет мелкие дефекты без выбора источника."),
    ("Лечение", "healing", "Копирует источник и подгоняет цвет под место назначения."),
    ("Заплатка", "patch", "Перетащите активное выделение на область-источник для ретуши."),
    ("Кисть", "brush", "Рисует выбранным передним цветом по пикселям или маске."),
    ("Штамп", "clone", "Копирует пиксели из источника. Alt+клик задает источник."),
    ("Ластик", "eraser", "Стирает пиксели слоя или закрашивает маску черным."),
    ("Градиент", "gradient", "Растягивает переход от переднего цвета к фоновому."),
    ("Заливка", "fill", "Заполняет связанную область выбранным цветом."),
    ("Размытие", "blur_tool", "Локально смягчает участок активного слоя."),
    ("Резкость", "sharpen_tool", "Локально усиливает контраст деталей."),
    ("Осветлитель", "dodge", "Осветляет область под кистью."),
    ("Затемнитель", "burn", "Затемняет область под кистью."),
    ("Кривая Безье", "bezier_shape", "Создает редактируемую кубическую кривую Безье."),
    ("Выбор контура", "path_select", "Выбирает контур целиком для дальнейшего редактирования."),
    ("Выбор узлов", "direct_select", "Выбирает и перемещает опорные точки и направляющие контура."),
    ("Добавить узел", "add_anchor", "Добавляет опорную точку в ближайший участок выбранного контура."),
    ("Удалить узел", "delete_anchor", "Удаляет опорную точку выбранного контура."),
    ("Преобразовать узел", "convert_anchor", "Переключает опорную точку между угловой и плавной."),
    ("Текст", "text", "Создает редактируемый текстовый слой."),
    ("Прямоуг. фигура", "rect_shape", "Создает редактируемый прямоугольник."),
    ("Эллипс", "ellipse_shape", "Создает редактируемый эллипс."),
    ("Линия", "line_shape", "Создает редактируемую линию."),
    ("Многоугольник", "polygon_shape", "Создает редактируемый многоугольник."),
    ("Звезда", "star_shape", "Создает редактируемую звезду."),
    ("Своя фигура", "custom_shape", "Создает фигуру из библиотеки пользовательских контуров."),
    ("Рука", "hand", "Перемещает холст по экрану без изменения изображения."),
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
    "brush": {"size": 28, "hardness": 1.0, "opacity": 1.0, "flow": 1.0, "spacing": 0.0, "smoothing": 0.15, "blend_mode": "Normal", "pressure_size": False, "pressure_opacity": False, "pressure_flow": False},
    "eraser": {"size": 28, "hardness": 1.0, "opacity": 1.0, "flow": 1.0, "spacing": 0.0, "smoothing": 0.15, "pressure_size": False, "pressure_opacity": False, "pressure_flow": False},
    "blur_tool": {"size": 32, "hardness": 1.0, "strength": 0.25, "flow": 0.35, "spacing": 0.0, "smoothing": 0.15, "pressure_size": False, "pressure_opacity": False, "pressure_flow": False},
    "sharpen_tool": {"size": 28, "hardness": 1.0, "strength": 0.2, "flow": 0.3, "spacing": 0.0, "smoothing": 0.15, "pressure_size": False, "pressure_opacity": False, "pressure_flow": False},
    "dodge": {"size": 32, "hardness": 1.0, "flow": 1.0, "spacing": 0.0, "smoothing": 0.15, "exposure": 0.15, "range": "Средние тона"},
    "burn": {"size": 32, "hardness": 1.0, "flow": 1.0, "spacing": 0.0, "smoothing": 0.15, "exposure": 0.15, "range": "Средние тона"},
    "clone": {"size": 28, "hardness": 1.0, "flow": 1.0, "spacing": 0.0, "smoothing": 0.15, "opacity": 0.8, "pressure_size": False, "pressure_opacity": False, "pressure_flow": False},
    "healing": {"size": 26, "hardness": 1.0, "flow": 1.0, "spacing": 0.0, "smoothing": 0.15, "strength": 0.65, "pressure_size": False, "pressure_opacity": False, "pressure_flow": False},
    "spot_healing": {"size": 10, "hardness": 1.0, "flow": 1.0, "spacing": 0.0, "smoothing": 0.1, "strength": 0.65, "pressure_size": False, "pressure_opacity": False, "pressure_flow": False},
}

BRUSH_PRESET_DEFAULTS = {
    "Круглая жёсткая": {"size": 28, "hardness": 1.0, "opacity": 1.0, "flow": 1.0, "spacing": 0.0, "smoothing": 0.15, "blend_mode": "Normal"},
    "Мягкая кисть": {"size": 80, "hardness": 0.0, "opacity": 1.0, "flow": 0.25, "spacing": 0.12, "smoothing": 0.25, "blend_mode": "Normal"},
    "Круглая кисть": {"size": 28, "hardness": 1.0, "opacity": 1.0, "flow": 1.0, "spacing": 0.0, "smoothing": 0.15, "blend_mode": "Normal"},
    "Точная ретушь": {"size": 16, "hardness": 0.45, "opacity": 1.0, "flow": 0.2, "spacing": 0.15, "smoothing": 0.35, "blend_mode": "Normal"},
    "Карандаш": {"size": 4, "hardness": 1.0, "opacity": 1.0, "flow": 1.0, "spacing": 0.0, "smoothing": 0.05, "blend_mode": "Normal"},
    "Аэрограф": {"size": 90, "hardness": 0.0, "opacity": 0.7, "flow": 0.08, "spacing": 0.08, "smoothing": 0.25, "blend_mode": "Normal"},
    "Маркер": {"size": 34, "hardness": 0.85, "opacity": 0.7, "flow": 0.45, "spacing": 0.05, "smoothing": 0.25, "blend_mode": "Multiply", "advanced": {"angle": -18.0, "roundness": 0.35}},
    "Плоская каллиграфия": {"size": 42, "hardness": 1.0, "opacity": 1.0, "flow": 1.0, "spacing": 0.04, "smoothing": 0.3, "blend_mode": "Normal", "advanced": {"angle": 35.0, "roundness": 0.18}},
    "Тушь": {"size": 18, "hardness": 0.9, "opacity": 1.0, "flow": 0.8, "spacing": 0.03, "smoothing": 0.45, "blend_mode": "Normal", "pressure_size": True, "pressure_opacity": True},
    "Мел": {"size": 36, "hardness": 0.75, "opacity": 0.75, "flow": 0.35, "spacing": 0.12, "smoothing": 0.2, "blend_mode": "Normal", "advanced": {"size_jitter": 0.12, "angle_jitter": 0.4, "opacity_jitter": 0.18}},
    "Уголь": {"size": 52, "hardness": 0.35, "opacity": 0.68, "flow": 0.22, "spacing": 0.1, "smoothing": 0.18, "blend_mode": "Multiply", "advanced": {"size_jitter": 0.08, "roundness": 0.55, "angle_jitter": 0.25, "opacity_jitter": 0.22}},
    "Брызги": {"size": 26, "hardness": 0.9, "opacity": 0.9, "flow": 0.65, "spacing": 0.24, "smoothing": 0.05, "blend_mode": "Normal", "advanced": {"scatter": 1.8, "scatter_both_axes": True, "scatter_count": 4, "size_jitter": 0.55, "opacity_jitter": 0.25}},
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

SELECT_MASK_PREVIEW_CHANNEL = "Чёрно-белая маска"
SELECT_MASK_PREVIEW_OVERLAY = "Красное перекрытие"
SELECT_MASK_PREVIEW_CUTOUT = "Вырез на прозрачности"
SELECT_MASK_PREVIEW_EDGE_CONFIDENCE = "Уверенность края"
SELECT_MASK_PREVIEW_ONION = "Луковая плёнка"
SELECT_MASK_PREVIEW_MARCHING = "Бегущие муравьи"
SELECT_MASK_PREVIEW_BLACK = "На чёрном"
SELECT_MASK_PREVIEW_WHITE = "На белом"
SELECT_MASK_PREVIEW_LAYERS = "На слоях"
SELECT_MASK_PREVIEW_MODES = [
    SELECT_MASK_PREVIEW_ONION, SELECT_MASK_PREVIEW_MARCHING, SELECT_MASK_PREVIEW_BLACK,
    SELECT_MASK_PREVIEW_WHITE, SELECT_MASK_PREVIEW_CHANNEL, SELECT_MASK_PREVIEW_LAYERS,
    SELECT_MASK_PREVIEW_OVERLAY, SELECT_MASK_PREVIEW_CUTOUT, SELECT_MASK_PREVIEW_EDGE_CONFIDENCE,
]

FILTER_TYPES = ["blur", "motion_blur", "sharpen", "unsharp_mask", "smart_sharpen", "noise", "reduce_noise", "median", "high_pass", "edge", "emboss"]
FILTER_LABELS = {
    "blur": "Размытие",
    "motion_blur": "Размытие в движении",
    "sharpen": "Резкость",
    "unsharp_mask": "Контурная резкость",
    "smart_sharpen": "Умная резкость",
    "noise": "Шум",
    "reduce_noise": "Уменьшение шума",
    "median": "Медиана",
    "high_pass": "Цветовой контраст",
    "edge": "Края",
    "emboss": "Тиснение",
}
FILTER_VALUES = {label: value for value, label in FILTER_LABELS.items()}
CHANNEL_LABELS = {"RGB": "RGB", "Red": "Красный", "Green": "Зелёный", "Blue": "Синий", "Alpha": "Альфа"}
CHANNEL_VALUES = {label: value for value, label in CHANNEL_LABELS.items()}

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
    "black_white",
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
    "black_white": "Чёрно-белое",
}
ADJUSTMENT_VALUES = {label: value for value, label in ADJUSTMENT_LABELS.items()}

ADJUSTMENT_PRESETS = {
    "Яркий портрет": {"type": "brightness_contrast", "brightness": 12, "contrast": 1.18},
    "Теплый тон": {"type": "color_balance", "red": 12, "green": 4, "blue": -10},
    "Кино-контраст": {"type": "curves", "shadows": 48, "midtones": 128, "highlights": 210},
    "Мягкая экспозиция": {"type": "exposure", "exposure": 0.25, "offset": 0.0, "gamma": 1.08},
    "Холодные тени": {"type": "color_balance", "red": -8, "green": 0, "blue": 14},
    "Чистый чёрно-белый": {"type": "black_white", "red": 0.299, "green": 0.587, "blue": 0.114},
}


ADJUSTMENT_PRESETS.update(
    {
        "Живые цвета": {"type": "vibrance", "vibrance": 0.42, "saturation": 1.05},
        "Золотой час": {"type": "temperature_tint", "temperature": 26, "tint": 5},
        "Холодный свет": {"type": "temperature_tint", "temperature": -22, "tint": -3},
    }
)

__all__ = [name for name in globals() if not name.startswith("__")]
