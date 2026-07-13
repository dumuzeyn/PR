from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import colorchooser, filedialog, messagebox, simpledialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .core import (
    Document,
    BLEND_MODES,
    add_noise,
    adjust_brightness_contrast,
    adjust_color_balance,
    adjust_exposure,
    adjust_hue_saturation,
    adjust_posterize,
    adjust_saturation,
    adjust_threshold,
    apply_gradient,
    apply_filter_stack,
    blur,
    curves,
    content_aware_fill,
    clone_or_heal,
    draw_mask_brush,
    draw_brush,
    encode_png,
    edge_aware_cleanup,
    flood_fill,
    image_statistics,
    effective_layer_mask,
    levels,
    local_retouch,
    paste_mask,
    refine_selection_mask,
    rgba_array_to_pil,
    reduce_red_eye,
    selection_edge_confidence,
    union_rect,
    sharpen,
)
from .history import DocumentStateCommand, History, LayerBlendModeCommand, LayerMoveCommand, LayerOpacityCommand, MaskPatchCommand, PixelPatchCommand, SelectionMaskCommand
from .ui.tool_options import ToolOptionsPanel
from .ui.tool_palette import ToolPalette, ToolPaletteDialog, normalize_tool_order, normalize_visible_tools


class ToolTip:
    def __init__(self, widget: tk.Widget, text: str, delay: int = 450) -> None:
        self.widget = widget
        self.text = text
        self.delay = delay
        self._after_id: str | None = None
        self._tip: tk.Toplevel | None = None
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
        x = self.widget.winfo_rootx() + self.widget.winfo_width() + 10
        y = self.widget.winfo_rooty()
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self._tip,
            text=self.text,
            justify=tk.LEFT,
            background="#fff7d6",
            foreground="#1f2328",
            relief=tk.SOLID,
            borderwidth=1,
            padx=8,
            pady=5,
            wraplength=280,
        )
        label.pack()

    def hide(self, _event=None) -> None:
        self.cancel()
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


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
    "Мягкая ретушь": {"brush_size": 18, "opacity": 0.22, "tolerance": 18},
    "Средняя ретушь": {"brush_size": 34, "opacity": 0.45, "tolerance": 28},
    "Сильная ретушь": {"brush_size": 58, "opacity": 0.72, "tolerance": 44},
    "Детальная ретушь": {"brush_size": 9, "opacity": 0.34, "tolerance": 12},
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


class PhotoRedactorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
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
        self.action_recording = False
        self.recorded_actions: list[str] = []
        self.tool = tk.StringVar(value="brush")
        self.tool_order = [value for _label, value, _description in TOOL_DEFINITIONS]
        self.visible_tools = list(self.tool_order)
        self.tool_pane_position = 360
        self.paint_target = tk.StringVar(value="pixels")
        self.retouch_preset = tk.StringVar(value="Средняя ретушь")
        self.zoom = tk.DoubleVar(value=1.0)
        self.brush_size = tk.IntVar(value=28)
        self.opacity = tk.DoubleVar(value=1.0)
        self.tolerance = tk.IntVar(value=24)
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
        self._stroke_layer_id: str | None = None
        self._stroke_kind = "pixels"
        self._stroke_rect: tuple[int, int, int, int] | None = None
        self._stroke_before: np.ndarray | None = None
        self._opacity_layer_id: str | None = None
        self._opacity_before: float | None = None
        self._space_down = False
        self._panning = False
        self.selection_id: int | None = None
        self.selection_box: tuple[int, int, int, int] | None = None
        self._lasso_points: list[tuple[int, int]] = []
        self._polygon_points: list[tuple[int, int]] = []
        self._polygon_ids: list[int] = []
        self._magnetic_edges: np.ndarray | None = None
        self._quick_points: list[tuple[int, int]] = []
        self._quick_mode = "replace"
        self._quick_preview_id: int | None = None
        self._quick_preview_image: ImageTk.PhotoImage | None = None
        self._last_quick_preview_time = 0.0
        self._brush_preview_ids: list[int] = []
        self._last_pointer_event = None
        self._clone_source: tuple[int, int] | None = None
        self._clone_anchor_target: tuple[int, int] | None = None
        self._clone_anchor_source: tuple[int, int] | None = None
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
        self._canvas_image_id: int | None = None
        self._canvas_origin = (0, 0)
        self._render_after_id: str | None = None
        self._last_render_time = 0.0
        self._composite_cache = None
        self._composite_dirty = True
        self._view_dirty = True

        self._build_ui()
        self.tool.trace_add("write", self.tool_changed)
        self.brush_size.trace_add("write", self.brush_size_changed)
        self.tolerance.trace_add("write", self.quick_preview_settings_changed)
        self.load_settings()
        self.refresh_recent_menu()
        self.refresh()
        self.after(500, self.check_recovery_file)
        self.schedule_autosave()

    def destroy(self) -> None:
        self.autosave_recovery()
        self.save_settings()
        self.executor.shutdown(wait=False, cancel_futures=True)
        super().destroy()

    def load_settings(self) -> None:
        try:
            if self.settings_path.exists():
                data = json.loads(self.settings_path.read_text(encoding="utf-8"))
                self.recent_files = [str(path) for path in data.get("recent_files", []) if Path(path).exists()]
                self.tool_order = normalize_tool_order(data.get("tool_order"), TOOL_DEFINITIONS)
                self.visible_tools = normalize_visible_tools(data.get("visible_tools"), self.tool_order)
                self.tool_pane_position = int(data.get("tool_pane_position", self.tool_pane_position))
                if self.tool.get() not in self.tool_order:
                    self.tool.set(self.visible_tools[0])
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
            self.settings_path.write_text(
                json.dumps(
                    {
                        "recent_files": self.recent_files[:12],
                        "tool_order": self.tool_order,
                        "visible_tools": self.visible_tools,
                        "tool_pane_position": self.tool_pane_position,
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
            return
        for path in self.recent_files:
            label = Path(path).name
            self.recent_menu.add_command(label=label, command=lambda p=path: self.open_path(p))
        self.recent_menu.add_separator()
        self.recent_menu.add_command(label="Очистить список", command=self.clear_recent_files)

    def clear_recent_files(self) -> None:
        self.recent_files.clear()
        self.save_settings()
        self.refresh_recent_menu()

    def _build_ui(self) -> None:
        self._build_menu()
        root = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        root.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(root, width=250)
        center = ttk.Frame(root)
        right = ttk.Frame(root, width=280)
        root.add(left, weight=0)
        root.add(center, weight=1)
        root.add(right, weight=0)

        self._build_tools(left)
        self._build_canvas(center)
        self._build_panels(right)

        self.status = ttk.Label(self, text="", anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        self.bind_all("<Control-z>", lambda _e: self.undo())
        self.bind_all("<Control-y>", lambda _e: self.redo())
        self.bind_all("<Control-s>", lambda _e: self.save())
        self.bind_all("<Control-o>", lambda _e: self.open_file())
        self.bind_all("<Control-n>", lambda _e: self.new_document())
        self.bind_all("<plus>", lambda _e: self.set_zoom(self.zoom.get() * 1.25))
        self.bind_all("<minus>", lambda _e: self.set_zoom(self.zoom.get() / 1.25))
        self.bind_all("<KeyPress-space>", self.space_down)
        self.bind_all("<KeyRelease-space>", self.space_up)

    def _build_menu(self) -> None:
        menu = tk.Menu(self)
        self.config(menu=menu)

        file_menu = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Новый", command=self.new_document, accelerator="Ctrl+N")
        file_menu.add_command(label="Новый из пресета", command=self.new_from_preset)
        file_menu.add_command(label="Открыть изображение/проект", command=self.open_file, accelerator="Ctrl+O")
        self.recent_menu = tk.Menu(file_menu, tearoff=False)
        file_menu.add_cascade(label="Недавние файлы", menu=self.recent_menu)
        file_menu.add_command(label="Поместить встроенное", command=self.place_embedded)
        file_menu.add_command(label="Поместить связанное", command=self.place_linked)
        file_menu.add_command(label="Загрузить файлы как слои", command=self.load_files_as_layers)
        file_menu.add_separator()
        file_menu.add_command(label="Сохранить проект", command=self.save, accelerator="Ctrl+S")
        file_menu.add_command(label="Сохранить проект как", command=self.save_as_project)
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
        menu.add_cascade(label="Правка", menu=edit)
        edit.add_command(label="Отменить", command=self.undo, accelerator="Ctrl+Z")
        edit.add_command(label="Повторить", command=self.redo, accelerator="Ctrl+Y")
        edit.add_separator()
        edit.add_command(label="Снять выделение", command=self.clear_selection)

        self.tools_menu = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Инструменты", menu=self.tools_menu)
        self.refresh_tool_menu()

        select = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Выделение", menu=select)
        select.add_command(label="Выделить все", command=self.select_all)
        select.add_command(label="Инвертировать выделение", command=self.invert_selection)
        select.add_command(label="Снять выделение", command=self.clear_selection)
        select.add_separator()
        select.add_command(label="Выделить непрозрачные пиксели", command=self.select_opaque_pixels)
        select.add_command(label="Выделить объект", command=self.select_subject)
        select.add_command(label="Выделить фон", command=self.select_background)
        select.add_command(label="Выделить небо", command=self.select_sky)
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
        image.add_command(label="Обрезать по выделению", command=self.crop_to_selection)
        image.add_command(label="Обрезать прозрачные пиксели", command=self.trim_transparent)
        image.add_command(label="Показать все слои", command=self.reveal_all)
        image.add_separator()
        image.add_command(label="Повернуть на 90 по часовой", command=lambda: self.rotate(90))
        image.add_command(label="Повернуть на 180", command=lambda: self.rotate(180))
        image.add_command(label="Отразить горизонтально", command=lambda: self.flip(horizontal=True))
        image.add_command(label="Отразить вертикально", command=lambda: self.flip(horizontal=False))

        layer = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Слой", menu=layer)
        layer.add_command(label="Новый слой", command=self.new_layer)
        layer.add_command(label="Дублировать слой", command=self.duplicate_layer)
        layer.add_command(label="Удалить слой", command=self.delete_layer)
        layer.add_command(label="Переименовать слой", command=self.rename_layer)
        layer.add_command(label="Заблокировать/разблокировать", command=self.toggle_layer_lock)
        layer.add_command(label="Редактировать текстовый слой", command=self.edit_text_layer)
        layer.add_command(label="Редактировать фигуру", command=self.edit_shape_layer)
        layer.add_command(label="Булева операция фигур", command=self.boolean_shape_layers)
        layer.add_command(label="\u0420\u0435\u0434\u0430\u043a\u0442\u0438\u0440\u043e\u0432\u0430\u0442\u044c \u043a\u043e\u0440\u0440\u0435\u043a\u0442\u0438\u0440\u0443\u044e\u0449\u0438\u0439 \u0441\u043b\u043e\u0439", command=self.edit_adjustment_layer)
        layer.add_separator()
        layer.add_command(label="Поднять выше", command=lambda: self.move_layer(1))
        layer.add_command(label="Опустить ниже", command=lambda: self.move_layer(-1))
        layer.add_command(label="Свободная трансформация", command=self.free_transform_layer)
        layer.add_command(label="Трансформировать выделенные пиксели", command=self.transform_selected_pixels)
        layer.add_command(label="Перспективная трансформация", command=self.perspective_transform_layer)
        layer.add_command(label="Деформация слоя", command=self.warp_layer)
        layer.add_command(label="Обновить связанный слой", command=self.update_linked_layer)
        layer.add_command(label="Перелинковать слой", command=self.relink_layer)
        layer.add_command(label="Переключить обтравочную маску", command=self.toggle_clipping_mask)
        layer.add_command(label="Стили слоя", command=self.edit_layer_styles)
        layer.add_command(label="Фильтры слоя", command=self.edit_layer_filters)
        layer.add_command(label="Очистить фильтры слоя", command=self.clear_layer_filters)
        layer.add_command(label="Объединить с нижним", command=self.merge_down)
        layer.add_command(label="Свести изображение", command=self.flatten)
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

        analysis = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Анализ", menu=analysis)
        analysis.add_command(label="Статистика изображения", command=self.show_image_statistics)
        analysis.add_command(label="Гистограмма", command=self.show_histogram)
        analysis.add_command(label="Метаданные / EXIF", command=self.show_metadata)

        actions = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Действия", menu=actions)
        actions.add_command(label="Начать запись", command=self.start_action_recording)
        actions.add_command(label="Остановить запись", command=self.stop_action_recording)
        actions.add_command(label="Сохранить запись", command=self.save_action_recording)
        actions.add_command(label="Очистить запись", command=self.clear_action_recording)

        view = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Вид", menu=view)
        view.add_command(label="Увеличить", command=lambda: self.set_zoom(self.zoom.get() * 1.25))
        view.add_command(label="Уменьшить", command=lambda: self.set_zoom(self.zoom.get() / 1.25))
        view.add_command(label="100%", command=lambda: self.set_zoom(1.0))
        view.add_command(label="По размеру окна", command=self.fit_to_screen)
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
        parent.configure(width=250)
        self.tool_split = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        self.tool_split.pack(fill=tk.BOTH, expand=True)
        tool_area = ttk.Frame(self.tool_split)
        options_area = ttk.Frame(self.tool_split)
        self.tool_split.add(tool_area, weight=1)
        self.tool_split.add(options_area, weight=1)
        self.tool_palette = ToolPalette(
            tool_area,
            definitions=TOOL_DEFINITIONS,
            tool_var=self.tool,
            order=self.tool_order,
            visible=self.visible_tools,
            select_tool=self.select_tool,
            configure_tools=self.configure_tool_palette,
            tooltip_factory=ToolTip,
        )
        self.tool_palette.pack(fill=tk.BOTH, expand=True)
        self.tool_options_panel = ToolOptionsPanel(
            options_area,
            tool_var=self.tool,
            definitions=TOOL_DEFINITIONS,
            brush_size=self.brush_size,
            opacity=self.opacity,
            tolerance=self.tolerance,
            paint_target=self.paint_target,
            retouch_preset=self.retouch_preset,
            retouch_presets=RETOUCH_PRESETS,
            pick_foreground=self.pick_foreground,
            pick_background=self.pick_background,
            set_paint_target=self.set_paint_target,
            apply_retouch_preset=self.apply_retouch_preset,
            tooltip_factory=ToolTip,
        )
        self.tool_options_panel.pack(fill=tk.BOTH, expand=True)
        self.after_idle(self.apply_tool_pane_position)

    def select_tool(self, value: str) -> None:
        if value in self.tool_order:
            self.tool.set(value)

    def tool_changed(self, *_args) -> None:
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
        label = self.tool_label(self.tool.get())
        if label:
            self.status_text(f"Инструмент: {label}")

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
            self.tools_menu.add_radiobutton(label=label, value=value, variable=self.tool, command=lambda v=value: self.select_tool(v))
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
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X)
        ttk.Button(toolbar, text="-", command=lambda: self.set_zoom(self.zoom.get() / 1.25), width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="+", command=lambda: self.set_zoom(self.zoom.get() * 1.25), width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="100%", command=lambda: self.set_zoom(1.0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Вписать", command=self.fit_to_screen).pack(side=tk.LEFT, padx=2)
        self.zoom_label = ttk.Label(toolbar, text="100%")
        self.zoom_label.pack(side=tk.LEFT, padx=10)

        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(frame, bg="#24262b", highlightthickness=0)
        xbar = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        ybar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self.canvas.bind("<ButtonPress-1>", self.pointer_down)
        self.canvas.bind("<B1-Motion>", self.pointer_drag)
        self.canvas.bind("<ButtonRelease-1>", self.pointer_up)
        self.canvas.bind("<Double-Button-1>", self.pointer_double_click)
        self.canvas.bind("<ButtonPress-2>", self.pan_down)
        self.canvas.bind("<B2-Motion>", self.pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self.pan_up)
        self.canvas.bind("<Motion>", self.pointer_motion)
        self.canvas.bind("<Leave>", self.pointer_leave)
        self.canvas.bind("<MouseWheel>", self.mouse_wheel)

    def apply_retouch_preset(self) -> None:
        preset = RETOUCH_PRESETS.get(self.retouch_preset.get())
        if preset is None:
            return
        self.brush_size.set(int(preset["brush_size"]))
        self.opacity.set(float(preset["opacity"]))
        self.tolerance.set(int(preset["tolerance"]))
        if self.tool.get() not in {"blur_tool", "sharpen_tool", "dodge", "burn", "clone", "healing"}:
            self.tool.set("healing")
        self.status_text(f"Пресет ретуши: {self.retouch_preset.get()}")

    def _build_panels(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Слои").pack(anchor=tk.W, padx=8, pady=(8, 4))
        self.layer_list = tk.Listbox(parent, height=16, exportselection=False)
        self.layer_list.pack(fill=tk.BOTH, expand=False, padx=8)
        self.layer_list.bind("<<ListboxSelect>>", self.layer_selected)
        buttons = ttk.Frame(parent)
        buttons.pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(buttons, text="+", width=3, command=self.new_layer).pack(side=tk.LEFT)
        ttk.Button(buttons, text="x", width=3, command=self.delete_layer).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Дубль", command=self.duplicate_layer).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Вверх", command=lambda: self.move_layer(1)).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Вниз", command=lambda: self.move_layer(-1)).pack(side=tk.LEFT)
        ttk.Label(parent, text="Непрозрачность слоя").pack(anchor=tk.W, padx=8)
        self.layer_opacity = tk.DoubleVar(value=1.0)
        self.layer_opacity_scale = ttk.Scale(parent, from_=0.0, to=1.0, variable=self.layer_opacity, command=self.change_layer_opacity)
        self.layer_opacity_scale.pack(fill=tk.X, padx=8)
        self.layer_opacity_scale.bind("<ButtonPress-1>", self.begin_layer_opacity_change)
        self.layer_opacity_scale.bind("<ButtonRelease-1>", self.end_layer_opacity_change)
        ttk.Label(parent, text="Режим наложения").pack(anchor=tk.W, padx=8, pady=(6, 0))
        self.blend_mode = tk.StringVar(value="Normal")
        self.blend_mode_box = ttk.Combobox(parent, textvariable=self.blend_mode, values=BLEND_MODES, state="readonly")
        self.blend_mode_box.pack(fill=tk.X, padx=8)
        self.blend_mode_box.bind("<<ComboboxSelected>>", self.change_blend_mode)
        ttk.Button(parent, text="Показать / скрыть", command=self.toggle_layer_visible).pack(fill=tk.X, padx=8, pady=(6, 2))
        ttk.Button(parent, text="Блокировка", command=self.toggle_layer_lock).pack(fill=tk.X, padx=8, pady=(0, 6))
        ttk.Separator(parent).pack(fill=tk.X, pady=8)
        ttk.Label(parent, text="Миниатюры").pack(anchor=tk.W, padx=8, pady=(0, 4))
        thumbs = ttk.Frame(parent)
        thumbs.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Label(thumbs, text="Слой").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(thumbs, text="Маска").grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        self.layer_thumb = ttk.Label(thumbs)
        self.layer_thumb.grid(row=1, column=0, sticky=tk.W)
        self.mask_thumb = ttk.Label(thumbs)
        self.mask_thumb.grid(row=1, column=1, sticky=tk.W, padx=(10, 0))
        self.layer_thumb.bind("<Button-1>", self.edit_pixels_channel)
        self.mask_thumb.bind("<Button-1>", self.edit_mask_channel)
        ToolTip(self.layer_thumb, "Миниатюра пикселей активного слоя. Клик переключает кисть обратно на пиксели.")
        ToolTip(self.mask_thumb, "Миниатюра маски активного слоя. Клик включает рисование по маске и черно-белый просмотр канала.")
        ttk.Label(parent, text="Просмотр маски").pack(anchor=tk.W, padx=8)
        self.mask_preview_box = ttk.Combobox(parent, textvariable=self.mask_preview, values=MASK_PREVIEW_MODES, state="readonly")
        self.mask_preview_box.pack(fill=tk.X, padx=8, pady=(0, 6))
        self.mask_preview_box.bind("<<ComboboxSelected>>", lambda _event: self.set_mask_preview())
        ToolTip(self.mask_preview_box, "Показывает активную маску поверх холста или как черно-белый канал без изменения документа.")
        ttk.Separator(parent).pack(fill=tk.X, pady=8)
        self.info = ttk.Label(parent, text="", justify=tk.LEFT)
        self.info.pack(anchor=tk.W, padx=8)

    def push_command(self, command) -> None:
        self.history.push(command)
        self.record_action(command.label)
        self.status_text(command.label)

    def run_document_command(self, label: str, fn) -> None:
        before = self.doc.raw_state()
        fn()
        after = self.doc.raw_state()
        self.history.push(DocumentStateCommand(label, before, after))
        self.record_action(label)
        self.status_text(label)

    def run_selection_command(self, label: str, fn) -> None:
        before = None if self.doc.selection_mask is None else self.doc.selection_mask.copy()
        fn()
        after = None if self.doc.selection_mask is None else self.doc.selection_mask.copy()
        self.history.push(SelectionMaskCommand(label, before, after))
        self.record_action(label)
        self.selection_box = self.doc.selection_bounds()
        self.update_selection_overlay()
        self.status_text(label)

    def record_action(self, label: str) -> None:
        if self.action_recording:
            self.recorded_actions.append(label)

    def start_action_recording(self) -> None:
        self.action_recording = True
        self.recorded_actions.clear()
        self.status_text("Action recording started")

    def stop_action_recording(self) -> None:
        self.action_recording = False
        self.status_text(f"Action recording stopped: {len(self.recorded_actions)} steps")

    def clear_action_recording(self) -> None:
        self.recorded_actions.clear()
        self.status_text("Action recording cleared")

    def save_action_recording(self) -> None:
        if not self.recorded_actions:
            messagebox.showinfo("Actions", "No recorded steps.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Action JSON", "*.json")])
        if not path:
            return
        data = {"name": Path(path).stem, "steps": list(self.recorded_actions), "format": "PhotoRedactor action log v1"}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self.status_text(f"Saved action: {path}")

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
            messagebox.showinfo("Recovery", "No recovery file found.")
            return
        self.doc = Document.open_project(self.recovery_path)
        self.history.clear()
        self.selection_box = self.doc.selection_bounds()
        self.refresh()
        self.status_text(f"Opened recovery: {self.recovery_path}")

    def clear_recovery(self) -> None:
        if self.recovery_path.exists():
            self.recovery_path.unlink()
        self.status_text("Recovery cleared")

    def undo(self) -> None:
        label = self.history.undo(self.doc)
        if label:
            self.invalidate_pixels()
            self.refresh()
            self.status_text(f"Undo: {label}")

    def redo(self) -> None:
        label = self.history.redo(self.doc)
        if label:
            self.invalidate_pixels()
            self.refresh()
            self.status_text(f"Redo: {label}")

    def invalidate_pixels(self) -> None:
        self._composite_dirty = True
        self._view_dirty = True

    def invalidate_view(self) -> None:
        self._view_dirty = True

    def refresh_canvas(self) -> None:
        if self._composite_dirty or self._composite_cache is None:
            self._composite_cache = self.doc.composite(checker=True)
            self._composite_dirty = False
        display = self._channel_display(self._composite_cache)
        display = self._mask_preview_display(display)
        image = rgba_array_to_pil(display)
        scale = self.zoom.get()
        if scale != 1.0:
            resample = Image.Resampling.NEAREST if scale >= 4 else Image.Resampling.BILINEAR
            image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), resample)
        self._preview_image = ImageTk.PhotoImage(image)
        pad_x = max(40, self.canvas.winfo_width() // 2)
        pad_y = max(40, self.canvas.winfo_height() // 2)
        self._canvas_origin = (pad_x, pad_y)
        if self._canvas_image_id is None:
            self._canvas_image_id = self.canvas.create_image(pad_x, pad_y, image=self._preview_image, anchor=tk.NW)
        else:
            self.canvas.itemconfigure(self._canvas_image_id, image=self._preview_image)
            self.canvas.coords(self._canvas_image_id, pad_x, pad_y)
        self.canvas.configure(scrollregion=(0, 0, image.width + pad_x * 2, image.height + pad_y * 2))
        self.zoom_label.configure(text=f"{round(scale * 100)}%")
        self._last_render_time = time.perf_counter()
        self._view_dirty = False
        self.update_selection_overlay()
        self.update_grid_and_guides()
        if self.tool.get() == "quick_selection" and self._quick_points:
            self.update_quick_selection_preview(force=True)
        if self._last_pointer_event is not None:
            self.update_brush_preview(self._last_pointer_event)

    def _channel_display(self, composite: np.ndarray) -> np.ndarray:
        channel = self.view_channel.get() if hasattr(self, "view_channel") else "RGB"
        if channel == "RGB":
            return composite
        source = self.doc.composite(checker=False)
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

    def request_canvas_refresh(self) -> None:
        self.invalidate_pixels()
        if self._render_after_id is not None:
            return
        elapsed_ms = (time.perf_counter() - self._last_render_time) * 1000
        delay = 0 if elapsed_ms >= 33 else int(33 - elapsed_ms)
        self._render_after_id = self.after(delay, self._run_scheduled_canvas_refresh)

    def _run_scheduled_canvas_refresh(self) -> None:
        self._render_after_id = None
        self.refresh_canvas()

    def refresh(self) -> None:
        self.invalidate_pixels()
        if self._render_after_id is not None:
            self.after_cancel(self._render_after_id)
            self._render_after_id = None
        self.refresh_canvas()
        self.refresh_layers()
        self.info.configure(text=f"{self.doc.width} x {self.doc.height}px\nСлоев: {len(self.doc.layers)}\nАктивный: {self.doc.layer.name}")

    def refresh_layers(self) -> None:
        self.layer_list.delete(0, tk.END)
        for i, layer in enumerate(reversed(self.doc.layers)):
            marker = "*" if layer.visible else "-"
            mask_marker = "U" if layer.mask is not None and not layer.mask_linked else "M" if layer.mask is not None and layer.mask_enabled else "m" if layer.mask is not None else " "
            lock_marker = "L" if layer.locked else " "
            kind_marker = "T" if layer.kind == "text" else "A" if layer.kind == "adjustment" else "S" if layer.kind == "shape" else "L" if layer.kind == "linked" else "E" if layer.kind == "embedded" else " "
            clip_marker = "C" if layer.clipping else " "
            fx_marker = "F" if layer.effects else " "
            filter_marker = "P" if layer.filters else " "
            self.layer_list.insert(tk.END, f"{marker}{mask_marker}{lock_marker}{kind_marker}{clip_marker}{fx_marker}{filter_marker} {layer.name}  {round(layer.opacity * 100)}%")
        self.layer_list.selection_clear(0, tk.END)
        self.layer_list.selection_set(len(self.doc.layers) - 1 - self.doc.active_layer)
        self.layer_opacity.set(self.doc.layer.opacity)
        self.blend_mode.set(self.doc.layer.blend_mode)
        self.refresh_layer_previews()

    def refresh_layer_previews(self) -> None:
        layer = self.doc.layer
        self._layer_thumb_image = ImageTk.PhotoImage(self.make_layer_thumbnail(layer.pixels))
        self.layer_thumb.configure(image=self._layer_thumb_image)
        self._mask_thumb_image = ImageTk.PhotoImage(self.make_mask_thumbnail(layer.mask))
        self.mask_thumb.configure(image=self._mask_thumb_image)

    def make_layer_thumbnail(self, pixels: np.ndarray, size: int = 64) -> Image.Image:
        image = rgba_array_to_pil(pixels)
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (size, size), (44, 46, 52, 255))
        x = (size - image.width) // 2
        y = (size - image.height) // 2
        canvas.alpha_composite(image, (x, y))
        return canvas

    def make_mask_thumbnail(self, mask: np.ndarray | None, size: int = 64) -> Image.Image:
        if mask is None:
            return Image.new("RGBA", (size, size), (72, 74, 82, 255))
        image = Image.fromarray(mask.astype(np.uint8), "L")
        image.thumbnail((size, size), Image.Resampling.NEAREST)
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

    def run_background(self, label: str, worker, done=None) -> None:
        self.status_text(f"{label}...")
        future = self.executor.submit(worker)

        def complete() -> None:
            try:
                result = future.result()
            except Exception as exc:
                messagebox.showerror(label, str(exc))
                self.status_text(f"{label}: error")
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

    @staticmethod
    def selection_mode_from_event(event) -> str:
        shift = bool(event.state & 0x0001)
        ctrl = bool(event.state & 0x0004)
        if shift and ctrl:
            return "intersect"
        if shift:
            return "add"
        if ctrl:
            return "subtract"
        return "replace"

    @staticmethod
    def brush_preview_tools() -> set[str]:
        return {"brush", "eraser", "blur_tool", "sharpen_tool", "dodge", "burn", "clone", "healing", "quick_selection"}

    def pointer_motion(self, event) -> None:
        self._last_pointer_event = event
        if not self._panning:
            self.update_brush_preview(event)

    def pointer_leave(self, _event) -> None:
        self._last_pointer_event = None
        self.clear_brush_preview()

    def brush_size_changed(self, *_args) -> None:
        if self._last_pointer_event is not None:
            self.update_brush_preview(self._last_pointer_event)
        self.quick_preview_settings_changed()

    def quick_preview_settings_changed(self, *_args) -> None:
        if self.tool.get() == "quick_selection" and self._quick_points:
            self.update_quick_selection_preview(force=True)

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
        mask = self.doc.preview_quick_selection_brush(
            self.doc.layer,
            self._quick_points,
            max(2, int(self.brush_size.get())),
            int(self.tolerance.get()),
            self._quick_mode,
        )
        if mask is None or not np.any(mask):
            self.clear_quick_selection_preview()
            return
        mask_image = Image.fromarray(mask, mode="L")
        scale = self.zoom.get()
        if scale != 1.0:
            mask_image = mask_image.resize(
                (max(1, int(self.doc.width * scale)), max(1, int(self.doc.height * scale))),
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
        if self._quick_preview_id is None:
            self._quick_preview_id = self.canvas.create_image(ox, oy, image=self._quick_preview_image, anchor=tk.NW)
        else:
            self.canvas.itemconfigure(self._quick_preview_id, image=self._quick_preview_image)
            self.canvas.coords(self._quick_preview_id, ox, oy)
        self.canvas.tag_raise(self._quick_preview_id)
        for item_id in self._brush_preview_ids:
            self.canvas.tag_raise(item_id)

    def clear_quick_selection_preview(self) -> None:
        if self._quick_preview_id is not None:
            self.canvas.delete(self._quick_preview_id)
            self._quick_preview_id = None
        self._quick_preview_image = None

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
        point = self.canvas_to_doc(event)
        self.drag_start = point
        self.last_point = point
        tool = self.tool.get()
        if tool in ["brush", "eraser", "blur_tool", "sharpen_tool", "dodge", "burn", "clone", "healing"]:
            if tool in ["clone", "healing"]:
                if event.state & 0x0008:
                    self._clone_source = point
                    self.status_text(f"Источник штампа: {point[0]}, {point[1]}")
                    self.drag_start = None
                    return
                if self._clone_source is None:
                    self.status_text("Сначала Alt+клик задает источник штампа")
                    self.drag_start = None
                    return
                self._clone_anchor_target = point
                self._clone_anchor_source = self._clone_source
            kind = "mask" if tool in ["brush", "eraser"] and self.paint_target.get() == "mask" else "pixels"
            self.begin_stroke(kind)
            self.paint_at(point)
        elif tool == "fill":
            self.run_document_command("Fill", lambda: flood_fill(self.doc.layer, point[0], point[1], self.foreground, int(self.tolerance.get()), self.doc.layer_selection_mask(self.doc.layer)))
            self.doc.dirty = True
            self.refresh()
        elif tool == "magic_wand":
            mode = self.selection_mode_from_event(event)
            self.run_selection_command("Magic wand selection", lambda: self.doc.magic_wand_selection(self.doc.layer, point[0], point[1], int(self.tolerance.get()), mode))
        elif tool == "color_range":
            mode = self.selection_mode_from_event(event)
            self.run_selection_command("Color range selection", lambda: self.doc.color_range_selection(self.doc.layer, point[0], point[1], int(self.tolerance.get()), mode))
        elif tool == "quick_selection":
            self._quick_points = [point]
            self._quick_mode = self.selection_mode_from_event(event)
            self.update_quick_selection_preview(force=True)
            self.status_text("Кисть быстрого выделения")
        elif tool == "patch":
            self.begin_patch_drag(point)
        elif tool == "eyedropper":
            self.pick_color_from_document(point)
        elif tool == "text":
            data = self.text_layer_dialog("Text layer", {"text": "", "size": 48, "font_family": "Arial", "box_width": 0, "align": "left", "line_spacing": 10, "tracking": 0})
            if data and data["text"]:
                self.run_document_command(
                    "Text layer",
                    lambda: self.doc.add_text_layer(
                        data["text"],
                        point[0],
                        point[1],
                        self.foreground,
                        data["size"],
                        data["font_family"],
                        data["box_width"],
                        data["align"],
                        data["line_spacing"],
                        data["tracking"],
                    ),
                )
                self.doc.dirty = True
                self.refresh()
        elif tool == "move":
            if self.doc.layer.locked:
                self.status_text("Слой заблокирован")
                return
            self._move_layer_id = self.doc.layer.id
            self._move_start = (self.doc.layer.x, self.doc.layer.y)
            self._move_start_mask = None if self.doc.layer.mask is None else self.doc.layer.mask.copy()
        elif tool == "polygon_lasso":
            self._polygon_points.append(point)
            self.draw_polygon_lasso()
        elif tool == "lasso":
            self._lasso_points = [point]
        elif tool == "magnetic_lasso":
            self._magnetic_edges = self.doc.magnetic_edge_map()
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
        if tool in ["brush", "eraser", "blur_tool", "sharpen_tool", "dodge", "burn", "clone", "healing"]:
            self.paint_line(self.last_point or point, point)
            self.last_point = point
        elif tool == "move" and self.drag_start:
            dx, dy = point[0] - self.drag_start[0], point[1] - self.drag_start[1]
            self.doc.move_active_layer(dx, dy)
            self.drag_start = point
            self.request_canvas_refresh()
        elif tool in ["select", "ellipse_select", "crop", "gradient", "rect_shape", "ellipse_shape", "line_shape", "bezier_shape", "polygon_shape", "star_shape"]:
            self.draw_selection(self.drag_start, point)
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
        if tool in ["brush", "eraser", "blur_tool", "sharpen_tool", "dodge", "burn", "clone", "healing"]:
            self.end_stroke(f"{tool.title()} stroke")
            self._clone_anchor_target = None
            self._clone_anchor_source = None
        elif tool == "move":
            self.end_move_layer()
        elif tool == "gradient" and self.drag_start:
            self.run_document_command("Gradient", lambda: apply_gradient(self.doc.layer, (*self.drag_start, *point), self.foreground, self.background, self.doc.layer_selection_mask(self.doc.layer)))
            self.doc.dirty = True
            self.refresh()
        elif tool in ["rect_shape", "ellipse_shape", "line_shape", "bezier_shape", "polygon_shape", "star_shape"] and self.drag_start:
            self.create_shape_from_drag(tool, (*self.drag_start, *point))
            self.refresh()
        elif tool in ["select", "ellipse_select", "crop"] and self.drag_start:
            self.selection_box = (*self.drag_start, *point)
            if tool == "select":
                mode = self.selection_mode_from_event(event)
                self.run_selection_command("Rectangular selection", lambda: self.doc.set_rect_selection(self.selection_box, mode))
            elif tool == "ellipse_select":
                mode = self.selection_mode_from_event(event)
                self.run_selection_command("Elliptical selection", lambda: self.doc.set_ellipse_selection(self.selection_box, mode))
            self.draw_selection(self.drag_start, point)
        elif tool == "lasso" and len(self._lasso_points) >= 3:
            mode = self.selection_mode_from_event(event)
            points = list(self._lasso_points)
            self.run_selection_command("Lasso selection", lambda: self.doc.set_polygon_selection(points, mode))
            self.clear_lasso_overlay()
        elif tool == "magnetic_lasso":
            if self.drag_start:
                self._lasso_points.append(self.magnetic_lasso_point(point))
            if len(self._lasso_points) >= 3:
                mode = self.selection_mode_from_event(event)
                points = list(self._lasso_points)
                self.run_selection_command("Magnetic lasso selection", lambda: self.doc.set_polygon_selection(points, mode))
            self.clear_lasso_overlay()
            self._magnetic_edges = None
        elif tool == "quick_selection" and self._quick_points:
            points = list(self._quick_points)
            mode = self._quick_mode
            radius = max(2, int(self.brush_size.get()))
            tolerance = int(self.tolerance.get())
            self._quick_points.clear()
            self.clear_quick_selection_preview()
            self.run_selection_command("Quick selection", lambda: self.doc.quick_selection_brush(self.doc.layer, points, radius, tolerance, mode))
        elif tool == "patch" and self.drag_start:
            self.finish_patch_drag(point)
        self.drag_start = None
        self.last_point = None

    def pointer_double_click(self, event) -> None:
        if self.tool.get() == "polygon_lasso" and len(self._polygon_points) >= 3:
            mode = self.selection_mode_from_event(event)
            points = list(self._polygon_points)
            self.run_selection_command("Polygon lasso selection", lambda: self.doc.set_polygon_selection(points, mode))
            self.clear_lasso_overlay()

    def begin_stroke(self, kind: str = "pixels") -> None:
        self._stroke_layer_id = self.doc.layer.id
        self._stroke_kind = kind
        if kind == "mask" and self.doc.layer.mask is None:
            self.doc.layer.mask = np.full(self.doc.layer.pixels.shape[:2], 255, dtype=np.uint8)
            self.doc.layer.mask_enabled = True
        self._stroke_rect = None
        self._stroke_before = None

    def brush_local_rect(self, point: tuple[int, int]) -> tuple[int, int, int, int] | None:
        layer = self.doc.layer
        radius = int(self.brush_size.get())
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
        layer = self.doc.layer
        target = layer.mask if self._stroke_kind == "mask" else layer.pixels
        if target is None:
            return
        if self._stroke_rect is None or self._stroke_before is None:
            x1, y1, x2, y2 = rect
            self._stroke_rect = rect
            self._stroke_before = target[y1:y2, x1:x2].copy()
            return
        old = self._stroke_rect
        new = union_rect(old, rect)
        if new == old:
            return
        nx1, ny1, nx2, ny2 = new
        ox1, oy1, ox2, oy2 = old
        merged = target[ny1:ny2, nx1:nx2].copy()
        merged[oy1 - ny1 : oy2 - ny1, ox1 - nx1 : ox2 - nx1] = self._stroke_before
        self._stroke_rect = new
        self._stroke_before = merged

    def end_stroke(self, label: str) -> None:
        if self._stroke_layer_id and self._stroke_rect and self._stroke_before is not None:
            layer = self.doc.get_layer(self._stroke_layer_id)
            if layer is not None:
                x1, y1, x2, y2 = self._stroke_rect
                if self._stroke_kind == "mask" and layer.mask is not None:
                    after = layer.mask[y1:y2, x1:x2].copy()
                    self.push_command(MaskPatchCommand(label, self._stroke_layer_id, self._stroke_rect, self._stroke_before, after))
                elif self._stroke_kind == "pixels":
                    after = layer.pixels[y1:y2, x1:x2].copy()
                    self.push_command(PixelPatchCommand(label, self._stroke_layer_id, self._stroke_rect, self._stroke_before, after))
        self._stroke_layer_id = None
        self._stroke_kind = "pixels"
        self._stroke_rect = None
        self._stroke_before = None

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

    def paint_at(self, point: tuple[int, int]) -> None:
        self.capture_stroke_before(self.brush_local_rect(point))
        tool = self.tool.get()
        selection_mask = self.doc.layer_selection_mask(self.doc.layer)
        if tool in ["clone", "healing"]:
            source = self.clone_source_for_point(point)
            if source is not None:
                clone_or_heal(self.doc.layer, source[0], source[1], point[0], point[1], int(self.brush_size.get()), float(self.opacity.get()), tool == "healing", selection_mask)
        elif self._stroke_kind == "mask":
            draw_mask_brush(self.doc.layer, point[0], point[1], int(self.brush_size.get()), 0 if tool == "eraser" else 255, float(self.opacity.get()), selection_mask)
        elif tool in ["blur_tool", "sharpen_tool", "dodge", "burn"]:
            mode = "blur" if tool == "blur_tool" else "sharpen" if tool == "sharpen_tool" else tool
            local_retouch(self.doc.layer, point[0], point[1], int(self.brush_size.get()), mode, float(self.opacity.get()), selection_mask)
        else:
            draw_brush(
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
        self.request_canvas_refresh()

    def paint_line(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        steps = max(1, int(((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5 / max(1, self.brush_size.get() / 4)))
        for i in range(steps + 1):
            t = i / steps
            x = round(start[0] * (1 - t) + end[0] * t)
            y = round(start[1] * (1 - t) + end[1] * t)
            self.capture_stroke_before(self.brush_local_rect((x, y)))
            tool = self.tool.get()
            selection_mask = self.doc.layer_selection_mask(self.doc.layer)
            if tool in ["clone", "healing"]:
                source = self.clone_source_for_point((x, y))
                if source is not None:
                    clone_or_heal(self.doc.layer, source[0], source[1], x, y, int(self.brush_size.get()), float(self.opacity.get()), tool == "healing", selection_mask)
            elif self._stroke_kind == "mask":
                draw_mask_brush(self.doc.layer, x, y, int(self.brush_size.get()), 0 if tool == "eraser" else 255, float(self.opacity.get()), selection_mask)
            elif tool in ["blur_tool", "sharpen_tool", "dodge", "burn"]:
                mode = "blur" if tool == "blur_tool" else "sharpen" if tool == "sharpen_tool" else tool
                local_retouch(self.doc.layer, x, y, int(self.brush_size.get()), mode, float(self.opacity.get()), selection_mask)
            else:
                draw_brush(
                    self.doc.layer,
                    x,
                    y,
                    int(self.brush_size.get()),
                    self.foreground,
                    float(self.opacity.get()),
                    tool == "eraser",
                    selection_mask,
                )
        self.doc.dirty = True
        self.request_canvas_refresh()

    def clone_source_for_point(self, point: tuple[int, int]) -> tuple[int, int] | None:
        if self._clone_anchor_source is None or self._clone_anchor_target is None:
            return self._clone_source
        return (
            self._clone_anchor_source[0] + point[0] - self._clone_anchor_target[0],
            self._clone_anchor_source[1] + point[1] - self._clone_anchor_target[1],
        )

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

    def draw_selection(self, start: tuple[int, int] | None, end: tuple[int, int]) -> None:
        if not start:
            return
        x1, y1 = self.doc_to_canvas(start[0], start[1])
        x2, y2 = self.doc_to_canvas(end[0], end[1])
        coords = [x1, y1, x2, y2]
        if self.selection_id is None:
            self.selection_id = self.canvas.create_rectangle(*coords, outline="#50e3ff", dash=(5, 4), width=2)
        else:
            self.canvas.coords(self.selection_id, *coords)

    def draw_lasso(self) -> None:
        self.delete_lasso_overlay()
        if len(self._lasso_points) < 2:
            return
        coords = [coord for point in self._lasso_points for xy in [self.doc_to_canvas(point[0], point[1])] for coord in xy]
        self._polygon_ids.append(self.canvas.create_line(*coords, fill="#50e3ff", dash=(4, 3), width=2, smooth=True))

    def draw_polygon_lasso(self) -> None:
        self.delete_lasso_overlay()
        if len(self._polygon_points) >= 2:
            coords = [coord for point in self._polygon_points for xy in [self.doc_to_canvas(point[0], point[1])] for coord in xy]
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
            self._magnetic_edges = self.doc.magnetic_edge_map()
        return self.doc.snap_point_to_edge(point, self._magnetic_edges, max(8, int(self.tolerance.get())))

    def update_selection_overlay(self) -> None:
        bounds = self.doc.selection_bounds()
        if bounds is None:
            if self.selection_id is not None:
                self.canvas.delete(self.selection_id)
                self.selection_id = None
            return
        x1, y1 = self.doc_to_canvas(bounds[0], bounds[1])
        x2, y2 = self.doc_to_canvas(bounds[2], bounds[3])
        coords = [x1, y1, x2, y2]
        if self.selection_id is None:
            self.selection_id = self.canvas.create_rectangle(*coords, outline="#50e3ff", dash=(5, 4), width=2)
        else:
            self.canvas.coords(self.selection_id, *coords)
        self.canvas.tag_raise(self.selection_id)
        self.update_grid_and_guides()

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

    def create_shape_from_drag(self, tool: str, box: tuple[int, int, int, int]) -> None:
        shape = "rectangle"
        sides = 5
        inner_ratio = 0.5
        if tool == "ellipse_shape":
            shape = "ellipse"
        elif tool == "line_shape":
            shape = "line"
        elif tool == "bezier_shape":
            shape = "bezier"
        elif tool == "polygon_shape":
            shape = "polygon"
            value = simpledialog.askinteger("Polygon shape", "Sides:", initialvalue=5, minvalue=3, maxvalue=64)
            if value is None:
                return
            sides = value
        elif tool == "star_shape":
            shape = "star"
            value = simpledialog.askinteger("Star shape", "Points:", initialvalue=5, minvalue=3, maxvalue=64)
            if value is None:
                return
            sides = value
            ratio = simpledialog.askfloat("Star shape", "Inner radius 0.05..0.95:", initialvalue=0.5, minvalue=0.05, maxvalue=0.95)
            if ratio is None:
                return
            inner_ratio = ratio
        stroke_width = int(self.brush_size.get()) if shape in {"line", "bezier"} else 2
        self.run_document_command(
            "Shape layer",
            lambda: self.doc.add_shape_layer(shape, box, self.foreground, self.background, stroke_width, sides, inner_ratio),
        )

    def clear_selection(self) -> None:
        self.run_selection_command("Clear selection", self.doc.clear_selection)

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
        smooth = data["smooth"]
        feather = data["feather"]
        contrast = data["contrast"]
        shift = data["shift"]
        output = data["output"]
        if output == "Маска слоя":
            def edit() -> None:
                self.doc.refine_selection(smooth, feather, contrast, shift)
                self.doc.add_mask_from_selection()

            self.run_document_command("Select and Mask to layer mask", edit)
            self.paint_target.set("mask")
            self.mask_preview.set(MASK_PREVIEW_CHANNEL)
            self.selection_box = self.doc.selection_bounds()
            self.refresh()
            self.status_text("Select and Mask: маска слоя")
        else:
            self.run_selection_command("Select and Mask", lambda: self.doc.refine_selection(smooth, feather, contrast, shift))
            self.refresh_canvas()

    def select_and_mask_dialog(self) -> dict[str, object] | None:
        source = self.doc.selection_mask.copy()
        dialog = tk.Toplevel(self)
        dialog.title("Выделить и маска")
        dialog.transient(self)
        dialog.resizable(False, False)
        dialog.grab_set()

        smooth = tk.IntVar(value=2)
        feather = tk.IntVar(value=2)
        contrast = tk.DoubleVar(value=1.25)
        shift = tk.IntVar(value=0)
        output = tk.StringVar(value="Выделение")
        preview_mode = tk.StringVar(value=SELECT_MASK_PREVIEW_CHANNEL)
        result: dict[str, object] | None = None

        preview = ttk.Label(dialog)
        preview.grid(row=0, column=0, rowspan=8, padx=12, pady=12, sticky="n")
        stats = ttk.Label(dialog, text="", justify=tk.LEFT)
        stats.grid(row=8, column=0, padx=12, pady=(0, 12), sticky="w")

        def current_mask() -> np.ndarray:
            return refine_selection_mask(source, int(smooth.get()), int(feather.get()), float(contrast.get()), int(shift.get()))

        def update_preview(*_args) -> None:
            mask = current_mask()
            canvas = self.render_select_mask_preview(self.doc.composite(checker=False), mask, preview_mode.get(), 160)
            self._select_mask_preview_image = ImageTk.PhotoImage(canvas)
            preview.configure(image=self._select_mask_preview_image)
            selected = int(np.count_nonzero(mask))
            stats.configure(text=f"Активных пикселей: {selected}\nГраницы: {self.mask_bounds(mask) or '-'}")

        def add_spin(row: int, label: str, variable, from_: float, to: float, increment: float = 1.0) -> None:
            ttk.Label(dialog, text=label).grid(row=row, column=1, sticky="w", padx=(0, 12), pady=(8, 0))
            spin = ttk.Spinbox(dialog, textvariable=variable, from_=from_, to=to, increment=increment, width=10, command=update_preview)
            spin.grid(row=row, column=2, sticky="ew", padx=(0, 12), pady=(8, 0))

        add_spin(0, "Сглаживание", smooth, 0, 100)
        add_spin(1, "Растушевка", feather, 0, 500)
        add_spin(2, "Контраст", contrast, 0.0, 5.0, 0.05)
        add_spin(3, "Сдвиг края", shift, -500, 500)
        ttk.Label(dialog, text="Результат").grid(row=4, column=1, sticky="w", padx=(0, 12), pady=(8, 0))
        output_box = ttk.Combobox(dialog, textvariable=output, values=["Выделение", "Маска слоя"], state="readonly", width=14)
        output_box.grid(row=4, column=2, sticky="ew", padx=(0, 12), pady=(8, 0))
        ttk.Label(dialog, text="Просмотр").grid(row=5, column=1, sticky="w", padx=(0, 12), pady=(8, 0))
        preview_box = ttk.Combobox(dialog, textvariable=preview_mode, values=SELECT_MASK_PREVIEW_MODES, state="readonly", width=18)
        preview_box.grid(row=5, column=2, sticky="ew", padx=(0, 12), pady=(8, 0))

        buttons = ttk.Frame(dialog)
        buttons.grid(row=7, column=1, columnspan=2, sticky="e", padx=12, pady=12)

        def accept() -> None:
            nonlocal result
            result = {
                "smooth": max(0, int(smooth.get())),
                "feather": max(0, int(feather.get())),
                "contrast": max(0.0, float(contrast.get())),
                "shift": int(shift.get()),
                "output": output.get(),
            }
            dialog.destroy()

        def cancel() -> None:
            dialog.destroy()

        ttk.Button(buttons, text="ОК", command=accept).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Отмена", command=cancel).pack(side=tk.RIGHT)
        preview_box.bind("<<ComboboxSelected>>", update_preview)
        for variable in [smooth, feather, contrast, shift]:
            variable.trace_add("write", update_preview)
        dialog.protocol("WM_DELETE_WINDOW", cancel)
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
        mask_image = Image.fromarray(mask.astype(np.uint8), "L")
        source = rgba_array_to_pil(composite)
        source.thumbnail(preview_size, Image.Resampling.LANCZOS)
        mask_image.thumbnail(preview_size, Image.Resampling.NEAREST)
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
        width = simpledialog.askinteger("New document", "Width px:", initialvalue=1280, minvalue=1, maxvalue=50000)
        if not width:
            return
        height = simpledialog.askinteger("New document", "Height px:", initialvalue=900, minvalue=1, maxvalue=50000)
        if not height:
            return
        color = colorchooser.askcolor(title="Background color", initialcolor="#ffffff")[0] or (255, 255, 255)
        self.doc = Document.new(width, height, tuple(map(int, color)) + (255,))
        self.history.clear()
        self.selection_box = None
        self.refresh()

    def new_from_preset(self) -> None:
        presets = {
            "photo": (1800, 1200, 300, (255, 255, 255, 255)),
            "web": (1920, 1080, 72, (255, 255, 255, 255)),
            "4k": (3840, 2160, 72, (0, 0, 0, 255)),
            "mobile_story": (1080, 1920, 72, (255, 255, 255, 255)),
            "icon": (1024, 1024, 72, (0, 0, 0, 0)),
            "print_a4": (2480, 3508, 300, (255, 255, 255, 255)),
        }
        names = ", ".join(presets)
        name = simpledialog.askstring("New preset", f"Preset name:\n{names}", initialvalue="web")
        if not name:
            return
        key = name.strip().lower()
        if key not in presets:
            messagebox.showerror("New preset", "Unknown preset.")
            return
        width, height, dpi, background = presets[key]
        self.doc = Document.new(width, height, background)
        self.doc.dpi = dpi
        self.doc.metadata = {"source": "preset", "preset": key}
        self.history.clear()
        self.selection_box = None
        self.refresh()

    def open_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Supported", "*.prdx *.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"), ("All", "*.*")])
        if not path:
            return
        self.open_path(path)

    def open_path(self, path: str) -> None:
        if not Path(path).exists():
            messagebox.showerror("Open", f"File not found:\n{path}")
            self.recent_files = [item for item in self.recent_files if item.lower() != path.lower()]
            self.refresh_recent_menu()
            return
        self.doc = Document.open_project(path) if path.lower().endswith(".prdx") else Document.from_image(path)
        self.history.clear()
        self.selection_box = self.doc.selection_bounds()
        self.add_recent_file(path)
        self.refresh()

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

        def worker():
            snapshot.save_project(path)
            return path

        def done(saved_path):
            self.doc.path = saved_path
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
        color = colorchooser.askcolor(title="Foreground")[0]
        if color:
            self.foreground = tuple(map(int, color)) + (255,)

    def pick_background(self) -> None:
        color = colorchooser.askcolor(title="Background")[0]
        if color:
            self.background = tuple(map(int, color)) + (255,)

    def pick_color_from_document(self, point: tuple[int, int]) -> None:
        x, y = point
        if x < 0 or y < 0 or x >= self.doc.width or y >= self.doc.height:
            return
        rgba = self.doc.composite(False)[y, x]
        self.foreground = tuple(int(v) for v in rgba)
        self.status_text(f"Picked RGBA: {self.foreground}")

    def show_image_statistics(self) -> None:
        stats = image_statistics(self.doc.composite(False))
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
        stats = image_statistics(self.doc.composite(False))
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
        window.geometry("560x470")
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

        families = sorted(set(tkfont.families()))
        ttk.Label(frame, text="Font").grid(row=1, column=0, sticky=tk.W)
        ttk.Combobox(frame, textvariable=font_var, values=families, state="normal").grid(row=1, column=1, columnspan=3, sticky="ew", pady=2)
        ttk.Label(frame, text="Size").grid(row=2, column=0, sticky=tk.W)
        ttk.Spinbox(frame, from_=4, to=500, textvariable=size_var, width=8).grid(row=2, column=1, sticky="ew", pady=2)
        ttk.Label(frame, text="Box width").grid(row=2, column=2, sticky=tk.W, padx=(10, 0))
        ttk.Spinbox(frame, from_=0, to=100000, textvariable=width_var, width=10).grid(row=2, column=3, sticky="ew", pady=2)
        ttk.Label(frame, text="Align").grid(row=3, column=0, sticky=tk.W)
        ttk.Combobox(frame, textvariable=align_var, values=["left", "center", "right"], state="readonly").grid(row=3, column=1, sticky="ew", pady=2)
        ttk.Label(frame, text="Line spacing").grid(row=3, column=2, sticky=tk.W, padx=(10, 0))
        ttk.Spinbox(frame, from_=0, to=500, textvariable=spacing_var, width=10).grid(row=3, column=3, sticky="ew", pady=2)
        ttk.Label(frame, text="Tracking").grid(row=4, column=0, sticky=tk.W)
        ttk.Spinbox(frame, from_=-50, to=500, textvariable=tracking_var, width=8).grid(row=4, column=1, sticky="ew", pady=2)

        buttons = ttk.Frame(frame)
        buttons.grid(row=5, column=0, columnspan=4, sticky="e", pady=(12, 0))

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
            }
            window.destroy()

        ttk.Button(buttons, text="OK", command=accept).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side=tk.RIGHT)
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
        self.run_document_command("New layer", lambda: self.doc.add_layer(f"Layer {len(self.doc.layers) + 1}"))
        self.refresh()

    def duplicate_layer(self) -> None:
        self.run_document_command("Duplicate layer", self.doc.duplicate_active_layer)
        self.refresh()

    def delete_layer(self) -> None:
        self.run_document_command("Delete layer", self.doc.delete_active_layer)
        self.refresh()

    def rename_layer(self) -> None:
        name = simpledialog.askstring("Rename layer", "Name:", initialvalue=self.doc.layer.name)
        if not name:
            return

        def edit():
            self.doc.layer.name = name
            self.doc.dirty = True

        self.run_document_command("Rename layer", edit)
        self.refresh()

    def move_layer(self, delta: int) -> None:
        i = self.doc.active_layer
        j = i + delta
        if 0 <= j < len(self.doc.layers):
            def edit():
                self.doc.layers[i], self.doc.layers[j] = self.doc.layers[j], self.doc.layers[i]
                self.doc.active_layer = j
                self.doc.dirty = True

            self.run_document_command("Layer reorder", edit)
            self.refresh()

    def free_transform_layer(self) -> None:
        layer = self.doc.layer
        if layer.locked:
            self.status_text("Слой заблокирован")
            return
        initial = f"{layer.x},{layer.y},{layer.pixels.shape[1]},{layer.pixels.shape[0]},0,false,false"
        raw = simpledialog.askstring("Free transform", "x,y,width,height,rotation,flipH,flipV:", initialvalue=initial)
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
            messagebox.showerror("Free transform", "Use: x,y,width,height,rotation,flipH,flipV")
            return
        self.run_document_command("Free transform", lambda: self.doc.transform_active_layer(x, y, width, height, angle, flip_h, flip_v))
        self.refresh()

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
        raw = simpledialog.askstring(
            "Деформация слоя",
            "mode,amount,wavelength: arc, arc_vertical, bulge, pinch, wave_x, wave_y",
            initialvalue="arc,0.35,96",
        )
        if not raw:
            return
        try:
            parts = [part.strip() for part in raw.split(",")]
            if len(parts) not in {2, 3}:
                raise ValueError
            mode = parts[0].lower()
            if mode not in {"arc", "arc_vertical", "bulge", "pinch", "wave_x", "wave_y"}:
                raise ValueError
            amount = float(parts[1])
            wavelength = float(parts[2]) if len(parts) == 3 else 96.0
        except ValueError:
            messagebox.showerror("Деформация слоя", "Используйте: mode,amount,wavelength. Пример: arc,0.35,96")
            return
        self.run_document_command("Warp layer", lambda: self.doc.warp_active_layer(mode, amount, wavelength))
        self.refresh()

    def toggle_clipping_mask(self) -> None:
        if self.doc.active_layer <= 0:
            messagebox.showinfo("Clipping mask", "The bottom layer cannot be clipped.")
            return
        self.run_document_command("Toggle clipping mask", self.doc.toggle_active_clipping)
        self.refresh()

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
        self.run_document_command("Layer styles", lambda: self.doc.set_active_layer_effects(effects))
        self.refresh()

    def edit_layer_filters(self) -> None:
        layer = self.doc.layer
        filters = self.layer_filters_dialog(layer.filters, layer.pixels, self.doc.layer_selection_mask(layer))
        if filters is None:
            return
        self.run_document_command("Layer filters", lambda: self.doc.set_active_layer_filters(filters))
        self.refresh()

    def clear_layer_filters(self) -> None:
        self.run_document_command("Clear layer filters", self.doc.clear_active_layer_filters)
        self.refresh()

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

        def update_preview() -> None:
            source = rgba_array_to_pil(pixels)
            source.thumbnail((180, 180), Image.Resampling.LANCZOS)
            thumb = np.array(source.convert("RGBA"), dtype=np.uint8)
            shown = apply_filter_stack(thumb, filters) if filters else thumb
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
        self.run_document_command("Toggle mask", self.doc.toggle_active_mask)
        self.refresh()

    def toggle_layer_mask_link(self) -> None:
        if self.doc.layer.mask is None:
            messagebox.showinfo("Маска слоя", "У активного слоя нет маски.")
            return
        self.run_document_command("Toggle mask link", self.doc.toggle_active_mask_link)
        self.refresh()

    def set_mask_density(self) -> None:
        layer = self.doc.layer
        if layer.mask is None:
            messagebox.showinfo("Mask density", "Active layer has no mask.")
            return
        value = simpledialog.askfloat("Mask density", "Density 0..1:", initialvalue=float(layer.mask_density), minvalue=0.0, maxvalue=1.0)
        if value is not None:
            self.run_document_command("Mask density", lambda: self.doc.set_active_mask_density(value))
            self.refresh()

    def set_mask_feather(self) -> None:
        layer = self.doc.layer
        if layer.mask is None:
            messagebox.showinfo("Mask feather", "Active layer has no mask.")
            return
        value = simpledialog.askfloat("Mask feather", "Radius px:", initialvalue=float(layer.mask_feather), minvalue=0.0, maxvalue=500.0)
        if value is not None:
            self.run_document_command("Mask feather", lambda: self.doc.set_active_mask_feather(value))
            self.refresh()

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
            self.doc.active_layer = len(self.doc.layers) - 1 - sel[0]
            self.refresh_layers()

    def begin_layer_opacity_change(self, _event) -> None:
        self._opacity_layer_id = self.doc.layer.id
        self._opacity_before = self.doc.layer.opacity

    def change_layer_opacity(self, _value) -> None:
        self.doc.layer.opacity = float(self.layer_opacity.get())
        self.doc.dirty = True
        self.request_canvas_refresh()

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
        self.refresh()

    def toggle_layer_visible(self) -> None:
        def edit():
            self.doc.layer.visible = not self.doc.layer.visible
            self.doc.dirty = True

        self.run_document_command("Toggle layer visible", edit)
        self.refresh()

    def toggle_layer_lock(self) -> None:
        def edit():
            self.doc.layer.locked = not self.doc.layer.locked
            self.doc.dirty = True

        self.run_document_command("Toggle layer lock", edit)
        self.refresh()

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
            if shape not in {"rectangle", "ellipse", "line", "bezier", "polygon", "star"}:
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
        self.run_document_command(
            "Edit shape layer",
            lambda: self.doc.edit_shape_layer(shape=shape, fill=self.foreground, stroke=self.background, stroke_width=stroke_width, sides=sides, inner_ratio=inner_ratio, control_points=new_control_points),
        )
        self.refresh()

    def boolean_shape_layers(self) -> None:
        if self.doc.active_layer <= 0:
            messagebox.showinfo("Булева операция фигур", "Активная фигура должна быть над другой фигурой.")
            return
        upper = self.doc.layer
        lower = self.doc.layers[self.doc.active_layer - 1]
        if upper.kind != "shape" or lower.kind != "shape" or upper.shape_data is None or lower.shape_data is None:
            messagebox.showinfo("Булева операция фигур", "Выберите фигуру, расположенную прямо над другой фигурой.")
            return
        raw = simpledialog.askstring("Булева операция фигур", "mode: union, subtract, intersect, xor", initialvalue="union")
        if raw is None:
            return
        mode = raw.strip().lower()
        if mode not in {"union", "subtract", "intersect", "xor"}:
            messagebox.showerror("Булева операция фигур", "Используйте: union, subtract, intersect или xor.")
            return
        self.run_document_command("Boolean shape operation", lambda: self.doc.boolean_active_shape_with_lower(mode))
        self.refresh()

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

        self.run_document_command("Rotate", edit)
        self.refresh()

    def flip(self, horizontal: bool) -> None:
        def edit():
            code = 1 if horizontal else 0
            for layer in self.doc.layers:
                layer.pixels = cv2.flip(layer.pixels, code)
                if layer.mask is not None:
                    layer.mask = cv2.flip(layer.mask, code)
                if horizontal:
                    layer.x = self.doc.width - (layer.x + layer.pixels.shape[1])
                else:
                    layer.y = self.doc.height - (layer.y + layer.pixels.shape[0])
            if self.doc.selection_mask is not None:
                self.doc.selection_mask = cv2.flip(self.doc.selection_mask, code)
            for name, mask in list(self.doc.saved_selections.items()):
                self.doc.saved_selections[name] = cv2.flip(mask, code)
            self.doc.dirty = True

        self.run_document_command("Flip", edit)
        self.refresh()

    def apply_to_layer(self, label: str, fn) -> None:
        layer = self.doc.layer
        if layer.locked:
            self.status_text("Слой заблокирован")
            return
        layer_id = layer.id
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
            self.doc.dirty = True
            self.push_command(PixelPatchCommand(label, layer_id, rect, before, after.copy()))
            self.invalidate_pixels()
            self.refresh()

        self.run_background(label, worker, done)

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

        source = self.doc.composite(checker=False)
        adjustment_type = tk.StringVar(value=str(initial.get("type", "brightness_contrast")))
        values = [tk.DoubleVar(value=0.0), tk.DoubleVar(value=0.0), tk.DoubleVar(value=0.0)]
        labels: list[ttk.Label] = []
        spins: list[ttk.Spinbox] = []
        updating = False

        preview = ttk.Label(dialog)
        preview.grid(row=0, column=0, rowspan=8, padx=12, pady=12, sticky="n")
        ttk.Label(dialog, text="Пресет").grid(row=0, column=1, sticky="w", padx=(0, 12), pady=(12, 4))
        adjustment_preset = tk.StringVar(value=next(iter(ADJUSTMENT_PRESETS)))
        preset_box = ttk.Combobox(dialog, textvariable=adjustment_preset, values=list(ADJUSTMENT_PRESETS), state="readonly", width=22)
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

        def update_preview(*_args) -> None:
            if updating:
                return
            thumb = rgba_array_to_pil(source)
            thumb.thumbnail((180, 180), Image.Resampling.LANCZOS)
            arr = np.array(thumb.convert("RGBA"), dtype=np.uint8)
            shown = self.apply_adjustment_preview(arr, current_adjustment())
            image = rgba_array_to_pil(shown)
            canvas = Image.new("RGBA", (180, 180), (44, 46, 52, 255))
            canvas.alpha_composite(image, ((180 - image.width) // 2, (180 - image.height) // 2))
            self._adjustment_preview_image = ImageTk.PhotoImage(canvas)
            preview.configure(image=self._adjustment_preview_image)

        def type_changed(_event=None) -> None:
            set_values_for_kind(adjustment_type.get())
            update_preview()

        def apply_adjustment_preset() -> None:
            preset = ADJUSTMENT_PRESETS.get(adjustment_preset.get())
            if preset is None:
                return
            kind = str(preset.get("type", "brightness_contrast"))
            if kind not in ADJUSTMENT_TYPES:
                return
            adjustment_type.set(kind)
            set_values_for_kind(kind, preset)
            update_preview()

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
            messagebox.showinfo("Content-aware fill", "Create a selection on the active layer first.")
            return
        radius = simpledialog.askinteger("Content-aware fill", "Search radius:", initialvalue=3, minvalue=1, maxvalue=30)
        if radius:
            self.apply_to_layer("content-aware fill", lambda arr: content_aware_fill(arr, selection_mask, radius))

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
        strength = simpledialog.askfloat("Red-eye reduction", "Strength 0..1:", initialvalue=0.85, minvalue=0.0, maxvalue=1.0)
        if strength is not None:
            self.apply_to_layer("red-eye reduction", lambda arr: reduce_red_eye(arr, selection_mask, strength))

    def filter_patch_selection(self) -> None:
        if self.doc.selection_mask is None:
            messagebox.showinfo("Patch selection", "Create a selection first.")
            return
        x = simpledialog.askinteger("Patch selection", "Source X top-left:", initialvalue=0, minvalue=-100000, maxvalue=100000)
        if x is None:
            return
        y = simpledialog.askinteger("Patch selection", "Source Y top-left:", initialvalue=0, minvalue=-100000, maxvalue=100000)
        if y is None:
            return
        heal = messagebox.askyesno("Patch selection", "Blend source color into the destination?")
        self.run_document_command("Patch selection", lambda: self.doc.patch_active_selection(x, y, heal))
        self.refresh()

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
        bbox = self.canvas.bbox("all")
        if bbox is None:
            return
        target_x, target_y = self.doc_to_canvas(doc_x, doc_y)
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        scroll_w = max(1, bbox[2] - bbox[0])
        scroll_h = max(1, bbox[3] - bbox[1])
        self.canvas.xview_moveto(max(0.0, min(1.0, (target_x - width / 2 - bbox[0]) / scroll_w)))
        self.canvas.yview_moveto(max(0.0, min(1.0, (target_y - height / 2 - bbox[1]) / scroll_h)))

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
            exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
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


def main() -> None:
    app = PhotoRedactorApp()
    app.mainloop()
