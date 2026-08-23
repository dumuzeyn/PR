from __future__ import annotations

from ..app_shared import *
from ..brush_config import BRUSH_ADVANCED_DEFAULTS
from ..brush_dynamics import CONTROL_SOURCES
from ..brush_tip import BRUSH_TIP_CACHE


class BrushSettingsMixin:
    def open_brush_engine_panel(self) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("Движок кисти")
        dialog.geometry("780x660")
        dialog.minsize(700, 570)
        dialog.transient(self)
        dialog.grab_set()
        variables = self._brush_dialog_variables()

        header = ttk.Frame(dialog, padding=(12, 10))
        header.pack(fill=tk.X)
        ttk.Label(header, text="Движок кисти", style="PanelTitle.TLabel").pack(side=tk.LEFT)
        ttk.Label(header, text="Форма и поведение текущего пресета", style="Secondary.TLabel").pack(side=tk.LEFT, padx=10)
        notebook = ttk.Notebook(dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=12)

        tip = self._brush_tab(notebook, "Форма отпечатка")
        self._dialog_scale(tip, "Угол", variables["angle"], -180, 180, "°")
        self._dialog_scale(tip, "Округлость", variables["roundness"], 0.01, 1.0, "%", percent=True)
        self._dialog_check(tip, "Отразить по горизонтали", variables["flip_x"])
        self._dialog_check(tip, "Отразить по вертикали", variables["flip_y"])
        self._dialog_file(tip, "Пользовательский отпечаток", variables["custom_tip_path"], dialog)
        self._dialog_file(tip, "Вторая кисть", variables["dual_tip_path"], dialog)
        preview = tk.Canvas(tip, width=180, height=180, background="#f5f5f5", highlightthickness=1, highlightbackground=TOKENS.BORDER)
        preview.pack(pady=12)

        shape = self._brush_tab(notebook, "Динамика формы")
        self._dialog_scale(shape, "Колебание размера", variables["size_jitter"], 0, 1, "%", percent=True)
        self._dialog_scale(shape, "Минимальный диаметр", variables["minimum_diameter"], 0.01, 1, "%", percent=True)
        self._dialog_combo(shape, "Управление размером", variables["size_control"], CONTROL_SOURCES)
        self._dialog_scale(shape, "Колебание угла", variables["angle_jitter"], 0, 1, "%", percent=True)
        self._dialog_combo(shape, "Управление углом", variables["angle_control"], CONTROL_SOURCES)
        self._dialog_scale(shape, "Колебание округлости", variables["roundness_jitter"], 0, 1, "%", percent=True)
        self._dialog_scale(shape, "Минимальная округлость", variables["minimum_roundness"], 0.01, 1, "%", percent=True)
        self._dialog_combo(shape, "Управление округлостью", variables["roundness_control"], CONTROL_SOURCES)

        scatter = self._brush_tab(notebook, "Рассеивание")
        self._dialog_scale(scatter, "Рассеивание", variables["scatter"], 0, 10, "×")
        self._dialog_check(scatter, "По обеим осям", variables["scatter_both_axes"])
        self._dialog_scale(scatter, "Количество", variables["scatter_count"], 1, 16, integer=True)
        self._dialog_scale(scatter, "Колебание количества", variables["count_jitter"], 0, 1, "%", percent=True)

        transfer = self._brush_tab(notebook, "Передача и цвет")
        for label, key in (
            ("Колебание непрозрачности", "opacity_jitter"), ("Минимальная непрозрачность", "minimum_opacity"),
            ("Колебание потока", "flow_jitter"), ("Минимальный поток", "minimum_flow"),
            ("Основной / дополнительный", "foreground_background_jitter"), ("Колебание оттенка", "hue_jitter"),
            ("Колебание насыщенности", "saturation_jitter"), ("Колебание яркости", "brightness_jitter"),
        ):
            self._dialog_scale(transfer, label, variables[key], 0, 1, "%", percent=True)

        texture = self._brush_tab(notebook, "Текстура")
        self._dialog_file(texture, "Изображение текстуры", variables["texture_path"], dialog)
        self._dialog_scale(texture, "Масштаб", variables["texture_scale"], 0.05, 4.0, "×")
        self._dialog_scale(texture, "Глубина", variables["texture_depth"], 0, 1, "%", percent=True)
        self._dialog_check(texture, "Инвертировать", variables["texture_invert"])
        self._dialog_check(texture, "Закрепить относительно холста", variables["texture_canvas_space"])

        stabilize = self._brush_tab(notebook, "Стабилизация")
        self._dialog_combo(stabilize, "Режим", variables["smoothing_mode"], ("basic", "stabilizer", "pulled_string"),
                           labels=("Базовое сглаживание", "Стабилизатор", "Натянутая нить"))
        self._dialog_scale(stabilize, "Сила стабилизатора", variables["stabilizer_strength"], 0, 1, "%", percent=True)
        self._dialog_scale(stabilize, "Окно стабилизатора", variables["stabilizer_window"], 2, 64, integer=True)
        self._dialog_scale(stabilize, "Радиус натянутой нити", variables["pulled_string_radius"], 0, 300, "px", integer=True)
        self._dialog_scale(stabilize, "Seed", variables["random_seed"], 0, 99999, integer=True)

        def update_preview(*_args) -> None:
            mask = BRUSH_TIP_CACHE.stamp(
                70, float(self.hardness.get()), float(variables["angle"].get()), float(variables["roundness"].get()),
                bool(variables["flip_x"].get()), bool(variables["flip_y"].get()),
                str(variables["custom_tip_path"].get()), str(variables["dual_tip_path"].get()),
            )
            image = Image.fromarray(np.rint((1.0 - mask) * 255.0).astype(np.uint8), "L").convert("RGB")
            self._brush_tip_preview = ImageTk.PhotoImage(image)
            preview.delete("all")
            preview.create_image(90, 90, image=self._brush_tip_preview)

        for key in ("angle", "roundness", "flip_x", "flip_y", "custom_tip_path", "dual_tip_path"):
            variables[key].trace_add("write", update_preview)
        update_preview()

        footer = ttk.Frame(dialog, padding=12)
        footer.pack(fill=tk.X)

        def apply() -> None:
            for key, variable in variables.items():
                self.brush_advanced[key] = variable.get()
            self.save_active_tool_settings()
            self.save_settings()
            dialog.destroy()
            self.status_text("Настройки движка кисти применены")

        def reset() -> None:
            for key, value in BRUSH_ADVANCED_DEFAULTS.items():
                variables[key].set(value)

        ttk.Button(footer, text="Применить", command=apply, style="Primary.TButton").pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(footer, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT)
        ttk.Button(footer, text="По умолчанию", command=reset).pack(side=tk.LEFT)

    def _brush_dialog_variables(self) -> dict[str, tk.Variable]:
        result: dict[str, tk.Variable] = {}
        for key, default in BRUSH_ADVANCED_DEFAULTS.items():
            value = self.brush_advanced.get(key, default)
            kind = tk.BooleanVar if isinstance(default, bool) else tk.IntVar if isinstance(default, int) else tk.DoubleVar if isinstance(default, float) else tk.StringVar
            result[key] = kind(value=value)
        return result

    @staticmethod
    def _brush_tab(notebook: ttk.Notebook, title: str) -> ttk.Frame:
        frame = ttk.Frame(notebook, padding=12)
        notebook.add(frame, text=title)
        return frame

    @staticmethod
    def _dialog_check(parent: ttk.Frame, label: str, variable: tk.Variable) -> None:
        ttk.Checkbutton(parent, text=label, variable=variable).pack(anchor=tk.W, pady=4)

    @staticmethod
    def _dialog_combo(parent: ttk.Frame, label: str, variable: tk.Variable, values, labels=None) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text=label).pack(side=tk.LEFT)
        display = list(labels or values)
        box = ttk.Combobox(row, values=display, state="readonly", width=24)
        current = list(values).index(variable.get()) if variable.get() in values else 0
        box.current(current)
        box.pack(side=tk.RIGHT)
        box.bind("<<ComboboxSelected>>", lambda _event: variable.set(list(values)[box.current()]))

    @staticmethod
    def _dialog_scale(parent: ttk.Frame, label: str, variable: tk.Variable, start: float, end: float, unit: str = "", *, percent: bool = False, integer: bool = False) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text=label, width=30).pack(side=tk.LEFT)
        value = ttk.Label(row, width=9, anchor=tk.E)
        value.pack(side=tk.RIGHT)
        def show(raw=None) -> None:
            number = float(variable.get() if raw is None else raw)
            if integer and raw is not None:
                variable.set(round(number))
                number = float(variable.get())
            text = f"{round(number * 100)}%" if percent else f"{round(number)} {unit}" if integer else f"{number:.2f} {unit}"
            value.configure(text=text.strip())
        scale = AccentScale(row, from_=start, to=end, command=show)
        scale.set(float(variable.get()))
        scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=8)
        if not integer:
            scale.configure(variable=variable)
        show()

    @staticmethod
    def _dialog_file(parent: ttk.Frame, label: str, variable: tk.StringVar, owner: tk.Toplevel) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text=label, width=25).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=variable).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        def browse() -> None:
            path = filedialog.askopenfilename(parent=owner, filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("Все файлы", "*.*")])
            if path:
                variable.set(path)
        ttk.Button(row, text="Обзор...", command=browse).pack(side=tk.RIGHT)


__all__ = ["BrushSettingsMixin"]
