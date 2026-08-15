from __future__ import annotations

from ..app_shared import *


GRADIENT_PRESETS: dict[str, dict[str, object]] = {
    "Передний → фон": {
        "stops": [],
        "opacity_stops": [{"position": 0.0, "opacity": 1.0}, {"position": 1.0, "opacity": 1.0}],
    },
    "Чёрный → белый": {
        "stops": [{"position": 0.0, "color": [0, 0, 0, 255]}, {"position": 1.0, "color": [255, 255, 255, 255]}],
        "opacity_stops": [{"position": 0.0, "opacity": 1.0}, {"position": 1.0, "opacity": 1.0}],
    },
    "Прозрачный → цвет": {
        "stops": [],
        "opacity_stops": [{"position": 0.0, "opacity": 0.0}, {"position": 1.0, "opacity": 1.0}],
    },
    "Закат": {
        "stops": [
            {"position": 0.0, "color": [35, 55, 145, 255], "midpoint": 0.45},
            {"position": 0.52, "color": [235, 75, 80, 255], "midpoint": 0.55},
            {"position": 1.0, "color": [255, 200, 75, 255]},
        ],
        "opacity_stops": [{"position": 0.0, "opacity": 1.0}, {"position": 1.0, "opacity": 1.0}],
    },
}


class GradientEditorMixin:
    def open_gradient_editor(self) -> None:
        self.gradient_editor_dialog(self.current_gradient_definition())

    def gradient_editor_dialog(self, initial: dict[str, object], on_apply=None) -> None:
        working = copy.deepcopy(initial)
        if not working.get("stops"):
            working["stops"] = [
                {"position": 0.0, "color": list(self.foreground), "midpoint": 0.5},
                {"position": 1.0, "color": list(self.background)},
            ]
        working.setdefault("opacity_stops", [{"position": 0.0, "opacity": 1.0}, {"position": 1.0, "opacity": 1.0}])
        working.setdefault("reverse", False)
        working.setdefault("dither", False)
        working.setdefault("transparency", True)
        working.setdefault("interpolation_space", "srgb")
        working.setdefault("noise", {"enabled": False, "roughness": 0.5, "color_model": "rgb", "seed": 0, "restrict_colors": False})
        noise_options = working["noise"]
        if not isinstance(noise_options, dict):
            noise_options = {"enabled": False}
            working["noise"] = noise_options
        noise_options.setdefault("roughness", 0.5)
        noise_options.setdefault("color_model", "rgb")
        noise_options.setdefault("seed", 0)
        noise_options.setdefault("restrict_colors", False)
        noise_options.setdefault("channels", [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        dialog = tk.Toplevel(self)
        dialog.title("Редактор градиента")
        dialog.transient(self)
        dialog.grab_set()
        dialog.minsize(760, 560)

        preset = tk.StringVar(value="Передний → фон")
        selected_kind = tk.StringVar(value="color")
        selected_index = tk.IntVar(value=0)
        position = tk.DoubleVar(value=0.0)
        midpoint = tk.DoubleVar(value=0.5)
        opacity = tk.DoubleVar(value=1.0)
        reverse = tk.BooleanVar(value=bool(working["reverse"]))
        dither = tk.BooleanVar(value=bool(working["dither"]))
        transparency = tk.BooleanVar(value=bool(working["transparency"]))
        generation_mode = tk.StringVar(value="Шумовой" if bool(noise_options.get("enabled", False)) else "Обычный")
        interpolation = tk.StringVar(value={"srgb": "sRGB", "linear_rgb": "Linear RGB", "oklab": "OKLab"}.get(str(working["interpolation_space"]), "sRGB"))
        noise_model = tk.StringVar(value={"rgb": "RGB", "hsv": "HSV", "grayscale": "Оттенки серого"}.get(str(noise_options["color_model"]), "RGB"))
        roughness = tk.DoubleVar(value=round(float(noise_options["roughness"]) * 100.0))
        noise_seed = tk.IntVar(value=int(noise_options["seed"]))
        restrict_colors = tk.BooleanVar(value=bool(noise_options["restrict_colors"]))
        channel_values = noise_options.get("channels", [[0.0, 1.0]] * 3)
        channel_vars = [
            (
                tk.DoubleVar(value=round(float(channel_values[index][0]) * 100.0)),
                tk.DoubleVar(value=round(float(channel_values[index][1]) * 100.0)),
            )
            for index in range(3)
        ]
        dragging: list[tuple[str, int] | None] = [None]
        syncing = [False]

        header = ttk.Frame(dialog, padding=(12, 10, 12, 4))
        header.pack(fill=tk.X)
        ttk.Label(header, text="Пресет").pack(side=tk.LEFT)
        preset_box = ttk.Combobox(header, textvariable=preset, values=list(GRADIENT_PRESETS), state="readonly", width=24)
        preset_box.pack(side=tk.LEFT, padx=(8, 4))
        ttk.Button(header, text="Применить пресет", command=lambda: apply_preset()).pack(side=tk.LEFT)
        ttk.Checkbutton(header, text="Обратить", variable=reverse).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Checkbutton(header, text="Дизеринг", variable=dither).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Checkbutton(header, text="Прозрачность", variable=transparency).pack(side=tk.RIGHT)

        mode_bar = ttk.Frame(dialog, padding=(12, 2, 12, 4))
        mode_bar.pack(fill=tk.X)
        ttk.Label(mode_bar, text="Режим").pack(side=tk.LEFT)
        ttk.Combobox(mode_bar, textvariable=generation_mode, values=("Обычный", "Шумовой"), state="readonly", width=14).pack(side=tk.LEFT, padx=(8, 18))
        standard_settings = ttk.Frame(mode_bar)
        ttk.Label(standard_settings, text="Интерполяция").pack(side=tk.LEFT)
        ttk.Combobox(
            standard_settings, textvariable=interpolation,
            values=("sRGB", "Linear RGB", "OKLab"), state="readonly", width=14,
        ).pack(side=tk.LEFT, padx=(8, 0))
        noise_settings = ttk.Frame(mode_bar)
        ttk.Label(noise_settings, text="Цветовая модель").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            noise_settings, textvariable=noise_model,
            values=("RGB", "HSV", "Оттенки серого"), state="readonly", width=16,
        ).grid(row=0, column=1, padx=(6, 14))
        ttk.Label(noise_settings, text="Шероховатость, %").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(noise_settings, textvariable=roughness, from_=0, to=100, increment=1, width=6).grid(row=0, column=3, padx=(6, 14))
        ttk.Label(noise_settings, text="Seed").grid(row=0, column=4, sticky="w")
        ttk.Spinbox(noise_settings, textvariable=noise_seed, from_=0, to=2147483647, increment=1, width=10).grid(row=0, column=5, padx=(6, 8))

        channel_bar = ttk.Frame(dialog, padding=(12, 0, 12, 4))
        ttk.Label(channel_bar, text="Диапазоны каналов:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        channel_rows = []
        for index, (low, high) in enumerate(channel_vars):
            label = ttk.Label(channel_bar, width=3)
            low_box = ttk.Spinbox(channel_bar, textvariable=low, from_=0, to=100, increment=1, width=5)
            separator = ttk.Label(channel_bar, text="-")
            high_box = ttk.Spinbox(channel_bar, textvariable=high, from_=0, to=100, increment=1, width=5)
            column = 1 + index * 4
            label.grid(row=0, column=column)
            low_box.grid(row=0, column=column + 1)
            separator.grid(row=0, column=column + 2, padx=3)
            high_box.grid(row=0, column=column + 3, padx=(0, 12))
            channel_rows.append((label, low_box, separator, high_box))
        ttk.Checkbutton(channel_bar, text="Ограничить цвета", variable=restrict_colors).grid(row=0, column=13, padx=(4, 0))

        ramp = tk.Canvas(dialog, height=145, background=TOKENS.SURFACE, highlightthickness=1, highlightbackground=TOKENS.BORDER, cursor="hand2")
        ramp.pack(fill=tk.X, padx=12, pady=(6, 8))
        body = ttk.PanedWindow(dialog, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=12)
        color_frame = ttk.LabelFrame(body, text="Цветовые точки", padding=8)
        opacity_frame = ttk.LabelFrame(body, text="Точки прозрачности", padding=8)
        body.add(color_frame, weight=1)
        body.add(opacity_frame, weight=1)
        color_list = tk.Listbox(color_frame, exportselection=False, height=9)
        color_list.pack(fill=tk.BOTH, expand=True)
        opacity_list = tk.Listbox(opacity_frame, exportselection=False, height=9)
        opacity_list.pack(fill=tk.BOTH, expand=True)

        color_buttons = ttk.Frame(color_frame)
        color_buttons.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(color_buttons, text="Добавить", command=lambda: add_stop("color")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(color_buttons, text="Удалить", command=lambda: delete_stop("color")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        opacity_buttons = ttk.Frame(opacity_frame)
        opacity_buttons.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(opacity_buttons, text="Добавить", command=lambda: add_stop("opacity")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(opacity_buttons, text="Удалить", command=lambda: delete_stop("opacity")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        editor = ttk.LabelFrame(dialog, text="Выбранная точка", padding=8)
        editor.pack(fill=tk.X, padx=12, pady=8)
        ttk.Label(editor, text="Позиция").grid(row=0, column=0, sticky="w")
        position_spin = ttk.Spinbox(editor, textvariable=position, from_=0.0, to=1.0, increment=0.01, width=9)
        position_spin.grid(row=0, column=1, padx=(6, 18))
        ttk.Label(editor, text="Средняя точка").grid(row=0, column=2, sticky="w")
        midpoint_spin = ttk.Spinbox(editor, textvariable=midpoint, from_=0.01, to=0.99, increment=0.01, width=9)
        midpoint_spin.grid(row=0, column=3, padx=(6, 18))
        color_button = tk.Button(editor, text="Цвет", width=12, cursor="hand2", command=lambda: pick_stop_color())
        color_button.grid(row=0, column=4, padx=(0, 18))
        ttk.Label(editor, text="Непрозрачность").grid(row=0, column=5, sticky="w")
        opacity_spin = ttk.Spinbox(editor, textvariable=opacity, from_=0.0, to=1.0, increment=0.05, width=9)
        opacity_spin.grid(row=0, column=6, padx=(6, 0))

        footer = ttk.Frame(dialog, padding=(12, 0, 12, 12))
        footer.pack(fill=tk.X)
        ttk.Label(footer, text="Перетаскивайте маркеры над и под полосой", style="Secondary.TLabel").pack(side=tk.LEFT)

        def stops(kind: str) -> list[dict[str, object]]:
            return working["stops" if kind == "color" else "opacity_stops"]  # type: ignore[return-value]

        def sort_stops(kind: str) -> None:
            stops(kind).sort(key=lambda item: float(item.get("position", 0.0)))

        def marker_x(value: float) -> float:
            return 28 + float(np.clip(value, 0.0, 1.0)) * max(1, ramp.winfo_width() - 56)

        def draw_ramp(*_args) -> None:
            if not ramp.winfo_exists():
                return
            width = max(200, ramp.winfo_width() - 56)
            image = GradientEngine.render(
                width, 70, (0, 0), (width - 1, 0), working["stops"], "linear",
                opacity_stops=working["opacity_stops"], reverse=reverse.get(),
                dither=dither.get(), transparency=transparency.get(),
                interpolation_space={"sRGB": "srgb", "Linear RGB": "linear_rgb", "OKLab": "oklab"}[interpolation.get()],
                noise=current_noise_options(),
            )
            checker = np.zeros((70, width, 4), dtype=np.uint8)
            yy, xx = np.indices((70, width))
            shade = np.where(((xx // 8 + yy // 8) % 2)[:, :, None], 92, 60).astype(np.uint8)
            checker[:, :, :3] = shade
            checker[:, :, 3] = 255
            alpha = image[:, :, 3:4].astype(np.float32) / 255.0
            checker[:, :, :3] = np.clip(image[:, :, :3] * alpha + checker[:, :, :3] * (1.0 - alpha), 0, 255).astype(np.uint8)
            self._gradient_editor_image = ImageTk.PhotoImage(Image.fromarray(checker, "RGBA"))
            ramp.delete("all")
            ramp.create_image(28, 36, image=self._gradient_editor_image, anchor=tk.NW)
            ramp.create_rectangle(28, 36, 28 + width, 106, outline=TOKENS.BORDER)
            for index, stop in enumerate(stops("opacity")):
                x = marker_x(float(stop["position"]))
                ramp.create_polygon(x - 7, 27, x + 7, 27, x, 36, fill="#f3f5f7", outline="#111318", tags=("marker", f"opacity:{index}"))
            for index, stop in enumerate(stops("color")):
                x = marker_x(float(stop["position"]))
                color = self.color_hex(tuple(stop["color"]))
                ramp.create_polygon(x - 7, 115, x + 7, 115, x, 106, fill=color, outline="#111318", tags=("marker", f"color:{index}"))
            refresh_lists()

        def current_noise_options() -> dict[str, object]:
            return {
                "enabled": generation_mode.get() == "Шумовой",
                "roughness": float(np.clip(roughness.get() / 100.0, 0.0, 1.0)),
                "color_model": {"RGB": "rgb", "HSV": "hsv", "Оттенки серого": "grayscale"}[noise_model.get()],
                "seed": int(noise_seed.get()),
                "restrict_colors": bool(restrict_colors.get()),
                "channels": [
                    [float(np.clip(low.get() / 100.0, 0.0, 1.0)), float(np.clip(high.get() / 100.0, 0.0, 1.0))]
                    for low, high in channel_vars
                ],
            }

        def refresh_mode(*_args) -> None:
            standard_settings.pack_forget()
            noise_settings.pack_forget()
            channel_bar.pack_forget()
            if generation_mode.get() == "Шумовой":
                noise_settings.pack(side=tk.LEFT, fill=tk.X, expand=True)
                channel_bar.pack(fill=tk.X, before=ramp)
                names = ("R", "G", "B") if noise_model.get() == "RGB" else (("H", "S", "V") if noise_model.get() == "HSV" else ("Y",))
                for index, row in enumerate(channel_rows):
                    if index < len(names):
                        row[0].configure(text=names[index])
                        for widget in row:
                            widget.grid()
                    else:
                        for widget in row:
                            widget.grid_remove()
            else:
                standard_settings.pack(side=tk.LEFT)
            draw_ramp()

        def refresh_lists() -> None:
            current_kind, current_index = selected_kind.get(), selected_index.get()
            color_list.delete(0, tk.END)
            for item in stops("color"):
                color_list.insert(tk.END, f"{round(float(item['position']) * 100):3d}%   {self.color_hex(tuple(item['color'])).upper()}")
            opacity_list.delete(0, tk.END)
            for item in stops("opacity"):
                opacity_list.insert(tk.END, f"{round(float(item['position']) * 100):3d}%   {round(float(item['opacity']) * 100):3d}%")
            target = color_list if current_kind == "color" else opacity_list
            if target.size():
                index = min(current_index, target.size() - 1)
                target.selection_set(index)

        def select_stop(kind: str, index: int) -> None:
            selected_kind.set(kind)
            selected_index.set(max(0, min(index, len(stops(kind)) - 1)))
            item = stops(kind)[selected_index.get()]
            syncing[0] = True
            position.set(float(item.get("position", 0.0)))
            midpoint.set(float(item.get("midpoint", 0.5)))
            opacity.set(float(item.get("opacity", 1.0)))
            syncing[0] = False
            is_color = kind == "color"
            color_button.configure(state=tk.NORMAL if is_color else tk.DISABLED)
            opacity_spin.configure(state=tk.DISABLED if is_color else tk.NORMAL)
            midpoint_spin.configure(state=tk.NORMAL if selected_index.get() < len(stops(kind)) - 1 else tk.DISABLED)
            if is_color:
                color_button.configure(background=self.color_hex(tuple(item["color"])), activebackground=self.color_hex(tuple(item["color"])))
            refresh_lists()

        def list_select(kind: str, widget: tk.Listbox) -> None:
            selected = widget.curselection()
            if selected:
                select_stop(kind, int(selected[0]))

        def add_stop(kind: str) -> None:
            values = stops(kind)
            if kind == "color":
                rendered = GradientEngine.render(1, 1, (0, 0), (1, 0), working["stops"], "linear", origin=(0.5, 0.0))
                values.append({"position": 0.5, "color": rendered[0, 0].tolist(), "midpoint": 0.5})
            else:
                values.append({"position": 0.5, "opacity": 1.0, "midpoint": 0.5})
            sort_stops(kind)
            select_stop(kind, values.index(next(item for item in values if float(item["position"]) == 0.5)))
            draw_ramp()

        def delete_stop(kind: str) -> None:
            values = stops(kind)
            if len(values) <= 2:
                return
            values.pop(min(selected_index.get(), len(values) - 1))
            select_stop(kind, min(selected_index.get(), len(values) - 1))
            draw_ramp()

        def update_selected(*_args) -> None:
            if syncing[0]:
                return
            kind = selected_kind.get()
            values = stops(kind)
            if not values:
                return
            item = values[min(selected_index.get(), len(values) - 1)]
            item["position"] = float(np.clip(position.get(), 0.0, 1.0))
            item["midpoint"] = float(np.clip(midpoint.get(), 0.01, 0.99))
            if kind == "opacity":
                item["opacity"] = float(np.clip(opacity.get(), 0.0, 1.0))
            sort_stops(kind)
            selected_index.set(values.index(item))
            draw_ramp()

        def pick_stop_color() -> None:
            item = stops("color")[selected_index.get()]
            chosen = colorchooser.askcolor(self.color_hex(tuple(item["color"])), parent=dialog)[0]
            if chosen:
                item["color"] = [round(value) for value in chosen] + [255]
                draw_ramp()
                select_stop("color", selected_index.get())

        def apply_preset() -> None:
            value = copy.deepcopy(GRADIENT_PRESETS[preset.get()])
            if not value.get("stops"):
                value["stops"] = [
                    {"position": 0.0, "color": list(self.foreground), "midpoint": 0.5},
                    {"position": 1.0, "color": list(self.background)},
                ]
            working["stops"] = value["stops"]
            working["opacity_stops"] = value["opacity_stops"]
            select_stop("color", 0)
            draw_ramp()

        def marker_press(event) -> None:
            item = ramp.find_closest(event.x, event.y)
            tags = ramp.gettags(item)
            marker = next((tag for tag in tags if ":" in tag), None)
            if marker:
                kind, raw_index = marker.split(":", 1)
                dragging[0] = (kind, int(raw_index))
                select_stop(kind, int(raw_index))

        def marker_drag(event) -> None:
            if dragging[0] is None:
                return
            kind, index = dragging[0]
            item = stops(kind)[min(index, len(stops(kind)) - 1)]
            item["position"] = float(np.clip((event.x - 28) / max(1, ramp.winfo_width() - 56), 0.0, 1.0))
            sort_stops(kind)
            new_index = stops(kind).index(item)
            dragging[0] = (kind, new_index)
            select_stop(kind, new_index)
            draw_ramp()

        def accept() -> None:
            working["reverse"] = bool(reverse.get())
            working["dither"] = bool(dither.get())
            working["transparency"] = bool(transparency.get())
            working["interpolation_space"] = {"sRGB": "srgb", "Linear RGB": "linear_rgb", "OKLab": "oklab"}[interpolation.get()]
            working["noise"] = current_noise_options()
            if on_apply is None:
                self.gradient_definition = copy.deepcopy(working)
            else:
                on_apply(copy.deepcopy(working))
            dialog.destroy()
            if hasattr(self, "tool_options_panel"):
                self.tool_options_panel.render()

        color_list.bind("<<ListboxSelect>>", lambda _event: list_select("color", color_list))
        opacity_list.bind("<<ListboxSelect>>", lambda _event: list_select("opacity", opacity_list))
        ramp.bind("<ButtonPress-1>", marker_press)
        ramp.bind("<B1-Motion>", marker_drag)
        ramp.bind("<ButtonRelease-1>", lambda _event: dragging.__setitem__(0, None))
        ramp.bind("<Configure>", draw_ramp)
        for variable in (position, midpoint, opacity):
            variable.trace_add("write", update_selected)
        for variable in (reverse, dither, transparency):
            variable.trace_add("write", draw_ramp)
        generation_mode.trace_add("write", refresh_mode)
        noise_model.trace_add("write", refresh_mode)
        for variable in (interpolation, roughness, noise_seed, restrict_colors):
            variable.trace_add("write", draw_ramp)
        for low, high in channel_vars:
            low.trace_add("write", draw_ramp)
            high.trace_add("write", draw_ramp)
        ttk.Button(footer, text="Применить", command=accept, style="Primary.TButton").pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(footer, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT)
        self._gradient_editor_ramp = ramp
        self._gradient_editor_color_list = color_list
        self._gradient_editor_opacity_list = opacity_list
        self._gradient_editor_apply = accept
        self._gradient_editor_mode = generation_mode
        self._gradient_editor_standard_settings = standard_settings
        self._gradient_editor_noise_settings = noise_settings
        self._gradient_editor_channel_bar = channel_bar
        self.center_toplevel(dialog, 820, 620)
        select_stop("color", 0)
        refresh_mode()
