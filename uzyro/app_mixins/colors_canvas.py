from __future__ import annotations

from ..app_shared import *
from ..brush_config import default_brush_config


class ColorsCanvasMixin:
    def _build_color_control(self, parent: ttk.Frame) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=(3, 7))
        self.color_control_canvas = tk.Canvas(frame, width=58, height=44, highlightthickness=0, background=TOKENS.SURFACE)
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
        canvas.create_text(49, 7, text="⇄", fill=TOKENS.TEXT_SECONDARY, font=("Segoe UI Symbol", 10), tags="swap")
        canvas.create_rectangle(3, 32, 11, 40, fill="#000000", outline=TOKENS.TEXT_SECONDARY, width=1, tags="reset")
        canvas.create_rectangle(9, 28, 17, 36, fill="#ffffff", outline=TOKENS.TEXT_SECONDARY, width=1, tags="reset")

    def color_control_click(self, event) -> None:
        if event.x >= 39 and event.y <= 15:
            self.swap_colors()
        elif event.x <= 18 and event.y >= 27:
            self.reset_colors()
        elif event.x >= 19 and event.y >= 14:
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
        if "flow" in settings:
            settings["flow"] = float(self.brush_flow.get())
        if "spacing" in settings:
            settings["spacing"] = float(self.brush_spacing.get())
        if "smoothing" in settings:
            settings["smoothing"] = float(self.brush_smoothing.get())
        if "blend_mode" in settings:
            settings["blend_mode"] = self.brush_blend_mode.get()
        for key, variable in (("pressure_size", self.pressure_size), ("pressure_opacity", self.pressure_opacity), ("pressure_flow", self.pressure_flow)):
            if key in settings:
                settings[key] = bool(variable.get())
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
        if "flow" in settings:
            self.brush_flow.set(float(settings["flow"]))
        if "spacing" in settings:
            self.brush_spacing.set(float(settings["spacing"]))
        if "smoothing" in settings:
            self.brush_smoothing.set(float(settings["smoothing"]))
        if "blend_mode" in settings:
            self.brush_blend_mode.set(str(settings["blend_mode"]))
        for key, variable in (("pressure_size", self.pressure_size), ("pressure_opacity", self.pressure_opacity), ("pressure_flow", self.pressure_flow)):
            if key in settings:
                variable.set(bool(settings[key]))
        if "strength" in settings:
            self.retouch_strength.set(float(settings["strength"]))
        if "exposure" in settings:
            self.exposure.set(float(settings["exposure"]))
        if "range" in settings:
            self.tonal_range.set(str(settings["range"]))

    def apply_brush_preset(self) -> None:
        values = self.brush_presets.get(self.brush_preset.get())
        if not isinstance(values, dict):
            return
        mapping = {
            "size": self.brush_size,
            "hardness": self.hardness,
            "opacity": self.opacity,
            "flow": self.brush_flow,
            "spacing": self.brush_spacing,
            "smoothing": self.brush_smoothing,
            "blend_mode": self.brush_blend_mode,
            "pressure_size": self.pressure_size,
            "pressure_opacity": self.pressure_opacity,
            "pressure_flow": self.pressure_flow,
        }
        for key, variable in mapping.items():
            if key in values:
                variable.set(values[key])
        self.brush_advanced.clear()
        self.brush_advanced.update(default_brush_config())
        advanced = values.get("advanced")
        if isinstance(advanced, dict):
            self.brush_advanced.update({key: advanced[key] for key in self.brush_advanced if key in advanced})
        self.save_active_tool_settings()
        if hasattr(self, "tool_options_panel"):
            self.tool_options_panel.render()
        self.status_text(f"Пресет кисти: {self.brush_preset.get()}")

    def import_custom_brush(self) -> None:
        path = filedialog.askopenfilename(
            parent=self,
            title="Добавить свою кисть",
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("Все файлы", "*.*")],
        )
        if not path:
            return
        try:
            with Image.open(path) as source:
                source.verify()
        except (OSError, ValueError):
            messagebox.showerror("Кисть", "Не удалось прочитать изображение кисти.", parent=self)
            return
        base_name = Path(path).stem.strip() or "Своя кисть"
        name = base_name
        suffix = 2
        while name in self.brush_presets:
            name = f"{base_name} {suffix}"
            suffix += 1
        self.brush_advanced.clear()
        self.brush_advanced.update(default_brush_config())
        self.brush_advanced["custom_tip_path"] = str(Path(path).resolve())
        self.brush_preset_name.set(name)
        self.save_brush_preset()
        self.status_text(f"Добавлена кисть «{name}»")

    def save_brush_preset(self, automatic_name: bool = False) -> None:
        name = self.brush_preset_name.get().strip()
        if automatic_name:
            index = 1
            while f"Мой пресет {index}" in self.brush_presets:
                index += 1
            name = f"Мой пресет {index}"
            self.brush_preset_name.set(name)
        if not name:
            self.status_text("Введите название пресета")
            return
        self.brush_presets[name] = {
            "size": int(self.brush_size.get()),
            "hardness": float(self.hardness.get()),
            "opacity": float(self.opacity.get()),
            "flow": float(self.brush_flow.get()),
            "spacing": float(self.brush_spacing.get()),
            "smoothing": float(self.brush_smoothing.get()),
            "blend_mode": self.brush_blend_mode.get(),
            "pressure_size": bool(self.pressure_size.get()),
            "pressure_opacity": bool(self.pressure_opacity.get()),
            "pressure_flow": bool(self.pressure_flow.get()),
            "advanced": copy.deepcopy(self.brush_advanced),
        }
        self.brush_preset.set(name)
        self.save_settings()
        self.tool_options_panel.render()
        self.status_text(f"Пресет «{name}» сохранён")

    def delete_brush_preset(self) -> None:
        name = self.brush_preset.get()
        if name in BRUSH_PRESET_DEFAULTS:
            self.status_text("Стандартный пресет нельзя удалить")
            return
        if self.brush_presets.pop(name, None) is None:
            return
        self.brush_preset.set(next(iter(self.brush_presets)))
        self.save_settings()
        self.tool_options_panel.render()

    def reset_brush_presets(self) -> None:
        self.brush_presets.clear()
        self.brush_presets.update(copy.deepcopy(BRUSH_PRESET_DEFAULTS))
        self.brush_preset.set(next(iter(self.brush_presets)))
        self.apply_brush_preset()
        self.save_settings()

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
        self.update_path_overlay()

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
        self.canvas.bind("<B3-Motion>", self.patch_selection_drag)
        self.canvas.bind("<ButtonRelease-3>", self.patch_selection_up)
        self.canvas.bind("<Double-Button-1>", self.pointer_double_click)
        self.canvas.bind("<ButtonPress-2>", self.pan_down)
        self.canvas.bind("<B2-Motion>", self.pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self.pan_up)
        self.canvas.bind("<Motion>", self.pointer_motion)
        self.canvas.bind("<Leave>", self.pointer_leave)
        self.canvas.bind("<FocusOut>", self.canvas_focus_lost)
        self.canvas.bind("<MouseWheel>", self.mouse_wheel)

    def selection_right_click(self, event) -> str | None:
        if self.tool.get() == "patch":
            self.patch_selection_down(event)
            return "break"
        selection_tools = {
            "select", "ellipse_select", "lasso", "magnetic_lasso", "polygon_lasso",
            "quick_selection", "magic_wand", "color_range",
        }
        mask = self.doc.selection_mask
        if self.tool.get() not in selection_tools or mask is None:
            return None
        if self.selection_mode.get() != "subtract":
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
        self.right_tabs = ttk.Notebook(parent, style="Sidebar.TNotebook")
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
        add = ttk.Button(buttons, image=self._panel_icons["add"], width=2, command=self.new_layer, style="PanelIcon.TButton")
        delete = ttk.Button(buttons, image=self._panel_icons["delete"], width=2, command=self.delete_layer, style="PanelIcon.TButton")
        duplicate = ttk.Button(buttons, image=self._panel_icons["duplicate"], width=2, command=self.duplicate_layer, style="PanelIcon.TButton")
        up = ttk.Button(buttons, image=self._panel_icons["up"], width=2, command=lambda: self.move_layer(1), style="PanelIcon.TButton")
        down = ttk.Button(buttons, image=self._panel_icons["down"], width=2, command=lambda: self.move_layer(-1), style="PanelIcon.TButton")
        for button in (add, delete, duplicate, up, down):
            button.pack(side=tk.LEFT, padx=(0, 3))
        solo = ttk.Button(buttons, text="Только", command=self.toggle_layer_solo)
        solo.pack(side=tk.RIGHT)
        ToolTip(add, "Новый слой")
        ToolTip(delete, "Удалить выбранные слои")
        ToolTip(duplicate, "Дублировать слой")
        ToolTip(up, "Поднять слой")
        ToolTip(down, "Опустить слой")
        ToolTip(solo, "Показать только активный слой; повторный клик вернёт прежнюю видимость")

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
