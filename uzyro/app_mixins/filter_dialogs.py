from __future__ import annotations

from ..app_shared import *


class FilterDialogsMixin:
    def toggle_clipping_mask(self) -> None:
        if self.doc.active_layer <= 0:
            messagebox.showinfo("Clipping mask", "The bottom layer cannot be clipped.")
            return
        self.set_layer_property("Toggle clipping mask", "clipping", not self.doc.layer.clipping)

    def edit_layer_styles(self) -> None:
        layer = self.doc.layer
        effects = self.layer_styles_dialog(layer)
        if effects is None:
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

    def add_smart_filter(self, kind: str) -> None:
        item = self.make_filter_item(kind, self.filter_primary_value({"type": kind}))
        self.set_layer_property(f"Smart filter: {FILTER_LABELS.get(kind, kind)}", "filters", [*self.doc.layer.filters, item], preserve_render_cache=False)

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
        preview.grid(row=0, column=0, rowspan=14, padx=12, pady=12, sticky="n")
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
        filter_type = tk.StringVar(value=FILTER_LABELS[FILTER_TYPES[0]])
        type_box = ttk.Combobox(controls, textvariable=filter_type, values=list(FILTER_VALUES), state="readonly", width=14)
        type_box.grid(row=2, column=1, sticky="ew", pady=(0, 6))
        ttk.Label(controls, text="Параметр").grid(row=3, column=0, sticky="w")
        value_var = tk.DoubleVar(value=3.0)
        value_spin = ttk.Spinbox(controls, textvariable=value_var, from_=0.0, to=500.0, increment=1.0, width=12)
        value_spin.grid(row=3, column=1, sticky="ew", pady=(0, 8))
        extra_frame = ttk.Frame(controls)
        extra_frame.grid(row=4, column=0, columnspan=2, sticky="ew")
        extra_frame.columnconfigure(1, weight=1)
        angle_var = tk.DoubleVar(value=0.0)
        radius_var = tk.DoubleVar(value=2.0)
        threshold_var = tk.DoubleVar(value=0.0)
        extra_controls: dict[str, tuple[ttk.Label, ttk.Spinbox]] = {}
        for row, (key, label, variable, low, high, step) in enumerate([
            ("angle", "Угол", angle_var, -180.0, 180.0, 1.0),
            ("radius", "Радиус", radius_var, 0.1, 100.0, 0.1),
            ("threshold", "Порог", threshold_var, 0.0, 255.0, 1.0),
        ]):
            field_label = ttk.Label(extra_frame, text=label)
            field_spin = ttk.Spinbox(extra_frame, textvariable=variable, from_=low, to=high, increment=step, width=12)
            field_label.grid(row=row, column=0, sticky="w")
            field_spin.grid(row=row, column=1, sticky="ew", pady=(0, 6))
            extra_controls[key] = field_label, field_spin
        enabled_var = tk.BooleanVar(value=True)
        enabled_check = ttk.Checkbutton(controls, text="Включен", variable=enabled_var)
        enabled_check.grid(row=5, column=0, columnspan=2, sticky="w", pady=(0, 6))
        ttk.Label(controls, text="Opacity").grid(row=6, column=0, sticky="w")
        opacity_var = tk.DoubleVar(value=100.0)
        opacity_spin = ttk.Spinbox(controls, textvariable=opacity_var, from_=0.0, to=100.0, increment=5.0, width=12)
        opacity_spin.grid(row=6, column=1, sticky="ew", pady=(0, 8))
        ttk.Label(controls, text="Режим").grid(row=7, column=0, sticky="w")
        filter_blend_mode = tk.StringVar(value="Normal")
        filter_blend_box = ttk.Combobox(controls, textvariable=filter_blend_mode, values=BLEND_MODES, state="readonly", width=14)
        filter_blend_box.grid(row=7, column=1, sticky="ew", pady=(0, 8))
        ttk.Label(controls, text="Канал").grid(row=8, column=0, sticky="w")
        filter_channel = tk.StringVar(value="RGB")
        filter_channel_box = ttk.Combobox(controls, textvariable=filter_channel, values=list(CHANNEL_VALUES), state="readonly", width=14)
        filter_channel_box.grid(row=8, column=1, sticky="ew", pady=(0, 8))
        mask_inverted = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="Инвертировать маску", variable=mask_inverted).grid(row=9, column=0, columnspan=2, sticky="w", pady=(0, 6))
        ttk.Label(controls, text="Плотность маски").grid(row=10, column=0, sticky="w")
        mask_density = tk.DoubleVar(value=100.0)
        mask_density_spin = ttk.Spinbox(controls, textvariable=mask_density, from_=0, to=100, increment=5, width=12)
        mask_density_spin.grid(row=10, column=1, sticky="ew", pady=(0, 6))
        ttk.Label(controls, text="Растушёвка").grid(row=11, column=0, sticky="w")
        mask_feather = tk.DoubleVar(value=0.0)
        mask_feather_spin = ttk.Spinbox(controls, textvariable=mask_feather, from_=0, to=500, increment=1, width=12)
        mask_feather_spin.grid(row=11, column=1, sticky="ew", pady=(0, 8))
        hint = ttk.Label(controls, text="", wraplength=190, justify=tk.LEFT)
        hint.grid(row=12, column=0, columnspan=2, sticky="w", pady=(0, 8))

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

        def selected_filter_kind() -> str:
            return FILTER_VALUES.get(filter_type.get(), filter_type.get())

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
            channel = CHANNEL_LABELS.get(str(item.get("channel", "RGB")), "RGB")
            return f"{prefix}{label}: {value:g}, {opacity}%, {channel}{blend}{mask}"

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
            visible_extras = {
                "motion_blur": {"angle"},
                "unsharp_mask": {"radius", "threshold"},
                "smart_sharpen": {"radius"},
            }.get(kind, set())
            for key, widgets in extra_controls.items():
                for widget in widgets:
                    widget.grid() if key in visible_extras else widget.grid_remove()
            if kind == "blur":
                value_spin.configure(from_=1, to=500, increment=1)
                hint.configure(text="Радиус размытия в пикселях.")
            elif kind == "motion_blur":
                value_spin.configure(from_=1, to=300, increment=1); hint.configure(text="Длина размытия; угол хранится в параметрах фильтра.")
            elif kind == "sharpen":
                value_spin.configure(from_=0, to=10, increment=0.1)
                hint.configure(text="Сила повышения резкости.")
            elif kind in {"unsharp_mask", "smart_sharpen"}:
                value_spin.configure(from_=0, to=5, increment=0.05); hint.configure(text="Сила повышения резкости без изменения исходного слоя.")
            elif kind == "noise":
                value_spin.configure(from_=0, to=1, increment=0.01)
                hint.configure(text="Количество детерминированного шума 0..1.")
            elif kind == "reduce_noise":
                value_spin.configure(from_=0, to=1, increment=0.01); hint.configure(text="Сила шумоподавления с сохранением границ.")
            elif kind == "high_pass":
                value_spin.configure(from_=0.1, to=100, increment=0.1); hint.configure(text="Радиус выделения деталей на нейтральном сером.")
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
                update_value_controls(selected_filter_kind())
                return
            updating_controls = True
            item = filters[index]
            kind = str(item.get("type", "blur")).lower()
            filter_type.set(FILTER_LABELS.get(kind, FILTER_LABELS["blur"]))
            update_value_controls(selected_filter_kind())
            value_var.set(self.filter_primary_value(item))
            angle_var.set(float(item.get("angle", 0.0)))
            radius_var.set(float(item.get("radius", 2.0 if kind == "unsharp_mask" else 1.2)))
            threshold_var.set(float(item.get("threshold", 0.0)))
            enabled_var.set(bool(item.get("enabled", True)))
            opacity_var.set(float(item.get("opacity", 1.0)) * 100.0)
            blend_mode = str(item.get("blend_mode", "Normal"))
            filter_blend_mode.set(blend_mode if blend_mode in BLEND_MODES else "Normal")
            filter_channel.set(CHANNEL_LABELS.get(str(item.get("channel", "RGB")), "RGB"))
            mask_inverted.set(bool(item.get("mask_inverted", False)))
            mask_density.set(float(item.get("mask_density", 1.0)) * 100.0)
            mask_feather.set(float(item.get("mask_feather", 0.0)))
            updating_controls = False

        def current_item(original: dict | None = None) -> dict:
            metadata = dict(original or {})
            metadata["enabled"] = bool(enabled_var.get())
            metadata["opacity"] = float(opacity_var.get()) / 100.0
            metadata["blend_mode"] = filter_blend_mode.get()
            metadata["channel"] = CHANNEL_VALUES.get(filter_channel.get(), "RGB")
            metadata["mask_inverted"] = bool(mask_inverted.get())
            metadata["mask_density"] = float(mask_density.get()) / 100.0
            metadata["mask_feather"] = float(mask_feather.get())
            metadata["angle"] = float(angle_var.get())
            metadata["radius"] = float(radius_var.get())
            metadata["threshold"] = float(threshold_var.get())
            return self.make_filter_item(selected_filter_kind(), value_var.get(), metadata)

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
            path = filedialog.askopenfilename(filetypes=[("UZYRO filter preset", "*.json"), ("JSON", "*.json")])
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
            path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("UZYRO filter preset", "*.json"), ("JSON", "*.json")])
            if not path:
                return
            name = Path(path).stem
            data = {"format": "UZYRO filter preset", "version": 1, "name": name, "filters": normalized_filters}
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

        def edit_filter_mask() -> None:
            index = selected_index()
            if index is None:
                return
            filters[index] = current_item(filters[index])
            encoded = filters[index].get("mask")
            if isinstance(encoded, str) and encoded:
                try:
                    initial_mask = decode_png(encoded)[:, :, 0]
                except Exception:
                    initial_mask = np.full(pixels.shape[:2], 255, dtype=np.uint8)
            else:
                initial_mask = np.full(pixels.shape[:2], 255, dtype=np.uint8)
            edited = self.filter_mask_editor(initial_mask, selection_mask)
            if edited is None:
                return
            filters[index]["mask"] = encode_png(np.dstack([edited] * 4))
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
        ttk.Button(mask_buttons, text="Редактировать маску", command=edit_filter_mask).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(mask_buttons, text="Удалить маску", command=clear_filter_mask).pack(side=tk.LEFT)
        ttk.Button(preset_file_buttons, text="Загрузить пресет", command=load_preset_file).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(preset_file_buttons, text="Сохранить пресет", command=save_preset_file).pack(side=tk.LEFT)
        ttk.Button(bottom, text="ОК", command=accept).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(bottom, text="Отмена", command=cancel).pack(side=tk.RIGHT)
        listbox.bind("<<ListboxSelect>>", load_selected)
        type_box.bind("<<ComboboxSelected>>", lambda _event: (update_value_controls(selected_filter_kind()), apply_current()))
        value_spin.bind("<KeyRelease>", apply_current)
        value_spin.bind("<FocusOut>", apply_current)
        opacity_spin.bind("<KeyRelease>", apply_current)
        opacity_spin.bind("<FocusOut>", apply_current)
        filter_blend_box.bind("<<ComboboxSelected>>", apply_current)
        filter_channel_box.bind("<<ComboboxSelected>>", apply_current)
        value_var.trace_add("write", lambda *_args: apply_current())
        angle_var.trace_add("write", lambda *_args: apply_current())
        radius_var.trace_add("write", lambda *_args: apply_current())
        threshold_var.trace_add("write", lambda *_args: apply_current())
        enabled_var.trace_add("write", lambda *_args: apply_current())
        opacity_var.trace_add("write", lambda *_args: apply_current())
        filter_blend_mode.trace_add("write", lambda *_args: apply_current())
        filter_channel.trace_add("write", lambda *_args: apply_current())
        mask_inverted.trace_add("write", lambda *_args: apply_current())
        mask_density.trace_add("write", lambda *_args: apply_current())
        mask_feather.trace_add("write", lambda *_args: apply_current())
        dialog.protocol("WM_DELETE_WINDOW", cancel)
        self._filter_dialog_channel = filter_channel
        self._filter_dialog_mask_inverted = mask_inverted
        self._filter_dialog_mask_density = mask_density
        self._filter_dialog_mask_feather = mask_feather
        self._filter_dialog_extra_values = {"angle": angle_var, "radius": radius_var, "threshold": threshold_var}
        self._filter_dialog_extra_controls = extra_controls
        self._filter_dialog_accept = accept
        refresh_list(0 if filters else None)
        if not filters:
            update_value_controls(selected_filter_kind())
            update_preview()
        dialog.wait_window()
        return result
