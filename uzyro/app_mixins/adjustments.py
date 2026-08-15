from __future__ import annotations

from ..app_shared import *


class AdjustmentsMixin:
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
        if initial.get("type") == "grayscale":
            initial = {"type": "black_white", "red": 0.299, "green": 0.587, "blue": 0.114}
        result: dict | None = None
        dialog = tk.Toplevel(self)
        dialog.title("\u041a\u043e\u0440\u0440\u0435\u043a\u0442\u0438\u0440\u0443\u044e\u0449\u0438\u0439 \u0441\u043b\u043e\u0439")
        dialog.transient(self)
        dialog.resizable(False, False)
        dialog.grab_set()

        source = self.render_engine.render(self.doc, checker=False)
        initial_kind = str(initial.get("type", "brightness_contrast"))
        adjustment_type = tk.StringVar(value=ADJUSTMENT_LABELS.get(initial_kind, ADJUSTMENT_LABELS["brightness_contrast"]))
        adjustment_channel = tk.StringVar(value=CHANNEL_LABELS.get(str(initial.get("channel", "RGB")), "RGB"))
        values = [tk.DoubleVar(value=0.0), tk.DoubleVar(value=0.0), tk.DoubleVar(value=0.0)]
        labels: list[ttk.Label] = []
        spins: list[ttk.Spinbox] = []
        updating = False

        preview = ttk.Label(dialog)
        preview.grid(row=0, column=0, rowspan=9, padx=12, pady=12, sticky="n")
        ttk.Label(dialog, text="Пресет").grid(row=0, column=1, sticky="w", padx=(0, 12), pady=(12, 4))
        adjustment_preset = tk.StringVar(value=next(iter(self.adjustment_presets)))
        preset_box = ttk.Combobox(dialog, textvariable=adjustment_preset, values=list(self.adjustment_presets), state="readonly", width=22)
        preset_box.grid(row=0, column=2, sticky="ew", padx=(0, 12), pady=(12, 4))
        ttk.Button(dialog, text="Применить пресет", command=lambda: apply_adjustment_preset()).grid(row=1, column=1, columnspan=2, sticky="ew", padx=(0, 12), pady=(0, 8))
        ttk.Label(dialog, text="\u0422\u0438\u043f").grid(row=2, column=1, sticky="w", padx=(0, 12), pady=(4, 4))
        type_box = ttk.Combobox(dialog, textvariable=adjustment_type, values=list(ADJUSTMENT_VALUES), state="readonly", width=22)
        type_box.grid(row=2, column=2, sticky="ew", padx=(0, 12), pady=(4, 4))
        ttk.Label(dialog, text="Канал").grid(row=6, column=1, sticky="w", padx=(0, 12), pady=(4, 4))
        channel_box = ttk.Combobox(dialog, textvariable=adjustment_channel, values=list(CHANNEL_VALUES), state="readonly", width=22)
        channel_box.grid(row=6, column=2, sticky="ew", padx=(0, 12), pady=(4, 4))
        hint = ttk.Label(dialog, text="", wraplength=240, justify=tk.LEFT)
        hint.grid(row=7, column=1, columnspan=2, sticky="w", padx=(0, 12), pady=(8, 0))

        for index in range(3):
            label = ttk.Label(dialog, text="")
            spin = ttk.Spinbox(dialog, textvariable=values[index], from_=0, to=255, increment=1, width=12)
            label.grid(row=index + 3, column=1, sticky="w", padx=(0, 12), pady=4)
            spin.grid(row=index + 3, column=2, sticky="ew", padx=(0, 12), pady=4)
            labels.append(label)
            spins.append(spin)

        buttons = ttk.Frame(dialog)
        buttons.grid(row=8, column=1, columnspan=2, sticky="e", padx=12, pady=12)

        def current_adjustment() -> dict:
            kind = ADJUSTMENT_VALUES.get(adjustment_type.get(), adjustment_type.get())
            item = self.make_adjustment_item(kind, [value.get() for value in values])
            item["channel"] = CHANNEL_VALUES.get(adjustment_channel.get(), "RGB")
            return item

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
            adjustment_channel.set(CHANNEL_LABELS.get(str(adjustment.get("channel", "RGB")), "RGB"))
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
            set_values_for_kind(ADJUSTMENT_VALUES.get(adjustment_type.get(), adjustment_type.get()))
            update_preview()

        def apply_adjustment_preset() -> None:
            preset = self.adjustment_presets.get(adjustment_preset.get())
            if preset is None:
                return
            kind = str(preset.get("type", "brightness_contrast"))
            if kind not in ADJUSTMENT_TYPES:
                return
            adjustment_type.set(ADJUSTMENT_LABELS[kind])
            set_values_for_kind(kind, preset)
            update_preview()

        def import_adjustment_presets() -> None:
            path = filedialog.askopenfilename(filetypes=[("UZYRO presets", "*.json"), ("JSON", "*.json")], parent=dialog)
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
            path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("UZYRO presets", "*.json"), ("JSON", "*.json")], parent=dialog)
            if path:
                payload = {"format": "UZYRO adjustment presets", "version": 1, "presets": self.adjustment_presets}
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
        channel_box.bind("<<ComboboxSelected>>", update_preview)
        for value in values:
            value.trace_add("write", update_preview)
        ttk.Button(buttons, text="\u041e\u041a", command=accept).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="\u041e\u0442\u043c\u0435\u043d\u0430", command=cancel).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Экспорт", command=export_adjustment_presets).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(buttons, text="Импорт", command=import_adjustment_presets).pack(side=tk.LEFT)
        dialog.protocol("WM_DELETE_WINDOW", cancel)
        self._adjustment_dialog_channel = adjustment_channel
        self._adjustment_dialog_accept = accept
        set_values_for_kind(initial_kind, initial)
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
        if kind == "black_white":
            return [("Красный", float(adjustment.get("red", 0.299)), 0, 1, 0.01), ("Зелёный", float(adjustment.get("green", 0.587)), 0, 1, 0.01), ("Синий", float(adjustment.get("blue", 0.114)), 0, 1, 0.01)]
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
        if kind == "invert":
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
        if kind == "black_white":
            return {"type": kind, "red": float(values[0]), "green": float(values[1]), "blue": float(values[2])}
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
        return apply_adjustment(arr, adjustment)

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
