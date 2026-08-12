from __future__ import annotations

from ..app_shared import *
from ..layer_effects import EFFECT_LABELS, EFFECT_ORDER, LayerEffectsStack
from ..geometry_ops import checker_background
from ..render_ops import alpha_blend_inplace, render_layer_style


STYLE_PARAMETERS = {
    "drop_shadow": (("x", "Смещение X", -200, 200, 1), ("y", "Смещение Y", -200, 200, 1), ("blur", "Размытие", 0, 200, 1)),
    "outer_glow": (("blur", "Размер", 1, 200, 1),),
    "stroke": (("size", "Размер", 1, 200, 1),),
    "bevel_emboss": (("size", "Размер", 1, 100, 1), ("depth", "Глубина", 0, 5, 0.05), ("angle", "Угол", -180, 180, 1)),
    "inner_shadow": (("x", "Смещение X", -200, 200, 1), ("y", "Смещение Y", -200, 200, 1), ("blur", "Размытие", 1, 200, 1)),
    "inner_glow": (("blur", "Размер", 1, 200, 1),),
    "satin": (("distance", "Расстояние", -200, 200, 1), ("size", "Размер", 1, 200, 1), ("angle", "Угол", -180, 180, 1)),
    "color_overlay": (),
    "gradient_overlay": (("angle", "Угол", -180, 180, 1), ("scale", "Масштаб", 1, 400, 1)),
    "pattern_overlay": (("scale", "Размер узора", 2, 200, 1), ("angle", "Угол", -180, 180, 1)),
}


class LayerStylesMixin:
    def layer_styles_dialog(self, layer: Layer) -> dict[str, dict] | None:
        effects = LayerEffectsStack.normalize(layer.effects)
        result: dict[str, dict] | None = None
        current_kind = [EFFECT_ORDER[0]]
        loading = [False]
        dialog = tk.Toplevel(self)
        dialog.title("Стили слоя")
        dialog.transient(self); dialog.grab_set(); dialog.minsize(760, 520)

        body = ttk.Frame(dialog, padding=12); body.pack(fill=tk.BOTH, expand=True)
        preview = ttk.Label(body); preview.grid(row=0, column=0, rowspan=2, sticky="n", padx=(0, 12))
        listbox = tk.Listbox(body, width=27, height=16, exportselection=False, activestyle="none")
        listbox.grid(row=0, column=1, rowspan=2, sticky="ns", padx=(0, 12))
        controls = ttk.Frame(body); controls.grid(row=0, column=2, sticky="new")
        body.columnconfigure(2, weight=1); body.rowconfigure(1, weight=1)

        enabled = tk.BooleanVar(value=False); opacity = tk.DoubleVar(value=65.0); blend_mode = tk.StringVar(value="Normal")
        ttk.Checkbutton(controls, text="Включён", variable=enabled).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Label(controls, text="Непрозрачность").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Spinbox(controls, textvariable=opacity, from_=0, to=100, increment=1, width=12).grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Label(controls, text="Режим наложения").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Combobox(controls, textvariable=blend_mode, values=BLEND_MODES, state="readonly", width=16).grid(row=2, column=1, sticky="ew", pady=3)

        color_frame = ttk.Frame(controls); color_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 4))
        color_buttons: list[tk.Button] = []
        colors: list[list[int]] = [[0, 0, 0, 255], [255, 255, 255, 255]]
        parameter_frame = ttk.Frame(controls); parameter_frame.grid(row=4, column=0, columnspan=2, sticky="ew")
        parameter_vars = [tk.DoubleVar(value=0.0) for _ in range(3)]
        parameter_labels: list[ttk.Label] = []; parameter_spins: list[ttk.Spinbox] = []
        for index, variable in enumerate(parameter_vars):
            label = ttk.Label(parameter_frame, text=""); spin = ttk.Spinbox(parameter_frame, textvariable=variable, width=12)
            label.grid(row=index, column=0, sticky="w", pady=3); spin.grid(row=index, column=1, sticky="ew", pady=3)
            parameter_labels.append(label); parameter_spins.append(spin)
        position_label = ttk.Label(parameter_frame, text="Положение")
        position = tk.StringVar(value="outside")
        position_box = ttk.Combobox(parameter_frame, textvariable=position, values=["outside", "center", "inside"], state="readonly", width=16)

        source = layer.pixels
        scale = min(1.0, 220 / max(1, source.shape[1]), 220 / max(1, source.shape[0]))
        size = max(1, round(source.shape[1] * scale)), max(1, round(source.shape[0] * scale))
        preview_source = source.copy() if size == (source.shape[1], source.shape[0]) else cv2.resize(source, size, interpolation=cv2.INTER_AREA)

        def refresh_list(selected: int | None = None) -> None:
            listbox.delete(0, tk.END)
            for kind in EFFECT_ORDER:
                active = bool(effects.get(kind, {}).get("enabled", False))
                listbox.insert(tk.END, f"{'●' if active else '○'}  {EFFECT_LABELS[kind]}")
            target = EFFECT_ORDER.index(current_kind[0]) if selected is None else selected
            listbox.selection_set(target); listbox.activate(target)

        def effect_colors(kind: str, item: dict) -> tuple[str, str | None]:
            if kind == "bevel_emboss":
                return "highlight", "shadow"
            if kind in {"gradient_overlay", "pattern_overlay"}:
                return "color", "color2"
            return "color", None

        def pick_color(index: int) -> None:
            chosen = colorchooser.askcolor(tuple(colors[index][:3]), parent=dialog)[0]
            if chosen is None:
                return
            colors[index] = [int(round(value)) for value in chosen] + [255]
            update_color_buttons(); save_current()

        def update_color_buttons() -> None:
            for button in color_buttons:
                button.destroy()
            color_buttons.clear()
            keys = effect_colors(current_kind[0], effects[current_kind[0]])
            for index, key in enumerate(keys):
                if key is None:
                    continue
                label = "Основной цвет" if index == 0 else "Второй цвет"
                ttk.Label(color_frame, text=label).grid(row=index, column=0, sticky="w", pady=2)
                value = colors[index]; color_hex = f"#{value[0]:02x}{value[1]:02x}{value[2]:02x}"
                button = tk.Button(color_frame, background=color_hex, activebackground=color_hex, width=5, command=lambda i=index: pick_color(i))
                button.grid(row=index, column=1, sticky="e", padx=(10, 0), pady=2); color_buttons.append(button)

        def load_selected(_event=None) -> None:
            selection = listbox.curselection()
            if not selection:
                return
            loading[0] = True
            kind = EFFECT_ORDER[int(selection[0])]; current_kind[0] = kind
            item = LayerEffectsStack.item(kind, effects.get(kind)); effects[kind] = item
            enabled.set(bool(item["enabled"])); opacity.set(float(item["opacity"]) * 100.0); blend_mode.set(str(item["blend_mode"]))
            first, second = effect_colors(kind, item)
            colors[0] = list(item.get(first, [0, 0, 0, 255])); colors[1] = list(item.get(second, [255, 255, 255, 255])) if second else [255, 255, 255, 255]
            specs = STYLE_PARAMETERS[kind]
            for index, (key, label, low, high, increment) in enumerate(specs):
                parameter_labels[index].configure(text=label); parameter_spins[index].configure(from_=low, to=high, increment=increment)
                parameter_vars[index].set(float(item.get(key, 0))); parameter_labels[index].grid(); parameter_spins[index].grid()
            for index in range(len(specs), 3):
                parameter_labels[index].grid_remove(); parameter_spins[index].grid_remove()
            if kind == "stroke":
                position.set(str(item.get("position", "outside"))); position_label.grid(row=3, column=0, sticky="w", pady=3); position_box.grid(row=3, column=1, sticky="ew", pady=3)
            else:
                position_label.grid_remove(); position_box.grid_remove()
            update_color_buttons(); loading[0] = False; update_preview()

        def save_current(*_args) -> None:
            if loading[0]:
                return
            kind = current_kind[0]; item = LayerEffectsStack.item(kind, effects.get(kind))
            item.update(enabled=enabled.get(), opacity=float(opacity.get()) / 100.0, blend_mode=blend_mode.get())
            for index, (key, _label, _low, _high, _increment) in enumerate(STYLE_PARAMETERS[kind]):
                item[key] = float(parameter_vars[index].get())
            first, second = effect_colors(kind, item); item[first] = list(colors[0])
            if second:
                item[second] = list(colors[1])
            if kind == "stroke":
                item["position"] = position.get()
            effects[kind] = item; refresh_list(); update_preview()

        def update_preview() -> None:
            temporary = Layer("Предпросмотр", preview_source, x=18, y=18, effects=effects)
            underlays, styled = render_layer_style(temporary, preview_source)
            canvas = checker_background(256, 256).copy()
            for effect, x, y, value, mode in underlays:
                alpha_blend_inplace(canvas, effect, x, y, value, blend_mode=mode)
            alpha_blend_inplace(canvas, styled, 18, 18, 1.0)
            self._layer_styles_preview_image = ImageTk.PhotoImage(rgba_array_to_pil(canvas))
            preview.configure(image=self._layer_styles_preview_image)

        def accept() -> None:
            nonlocal result
            save_current(); result = LayerEffectsStack.normalize(effects); dialog.destroy()

        footer = ttk.Frame(dialog, padding=12); footer.pack(fill=tk.X)
        ttk.Button(footer, text="Применить", command=accept, style="Primary.TButton").pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(footer, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT)
        listbox.bind("<<ListboxSelect>>", load_selected)
        for variable in [enabled, opacity, blend_mode, position, *parameter_vars]:
            variable.trace_add("write", save_current)
        self._layer_styles_effects = effects; self._layer_styles_enabled = enabled; self._layer_styles_accept = accept; self._layer_styles_listbox = listbox
        refresh_list(0); load_selected(); self.center_toplevel(dialog, 820, 570); dialog.wait_window()
        return result
