from __future__ import annotations

import colorsys

from ..app_shared import *


class ColorPickerMixin:
    def color_picker_dialog(self, initial: tuple[int, int, int, int], title: str) -> tuple[int, int, int, int] | None:
        current = tuple(int(np.clip(value, 0, 255)) for value in (*initial[:3], initial[3] if len(initial) > 3 else 255))
        result: tuple[int, int, int, int] | None = None
        updating = [False]
        dialog = tk.Toplevel(self)
        dialog.title(title); dialog.transient(self); dialog.grab_set(); dialog.resizable(False, False)
        body = ttk.Frame(dialog, padding=14); body.pack(fill=tk.BOTH, expand=True)
        picker = ttk.Frame(body); picker.grid(row=0, column=0, rowspan=3, sticky="n")
        plane = tk.Canvas(picker, width=300, height=220, highlightthickness=1, highlightbackground=TOKENS.BORDER, cursor="crosshair")
        plane.grid(row=0, column=0)
        hue_bar = tk.Canvas(picker, width=26, height=220, highlightthickness=1, highlightbackground=TOKENS.BORDER, cursor="sb_v_double_arrow")
        hue_bar.grid(row=0, column=1, padx=(8, 0))

        controls = ttk.Frame(body); controls.grid(row=0, column=1, sticky="new", padx=(16, 0))
        tabs = ttk.Notebook(controls); tabs.pack(fill=tk.X)
        rgb_vars = [tk.IntVar(value=current[index]) for index in range(3)]
        alpha_var = tk.IntVar(value=current[3])
        hsv_vars = [tk.DoubleVar() for _ in range(3)]
        hsl_vars = [tk.DoubleVar() for _ in range(3)]
        hex_var = tk.StringVar()

        def fields(tab_name: str, labels: tuple[str, ...], variables: list, ranges: tuple[tuple[float, float], ...], command) -> None:
            tab = ttk.Frame(tabs, padding=10); tabs.add(tab, text=tab_name)
            for row, (label, variable, limits) in enumerate(zip(labels, variables, ranges)):
                ttk.Label(tab, text=label).grid(row=row, column=0, sticky="w", pady=3)
                spin = ttk.Spinbox(tab, textvariable=variable, from_=limits[0], to=limits[1], increment=1, width=12, command=command)
                spin.grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=3)
                spin.bind("<Return>", command); spin.bind("<FocusOut>", command)
            tab.columnconfigure(1, weight=1)

        preview = tk.Canvas(controls, width=230, height=58, highlightthickness=1, highlightbackground=TOKENS.BORDER)
        previews = ttk.Frame(controls); previews.pack(fill=tk.X, pady=(12, 3))
        ttk.Label(previews, text="Текущий").pack(side=tk.LEFT)
        ttk.Label(previews, text="Новый").pack(side=tk.RIGHT)
        preview.pack(fill=tk.X)
        hex_row = ttk.Frame(controls); hex_row.pack(fill=tk.X, pady=(10, 3))
        ttk.Label(hex_row, text="HEX").pack(side=tk.LEFT)
        hex_entry = ttk.Entry(hex_row, textvariable=hex_var, width=14); hex_entry.pack(side=tk.RIGHT)
        alpha_row = ttk.Frame(controls); alpha_row.pack(fill=tk.X, pady=3)
        ttk.Label(alpha_row, text="Alpha").pack(side=tk.LEFT)
        alpha_spin = ttk.Spinbox(alpha_row, textvariable=alpha_var, from_=0, to=255, width=12); alpha_spin.pack(side=tk.RIGHT)

        plane_photo = [None]; hue_photo = [None]; plane_marker = [None]; hue_marker = [None]; cached_hue = [None]

        def rgba() -> tuple[int, int, int, int]:
            return tuple(int(np.clip(variable.get(), 0, 255)) for variable in (*rgb_vars, alpha_var))

        def draw_hue_bar() -> None:
            hsv = np.zeros((220, 26, 3), dtype=np.uint8)
            hsv[:, :, 0] = np.linspace(0, 179, 220, dtype=np.uint8)[:, None]
            hsv[:, :, 1:] = 255
            hue_photo[0] = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), "RGB"))
            hue_bar.create_image(0, 0, anchor=tk.NW, image=hue_photo[0])

        def redraw() -> None:
            hue, saturation, value = (float(variable.get()) for variable in hsv_vars)
            if cached_hue[0] != round(hue, 3):
                hsv = np.empty((220, 300, 3), dtype=np.uint8)
                hsv[:, :, 0] = round((hue % 360.0) / 2.0)
                hsv[:, :, 1] = np.linspace(0, 255, 300, dtype=np.uint8)[None, :]
                hsv[:, :, 2] = np.linspace(255, 0, 220, dtype=np.uint8)[:, None]
                plane_photo[0] = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), "RGB"))
                plane.delete("image"); plane.create_image(0, 0, anchor=tk.NW, image=plane_photo[0], tags="image")
                cached_hue[0] = round(hue, 3)
            x = float(np.clip(saturation, 0, 100)) * 2.99
            y = (100.0 - float(np.clip(value, 0, 100))) * 2.19
            if plane_marker[0] is None: plane_marker[0] = plane.create_oval(0, 0, 0, 0, outline="white", width=2)
            plane.coords(plane_marker[0], x - 5, y - 5, x + 5, y + 5); plane.tag_raise(plane_marker[0])
            hy = (hue % 360.0) / 360.0 * 219
            if hue_marker[0] is None: hue_marker[0] = hue_bar.create_rectangle(0, 0, 25, 0, outline="white", width=2)
            hue_bar.coords(hue_marker[0], 0, hy - 2, 25, hy + 2); hue_bar.tag_raise(hue_marker[0])
            color = rgba(); preview.delete("all")
            preview.create_rectangle(0, 0, 115, 58, fill=self.color_hex(current), outline="")
            preview.create_rectangle(115, 0, 230, 58, fill=self.color_hex(color), outline="")

        def set_color(color: tuple[int, int, int, int]) -> None:
            updating[0] = True
            for variable, value in zip(rgb_vars, color[:3]): variable.set(int(value))
            alpha_var.set(int(color[3]))
            rf, gf, bf = (value / 255.0 for value in color[:3])
            h, s, v = colorsys.rgb_to_hsv(rf, gf, bf); lh, light, ls = colorsys.rgb_to_hls(rf, gf, bf)
            for variable, value in zip(hsv_vars, (h * 360, s * 100, v * 100)): variable.set(round(value, 2))
            for variable, value in zip(hsl_vars, (lh * 360, ls * 100, light * 100)): variable.set(round(value, 2))
            hex_var.set("#{:02X}{:02X}{:02X}{:02X}".format(*color))
            updating[0] = False; redraw()

        def from_rgb(_event=None) -> None:
            if not updating[0]: set_color(rgba())

        def from_hsv(_event=None) -> None:
            if updating[0]: return
            h, s, v = (float(variable.get()) for variable in hsv_vars)
            rgb = colorsys.hsv_to_rgb((h % 360) / 360.0, np.clip(s / 100, 0, 1), np.clip(v / 100, 0, 1))
            set_color(tuple(round(value * 255) for value in rgb) + (int(np.clip(alpha_var.get(), 0, 255)),))

        def from_hsl(_event=None) -> None:
            if updating[0]: return
            h, s, light = (float(variable.get()) for variable in hsl_vars)
            rgb = colorsys.hls_to_rgb((h % 360) / 360.0, np.clip(light / 100, 0, 1), np.clip(s / 100, 0, 1))
            set_color(tuple(round(value * 255) for value in rgb) + (int(np.clip(alpha_var.get(), 0, 255)),))

        def from_hex(_event=None) -> None:
            raw = hex_var.get().strip().lstrip("#")
            if len(raw) not in {6, 8}: return
            try: values = tuple(int(raw[index:index + 2], 16) for index in range(0, len(raw), 2))
            except ValueError: return
            set_color(values if len(values) == 4 else (*values, int(np.clip(alpha_var.get(), 0, 255))))

        def pick_plane(event) -> None:
            hsv_vars[1].set(np.clip(event.x / 299 * 100, 0, 100)); hsv_vars[2].set(np.clip((219 - event.y) / 219 * 100, 0, 100)); from_hsv()

        def pick_hue(event) -> None:
            hsv_vars[0].set(np.clip(event.y / 219 * 360, 0, 359.99)); from_hsv()

        def accept() -> None:
            nonlocal result
            result = rgba(); dialog.destroy()

        fields("RGB", ("R", "G", "B"), rgb_vars, ((0, 255),) * 3, from_rgb)
        fields("HSV", ("H", "S", "V"), hsv_vars, ((0, 359.99), (0, 100), (0, 100)), from_hsv)
        fields("HSL", ("H", "S", "L"), hsl_vars, ((0, 359.99), (0, 100), (0, 100)), from_hsl)
        footer = ttk.Frame(body); footer.grid(row=3, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(footer, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT)
        ttk.Button(footer, text="ОК", command=accept, style="Primary.TButton").pack(side=tk.RIGHT, padx=(0, 6))
        plane.bind("<Button-1>", pick_plane); plane.bind("<B1-Motion>", pick_plane)
        hue_bar.bind("<Button-1>", pick_hue); hue_bar.bind("<B1-Motion>", pick_hue)
        hex_entry.bind("<Return>", from_hex); hex_entry.bind("<FocusOut>", from_hex)
        alpha_spin.bind("<Return>", from_rgb); alpha_spin.bind("<FocusOut>", from_rgb)
        dialog.bind("<Return>", lambda _event: accept()); dialog.bind("<Escape>", lambda _event: dialog.destroy())
        self._color_picker_vars = {"rgb": rgb_vars, "hsv": hsv_vars, "hsl": hsl_vars, "hex": hex_var, "alpha": alpha_var}
        self._color_picker_apply_hex = from_hex
        self._color_picker_apply_hsv = from_hsv
        self._color_picker_apply_hsl = from_hsl
        self._color_picker_accept = accept
        draw_hue_bar(); set_color(current); self.center_toplevel(dialog, 650, 390); dialog.wait_window()
        return result
