from __future__ import annotations

from ..app_shared import *


class DestructiveFiltersMixin:
    def destructive_filter_dialog(self, title: str, specs: list[tuple[str, str, float, float, float, float]], operation) -> dict[str, float] | None:
        source = self.doc.layer.working_rgba() if self.doc.layer.working_pixels is not None else self.doc.layer.pixels.copy()
        result: dict[str, float] | None = None
        dialog = tk.Toplevel(self); dialog.title(title); dialog.transient(self); dialog.grab_set(); dialog.resizable(False, False)
        previews = ttk.Frame(dialog, padding=12); previews.pack(fill=tk.X)
        original_label = ttk.Label(previews); original_label.pack(side=tk.LEFT, padx=(0, 8))
        result_label = ttk.Label(previews); result_label.pack(side=tk.LEFT)
        controls = ttk.Frame(dialog, padding=(12, 0, 12, 8)); controls.pack(fill=tk.X)
        variables = {key: tk.DoubleVar(value=default) for key, _label, default, _low, _high, _step in specs}
        value_labels: dict[str, ttk.Label] = {}
        for row, (key, label, _default, low, high, step) in enumerate(specs):
            ttk.Label(controls, text=label).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Scale(controls, from_=low, to=high, variable=variables[key], orient=tk.HORIZONTAL).grid(row=row, column=1, sticky="ew", padx=8)
            value = ttk.Label(controls, width=8, anchor=tk.E); value.grid(row=row, column=2, sticky="e")
            value_labels[key] = value
            variables[key]._filter_step = step
        controls.columnconfigure(1, weight=1)

        thumb = rgba_array_to_pil(display_rgba(source)); thumb.thumbnail((250, 250), Image.Resampling.LANCZOS)
        preview_source = np.asarray(thumb.convert("RGBA"), dtype=np.uint8)
        self._destructive_filter_original = ImageTk.PhotoImage(thumb); original_label.configure(image=self._destructive_filter_original)
        redraw_after = [None]

        def values() -> dict[str, float]:
            return {key: float(variable.get()) for key, variable in variables.items()}

        def redraw() -> None:
            redraw_after[0] = None
            current = values(); shown = operation(preview_source, current)
            self._destructive_filter_preview = ImageTk.PhotoImage(rgba_array_to_pil(display_rgba(shown)))
            result_label.configure(image=self._destructive_filter_preview)
            for key, label in value_labels.items():
                step = float(getattr(variables[key], "_filter_step", 1.0))
                label.configure(text=f"{current[key]:.2f}" if step < 1 else f"{current[key]:.0f}")

        def schedule(*_args) -> None:
            if redraw_after[0] is None:
                redraw_after[0] = dialog.after(35, redraw)

        def accept() -> None:
            nonlocal result
            result = values(); dialog.destroy()

        footer = ttk.Frame(dialog, padding=12); footer.pack(fill=tk.X)
        ttk.Button(footer, text="Применить", command=accept, style="Primary.TButton").pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(footer, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT)
        for variable in variables.values():
            variable.trace_add("write", schedule)
        dialog.bind("<Return>", lambda _event: accept()); dialog.bind("<Escape>", lambda _event: dialog.destroy())
        self._destructive_filter_values = variables; self._destructive_filter_accept = accept
        self.center_toplevel(dialog, 570, 430); redraw(); dialog.wait_window()
        return result

    def run_destructive_filter(self, title: str, specs, operation) -> None:
        values = self.destructive_filter_dialog(title, specs, operation)
        if values is not None:
            self.apply_to_layer(title, lambda pixels: operation(pixels, values))

    def filter_blur(self) -> None:
        self.run_destructive_filter("Размытие по Гауссу", [("radius", "Радиус", 3, 1, 200, 1)], lambda arr, p: blur(arr, int(p["radius"])))

    def filter_motion_blur(self) -> None:
        self.run_destructive_filter("Размытие в движении", [("distance", "Расстояние", 16, 1, 150, 1), ("angle", "Угол", 0, -180, 180, 1)], lambda arr, p: motion_blur(arr, int(p["distance"]), p["angle"]))

    def filter_sharpen(self) -> None:
        self.run_destructive_filter("Контурная резкость", [("amount", "Сила", 1.0, 0, 5, 0.05), ("radius", "Радиус", 2.0, 0.1, 20, 0.1), ("threshold", "Порог", 0, 0, 255, 1)], lambda arr, p: unsharp_mask(arr, p["amount"], p["radius"], p["threshold"]))

    def filter_smart_sharpen(self) -> None:
        self.run_destructive_filter("Умная резкость", [("amount", "Сила", 1.0, 0, 5, 0.05), ("radius", "Радиус", 1.2, 0.1, 20, 0.1)], lambda arr, p: smart_sharpen(arr, p["amount"], p["radius"]))

    def filter_noise(self) -> None:
        self.run_destructive_filter("Добавить шум", [("amount", "Количество", 0.04, 0, 0.5, 0.01)], lambda arr, p: deterministic_noise(arr, p["amount"]))

    def filter_reduce_noise(self) -> None:
        self.run_destructive_filter("Уменьшение шума", [("strength", "Сила", 0.35, 0, 1, 0.01)], lambda arr, p: reduce_noise(arr, p["strength"]))

    def filter_high_pass(self) -> None:
        self.run_destructive_filter("Цветовой контраст", [("radius", "Радиус", 4, 0.1, 100, 0.1)], lambda arr, p: high_pass(arr, p["radius"]))
