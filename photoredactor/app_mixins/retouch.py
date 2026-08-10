from __future__ import annotations

from ..app_shared import *


class RetouchMixin:
    def frequency_separation_layers(self) -> None:
        layer = self.doc.layer
        if layer.locked or layer.kind == "adjustment":
            messagebox.showinfo("Частотное разложение", "Выберите незаблокированный слой с изображением.")
            return
        settings = self.frequency_separation_dialog(layer.pixels)
        if settings is None:
            return
        self.run_document_command(
            "Частотное разложение",
            lambda: self.doc.frequency_separate_active(settings["radius"], settings["texture_strength"]),
        )
        self.refresh()

    def frequency_separation_dialog(self, source: np.ndarray) -> dict[str, float] | None:
        dialog = tk.Toplevel(self)
        dialog.title("Частотное разложение")
        dialog.transient(self)
        dialog.resizable(False, False)
        dialog.grab_set()
        result: dict[str, float] | None = None
        radius = tk.DoubleVar(value=8.0)
        texture_strength = tk.DoubleVar(value=1.0)

        preview_row = ttk.Frame(dialog)
        preview_row.pack(fill=tk.X, padx=12, pady=(12, 8))
        preview_labels: list[ttk.Label] = []
        for title in ["Цвет и тон", "Текстура", "Результат"]:
            column = ttk.Frame(preview_row)
            column.pack(side=tk.LEFT, padx=4)
            ttk.Label(column, text=title).pack(pady=(0, 4))
            label = ttk.Label(column)
            label.pack()
            preview_labels.append(label)

        controls = ttk.Frame(dialog)
        controls.pack(fill=tk.X, padx=16)
        radius_value = ttk.Label(controls, width=8)
        texture_value = ttk.Label(controls, width=8)
        ttk.Label(controls, text="Радиус размытия").grid(row=0, column=0, sticky="w")
        ttk.Scale(controls, from_=0.5, to=40.0, variable=radius, orient=tk.HORIZONTAL).grid(row=0, column=1, sticky="ew", padx=8)
        radius_value.grid(row=0, column=2, sticky="e")
        ttk.Label(controls, text="Сила текстуры").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Scale(controls, from_=0.0, to=2.0, variable=texture_strength, orient=tk.HORIZONTAL).grid(row=1, column=1, sticky="ew", padx=8, pady=(8, 0))
        texture_value.grid(row=1, column=2, sticky="e", pady=(8, 0))
        controls.columnconfigure(1, weight=1)

        thumb_image = rgba_array_to_pil(source)
        thumb_image.thumbnail((190, 190), Image.Resampling.LANCZOS)
        thumb = np.array(thumb_image.convert("RGBA"), dtype=np.uint8)

        def photo_for(arr: np.ndarray) -> ImageTk.PhotoImage:
            image = rgba_array_to_pil(arr)
            canvas = Image.new("RGBA", (190, 190), (44, 46, 52, 255))
            canvas.alpha_composite(image, ((190 - image.width) // 2, (190 - image.height) // 2))
            return ImageTk.PhotoImage(canvas)

        def update_preview(*_args) -> None:
            low, high = frequency_separation(thumb, radius.get(), texture_strength.get())
            recombined = low.copy()
            recombined[:, :, :3] = np.clip(
                low[:, :, :3].astype(np.float32) + high[:, :, :3].astype(np.float32) * 2.0 - 255.0,
                0,
                255,
            ).astype(np.uint8)
            self._frequency_preview_images = [photo_for(low), photo_for(high), photo_for(recombined)]
            for label, image in zip(preview_labels, self._frequency_preview_images):
                label.configure(image=image)
            radius_value.configure(text=f"{radius.get():.1f} px")
            texture_value.configure(text=f"{texture_strength.get():.2f}")

        def accept() -> None:
            nonlocal result
            result = {"radius": float(radius.get()), "texture_strength": float(texture_strength.get())}
            dialog.destroy()

        buttons = ttk.Frame(dialog)
        buttons.pack(fill=tk.X, padx=12, pady=12)
        ttk.Button(buttons, text="ОК", command=accept).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT)
        ToolTip(preview_labels[0], "На этом слое удобно выравнивать цвет и тон кожи.")
        ToolTip(preview_labels[1], "Этот слой сохраняет поры, волосы и мелкие детали.")
        radius.trace_add("write", update_preview)
        texture_strength.trace_add("write", update_preview)
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        update_preview()
        dialog.wait_window()
        return result

    def portrait_cleanup_layer(self) -> None:
        layer = self.doc.layer
        if layer.locked or layer.kind == "adjustment":
            messagebox.showinfo("Портретная обработка", "Выберите незаблокированный слой с портретом.")
            return
        settings = self.portrait_cleanup_dialog(layer.pixels)
        if settings is None:
            return
        self.apply_to_layer(
            "Портретная обработка",
            lambda arr: portrait_cleanup(arr, settings["smoothing"], settings["texture"], settings["even_tone"], settings["redness"]),
        )

    def portrait_cleanup_dialog(self, source: np.ndarray) -> dict[str, float] | None:
        dialog = tk.Toplevel(self)
        dialog.title("Портретная обработка")
        dialog.transient(self)
        dialog.resizable(False, False)
        dialog.grab_set()
        result: dict[str, float] | None = None
        variables = {
            "smoothing": tk.DoubleVar(value=0.35),
            "texture": tk.DoubleVar(value=0.7),
            "even_tone": tk.DoubleVar(value=0.2),
            "redness": tk.DoubleVar(value=0.2),
        }

        preview_row = ttk.Frame(dialog)
        preview_row.pack(fill=tk.X, padx=12, pady=(12, 8))
        preview_labels: list[ttk.Label] = []
        for title in ["До", "После"]:
            column = ttk.Frame(preview_row)
            column.pack(side=tk.LEFT, padx=5)
            ttk.Label(column, text=title).pack(pady=(0, 4))
            label = ttk.Label(column)
            label.pack()
            preview_labels.append(label)

        thumb_image = rgba_array_to_pil(source)
        thumb_image.thumbnail((250, 220), Image.Resampling.LANCZOS)
        thumb = np.array(thumb_image.convert("RGBA"), dtype=np.uint8)

        def photo_for(arr: np.ndarray) -> ImageTk.PhotoImage:
            image = rgba_array_to_pil(arr)
            canvas = Image.new("RGBA", (250, 220), (44, 46, 52, 255))
            canvas.alpha_composite(image, ((250 - image.width) // 2, (220 - image.height) // 2))
            return ImageTk.PhotoImage(canvas)

        controls = ttk.Frame(dialog)
        controls.pack(fill=tk.X, padx=16)
        value_labels: dict[str, ttk.Label] = {}
        specs = [
            ("smoothing", "Сглаживание кожи", 0.0, 1.0),
            ("texture", "Сохранение текстуры", 0.0, 1.5),
            ("even_tone", "Выравнивание тона", 0.0, 1.0),
            ("redness", "Уменьшение покраснений", 0.0, 1.0),
        ]
        for row, (key, title, start, end) in enumerate(specs):
            ttk.Label(controls, text=title).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Scale(controls, from_=start, to=end, variable=variables[key], orient=tk.HORIZONTAL).grid(row=row, column=1, sticky="ew", padx=8, pady=3)
            value_labels[key] = ttk.Label(controls, width=7)
            value_labels[key].grid(row=row, column=2, sticky="e")
        controls.columnconfigure(1, weight=1)

        original_photo = photo_for(thumb)

        def update_preview(*_args) -> None:
            cleaned = portrait_cleanup(
                thumb,
                variables["smoothing"].get(),
                variables["texture"].get(),
                variables["even_tone"].get(),
                variables["redness"].get(),
            )
            self._portrait_preview_images = [original_photo, photo_for(cleaned)]
            for label, image in zip(preview_labels, self._portrait_preview_images):
                label.configure(image=image)
            for key, label in value_labels.items():
                label.configure(text=f"{variables[key].get():.2f}")

        def accept() -> None:
            nonlocal result
            result = {key: float(variable.get()) for key, variable in variables.items()}
            dialog.destroy()

        buttons = ttk.Frame(dialog)
        buttons.pack(fill=tk.X, padx=12, pady=12)
        ttk.Button(buttons, text="ОК", command=accept).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT)
        ToolTip(preview_labels[1], "Обработка автоматически ограничивается тонами кожи.")
        for variable in variables.values():
            variable.trace_add("write", update_preview)
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        update_preview()
        dialog.wait_window()
        return result

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
