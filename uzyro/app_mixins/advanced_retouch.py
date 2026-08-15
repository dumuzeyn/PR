from __future__ import annotations

from ..app_shared import *


class AdvancedRetouchMixin:
    def filter_red_eye(self) -> None:
        layer = self.doc.layer
        if layer.locked or layer.kind == "adjustment":
            messagebox.showinfo("Удаление красных глаз", "Выберите незаблокированный слой с изображением.")
            return
        selection_mask = self.doc.layer_selection_mask(layer)
        settings = self.red_eye_dialog(layer.pixels, selection_mask)
        if settings is None:
            return
        self.apply_to_layer(
            "Удаление красных глаз",
            lambda arr: reduce_red_eye(
                arr,
                selection_mask,
                settings["strength"],
                settings["threshold"],
                settings["darken"],
                settings["feather"],
            ),
        )

    def red_eye_dialog(self, source: np.ndarray, selection_mask: np.ndarray | None) -> dict[str, float] | None:
        dialog = tk.Toplevel(self)
        dialog.title("Удаление красных глаз")
        dialog.transient(self)
        dialog.resizable(False, False)
        dialog.grab_set()
        result: dict[str, float] | None = None
        variables = {
            "strength": tk.DoubleVar(value=0.85),
            "threshold": tk.DoubleVar(value=0.35),
            "darken": tk.DoubleVar(value=0.18),
            "feather": tk.DoubleVar(value=2.0),
        }

        preview_row = ttk.Frame(dialog)
        preview_row.pack(fill=tk.X, padx=12, pady=(12, 8))
        preview_labels: list[ttk.Label] = []
        for title in ("До", "После"):
            column = ttk.Frame(preview_row)
            column.pack(side=tk.LEFT, padx=5)
            ttk.Label(column, text=title).pack(pady=(0, 4))
            label = ttk.Label(column)
            label.pack()
            preview_labels.append(label)

        thumb_image = rgba_array_to_pil(source)
        thumb_image.thumbnail((270, 220), Image.Resampling.LANCZOS)
        thumb = np.array(thumb_image.convert("RGBA"), dtype=np.uint8)
        thumb_mask = None
        if selection_mask is not None:
            mask_image = Image.fromarray(np.asarray(selection_mask, dtype=np.uint8), mode="L")
            mask_image = mask_image.resize(thumb_image.size, Image.Resampling.LANCZOS)
            thumb_mask = np.array(mask_image, dtype=np.uint8)

        def photo_for(arr: np.ndarray) -> ImageTk.PhotoImage:
            image = rgba_array_to_pil(arr)
            canvas = Image.new("RGBA", (270, 220), (44, 46, 52, 255))
            canvas.alpha_composite(image, ((270 - image.width) // 2, (220 - image.height) // 2))
            return ImageTk.PhotoImage(canvas)

        controls = ttk.Frame(dialog)
        controls.pack(fill=tk.X, padx=16)
        value_labels: dict[str, ttk.Label] = {}
        specs = (
            ("strength", "Сила коррекции", 0.0, 1.0, 2),
            ("threshold", "Порог красного", 0.0, 1.0, 2),
            ("darken", "Затемнение зрачка", 0.0, 0.8, 2),
            ("feather", "Мягкость края", 0.0, 8.0, 1),
        )
        for row, (key, title, start, end, _digits) in enumerate(specs):
            ttk.Label(controls, text=title).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Scale(controls, from_=start, to=end, variable=variables[key], orient=tk.HORIZONTAL).grid(
                row=row, column=1, sticky="ew", padx=8, pady=3
            )
            value_labels[key] = ttk.Label(controls, width=7)
            value_labels[key].grid(row=row, column=2, sticky="e")
        controls.columnconfigure(1, weight=1)

        selection_note = (
            "Коррекция ограничена текущим выделением."
            if selection_mask is not None and np.any(selection_mask)
            else "Для точной коррекции сначала выделите область глаз."
        )
        ttk.Label(dialog, text=selection_note, style="Secondary.TLabel").pack(anchor="w", padx=16, pady=(8, 0))
        original_photo = photo_for(thumb)

        def update_preview(*_args) -> None:
            corrected = reduce_red_eye(
                thumb,
                thumb_mask,
                variables["strength"].get(),
                variables["threshold"].get(),
                variables["darken"].get(),
                variables["feather"].get() * max(0.35, thumb.shape[1] / max(1, source.shape[1])),
            )
            self._red_eye_preview_images = [original_photo, photo_for(corrected)]
            for label, image in zip(preview_labels, self._red_eye_preview_images):
                label.configure(image=image)
            for key, _title, _start, _end, digits in specs:
                value_labels[key].configure(text=f"{variables[key].get():.{digits}f}")

        def accept() -> None:
            nonlocal result
            result = {key: float(variable.get()) for key, variable in variables.items()}
            dialog.destroy()

        buttons = ttk.Frame(dialog)
        buttons.pack(fill=tk.X, padx=12, pady=12)
        ttk.Button(buttons, text="Применить", command=accept, style="Primary.TButton").pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT)
        for variable in variables.values():
            variable.trace_add("write", update_preview)
        self._red_eye_variables = variables
        self._red_eye_accept = accept
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        self.center_toplevel(dialog, 600, 460)
        update_preview()
        dialog.wait_window()
        return result

    def filter_patch_selection(self) -> None:
        if self.doc.selection_mask is None or not np.any(self.doc.selection_mask):
            messagebox.showinfo("Заплатка", "Сначала выделите дефект, который нужно заменить.")
            return
        self.tool.set("patch")
        if hasattr(self, "tool_options_panel"):
            self.tool_options_panel.render()
        self.status_text("Заплатка: перетащите выделенную область на подходящий источник текстуры")
