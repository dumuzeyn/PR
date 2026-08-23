from __future__ import annotations

from ..app_shared import *
from ..selection_color import combine_sample_masks


class ColorRangeMixin:
    def color_range_dialog(self) -> None:
        layer = self.doc.layer
        sample_all = bool(self.color_range_sample_all_layers.get())
        source = self.doc.composite(False) if sample_all else layer.pixels
        origin = (0, 0) if sample_all else (layer.x, layer.y)
        if source.size == 0:
            return
        dialog = tk.Toplevel(self)
        dialog.title("Цветовой диапазон")
        dialog.transient(self)
        dialog.grab_set()
        dialog.minsize(820, 580)

        fuzziness = tk.IntVar(value=int(self.tolerance.get()))
        sample_mode = tk.StringVar(value="Пипетка")
        preview_mode = tk.StringVar(value=SELECT_MASK_PREVIEW_OVERLAY)
        antialias = tk.BooleanVar(value=bool(self.selection_antialias.get()))
        included: list[tuple[int, int, int, int]] = []
        excluded: list[tuple[int, int, int, int]] = []
        transform = {"scale": 1.0, "left": 0.0, "top": 0.0}
        preview_after: list[str | None] = [None]

        center_x = min(source.shape[1] - 1, max(0, source.shape[1] // 2))
        center_y = min(source.shape[0] - 1, max(0, source.shape[0] // 2))
        included.append(tuple(int(value) for value in source[center_y, center_x]))

        header = ttk.Frame(dialog, padding=(12, 10, 12, 6))
        header.pack(fill=tk.X)
        ttk.Label(header, text="Цветовой диапазон", style="PanelTitle.TLabel").pack(side=tk.LEFT)
        stats = ttk.Label(header, style="Secondary.TLabel")
        stats.pack(side=tk.RIGHT)
        body = ttk.PanedWindow(dialog, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=12)
        preview = tk.Canvas(body, width=560, height=500, background=TOKENS.WORKSPACE, highlightthickness=1, highlightbackground=TOKENS.BORDER, cursor="crosshair")
        controls = ttk.Frame(body, width=245, padding=(12, 2))
        body.add(preview, weight=1)
        body.add(controls, weight=0)
        footer = ttk.Frame(dialog, padding=12)
        footer.pack(fill=tk.X)

        ttk.Label(controls, text="Образцы", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(3, 4))
        modes = ttk.Frame(controls)
        modes.pack(fill=tk.X)
        for label in ("Пипетка", "Добавить", "Вычесть"):
            ttk.Radiobutton(modes, text=label, value=label, variable=sample_mode).pack(side=tk.LEFT, padx=(0, 5))
        sample_info = ttk.Label(controls, text="", wraplength=225, justify=tk.LEFT, style="Secondary.TLabel")
        sample_info.pack(fill=tk.X, pady=(7, 9))
        ttk.Label(controls, text="Разброс", style="PanelTitle.TLabel").pack(anchor=tk.W)
        fuzz_value = ttk.Label(controls, style="Secondary.TLabel")
        fuzz_value.pack(anchor=tk.E)
        AccentScale(controls, variable=fuzziness, from_=0, to=128).pack(fill=tk.X)
        ttk.Checkbutton(controls, text="Сглаживание", variable=antialias).pack(anchor=tk.W, pady=(10, 2))
        ttk.Label(controls, text="Просмотр", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(12, 3))
        ttk.Combobox(controls, textvariable=preview_mode, values=SELECT_MASK_PREVIEW_MODES, state="readonly").pack(fill=tk.X)
        ttk.Button(controls, text="Сбросить образцы", command=lambda: reset_samples()).pack(fill=tk.X, pady=(14, 0))

        def source_mask() -> np.ndarray:
            return combine_sample_masks(source, included, excluded, int(fuzziness.get()), bool(antialias.get()))

        def document_mask() -> np.ndarray:
            local = source_mask()
            if sample_all:
                return local
            result = np.zeros((self.doc.height, self.doc.width), dtype=np.uint8)
            paste_mask(result, local, origin[0], origin[1])
            return result

        def update_preview(*_args) -> None:
            preview_after[0] = None
            try:
                mask = document_mask()
            except (tk.TclError, ValueError):
                return
            preview.update_idletasks()
            available = max(300, min(preview.winfo_width(), preview.winfo_height()))
            size = min(700, available)
            scale = min(1.0, size / max(1, self.doc.width), size / max(1, self.doc.height))
            width, height = max(1, round(self.doc.width * scale)), max(1, round(self.doc.height * scale))
            composite = self.render_engine.render(self.doc, checker=False)
            reduced = composite if (width, height) == (self.doc.width, self.doc.height) else cv2.resize(composite, (width, height), interpolation=cv2.INTER_AREA)
            reduced_mask = mask if (width, height) == (self.doc.width, self.doc.height) else cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)
            image = self.render_select_mask_preview(reduced, reduced_mask, preview_mode.get(), size)
            self._color_range_preview_image = ImageTk.PhotoImage(image)
            preview.delete("preview")
            preview.create_image(preview.winfo_width() / 2, preview.winfo_height() / 2, image=self._color_range_preview_image, tags="preview")
            transform.update({"scale": scale, "left": (preview.winfo_width() - width) / 2, "top": (preview.winfo_height() - height) / 2})
            selected = int(np.count_nonzero(mask))
            soft = int(np.count_nonzero((mask > 0) & (mask < 255)))
            stats.configure(text=f"Выбрано: {selected} px  |  Мягких: {soft} px")
            fuzz_value.configure(text=str(int(fuzziness.get())))
            sample_info.configure(text=f"Добавлено образцов: {len(included)}\nВычтено образцов: {len(excluded)}")

        def schedule_preview(*_args) -> None:
            if preview_after[0] is not None:
                try:
                    dialog.after_cancel(preview_after[0])
                except tk.TclError:
                    pass
            preview_after[0] = dialog.after(60, update_preview)

        def preview_point(event) -> tuple[int, int] | None:
            scale = float(transform["scale"])
            x = round((event.x - float(transform["left"])) / max(scale, 1e-8))
            y = round((event.y - float(transform["top"])) / max(scale, 1e-8))
            if 0 <= x < self.doc.width and 0 <= y < self.doc.height:
                return x, y
            return None

        def add_sample(event) -> None:
            point = preview_point(event)
            if point is None:
                return
            lx, ly = point[0] - origin[0], point[1] - origin[1]
            if lx < 0 or ly < 0 or lx >= source.shape[1] or ly >= source.shape[0]:
                return
            value = tuple(int(channel) for channel in source[ly, lx])
            mode = "Добавить" if event.state & 0x0001 else "Вычесть" if event.state & 0x0004 else sample_mode.get()
            if mode == "Пипетка":
                included[:] = [value]
                excluded.clear()
            elif mode == "Добавить" and value not in included:
                included.append(value)
            elif mode == "Вычесть" and value not in excluded:
                excluded.append(value)
            self.color_range_sample_hex.set(self.color_hex(value).upper())
            update_preview()

        def reset_samples() -> None:
            included[:] = [tuple(int(value) for value in source[center_y, center_x])]
            excluded.clear()
            update_preview()

        def accept() -> None:
            mask = document_mask()
            self.tolerance.set(int(fuzziness.get()))
            self.selection_antialias.set(bool(antialias.get()))
            self.run_selection_command("Цветовой диапазон", lambda: self.doc.apply_selection_mask(mask, self.selection_mode.get()))
            dialog.destroy()

        ttk.Label(footer, text="Клик: новый образец  |  Shift: добавить  |  Ctrl: вычесть", style="Secondary.TLabel").pack(side=tk.LEFT)
        ttk.Button(footer, text="Применить", command=accept, style="Primary.TButton").pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(footer, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT)
        preview.bind("<Button-1>", add_sample)
        preview.bind("<Configure>", schedule_preview)
        for variable in (fuzziness, preview_mode, antialias):
            variable.trace_add("write", schedule_preview)
        self._color_range_preview = preview
        self._color_range_sample_mode = sample_mode
        self._color_range_accept = accept
        self.center_toplevel(dialog, 900, 650)
        update_preview()
