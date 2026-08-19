from __future__ import annotations

from ..app_shared import *


class DocumentsMixin:
    def new_document(self) -> None:
        clipboard_image = self.read_clipboard_image()
        settings = self.new_document_dialog(clipboard_image)
        if settings is None:
            return
        self.create_document_from_settings(settings, clipboard_image)

    def available_document_presets(self, clipboard_image: Image.Image | None = None) -> list[dict[str, object]]:
        presets = [dict(item) for item in DOCUMENT_PRESETS]
        for preset in presets:
            if preset["name"] == "Свой размер":
                preset.update(
                    width=self.custom_canvas_width,
                    height=self.custom_canvas_height,
                    dpi=self.custom_canvas_dpi,
                    background=self.custom_canvas_background,
                )
                break
        if clipboard_image is not None:
            presets.insert(
                0,
                {
                    "name": "Из буфера обмена",
                    "description": "Размер скопированного изображения",
                    "width": clipboard_image.width,
                    "height": clipboard_image.height,
                    "dpi": 72,
                    "background": "Прозрачный",
                    "clipboard": True,
                },
            )
        return presets

    def new_document_dialog(self, clipboard_image: Image.Image | None = None) -> dict[str, object] | None:
        presets = self.available_document_presets(clipboard_image)
        dialog = tk.Toplevel(self)
        dialog.title("Новый холст")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)
        result: dict[str, object] | None = None

        width = tk.IntVar(value=int(presets[0]["width"]))
        height = tk.IntVar(value=int(presets[0]["height"]))
        dpi = tk.IntVar(value=int(presets[0]["dpi"]))
        background = tk.StringVar(value=str(presets[0]["background"]))
        include_clipboard = tk.BooleanVar(value=bool(presets[0].get("clipboard", False)))

        body = ttk.Frame(dialog)
        body.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)
        preset_panel = ttk.Frame(body, width=320)
        preset_panel.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 16))
        ttk.Label(preset_panel, text="Форматы", font=("Segoe UI Semibold", 12)).pack(anchor=tk.W, pady=(0, 6))
        preset_list_area = ttk.Frame(preset_panel)
        preset_list_area.pack(fill=tk.BOTH, expand=True)
        preset_row_height = 52
        preset_list_width = 306
        preset_list = tk.Canvas(
            preset_list_area,
            width=preset_list_width,
            height=572,
            background=TOKENS.SURFACE,
            highlightbackground=TOKENS.BORDER,
            highlightthickness=1,
            cursor="hand2",
            takefocus=True,
        )
        preset_scroll = ttk.Scrollbar(preset_list_area, orient=tk.VERTICAL, command=preset_list.yview)
        preset_list.configure(yscrollcommand=preset_scroll.set)
        preset_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        preset_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._new_document_preset_canvas = preset_list
        preset_items: list[tuple[int, int, int]] = []
        for index, preset in enumerate(presets):
            y1 = index * preset_row_height
            tag = f"preset-{index}"
            rectangle = preset_list.create_rectangle(
                0,
                y1,
                preset_list_width,
                y1 + preset_row_height,
                fill=TOKENS.SURFACE,
                outline=TOKENS.SEPARATOR,
                tags=(tag, "preset"),
            )
            name_item = preset_list.create_text(
                12,
                y1 + 16,
                text=str(preset["name"]),
                anchor=tk.W,
                fill=TOKENS.TEXT_PRIMARY,
                font=("Segoe UI Semibold", 10),
                tags=(tag, "preset"),
            )
            size_item = preset_list.create_text(
                12,
                y1 + 36,
                text=f"Размер холста: {preset['width']} x {preset['height']} px",
                anchor=tk.W,
                fill=TOKENS.TEXT_SECONDARY,
                font=("Segoe UI", 9),
                tags=(tag, "preset"),
            )
            preset_items.append((rectangle, name_item, size_item))
            preset_list.tag_bind(tag, "<Button-1>", lambda _event, selected=index: select_preset(selected))
        preset_list.configure(scrollregion=(0, 0, preset_list_width, len(presets) * preset_row_height))

        right_panel = ttk.Frame(body)
        right_panel.grid(row=0, column=1, rowspan=2, sticky="new")
        preview_panel = ttk.Frame(right_panel)
        preview_panel.pack(fill=tk.X)
        ttk.Label(preview_panel, text="Предпросмотр", font=("Segoe UI Semibold", 12)).pack(anchor=tk.W, pady=(0, 6))
        current_size_label = ttk.Label(preview_panel, text="", font=("Segoe UI Semibold", 14))
        current_size_label.pack(anchor=tk.W, pady=(0, 7))
        self._new_document_size_label = current_size_label
        preview = tk.Canvas(preview_panel, width=330, height=210, background=TOKENS.WORKSPACE, highlightthickness=1, highlightbackground=TOKENS.BORDER)
        preview.pack()
        preset_description = ttk.Label(preview_panel, text="", wraplength=330, justify=tk.LEFT)
        preset_description.pack(fill=tk.X, pady=(7, 0))

        settings = ttk.LabelFrame(right_panel, text="Параметры")
        settings.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(settings, text="Ширина, px").grid(row=0, column=0, sticky="w", padx=10, pady=(10, 4))
        width_entry = ttk.Spinbox(settings, textvariable=width, from_=1, to=50000, width=12)
        width_entry.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=(10, 4))
        ttk.Label(settings, text="Высота, px").grid(row=1, column=0, sticky="w", padx=10, pady=4)
        height_entry = ttk.Spinbox(settings, textvariable=height, from_=1, to=50000, width=12)
        height_entry.grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=4)
        swap_button = ttk.Button(settings, text="Поменять ориентацию")
        swap_button.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=4)
        ttk.Label(settings, text="DPI").grid(row=3, column=0, sticky="w", padx=10, pady=4)
        ttk.Spinbox(settings, textvariable=dpi, from_=1, to=2400, width=12).grid(row=3, column=1, sticky="ew", padx=(0, 10), pady=4)
        ttk.Label(settings, text="Фон").grid(row=4, column=0, sticky="w", padx=10, pady=4)
        ttk.Combobox(settings, textvariable=background, values=list(DOCUMENT_BACKGROUNDS), state="readonly", width=16).grid(row=4, column=1, sticky="ew", padx=(0, 10), pady=4)
        clipboard_check = ttk.Checkbutton(settings, text="Добавить изображение из буфера", variable=include_clipboard)
        if clipboard_image is not None:
            clipboard_check.grid(row=5, column=0, columnspan=2, sticky="w", padx=10, pady=(5, 10))
        settings.columnconfigure(1, weight=1)

        def safe_dimension(variable: tk.IntVar, fallback: int) -> int:
            try:
                return max(1, min(50000, int(variable.get())))
            except (tk.TclError, ValueError):
                return fallback

        def update_preview(*_args) -> None:
            canvas_width = safe_dimension(width, 1280)
            canvas_height = safe_dimension(height, 900)
            current_size_label.configure(text=f"{canvas_width} x {canvas_height} px")
            preview.delete("all")
            margin = 16
            scale = min((330 - margin * 2) / canvas_width, (210 - margin * 2) / canvas_height)
            shown_width = max(1, round(canvas_width * scale))
            shown_height = max(1, round(canvas_height * scale))
            x1 = (330 - shown_width) // 2
            y1 = (210 - shown_height) // 2
            x2, y2 = x1 + shown_width, y1 + shown_height
            if background.get() == "Прозрачный":
                tile = max(5, min(12, shown_width // 12 if shown_width else 5))
                for py in range(y1, y2, tile):
                    for px in range(x1, x2, tile):
                        color = "#d4d6d8" if ((px - x1) // tile + (py - y1) // tile) % 2 == 0 else "#ffffff"
                        preview.create_rectangle(px, py, min(px + tile, x2), min(py + tile, y2), fill=color, outline="")
            else:
                fill = "#ffffff" if background.get() == "Белый" else "#050505"
                preview.create_rectangle(x1, y1, x2, y2, fill=fill, outline="")
            if clipboard_image is not None and include_clipboard.get():
                image = clipboard_image.copy()
                image.thumbnail((shown_width, shown_height), Image.Resampling.LANCZOS)
                self._new_document_preview = ImageTk.PhotoImage(image)
                preview.create_image((x1 + x2) // 2, (y1 + y2) // 2, image=self._new_document_preview)
            preview.create_rectangle(x1, y1, x2, y2, outline=TOKENS.BORDER_ACTIVE, width=1)

        selected_preset_index = 0

        def paint_preset_selection() -> None:
            for index, (rectangle, name_item, size_item) in enumerate(preset_items):
                selected = index == selected_preset_index
                preset_list.itemconfigure(rectangle, fill=TOKENS.SURFACE_SELECTED if selected else TOKENS.SURFACE)
                preset_list.itemconfigure(rectangle, outline=TOKENS.ACCENT if selected else TOKENS.SEPARATOR)
                preset_list.itemconfigure(name_item, fill=TOKENS.TEXT_PRIMARY)
                preset_list.itemconfigure(size_item, fill=TOKENS.ACCENT_HOVER if selected else TOKENS.TEXT_SECONDARY)

        def select_preset(index: int) -> None:
            nonlocal selected_preset_index
            selected_preset_index = max(0, min(len(presets) - 1, int(index)))
            preset = presets[selected_preset_index]
            width.set(int(preset["width"]))
            height.set(int(preset["height"]))
            dpi.set(int(preset["dpi"]))
            background.set(str(preset["background"]))
            include_clipboard.set(bool(preset.get("clipboard", False)))
            preset_description.configure(text=str(preset["description"]))
            paint_preset_selection()
            row_top = selected_preset_index * preset_row_height
            row_bottom = row_top + preset_row_height
            visible_top = preset_list.canvasy(0)
            visible_bottom = visible_top + preset_list.winfo_height()
            total_height = max(1, len(presets) * preset_row_height)
            if preset_list.winfo_height() > preset_row_height:
                if row_top < visible_top:
                    preset_list.yview_moveto(row_top / total_height)
                elif row_bottom > visible_bottom:
                    preset_list.yview_moveto(max(0.0, (row_bottom - preset_list.winfo_height()) / total_height))
            update_preview()

        def move_preset_selection(delta: int) -> str:
            select_preset(selected_preset_index + delta)
            return "break"

        def scroll_presets(event) -> str:
            preset_list.yview_scroll(-1 if event.delta > 0 else 1, "units")
            return "break"

        def swap_orientation() -> None:
            old_width = safe_dimension(width, 1280)
            old_height = safe_dimension(height, 900)
            width.set(old_height)
            height.set(old_width)
            update_preview()

        def accept() -> None:
            nonlocal result
            try:
                canvas_width = int(width.get())
                canvas_height = int(height.get())
                canvas_dpi = int(dpi.get())
            except (tk.TclError, ValueError):
                messagebox.showerror("Новый холст", "Укажите целые числа для размера и DPI.", parent=dialog)
                return
            if not (1 <= canvas_width <= 50000 and 1 <= canvas_height <= 50000 and 1 <= canvas_dpi <= 2400):
                messagebox.showerror("Новый холст", "Размер должен быть от 1 до 50000 px, DPI - от 1 до 2400.", parent=dialog)
                return
            result = {
                "width": canvas_width,
                "height": canvas_height,
                "dpi": canvas_dpi,
                "background": DOCUMENT_BACKGROUNDS.get(background.get(), DOCUMENT_BACKGROUNDS["Белый"]),
                "include_clipboard": bool(include_clipboard.get() and clipboard_image is not None),
            }
            selected_preset = presets[selected_preset_index]
            uses_custom_size = selected_preset["name"] == "Свой размер"
            uses_custom_size = uses_custom_size or canvas_width != int(selected_preset["width"]) or canvas_height != int(selected_preset["height"])
            if uses_custom_size:
                self.remember_custom_canvas(canvas_width, canvas_height, canvas_dpi, background.get())
            dialog.destroy()

        def accept_preset(index: int) -> str:
            select_preset(index)
            accept()
            return "break"

        swap_button.configure(command=swap_orientation)
        for index in range(len(presets)):
            preset_list.tag_bind(
                f"preset-{index}",
                "<Double-Button-1>",
                lambda _event, selected=index: accept_preset(selected),
            )
        self._new_document_accept_preset = accept_preset
        preset_list.bind("<Up>", lambda _event: move_preset_selection(-1))
        preset_list.bind("<Down>", lambda _event: move_preset_selection(1))
        preset_list.bind("<MouseWheel>", scroll_presets)
        for variable in (width, height, background, include_clipboard):
            variable.trace_add("write", update_preview)
        buttons = ttk.Frame(dialog)
        buttons.pack(fill=tk.X, padx=14, pady=(0, 14))
        ttk.Button(buttons, text="Создать", command=accept).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT)
        ToolTip(preset_list, "Выберите готовый формат, затем при необходимости измените его параметры.")
        select_preset(0)
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        self.center_toplevel(dialog, 820, 680)
        preset_list.yview_moveto(0.0)
        dialog.wait_window()
        return result

    def remember_custom_canvas(self, width: int, height: int, dpi: int, background: str) -> None:
        self.custom_canvas_width = max(1, min(50000, int(width)))
        self.custom_canvas_height = max(1, min(50000, int(height)))
        self.custom_canvas_dpi = max(1, min(2400, int(dpi)))
        self.custom_canvas_background = background if background in DOCUMENT_BACKGROUNDS else "Белый"
        self.save_settings()

    def create_document_from_settings(self, settings: dict[str, object], clipboard_image: Image.Image | None = None) -> None:
        width = int(settings["width"])
        height = int(settings["height"])
        dpi = int(settings["dpi"])
        background = tuple(settings["background"])
        include_clipboard = bool(settings.get("include_clipboard", False) and clipboard_image is not None)
        self.doc = Document.new(width, height, background)
        self._edit_generation += 1
        self.doc.dpi = dpi
        self.doc.metadata = {"source": "clipboard" if include_clipboard else "new document", "preset_size": [width, height]}
        if include_clipboard and clipboard_image is not None:
            canvas = Image.new("RGBA", (width, height), background)
            image = clipboard_image.convert("RGBA").copy()
            if image.width > width or image.height > height:
                image.thumbnail((width, height), Image.Resampling.LANCZOS)
            canvas.alpha_composite(image, ((width - image.width) // 2, (height - image.height) // 2))
            self.doc.layer.pixels = np.array(canvas, dtype=np.uint8)
            self.doc.layer.name = "Из буфера обмена"
        self.history.clear()
        self.selection_box = None
        self.show_editor()

    def create_from_clipboard(self) -> None:
        image = self.read_clipboard_image() or getattr(self, "_startup_clipboard_image", None)
        if image is None:
            messagebox.showinfo("Буфер обмена", "В буфере больше нет изображения.")
            return
        self.create_document_from_settings(
            {"width": image.width, "height": image.height, "dpi": 72, "background": DOCUMENT_BACKGROUNDS["Прозрачный"], "include_clipboard": True},
            image,
        )

    def new_from_preset(self) -> None:
        self.new_document()
