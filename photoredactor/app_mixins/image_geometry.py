from __future__ import annotations

from ..app_shared import *


class ImageGeometryMixin:
    def resize_image(self) -> None:
        width = simpledialog.askinteger("Resize image", "Width px:", initialvalue=self.doc.width, minvalue=1, maxvalue=100000)
        height = simpledialog.askinteger("Resize image", "Height px:", initialvalue=self.doc.height, minvalue=1, maxvalue=100000)
        if width and height:
            self.run_document_command("Resize image", lambda: self.doc.resize_image(width, height))
            self.refresh()

    def resize_canvas(self) -> None:
        width = simpledialog.askinteger("Resize canvas", "Width px:", initialvalue=self.doc.width, minvalue=1, maxvalue=100000)
        height = simpledialog.askinteger("Resize canvas", "Height px:", initialvalue=self.doc.height, minvalue=1, maxvalue=100000)
        if width and height:
            self.run_document_command("Resize canvas", lambda: self.doc.resize_canvas(width, height))
            self.refresh()

    def generative_expand_dialog(self) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("Генеративное расширение холста")
        dialog.geometry("560x500")
        dialog.transient(self)
        dialog.grab_set()
        values = {name: tk.IntVar(value=max(64, size // 8)) for name, size in (
            ("Слева", self.doc.width), ("Сверху", self.doc.height), ("Справа", self.doc.width), ("Снизу", self.doc.height)
        )}
        method = tk.StringVar(value="content-aware")
        controls = ttk.Frame(dialog, padding=12)
        controls.pack(fill=tk.X)
        for column, (name, variable) in enumerate(values.items()):
            ttk.Label(controls, text=name).grid(row=0, column=column, padx=4, sticky="w")
            ttk.Spinbox(controls, from_=0, to=100000, textvariable=variable, width=10).grid(row=1, column=column, padx=4)
        ttk.Label(controls, text="Заполнение").grid(row=2, column=0, sticky="w", pady=(12, 2))
        ttk.Combobox(
            controls,
            textvariable=method,
            values=("content-aware", "mirror", "edge"),
            state="readonly",
        ).grid(row=3, column=0, columnspan=4, sticky="ew", padx=4)
        preview = ttk.Label(dialog, anchor=tk.CENTER)
        preview.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        def update_preview(*_args) -> None:
            try:
                source = self.render_engine.render(self.doc, False)
                scale = min(1.0, 360 / max(source.shape[0], source.shape[1]))
                small = cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                document_scale = small.shape[1] / self.doc.width
                margins = [max(0, round(values[name].get() * document_scale)) for name in ("Слева", "Сверху", "Справа", "Снизу")]
                result = generative_expand_pixels(small, margins[0], margins[1], margins[2], margins[3], method.get())
                image = rgba_array_to_pil(result)
                image.thumbnail((500, 320), Image.Resampling.LANCZOS)
                self._generative_preview_image = ImageTk.PhotoImage(image)
                preview.configure(image=self._generative_preview_image)
            except (tk.TclError, ValueError):
                pass

        for variable in values.values():
            variable.trace_add("write", update_preview)
        method.trace_add("write", update_preview)

        buttons = ttk.Frame(dialog, padding=(12, 0, 12, 12))
        buttons.pack(fill=tk.X)

        def apply() -> None:
            margins = [values[name].get() for name in ("Слева", "Сверху", "Справа", "Снизу")]
            if not any(value > 0 for value in margins):
                messagebox.showinfo("Расширение", "Укажите размер расширения хотя бы с одной стороны.", parent=dialog)
                return
            dialog.destroy()
            self.run_document_command(
                "Генеративное расширение",
                lambda: self.doc.generative_expand(*margins, method.get()),
            )
            self.refresh()

        ttk.Button(buttons, text="Расширить", command=apply).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT, padx=6)
        update_preview()

    def change_bit_depth(self, bit_depth: int) -> None:
        self.run_document_command(f"Глубина каналов {bit_depth} бит", lambda: self.doc.set_bit_depth(bit_depth))
        self.refresh()

    def change_color_model(self, color_model: str) -> None:
        self.run_document_command(f"Цветовая модель {color_model}", lambda: self.doc.set_color_model(color_model))
        self.refresh()

    def _choose_icc_profile(self, title: str) -> str | None:
        use_srgb = messagebox.askyesnocancel(title, "Использовать стандартный профиль sRGB?\n\nДа: sRGB\nНет: выбрать ICC-файл")
        if use_srgb is None:
            return None
        if use_srgb:
            return "sRGB"
        return filedialog.askopenfilename(title=title, filetypes=[("ICC-профили", "*.icc *.icm"), ("Все файлы", "*.*")]) or None

    def assign_icc_profile(self) -> None:
        profile = self._choose_icc_profile("Назначить ICC-профиль")
        if profile:
            self.run_document_command("Назначить ICC-профиль", lambda: self.doc.assign_color_profile(profile))
            self.refresh()

    def convert_icc_profile(self) -> None:
        profile = self._choose_icc_profile("Преобразовать в ICC-профиль")
        if not profile:
            return
        try:
            self.run_document_command("Преобразовать ICC-профиль", lambda: self.doc.convert_color_profile(profile))
            self.refresh()
        except Exception as exc:
            messagebox.showerror("Управление цветом", str(exc))

    def crop_to_selection(self) -> None:
        crop_box = self.doc.selection_bounds() or self.selection_box
        if not crop_box:
            messagebox.showinfo("Crop", "Create a rectangular selection first.")
            return
        self.run_document_command("Crop", lambda: self.doc.crop(crop_box))
        self.doc.clear_selection()
        self.selection_box = None
        self.update_selection_overlay()
        self.refresh()

    def trim_transparent(self) -> None:
        self.run_document_command("Trim transparent pixels", self.doc.trim_transparent)
        self.selection_box = self.doc.selection_bounds()
        self.refresh()

    def reveal_all(self) -> None:
        self.run_document_command("Reveal all layers", self.doc.reveal_all)
        self.selection_box = self.doc.selection_bounds()
        self.refresh()

    def rotate(self, angle: int) -> None:
        def edit():
            old_w, old_h = self.doc.width, self.doc.height
            for layer in self.doc.layers:
                lx, ly, lw, lh = layer.x, layer.y, layer.pixels.shape[1], layer.pixels.shape[0]
                if layer.kind in {"linked", "embedded"} and layer.smart_data is not None:
                    transform = dict(layer.smart_data.get("transform") or {})
                    transform["angle"] = float(transform.get("angle", 0.0)) + angle
                    layer.smart_data = {**layer.smart_data, "transform": transform}
                if angle == 90:
                    pixels = layer.working_rgba() if layer.working_pixels is not None else layer.pixels
                    layer.replace_pixels(cv2.rotate(pixels, cv2.ROTATE_90_CLOCKWISE))
                    if layer.mask is not None:
                        layer.mask = cv2.rotate(layer.mask, cv2.ROTATE_90_CLOCKWISE)
                    layer.x = old_h - (ly + lh)
                    layer.y = lx
                elif angle == 180:
                    pixels = layer.working_rgba() if layer.working_pixels is not None else layer.pixels
                    layer.replace_pixels(cv2.rotate(pixels, cv2.ROTATE_180))
                    if layer.mask is not None:
                        layer.mask = cv2.rotate(layer.mask, cv2.ROTATE_180)
                    layer.x = old_w - (lx + lw)
                    layer.y = old_h - (ly + lh)
                if angle not in {90, 180}:
                    layer.touch_pixels()
            if self.doc.selection_mask is not None:
                if angle == 90:
                    self.doc.selection_mask = cv2.rotate(self.doc.selection_mask, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    self.doc.selection_mask = cv2.rotate(self.doc.selection_mask, cv2.ROTATE_180)
            for name, mask in list(self.doc.saved_selections.items()):
                if angle == 90:
                    self.doc.saved_selections[name] = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    self.doc.saved_selections[name] = cv2.rotate(mask, cv2.ROTATE_180)
            if angle in [90, 270]:
                self.doc.width, self.doc.height = self.doc.height, self.doc.width
            self.doc.dirty = True

        self.run_document_command(f"Rotate {angle}", edit)
        self.refresh()

    def flip(self, horizontal: bool) -> None:
        def edit():
            code = 1 if horizontal else 0
            for layer in self.doc.layers:
                pixels = layer.working_rgba() if layer.working_pixels is not None else layer.pixels
                layer.replace_pixels(cv2.flip(pixels, code))
                if layer.kind in {"linked", "embedded"} and layer.smart_data is not None:
                    transform = dict(layer.smart_data.get("transform") or {})
                    key = "flip_horizontal" if horizontal else "flip_vertical"
                    transform[key] = not bool(transform.get(key, False))
                    layer.smart_data = {**layer.smart_data, "transform": transform}
                if layer.mask is not None:
                    layer.mask = cv2.flip(layer.mask, code)
                if horizontal:
                    layer.x = self.doc.width - (layer.x + layer.pixels.shape[1])
                else:
                    layer.y = self.doc.height - (layer.y + layer.pixels.shape[0])
            if self.doc.selection_mask is not None:
                self.doc.selection_mask = cv2.flip(self.doc.selection_mask, code)
            for name, mask in list(self.doc.saved_selections.items()):
                self.doc.saved_selections[name] = cv2.flip(mask, code)
            self.doc.dirty = True

        self.run_document_command("Flip horizontal" if horizontal else "Flip vertical", edit)
        self.refresh()

    def apply_to_layer(self, label: str, fn) -> None:
        layer = self.doc.layer
        if layer.locked:
            self.status_text("Слой заблокирован")
            return
        layer_id = layer.id
        generation = self._edit_generation
        pixels_revision = layer.pixels_revision
        before = layer.pixels.copy()
        before_working = layer.working_rgba().copy() if layer.working_pixels is not None else None
        selection_mask = self.doc.layer_selection_mask(layer)
        rect = (0, 0, before.shape[1], before.shape[0])

        def worker():
            source = before.copy() if before_working is None else before_working.copy()
            after = fn(source)
            if selection_mask is not None:
                alpha = (selection_mask.astype(np.float32) / 255.0)[:, :, None]
                if before_working is None:
                    after = (after.astype(np.float32) * alpha + before.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
                else:
                    after = normalize_rgba(after) * alpha + before_working * (1.0 - alpha)
            return after

        def done(after):
            target = self.doc.get_layer(layer_id)
            if target is None:
                return
            if before_working is None:
                target.pixels = display_rgba(after)
                target.touch_pixels()
                after_working = None
            else:
                after_working = normalize_rgba(after)
                target.set_working_rgba(after_working, self.doc.bit_depth, target.working_model)
            self.doc.dirty = True
            self.push_command(PixelPatchCommand(
                label,
                layer_id,
                rect,
                before,
                target.pixels.copy(),
                before_working,
                None if before_working is None else after_working.copy(),
            ))
            self.invalidate_pixels()
            self.refresh()

        self.run_background(
            label,
            worker,
            done,
            lambda: self._edit_generation == generation
            and (target := self.doc.get_layer(layer_id)) is not None
            and target.pixels_revision == pixels_revision,
        )
