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
        if layer.kind in {"linked", "embedded"}:
            approved = messagebox.askyesno(
                "Растеризация Smart Object",
                "Эта операция изменяет пиксели. Растеризовать Smart Object? Для неразрушающей обработки используйте «Слой -> Фильтры слоя».",
            )
            if not approved:
                return
            def rasterize() -> None:
                layer.kind = "raster"; layer.smart_data = None; layer.smart_source = None
                layer.transform_data = None; layer.transform_source = None; layer.transform_mask_source = None
                self.doc.dirty = True
            self.run_document_command("Растеризация Smart Object", rasterize)
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
