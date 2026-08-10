from __future__ import annotations

from ..app_shared import *


class PluginsViewMixin:
    def refresh_plugin_filter_menu(self) -> None:
        self.plugin_filters_menu.delete(0, tk.END)
        if not self.plugin_registry.filters:
            self.plugin_filters_menu.add_command(label="Нет доступных фильтров", state=tk.DISABLED)
            return
        for name, plugin in sorted(self.plugin_registry.filters.items()):
            self.plugin_filters_menu.add_command(label=name, command=lambda value=name: self.apply_plugin_filter(value))

    def apply_plugin_filter(self, name: str) -> None:
        raw = simpledialog.askstring("Фильтр-плагин", "Параметры JSON:", initialvalue="{}")
        if raw is None:
            return
        try:
            params = json.loads(raw)
            if not isinstance(params, dict):
                raise ValueError("Параметры должны быть JSON-объектом")
            self.apply_to_layer(name, lambda pixels: self.plugin_registry.apply_filter(name, pixels, params))
        except Exception as exc:
            messagebox.showerror("Фильтр-плагин", str(exc))

    def reload_plugins(self) -> None:
        count = self.plugin_registry.discover()
        for name, callback in self.plugin_registry.action_commands.items():
            if name not in self.action_runner.commands:
                self.action_runner.register(name, callback)
        self.status_text(f"Плагины перезагружены: {count}")
        if self.plugin_registry.errors:
            self.show_plugin_errors()

    def show_plugin_errors(self) -> None:
        if not self.plugin_registry.errors:
            messagebox.showinfo("Плагины", "Ошибок загрузки нет.")
            return
        self.show_text_window("Ошибки плагинов", "\n".join(self.plugin_registry.errors))

    def set_view_channel(self) -> None:
        self.invalidate_view()
        self.refresh_canvas()

    def set_mask_preview(self) -> None:
        self.invalidate_view()
        self.refresh_canvas()

    def set_paint_target(self) -> None:
        if self.paint_target.get() == "mask":
            self.prepare_mask_editing(create_if_missing=True)
        else:
            self.mask_preview.set(MASK_PREVIEW_NORMAL)
            self.set_mask_preview()
            self.status_text("Рисование по пикселям активного слоя")

    def edit_pixels_channel(self, _event=None) -> None:
        self.paint_target.set("pixels")
        self.mask_preview.set(MASK_PREVIEW_NORMAL)
        self.set_mask_preview()
        self.status_text("Рисование по пикселям активного слоя")

    def edit_mask_channel(self, _event=None) -> None:
        self.prepare_mask_editing(create_if_missing=True)

    def edit_active_mask_channel(self) -> None:
        self.prepare_mask_editing(create_if_missing=True)

    def prepare_mask_editing(self, create_if_missing: bool) -> None:
        layer = self.doc.layer
        if layer.mask is None:
            if not create_if_missing:
                self.mask_preview.set(MASK_PREVIEW_NORMAL)
                self.set_mask_preview()
                self.status_text("У активного слоя нет маски")
                return
            self.run_document_command("Add reveal-all mask", self.doc.add_reveal_all_mask)
        self.paint_target.set("mask")
        self.mask_preview.set(MASK_PREVIEW_CHANNEL)
        self.refresh()
        self.status_text("Рисование по маске активного слоя")

    def set_zoom(self, value: float) -> None:
        old_zoom = max(0.0001, float(self.zoom.get()))
        ox, oy = self._canvas_origin
        center_x = self.canvas.canvasx(max(1, self.canvas.winfo_width()) / 2)
        center_y = self.canvas.canvasy(max(1, self.canvas.winfo_height()) / 2)
        doc_center_x = (center_x - ox) / old_zoom
        doc_center_y = (center_y - oy) / old_zoom
        self.zoom.set(max(0.05, min(16.0, value)))
        self.invalidate_view()
        self.refresh_canvas()
        self.center_canvas_on_doc(doc_center_x, doc_center_y)

    def center_canvas_on_doc(self, doc_x: float, doc_y: float) -> None:
        if self._initial_fit_after_id is not None and not self._performing_initial_fit:
            try:
                self.after_cancel(self._initial_fit_after_id)
            except tk.TclError:
                pass
            self._initial_fit_after_id = None
        raw_region = str(self.canvas.cget("scrollregion")).split()
        if len(raw_region) != 4:
            return
        try:
            region = tuple(float(value) for value in raw_region)
        except ValueError:
            return
        target_x, target_y = self.doc_to_canvas(doc_x, doc_y)
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        scroll_w = max(1.0, region[2] - region[0])
        scroll_h = max(1.0, region[3] - region[1])
        left = target_x - width / 2.0
        top = target_y - height / 2.0
        self.canvas.xview_moveto(max(0.0, min(1.0, (left - region[0]) / scroll_w)))
        self.canvas.yview_moveto(max(0.0, min(1.0, (top - region[1]) / scroll_h)))

    def fit_to_screen(self) -> None:
        self.update_idletasks()
        w = max(1, self.canvas.winfo_width() - 20)
        h = max(1, self.canvas.winfo_height() - 20)
        self.set_zoom(min(w / self.doc.width, h / self.doc.height))
        self.center_canvas_on_doc(self.doc.width / 2, self.doc.height / 2)

    def set_grid_spacing(self) -> None:
        spacing = simpledialog.askinteger("Grid", "Spacing px:", initialvalue=int(self.grid_spacing.get()), minvalue=4, maxvalue=5000)
        if spacing:
            self.grid_spacing.set(spacing)
            self.refresh_canvas()

    def add_horizontal_guide(self) -> None:
        y = simpledialog.askinteger("Horizontal guide", "Y coordinate:", initialvalue=self.doc.height // 2, minvalue=0, maxvalue=max(0, self.doc.height))
        if y is not None:
            self._guide_doc_lines.append(("h", y))
            self.refresh_canvas()

    def add_vertical_guide(self) -> None:
        x = simpledialog.askinteger("Vertical guide", "X coordinate:", initialvalue=self.doc.width // 2, minvalue=0, maxvalue=max(0, self.doc.width))
        if x is not None:
            self._guide_doc_lines.append(("v", x))
            self.refresh_canvas()

    def clear_guides(self) -> None:
        self._guide_doc_lines.clear()
        self.refresh_canvas()

    def mouse_wheel(self, event) -> None:
        if event.state & 0x0004:
            self.set_zoom(self.zoom.get() * (1.1 if event.delta > 0 else 0.9))
        elif event.state & 0x0001:
            self.canvas.xview_scroll(-1 if event.delta > 0 else 1, "units")
        else:
            self.canvas.yview_scroll(-3 if event.delta > 0 else 3, "units")

    def batch_process(self) -> None:
        src = filedialog.askdirectory(title="Source folder")
        if not src:
            return
        dst = filedialog.askdirectory(title="Destination folder")
        if not dst:
            return
        width = simpledialog.askinteger("Batch", "Max width px, empty for original:", initialvalue=1920, minvalue=1, maxvalue=50000)

        def worker():
            exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"} | RAW_EXTENSIONS
            count = 0
            for path in Path(src).rglob("*"):
                if path.suffix.lower() not in exts:
                    continue
                doc = Document.from_image(path)
                if width and doc.width > width:
                    doc.resize_image(width, max(1, round(doc.height * width / doc.width)))
                out = Path(dst) / f"{path.stem}.png"
                doc.export_flat(out)
                count += 1
            return count

        self.run_background("Batch", worker, lambda count: messagebox.showinfo("Batch", f"Processed {count} files."))
