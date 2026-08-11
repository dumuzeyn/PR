from __future__ import annotations

from ..app_shared import *


class SourceOverlayMixin:
    def open_clone_source_panel(self) -> None:
        dialog = getattr(self, "_clone_source_dialog", None)
        if dialog is not None and dialog.winfo_exists():
            dialog.deiconify()
            dialog.lift()
            return
        dialog = tk.Toplevel(self)
        self._clone_source_dialog = dialog
        dialog.title("Источник клонирования")
        dialog.geometry("410x540")
        dialog.minsize(380, 500)
        dialog.transient(self)
        body = ttk.Frame(dialog, padding=12)
        body.pack(fill=tk.BOTH, expand=True)
        ttk.Checkbutton(body, text="Выровненный источник", variable=self.clone_aligned).pack(anchor=tk.W)
        ttk.Label(body, text="Образец").pack(anchor=tk.W, pady=(10, 2))
        ttk.Combobox(
            body,
            textvariable=self.clone_sampling,
            values=["Текущий слой", "Текущий и ниже", "Все видимые"],
            state="readonly",
        ).pack(fill=tk.X)
        source = ttk.LabelFrame(body, text="Положение", padding=8)
        source.pack(fill=tk.X, pady=(10, 0))
        self._source_dialog_pair(source, "Источник X", self.clone_source_x, "Y", self.clone_source_y, -50000, 50000)
        self._source_dialog_pair(source, "Смещение X", self.clone_offset_x, "Y", self.clone_offset_y, -50000, 50000)
        transform = ttk.LabelFrame(body, text="Преобразование", padding=8)
        transform.pack(fill=tk.X, pady=(10, 0))
        self._source_dialog_pair(transform, "Масштаб X, %", self.clone_scale_x, "Y, %", self.clone_scale_y, 5, 2000)
        self._source_dialog_pair(transform, "Поворот, °", self.clone_rotation, "", None, -180, 180)
        flip = ttk.Frame(transform)
        flip.pack(fill=tk.X, pady=(5, 0))
        ttk.Checkbutton(flip, text="Отразить X", variable=self.clone_flip_horizontal).pack(side=tk.LEFT)
        ttk.Checkbutton(flip, text="Отразить Y", variable=self.clone_flip_vertical).pack(side=tk.LEFT, padx=(12, 0))
        overlay = ttk.LabelFrame(body, text="Наложение источника", padding=8)
        overlay.pack(fill=tk.X, pady=(10, 0))
        ttk.Checkbutton(overlay, text="Показывать до рисования", variable=self.clone_overlay_visible).pack(anchor=tk.W)
        opacity_row = ttk.Frame(overlay)
        opacity_row.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(opacity_row, text="Непрозрачность").pack(side=tk.LEFT)
        ttk.Scale(opacity_row, variable=self.clone_overlay_opacity, from_=0.05, to=1.0).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        ttk.Button(body, text="Закрыть", command=dialog.destroy).pack(side=tk.BOTTOM, anchor=tk.E, pady=(12, 0))
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        dialog.lift()
        dialog.focus_set()

    @staticmethod
    def _source_dialog_pair(parent, first_label, first_var, second_label, second_var, start, end) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=first_label, width=16).pack(side=tk.LEFT)
        ttk.Spinbox(row, textvariable=first_var, from_=start, to=end, width=8).pack(side=tk.LEFT)
        if second_var is not None:
            ttk.Label(row, text=second_label, width=3).pack(side=tk.LEFT, padx=(10, 0))
            ttk.Spinbox(row, textvariable=second_var, from_=start, to=end, width=8).pack(side=tk.LEFT)

    def clone_source_settings_changed(self, *_args) -> None:
        if not hasattr(self, "_source_anchor"):
            return
        if self._source_anchor.point is not None:
            requested = int(self.clone_source_x.get()), int(self.clone_source_y.get())
            if requested != self._source_anchor.point:
                self._source_anchor.set_source(requested)
                self._clone_source = requested
        self._clone_sample_pixels = None
        if hasattr(self, "canvas"):
            self.update_clone_source_marker()
            if self._last_pointer_event is not None:
                self.update_clone_overlay(self.canvas_to_doc(self._last_pointer_event))

    def update_clone_overlay(self, point: tuple[int, int]) -> None:
        if self.tool.get() not in {"clone", "healing"} or not bool(self.clone_overlay_visible.get()):
            self.clear_clone_overlay()
            return
        if self._source_anchor.point is None:
            self.clear_clone_overlay()
            return
        if self._clone_sample_pixels is None:
            self.prepare_clone_sample()
        if self._source_anchor.stroke_source is None and self._source_anchor.aligned and self._source_anchor.offset is not None:
            source = (
                point[0] + self._source_anchor.offset[0] + int(self.clone_offset_x.get()),
                point[1] + self._source_anchor.offset[1] + int(self.clone_offset_y.get()),
            )
        else:
            source = self.clone_source_for_point(point)
        if source is None or self._clone_sample_pixels is None:
            self.clear_clone_overlay()
            return
        radius = max(1, int(self.brush_size.get()))
        size = radius * 2 + 1
        sampled, valid = sample_source_patch(
            self._clone_sample_pixels,
            self._clone_sample_origin,
            source,
            size,
            size,
            self.current_source_transform(),
        )
        falloff = retouch_falloff_mask(radius, float(self.hardness.get())) * valid
        preview = sampled.copy()
        alpha = preview[:, :, 3].astype(np.float32) * falloff * float(self.clone_overlay_opacity.get())
        preview[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
        image = Image.fromarray(preview, mode="RGBA")
        scale = float(self.zoom.get())
        if scale != 1.0:
            image = image.resize((max(1, round(size * scale)), max(1, round(size * scale))), Image.Resampling.BILINEAR)
        self._clone_overlay_image = ImageTk.PhotoImage(image)
        cx, cy = self.doc_to_canvas(point[0] - radius, point[1] - radius)
        if self._clone_overlay_id is None:
            self._clone_overlay_id = self.canvas.create_image(cx, cy, image=self._clone_overlay_image, anchor=tk.NW)
        else:
            self.canvas.itemconfigure(self._clone_overlay_id, image=self._clone_overlay_image)
            self.canvas.coords(self._clone_overlay_id, cx, cy)
        self.canvas.tag_raise(self._clone_overlay_id)
        for item_id in self._brush_preview_ids:
            self.canvas.tag_raise(item_id)

    def clear_clone_overlay(self) -> None:
        if self._clone_overlay_id is not None:
            self.canvas.delete(self._clone_overlay_id)
            self._clone_overlay_id = None
        self._clone_overlay_image = None
