from __future__ import annotations

from ..app_shared import *


class SourceOverlayMixin:
    CLONE_SOURCE_NAMES = ("A", "B", "C", "D", "E")

    def ensure_clone_sources(self) -> None:
        if hasattr(self, "_clone_sources"):
            return
        self.clone_source_slot = tk.StringVar(value="A")
        self.clone_overlay_auto_hide = tk.BooleanVar(value=False)
        self.clone_overlay_clipped = tk.BooleanVar(value=True)
        self.clone_overlay_invert = tk.BooleanVar(value=False)
        base = self._clone_slot_from_controls()
        self._clone_sources = {name: dict(base, point=None, document_path="", layer_id="", layer_name="") for name in self.CLONE_SOURCE_NAMES}
        self._clone_sources["A"].update(self._clone_source_metadata())
        self._clone_source_switching = False
        self._active_clone_source_name = "A"

    def _clone_source_metadata(self) -> dict[str, object]:
        point = self._source_anchor.point if hasattr(self, "_source_anchor") else None
        layer = self.doc.layer
        return {
            "point": list(point) if point is not None else None,
            "document_path": str(self.doc.path or ""),
            "layer_id": str(layer.id),
            "layer_name": str(layer.name),
        }

    def _clone_slot_from_controls(self) -> dict[str, object]:
        return {
            "source_x": int(self.clone_source_x.get()), "source_y": int(self.clone_source_y.get()),
            "offset_x": int(self.clone_offset_x.get()), "offset_y": int(self.clone_offset_y.get()),
            "scale_x": float(self.clone_scale_x.get()), "scale_y": float(self.clone_scale_y.get()),
            "rotation": float(self.clone_rotation.get()),
            "flip_x": bool(self.clone_flip_horizontal.get()), "flip_y": bool(self.clone_flip_vertical.get()),
            "overlay_opacity": float(self.clone_overlay_opacity.get()), "aligned": bool(self.clone_aligned.get()),
            "sampling": self.clone_sampling.get(),
            "overlay_visible": bool(self.clone_overlay_visible.get()),
            "overlay_auto_hide": bool(self.clone_overlay_auto_hide.get()) if hasattr(self, "clone_overlay_auto_hide") else False,
            "overlay_clipped": bool(self.clone_overlay_clipped.get()) if hasattr(self, "clone_overlay_clipped") else True,
            "overlay_invert": bool(self.clone_overlay_invert.get()) if hasattr(self, "clone_overlay_invert") else False,
        }

    def save_active_clone_source(self) -> None:
        self.ensure_clone_sources()
        if self._clone_source_switching:
            return
        slot = self._active_clone_source_name
        self._clone_sources[slot].update(self._clone_slot_from_controls())
        self._clone_sources[slot].update(self._clone_source_metadata())

    def switch_clone_source(self, name: str) -> None:
        self.ensure_clone_sources()
        if name not in self._clone_sources:
            return
        self.save_active_clone_source()
        self._clone_source_switching = True
        self.clone_source_slot.set(name)
        values = self._clone_sources[name]
        for key, variable in (
            ("source_x", self.clone_source_x), ("source_y", self.clone_source_y),
            ("offset_x", self.clone_offset_x), ("offset_y", self.clone_offset_y),
            ("scale_x", self.clone_scale_x), ("scale_y", self.clone_scale_y),
            ("rotation", self.clone_rotation), ("flip_x", self.clone_flip_horizontal),
            ("flip_y", self.clone_flip_vertical), ("overlay_opacity", self.clone_overlay_opacity),
            ("aligned", self.clone_aligned), ("sampling", self.clone_sampling),
            ("overlay_visible", self.clone_overlay_visible), ("overlay_auto_hide", self.clone_overlay_auto_hide),
            ("overlay_clipped", self.clone_overlay_clipped), ("overlay_invert", self.clone_overlay_invert),
        ):
            variable.set(values.get(key, variable.get()))
        point = values.get("point")
        self._source_anchor.reset()
        if isinstance(point, (list, tuple)) and len(point) == 2:
            self._source_anchor.set_source((int(point[0]), int(point[1])))
            self._clone_source = self._source_anchor.point
        else:
            self._clone_source = None
        self._source_anchor.aligned = bool(self.clone_aligned.get())
        self._active_clone_source_name = name
        self._clone_source_switching = False
        self._clone_sample_pixels = None
        self.update_clone_source_marker()
        self._update_clone_source_info()

    def _update_clone_source_info(self) -> None:
        label = getattr(self, "_clone_source_info", None)
        if label is None or not label.winfo_exists():
            return
        active = self._clone_sources[self._active_clone_source_name]
        document_name = Path(str(active.get("document_path") or "Текущий документ")).name
        label.configure(text=f"{document_name} · {active.get('layer_name') or self.doc.layer.name}")

    def source_retouch_settings_payload(self) -> dict[str, object]:
        self.save_active_clone_source()
        stored_sources = {
            name: {key: value for key, value in source.items() if not key.startswith("_")}
            for name, source in self._clone_sources.items()
        }
        return {
            "active_source": self.clone_source_slot.get(), "sources": stored_sources,
            "diffusion": int(self.healing_diffusion.get()), "spot_mode": self.spot_healing_mode.get(),
            "patch_structure": int(self.patch_structure.get()), "patch_color": float(self.patch_color_adaptation.get()),
            "patch_all_layers": bool(self.patch_sample_all_layers.get()),
        }

    def load_source_retouch_settings(self, settings: dict[str, object]) -> None:
        self.ensure_clone_sources()
        stored = settings.get("sources")
        if isinstance(stored, dict):
            for name in self.CLONE_SOURCE_NAMES:
                if isinstance(stored.get(name), dict):
                    self._clone_sources[name].update(stored[name])
        elif settings:
            self._clone_sources["A"].update({
                "aligned": settings.get("aligned", True), "sampling": settings.get("sampling", "Текущий слой"),
                "scale_x": settings.get("scale_x", 100.0), "scale_y": settings.get("scale_y", 100.0),
                "rotation": settings.get("rotation", 0.0), "flip_x": settings.get("flip_horizontal", False),
                "flip_y": settings.get("flip_vertical", False), "overlay_visible": settings.get("overlay_visible", True),
                "overlay_opacity": settings.get("overlay_opacity", 0.45),
            })
        self.healing_diffusion.set(max(1, min(7, int(settings.get("diffusion", 4)))))
        self.spot_healing_mode.set(str(settings.get("spot_mode", self.spot_healing_mode.get())))
        self.patch_structure.set(max(1, min(7, int(settings.get("patch_structure", 5)))))
        self.patch_color_adaptation.set(float(np.clip(settings.get("patch_color", 8.0), 0.0, 10.0)))
        self.patch_sample_all_layers.set(bool(settings.get("patch_all_layers", False)))
        self._clone_source_switching = True
        self.switch_clone_source(str(settings.get("active_source", "A")))

    def open_clone_source_panel(self) -> None:
        self.ensure_clone_sources()
        dialog = getattr(self, "_clone_source_dialog", None)
        if dialog is not None and dialog.winfo_exists():
            dialog.deiconify()
            dialog.lift()
            return
        dialog = tk.Toplevel(self)
        self._clone_source_dialog = dialog
        dialog.title("Источник клонирования")
        dialog.geometry("430x670")
        dialog.minsize(400, 600)
        dialog.transient(self)
        body = ttk.Frame(dialog, padding=12)
        body.pack(fill=tk.BOTH, expand=True)
        slots = ttk.LabelFrame(body, text="Источники", padding=6)
        slots.pack(fill=tk.X, pady=(0, 8))
        for name in self.CLONE_SOURCE_NAMES:
            ttk.Radiobutton(slots, text=name, value=name, variable=self.clone_source_slot, command=lambda value=name: self.switch_clone_source(value), style="Toolbutton").pack(side=tk.LEFT, fill=tk.X, expand=True)
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
        self._clone_source_info = ttk.Label(source, style="Secondary.TLabel")
        self._clone_source_info.pack(anchor=tk.W, pady=(0, 5))
        self._update_clone_source_info()
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
        ttk.Checkbutton(overlay, text="Автоматически скрывать во время мазка", variable=self.clone_overlay_auto_hide).pack(anchor=tk.W)
        ttk.Checkbutton(overlay, text="Обрезать по форме кисти", variable=self.clone_overlay_clipped).pack(anchor=tk.W)
        ttk.Checkbutton(overlay, text="Инвертировать", variable=self.clone_overlay_invert).pack(anchor=tk.W)
        opacity_row = ttk.Frame(overlay)
        opacity_row.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(opacity_row, text="Непрозрачность").pack(side=tk.LEFT)
        AccentScale(opacity_row, variable=self.clone_overlay_opacity, from_=0.05, to=1.0).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        healing = ttk.LabelFrame(body, text="Восстанавливающая кисть", padding=8)
        healing.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(healing, text="Диффузия 1–7").pack(side=tk.LEFT)
        ttk.Spinbox(healing, textvariable=self.healing_diffusion, from_=1, to=7, width=6).pack(side=tk.RIGHT)
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
        self.ensure_clone_sources()
        if self._clone_source_switching:
            return
        active = self._clone_sources[self._active_clone_source_name]
        previous_sampling = active.get("sampling")
        previous_point = active.get("point")
        if not hasattr(self, "_source_anchor"):
            return
        if self._source_anchor.point is not None:
            requested = int(self.clone_source_x.get()), int(self.clone_source_y.get())
            if requested != self._source_anchor.point:
                self._source_anchor.set_source(requested)
                self._clone_source = requested
        self.save_active_clone_source()
        self._update_clone_source_info()
        current_point = list(self._source_anchor.point) if self._source_anchor.point is not None else None
        if previous_sampling != self.clone_sampling.get() or previous_point != current_point:
            active.pop("_sample_pixels", None)
            active.pop("_sample_origin", None)
            self._clone_sample_pixels = None
        if hasattr(self, "canvas"):
            self.update_clone_source_marker()
            if self._last_pointer_event is not None:
                self.update_clone_overlay(self.canvas_to_doc(self._last_pointer_event))

    def active_clone_snapshot(self) -> tuple[np.ndarray, tuple[int, int]] | None:
        self.ensure_clone_sources()
        active = self._clone_sources[self._active_clone_source_name]
        pixels = active.get("_sample_pixels")
        origin = active.get("_sample_origin")
        if isinstance(pixels, np.ndarray) and isinstance(origin, tuple):
            return pixels, origin
        return None

    def store_active_clone_snapshot(self, pixels: np.ndarray, origin: tuple[int, int]) -> None:
        self.ensure_clone_sources()
        active = self._clone_sources[self._active_clone_source_name]
        active["_sample_pixels"] = pixels
        active["_sample_origin"] = (int(origin[0]), int(origin[1]))

    def update_clone_overlay(self, point: tuple[int, int]) -> None:
        self.ensure_clone_sources()
        if self.tool.get() not in {"clone", "healing"} or not bool(self.clone_overlay_visible.get()):
            self.clear_clone_overlay()
            return
        if bool(self.clone_overlay_auto_hide.get()) and self.drag_start is not None:
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
        falloff = (retouch_falloff_mask(radius, float(self.hardness.get())) if bool(self.clone_overlay_clipped.get()) else np.ones((size, size), dtype=np.float32)) * valid
        preview = sampled.copy()
        if bool(self.clone_overlay_invert.get()):
            preview[:, :, :3] = 255 - preview[:, :, :3]
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
