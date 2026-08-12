from __future__ import annotations

from ..app_shared import *


class LayerMasksMixin:
    def group_selected_layers(self) -> None:
        selected = [layer for layer in self.doc.layers if layer.id in self.selected_layer_ids]
        if len(selected) < 2:
            self.status_text("Для группы выберите минимум два слоя")
            return
        group_id = uuid.uuid4().hex
        self.run_document_command("Сгруппировать слои", lambda: [setattr(layer, "group_id", group_id) for layer in selected])
        self.refresh_layers()

    def ungroup_selected_layers(self) -> None:
        group_ids = {layer.group_id for layer in self.doc.layers if layer.id in self.selected_layer_ids and layer.group_id}
        if not group_ids:
            self.status_text("Выбранные слои не входят в группу")
            return
        self.run_document_command("Распустить группу", lambda: [setattr(layer, "group_id", None) for layer in self.doc.layers if layer.group_id in group_ids])
        self.selected_layer_ids = {layer.id for layer in self.doc.layers if layer.id in self.selected_layer_ids}
        self.refresh_layers()

    def merge_down(self) -> None:
        self.run_document_command("Merge down", self.doc.merge_down)
        self.refresh()

    def flatten(self) -> None:
        self.run_document_command("Flatten", self.doc.flatten)
        self.refresh()

    def add_mask_from_selection(self) -> None:
        self.run_document_command("Add mask from selection", self.doc.add_mask_from_selection)
        self.refresh()

    def add_reveal_all_mask(self) -> None:
        self.run_document_command("Add reveal-all mask", self.doc.add_reveal_all_mask)
        self.refresh()

    def add_hide_all_mask(self) -> None:
        self.run_document_command("Add hide-all mask", self.doc.add_hide_all_mask)
        self.refresh()

    def invert_layer_mask(self) -> None:
        self.run_document_command("Invert mask", self.doc.invert_active_mask)
        self.refresh()

    def toggle_layer_mask(self) -> None:
        self.set_layer_property("Toggle mask", "mask_enabled", not self.doc.layer.mask_enabled)

    def toggle_layer_mask_link(self) -> None:
        if self.doc.layer.mask is None:
            messagebox.showinfo("Маска слоя", "У активного слоя нет маски.")
            return
        self.set_layer_property("Toggle mask link", "mask_linked", not self.doc.layer.mask_linked, affects_canvas=False)

    def set_mask_density(self) -> None:
        layer = self.doc.layer
        if layer.mask is None:
            messagebox.showinfo("Mask density", "Active layer has no mask.")
            return
        value = simpledialog.askfloat("Mask density", "Density 0..1:", initialvalue=float(layer.mask_density), minvalue=0.0, maxvalue=1.0)
        if value is not None:
            self.set_layer_property("Mask density", "mask_density", float(value))

    def set_mask_feather(self) -> None:
        layer = self.doc.layer
        if layer.mask is None:
            messagebox.showinfo("Mask feather", "Active layer has no mask.")
            return
        value = simpledialog.askfloat("Mask feather", "Radius px:", initialvalue=float(layer.mask_feather), minvalue=0.0, maxvalue=500.0)
        if value is not None:
            self.set_layer_property("Mask feather", "mask_feather", float(value), preserve_render_cache=False)

    def refine_layer_mask(self) -> None:
        if self.doc.layer.mask is None:
            messagebox.showinfo("Уточнить край маски", "У активного слоя нет маски.")
            return
        data = self.refine_layer_mask_dialog()
        if data is None:
            return
        self.run_document_command(
            "Уточнить край маски",
            lambda: self.doc.refine_active_mask(
                int(data["smooth"]),
                int(data["feather"]),
                float(data["contrast"]),
                int(data["shift"]),
                int(data["edge_radius"]),
                float(data["edge_strength"]),
                int(data["confidence_threshold"]),
            ),
        )
        self.refresh()

    def refine_layer_mask_dialog(self) -> dict[str, object] | None:
        layer = self.doc.layer
        dialog = tk.Toplevel(self)
        dialog.title("Уточнить край маски")
        dialog.transient(self)
        dialog.resizable(False, False)
        dialog.grab_set()

        smooth = tk.IntVar(value=2)
        feather = tk.IntVar(value=1)
        contrast = tk.DoubleVar(value=1.15)
        shift = tk.IntVar(value=0)
        edge_radius = tk.IntVar(value=3)
        edge_strength = tk.DoubleVar(value=0.65)
        confidence_threshold = tk.IntVar(value=96)
        preview_mode = tk.StringVar(value=SELECT_MASK_PREVIEW_CHANNEL)
        result: dict[str, object] | None = None

        preview = ttk.Label(dialog)
        preview.grid(row=0, column=0, rowspan=9, padx=12, pady=12, sticky="n")
        stats = ttk.Label(dialog, text="", justify=tk.LEFT)
        stats.grid(row=9, column=0, padx=12, pady=(0, 12), sticky="w")

        def values() -> dict[str, object]:
            return {
                "smooth": max(0, int(smooth.get())),
                "feather": max(0, int(feather.get())),
                "contrast": max(0.0, float(contrast.get())),
                "shift": int(shift.get()),
                "edge_radius": max(0, int(edge_radius.get())),
                "edge_strength": float(np.clip(edge_strength.get(), 0.0, 1.0)),
                "confidence_threshold": int(np.clip(confidence_threshold.get(), 0, 255)),
            }

        def update_preview(*_args) -> None:
            try:
                current = values()
            except (tk.TclError, ValueError):
                return
            mask = self.doc.preview_active_mask_refinement(**current)
            if mask is None:
                return
            canvas = self.render_select_mask_preview(layer.pixels, mask, preview_mode.get(), 180)
            self._mask_edge_preview_image = ImageTk.PhotoImage(canvas)
            preview.configure(image=self._mask_edge_preview_image)
            active = int(np.count_nonzero(mask))
            stats.configure(text=f"Активных пикселей: {active}\nГраницы: {self.mask_bounds(mask) or '-'}")

        def add_spin(row: int, label: str, variable, from_: float, to: float, increment: float = 1.0) -> None:
            ttk.Label(dialog, text=label).grid(row=row, column=1, sticky="w", padx=(0, 12), pady=(8, 0))
            spin = ttk.Spinbox(dialog, textvariable=variable, from_=from_, to=to, increment=increment, width=10, command=update_preview)
            spin.grid(row=row, column=2, sticky="ew", padx=(0, 12), pady=(8, 0))

        add_spin(0, "Сглаживание", smooth, 0, 100)
        add_spin(1, "Растушёвка", feather, 0, 500)
        add_spin(2, "Контраст", contrast, 0.0, 5.0, 0.05)
        add_spin(3, "Сдвиг края", shift, -500, 500)
        add_spin(4, "Радиус анализа", edge_radius, 0, 40)
        add_spin(5, "Сила привязки", edge_strength, 0.0, 1.0, 0.05)
        add_spin(6, "Порог уверенности", confidence_threshold, 0, 255)
        ttk.Label(dialog, text="Просмотр").grid(row=7, column=1, sticky="w", padx=(0, 12), pady=(8, 0))
        preview_box = ttk.Combobox(dialog, textvariable=preview_mode, values=SELECT_MASK_PREVIEW_MODES, state="readonly", width=18)
        preview_box.grid(row=7, column=2, sticky="ew", padx=(0, 12), pady=(8, 0))

        buttons = ttk.Frame(dialog)
        buttons.grid(row=8, column=1, columnspan=2, sticky="e", padx=12, pady=12)

        def accept() -> None:
            nonlocal result
            try:
                result = values()
            except (tk.TclError, ValueError):
                return
            dialog.destroy()

        ttk.Button(buttons, text="ОК", command=accept).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT)
        preview_box.bind("<<ComboboxSelected>>", update_preview)
        for variable in [smooth, feather, contrast, shift, edge_radius, edge_strength, confidence_threshold]:
            variable.trace_add("write", update_preview)
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        update_preview()
        dialog.wait_window()
        return result

    def apply_layer_mask(self) -> None:
        self.run_document_command("Apply mask", self.doc.apply_active_mask)
        self.refresh()

    def delete_layer_mask(self) -> None:
        self.run_document_command("Delete mask", self.doc.delete_active_mask)
        self.refresh()

    def layer_selected(self, _event) -> None:
        sel = self.layer_list.curselection()
        if sel:
            self.selected_layer_ids = {self.doc.layers[len(self.doc.layers) - 1 - row].id for row in sel}
            active_row = int(self.layer_list.index(tk.ACTIVE))
            active_index = len(self.doc.layers) - 1 - active_row
            if not (0 <= active_index < len(self.doc.layers)) or self.doc.layers[active_index].id not in self.selected_layer_ids:
                active_index = len(self.doc.layers) - 1 - sel[0]
            self.doc.active_layer = active_index
            if self.tool.get() == "text" and self.doc.layer.kind == "text":
                self.load_text_properties_from_layer(self.doc.layer)
            self.layer_opacity.set(self.doc.layer.opacity)
            self.blend_mode.set(self.doc.layer.blend_mode)
            self.refresh_layer_previews()
            self.refresh_properties()
            self.update_object_bounds()
            self.info.configure(text=f"{self.doc.width} x {self.doc.height}px\nСлоев: {len(self.doc.layers)}\nВыбрано: {len(self.selected_layer_ids)}")

    def layer_list_click(self, event) -> str | None:
        if int(event.x) > 34 or not self.doc.layers:
            return None
        row = int(self.layer_list.nearest(event.y))
        bounds = self.layer_list.bbox(row)
        if bounds is None or event.y < bounds[1] or event.y >= bounds[1] + bounds[3]:
            return "break"
        layer_index = len(self.doc.layers) - 1 - row
        if layer_index < 0 or layer_index >= len(self.doc.layers):
            return "break"
        active_index = self.doc.active_layer
        layer = self.doc.layers[layer_index]
        if bool(getattr(event, "state", 0) & 0x0008):
            saved = getattr(self, "_visibility_solo_state", None)
            if saved is not None and saved[0] == layer.id:
                states = saved[1]
                self.run_document_command("Восстановить видимость слоёв", lambda: [setattr(item, "visible", states.get(item.id, item.visible)) for item in self.doc.layers])
                self._visibility_solo_state = None
            else:
                if saved is not None:
                    for item in self.doc.layers:
                        item.visible = saved[1].get(item.id, item.visible)
                states = {item.id: bool(item.visible) for item in self.doc.layers}
                self._visibility_solo_state = (layer.id, states)
                self.run_document_command("Показать только выбранный слой", lambda: [setattr(item, "visible", item.id == layer.id) for item in self.doc.layers])
            self.doc.active_layer = active_index; self.refresh(preserve_render_cache=True)
            return "break"
        self._visibility_solo_state = None
        before = bool(layer.visible)
        layer.visible = not before
        self.doc.dirty = True
        self.push_command(LayerVisibilityCommand("Видимость слоя", layer.id, before, layer.visible))
        self.doc.active_layer = active_index
        self.refresh(preserve_render_cache=True)
        return "break"

    def begin_layer_opacity_change(self, _event) -> None:
        self._opacity_layer_id = self.doc.layer.id
        self._opacity_before = self.doc.layer.opacity

    def change_layer_opacity(self, _value) -> None:
        self.doc.layer.opacity = float(self.layer_opacity.get())
        self.doc.dirty = True
        self.request_canvas_refresh(preserve_layer_caches=True)

    def end_layer_opacity_change(self, _event) -> None:
        if self._opacity_layer_id is not None and self._opacity_before is not None:
            layer = self.doc.get_layer(self._opacity_layer_id)
            if layer is not None and layer.opacity != self._opacity_before:
                self.push_command(LayerOpacityCommand("Layer opacity", self._opacity_layer_id, self._opacity_before, layer.opacity))
        self._opacity_layer_id = None
        self._opacity_before = None

    def change_blend_mode(self, _event) -> None:
        layer = self.doc.layer
        before = layer.blend_mode
        after = self.blend_mode.get()
        if before == after:
            return
        layer.blend_mode = after
        self.doc.dirty = True
        self.push_command(LayerBlendModeCommand("Layer blend mode", layer.id, before, after))
        self.refresh(preserve_render_cache=True)

    def toggle_layer_visible(self) -> None:
        layer = self.doc.layer
        before = bool(layer.visible)
        layer.visible = not before
        self.doc.dirty = True
        self.push_command(LayerVisibilityCommand("Видимость слоя", layer.id, before, layer.visible))
        self.refresh(preserve_render_cache=True)

    def toggle_layer_lock(self) -> None:
        self.set_layer_property("Toggle layer lock", "locked", not self.doc.layer.locked, affects_canvas=False)
