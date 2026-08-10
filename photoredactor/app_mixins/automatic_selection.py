from __future__ import annotations

from ..app_shared import *


class AutomaticSelectionMixin:
    def select_opaque_pixels(self) -> None:
        self.run_selection_command("Select opaque pixels", lambda: self.doc.select_opaque_pixels(self.doc.layer))

    def automatic_selection_workspace(self) -> None:
        data = self.automatic_selection_dialog()
        if data is None:
            return
        mask = np.asarray(data["mask"], dtype=np.uint8)
        self.run_selection_command("Автоматическое выделение", lambda: setattr(self.doc, "selection_mask", mask.copy()))
        self.refresh_canvas()
        if data["output"] == "Уточнить в «Выделить и маска»":
            self.select_and_mask_workspace()

    def automatic_selection_dialog(self) -> dict[str, object] | None:
        layer = self.doc.layer
        if layer.kind == "adjustment" or layer.pixels.size == 0:
            messagebox.showinfo("Автоматическое выделение", "Активный слой не содержит изображения.")
            return None
        dialog = tk.Toplevel(self)
        dialog.title("Автоматическое выделение")
        dialog.transient(self)
        dialog.grab_set()
        dialog.minsize(820, 590)
        target = tk.StringVar(value="Объект")
        sensitivity = tk.DoubleVar(value=0.55)
        preview_mode = tk.StringVar(value=SELECT_MASK_PREVIEW_OVERLAY)
        output = tk.StringVar(value="Уточнить в «Выделить и маска»")
        result: dict[str, object] | None = None
        composite = self.render_engine.render(self.doc, checker=False)

        header = ttk.Frame(dialog, padding=(12, 10, 12, 6))
        header.pack(fill=tk.X)
        ttk.Label(header, text="Автоматическое выделение", style="PanelTitle.TLabel").pack(side=tk.LEFT)
        quality = ttk.Label(header, text="", style="Secondary.TLabel")
        quality.pack(side=tk.RIGHT)
        body = ttk.PanedWindow(dialog, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=12)
        preview = ttk.Label(body, anchor=tk.CENTER)
        controls = ttk.Frame(body, width=250, padding=(12, 4))
        body.add(preview, weight=1)
        body.add(controls, weight=0)
        footer = ttk.Frame(dialog, padding=12)
        footer.pack(fill=tk.X)

        ttk.Label(controls, text="Что выделить", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(4, 3))
        target_box = ttk.Combobox(controls, textvariable=target, values=["Объект", "Фон", "Небо"], state="readonly")
        target_box.pack(fill=tk.X)
        ttk.Label(controls, text="Чувствительность", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(14, 3))
        sensitivity_value = ttk.Label(controls, text="55%", style="Secondary.TLabel")
        sensitivity_value.pack(anchor=tk.E)
        ttk.Scale(controls, variable=sensitivity, from_=0.0, to=1.0).pack(fill=tk.X)
        ttk.Label(controls, text="Просмотр", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(14, 3))
        mode_box = ttk.Combobox(controls, textvariable=preview_mode, values=SELECT_MASK_PREVIEW_MODES, state="readonly")
        mode_box.pack(fill=tk.X)
        ttk.Label(controls, text="После выбора", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(14, 3))
        ttk.Combobox(controls, textvariable=output, values=["Применить выделение", "Уточнить в «Выделить и маска»"], state="readonly").pack(fill=tk.X)
        description = ttk.Label(controls, wraplength=230, justify=tk.LEFT, style="Secondary.TLabel")
        description.pack(fill=tk.X, pady=(16, 0))

        latest_mask: list[np.ndarray | None] = [None]
        latest_signature: list[tuple[str, float] | None] = [None]
        preview_after: list[str | None] = [None]

        def calculate_mask() -> np.ndarray:
            local = automatic_selection_mask(layer.pixels, target.get(), float(sensitivity.get()))
            return self.doc._layer_mask_to_document(layer, local)

        def update_preview(*_args) -> None:
            preview_after[0] = None
            try:
                mask = calculate_mask()
            except (tk.TclError, ValueError):
                return
            latest_mask[0] = mask
            latest_signature[0] = (target.get(), round(float(sensitivity.get()), 4))
            size = 520
            image = self.render_select_mask_preview(composite, mask, preview_mode.get(), size)
            self._automatic_selection_preview_image = ImageTk.PhotoImage(image)
            preview.configure(image=self._automatic_selection_preview_image)
            sensitivity_value.configure(text=f"{round(float(sensitivity.get()) * 100)}%")
            selected = np.count_nonzero(mask >= 128)
            soft = np.count_nonzero((mask > 0) & (mask < 255))
            quality.configure(text=f"Выбрано: {selected} px  |  Мягкий край: {soft} px")
            notes = {
                "Объект": "Ищет отличающиеся от краёв изображения области и сохраняет мелкие связанные детали.",
                "Фон": "Строит мягкое дополнение к найденному объекту, включая прозрачные края.",
                "Небо": "Анализирует связанные с верхним краем синие и светлые области.",
            }
            description.configure(text=notes[target.get()])

        def schedule_preview(*_args) -> None:
            if preview_after[0] is not None:
                try:
                    dialog.after_cancel(preview_after[0])
                except tk.TclError:
                    pass
            preview_after[0] = dialog.after(90, update_preview)

        def accept() -> None:
            nonlocal result
            signature = (target.get(), round(float(sensitivity.get()), 4))
            mask = latest_mask[0] if latest_mask[0] is not None and latest_signature[0] == signature else calculate_mask()
            result = {"mask": mask.copy(), "target": target.get(), "sensitivity": float(sensitivity.get()), "output": output.get()}
            dialog.destroy()

        ttk.Label(footer, text="Результат можно сразу доработать кистями сложного края", style="Secondary.TLabel").pack(side=tk.LEFT)
        ttk.Button(footer, text="Применить", command=accept, style="Primary.TButton").pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(footer, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT)
        for variable in (target, sensitivity, preview_mode):
            variable.trace_add("write", schedule_preview)
        self._automatic_selection_target = target
        self._automatic_selection_sensitivity = sensitivity
        self._automatic_selection_output = output
        self._automatic_selection_accept = accept
        self._automatic_selection_preview = preview
        self.center_toplevel(dialog, 900, 650)
        update_preview()
        dialog.wait_window()
        return result

    def select_subject(self) -> None:
        self.run_selection_command("Выделить объект", lambda: self.doc.select_subject(self.doc.layer))

    def select_background(self) -> None:
        self.run_selection_command("Выделить фон", lambda: self.doc.select_background(self.doc.layer))

    def select_sky(self) -> None:
        self.run_selection_command("Выделить небо", lambda: self.doc.select_sky(self.doc.layer))

    def single_row_selection(self) -> None:
        y = simpledialog.askinteger("Single row", "Y coordinate:", initialvalue=self.doc.height // 2, minvalue=0, maxvalue=max(0, self.doc.height - 1))
        if y is not None:
            self.run_selection_command("Single row selection", lambda: self.doc.set_single_row_selection(y))

    def single_column_selection(self) -> None:
        x = simpledialog.askinteger("Single column", "X coordinate:", initialvalue=self.doc.width // 2, minvalue=0, maxvalue=max(0, self.doc.width - 1))
        if x is not None:
            self.run_selection_command("Single column selection", lambda: self.doc.set_single_column_selection(x))

    def save_selection(self) -> None:
        if self.doc.selection_mask is None:
            messagebox.showinfo("Save selection", "There is no active selection.")
            return
        default = f"Selection {len(self.doc.saved_selections) + 1}"
        name = simpledialog.askstring("Save selection", "Name:", initialvalue=default)
        if name:
            self.run_document_command("Save selection", lambda: self.doc.save_selection(name))
            self.refresh()

    def load_selection(self) -> None:
        if not self.doc.saved_selections:
            messagebox.showinfo("Load selection", "No saved selections.")
            return
        names = ", ".join(self.doc.saved_selections)
        name = simpledialog.askstring("Load selection", f"Name:\n{names}")
        if name:
            self.run_selection_command("Load selection", lambda: self.doc.load_selection(name))

    def delete_saved_selection(self) -> None:
        if not self.doc.saved_selections:
            messagebox.showinfo("Delete selection", "No saved selections.")
            return
        names = ", ".join(self.doc.saved_selections)
        name = simpledialog.askstring("Delete selection", f"Name:\n{names}")
        if name:
            self.run_document_command("Delete saved selection", lambda: self.doc.delete_saved_selection(name))
            self.refresh()
