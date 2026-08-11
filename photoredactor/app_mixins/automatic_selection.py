from __future__ import annotations

from ..app_shared import *
from ..segmentation import SegmentationService


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
        service = SegmentationService.from_local_resources()
        roi_doc: list[tuple[int, int, int, int] | None] = [None]
        roi_start: list[tuple[int, int] | None] = [None]

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
        target_box = ttk.Combobox(controls, textvariable=target, values=["Объект", "Объект в области", "Фон", "Небо"], state="readonly")
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
        backend_status = ttk.Label(controls, text=f"Движок: {service.backend_name}", wraplength=230, justify=tk.LEFT, style="Secondary.TLabel")
        backend_status.pack(fill=tk.X, pady=(14, 3))

        def open_model_folder() -> None:
            folder = service.model_folder()
            folder.mkdir(parents=True, exist_ok=True)
            if os.name == "nt":
                os.startfile(folder)

        ttk.Button(controls, text="Папка модели сегментации", command=open_model_folder).pack(fill=tk.X)

        latest_mask: list[np.ndarray | None] = [None]
        latest_signature: list[tuple[str, float, tuple[int, int, int, int] | None] | None] = [None]
        preview_after: list[str | None] = [None]
        latest_backend: list[str] = [service.backend_name]

        def calculate_mask() -> np.ndarray:
            local_box = None
            if target.get() == "Объект в области" and roi_doc[0] is not None:
                x1, y1, x2, y2 = roi_doc[0]
                local_box = (x1 - layer.x, y1 - layer.y, x2 - layer.x, y2 - layer.y)
            if target.get() == "Объект в области" and local_box is None:
                local_box = (0, 0, layer.pixels.shape[1], layer.pixels.shape[0])
            selection = service.select(layer.pixels, target.get(), float(sensitivity.get()), local_box)
            latest_backend[0] = selection.backend + (" (fallback)" if selection.fallback else "")
            return self.doc._layer_mask_to_document(layer, selection.mask)

        def update_preview(*_args) -> None:
            preview_after[0] = None
            try:
                mask = calculate_mask()
            except (tk.TclError, ValueError):
                return
            latest_mask[0] = mask
            latest_signature[0] = (target.get(), round(float(sensitivity.get()), 4), roi_doc[0])
            size = 520
            image = self.render_select_mask_preview(composite, mask, preview_mode.get(), size)
            if target.get() == "Объект в области" and roi_doc[0] is not None:
                scale = min(1.0, size / max(1, self.doc.width), size / max(1, self.doc.height))
                offset_x = (size - self.doc.width * scale) / 2.0
                offset_y = (size - self.doc.height * scale) / 2.0
                x1, y1, x2, y2 = roi_doc[0]
                draw = ImageDraw.Draw(image)
                draw.rectangle((offset_x + x1 * scale, offset_y + y1 * scale, offset_x + x2 * scale, offset_y + y2 * scale), outline=(50, 205, 255, 255), width=2)
            self._automatic_selection_preview_image = ImageTk.PhotoImage(image)
            preview.configure(image=self._automatic_selection_preview_image)
            sensitivity_value.configure(text=f"{round(float(sensitivity.get()) * 100)}%")
            selected = np.count_nonzero(mask >= 128)
            soft = np.count_nonzero((mask > 0) & (mask < 255))
            quality.configure(text=f"Выбрано: {selected} px  |  Мягкий край: {soft} px")
            backend_status.configure(text=f"Движок: {latest_backend[0]}")
            notes = {
                "Объект": "Ищет отличающиеся от краёв изображения области и сохраняет мелкие связанные детали.",
                "Объект в области": "Протяните прямоугольник по объекту в предпросмотре. Анализ ограничится этой областью.",
                "Фон": "Строит мягкое дополнение к найденному объекту, включая прозрачные края.",
                "Небо": "Анализирует связанные с верхним краем синие и светлые области.",
            }
            description.configure(text=notes[target.get()])

        def preview_point(event) -> tuple[int, int] | None:
            size = 520
            scale = min(1.0, size / max(1, self.doc.width), size / max(1, self.doc.height))
            image_left = (preview.winfo_width() - size) / 2.0 + (size - self.doc.width * scale) / 2.0
            image_top = (preview.winfo_height() - size) / 2.0 + (size - self.doc.height * scale) / 2.0
            point = (round((event.x - image_left) / max(scale, 1e-8)), round((event.y - image_top) / max(scale, 1e-8)))
            if 0 <= point[0] < self.doc.width and 0 <= point[1] < self.doc.height:
                return point
            return None

        def roi_press(event) -> None:
            if target.get() != "Объект в области":
                return
            roi_start[0] = preview_point(event)

        def roi_drag(event) -> None:
            point = preview_point(event)
            if roi_start[0] is None or point is None:
                return
            x1, y1 = roi_start[0]
            roi_doc[0] = (min(x1, point[0]), min(y1, point[1]), max(x1, point[0]) + 1, max(y1, point[1]) + 1)
            schedule_preview()

        def roi_release(event) -> None:
            roi_drag(event)
            roi_start[0] = None

        def schedule_preview(*_args) -> None:
            if preview_after[0] is not None:
                try:
                    dialog.after_cancel(preview_after[0])
                except tk.TclError:
                    pass
            preview_after[0] = dialog.after(90, update_preview)

        def accept() -> None:
            nonlocal result
            signature = (target.get(), round(float(sensitivity.get()), 4), roi_doc[0])
            mask = latest_mask[0] if latest_mask[0] is not None and latest_signature[0] == signature else calculate_mask()
            result = {"mask": mask.copy(), "target": target.get(), "sensitivity": float(sensitivity.get()), "output": output.get()}
            dialog.destroy()

        ttk.Label(footer, text="Результат можно сразу доработать кистями сложного края", style="Secondary.TLabel").pack(side=tk.LEFT)
        ttk.Button(footer, text="Применить", command=accept, style="Primary.TButton").pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(footer, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT)
        for variable in (target, sensitivity, preview_mode):
            variable.trace_add("write", schedule_preview)
        preview.bind("<ButtonPress-1>", roi_press)
        preview.bind("<B1-Motion>", roi_drag)
        preview.bind("<ButtonRelease-1>", roi_release)
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
        self.semantic_select("Объект", "Выделить объект")

    def select_background(self) -> None:
        self.semantic_select("Фон", "Выделить фон")

    def select_sky(self) -> None:
        self.semantic_select("Небо", "Выделить небо")

    def semantic_select(self, target: str, command_name: str) -> None:
        service = SegmentationService.from_local_resources()
        result = service.select(self.doc.layer.pixels, target, 0.55)
        mask = self.doc._layer_mask_to_document(self.doc.layer, result.mask)
        self.run_selection_command(command_name, lambda: setattr(self.doc, "selection_mask", mask.copy()))
        suffix = " через резервный CPU-алгоритм" if result.fallback else ""
        self.status_text(f"{command_name}: {result.backend}{suffix}")

    def segmentation_model_folder(self) -> None:
        service = SegmentationService.from_local_resources()
        folder = service.model_folder()
        folder.mkdir(parents=True, exist_ok=True)
        messagebox.showinfo(
            "Модель выделения",
            f"Активный движок: {service.backend_name}\n\n"
            "Приложение не скачивает модель автоматически. Локальную ONNX-модель foreground.onnx можно поместить в открываемую папку.",
        )
        if os.name == "nt":
            os.startfile(folder)

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
