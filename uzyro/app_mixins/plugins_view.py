from __future__ import annotations

from ..app_shared import *
from ..security.validation import loads_bounded_json


PLUGIN_PERMISSION_INFO = {
    "pixels": ("Изменение пикселей", "низкий", "Работа с переданным изображением."),
    "document": ("Документ", "средний", "Чтение и изменение структуры открытого документа."),
    "filesystem.read": ("Чтение файлов", "высокий", "Чтение файлов, доступных вашей учётной записи."),
    "filesystem.write": ("Запись файлов", "высокий", "Создание и изменение файлов на компьютере."),
    "network": ("Сеть", "высокий", "Подключение к интернету и локальной сети."),
    "process": ("Запуск программ", "очень высокий", "Запуск внешних процессов от вашего имени."),
    "native": ("Нативный код", "очень высокий", "Загрузка системных библиотек; выдавайте только доверенным плагинам."),
}


class PluginsViewMixin:
    def refresh_plugin_import_menu(self) -> None:
        self.plugin_import_menu.delete(0, tk.END)
        if not self.plugin_registry.importers:
            self.plugin_import_menu.add_command(label="Нет доступных импортёров", state=tk.DISABLED)
            return
        for name in sorted(self.plugin_registry.importers):
            self.plugin_import_menu.add_command(label=name, command=lambda value=name: self.import_with_plugin(value))

    def refresh_plugin_export_menu(self) -> None:
        self.plugin_export_menu.delete(0, tk.END)
        if not self.plugin_registry.exporters:
            self.plugin_export_menu.add_command(label="Нет доступных экспортёров", state=tk.DISABLED)
            return
        for name in sorted(self.plugin_registry.exporters):
            self.plugin_export_menu.add_command(label=name, command=lambda value=name: self.export_with_plugin(value))

    def import_with_plugin(self, name: str) -> None:
        extension = self.plugin_registry.importers[name]
        pattern = " ".join(f"*{item}" for item in extension.extensions) or "*.*"
        path = filedialog.askopenfilename(filetypes=[(name, pattern), ("Все файлы", "*.*")])
        if not path:
            return
        raw = simpledialog.askstring("Импорт через плагин", "Параметры JSON:", initialvalue="{}")
        if raw is None:
            return
        try:
            params = loads_bounded_json(raw, maximum=1024 * 1024)
            if not isinstance(params, dict):
                raise ValueError("Параметры должны быть JSON-объектом")
        except Exception as exc:
            messagebox.showerror("Импорт через плагин", str(exc))
            return

        def done(document) -> None:
            self.open_document_session(document)

        self.run_background("Импорт через плагин", lambda: self.plugin_registry.import_document(name, path, params), done)

    def export_with_plugin(self, name: str) -> None:
        extension = self.plugin_registry.exporters[name]
        default_extension = extension.extensions[0] if extension.extensions else ""
        pattern = " ".join(f"*{item}" for item in extension.extensions) or "*.*"
        path = filedialog.asksaveasfilename(defaultextension=default_extension, filetypes=[(name, pattern), ("Все файлы", "*.*")])
        if not path:
            return
        raw = simpledialog.askstring("Экспорт через плагин", "Параметры JSON:", initialvalue="{}")
        if raw is None:
            return
        try:
            params = loads_bounded_json(raw, maximum=1024 * 1024)
            if not isinstance(params, dict):
                raise ValueError("Параметры должны быть JSON-объектом")
        except Exception as exc:
            messagebox.showerror("Экспорт через плагин", str(exc))
            return
        snapshot = self.document_copy()
        self.run_background("Экспорт через плагин", lambda: self.plugin_registry.export_document(name, snapshot, path, params))

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
            params = loads_bounded_json(raw, maximum=1024 * 1024)
            if not isinstance(params, dict):
                raise ValueError("Параметры должны быть JSON-объектом")
            self.apply_to_layer(name, lambda pixels: self.plugin_registry.apply_filter(name, pixels, params))
        except Exception as exc:
            messagebox.showerror("Фильтр-плагин", str(exc))

    def reload_plugins(self) -> None:
        for name in getattr(self, "_plugin_action_names", set()):
            self.action_runner.commands.pop(name, None)
        count = self.plugin_registry.discover()
        for name, callback in self.plugin_registry.action_commands.items():
            if name not in self.action_runner.commands:
                self.action_runner.register(name, callback)
        self._plugin_action_names = set(self.plugin_registry.action_commands)
        self.status_text(f"Плагины перезагружены: {count}")
        if self.plugin_registry.errors:
            self.show_plugin_errors()

    def show_plugin_errors(self) -> None:
        if not self.plugin_registry.errors:
            messagebox.showinfo("Плагины", "Ошибок загрузки нет.")
            return
        self.show_text_window("Ошибки плагинов", "\n".join(self.plugin_registry.errors))

    def plugin_manager(self) -> None:
        window = tk.Toplevel(self)
        self._plugin_manager_window = window
        window.title("Управление плагинами")
        window.geometry("980x560")
        window.minsize(820, 480)
        window.transient(self)
        header = ttk.Frame(window, padding=(16, 14, 16, 8))
        header.pack(fill=tk.X)
        ttk.Label(header, text="Плагины", style="PanelTitle.TLabel").pack(anchor=tk.W)
        ttk.Label(
            header,
            text="Новые плагины отключены до выдачи запрошенных разрешений.",
            style="Secondary.TLabel",
        ).pack(anchor=tk.W, pady=(3, 0))
        tree = ttk.Treeview(window, columns=("author", "version", "status", "permissions"), show="tree headings", selectmode="browse")
        self._plugin_manager_tree = tree
        tree.heading("#0", text="Плагин")
        tree.heading("author", text="Автор")
        tree.heading("version", text="Версия")
        tree.heading("status", text="Состояние")
        tree.heading("permissions", text="Разрешения")
        tree.column("#0", width=190)
        tree.column("author", width=145)
        tree.column("version", width=75, anchor=tk.CENTER)
        tree.column("status", width=150)
        tree.column("permissions", width=350)
        tree.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 8))
        tree.tag_configure("blocked", foreground=TOKENS.WARNING)
        tree.tag_configure("critical", foreground=TOKENS.DANGER)
        tree.tag_configure("ready", foreground=TOKENS.SUCCESS)

        def refresh() -> None:
            tree.delete(*tree.get_children())
            for info in self.plugin_registry.plugins.values():
                blocked = bool(info.blocked_permissions)
                critical = bool(info.requested_permissions & {"process", "native"})
                status = "Ожидает разрешения" if blocked else "Разрешён"
                if info.legacy and blocked:
                    status = "Совместимый, отключён"
                labels = [PLUGIN_PERMISSION_INFO[value][0] for value in sorted(info.requested_permissions)]
                tag = "critical" if critical else "blocked" if blocked else "ready"
                tree.insert(
                    "", "end", iid=info.id, text=info.name,
                    values=(info.author or "Не указан", info.version, status, ", ".join(labels) or "Не запрошены"),
                    tags=(tag,),
                )

        def configure_permissions() -> None:
            selected = tree.selection()
            if not selected:
                return
            info = self.plugin_registry.plugins[selected[0]]
            dialog = tk.Toplevel(window)
            dialog.title(f"Разрешения: {info.name}")
            dialog.transient(window)
            dialog.grab_set()
            dialog.minsize(600, 380)
            body = ttk.Frame(dialog, padding=16)
            body.pack(fill=tk.BOTH, expand=True)
            ttk.Label(body, text=info.name, style="PanelTitle.TLabel").pack(anchor=tk.W)
            ttk.Label(
                body,
                text=f"Автор: {info.author or 'Не указан'}  |  Версия: {info.version}  |  API: {info.api_version}",
                style="Secondary.TLabel",
            ).pack(anchor=tk.W, pady=(3, 12))
            variables = {}
            for permission in sorted(info.requested_permissions):
                variable = tk.BooleanVar(value=permission in info.granted_permissions)
                variables[permission] = variable
                label, risk, description = PLUGIN_PERMISSION_INFO[permission]
                row = ttk.Frame(body, style="Elevated.TFrame", padding=(10, 8))
                row.pack(fill=tk.X, pady=3)
                check = ttk.Checkbutton(row, text=label, variable=variable)
                check.pack(anchor=tk.W)
                tone = "Danger.TLabel" if risk == "очень высокий" else "Warning.TLabel" if risk == "высокий" else "Secondary.TLabel"
                ttk.Label(row, text=f"Риск: {risk}. {description}", style=tone, wraplength=530).pack(anchor=tk.W, padx=(24, 0), pady=(2, 0))

            if info.requested_permissions & {"process", "native"}:
                ttk.Label(
                    body,
                    text="Python-плагин запускается отдельно, но это не системная песочница. Разрешайте запуск программ и нативный код только проверенному автору.",
                    style="Danger.TLabel", wraplength=550,
                ).pack(fill=tk.X, pady=(10, 0))

            def apply() -> None:
                self.plugin_registry.set_permissions(info.id, {key for key, value in variables.items() if value.get()})
                dialog.destroy()
                self.reload_plugins()
                refresh()

            actions = ttk.Frame(body)
            actions.pack(fill=tk.X, pady=(14, 0))
            ttk.Button(actions, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT)
            ttk.Button(actions, text="Применить", command=apply, style="Primary.TButton").pack(side=tk.RIGHT, padx=(0, 6))
            self.center_toplevel(dialog, 640, max(420, 225 + len(variables) * 58))

        controls = ttk.Frame(window)
        controls.pack(fill=tk.X, padx=16, pady=(0, 14))
        ttk.Button(controls, text="Настроить разрешения...", command=configure_permissions).pack(side=tk.LEFT)
        ttk.Button(controls, text="Перезагрузить", command=lambda: (self.reload_plugins(), refresh())).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Показать ошибки", command=self.show_plugin_errors).pack(side=tk.LEFT)
        refresh()

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
