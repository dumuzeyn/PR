from __future__ import annotations

import zipfile

from ..app_shared import *
from ..security.paths import validate_local_input
from ..security.temporary import temporary_root


class SmartFilesMixin:
    def open_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Проекты, PSD/PSB, изображения и RAW", "*.prdx *.psd *.psb *.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.dng *.arw *.cr2 *.cr3 *.nef *.nrw *.orf *.raf *.rw2 *.pef *.raw"), ("Все файлы", "*.*")])
        if not path:
            return
        self.open_path(path)

    def open_path(self, path: str) -> None:
        try:
            source = validate_local_input(path)
        except (OSError, ValueError):
            messagebox.showerror("Открытие", f"Файл не найден:\n{path}")
            self.recent_files = [item for item in self.recent_files if item.lower() != path.lower()]
            self.refresh_recent_menu()
            return
        source_path = str(source)

        def worker() -> Document:
            return Document.open_project(source_path) if source_path.lower().endswith(".prdx") else Document.from_image(source_path)

        def opened(document: Document) -> None:
            self.add_recent_file(source_path)
            self.open_document_session(document)
            compatibility = document.metadata.get("psd_compatibility", {})
            warnings = compatibility.get("warnings", []) if isinstance(compatibility, dict) else []
            if warnings:
                messagebox.showwarning("Совместимость PSD/PSB", "\n".join(str(item) for item in warnings[:8]))

        self.run_background("Открытие проекта", worker, opened)

    def place_embedded(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Изображения и проекты", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.prdx"), ("Все файлы", "*.*")])
        if not path:
            return
        self.run_document_command("Place embedded", lambda: self.doc.place_image(path))
        self.add_recent_file(path)
        self.refresh()

    def place_linked(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Изображения и проекты", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.prdx"), ("Все файлы", "*.*")])
        if not path:
            return
        self.run_document_command("Place linked", lambda: self.doc.place_image(path, linked=True))
        self.add_recent_file(path)
        self.refresh()

    def update_linked_layer(self) -> None:
        layer = self.doc.layer
        source_path = (layer.smart_data or {}).get("source_path")
        if layer.kind != "linked" or not source_path:
            messagebox.showinfo("Linked layer", "The active layer is not linked to an external file.")
            return
        if not Path(source_path).exists():
            messagebox.showerror("Linked layer", f"Linked source file not found:\n{source_path}")
            return
        self.run_document_command("Update linked layer", self.doc.update_linked_layer)
        self.refresh()

    def show_linked_layer_status(self) -> None:
        status = self.doc.linked_layer_status()
        labels = {
            "embedded": "Объект встроен в проект",
            "current": "Связанный файл актуален",
            "modified": "Связанный файл изменён вне редактора",
            "missing": "Связанный файл не найден",
        }
        messagebox.showinfo("Smart Object", f"{labels.get(status['status'], status['status'])}\n\n{status.get('path') or ''}")

    def convert_to_smart_object(self) -> None:
        selected = self.selected_layer_ids or {self.doc.layer.id}
        self.run_document_command("Преобразовать в Smart Object", lambda: self.doc.convert_layers_to_smart_object(selected))
        self.selected_layer_ids = {self.doc.layer.id}
        self.refresh()

    def edit_smart_object_contents(self) -> None:
        nested = self.doc.active_smart_document()
        if nested is None:
            messagebox.showinfo("Smart Object", "У выбранного Smart Object нет вложенного редактируемого документа.")
            return
        layer_id = self.doc.layer.id
        smart_root = Path(tempfile.mkdtemp(prefix="smart-", dir=temporary_root()))
        temporary = smart_root / "document.prdx"
        nested.save_project(temporary)
        process = self.launch_smart_document_editor(temporary)

        def worker() -> Document:
            process.wait()
            return Document.open_project(temporary)

        def apply_and_close(edited: Document) -> None:
            target = self.doc.get_layer(layer_id)
            if target is not None:
                self.doc.active_layer = self.doc.layers.index(target)
                self.run_document_command("Обновить содержимое Smart Object", lambda: self.doc.update_active_smart_document(edited))
                self.refresh()
            try:
                temporary.unlink()
            except OSError:
                pass
            try:
                smart_root.rmdir()
            except OSError:
                pass

        self._smart_object_editor_process = process
        self._smart_object_editor_path = temporary
        self.run_background("Редактирование Smart Object", worker, apply_and_close)

    @staticmethod
    def launch_smart_document_editor(path: Path):
        if getattr(sys, "frozen", False):
            command = [sys.executable, "--smart-edit", str(path)]
        else:
            command = [sys.executable, str(Path(__file__).resolve().parents[1] / "uzyro_launcher.py"), "--smart-edit", str(path)]
        return subprocess.Popen(command)

    def resolve_linked_conflict_dialog(self) -> None:
        layer = self.doc.layer
        status = self.doc.linked_layer_status(layer)
        if layer.kind != "linked":
            messagebox.showinfo("Связанный Smart Object", "Выбранный Smart Object является встроенным.")
            return
        if status["status"] == "current":
            messagebox.showinfo("Связанный Smart Object", "Связанный файл не изменён.")
            return
        dialog = tk.Toplevel(self)
        dialog.title("Конфликт связанного Smart Object")
        dialog.transient(self)
        dialog.grab_set()
        result: list[str | None] = [None]
        title = "Связанный файл не найден" if status["status"] == "missing" else "Связанный файл изменён"
        ttk.Label(dialog, text=title, style="PanelTitle.TLabel").pack(anchor=tk.W, padx=16, pady=(16, 5))
        ttk.Label(dialog, text=str(status.get("path") or ""), wraplength=500, style="Secondary.TLabel").pack(anchor=tk.W, padx=16, pady=(0, 14))
        buttons = ttk.Frame(dialog, padding=16)
        buttons.pack(fill=tk.X)

        def choose(action: str) -> None:
            result[0] = action
            dialog.destroy()

        if status["status"] == "modified":
            ttk.Button(buttons, text="Обновить из файла", command=lambda: choose("update"), style="Primary.TButton").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(buttons, text="Сохранить текущую версию внутри проекта", command=lambda: choose("embed")).pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="Выбрать другой файл", command=lambda: choose("relink")).pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT)
        self._linked_conflict_choice = choose
        self.center_toplevel(dialog, 720, 220)
        dialog.wait_window()
        action = result[0]
        if action is None:
            return
        path = None
        if action == "relink":
            path = filedialog.askopenfilename(filetypes=[("Изображения и проекты", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.prdx"), ("Все файлы", "*.*")])
            if not path:
                return
        self.run_document_command("Разрешить конфликт Smart Object", lambda: self.doc.resolve_linked_conflict(action, path))
        self.refresh()

    def replace_smart_contents(self) -> None:
        if self.doc.layer.kind not in {"linked", "embedded"}:
            messagebox.showinfo("Smart Object", "Активный слой не является Smart Object.")
            return
        path = filedialog.askopenfilename(filetypes=[("Изображения и проекты", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.prdx"), ("Все файлы", "*.*")])
        if not path:
            return
        linked = messagebox.askyesno("Smart Object", "Оставить содержимое связанным с внешним файлом?")
        self.run_document_command("Заменить содержимое Smart Object", lambda: self.doc.replace_active_smart_contents(path, linked))
        self.refresh()

    def convert_smart_to_embedded(self) -> None:
        if self.doc.layer.kind not in {"linked", "embedded"}:
            messagebox.showinfo("Smart Object", "Активный слой не является Smart Object.")
            return
        self.run_document_command("Преобразовать Smart Object во встроенный", self.doc.convert_active_smart_to_embedded)
        self.refresh()

    def reset_smart_transform(self) -> None:
        self.run_document_command("Сбросить трансформацию Smart Object", self.doc.reset_active_smart_transform)
        self.refresh()

    def relink_layer(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Изображения и проекты", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.prdx"), ("Все файлы", "*.*")])
        if not path:
            return
        self.run_document_command("Relink layer", lambda: self.doc.relink_active_layer(path))
        self.add_recent_file(path)
        self.refresh()

    def load_files_as_layers(self) -> None:
        paths = filedialog.askopenfilenames(filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"), ("All", "*.*")])
        if not paths:
            return

        def edit():
            for path in paths:
                self.doc.place_image(path)

        self.run_document_command("Load files as layers", edit)
        for path in paths:
            self.add_recent_file(path)
        self.refresh()

    def save(self) -> None:
        if self.doc.path and self.doc.path.lower().endswith(".prdx"):
            self.save_project_async(self.doc.path)
        else:
            self.save_as_project()

    def save_as_project(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".prdx", filetypes=[("UZYRO project", "*.prdx")])
        if path:
            self.save_project_async(path)

    def save_project_async(self, path: str) -> None:
        snapshot = self.document_copy()
        generation = self._edit_generation
        document = self.doc
        session = self.document_manager.active

        def worker():
            snapshot.save_project(path)
            return path

        def done(saved_path):
            document.path = saved_path
            current_generation = self._edit_generation if self.doc is document else session.edit_generation if session is not None else -1
            if current_generation == generation:
                document.dirty = False
            self.add_recent_file(saved_path)
            self.refresh_document_tabs()

        self.run_background("Save project", worker, done)

    def export_image(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("WebP", "*.webp"), ("TIFF", "*.tiff"), ("BMP", "*.bmp")])
        if path:
            suffix = Path(path).suffix.lower()
            quality = 95
            if suffix in {".jpg", ".jpeg", ".webp"}:
                quality = simpledialog.askinteger("Export quality", "Quality 1..100:", initialvalue=95, minvalue=1, maxvalue=100) or 95
            snapshot = self.document_copy()

            def worker():
                snapshot.export_flat(path, quality)
                return path

            self.run_background("Export image", worker)

    def pick_foreground(self) -> None:
        color = self.color_picker_dialog(self.foreground, "Основной цвет")
        if color is not None:
            self.foreground = color
            self.refresh_color_control()
            self.text_properties_changed()

    def pick_background(self) -> None:
        color = self.color_picker_dialog(self.background, "Дополнительный цвет")
        if color is not None:
            self.background = color
            self.refresh_color_control()

    def pick_color_from_document(self, point: tuple[int, int]) -> None:
        x, y = point
        if x < 0 or y < 0 or x >= self.doc.width or y >= self.doc.height:
            return
        rgba = self.render_engine.render(self.doc, False)[y, x]
        self.foreground = tuple(int(v) for v in rgba)
        self.refresh_color_control()
        self.status_text(f"Основной цвет: {self.color_hex(self.foreground).upper()}")

    def show_image_statistics(self) -> None:
        stats = image_statistics(self.render_engine.render(self.doc, False))
        text = [
            f"Size: {stats['width']} x {stats['height']}",
            f"Opaque pixels: {stats['opaque_pixels']}",
            f"Transparent pixels: {stats['transparent_pixels']}",
            "",
        ]
        for name, values in stats["channels"].items():
            text.append(f"{name}: min {values['min']:.2f}, max {values['max']:.2f}, mean {values['mean']:.2f}, std {values['std']:.2f}")
        messagebox.showinfo("Image statistics", "\n".join(text))

    def show_histogram(self) -> None:
        stats = image_statistics(self.render_engine.render(self.doc, False))
        lines = []
        for channel, values in stats["histogram"].items():
            lines.append(channel.upper())
            peak = max(values) or 1
            for i, value in enumerate(values):
                start = i * 16
                end = start + 15
                bar = "#" * max(1, round(value / peak * 32)) if value else ""
                lines.append(f"{start:03d}-{end:03d}: {bar} {value}")
            lines.append("")
        self.show_text_window("Histogram", "\n".join(lines))

    def show_metadata(self) -> None:
        text = json.dumps(self.doc.metadata or {}, ensure_ascii=False, indent=2)
        self.show_text_window("Metadata / EXIF", text if text != "{}" else "No metadata.")

    def edit_metadata(self) -> None:
        window = tk.Toplevel(self)
        window.title("Редактор метаданных")
        window.geometry("720x500")
        window.transient(self)
        working = copy.deepcopy(self.doc.metadata or {})
        tree = ttk.Treeview(window, columns=("value",), show="tree headings")
        tree.heading("#0", text="Поле")
        tree.heading("value", text="Значение")
        tree.column("#0", width=230)
        tree.column("value", width=450)
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 4))

        def refill() -> None:
            tree.delete(*tree.get_children())
            for key, value in sorted(working.items(), key=lambda item: str(item[0]).lower()):
                tree.insert("", tk.END, iid=str(key), text=str(key), values=(json.dumps(value, ensure_ascii=False),))

        def set_value() -> None:
            selected = tree.selection()
            old_key = selected[0] if selected else ""
            key = simpledialog.askstring("Метаданные", "Название поля:", initialvalue=old_key, parent=window)
            if not key:
                return
            initial = json.dumps(working.get(old_key, ""), ensure_ascii=False)
            raw = simpledialog.askstring("Метаданные", "Значение:", initialvalue=initial, parent=window)
            if raw is None:
                return
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = raw
            if old_key and old_key != key:
                working.pop(old_key, None)
            working[key] = value
            refill()

        def remove_value() -> None:
            for key in tree.selection():
                working.pop(key, None)
            refill()

        buttons = ttk.Frame(window)
        buttons.pack(fill=tk.X, padx=10, pady=(4, 10))
        ttk.Button(buttons, text="Добавить / изменить", command=set_value).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Удалить", command=remove_value).pack(side=tk.LEFT, padx=6)

        def apply() -> None:
            self.run_document_command("Редактировать метаданные", lambda: setattr(self.doc, "metadata", copy.deepcopy(working)))
            self.doc.dirty = True
            window.destroy()
            self.refresh()

        ttk.Button(buttons, text="Применить", command=apply).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Отмена", command=window.destroy).pack(side=tk.RIGHT, padx=6)
        refill()

    def show_cache_status(self) -> None:
        status = self.render_engine.cache_status()
        gpu = status["gpu"]
        metrics = acceleration_metrics()
        calibration = metrics.get("calibration", {})
        project_info = None
        if self.doc.path and self.doc.path.lower().endswith(".prdx") and Path(self.doc.path).exists():
            try:
                project_info = self.doc.project_storage_info(self.doc.path)
            except (OSError, ValueError, zipfile.BadZipFile, KeyError):
                project_info = None
        text = (
            f"Кэш в памяти: {status['memory_bytes'] / 1024 / 1024:.1f} МБ\n"
            f"Объектов в памяти: {status['memory_items']}\n"
            f"Объектов на scratch-диске: {status['disk_items']}\n"
            f"GPU доступен: {'да' if gpu['available'] else 'нет'}\n"
            f"Backend: {gpu.get('backend', 'cpu')}\n"
            f"Устройство: {gpu.get('device') or 'CPU'}\n"
            f"Режим: {gpu.get('mode', 'auto')}\n"
            f"Автовыбор GPU: {'да' if calibration.get('selected') else 'нет'}\n"
            f"GPU-операций: {sum(value for key, value in metrics.get('counts', {}).items() if key.startswith('gpu.'))}"
        )
        if project_info is not None:
            text += (
                f"\n\nФормат проекта: {project_info['format']} v{project_info['version']}\n"
                f"Тайлов в проекте: {project_info['tiles']}\n"
                f"Несжатые данные: {project_info['bytes'] / 1024 / 1024:.1f} МБ"
            )
        messagebox.showinfo("Большие документы", text)

    def interactive_performance_dialog(self) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("Интерактивная производительность")
        dialog.transient(self)
        dialog.grab_set()
        body = ttk.Frame(dialog, padding=16)
        body.pack(fill=tk.BOTH, expand=True)
        ttk.Label(body, text="Проверка рабочих инструментов", style="PanelTitle.TLabel").pack(anchor=tk.W)
        result = ttk.Label(body, text="Тест ещё не запускался.", justify=tk.LEFT)
        result.pack(fill=tk.X, pady=(14, 18))
        run_button = ttk.Button(body, text="Запустить тест")
        run_button.pack(side=tk.LEFT)
        ttk.Button(body, text="Закрыть", command=dialog.destroy).pack(side=tk.RIGHT)

        def done(report: dict[str, float | bool]) -> None:
            state = "Целевые задержки соблюдены" if report["passed"] else "Есть операции выше целевой задержки"
            result.configure(text=(
                f"Кисть: {report['brush_dab_ms']:.1f} мс на штрих\n"
                f"Градиент: {report['gradient_preview_ms']:.1f} мс\n"
                f"Трансформация: {report['transform_preview_ms']:.1f} мс\n"
                f"Изменение среди 60 слоёв: {report['many_layers_tile_ms']:.1f} мс\n\n{state}"
            ))
            run_button.configure(state=tk.NORMAL)

        def run() -> None:
            run_button.configure(state=tk.DISABLED)
            result.configure(text="Выполняется проверка...")
            self.run_background("Тест производительности", benchmark_interactive_paths, done)

        run_button.configure(command=run)
        self._interactive_performance_result = result
        self._interactive_performance_run = run
        self._interactive_performance_dialog = dialog
        self.center_toplevel(dialog, 560, 340)

    def gpu_acceleration_dialog(self) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("Настройка GPU")
        dialog.transient(self)
        dialog.grab_set()
        current = os.environ.get("UZYRO_GPU", "auto").lower()
        mode = tk.StringVar(value=current if current in {"auto", "force", "off"} else "auto")
        body = ttk.Frame(dialog, padding=16)
        body.pack(fill=tk.BOTH, expand=True)
        ttk.Label(body, text="Режим вычислений", style="PanelTitle.TLabel").pack(anchor=tk.W)
        labels = (("Автоматически", "auto"), ("Всегда использовать GPU", "force"), ("Только CPU", "off"))
        for label, value in labels:
            ttk.Radiobutton(body, text=label, value=value, variable=mode).pack(anchor=tk.W, pady=2)
        status_label = ttk.Label(body, justify=tk.LEFT, style="Secondary.TLabel")
        status_label.pack(fill=tk.X, pady=(14, 8))
        result_label = ttk.Label(body, justify=tk.LEFT)
        result_label.pack(fill=tk.X, pady=8)

        def refresh_status() -> None:
            status = acceleration_status()
            status_label.configure(
                text=(
                    f"Backend: {status['backend']}\n"
                    f"Устройство: {status['device'] or 'CPU'}\n"
                    f"OpenCL: {'доступен' if status['opencl_available'] else 'недоступен'}"
                )
            )

        def benchmark_done(report: dict[str, object]) -> None:
            result_label.configure(
                text=(
                    f"CPU: {report['cpu_blur_ms']:.2f} мс   GPU: {report['gpu_blur_ms']:.2f} мс\n"
                    f"Ускорение: {report['blur_speedup']:.2f}x   Макс. ошибка: {report['blur_max_error']}"
                )
            )

        def run_benchmark() -> None:
            self.run_background("Тест GPU", benchmark_acceleration, benchmark_done)

        def apply() -> None:
            os.environ["UZYRO_GPU"] = mode.get()
            reset_acceleration_metrics()
            if mode.get() == "auto":
                calibrate_acceleration()
            self.render_engine.gpu = gpu_status()
            self.render_engine.mipmaps.gpu = dict(self.render_engine.gpu)
            self.save_settings()
            dialog.destroy()

        footer = ttk.Frame(body)
        footer.pack(fill=tk.X, side=tk.BOTTOM, pady=(12, 0))
        ttk.Button(footer, text="Применить", command=apply, style="Primary.TButton").pack(side=tk.RIGHT)
        ttk.Button(footer, text="Закрыть", command=dialog.destroy).pack(side=tk.RIGHT, padx=6)
        ttk.Button(footer, text="Запустить тест", command=run_benchmark).pack(side=tk.LEFT)
        refresh_status()
        self._gpu_mode_variable = mode
        self._gpu_benchmark_result = result_label
        self._gpu_apply = apply
        self._gpu_dialog = dialog
        self.center_toplevel(dialog, 620, 390)
