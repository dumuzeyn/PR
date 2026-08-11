from __future__ import annotations

import json

from ..app_shared import *


CONDITION_FIELDS = {
    "Ширина документа": "document.width",
    "Высота документа": "document.height",
    "Количество слоёв": "document.layer_count",
    "Есть выделение": "document.has_selection",
    "Глубина цвета": "document.bit_depth",
    "Цветовая модель": "document.color_model",
    "Тип активного слоя": "layer.kind",
    "Имя активного слоя": "layer.name",
    "Расширение файла": "source.extension",
    "Имя файла": "source.name",
}

CONDITION_OPERATORS = {
    "равно": "eq",
    "не равно": "ne",
    "больше": "gt",
    "не меньше": "gte",
    "меньше": "lt",
    "не больше": "lte",
    "содержит": "contains",
    "входит в список": "in",
    "существует": "exists",
}


class ActionWorkflowsMixin:
    def start_action_recording(self) -> None:
        self.action_recorder.start()
        self.status_text("Запись действия начата")

    def stop_action_recording(self) -> None:
        self.action_recorder.stop()
        self.status_text(f"Запись остановлена: {len(self.action_recorder.steps)} шагов")
        if self.action_recorder.steps:
            self.show_action_editor()

    def clear_action_recording(self) -> None:
        self.action_recorder.steps.clear()
        self.status_text("Запись действия очищена")

    def save_action_recording(self) -> None:
        if not self.action_recorder.steps:
            messagebox.showinfo("Действия", "Нет записанных шагов.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("Действие PhotoRedactor", "*.json")],
        )
        if path:
            self.action_recorder.save(path)
            self.status_text(f"Действие сохранено: {path}")

    def load_action_recording(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Действие PhotoRedactor", "*.json")])
        if not path:
            return
        try:
            data = load_action(path)
            self.action_recorder.stop()
            self.action_recorder.steps = [
                ActionStep(
                    command=str(raw["command"]),
                    params=dict(raw.get("params") or {}),
                    label=str(raw.get("label", "")),
                    condition=raw.get("condition"),
                    on_error=str(raw.get("on_error", "stop")),
                    enabled=bool(raw.get("enabled", True)),
                    stop_after=bool(raw.get("stop_after", False)),
                )
                for raw in data.get("steps", [])
                if isinstance(raw, dict)
            ]
            self.show_action_editor()
        except Exception as exc:
            messagebox.showerror("Действия", str(exc))

    def run_action_file(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Действие PhotoRedactor", "*.json"), ("Все файлы", "*.*")]
        )
        if not path:
            return
        try:
            holder = {}

            def execute() -> None:
                holder["report"] = self.action_runner.run_with_report(self.doc, path)

            self.run_document_command("Выполнить действие", execute)
            self.refresh()
            report = holder["report"]
            message = f"Выполнено шагов: {report.executed}\nПропущено: {report.skipped}"
            if report.errors:
                message += f"\nОшибок: {len(report.errors)}"
            if report.stop_message:
                message += f"\n{report.stop_message}"
            messagebox.showinfo("Действия", message)
        except Exception as exc:
            messagebox.showerror("Действия", str(exc))

    def show_action_editor(self) -> None:
        existing = getattr(self, "_action_editor", None)
        if existing is not None and existing.winfo_exists():
            existing.lift()
            return
        window = tk.Toplevel(self)
        self._action_editor = window
        window.title("Редактор действия")
        window.geometry("760x460")
        window.transient(self)

        tree = ttk.Treeview(window, columns=("step", "condition", "errors"), show="headings", selectmode="browse")
        self._action_editor_tree = tree
        tree.heading("step", text="Шаг")
        tree.heading("condition", text="Условие")
        tree.heading("errors", text="При ошибке")
        tree.column("step", width=350)
        tree.column("condition", width=230)
        tree.column("errors", width=110)
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 6))

        def condition_text(condition) -> str:
            if not condition:
                return "Всегда"
            items = condition if isinstance(condition, list) else [condition]
            return "; ".join(f"{item.get('field')} {item.get('operator')} {item.get('value', '')}" for item in items)

        def refresh() -> None:
            selected = tree.selection()
            index = tree.index(selected[0]) if selected else 0
            tree.delete(*tree.get_children())
            for item_index, step in enumerate(self.action_recorder.steps):
                tree.insert("", "end", iid=str(item_index), values=(step.label or step.command, condition_text(step.condition), step.on_error))
            children = tree.get_children()
            if children:
                tree.selection_set(children[min(index, len(children) - 1)])

        def selected_index() -> int | None:
            selected = tree.selection()
            return int(selected[0]) if selected else None

        def move(delta: int) -> None:
            index = selected_index()
            if index is None or not 0 <= index + delta < len(self.action_recorder.steps):
                return
            self.action_recorder.steps[index], self.action_recorder.steps[index + delta] = (
                self.action_recorder.steps[index + delta],
                self.action_recorder.steps[index],
            )
            refresh()
            tree.selection_set(str(index + delta))

        def remove() -> None:
            index = selected_index()
            if index is not None:
                self.action_recorder.steps.pop(index)
                refresh()

        def set_error_policy() -> None:
            index = selected_index()
            if index is None:
                return
            labels = {"Остановить действие": "stop", "Продолжить": "continue", "Пропустить файл": "skip_file"}
            answer = simpledialog.askstring(
                "Обработка ошибки",
                "Введите: stop, continue или skip_file",
                initialvalue=self.action_recorder.steps[index].on_error,
                parent=window,
            )
            if answer in labels.values():
                self.action_recorder.steps[index].on_error = answer
                refresh()

        def add_stop() -> None:
            message = simpledialog.askstring("Остановка", "Сообщение:", initialvalue="Проверьте результат", parent=window)
            if message:
                self.action_recorder.add_stop(message)
                refresh()

        controls = ttk.Frame(window)
        controls.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(controls, text="Вверх", command=lambda: move(-1)).pack(side=tk.LEFT)
        ttk.Button(controls, text="Вниз", command=lambda: move(1)).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Условие...", command=lambda: self._edit_action_condition(window, selected_index(), refresh)).pack(side=tk.LEFT, padx=(12, 4))
        ttk.Button(controls, text="При ошибке...", command=set_error_policy).pack(side=tk.LEFT)
        ttk.Button(controls, text="Добавить остановку", command=add_stop).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Удалить", command=remove).pack(side=tk.LEFT)
        ttk.Button(controls, text="Сохранить...", command=self.save_action_recording).pack(side=tk.RIGHT)
        refresh()

    def _edit_action_condition(self, parent, index: int | None, done) -> None:
        if index is None:
            return
        step = self.action_recorder.steps[index]
        current = step.condition if isinstance(step.condition, dict) else {}
        dialog = tk.Toplevel(parent)
        dialog.title("Условие шага")
        dialog.transient(parent)
        dialog.grab_set()
        body = ttk.Frame(dialog, padding=12)
        body.pack(fill=tk.BOTH, expand=True)
        field = tk.StringVar(value=next((label for label, key in CONDITION_FIELDS.items() if key == current.get("field")), next(iter(CONDITION_FIELDS))))
        operator = tk.StringVar(value=next((label for label, key in CONDITION_OPERATORS.items() if key == current.get("operator")), "равно"))
        value = tk.StringVar(value=json.dumps(current.get("value", ""), ensure_ascii=False) if current else "")
        ttk.Label(body, text="Поле").grid(row=0, column=0, sticky="w")
        ttk.Combobox(body, textvariable=field, values=list(CONDITION_FIELDS), state="readonly", width=28).grid(row=1, column=0, sticky="ew", pady=(2, 8))
        ttk.Label(body, text="Сравнение").grid(row=2, column=0, sticky="w")
        ttk.Combobox(body, textvariable=operator, values=list(CONDITION_OPERATORS), state="readonly", width=28).grid(row=3, column=0, sticky="ew", pady=(2, 8))
        ttk.Label(body, text="Значение").grid(row=4, column=0, sticky="w")
        ttk.Entry(body, textvariable=value, width=32).grid(row=5, column=0, sticky="ew", pady=(2, 10))

        def apply() -> None:
            try:
                parsed = json.loads(value.get())
            except json.JSONDecodeError:
                parsed = value.get()
            step.condition = {"field": CONDITION_FIELDS[field.get()], "operator": CONDITION_OPERATORS[operator.get()], "value": parsed}
            dialog.destroy()
            done()

        actions = ttk.Frame(body)
        actions.grid(row=6, column=0, sticky="e")
        ttk.Button(actions, text="Без условия", command=lambda: (setattr(step, "condition", None), dialog.destroy(), done())).pack(side=tk.LEFT)
        ttk.Button(actions, text="Применить", command=apply).pack(side=tk.LEFT, padx=(6, 0))

    def batch_action_file(self) -> None:
        action = filedialog.askopenfilename(filetypes=[("Действие PhotoRedactor", "*.json")])
        if not action:
            return
        sources = filedialog.askopenfilenames(filetypes=[("Изображения", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff")])
        if not sources:
            return
        destination = filedialog.askdirectory(title="Папка результата")
        if not destination:
            return
        options = self._batch_options_dialog()
        if options is None:
            return
        try:
            self.batch_queue.enqueue(action, list(sources), destination, **options)
            self.open_batch_queue()
        except Exception as exc:
            messagebox.showerror("Пакетная обработка", str(exc))

    def _batch_options_dialog(self) -> dict[str, str] | None:
        result: list[dict[str, str] | None] = [None]
        dialog = tk.Toplevel(self)
        dialog.title("Параметры пакетного задания")
        dialog.transient(self)
        dialog.grab_set()
        body = ttk.Frame(dialog, padding=12)
        body.pack(fill=tk.BOTH, expand=True)
        formats = {"PNG": ".png", "JPEG": ".jpg", "WebP": ".webp", "TIFF": ".tiff", "BMP": ".bmp"}
        conflicts = {"Добавить номер": "rename", "Перезаписать": "overwrite", "Пропустить": "skip"}
        errors = {"Продолжать очередь": "continue", "Остановить очередь": "stop"}
        format_var = tk.StringVar(value="PNG")
        conflict_var = tk.StringVar(value="Добавить номер")
        error_var = tk.StringVar(value="Продолжать очередь")
        for row, (label, variable, values) in enumerate((
            ("Формат результата", format_var, list(formats)),
            ("Если файл существует", conflict_var, list(conflicts)),
            ("При ошибке файла", error_var, list(errors)),
        )):
            ttk.Label(body, text=label).grid(row=row * 2, column=0, sticky="w")
            ttk.Combobox(body, textvariable=variable, values=values, state="readonly", width=30).grid(row=row * 2 + 1, column=0, sticky="ew", pady=(2, 8))

        def accept() -> None:
            result[0] = {"suffix": formats[format_var.get()], "conflict": conflicts[conflict_var.get()], "on_error": errors[error_var.get()]}
            dialog.destroy()

        controls = ttk.Frame(body)
        controls.grid(row=6, column=0, sticky="e", pady=(4, 0))
        ttk.Button(controls, text="Отмена", command=dialog.destroy).pack(side=tk.LEFT)
        ttk.Button(controls, text="Добавить", command=accept).pack(side=tk.LEFT, padx=(6, 0))
        dialog.wait_window()
        return result[0]

    def open_batch_queue(self) -> None:
        existing = getattr(self, "_batch_queue_window", None)
        if existing is not None and existing.winfo_exists():
            existing.lift()
            return
        window = tk.Toplevel(self)
        self._batch_queue_window = window
        window.title("Очередь пакетной обработки")
        window.geometry("820x430")
        window.transient(self)
        tree = ttk.Treeview(window, columns=("state", "progress", "errors", "folder"), show="headings")
        self._batch_queue_tree = tree
        for key, title, width in (("state", "Состояние", 150), ("progress", "Готово", 90), ("errors", "Ошибки", 70), ("folder", "Папка результата", 430)):
            tree.heading(key, text=title)
            tree.column(key, width=width)
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 6))
        progress = ttk.Progressbar(window, mode="determinate")
        progress.pack(fill=tk.X, padx=10, pady=(0, 6))

        def refresh() -> None:
            if not window.winfo_exists():
                return
            selected = tree.selection()
            tree.delete(*tree.get_children())
            for job in self.batch_queue.jobs:
                tree.insert("", "end", iid=job.id, values=(job.state, f"{job.completed_count}/{len(job.items)}", job.error_count, job.destination))
            if selected and tree.exists(selected[0]):
                tree.selection_set(selected[0])
            total = sum(len(job.items) for job in self.batch_queue.jobs)
            complete = sum(job.completed_count for job in self.batch_queue.jobs)
            progress.configure(maximum=max(1, total), value=complete)
            window.after(250, refresh)

        controls = ttk.Frame(window)
        controls.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(controls, text="Добавить...", command=self.batch_action_file).pack(side=tk.LEFT)
        ttk.Button(controls, text="Запустить", command=self._start_batch_queue).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Отменить", command=self.batch_queue.cancel).pack(side=tk.LEFT)
        ttk.Button(controls, text="Подробности", command=lambda: self._show_batch_job_details(tree)).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Убрать завершённые", command=self.batch_queue.remove_finished).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Сохранить очередь...", command=self._save_batch_queue).pack(side=tk.RIGHT)
        ttk.Button(controls, text="Загрузить очередь...", command=self._load_batch_queue).pack(side=tk.RIGHT, padx=4)
        refresh()

    def _show_batch_job_details(self, tree) -> None:
        selected = tree.selection()
        if not selected:
            return
        job = next((item for item in self.batch_queue.jobs if item.id == selected[0]), None)
        if job is None:
            return
        window = tk.Toplevel(self)
        window.title("Файлы пакетного задания")
        window.geometry("900x380")
        window.transient(self)
        details = ttk.Treeview(window, columns=("source", "state", "steps", "message"), show="headings")
        for key, title, width in (("source", "Исходный файл", 300), ("state", "Состояние", 120), ("steps", "Шагов", 70), ("message", "Результат", 380)):
            details.heading(key, text=title)
            details.column(key, width=width)
        for item in job.items:
            details.insert("", "end", values=(item.source, item.state, item.executed_steps, item.message or item.target))
        details.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _start_batch_queue(self) -> None:
        future = getattr(self, "_batch_queue_future", None)
        if future is not None and not future.done():
            return
        self.status_text("Пакетная очередь выполняется...")
        self._batch_queue_future = self.executor.submit(self.batch_queue.run_all)

        def complete() -> None:
            try:
                jobs = self._batch_queue_future.result()
                errors = sum(job.error_count for job in jobs)
                self.status_text(f"Очередь завершена: заданий {len(jobs)}, ошибок {errors}")
            except Exception as exc:
                messagebox.showerror("Пакетная обработка", str(exc))
        self._batch_queue_future.add_done_callback(lambda _future: self.after(0, complete))

    def _save_batch_queue(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Очередь PhotoRedactor", "*.json")])
        if path:
            self.batch_queue.save(path)

    def _load_batch_queue(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Очередь PhotoRedactor", "*.json")])
        if path:
            try:
                self.batch_queue.load(path)
            except Exception as exc:
                messagebox.showerror("Пакетная обработка", str(exc))


__all__ = ["ActionWorkflowsMixin"]
