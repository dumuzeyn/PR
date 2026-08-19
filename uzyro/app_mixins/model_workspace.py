from __future__ import annotations

import threading
import webbrowser

from ..app_shared import *
from ..local_generative import LocalGenerationOptions, LocalImageClient, SAMPLERS, shutdown_local_servers
from ..model_manager import LCM_ACCELERATOR, MODEL_BY_ID, MODEL_CATALOG, ModelStore, detect_hardware


BACKEND_LABELS = {
    "auto": "Автоматически",
    "cuda": "NVIDIA CUDA",
    "vulkan": "Универсальный Vulkan",
    "cpu": "Только CPU",
}


class ModelWorkspaceMixin:
    def local_model_store(self) -> ModelStore:
        if not hasattr(self, "_local_model_store"):
            self._local_model_store = ModelStore(self.app_data_dir / "local-ai")
        return self._local_model_store

    def local_hardware_profile(self):
        if not hasattr(self, "_local_hardware_profile"):
            self._local_hardware_profile = detect_hardware()
        return self._local_hardware_profile

    def effective_local_backend(self) -> str:
        selected = str(self.generative_settings.get("local_backend", "auto"))
        return self.local_hardware_profile().recommended_backend if selected == "auto" else selected

    def active_local_model_id(self) -> str:
        selected = str(self.generative_settings.get("local_model_id", MODEL_CATALOG[0].model_id))
        return selected if selected in MODEL_BY_ID else MODEL_CATALOG[0].model_id

    def active_local_model_name(self) -> str:
        return MODEL_BY_ID[self.active_local_model_id()].name

    def local_generation_ready(self) -> bool:
        store = self.local_model_store()
        return bool(
            store.model_installed(self.active_local_model_id()) and store.accelerator_installed()
            and store.engine_executable(self.effective_local_backend())
        )

    def create_local_image_client(self, settings: dict[str, object], cancel=None) -> LocalImageClient:
        store = self.local_model_store()
        model_id = str(settings.get("local_model_id", self.active_local_model_id()))
        backend = str(settings.get("local_backend", self.effective_local_backend()))
        if backend == "auto":
            backend = self.local_hardware_profile().recommended_backend
        model = MODEL_BY_ID.get(model_id)
        engine = store.engine_executable(backend)
        if model is None or engine is None or not store.model_verified(model):
            raise GenerativeAPIError("Локальная модель не установлена. Откройте «Модели» и нажмите «Скачать и выбрать».")
        profile = self.local_hardware_profile()
        quality = str(settings.get("performance_profile", "balanced"))
        max_side = 512 if quality == "fast" or backend == "cpu" or (profile.nvidia_vram_mb and profile.nvidia_vram_mb < 6000) else 768
        options = LocalGenerationOptions(
            steps=int(settings.get("steps", model.recommended_steps)),
            cfg_scale=float(settings.get("cfg_scale", model.recommended_cfg)),
            strength=float(settings.get("strength", 0.88)),
            sampler=str(settings.get("sampler", "DPM++ 2M")),
            max_side=max_side,
        )
        return LocalImageClient(engine, store.model_path(model), backend, options, cancel, store.accelerator_path())

    def local_generation_variants(
        self, operation: str, source: np.ndarray, mask: np.ndarray | None,
        margins: tuple[int, int, int, int], settings: dict[str, object], count: int, cancel=None,
    ):
        client = self.create_local_image_client(settings, cancel)
        prompt = str(settings["prompt"])
        negative = str(settings.get("negative_prompt", ""))
        style = str(settings.get("style", ""))
        if operation == "fill":
            if mask is None:
                raise GenerativeAPIError("Маска локального заполнения отсутствует")
            call = lambda value: client.inpaint(source, mask, prompt, negative, value, style)
        else:
            call = lambda value: client.outpaint(source, margins, prompt, negative, value, style)
        return client.variants(call, int(settings["seed"]), count)

    def model_manager_dialog(self) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("Локальные модели")
        dialog.transient(self)
        dialog.minsize(860, 560)
        dialog.grab_set()
        store = self.local_model_store()
        profile = self.local_hardware_profile()
        selected_backend = tk.StringVar(value=str(self.generative_settings.get("local_backend", "auto")))
        status = tk.StringVar(value="Готово")
        progress_text = tk.StringVar(value="")
        cancel_event = threading.Event()
        self._model_download_cancel = cancel_event

        header = ttk.Frame(dialog, padding=(16, 14, 16, 10))
        header.pack(fill=tk.X)
        ttk.Label(header, text="Локальные модели", style="PanelTitle.TLabel").pack(anchor=tk.W)
        ttk.Label(header, text=profile.summary, style="Secondary.TLabel").pack(anchor=tk.W, pady=(4, 0))

        backend_row = ttk.Frame(header)
        backend_row.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(backend_row, text="Ускорение").pack(side=tk.LEFT)
        backend_box = ttk.Combobox(
            backend_row, state="readonly", width=24, values=list(BACKEND_LABELS.values()),
        )
        backend_box.set(BACKEND_LABELS.get(selected_backend.get(), BACKEND_LABELS["auto"]))
        backend_box.pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(
            backend_row, text=f"Рекомендация: {BACKEND_LABELS[profile.recommended_backend]}",
            style="Secondary.TLabel",
        ).pack(side=tk.LEFT, padx=(12, 0))

        content = ttk.PanedWindow(dialog, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True, padx=16)
        list_panel = ttk.Frame(content)
        details = ttk.Frame(content, padding=(16, 4, 0, 0))
        content.add(list_panel, weight=3)
        content.add(details, weight=2)

        tree = ttk.Treeview(list_panel, columns=("size", "state"), show="tree headings", selectmode="browse")
        tree.heading("#0", text="Модель")
        tree.heading("size", text="Размер")
        tree.heading("state", text="Состояние")
        tree.column("#0", width=330, stretch=True)
        tree.column("size", width=85, anchor=tk.E, stretch=False)
        tree.column("state", width=120, anchor=tk.CENTER, stretch=False)
        tree.pack(fill=tk.BOTH, expand=True)
        tree.tag_configure("verified", foreground=TOKENS.SUCCESS)
        tree.tag_configure("missing", foreground=TOKENS.TEXT_SECONDARY)

        name_text = tk.StringVar()
        description_text = tk.StringVar()
        license_text = tk.StringVar()
        ttk.Label(details, textvariable=name_text, style="PanelTitle.TLabel", wraplength=300).pack(anchor=tk.W)
        ttk.Label(details, textvariable=description_text, wraplength=300, justify=tk.LEFT).pack(anchor=tk.W, pady=(10, 0))
        ttk.Label(details, textvariable=license_text, style="Secondary.TLabel", wraplength=300).pack(anchor=tk.W, pady=(14, 0))
        source_button = ttk.Button(details, text="Страница модели")
        source_button.pack(anchor=tk.W, pady=(12, 0))
        active_label = ttk.Label(details, text="", style="Accent.TLabel")
        active_label.pack(anchor=tk.W, pady=(16, 0))

        progress = ttk.Progressbar(dialog, mode="determinate", maximum=100)
        progress.pack(fill=tk.X, padx=16, pady=(12, 3))
        ttk.Label(dialog, textvariable=progress_text, style="Secondary.TLabel").pack(anchor=tk.W, padx=16)

        footer = ttk.Frame(dialog, padding=16)
        footer.pack(fill=tk.X)
        status_label = ttk.Label(footer, textvariable=status)
        status_label.pack(side=tk.LEFT)
        close_button = ttk.Button(footer, text="Закрыть", command=dialog.destroy)
        close_button.pack(side=tk.RIGHT)
        remove_button = ttk.Button(footer, text="Удалить")
        remove_button.pack(side=tk.RIGHT, padx=(0, 6))
        install_button = ttk.Button(footer, text="Скачать и выбрать", style="Primary.TButton")
        install_button.pack(side=tk.RIGHT, padx=(0, 6))
        cancel_button = ttk.Button(footer, text="Отменить загрузку", state=tk.DISABLED, command=cancel_event.set)
        cancel_button.pack(side=tk.RIGHT, padx=(0, 6))

        def backend_value() -> str:
            label = backend_box.get()
            return next((key for key, value in BACKEND_LABELS.items() if value == label), "auto")

        def current_model():
            selection = tree.selection()
            return MODEL_BY_ID.get(selection[0]) if selection else None

        def refresh_rows() -> None:
            active = self.active_local_model_id()
            for model in MODEL_CATALOG:
                complete = store.model_installed(model) and store.accelerator_installed()
                state = "Проверено" if complete else "Не загружена"
                if model.model_id == active and complete:
                    state = "Выбрана · Проверено"
                values = (f"{model.size_gb + LCM_ACCELERATOR.size_gb:.1f} ГБ", state)
                if tree.exists(model.model_id):
                    tree.item(model.model_id, text=model.name, values=values, tags=("verified" if complete else "missing",))
                else:
                    tree.insert("", tk.END, iid=model.model_id, text=model.name, values=values, tags=("verified" if complete else "missing",))

        def show_details(_event=None) -> None:
            model = current_model()
            if model is None:
                return
            name_text.set(model.name)
            description_text.set(model.description)
            license_text.set(
                f"Лицензия модели: {model.license_name}\n"
                f"Ускоритель LCM: {LCM_ACCELERATOR.license_name}, {LCM_ACCELERATOR.size_gb:.1f} ГБ"
            )
            source_button.configure(command=lambda: webbrowser.open(model.source_url))
            active_label.configure(text="Активная модель" if model.model_id == self.active_local_model_id() else "")
            remove_button.configure(state=tk.NORMAL if store.model_installed(model) else tk.DISABLED)

        def set_busy(value: bool) -> None:
            install_button.configure(state=tk.DISABLED if value else tk.NORMAL)
            remove_button.configure(state=tk.DISABLED if value else tk.NORMAL)
            backend_box.configure(state=tk.DISABLED if value else "readonly")
            cancel_button.configure(state=tk.NORMAL if value else tk.DISABLED)
            close_button.configure(state=tk.DISABLED if value else tk.NORMAL)

        def update_progress(stage: str, done: int, total: int) -> None:
            def apply_progress() -> None:
                if not dialog.winfo_exists():
                    return
                percent = 100.0 * done / max(1, total)
                progress.configure(value=percent)
                progress_text.set(f"{stage}: {done / 1024 ** 2:.0f} из {total / 1024 ** 2:.0f} МБ")
            self.after(0, apply_progress)

        def install() -> None:
            model = current_model()
            if model is None:
                return
            configured = backend_value()
            backend = profile.recommended_backend if configured == "auto" else configured
            cancel_event.clear()
            set_busy(True)
            status.set("Подготовка движка и модели...")
            status_label.configure(style="TLabel")
            progress.configure(value=0)

            def worker():
                try:
                    return store.ensure_ready(model.model_id, backend, update_progress, cancel_event)
                except Exception as exc:
                    return exc

            def done(result) -> None:
                set_busy(False)
                if isinstance(result, Exception):
                    status.set(str(result))
                    status_label.configure(style="Danger.TLabel")
                    return
                self.generative_settings.update({
                    "provider": "local", "local_model_id": model.model_id,
                    "local_backend": configured, "performance_profile": "balanced",
                    "steps": 6, "cfg_scale": 1.5, "sampler": "LCM",
                })
                self.save_settings()
                status.set("Модель установлена, проверена и выбрана")
                status_label.configure(style="Success.TLabel")
                progress.configure(value=100)
                refresh_rows()
                show_details()

            self.run_background("Установка локальной модели", worker, done, lambda: dialog.winfo_exists())

        def remove() -> None:
            model = current_model()
            if model is None or not store.model_installed(model):
                return
            if not messagebox.askyesno("Удаление модели", f"Удалить {model.name} с диска?", parent=dialog):
                return
            shutdown_local_servers()
            store.remove_model(model)
            status.set("Модель удалена")
            status_label.configure(style="TLabel")
            refresh_rows()
            show_details()

        def select_row(_event=None) -> None:
            show_details()

        tree.bind("<<TreeviewSelect>>", select_row)
        backend_box.bind("<<ComboboxSelected>>", lambda _event: selected_backend.set(backend_value()))
        install_button.configure(command=install)
        remove_button.configure(command=remove)
        refresh_rows()
        tree.selection_set(self.active_local_model_id())
        tree.focus(self.active_local_model_id())
        show_details()
        self._model_manager_tree = tree
        self._model_manager_install = install
        self._model_manager_backend = backend_box
        self._model_manager_status = status
        self.center_toplevel(dialog, 980, 680)
