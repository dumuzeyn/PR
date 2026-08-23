from __future__ import annotations

import threading

from ..app_shared import *
from ..local_generative import PERFORMANCE_LABELS, PERFORMANCE_VALUES, SAMPLERS


STYLE_LABELS = {
    "Без пресета": "",
    "Фотография": "photographic",
    "Кино": "cinematic",
    "Цифровое искусство": "digital-art",
    "Аниме": "anime",
    "Комикс": "comic-book",
    "Фэнтези": "fantasy-art",
    "Линейный рисунок": "line-art",
    "Аналоговая плёнка": "analog-film",
    "Пиксель-арт": "pixel-art",
    "3D-модель": "3d-model",
}
class GenerativeWorkspaceMixin:
    @staticmethod
    def _window_exists(window) -> bool:
        try:
            return bool(window.winfo_exists())
        except tk.TclError:
            return False

    @staticmethod
    def _generative_safe_call(operation):
        try:
            return operation()
        except Exception as exc:
            return exc

    def generative_fill_dialog(self) -> None:
        self._open_generative_workspace("fill")

    def generative_expand_dialog(self) -> None:
        self._open_generative_workspace("expand")

    def _generation_source(self, operation: str, existing: Layer | None) -> tuple[np.ndarray, np.ndarray | None, tuple[int, int, int, int] | None]:
        data = {} if existing is None else existing.generation_data or {}
        if existing is not None:
            visible = existing.visible
            existing.visible = False
            try:
                composite = self.doc.composite(False)
            finally:
                existing.visible = visible
        else:
            composite = self.render_engine.render(self.doc, False).copy()
        if operation == "fill":
            mask = self.doc.selection_mask
            if existing is not None and existing.pixels.shape[:2] == composite.shape[:2]:
                mask = existing.pixels[:, :, 3]
            return composite, None if mask is None else mask.copy(), None
        raw = data.get("margins", [0, 0, 0, 0])
        margins = tuple(max(0, int(value)) for value in raw)
        if existing is not None and any(margins):
            left, top, right, bottom = margins
            composite = composite[top:composite.shape[0] - bottom or None, left:composite.shape[1] - right or None].copy()
        return composite, None, margins if any(margins) else None

    def _open_generative_workspace(self, operation: str, existing: Layer | None = None) -> None:
        source, mask, stored_margins = self._generation_source(operation, existing)
        if operation == "fill" and (mask is None or not np.any(mask)):
            messagebox.showinfo("Генеративная заливка", "Сначала создайте выделение.")
            return
        data = {} if existing is None else existing.generation_data or {}
        dialog = tk.Toplevel(self)
        dialog.title("Генеративная заливка" if operation == "fill" else "Генеративное расширение")
        dialog.transient(self)
        dialog.grab_set()
        dialog.minsize(980, 680)
        prompt = tk.StringVar(value=str(data.get("prompt", "")))
        negative = tk.StringVar(value=str(data.get("negative_prompt", "")))
        seed = tk.IntVar(value=int(data.get("seed", 0)))
        variants_count = tk.IntVar(value=int(self.generative_settings.get("variants", 3)))
        style_slug = str(data.get("style", self.generative_settings.get("style", "photographic")))
        style = tk.StringVar(value=next((label for label, slug in STYLE_LABELS.items() if slug == style_slug), "Без пресета"))
        creativity = tk.DoubleVar(value=float(data.get("creativity", self.generative_settings.get("creativity", 0.5))))
        steps = tk.IntVar(value=int(data.get("steps", self.generative_settings.get("steps", 22))))
        cfg_scale = tk.DoubleVar(value=float(data.get("cfg_scale", self.generative_settings.get("cfg_scale", 7.0))))
        strength = tk.DoubleVar(value=float(data.get("strength", self.generative_settings.get("strength", 0.88))))
        sampler = tk.StringVar(value=str(data.get("sampler", self.generative_settings.get("sampler", "DPM++ 2M"))))
        profile_slug = str(data.get("performance_profile", self.generative_settings.get("performance_profile", "balanced")))
        performance = tk.StringVar(value=next((label for label, slug in PERFORMANCE_LABELS.items() if slug == profile_slug), "Вручную"))
        default_margins = stored_margins or (max(64, source.shape[1] // 8), max(64, source.shape[0] // 8)) * 2
        margin_vars = [tk.IntVar(value=value) for value in default_margins]
        history: list[tuple[GeneratedVariant, dict[str, object]]] = []
        selected = tk.IntVar(value=-1)
        preview_photos: list[ImageTk.PhotoImage] = []
        generation_cancel = threading.Event()

        header = ttk.Frame(dialog, padding=(12, 10, 12, 6))
        header.pack(fill=tk.X)
        ttk.Label(header, text=dialog.title(), style="PanelTitle.TLabel").pack(side=tk.LEFT)
        status = ttk.Label(header, text="Локальная модель", style="Secondary.TLabel")
        status.pack(side=tk.RIGHT)
        body = ttk.PanedWindow(dialog, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=12)
        preview_panel = ttk.Frame(body)
        controls = ttk.Frame(body, width=300, padding=(12, 4))
        body.add(preview_panel, weight=1)
        body.add(controls, weight=0)
        canvas = tk.Canvas(preview_panel, background=TOKENS.WORKSPACE_BG, highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        variants_bar = ttk.Frame(preview_panel, padding=(0, 8, 0, 0))
        variants_bar.pack(fill=tk.X)

        ttk.Label(controls, text="Запрос", style="PanelTitle.TLabel").pack(anchor=tk.W)
        prompt_entry = ttk.Entry(controls, textvariable=prompt)
        prompt_entry.pack(fill=tk.X, pady=(3, 10))
        ttk.Label(controls, text="Исключить", style="Secondary.TLabel").pack(anchor=tk.W)
        ttk.Entry(controls, textvariable=negative).pack(fill=tk.X, pady=(3, 10))
        ttk.Label(controls, text="Стиль", style="Secondary.TLabel").pack(anchor=tk.W)
        ttk.Combobox(controls, textvariable=style, values=list(STYLE_LABELS), state="readonly").pack(fill=tk.X, pady=(3, 10))
        seed_row = ttk.Frame(controls)
        seed_row.pack(fill=tk.X)
        ttk.Label(seed_row, text="Seed", style="Secondary.TLabel").pack(side=tk.LEFT)
        ttk.Spinbox(seed_row, textvariable=seed, from_=0, to=MAX_SEED, width=14).pack(side=tk.RIGHT)
        ttk.Button(controls, text="Новый случайный seed", command=lambda: seed.set(variant_seeds(0, 1)[0])).pack(fill=tk.X, pady=(5, 10))
        count_row = ttk.Frame(controls)
        count_row.pack(fill=tk.X)
        ttk.Label(count_row, text="Варианты", style="Secondary.TLabel").pack(side=tk.LEFT)
        ttk.Spinbox(count_row, textvariable=variants_count, from_=1, to=4, width=6).pack(side=tk.RIGHT)

        local_controls = ttk.Frame(controls)
        local_controls.pack(fill=tk.X, pady=(14, 0))
        model_row = ttk.Frame(local_controls)
        model_row.pack(fill=tk.X)
        ttk.Label(model_row, text="Локальная модель", style="Secondary.TLabel").pack(side=tk.LEFT)
        ttk.Button(model_row, text="Выбрать...", command=self.model_manager_dialog).pack(side=tk.RIGHT)
        profile_row = ttk.Frame(local_controls)
        profile_row.pack(fill=tk.X, pady=(7, 0))
        ttk.Label(profile_row, text="Скорость и качество").pack(side=tk.LEFT)
        profile_box = ttk.Combobox(profile_row, textvariable=performance, values=list(PERFORMANCE_LABELS), state="readonly", width=14)
        profile_box.pack(side=tk.RIGHT)
        for label, variable, start, end in (
            ("Шаги", steps, 4, 80), ("Точность запроса (CFG)", cfg_scale, 1.0, 20.0),
            ("Сила изменения", strength, 0.05, 1.0),
        ):
            row = ttk.Frame(local_controls)
            row.pack(fill=tk.X, pady=(7, 0))
            ttk.Label(row, text=label).pack(side=tk.LEFT)
            ttk.Spinbox(row, textvariable=variable, from_=start, to=end, increment=1 if variable is steps else 0.05, width=7).pack(side=tk.RIGHT)
        ttk.Combobox(local_controls, textvariable=sampler, values=list(SAMPLERS), state="readonly").pack(fill=tk.X, pady=(7, 0))

        changing_profile = [False]

        def apply_performance(_event=None) -> None:
            values = PERFORMANCE_VALUES.get(PERFORMANCE_LABELS.get(performance.get(), "custom"))
            if values is None:
                return
            changing_profile[0] = True
            steps.set(values[0])
            cfg_scale.set(values[1])
            sampler.set(values[2])
            changing_profile[0] = False

        def mark_manual(*_args) -> None:
            if not changing_profile[0]:
                performance.set("Вручную")

        profile_box.bind("<<ComboboxSelected>>", apply_performance)
        for variable in (steps, cfg_scale, sampler):
            variable.trace_add("write", mark_manual)

        if operation == "expand":
            ttk.Label(controls, text="Расширение, px", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(16, 5))
            for label, variable in zip(("Слева", "Сверху", "Справа", "Снизу"), margin_vars):
                row = ttk.Frame(controls)
                row.pack(fill=tk.X, pady=2)
                ttk.Label(row, text=label).pack(side=tk.LEFT)
                ttk.Spinbox(row, textvariable=variable, from_=0, to=100000, width=10).pack(side=tk.RIGHT)
            ttk.Label(controls, text="Творческая свобода", style="Secondary.TLabel").pack(anchor=tk.W, pady=(12, 2))
            AccentScale(controls, variable=creativity, from_=0.0, to=1.0).pack(fill=tk.X)

        action_row = ttk.Frame(controls)
        action_row.pack(fill=tk.X, pady=(20, 0))
        generate_button = ttk.Button(action_row, text="Создать варианты", style="Primary.TButton")
        generate_button.pack(fill=tk.X)
        repeat_button = ttk.Button(action_row, text="Повторить выбранный seed", state=tk.DISABLED)
        repeat_button.pack(fill=tk.X, pady=(6, 0))
        cancel_button = ttk.Button(action_row, text="Отменить генерацию", state=tk.DISABLED, command=generation_cancel.set)
        cancel_button.pack(fill=tk.X, pady=(6, 0))
        status.configure(text=self.active_local_model_name())
        footer = ttk.Frame(dialog, padding=12)
        footer.pack(fill=tk.X)
        apply_button = ttk.Button(footer, text="Применить", state=tk.DISABLED, style="Primary.TButton")
        apply_button.pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(footer, text="Закрыть", command=lambda: (generation_cancel.set(), dialog.destroy())).pack(side=tk.RIGHT)

        def current_margins() -> tuple[int, int, int, int]:
            values: list[int] = []
            for variable in margin_vars:
                try:
                    values.append(max(0, int(variable.get())))
                except (tk.TclError, ValueError):
                    values.append(0)
            return tuple(values)

        def preview_array() -> np.ndarray:
            index = selected.get()
            if 0 <= index < len(history):
                variant = history[index][0]
                if operation == "fill":
                    overlay = rgba_array_to_pil(self._generated_layer_pixels(variant.pixels, mask))
                    return np.asarray(Image.alpha_composite(rgba_array_to_pil(source), overlay), dtype=np.uint8)
                return variant.pixels
            if operation == "expand":
                left, top, right, bottom = current_margins()
                target_width = source.shape[1] + left + right
                target_height = source.shape[0] + top + bottom
                scale = min(1.0, 900 / max(1, target_width), 600 / max(1, target_height))
                frame_width = max(1, round(target_width * scale))
                frame_height = max(1, round(target_height * scale))
                frame = np.zeros((frame_height, frame_width, 4), dtype=np.uint8)
                source_width = max(1, round(source.shape[1] * scale))
                source_height = max(1, round(source.shape[0] * scale))
                source_preview = cv2.resize(source, (source_width, source_height), interpolation=cv2.INTER_AREA)
                preview_left, preview_top = round(left * scale), round(top * scale)
                paste_width = min(source_width, frame_width - preview_left)
                paste_height = min(source_height, frame_height - preview_top)
                if paste_width > 0 and paste_height > 0:
                    frame[preview_top:preview_top + paste_height, preview_left:preview_left + paste_width] = source_preview[:paste_height, :paste_width]
                return frame
            overlay = source.copy()
            overlay[mask > 0, :3] = np.clip(overlay[mask > 0, :3] * 0.45 + np.array((35, 145, 235)) * 0.55, 0, 255)
            return overlay

        def redraw(*_args) -> None:
            array = preview_array()
            width, height = max(200, canvas.winfo_width()), max(200, canvas.winfo_height())
            scale = min(width / array.shape[1], height / array.shape[0])
            image = rgba_array_to_pil(array)
            image = image.resize((max(1, round(array.shape[1] * scale)), max(1, round(array.shape[0] * scale))), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self._generative_workspace_photo = photo
            canvas.delete("all")
            canvas.create_image(width / 2, height / 2, image=photo)

        def select_variant(index: int) -> None:
            selected.set(index)
            variant, _settings = history[index]
            status.configure(text=f"Вариант {index + 1}  |  seed {variant.seed}")
            apply_button.configure(state=tk.NORMAL)
            repeat_button.configure(state=tk.NORMAL)
            redraw()

        def rebuild_variant_buttons() -> None:
            for child in variants_bar.winfo_children():
                child.destroy()
            preview_photos.clear()
            for index, (variant, _settings) in enumerate(history[-16:]):
                actual_index = len(history) - min(16, len(history)) + index
                thumb = rgba_array_to_pil(variant.pixels)
                thumb.thumbnail((72, 48), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(thumb)
                preview_photos.append(photo)
                ttk.Button(
                    variants_bar, text=str(variant.seed), image=photo, compound=tk.TOP,
                    command=lambda value=actual_index: select_variant(value), width=11,
                ).pack(side=tk.LEFT, padx=(0, 4))

        def request_settings(request_seed: int | None = None) -> dict[str, object]:
            try:
                current_seed = int(seed.get() if request_seed is None else request_seed)
                current_creativity = float(creativity.get())
            except (tk.TclError, TypeError, ValueError) as exc:
                raise GenerativeAPIError("Seed и творческая свобода должны быть числами") from exc
            return {
                "operation": operation,
                "provider": "local",
                "prompt": prompt.get().strip(),
                "negative_prompt": negative.get().strip(),
                "seed": current_seed,
                "style": STYLE_LABELS.get(style.get(), ""),
                "creativity": current_creativity,
                "margins": list(current_margins()) if operation == "expand" else None,
                "local_model_id": self.active_local_model_id(),
                "local_backend": self.effective_local_backend(),
                "performance_profile": PERFORMANCE_LABELS.get(performance.get(), "custom"),
                "steps": int(steps.get()),
                "cfg_scale": float(cfg_scale.get()),
                "strength": float(strength.get()),
                "sampler": sampler.get(),
            }

        def generate(forced_seed: int | None = None) -> None:
            try:
                settings = request_settings(forced_seed)
                count = 1 if forced_seed is not None else max(1, min(4, int(variants_count.get())))
            except (GenerativeAPIError, tk.TclError, ValueError) as exc:
                messagebox.showinfo("Генеративный ИИ", str(exc), parent=dialog)
                return
            if not settings["prompt"]:
                messagebox.showinfo(dialog.title(), "Введите, что должно появиться в выбранной области.", parent=dialog)
                return
            margins = tuple(settings["margins"] or ())
            if operation == "expand":
                try:
                    validate_outpaint_dimensions(source, margins)
                except GenerativeAPIError as exc:
                    messagebox.showinfo("Генеративное расширение", str(exc), parent=dialog)
                    return
            if not self.local_generation_ready():
                self.model_manager_dialog()
                return
            generate_button.configure(state=tk.DISABLED)
            cancel_button.configure(state=tk.NORMAL)
            generation_cancel.clear()
            status.configure(text="Создание вариантов...")

            def worker():
                return self._generative_safe_call(
                    lambda: self.local_generation_variants(operation, source, mask, margins, settings, count, generation_cancel)
                )

            def done(result) -> None:
                generate_button.configure(state=tk.NORMAL)
                cancel_button.configure(state=tk.DISABLED)
                if isinstance(result, Exception):
                    status.configure(text="Ошибка генерации")
                    messagebox.showerror("Генеративный ИИ", str(result), parent=dialog)
                    return
                for variant in result:
                    variant_settings = {**settings, "seed": variant.seed}
                    history.append((variant, variant_settings))
                rebuild_variant_buttons()
                select_variant(len(history) - len(result))

            self.run_background("Генеративный ИИ", worker, done, lambda: self._window_exists(dialog))

        def repeat() -> None:
            index = selected.get()
            if 0 <= index < len(history):
                generate(history[index][0].seed)

        def apply() -> None:
            index = selected.get()
            if not 0 <= index < len(history):
                return
            variant, settings = history[index]
            settings = {**settings, "variant_seeds": [item[0].seed for item in history]}
            if operation == "fill":
                self._apply_generative_fill(variant.pixels, mask, settings, existing)
            else:
                self._apply_generative_expand(variant.pixels, tuple(settings["margins"]), settings, existing)
            self.generative_settings.update({
                "provider": str(settings["provider"]),
                "local_model_id": str(settings["local_model_id"]),
                "local_backend": str(settings["local_backend"]),
                "performance_profile": str(settings["performance_profile"]),
                "variants": max(1, min(4, int(variants_count.get()))),
                "style": str(settings["style"]),
                "creativity": float(settings["creativity"]),
                "steps": int(settings["steps"]),
                "cfg_scale": float(settings["cfg_scale"]),
                "strength": float(settings["strength"]),
                "sampler": str(settings["sampler"]),
            })
            self.save_settings()
            dialog.destroy()

        generate_button.configure(command=generate)
        repeat_button.configure(command=repeat)
        apply_button.configure(command=apply)
        canvas.bind("<Configure>", redraw)
        for variable in margin_vars:
            variable.trace_add("write", redraw)
        self._generative_workspace_generate = generate
        self._generative_workspace_history = history
        self._generative_workspace_apply = apply
        self._generative_workspace_selected = selected
        self._generative_workspace_prompt = prompt
        self._generative_workspace_negative = negative
        self._generative_workspace_seed = seed
        self._generative_workspace_variants = variants_count
        self._generative_workspace_steps = steps
        self._generative_workspace_cfg = cfg_scale
        self._generative_workspace_strength = strength
        self._generative_workspace_sampler = sampler
        self._generative_workspace_performance = performance
        self._generative_workspace_profile_box = profile_box
        self._generative_workspace_margins = margin_vars
        self._generative_workspace_dialog = dialog
        self.center_toplevel(dialog, 1120, 760)
        redraw()
