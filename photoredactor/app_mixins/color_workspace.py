from __future__ import annotations

import base64

from ..app_shared import *


INTENT_LABELS = {
    "Перцепционный": "perceptual",
    "Относительный колориметрический": "relative",
    "Абсолютный колориметрический": "absolute",
    "Насыщенность": "saturation",
}


class ColorWorkspaceMixin:
    def current_proof_profile(self) -> bytes | None:
        settings = color_settings(self.doc.metadata)
        encoded = settings.get("proof_icc_base64")
        if not encoded:
            return None
        try:
            return base64.b64decode(encoded)
        except (ValueError, TypeError):
            return None

    def toggle_soft_proof(self) -> None:
        settings = color_settings(self.doc.metadata)
        if not settings.get("proof_icc_base64"):
            self.color_proof_workspace()
            return
        settings["soft_proof_enabled"] = not bool(settings.get("soft_proof_enabled", False))
        self.doc.dirty = True
        self._soft_proof_cache = None
        self.invalidate_view()
        self.refresh_canvas()
        state = "включена" if settings["soft_proof_enabled"] else "выключена"
        self.status_text(f"Цветопроба {state}")

    def apply_soft_proof_display(self, source: np.ndarray) -> np.ndarray:
        settings = color_settings(self.doc.metadata)
        profile = self.current_proof_profile()
        if not settings.get("soft_proof_enabled", False) or profile is None:
            return source
        signature = (
            self.render_engine.render_revision,
            settings.get("proof_profile_name"),
            settings.get("proof_rendering_intent", "relative"),
            bool(settings.get("proof_black_point_compensation", True)),
            bool(settings.get("gamut_warning", False)),
            float(settings.get("gamut_threshold", 8.0)),
        )
        cached = getattr(self, "_soft_proof_cache", None)
        if cached is not None and cached[0] == signature:
            return cached[1]
        proofed, warning = proof_document(
            self.doc,
            profile,
            str(settings.get("proof_rendering_intent", "relative")),
            bool(settings.get("proof_black_point_compensation", True)),
            bool(settings.get("gamut_warning", False)),
            float(settings.get("gamut_threshold", 8.0)),
        )
        alpha = proofed[:, :, 3:4].astype(np.float32) / 255.0
        result = source.copy()
        result[:, :, :3] = np.clip(
            proofed[:, :, :3].astype(np.float32) * alpha + source[:, :, :3].astype(np.float32) * (1.0 - alpha),
            0,
            255,
        ).astype(np.uint8)
        if bool(settings.get("gamut_warning", False)):
            active = warning > 0
            result[active, :3] = proofed[active, :3]
        self._soft_proof_cache = signature, result
        return result

    def color_proof_workspace(self) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("Цветопроба и подготовка к печати")
        dialog.transient(self)
        dialog.grab_set()
        settings = color_settings(self.doc.metadata)
        profile_path = tk.StringVar(value=str(settings.get("proof_profile_path", "")))
        intent_label = next(
            (label for label, value in INTENT_LABELS.items() if value == settings.get("proof_rendering_intent", "relative")),
            "Относительный колориметрический",
        )
        intent = tk.StringVar(value=intent_label)
        black_point = tk.BooleanVar(value=bool(settings.get("proof_black_point_compensation", True)))
        warning_enabled = tk.BooleanVar(value=bool(settings.get("gamut_warning", False)))
        warning_threshold = tk.DoubleVar(value=float(settings.get("gamut_threshold", 8.0)))
        ink_limit = tk.DoubleVar(value=float(settings.get("ink_limit", 300.0)))

        header = ttk.Frame(dialog, padding=12)
        header.pack(fill=tk.X)
        ttk.Label(header, text="Печатный ICC-профиль").grid(row=0, column=0, sticky="w")
        profile_entry = ttk.Entry(header, textvariable=profile_path, state="readonly")
        profile_entry.grid(row=1, column=0, sticky="ew", pady=(4, 0))

        def choose_profile() -> None:
            path = filedialog.askopenfilename(
                title="Выберите CMYK ICC-профиль печати",
                filetypes=[("ICC-профили", "*.icc *.icm"), ("Все файлы", "*.*")],
                parent=dialog,
            )
            if not path:
                return
            try:
                details = profile_details(path)
                if details["color_space"].upper() != "CMYK":
                    raise ValueError("Выбранный профиль не является CMYK-профилем печати")
            except Exception as exc:
                messagebox.showerror("ICC-профиль", str(exc), parent=dialog)
                return
            profile_path.set(path)
            update_preview()

        ttk.Button(header, text="Выбрать...", command=choose_profile).grid(row=1, column=1, padx=(8, 0), pady=(4, 0))
        header.columnconfigure(0, weight=1)

        options = ttk.Frame(dialog, padding=(12, 0, 12, 8))
        options.pack(fill=tk.X)
        ttk.Label(options, text="Метод преобразования").grid(row=0, column=0, sticky="w")
        ttk.Combobox(options, textvariable=intent, values=tuple(INTENT_LABELS), state="readonly", width=34).grid(row=1, column=0, sticky="w", pady=(3, 0))
        ttk.Checkbutton(options, text="Компенсация точки чёрного", variable=black_point).grid(row=1, column=1, sticky="w", padx=16)
        ttk.Checkbutton(options, text="Показать цвета вне охвата", variable=warning_enabled).grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Label(options, text="Порог ΔE").grid(row=2, column=1, sticky="e", pady=(8, 0))
        ttk.Spinbox(options, from_=1.0, to=30.0, increment=0.5, textvariable=warning_threshold, width=7).grid(row=2, column=2, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(options, text="Лимит суммарной краски, %").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(options, from_=100.0, to=400.0, increment=5.0, textvariable=ink_limit, width=8).grid(row=3, column=1, sticky="w", padx=16, pady=(8, 0))

        previews = ttk.Frame(dialog, padding=(12, 0))
        previews.pack(fill=tk.BOTH, expand=True)
        original_label = ttk.Label(previews, anchor=tk.CENTER)
        proof_label = ttk.Label(previews, anchor=tk.CENTER)
        for column, (title, widget) in enumerate((("Оригинал", original_label), ("Цветопроба", proof_label))):
            ttk.Label(previews, text=title).grid(row=0, column=column, pady=(0, 4))
            widget.grid(row=1, column=column, sticky="nsew", padx=5)
            previews.columnconfigure(column, weight=1)
        previews.rowconfigure(1, weight=1)
        report = ttk.Label(dialog, padding=(16, 8), justify=tk.LEFT, style="Secondary.TLabel")
        report.pack(fill=tk.X)
        original = self.doc.composite(False)
        preview_source = original
        preview_scale = min(1.0, 960.0 / max(original.shape[1], 1), 780.0 / max(original.shape[0], 1))
        if preview_scale < 1.0:
            preview_source = cv2.resize(
                original,
                (max(1, round(original.shape[1] * preview_scale)), max(1, round(original.shape[0] * preview_scale))),
                interpolation=cv2.INTER_AREA,
            )
        preflight_cache: dict[tuple[object, ...], dict[str, object]] = {}

        def preview_photo(pixels: np.ndarray) -> ImageTk.PhotoImage:
            image = rgba_array_to_pil(pixels)
            image.thumbnail((470, 300), Image.Resampling.LANCZOS)
            canvas = Image.new("RGBA", (470, 300), (44, 46, 52, 255))
            canvas.alpha_composite(image, ((470 - image.width) // 2, (300 - image.height) // 2))
            return ImageTk.PhotoImage(canvas)

        def update_preview(*_args) -> None:
            path = profile_path.get()
            self._color_workspace_images = [preview_photo(preview_source)]
            original_label.configure(image=self._color_workspace_images[0])
            if not path or not Path(path).exists():
                proof_label.configure(image="")
                report.configure(text="Выберите CMYK ICC-профиль печатной машины или стандарта печати.")
                return
            try:
                proofed = soft_proof_rgba(
                    preview_source,
                    document_source_profile(self.doc),
                    path,
                    "sRGB",
                    INTENT_LABELS[intent.get()],
                    black_point.get(),
                )
                warning = gamut_warning_mask(preview_source, proofed, warning_threshold.get())
                proof_preview = display_rgba(proofed)
                if warning_enabled.get() and np.any(warning):
                    proof_preview[warning > 0, :3] = (255, 0, 210)
                preflight_key = (path, INTENT_LABELS[intent.get()], black_point.get(), float(ink_limit.get()))
                if preflight_key not in preflight_cache:
                    preflight_cache[preflight_key] = print_preflight(
                        self.doc,
                        path,
                        INTENT_LABELS[intent.get()],
                        black_point.get(),
                        ink_limit.get(),
                    )
                preflight = preflight_cache[preflight_key]
                self._color_workspace_images.append(preview_photo(proof_preview))
                proof_label.configure(image=self._color_workspace_images[1])
                issue_text = "; ".join(preflight["issues"]) if preflight["issues"] else "критических замечаний нет"
                report.configure(
                    text=(
                        f"{preflight['profile']['name']}  |  {self.doc.dpi} DPI  |  "
                        f"Макс. краска: {preflight['maximum_ink']:.1f}%  |  "
                        f"Вне охвата: {np.count_nonzero(warning) / warning.size:.1%} предпросмотра\n{issue_text}"
                    )
                )
                self._color_workspace_preflight = preflight
            except Exception as exc:
                proof_label.configure(image="")
                report.configure(text=f"Ошибка цветопробы: {exc}")

        def save_settings(enable: bool) -> bool:
            path = profile_path.get()
            if not path or not Path(path).exists():
                messagebox.showinfo("Цветопроба", "Сначала выберите CMYK ICC-профиль.", parent=dialog)
                return False
            details = profile_details(path)
            settings.update(
                {
                    "proof_profile_path": path,
                    "proof_profile_name": details["name"],
                    "proof_icc_base64": base64.b64encode(Path(path).read_bytes()).decode("ascii"),
                    "proof_rendering_intent": INTENT_LABELS[intent.get()],
                    "proof_black_point_compensation": bool(black_point.get()),
                    "gamut_warning": bool(warning_enabled.get()),
                    "gamut_threshold": float(warning_threshold.get()),
                    "ink_limit": float(ink_limit.get()),
                    "soft_proof_enabled": bool(enable),
                }
            )
            self.doc.dirty = True
            self._soft_proof_cache = None
            return True

        def enable_proof() -> None:
            if save_settings(True):
                dialog.destroy()
                self.invalidate_view()
                self.refresh_canvas()

        def export_tiff() -> None:
            if not save_settings(settings.get("soft_proof_enabled", False)):
                return
            path = filedialog.asksaveasfilename(
                title="Экспорт CMYK TIFF",
                defaultextension=".tif",
                filetypes=[("CMYK TIFF", "*.tif *.tiff")],
                parent=dialog,
            )
            if not path:
                return
            try:
                export_cmyk_tiff(self.doc, path, profile_path.get(), INTENT_LABELS[intent.get()], black_point.get())
                messagebox.showinfo("Печать", f"CMYK TIFF сохранён:\n{path}", parent=dialog)
            except Exception as exc:
                messagebox.showerror("Печать", str(exc), parent=dialog)

        footer = ttk.Frame(dialog, padding=12)
        footer.pack(fill=tk.X)
        ttk.Button(footer, text="Включить цветопробу", command=enable_proof, style="Primary.TButton").pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(footer, text="Экспорт CMYK TIFF...", command=export_tiff).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(footer, text="Закрыть", command=dialog.destroy).pack(side=tk.RIGHT)
        self._color_workspace_profile = profile_path
        self._color_workspace_intent = intent
        self._color_workspace_warning = warning_enabled
        self._color_workspace_update = update_preview
        self._color_workspace_enable = enable_proof
        self._color_workspace_dialog = dialog
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        self.center_toplevel(dialog, 1020, 640)
        for variable in (intent, black_point, warning_enabled, warning_threshold, ink_limit):
            variable.trace_add("write", update_preview)
        update_preview()
