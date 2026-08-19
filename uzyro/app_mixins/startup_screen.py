from __future__ import annotations

from ..app_shared import *
from ..brand import branding_asset
from ..security.files import load_pillow_image, validate_dimensions


class StartupScreenMixin:
    @staticmethod
    def read_clipboard_image() -> Image.Image | None:
        try:
            value = ImageGrab.grabclipboard()
            if isinstance(value, Image.Image):
                validate_dimensions(value.width, value.height, copies=3)
                return value.convert("RGBA")
            if isinstance(value, list):
                for path in value:
                    suffix = Path(path).suffix.lower()
                    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
                        return load_pillow_image(path)
        except (OSError, ValueError):
            pass
        return None

    def show_start_screen(self) -> None:
        self._editor_active = False
        self.editor_root.pack_forget()
        self.status_frame.pack_forget()
        self.config(menu="")
        self.minsize(820, 560)
        self.title("UZYRO")
        self.restore_centered_window(1060, 700)
        if self.startup_frame is not None and self.startup_frame.winfo_exists():
            self.startup_frame.destroy()

        clipboard_image = self.read_clipboard_image()
        self._startup_clipboard_image = clipboard_image
        frame = tk.Frame(self, background=TOKENS.SURFACE)
        frame.pack(fill=tk.BOTH, expand=True)
        self.startup_frame = frame

        header = tk.Frame(frame, background=TOKENS.BACKGROUND, height=88)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        try:
            source_icon = tk.PhotoImage(file=str(branding_asset("uzyro-icon.png")))
            self._startup_brand_icon = source_icon.subsample(8, 8)
            tk.Label(header, image=self._startup_brand_icon, background=TOKENS.BACKGROUND).pack(side=tk.LEFT, padx=(32, 14), pady=12)
        except tk.TclError:
            self._startup_brand_icon = None
        brand_text = tk.Frame(header, background=TOKENS.BACKGROUND)
        brand_text.pack(side=tk.LEFT, fill=tk.Y, pady=13)
        tk.Label(brand_text, text="UZYRO", font=("Segoe UI Semibold", 21), foreground=TOKENS.TEXT_PRIMARY, background=TOKENS.BACKGROUND).pack(anchor=tk.W)
        tk.Label(brand_text, text="Рабочее пространство", font=("Segoe UI", 9), foreground=TOKENS.TEXT_SECONDARY, background=TOKENS.BACKGROUND).pack(anchor=tk.W)

        content = tk.Frame(frame, background=TOKENS.SURFACE)
        content.pack(fill=tk.BOTH, expand=True, padx=36, pady=26)
        content.columnconfigure(0, minsize=300)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(1, weight=1)

        tk.Label(content, text="Начать", font=("Segoe UI Semibold", 14), foreground=TOKENS.TEXT_PRIMARY, background=TOKENS.SURFACE).grid(row=0, column=0, sticky="w", pady=(0, 10))
        actions = tk.Frame(content, background=TOKENS.SURFACE)
        actions.grid(row=1, column=0, sticky="new", padx=(0, 34))

        new_button = tk.Button(
            actions,
            text="Создать новый холст",
            command=self.new_document,
            anchor="w",
            font=("Segoe UI Semibold", 11),
            background=TOKENS.ACCENT,
            foreground="white",
            activebackground=TOKENS.ACCENT_HOVER,
            activeforeground="white",
            relief=tk.FLAT,
            padx=18,
            pady=13,
            cursor="hand2",
        )
        new_button.pack(fill=tk.X, pady=(0, 8))
        open_button = tk.Button(
            actions,
            text="Открыть изображение или проект",
            command=self.open_file,
            anchor="w",
            font=("Segoe UI", 10),
            background=TOKENS.SURFACE_HOVER,
            foreground=TOKENS.TEXT_PRIMARY,
            activebackground=TOKENS.SURFACE_SELECTED,
            activeforeground=TOKENS.TEXT_PRIMARY,
            relief=tk.FLAT,
            padx=18,
            pady=12,
            cursor="hand2",
        )
        open_button.pack(fill=tk.X, pady=(0, 8))
        ToolTip(new_button, "Открывает выбор размера, формата и фона нового холста.")
        ToolTip(open_button, "Открывает PNG, JPEG, WebP, BMP, TIFF и проекты UZYRO.")

        if clipboard_image is not None:
            clipboard_box = tk.Frame(actions, background=TOKENS.SURFACE_HOVER, padx=12, pady=12)
            clipboard_box.pack(fill=tk.X, pady=(12, 8))
            preview = clipboard_image.copy()
            preview.thumbnail((72, 72), Image.Resampling.LANCZOS)
            preview_canvas = Image.new("RGBA", (72, 72), (34, 35, 41, 255))
            preview_canvas.alpha_composite(preview, ((72 - preview.width) // 2, (72 - preview.height) // 2))
            self._startup_clipboard_preview = ImageTk.PhotoImage(preview_canvas)
            tk.Label(clipboard_box, image=self._startup_clipboard_preview, background=TOKENS.SURFACE_HOVER).pack(side=tk.LEFT, padx=(0, 12))
            clipboard_text = tk.Frame(clipboard_box, background=TOKENS.SURFACE_HOVER)
            clipboard_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            tk.Label(clipboard_text, text="Изображение в буфере", font=("Segoe UI Semibold", 10), foreground=TOKENS.TEXT_PRIMARY, background=TOKENS.SURFACE_HOVER).pack(anchor=tk.W)
            tk.Label(clipboard_text, text=f"{clipboard_image.width} x {clipboard_image.height} px", foreground=TOKENS.TEXT_SECONDARY, background=TOKENS.SURFACE_HOVER).pack(anchor=tk.W)
            paste_button = ttk.Button(clipboard_text, text="Открыть как новый холст", command=self.create_from_clipboard)
            paste_button.pack(anchor=tk.W, pady=(7, 0))
            ToolTip(paste_button, "Создает холст точно по размеру изображения в буфере.")

        if self.recovery_path.exists():
            ttk.Button(actions, text="Восстановить последнюю сессию", command=self.open_recovery).pack(fill=tk.X, pady=(12, 0))

        recent_header = tk.Frame(content, background=TOKENS.SURFACE)
        recent_header.grid(row=0, column=1, sticky="ew", pady=(0, 12))
        tk.Label(recent_header, text="Недавние файлы", font=("Segoe UI Semibold", 14), foreground=TOKENS.TEXT_PRIMARY, background=TOKENS.SURFACE).pack(side=tk.LEFT)
        ttk.Button(recent_header, text="Очистить", command=self.clear_recent_files).pack(side=tk.RIGHT)

        recent_area = tk.Frame(content, background=TOKENS.SURFACE)
        recent_area.grid(row=1, column=1, sticky="nsew")
        recent_area.columnconfigure(0, weight=1)
        recent_area.rowconfigure(1, weight=1)
        tk.Label(recent_area, text="Имя файла  |  Тип  |  Изменен", anchor="w", padx=12, pady=8, foreground=TOKENS.TEXT_SECONDARY, background=TOKENS.SURFACE_HOVER).grid(row=0, column=0, sticky="ew")
        self.startup_recent_list = tk.Listbox(
            recent_area,
            exportselection=False,
            activestyle="none",
            background=TOKENS.SURFACE,
            foreground=TOKENS.TEXT_PRIMARY,
            selectbackground=TOKENS.SURFACE_SELECTED,
            selectforeground=TOKENS.TEXT_PRIMARY,
            highlightthickness=0,
            borderwidth=0,
            font=("Segoe UI", 10),
        )
        recent_scroll = ttk.Scrollbar(recent_area, orient=tk.VERTICAL, command=self.startup_recent_list.yview)
        self.startup_recent_list.configure(yscrollcommand=recent_scroll.set)
        self.startup_recent_list.grid(row=1, column=0, sticky="nsew")
        recent_scroll.grid(row=1, column=1, sticky="ns")
        self.startup_recent_list.bind("<Double-Button-1>", lambda _event: self.open_startup_recent())
        self.startup_recent_list.bind("<Return>", lambda _event: self.open_startup_recent())
        ttk.Button(recent_area, text="Открыть выбранный", command=self.open_startup_recent).grid(row=2, column=0, sticky="e", pady=(10, 0))
        self.refresh_startup_recent()

    def restore_centered_window(self, preferred_width: int, preferred_height: int) -> None:
        try:
            self.state("normal")
        except tk.TclError:
            try:
                self.attributes("-zoomed", False)
            except tk.TclError:
                pass
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        width = min(preferred_width, max(820, screen_width - 120))
        height = min(preferred_height, max(560, screen_height - 120))
        x = max(0, (screen_width - width) // 2)
        y = max(0, (screen_height - height) // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    @staticmethod
    def center_toplevel(window: tk.Toplevel, preferred_width: int, preferred_height: int) -> None:
        window.update_idletasks()
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        minimum_width, minimum_height = window.minsize()
        width = max(minimum_width, min(max(preferred_width, window.winfo_reqwidth()), max(640, screen_width - 80)))
        height = max(minimum_height, min(max(preferred_height, window.winfo_reqheight()), max(520, screen_height - 80)))
        x = max(0, (screen_width - width) // 2)
        y = max(0, (screen_height - height) // 2)
        window.geometry(f"{width}x{height}+{x}+{y}")
        window.update_idletasks()

    def refresh_startup_recent(self) -> None:
        if not hasattr(self, "startup_recent_list") or not self.startup_recent_list.winfo_exists():
            return
        recent_list = self.startup_recent_list
        recent_list.delete(0, tk.END)
        self._startup_recent_paths: list[str] = []
        for path in self.recent_files:
            item = Path(path)
            if not item.exists():
                continue
            kind = "Проект" if item.suffix.lower() == ".prdx" else "Изображение"
            modified = time.strftime("%d.%m.%Y %H:%M", time.localtime(item.stat().st_mtime))
            recent_list.insert(tk.END, f"  {item.name}  |  {kind}  |  {modified}")
            self._startup_recent_paths.append(path)
        if self._startup_recent_paths:
            recent_list.selection_set(0)
        else:
            recent_list.insert(tk.END, "  Недавних файлов пока нет")

    def open_startup_recent(self) -> None:
        selection = self.startup_recent_list.curselection() if hasattr(self, "startup_recent_list") else ()
        if not selection:
            return
        try:
            path = self._startup_recent_paths[int(selection[0])]
        except IndexError:
            return
        self.open_path(path)

    def show_editor(self) -> None:
        if self.startup_frame is not None and self.startup_frame.winfo_exists():
            self.startup_frame.destroy()
        self.startup_frame = None
        self.config(menu=self.editor_menu)
        self.geometry("1440x920")
        self.minsize(1000, 640)
        document_name = Path(self.doc.path).name if self.doc.path else "Новый документ"
        self.title(f"{document_name} - UZYRO")
        self.editor_root.pack(fill=tk.BOTH, expand=True)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self._editor_active = True
        self.refresh_recent_menu()
        self.refresh()
        self.maximize_window()
        if self._initial_fit_after_id is not None:
            self.after_cancel(self._initial_fit_after_id)
        self._initial_fit_after_id = self.after(80, self.finish_initial_fit)

    def finish_initial_fit(self) -> None:
        self._initial_fit_after_id = None
        if self._editor_active and self.winfo_exists():
            self.update_idletasks()
            self._performing_initial_fit = True
            try:
                self.fit_to_screen()
            finally:
                self._performing_initial_fit = False
            self._initial_fit_after_id = self.after(160, self.finish_initial_center)

    def finish_initial_center(self) -> None:
        self._initial_fit_after_id = None
        if self._editor_active and self.winfo_exists():
            self.update_idletasks()
            self.center_canvas_on_doc(self.doc.width / 2, self.doc.height / 2)

    def maximize_window(self) -> None:
        try:
            self.state("zoomed")
        except tk.TclError:
            try:
                self.attributes("-zoomed", True)
            except tk.TclError:
                pass
