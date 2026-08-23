from __future__ import annotations

from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from .desktop_controls import SlimScrollbar
from .theme import TOKENS


class ProjectSaveDialog(tk.Toplevel):
    def __init__(self, parent: tk.Misc, initial_path: str | Path | None = None) -> None:
        super().__init__(parent)
        self.parent = parent
        self.result: str | None = None
        self._entries: dict[str, Path] = {}
        self._overwrite_target: Path | None = None
        initial = Path(initial_path).expanduser() if initial_path else Path.home() / "Новый проект.prdx"
        self.current_directory = initial.parent if initial.parent.is_dir() else Path.home()
        self.filename = tk.StringVar(self, value=initial.name or "Новый проект.prdx")
        self.location = tk.StringVar(self, value=str(self.current_directory))
        self.status = tk.StringVar(self, value="")
        self.title("Сохранить проект - UZYRO")
        self.configure(background=TOKENS.APP_BG)
        self.transient(parent)
        self.minsize(700, 460)
        self.geometry("820x560")
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self._build()
        self._center()
        self.grab_set()
        self.filename_entry.focus_set()
        self.bind("<Escape>", lambda _event: self.cancel())
        self.bind("<Return>", lambda _event: self.accept())

    def _build(self) -> None:
        header = tk.Frame(self, height=64, background=TOKENS.APP_BG)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(
            header,
            text="Сохранить проект",
            font=("Segoe UI Semibold", 15),
            foreground=TOKENS.TEXT_PRIMARY,
            background=TOKENS.APP_BG,
        ).pack(anchor=tk.W, padx=18, pady=(11, 0))
        tk.Label(
            header,
            text="Формат UZYRO сохраняет слои, маски, текст и редактируемые объекты",
            foreground=TOKENS.TEXT_SECONDARY,
            background=TOKENS.APP_BG,
        ).pack(anchor=tk.W, padx=18)

        body = ttk.Frame(self, style="Panel.TFrame")
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(1, weight=1)

        navigation = ttk.Frame(body, style="Elevated.TFrame", width=166)
        navigation.grid(row=0, column=0, rowspan=3, sticky="nsew")
        navigation.grid_propagate(False)
        ttk.Label(navigation, text="Расположение", style="Elevated.TLabel", font=("Segoe UI Semibold", 9)).pack(anchor=tk.W, padx=12, pady=(12, 7))
        for label, path in self._shortcuts():
            button = tk.Button(
                navigation,
                text=label,
                anchor=tk.W,
                command=lambda target=path: self.navigate(target),
                background=TOKENS.PANEL_RAISED,
                foreground=TOKENS.TEXT_SECONDARY,
                activebackground=TOKENS.CONTROL_HOVER,
                activeforeground=TOKENS.TEXT_PRIMARY,
                relief=tk.FLAT,
                borderwidth=0,
                padx=12,
                pady=7,
                cursor="hand2",
            )
            button.pack(fill=tk.X)

        toolbar = ttk.Frame(body, style="Panel.TFrame")
        toolbar.grid(row=0, column=1, sticky="ew", padx=12, pady=10)
        toolbar.columnconfigure(1, weight=1)
        ttk.Button(toolbar, text="Вверх", width=8, command=self.go_up).grid(row=0, column=0, padx=(0, 6))
        location_entry = ttk.Entry(toolbar, textvariable=self.location)
        location_entry.grid(row=0, column=1, sticky="ew")
        location_entry.bind("<Return>", lambda _event: self.navigate_from_entry())
        ttk.Button(toolbar, text="Перейти", width=9, command=self.navigate_from_entry).grid(row=0, column=2, padx=(6, 0))
        ttk.Button(toolbar, text="Новая папка", command=self.create_folder).grid(row=0, column=3, padx=(6, 0))

        list_area = ttk.Frame(body, style="Panel.TFrame")
        list_area.grid(row=1, column=1, sticky="nsew", padx=12)
        list_area.columnconfigure(0, weight=1)
        list_area.rowconfigure(0, weight=1)
        self.file_list = ttk.Treeview(list_area, columns=("name", "type", "modified"), show="headings", selectmode="browse")
        self.file_list.heading("name", text="Имя")
        self.file_list.heading("type", text="Тип")
        self.file_list.heading("modified", text="Изменён")
        self.file_list.column("name", width=330, minwidth=180, anchor=tk.W)
        self.file_list.column("type", width=110, minwidth=90, anchor=tk.W, stretch=False)
        self.file_list.column("modified", width=140, minwidth=120, anchor=tk.W, stretch=False)
        scrollbar = SlimScrollbar(list_area, orient=tk.VERTICAL, command=self.file_list.yview)
        self.file_list.configure(yscrollcommand=scrollbar.set)
        self.file_list.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.file_list.bind("<<TreeviewSelect>>", self.select_entry)
        self.file_list.bind("<Double-Button-1>", self.open_entry)

        footer = ttk.Frame(body, style="Panel.TFrame")
        footer.grid(row=2, column=1, sticky="ew", padx=12, pady=12)
        footer.columnconfigure(1, weight=1)
        ttk.Label(footer, text="Имя файла", style="Secondary.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.filename_entry = ttk.Entry(footer, textvariable=self.filename)
        self.filename_entry.grid(row=0, column=1, sticky="ew")
        ttk.Label(footer, text="Проект UZYRO (*.prdx)", style="Secondary.TLabel").grid(row=1, column=1, sticky="w", pady=(5, 0))
        ttk.Label(footer, textvariable=self.status, style="Danger.TLabel").grid(row=2, column=0, columnspan=2, sticky="w", pady=(7, 0))
        actions = ttk.Frame(footer, style="Panel.TFrame")
        actions.grid(row=0, column=2, rowspan=3, sticky="se", padx=(12, 0))
        ttk.Button(actions, text="Отмена", width=10, command=self.cancel).pack(side=tk.LEFT)
        self.save_button = ttk.Button(actions, text="Сохранить", width=12, style="Primary.TButton", command=self.accept)
        self.save_button.pack(side=tk.LEFT, padx=(6, 0))
        self.refresh_files()

    def _shortcuts(self) -> list[tuple[str, Path]]:
        home = Path.home()
        candidates = [
            ("Домашняя папка", home),
            ("Рабочий стол", home / "Desktop"),
            ("Документы", home / "Documents"),
            ("Загрузки", home / "Downloads"),
            ("Диск " + self.current_directory.anchor.rstrip("\\"), Path(self.current_directory.anchor)),
        ]
        result: list[tuple[str, Path]] = []
        seen: set[str] = set()
        for label, path in candidates:
            key = str(path).casefold()
            if path.is_dir() and key not in seen:
                result.append((label, path))
                seen.add(key)
        return result

    def _center(self) -> None:
        self.update_idletasks()
        width, height = self.winfo_width(), self.winfo_height()
        x = self.parent.winfo_rootx() + max(0, (self.parent.winfo_width() - width) // 2)
        y = self.parent.winfo_rooty() + max(0, (self.parent.winfo_height() - height) // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def refresh_files(self) -> None:
        self.file_list.delete(*self.file_list.get_children())
        self._entries.clear()
        self.location.set(str(self.current_directory))
        try:
            entries = sorted(self.current_directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.casefold()))
        except (OSError, PermissionError) as exc:
            self.status.set(f"Не удалось открыть папку: {exc}")
            return
        for index, path in enumerate(entries):
            if path.name.startswith("."):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            item_id = f"item-{index}"
            kind = "Папка" if path.is_dir() else "Проект UZYRO" if path.suffix.lower() == ".prdx" else "Файл"
            modified = datetime.fromtimestamp(stat.st_mtime).strftime("%d.%m.%Y %H:%M")
            self.file_list.insert("", tk.END, iid=item_id, values=(path.name, kind, modified))
            self._entries[item_id] = path
        self.status.set("")
        self._overwrite_target = None
        self.save_button.configure(text="Сохранить")

    def selected_path(self) -> Path | None:
        selected = self.file_list.selection()
        return self._entries.get(selected[0]) if selected else None

    def select_entry(self, _event=None) -> None:
        path = self.selected_path()
        if path is not None and path.is_file() and path.suffix.lower() == ".prdx":
            self.filename.set(path.name)
            self._reset_overwrite()

    def open_entry(self, _event=None) -> None:
        path = self.selected_path()
        if path is None:
            return
        if path.is_dir():
            self.navigate(path)
        elif path.suffix.lower() == ".prdx":
            self.filename.set(path.name)
            self.accept()

    def navigate(self, path: Path) -> None:
        if path.is_dir():
            self.current_directory = path.resolve()
            self.refresh_files()
        else:
            self.status.set("Указанная папка не существует")

    def navigate_from_entry(self) -> None:
        self.navigate(Path(self.location.get()).expanduser())

    def go_up(self) -> None:
        parent = self.current_directory.parent
        if parent != self.current_directory:
            self.navigate(parent)

    def create_folder(self) -> None:
        name = FolderNameDialog.ask(self)
        if self.winfo_exists():
            self.grab_set()
        if not name:
            return
        target = self.current_directory / name
        try:
            target.mkdir()
            self.navigate(target)
        except OSError as exc:
            self.status.set(f"Не удалось создать папку: {exc}")

    def _reset_overwrite(self, _event=None) -> None:
        self._overwrite_target = None
        self.save_button.configure(text="Сохранить")
        self.status.set("")

    def accept(self) -> None:
        name = self.filename.get().strip()
        if not name or name in {".", ".."} or any(character in name for character in '<>:"/\\|?*'):
            self.status.set("Введите допустимое имя проекта")
            return
        if not name.lower().endswith(".prdx"):
            name += ".prdx"
        target = self.current_directory / name
        if target.exists() and self._overwrite_target != target:
            self._overwrite_target = target
            self.status.set("Файл уже существует. Нажмите «Перезаписать», чтобы заменить его.")
            self.save_button.configure(text="Перезаписать")
            return
        self.result = str(target)
        self.destroy()

    def cancel(self) -> None:
        self.result = None
        self.destroy()


class FolderNameDialog(tk.Toplevel):
    @classmethod
    def ask(cls, parent: tk.Misc) -> str | None:
        dialog = cls(parent)
        dialog.wait_window()
        return dialog.result

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.result: str | None = None
        self.value = tk.StringVar(self, value="Новая папка")
        self.title("Новая папка - UZYRO")
        self.configure(background=TOKENS.PANEL_BG)
        self.transient(parent)
        self.resizable(False, False)
        self.grab_set()
        content = ttk.Frame(self, style="Panel.TFrame", padding=16)
        content.pack(fill=tk.BOTH, expand=True)
        ttk.Label(content, text="Название папки", style="PanelTitle.TLabel").pack(anchor=tk.W, pady=(0, 7))
        entry = ttk.Entry(content, textvariable=self.value, width=38)
        entry.pack(fill=tk.X)
        actions = ttk.Frame(content, style="Panel.TFrame")
        actions.pack(fill=tk.X, pady=(14, 0))
        ttk.Button(actions, text="Создать", style="Primary.TButton", command=self.accept).pack(side=tk.RIGHT)
        ttk.Button(actions, text="Отмена", command=self.cancel).pack(side=tk.RIGHT, padx=(0, 6))
        self.bind("<Return>", lambda _event: self.accept())
        self.bind("<Escape>", lambda _event: self.cancel())
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.update_idletasks()
        x = parent.winfo_rootx() + max(0, (parent.winfo_width() - self.winfo_width()) // 2)
        y = parent.winfo_rooty() + max(0, (parent.winfo_height() - self.winfo_height()) // 2)
        self.geometry(f"+{x}+{y}")
        entry.selection_range(0, tk.END)
        entry.focus_set()

    def accept(self) -> None:
        value = self.value.get().strip()
        if value and value not in {".", ".."} and not any(character in value for character in '<>:"/\\|?*'):
            self.result = value
            self.destroy()

    def cancel(self) -> None:
        self.destroy()


def ask_project_save_path(parent: tk.Misc, initial_path: str | Path | None = None) -> str | None:
    dialog = ProjectSaveDialog(parent, initial_path)
    dialog.wait_window()
    return dialog.result


__all__ = ["ProjectSaveDialog", "ask_project_save_path"]
