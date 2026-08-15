from __future__ import annotations

import os
from pathlib import Path
import shutil
import tkinter as tk


APP_NAME = "UZYRO"


def branding_asset(name: str) -> Path:
    return Path(__file__).resolve().parent / "assets" / "branding" / name


def apply_window_branding(window: tk.Tk) -> None:
    try:
        icon = tk.PhotoImage(file=str(branding_asset("uzyro-icon.png")))
        window.iconphoto(True, icon)
        window._brand_icon = icon
    except tk.TclError:
        pass
    if os.name == "nt":
        try:
            window.iconbitmap(default=str(branding_asset("uzyro.ico")))
        except tk.TclError:
            pass


def user_data_directory(base: Path) -> Path:
    target = base / APP_NAME
    legacy = base / ("Photo" + "Redactor")
    if not target.exists() and legacy.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        for name in ("settings.json", "recovery.prdx"):
            source = legacy / name
            if source.is_file():
                shutil.copy2(source, target / name)
    return target
