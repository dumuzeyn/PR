from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import shutil
import tkinter as tk


APP_NAME = "UZYRO"


def branding_asset(name: str) -> Path:
    return Path(__file__).resolve().parent / "assets" / "branding" / name


def apply_window_branding(window: tk.Tk) -> None:
    icon_path = branding_asset("uzyro.ico")
    try:
        icon = tk.PhotoImage(file=str(branding_asset("uzyro-icon.png")))
        window.iconphoto(True, icon)
        window._brand_icon = icon
    except tk.TclError:
        pass
    if os.name == "nt":
        try:
            window.iconbitmap(str(icon_path))
        except tk.TclError:
            pass
        _apply_windows_titlebar_icon(window, icon_path)
        _apply_windows_dark_titlebar(window)
        if not getattr(window, "_brand_map_binding", False):
            window.bind(
                "<Map>",
                lambda _event: (
                    _apply_windows_titlebar_icon(window, icon_path),
                    _apply_windows_dark_titlebar(window),
                ),
                add="+",
            )
            window._brand_map_binding = True
        if not getattr(window, "_brand_destroy_binding", False):
            window.bind(
                "<Destroy>",
                lambda event: _release_windows_icons(window) if event.widget is window else None,
                add="+",
            )
            window._brand_destroy_binding = True
        window.after(
            200,
            lambda: (
                _apply_windows_titlebar_icon(window, icon_path),
                _apply_windows_dark_titlebar(window),
            ),
        )


def _apply_windows_dark_titlebar(window: tk.Misc) -> None:
    """Ask DWM for native dark chrome while keeping standard window behavior."""
    if os.name != "nt":
        return
    try:
        user32 = ctypes.windll.user32
        client_handle = int(window.winfo_id())
        window_handle = int(user32.GetParent(client_handle)) or client_handle
        enabled = ctypes.c_int(1)
        for attribute in (20, 19):
            result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                window_handle,
                attribute,
                ctypes.byref(enabled),
                ctypes.sizeof(enabled),
            )
            if result == 0:
                break
    except (AttributeError, OSError, tk.TclError):
        return


def _apply_windows_titlebar_icon(window: tk.Misc, icon_path: Path) -> None:
    if os.name != "nt" or not icon_path.is_file():
        return
    try:
        user32 = ctypes.windll.user32
        user32.GetParent.argtypes = [wintypes.HWND]
        user32.GetParent.restype = wintypes.HWND
        user32.LoadImageW.argtypes = [
            wintypes.HINSTANCE,
            wintypes.LPCWSTR,
            wintypes.UINT,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        ]
        user32.LoadImageW.restype = wintypes.HANDLE
        user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        user32.SendMessageW.restype = wintypes.LPARAM
        user32.DestroyIcon.argtypes = [wintypes.HICON]
        user32.DestroyIcon.restype = wintypes.BOOL
        client_handle = int(window.winfo_id())
        window_handle = int(user32.GetParent(client_handle)) or client_handle
        handles = tuple(
            int(user32.LoadImageW(None, str(icon_path), 1, size, size, 0x0010) or 0)
            for size in (16, 32)
        )
        if not all(handles):
            return
        user32.SendMessageW(window_handle, 0x0080, 0, handles[0])
        user32.SendMessageW(window_handle, 0x0080, 1, handles[1])
        for previous in getattr(window, "_brand_native_icons", ()):
            if previous and previous not in handles:
                user32.DestroyIcon(previous)
        window._brand_native_icons = handles
    except (AttributeError, OSError, tk.TclError):
        return


def _release_windows_icons(window: tk.Misc) -> None:
    if os.name != "nt":
        return
    try:
        for handle in getattr(window, "_brand_native_icons", ()):
            if handle:
                ctypes.windll.user32.DestroyIcon(handle)
        window._brand_native_icons = ()
    except (AttributeError, OSError):
        return


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
