from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk


@dataclass(frozen=True)
class DesignTokens:
    BACKGROUND: str = "#e8e9eb"
    WORKSPACE: str = "#34363a"
    SURFACE: str = "#f4f4f5"
    SURFACE_HOVER: str = "#e4e6e8"
    SURFACE_ACTIVE: str = "#d7dade"
    SURFACE_SELECTED: str = "#d6e7f4"
    BORDER: str = "#b8bdc3"
    BORDER_ACTIVE: str = "#2f78b7"
    TEXT_PRIMARY: str = "#1d2329"
    TEXT_SECONDARY: str = "#5f6872"
    TEXT_DISABLED: str = "#989fa7"
    ACCENT: str = "#2f78b7"
    ACCENT_HOVER: str = "#3f87c2"
    DANGER: str = "#b23b45"
    SPACING_XS: int = 4
    SPACING_SM: int = 8
    SPACING_MD: int = 12
    CONTROL_HEIGHT: int = 28
    ICON_SIZE: int = 18
    CORNER_RADIUS: int = 5


TOKENS = DesignTokens()


def configure_theme(root: tk.Misc, tokens: DesignTokens = TOKENS) -> ttk.Style:
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    root.option_add("*Font", "{Segoe UI} 9")
    root.option_add("*tearOff", False)
    root.option_add("*Menu.background", tokens.SURFACE)
    root.option_add("*Menu.foreground", tokens.TEXT_PRIMARY)
    root.option_add("*Menu.activeBackground", tokens.SURFACE_SELECTED)
    root.option_add("*Menu.activeForeground", tokens.TEXT_PRIMARY)
    root.option_add("*Menu.selectColor", tokens.ACCENT)

    style.configure(".", background=tokens.SURFACE, foreground=tokens.TEXT_PRIMARY, bordercolor=tokens.BORDER)
    style.configure("App.TFrame", background=tokens.BACKGROUND)
    style.configure("Panel.TFrame", background=tokens.SURFACE)
    style.configure("Workspace.TFrame", background=tokens.WORKSPACE)
    style.configure("Topbar.TFrame", background=tokens.BACKGROUND)
    style.configure("Status.TFrame", background=tokens.BACKGROUND)
    style.configure("TLabel", background=tokens.SURFACE, foreground=tokens.TEXT_PRIMARY)
    style.configure("Topbar.TLabel", background=tokens.BACKGROUND, foreground=tokens.TEXT_PRIMARY)
    style.configure("Status.TLabel", background=tokens.BACKGROUND, foreground=tokens.TEXT_SECONDARY, padding=(8, 3))
    style.configure("Secondary.TLabel", foreground=tokens.TEXT_SECONDARY)
    style.configure("PanelTitle.TLabel", font=("Segoe UI Semibold", 9), foreground=tokens.TEXT_PRIMARY)
    style.configure(
        "TButton",
        background=tokens.SURFACE_HOVER,
        foreground=tokens.TEXT_PRIMARY,
        bordercolor=tokens.BORDER,
        lightcolor=tokens.SURFACE_HOVER,
        darkcolor=tokens.SURFACE_HOVER,
        padding=(8, 4),
        relief="flat",
    )
    style.map(
        "TButton",
        background=[("pressed", tokens.SURFACE_ACTIVE), ("active", tokens.SURFACE_HOVER), ("disabled", tokens.SURFACE)],
        foreground=[("disabled", tokens.TEXT_DISABLED)],
        bordercolor=[("focus", tokens.BORDER_ACTIVE)],
    )
    style.configure("Primary.TButton", background=tokens.ACCENT, foreground="#ffffff", bordercolor=tokens.ACCENT, padding=(10, 5))
    style.map("Primary.TButton", background=[("active", tokens.ACCENT_HOVER), ("pressed", tokens.ACCENT)])
    style.configure("Quiet.TButton", background=tokens.SURFACE, bordercolor=tokens.SURFACE, padding=(5, 3))
    style.map("Quiet.TButton", background=[("active", tokens.SURFACE_HOVER), ("pressed", tokens.SURFACE_ACTIVE)])
    style.configure("Danger.TButton", foreground=tokens.DANGER)
    style.configure(
        "Tool.TRadiobutton",
        background=tokens.SURFACE,
        foreground=tokens.TEXT_PRIMARY,
        indicatorcolor=tokens.SURFACE,
        borderwidth=0,
        padding=(8, 6),
        relief="flat",
    )
    style.layout(
        "Tool.TRadiobutton",
        [("Radiobutton.padding", {"sticky": "nswe", "children": [("Radiobutton.label", {"sticky": "nswe"})]})],
    )
    style.map(
        "Tool.TRadiobutton",
        background=[("selected", tokens.SURFACE_SELECTED), ("active", tokens.SURFACE_HOVER)],
        foreground=[("selected", tokens.TEXT_PRIMARY)],
        indicatorcolor=[("selected", tokens.ACCENT)],
    )
    style.configure(
        "ToolOptionsTitle.TLabel",
        background=tokens.BACKGROUND,
        foreground=tokens.TEXT_PRIMARY,
        font=("Segoe UI Semibold", 9),
        padding=(10, 5),
    )
    for widget in ("TEntry", "TSpinbox", "TCombobox"):
        style.configure(widget, fieldbackground=tokens.SURFACE_HOVER, background=tokens.SURFACE_HOVER, foreground=tokens.TEXT_PRIMARY, bordercolor=tokens.BORDER, arrowcolor=tokens.TEXT_SECONDARY, padding=4)
        style.map(widget, bordercolor=[("focus", tokens.BORDER_ACTIVE)], fieldbackground=[("readonly", tokens.SURFACE_HOVER)], foreground=[("disabled", tokens.TEXT_DISABLED)])
    style.configure("TCheckbutton", background=tokens.SURFACE, foreground=tokens.TEXT_PRIMARY, indicatorcolor=tokens.SURFACE_HOVER, padding=3)
    style.map("TCheckbutton", indicatorcolor=[("selected", tokens.ACCENT), ("active", tokens.SURFACE_HOVER)])
    style.configure("TScale", background=tokens.SURFACE, troughcolor=tokens.SURFACE_HOVER, bordercolor=tokens.SURFACE, lightcolor=tokens.ACCENT, darkcolor=tokens.ACCENT)
    style.configure("TScrollbar", background=tokens.SURFACE_HOVER, troughcolor=tokens.SURFACE, bordercolor=tokens.SURFACE, arrowcolor=tokens.TEXT_SECONDARY, width=9)
    style.map("TScrollbar", background=[("active", tokens.TEXT_DISABLED)])
    style.configure("TNotebook", background=tokens.SURFACE, borderwidth=0, tabmargins=0)
    style.configure("TNotebook.Tab", background=tokens.SURFACE_HOVER, foreground=tokens.TEXT_SECONDARY, padding=(10, 7), borderwidth=0)
    style.map("TNotebook.Tab", background=[("selected", tokens.SURFACE_SELECTED), ("active", tokens.SURFACE_HOVER)], foreground=[("selected", tokens.TEXT_PRIMARY)])
    style.configure("TPanedwindow", background=tokens.BORDER, sashwidth=1)
    style.configure("TSeparator", background=tokens.BORDER)
    return style
