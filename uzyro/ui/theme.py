from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk


@dataclass(frozen=True)
class DesignTokens:
    BACKGROUND: str = "#111216"
    WORKSPACE: str = "#0c0d10"
    SURFACE: str = "#191a1f"
    SURFACE_ELEVATED: str = "#222329"
    SURFACE_HOVER: str = "#282a31"
    SURFACE_ACTIVE: str = "#31333c"
    SURFACE_SELECTED: str = "#312944"
    BORDER: str = "#292b32"
    SEPARATOR: str = "#2b2d34"
    BORDER_ACTIVE: str = "#9b73ff"
    TEXT_PRIMARY: str = "#f5f5f7"
    TEXT_SECONDARY: str = "#b2b4bd"
    TEXT_DISABLED: str = "#727680"
    ACCENT: str = "#9568fb"
    ACCENT_HOVER: str = "#a77cff"
    ACCENT_PRESSED: str = "#8054e8"
    SELECTION: str = "#3b3151"
    FOCUS: str = "#b190ff"
    SUCCESS: str = "#56b889"
    WARNING: str = "#e1b65f"
    DANGER: str = "#e06c78"
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
    root.option_add("*Listbox.background", tokens.SURFACE)
    root.option_add("*Listbox.foreground", tokens.TEXT_PRIMARY)
    root.option_add("*Listbox.selectBackground", tokens.SELECTION)
    root.option_add("*Listbox.selectForeground", tokens.TEXT_PRIMARY)
    root.option_add("*Listbox.highlightColor", tokens.FOCUS)
    root.option_add("*Listbox.highlightBackground", tokens.BORDER)
    root.option_add("*Text.background", tokens.SURFACE_ELEVATED)
    root.option_add("*Text.foreground", tokens.TEXT_PRIMARY)
    root.option_add("*Text.insertBackground", tokens.TEXT_PRIMARY)

    style.configure(".", background=tokens.SURFACE, foreground=tokens.TEXT_PRIMARY, bordercolor=tokens.BORDER)
    style.configure("App.TFrame", background=tokens.BACKGROUND)
    style.configure("Panel.TFrame", background=tokens.SURFACE)
    style.configure("Elevated.TFrame", background=tokens.SURFACE_ELEVATED)
    style.configure("Workspace.TFrame", background=tokens.WORKSPACE)
    style.configure("Topbar.TFrame", background=tokens.BACKGROUND)
    style.configure("Status.TFrame", background=tokens.BACKGROUND)
    style.configure("TLabel", background=tokens.SURFACE, foreground=tokens.TEXT_PRIMARY)
    style.configure("Topbar.TLabel", background=tokens.BACKGROUND, foreground=tokens.TEXT_PRIMARY)
    style.configure("Status.TLabel", background=tokens.BACKGROUND, foreground=tokens.TEXT_SECONDARY, padding=(8, 3))
    style.configure("Secondary.TLabel", foreground=tokens.TEXT_SECONDARY)
    style.configure("Accent.TLabel", foreground=tokens.ACCENT_HOVER, font=("Segoe UI Semibold", 9))
    style.configure("Elevated.TLabel", background=tokens.SURFACE_ELEVATED, foreground=tokens.TEXT_PRIMARY)
    style.configure("Success.TLabel", foreground=tokens.SUCCESS)
    style.configure("Warning.TLabel", foreground=tokens.WARNING)
    style.configure("Danger.TLabel", foreground=tokens.DANGER)
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
        borderwidth=0,
    )
    style.map(
        "TButton",
        background=[("pressed", tokens.SURFACE_ACTIVE), ("active", tokens.SURFACE_HOVER), ("disabled", tokens.SURFACE)],
        foreground=[("disabled", tokens.TEXT_DISABLED)],
        bordercolor=[("focus", tokens.BORDER_ACTIVE)],
    )
    style.configure("Primary.TButton", background=tokens.ACCENT, foreground="#ffffff", bordercolor=tokens.ACCENT, padding=(11, 5))
    style.map("Primary.TButton", background=[("active", tokens.ACCENT_HOVER), ("pressed", tokens.ACCENT_PRESSED), ("disabled", tokens.SURFACE_ACTIVE)], foreground=[("disabled", tokens.TEXT_DISABLED)])
    style.configure("Quiet.TButton", background=tokens.SURFACE, bordercolor=tokens.SURFACE, padding=(5, 3))
    style.map("Quiet.TButton", background=[("active", tokens.SURFACE_HOVER), ("pressed", tokens.SURFACE_ACTIVE)])
    style.configure("PanelIcon.TButton", padding=(4, 4))
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
        foreground=[("selected", "#ffffff")],
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
        style.configure(widget, fieldbackground=tokens.SURFACE_HOVER, background=tokens.SURFACE_HOVER, foreground=tokens.TEXT_PRIMARY, bordercolor=tokens.SURFACE_HOVER, lightcolor=tokens.SURFACE_HOVER, darkcolor=tokens.SURFACE_HOVER, arrowcolor=tokens.TEXT_SECONDARY, padding=4, borderwidth=0, relief="flat")
        style.map(widget, bordercolor=[("focus", tokens.BORDER_ACTIVE)], fieldbackground=[("readonly", tokens.SURFACE_HOVER)], foreground=[("disabled", tokens.TEXT_DISABLED)])
    style.configure("TCheckbutton", background=tokens.SURFACE, foreground=tokens.TEXT_PRIMARY, indicatorcolor=tokens.SURFACE_HOVER, padding=3)
    style.map("TCheckbutton", indicatorcolor=[("selected", tokens.ACCENT), ("active", tokens.SURFACE_HOVER)])
    style.configure("TScale", background=tokens.SURFACE, troughcolor=tokens.SURFACE_HOVER, bordercolor=tokens.SURFACE, lightcolor=tokens.ACCENT, darkcolor=tokens.ACCENT, borderwidth=0, relief="flat", sliderlength=14, sliderthickness=12)
    style.configure("TScrollbar", background=tokens.SURFACE_HOVER, troughcolor=tokens.SURFACE, bordercolor=tokens.SURFACE, arrowcolor=tokens.TEXT_SECONDARY, width=9)
    style.map("TScrollbar", background=[("active", tokens.TEXT_DISABLED)])
    style.configure("TNotebook", background=tokens.SURFACE, borderwidth=0, tabmargins=0)
    style.configure("TNotebook.Tab", background=tokens.SURFACE_HOVER, foreground=tokens.TEXT_SECONDARY, padding=(10, 7), borderwidth=0)
    style.map("TNotebook.Tab", background=[("selected", tokens.SURFACE_SELECTED), ("active", tokens.SURFACE_HOVER)], foreground=[("selected", tokens.TEXT_PRIMARY)])
    style.configure("Sidebar.TNotebook", background=tokens.SURFACE, borderwidth=0, tabmargins=0)
    style.configure("Sidebar.TNotebook.Tab", background=tokens.SURFACE_HOVER, foreground=tokens.TEXT_SECONDARY, padding=(6, 7), borderwidth=0)
    style.map("Sidebar.TNotebook.Tab", background=[("selected", tokens.SURFACE_SELECTED), ("active", tokens.SURFACE_HOVER)], foreground=[("selected", tokens.TEXT_PRIMARY)])
    style.configure("TPanedwindow", background=tokens.BORDER, sashwidth=1)
    style.configure("TSeparator", background=tokens.SEPARATOR)
    style.configure("DocumentTabs.TFrame", background=tokens.BACKGROUND)
    style.configure("DocumentTab.TButton", background=tokens.BACKGROUND, foreground=tokens.TEXT_SECONDARY, padding=(12, 6), borderwidth=0)
    style.map("DocumentTab.TButton", background=[("active", tokens.SURFACE_HOVER)], foreground=[("active", tokens.TEXT_PRIMARY)])
    style.configure("ActiveDocumentTab.TButton", background=tokens.SURFACE_SELECTED, foreground=tokens.TEXT_PRIMARY, padding=(12, 6), borderwidth=0)
    style.map("ActiveDocumentTab.TButton", background=[("active", tokens.SURFACE_SELECTED), ("pressed", tokens.SURFACE_ACTIVE)])
    style.configure("DocumentTabClose.TButton", background=tokens.BACKGROUND, foreground=tokens.TEXT_SECONDARY, padding=(5, 6), borderwidth=0)
    style.map("DocumentTabClose.TButton", background=[("active", tokens.SURFACE_HOVER)], foreground=[("active", tokens.TEXT_PRIMARY)])
    style.configure("TLabelframe", background=tokens.SURFACE, bordercolor=tokens.SURFACE, relief="flat", borderwidth=0)
    style.configure("TLabelframe.Label", background=tokens.SURFACE, foreground=tokens.TEXT_SECONDARY, font=("Segoe UI Semibold", 9))
    style.configure("Treeview", background=tokens.SURFACE, fieldbackground=tokens.SURFACE, foreground=tokens.TEXT_PRIMARY, bordercolor=tokens.SURFACE, rowheight=28, borderwidth=0, relief="flat")
    style.map("Treeview", background=[("selected", tokens.SELECTION)], foreground=[("selected", tokens.TEXT_PRIMARY)])
    style.configure("Treeview.Heading", background=tokens.SURFACE_ELEVATED, foreground=tokens.TEXT_SECONDARY, bordercolor=tokens.BORDER, padding=(8, 6), relief="flat", font=("Segoe UI Semibold", 9))
    style.map("Treeview.Heading", background=[("active", tokens.SURFACE_HOVER)])
    style.configure("TProgressbar", background=tokens.ACCENT, troughcolor=tokens.SURFACE_ELEVATED, bordercolor=tokens.BORDER, lightcolor=tokens.ACCENT, darkcolor=tokens.ACCENT)
    style.configure("Vertical.TProgressbar", background=tokens.ACCENT, troughcolor=tokens.SURFACE_ELEVATED)
    style.configure("TMenubutton", background=tokens.SURFACE_HOVER, foreground=tokens.TEXT_PRIMARY, bordercolor=tokens.BORDER, arrowcolor=tokens.TEXT_SECONDARY, padding=(8, 4))
    return style
