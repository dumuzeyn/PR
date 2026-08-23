from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk


@dataclass(frozen=True)
class DesignTokens:
    APP_BG: str = "#15161a"
    WORKSPACE_BG: str = "#0d0e11"
    PANEL_BG: str = "#1a1b20"
    PANEL_RAISED: str = "#212329"
    CONTROL_BG: str = "#24262c"
    CONTROL_HOVER: str = "#2a2c33"
    CONTROL_PRESSED: str = "#202126"
    CONTROL_SELECTED: str = "#292631"
    BORDER_SUBTLE: str = "#2d2f36"
    BORDER_STRONG: str = "#454852"
    ACTIVE_EDGE: str = "#755aa8"
    ACTIVE_SHADOW: str = "#101115"
    TEXT_PRIMARY: str = "#edeef2"
    TEXT_SECONDARY: str = "#b8bbc4"
    TEXT_MUTED: str = "#8a8e99"
    TEXT_DISABLED: str = "#626670"
    ACCENT: str = "#9568fb"
    ACCENT_HOVER: str = "#a77cff"
    ACCENT_PRESSED: str = "#8054e8"
    SELECTION: str = "#302a3d"
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

    # Compatibility names keep the rest of the UI on the same centralized palette.
    @property
    def BACKGROUND(self) -> str:
        return self.APP_BG

    @property
    def WORKSPACE(self) -> str:
        return self.WORKSPACE_BG

    @property
    def SURFACE(self) -> str:
        return self.PANEL_BG

    @property
    def SURFACE_ELEVATED(self) -> str:
        return self.PANEL_RAISED

    @property
    def SURFACE_HOVER(self) -> str:
        return self.CONTROL_HOVER

    @property
    def SURFACE_ACTIVE(self) -> str:
        return self.CONTROL_PRESSED

    @property
    def SURFACE_SELECTED(self) -> str:
        return self.CONTROL_SELECTED

    @property
    def BORDER(self) -> str:
        return self.BORDER_SUBTLE

    @property
    def SEPARATOR(self) -> str:
        return self.BORDER_SUBTLE

    @property
    def BORDER_ACTIVE(self) -> str:
        return self.ACTIVE_EDGE


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

    style.configure(".", background=tokens.PANEL_BG, foreground=tokens.TEXT_PRIMARY, bordercolor=tokens.BORDER_SUBTLE)
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
    style.configure("Muted.TLabel", foreground=tokens.TEXT_MUTED)
    style.configure("Accent.TLabel", foreground=tokens.ACCENT_HOVER, font=("Segoe UI Semibold", 9))
    style.configure("Elevated.TLabel", background=tokens.SURFACE_ELEVATED, foreground=tokens.TEXT_PRIMARY)
    style.configure("Success.TLabel", foreground=tokens.SUCCESS)
    style.configure("Warning.TLabel", foreground=tokens.WARNING)
    style.configure("Danger.TLabel", foreground=tokens.DANGER)
    style.configure("PanelTitle.TLabel", font=("Segoe UI Semibold", 9), foreground=tokens.TEXT_PRIMARY)
    style.configure(
        "TButton",
        background=tokens.CONTROL_BG,
        foreground=tokens.TEXT_PRIMARY,
        bordercolor=tokens.BORDER_SUBTLE,
        lightcolor=tokens.CONTROL_BG,
        darkcolor=tokens.ACTIVE_SHADOW,
        padding=(7, 3),
        relief="flat",
        borderwidth=1,
    )
    style.map(
        "TButton",
        background=[("pressed", tokens.CONTROL_PRESSED), ("active", tokens.CONTROL_HOVER), ("disabled", tokens.PANEL_BG)],
        foreground=[("disabled", tokens.TEXT_DISABLED)],
        bordercolor=[("focus", tokens.FOCUS), ("active", tokens.BORDER_STRONG)],
    )
    style.configure("Primary.TButton", background=tokens.ACCENT, foreground="#ffffff", bordercolor=tokens.ACCENT, padding=(11, 5))
    style.map("Primary.TButton", background=[("active", tokens.ACCENT_HOVER), ("pressed", tokens.ACCENT_PRESSED), ("disabled", tokens.SURFACE_ACTIVE)], foreground=[("disabled", tokens.TEXT_DISABLED)])
    style.configure("Quiet.TButton", background=tokens.SURFACE, bordercolor=tokens.SURFACE, padding=(5, 3))
    style.map("Quiet.TButton", background=[("active", tokens.SURFACE_HOVER), ("pressed", tokens.SURFACE_ACTIVE)])
    style.configure("PanelIcon.TButton", padding=(4, 4))
    style.configure("Danger.TButton", foreground=tokens.DANGER)
    style.configure(
        "Tool.TRadiobutton",
        background=tokens.PANEL_BG,
        foreground=tokens.TEXT_PRIMARY,
        indicatorcolor=tokens.PANEL_BG,
        bordercolor=tokens.PANEL_BG,
        lightcolor=tokens.PANEL_BG,
        darkcolor=tokens.PANEL_BG,
        borderwidth=1,
        padding=(7, 5),
        relief="flat",
    )
    style.layout(
        "Tool.TRadiobutton",
        [
            (
                "Radiobutton.border",
                {
                    "sticky": "nswe",
                    "children": [
                        ("Radiobutton.padding", {"sticky": "nswe", "children": [("Radiobutton.label", {"sticky": "nswe"})]})
                    ],
                },
            )
        ],
    )
    style.map(
        "Tool.TRadiobutton",
        background=[("selected", tokens.CONTROL_SELECTED), ("active", tokens.CONTROL_HOVER)],
        foreground=[("selected", tokens.TEXT_PRIMARY)],
        bordercolor=[("selected", tokens.ACTIVE_EDGE), ("focus", tokens.FOCUS)],
        lightcolor=[("selected", tokens.BORDER_STRONG)],
        darkcolor=[("selected", tokens.ACTIVE_SHADOW)],
        relief=[("selected", "raised")],
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
        style.configure(widget, fieldbackground=tokens.CONTROL_BG, background=tokens.CONTROL_BG, foreground=tokens.TEXT_PRIMARY, bordercolor=tokens.BORDER_SUBTLE, lightcolor=tokens.CONTROL_BG, darkcolor=tokens.ACTIVE_SHADOW, arrowcolor=tokens.TEXT_SECONDARY, padding=3, borderwidth=1, relief="flat")
        style.map(widget, bordercolor=[("focus", tokens.FOCUS), ("active", tokens.BORDER_STRONG)], fieldbackground=[("readonly", tokens.CONTROL_BG), ("disabled", tokens.PANEL_BG)], foreground=[("disabled", tokens.TEXT_DISABLED)])
    style.configure("TCheckbutton", background=tokens.SURFACE, foreground=tokens.TEXT_PRIMARY, indicatorcolor=tokens.SURFACE_HOVER, padding=3)
    style.map("TCheckbutton", indicatorcolor=[("selected", tokens.ACCENT), ("active", tokens.SURFACE_HOVER)])
    style.configure("TScale", background=tokens.SURFACE, troughcolor=tokens.SURFACE_HOVER, bordercolor=tokens.SURFACE, lightcolor=tokens.ACCENT, darkcolor=tokens.ACCENT, borderwidth=0, relief="flat", sliderlength=14, sliderthickness=12)
    style.configure("TScrollbar", background=tokens.SURFACE_HOVER, troughcolor=tokens.SURFACE, bordercolor=tokens.SURFACE, arrowcolor=tokens.TEXT_SECONDARY, width=9)
    style.map("TScrollbar", background=[("active", tokens.TEXT_DISABLED)])
    style.configure("TNotebook", background=tokens.SURFACE, borderwidth=0, tabmargins=0)
    style.configure("TNotebook.Tab", background=tokens.PANEL_BG, foreground=tokens.TEXT_SECONDARY, padding=(9, 6), borderwidth=1, relief="flat")
    style.map("TNotebook.Tab", background=[("selected", tokens.CONTROL_SELECTED), ("active", tokens.CONTROL_HOVER)], foreground=[("selected", tokens.TEXT_PRIMARY)], bordercolor=[("selected", tokens.ACTIVE_EDGE)], lightcolor=[("selected", tokens.BORDER_STRONG)], darkcolor=[("selected", tokens.ACTIVE_SHADOW)], relief=[("selected", "raised")])
    style.configure("Sidebar.TNotebook", background=tokens.SURFACE, borderwidth=0, tabmargins=0)
    style.configure("Sidebar.TNotebook.Tab", background=tokens.PANEL_BG, foreground=tokens.TEXT_SECONDARY, padding=(5, 6), borderwidth=1, relief="flat")
    style.map("Sidebar.TNotebook.Tab", background=[("selected", tokens.CONTROL_SELECTED), ("active", tokens.CONTROL_HOVER)], foreground=[("selected", tokens.TEXT_PRIMARY)], bordercolor=[("selected", tokens.ACTIVE_EDGE)], lightcolor=[("selected", tokens.BORDER_STRONG)], darkcolor=[("selected", tokens.ACTIVE_SHADOW)], relief=[("selected", "raised")])
    style.configure("TPanedwindow", background=tokens.BORDER, sashwidth=1)
    style.configure("TSeparator", background=tokens.SEPARATOR)
    style.configure("DocumentTabs.TFrame", background=tokens.BACKGROUND)
    style.configure("DocumentTab.TButton", background=tokens.BACKGROUND, foreground=tokens.TEXT_SECONDARY, padding=(12, 6), borderwidth=0)
    style.map("DocumentTab.TButton", background=[("active", tokens.SURFACE_HOVER)], foreground=[("active", tokens.TEXT_PRIMARY)])
    style.configure("ActiveDocumentTab.TButton", background=tokens.CONTROL_SELECTED, foreground=tokens.TEXT_PRIMARY, bordercolor=tokens.ACTIVE_EDGE, lightcolor=tokens.BORDER_STRONG, darkcolor=tokens.ACTIVE_SHADOW, padding=(11, 5), borderwidth=1, relief="raised")
    style.map("ActiveDocumentTab.TButton", background=[("active", tokens.CONTROL_HOVER), ("pressed", tokens.CONTROL_PRESSED)], bordercolor=[("focus", tokens.FOCUS), ("active", tokens.ACTIVE_EDGE)])
    style.configure("DocumentTabClose.TButton", background=tokens.BACKGROUND, foreground=tokens.TEXT_SECONDARY, padding=(5, 6), borderwidth=0)
    style.map("DocumentTabClose.TButton", background=[("active", tokens.SURFACE_HOVER)], foreground=[("active", tokens.TEXT_PRIMARY)])
    style.configure("TLabelframe", background=tokens.SURFACE, bordercolor=tokens.SURFACE, relief="flat", borderwidth=0)
    style.configure("TLabelframe.Label", background=tokens.SURFACE, foreground=tokens.TEXT_SECONDARY, font=("Segoe UI Semibold", 9))
    style.configure("Treeview", background=tokens.SURFACE, fieldbackground=tokens.SURFACE, foreground=tokens.TEXT_PRIMARY, bordercolor=tokens.SURFACE, rowheight=28, borderwidth=0, relief="flat")
    style.map("Treeview", background=[("selected", tokens.CONTROL_SELECTED)], foreground=[("selected", tokens.TEXT_PRIMARY)])
    style.configure("Treeview.Heading", background=tokens.SURFACE_ELEVATED, foreground=tokens.TEXT_SECONDARY, bordercolor=tokens.BORDER, padding=(8, 6), relief="flat", font=("Segoe UI Semibold", 9))
    style.map("Treeview.Heading", background=[("active", tokens.SURFACE_HOVER)])
    style.configure("TProgressbar", background=tokens.ACCENT, troughcolor=tokens.SURFACE_ELEVATED, bordercolor=tokens.BORDER, lightcolor=tokens.ACCENT, darkcolor=tokens.ACCENT)
    style.configure("Vertical.TProgressbar", background=tokens.ACCENT, troughcolor=tokens.SURFACE_ELEVATED)
    style.configure("TMenubutton", background=tokens.SURFACE_HOVER, foreground=tokens.TEXT_PRIMARY, bordercolor=tokens.BORDER, arrowcolor=tokens.TEXT_SECONDARY, padding=(8, 4))
    return style
