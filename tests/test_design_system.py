from __future__ import annotations

import tkinter as tk

import pytest

from uzyro.ui.icons import action_icon, tool_icon, tool_icon_bitmap
from uzyro.ui.theme import DesignTokens, TOKENS, configure_theme
from uzyro.ui.tooltip import ToolTip


def test_design_tokens_expose_semantic_neutral_palette() -> None:
    required = {
        "APP_BG", "WORKSPACE_BG", "PANEL_BG", "PANEL_RAISED",
        "CONTROL_BG", "CONTROL_HOVER", "CONTROL_PRESSED", "CONTROL_SELECTED",
        "BORDER_SUBTLE", "BORDER_STRONG", "TEXT_PRIMARY", "TEXT_SECONDARY",
        "TEXT_MUTED", "TEXT_DISABLED", "ACCENT", "ACCENT_HOVER",
        "ACCENT_PRESSED", "FOCUS", "SUCCESS", "WARNING", "DANGER",
    }
    assert required <= set(DesignTokens.__dataclass_fields__)
    assert TOKENS.CONTROL_SELECTED != TOKENS.ACCENT
    assert TOKENS.WORKSPACE_BG != TOKENS.PANEL_BG != TOKENS.CONTROL_BG


def test_tool_bitmap_rendering_is_sized_and_cached() -> None:
    first = tool_icon_bitmap("brush", 24)
    second = tool_icon_bitmap("brush", 24)
    assert first is second
    assert first.size == (24, 24)
    assert first.getbbox() is not None


@pytest.mark.ui
def test_selected_styles_use_neutral_raised_surface() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        style = configure_theme(root)
        selected = style.lookup("Tool.TRadiobutton", "background", ("selected",))
        edge = style.lookup("Tool.TRadiobutton", "bordercolor", ("selected",))
        assert selected == TOKENS.CONTROL_SELECTED
        assert selected != TOKENS.ACCENT
        assert edge == TOKENS.ACTIVE_EDGE
        assert style.lookup("ActiveDocumentTab.TButton", "relief") == "raised"
        assert style.lookup("Sidebar.TNotebook.Tab", "background", ("selected",)) == TOKENS.CONTROL_SELECTED
    finally:
        root.destroy()


@pytest.mark.ui
@pytest.mark.parametrize(
    ("tk_scaling", "expected_size"),
    [(4 / 3, 16), (5 / 3, 20), (2.0, 24), (7 / 3, 28), (8 / 3, 32)],
    ids=["100-percent", "125-percent", "150-percent", "175-percent", "200-percent"],
)
def test_icons_follow_tk_dpi_and_reuse_window_cache(tk_scaling: float, expected_size: int) -> None:
    root = tk.Tk()
    root.withdraw()
    original_scaling = float(root.tk.call("tk", "scaling"))
    try:
        root.tk.call("tk", "scaling", tk_scaling)
        brush = tool_icon(root, "brush", 16)
        assert brush is tool_icon(root, "brush", 16)
        assert brush.width() == expected_size
        add = action_icon(root, "add", 16)
        assert add is action_icon(root, "add", 16)
        assert add.width() == expected_size
    finally:
        root.tk.call("tk", "scaling", original_scaling)
        root.destroy()


@pytest.mark.ui
def test_tooltip_uses_raised_surface_and_cleans_up() -> None:
    root = tk.Tk()
    root.withdraw()
    button = tk.Button(root, text="Инструмент")
    button.pack()
    tooltip = ToolTip(button, "Описание")
    try:
        tooltip.show()
        assert tooltip._tip is not None
        shadow = tooltip._tip.winfo_children()[0]
        body = shadow.winfo_children()[0]
        assert shadow.cget("background") == TOKENS.ACTIVE_SHADOW
        assert body.cget("background") == TOKENS.PANEL_RAISED
        tooltip.hide()
        assert tooltip._tip is None
    finally:
        root.destroy()
