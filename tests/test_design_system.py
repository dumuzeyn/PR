from __future__ import annotations

import tkinter as tk
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from uzyro.ui.desktop_controls import AccentScale, LayerList, SlimScrollbar
from uzyro.ui.icons import action_icon, tool_icon, tool_icon_bitmap
from uzyro.ui.menu_bar import DarkMenuButton, DesktopMenuBar
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


@pytest.mark.ui
def test_desktop_controls_keep_geometry_and_listbox_contract() -> None:
    root = tk.Tk()
    root.geometry("360x260")
    configure_theme(root)
    value = tk.DoubleVar(root, value=0.25)
    changes: list[float] = []
    scale = AccentScale(root, variable=value, length=180, command=lambda raw: changes.append(float(raw)))
    scale.pack(fill=tk.X)
    scrollbar = SlimScrollbar(root, command=lambda *_args: None)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    layers = LayerList(root)
    layers.pack(fill=tk.BOTH, expand=True)
    layers.insert(tk.END, "Первый", "Второй")
    pixels = np.zeros((12, 12, 4), dtype=np.uint8)
    layer_data = [
        SimpleNamespace(id="one", pixels=pixels, pixels_revision=0, name="Первый", visible=True, locked=False, mask=None, effects=[]),
        SimpleNamespace(id="two", pixels=pixels, pixels_revision=0, name="Второй", visible=False, locked=True, mask=pixels[..., 0], effects=[]),
    ]
    layers.set_layers(layer_data, lambda _pixels, size: Image.new("RGBA", (size, size), (70, 80, 90, 255)))
    try:
        root.update()
        geometry = (scale.winfo_width(), scale.winfo_height())
        scale._enter(None)
        assert (scale.winfo_width(), scale.winfo_height()) == geometry
        scale.set(0.75, notify=True)
        assert value.get() == pytest.approx(0.75)
        assert changes[-1] == pytest.approx(0.75)
        layers.selection_set(1)
        layers.activate(1)
        assert layers.get(0, tk.END) == ("Первый", "Второй")
        assert layers.curselection() == (1,)
        assert layers.index(tk.ACTIVE) == 1
        assert layers.bbox(1) is not None
    finally:
        root.destroy()


@pytest.mark.ui
def test_dark_menu_components_render_custom_popup_rows() -> None:
    root = tk.Tk()
    root.geometry("480x240")
    configure_theme(root)
    menu = tk.Menu(root, tearoff=False)
    file_menu = tk.Menu(menu, tearoff=False)
    file_menu.add_command(label="Открыть", accelerator="Ctrl+O")
    file_menu.add_separator()
    file_menu.add_command(label="Закрыть")
    menu.add_cascade(label="Файл", menu=file_menu)
    bar = DesktopMenuBar(root, menu)
    bar.pack(fill=tk.X)
    button = DarkMenuButton(root, text="Пресеты", width=14)
    button_menu = tk.Menu(button, tearoff=False)
    button_menu.add_command(label="Круглая кисть")
    button.configure(menu=button_menu)
    button.pack(anchor=tk.W, pady=12)
    try:
        root.update()
        bar.open_popup(bar._buttons[0], 0)
        root.update()
        assert bar.popup is not None
        popup_edge = bar.popup.winfo_children()[0]
        assert popup_edge.cget("background") == TOKENS.PANEL_RAISED
        assert popup_edge.winfo_children()[0].winfo_height() == 28
        bar.close_popup()
        button._toggle()
        root.update()
        assert button.popup is not None
        assert button.winfo_height() == 28
        button.close_popup()
    finally:
        root.destroy()
