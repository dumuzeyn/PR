from __future__ import annotations

import ctypes
import os
from pathlib import Path
import tkinter as tk

from PIL import Image
import pytest

from uzyro.brand import APP_NAME, apply_window_branding, branding_asset, user_data_directory


ROOT = Path(__file__).resolve().parents[1]


def test_brand_assets_have_source_and_windows_icon_sizes() -> None:
    source = Image.open(ROOT / "design_assets" / "branding" / "UZYRO-logo-2048.png")
    application_icon = Image.open(branding_asset("uzyro-icon.png"))
    windows_icon = Image.open(branding_asset("uzyro.ico"))
    assert APP_NAME == "UZYRO"
    assert source.size == (2048, 2048)
    assert application_icon.size == (512, 512)
    assert windows_icon.info["sizes"] == {
        (16, 16), (20, 20), (24, 24), (32, 32), (40, 40),
        (48, 48), (64, 64), (128, 128), (256, 256),
    }


def test_user_settings_are_migrated_to_uzyro_directory(tmp_path: Path) -> None:
    legacy = tmp_path / ("Photo" + "Redactor")
    legacy.mkdir()
    (legacy / "settings.json").write_text('{"recent_files": []}', encoding="utf-8")
    target = user_data_directory(tmp_path)
    assert target.name == "UZYRO"
    assert (target / "settings.json").read_text(encoding="utf-8") == '{"recent_files": []}'


@pytest.mark.skipif(os.name != "nt", reason="Windows title-bar icon integration")
def test_titlebar_receives_native_small_and_large_icons() -> None:
    window = tk.Tk()
    try:
        window.title("UZYRO icon test")
        apply_window_branding(window)
        window.update()
        handles = getattr(window, "_brand_native_icons", ())
        assert len(handles) == 2 and all(handles)

        user32 = ctypes.windll.user32
        top_level = int(user32.GetParent(window.winfo_id())) or int(window.winfo_id())
        assert int(user32.SendMessageW(top_level, 0x007F, 0, 0)) == handles[0]
        assert int(user32.SendMessageW(top_level, 0x007F, 1, 0)) == handles[1]
    finally:
        window.destroy()
