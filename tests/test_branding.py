from __future__ import annotations

from pathlib import Path

from PIL import Image

from uzyro.brand import APP_NAME, branding_asset, user_data_directory


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
