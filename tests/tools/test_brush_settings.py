from __future__ import annotations

from uzyro.app_shared import BRUSH_PRESET_DEFAULTS, TOOL_SETTINGS_DEFAULTS
from uzyro.brush_config import is_test_polluted_brush_settings


def test_all_brush_based_tools_start_hard_and_continuous() -> None:
    for tool, settings in TOOL_SETTINGS_DEFAULTS.items():
        if "hardness" in settings:
            assert settings["hardness"] == 1.0, tool
        if "spacing" in settings:
            assert settings["spacing"] == 0.0, tool


def test_builtin_brush_library_contains_distinct_practical_presets() -> None:
    assert len(BRUSH_PRESET_DEFAULTS) >= 10
    assert any(values.get("advanced", {}).get("scatter", 0) > 0 for values in BRUSH_PRESET_DEFAULTS.values())
    assert any(values.get("advanced", {}).get("roundness", 1) < 0.5 for values in BRUSH_PRESET_DEFAULTS.values())


def test_old_ui_test_profile_is_recognized_without_matching_normal_user_settings() -> None:
    polluted = {
        "size": 18, "hardness": 0.17, "opacity": 0.83, "flow": 0.3,
        "spacing": 0.2, "smoothing": 0.15, "blend_mode": "Overlay",
    }
    advanced = {"angle": 37.0, "roundness": 0.42, "scatter": 1.25}
    assert is_test_polluted_brush_settings(polluted, advanced)

    changed_size = {**polluted, "size": 19, "smoothing": 0.0}
    assert is_test_polluted_brush_settings(changed_size, advanced)

    user_overlay = {**polluted, "hardness": 0.18}
    assert not is_test_polluted_brush_settings(user_overlay, advanced)
    assert not is_test_polluted_brush_settings(polluted, {**advanced, "scatter": 1.0})
