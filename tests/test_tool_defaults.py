from photoredactor.app_shared import BRUSH_PRESET_DEFAULTS, TOOL_SETTINGS_DEFAULTS


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
