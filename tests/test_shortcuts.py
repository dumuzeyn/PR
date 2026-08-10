from types import SimpleNamespace

from photoredactor.app import TOOL_DEFINITIONS
from photoredactor.ui.icons import SHORTCUTS
from photoredactor.ui.shortcuts import (
    TOOL_SHORTCUT_GROUPS,
    accelerator,
    command_for_event,
    event_key,
    validate_shortcut_registry,
)


def event(keycode: int, keysym: str, state: int = 0x0004) -> SimpleNamespace:
    return SimpleNamespace(keycode=keycode, keysym=keysym, state=state)


def test_shortcut_registry_has_no_conflicts() -> None:
    assert validate_shortcut_registry() == []


def test_control_commands_use_physical_keys_and_shift() -> None:
    assert command_for_event(event(90, "Cyrillic_ya")) == "undo"
    assert command_for_event(event(90, "Cyrillic_ya", 0x0005)) == "redo"
    assert command_for_event(event(83, "Cyrillic_yeru", 0x0005)) == "save_as"
    assert command_for_event(event(73, "Cyrillic_sha", 0x0005)) == "invert_selection"
    assert command_for_event(event(48, "0")) == "fit_to_screen"
    assert command_for_event(event(79, "o", 0x20004)) is None


def test_cyrillic_keys_have_a_non_windows_fallback() -> None:
    assert event_key(event(-1, "Cyrillic_em", 0)) == "v"
    assert event_key(event(-1, "Cyrillic_de", 0)) == "l"
    assert event_key(event(219, "Cyrillic_ha", 0)) == "["
    assert event_key(event(221, "Cyrillic_hardsign", 0)) == "]"


def test_every_tool_has_exactly_one_displayed_shortcut() -> None:
    tool_ids = {tool_id for _label, tool_id, _description in TOOL_DEFINITIONS}
    grouped = {tool for tools in TOOL_SHORTCUT_GROUPS.values() for tool in tools}
    assert grouped == tool_ids
    assert set(SHORTCUTS) == tool_ids
    for key, tools in TOOL_SHORTCUT_GROUPS.items():
        assert all(SHORTCUTS[tool] == key.upper() for tool in tools)


def test_primary_accelerators_match_visible_menu_text() -> None:
    assert accelerator("new_document") == "Ctrl+N"
    assert accelerator("save_as") == "Ctrl+Shift+S"
    assert accelerator("redo") == "Ctrl+Y"
    assert accelerator("flatten") == "Ctrl+Shift+E"
    assert accelerator("actual_size") == "Ctrl+1"
