from __future__ import annotations

from dataclasses import dataclass
import sys


@dataclass(frozen=True)
class Shortcut:
    command: str
    key: str
    accelerator: str
    shift: bool = False


COMMAND_SHORTCUTS = (
    Shortcut("new_document", "n", "Ctrl+N"),
    Shortcut("open", "o", "Ctrl+O"),
    Shortcut("save", "s", "Ctrl+S"),
    Shortcut("save_as", "s", "Ctrl+Shift+S", shift=True),
    Shortcut("undo", "z", "Ctrl+Z"),
    Shortcut("redo", "y", "Ctrl+Y"),
    Shortcut("redo", "z", "Ctrl+Shift+Z", shift=True),
    Shortcut("select_all", "a", "Ctrl+A"),
    Shortcut("deselect", "d", "Ctrl+D"),
    Shortcut("invert_selection", "i", "Ctrl+Shift+I", shift=True),
    Shortcut("copy", "c", "Ctrl+C"),
    Shortcut("cut", "x", "Ctrl+X"),
    Shortcut("paste", "v", "Ctrl+V"),
    Shortcut("new_layer", "n", "Ctrl+Shift+N", shift=True),
    Shortcut("duplicate_layer", "j", "Ctrl+J"),
    Shortcut("merge_down", "e", "Ctrl+E"),
    Shortcut("flatten", "e", "Ctrl+Shift+E", shift=True),
    Shortcut("free_transform", "t", "Ctrl+T"),
    Shortcut("fit_to_screen", "0", "Ctrl+0"),
    Shortcut("actual_size", "1", "Ctrl+1"),
)

PRIMARY_ACCELERATORS: dict[str, str] = {}
COMMAND_BY_GESTURE: dict[tuple[str, bool], str] = {}
for shortcut in COMMAND_SHORTCUTS:
    PRIMARY_ACCELERATORS.setdefault(shortcut.command, shortcut.accelerator)
    COMMAND_BY_GESTURE[(shortcut.key, shortcut.shift)] = shortcut.command


TOOL_SHORTCUT_GROUPS = {
    "v": ("move",),
    "h": ("hand",),
    "b": ("brush",),
    "e": ("eraser",),
    "r": ("blur_tool", "sharpen_tool"),
    "o": ("dodge", "burn"),
    "s": ("clone",),
    "j": ("healing", "spot_healing"),
    "k": ("patch",),
    "g": ("gradient", "fill"),
    "t": ("text",),
    "i": ("eyedropper",),
    "c": ("crop",),
    "m": ("select", "ellipse_select"),
    "l": ("lasso", "magnetic_lasso", "polygon_lasso"),
    "w": ("quick_selection", "magic_wand", "color_range"),
    "u": ("rect_shape", "ellipse_shape", "line_shape", "polygon_shape", "star_shape", "custom_shape"),
    "p": ("bezier_shape", "path_select", "direct_select", "add_anchor", "delete_anchor", "convert_anchor"),
}


CYRILLIC_KEYSYMS = {
    "cyrillic_ef": "a", "cyrillic_i": "b", "cyrillic_es": "c", "cyrillic_ve": "d",
    "cyrillic_u": "e", "cyrillic_a": "f", "cyrillic_pe": "g", "cyrillic_er": "h",
    "cyrillic_sha": "i", "cyrillic_o": "j", "cyrillic_el": "k", "cyrillic_de": "l",
    "cyrillic_softsign": "m", "cyrillic_te": "n", "cyrillic_shcha": "o", "cyrillic_ze": "p",
    "cyrillic_shorti": "q", "cyrillic_ka": "r", "cyrillic_yeru": "s", "cyrillic_ie": "t",
    "cyrillic_ghe": "u", "cyrillic_em": "v", "cyrillic_tse": "w", "cyrillic_che": "x",
    "cyrillic_en": "y", "cyrillic_ya": "z",
}


def event_key(event) -> str:
    """Return a layout-independent key for Windows and a keysym fallback elsewhere."""
    keycode = int(getattr(event, "keycode", -1))
    if 65 <= keycode <= 90 or 48 <= keycode <= 57:
        return chr(keycode).lower()
    if keycode == 219:
        return "["
    if keycode == 221:
        return "]"
    if keycode in {187, 107}:
        return "+"
    if keycode in {189, 109}:
        return "-"
    keysym = str(getattr(event, "keysym", "")).lower()
    if keysym in CYRILLIC_KEYSYMS:
        return CYRILLIC_KEYSYMS[keysym]
    return {
        "bracketleft": "[", "bracketright": "]", "plus": "+", "equal": "+",
        "kp_add": "+", "minus": "-", "kp_subtract": "-",
    }.get(keysym, keysym)


def command_for_event(event) -> str | None:
    state = int(getattr(event, "state", 0))
    if event_alt_down(state):
        return None
    shift = bool(state & 0x0001)
    return COMMAND_BY_GESTURE.get((event_key(event), shift))


def event_alt_down(event_or_state, platform: str | None = None) -> bool:
    state = int(event_or_state if isinstance(event_or_state, int) else getattr(event_or_state, "state", 0))
    alt_mask = 0x20000 if (platform or sys.platform) == "win32" else 0x0008
    return bool(state & alt_mask)


def accelerator(command: str) -> str:
    return PRIMARY_ACCELERATORS.get(command, "")


def validate_shortcut_registry() -> list[str]:
    errors: list[str] = []
    gestures: set[tuple[str, bool]] = set()
    for shortcut in COMMAND_SHORTCUTS:
        gesture = (shortcut.key, shortcut.shift)
        if gesture in gestures:
            errors.append(f"duplicate command gesture: {gesture}")
        gestures.add(gesture)
    assigned_tools: set[str] = set()
    for key, tools in TOOL_SHORTCUT_GROUPS.items():
        if not tools:
            errors.append(f"empty tool group: {key}")
        for tool in tools:
            if tool in assigned_tools:
                errors.append(f"tool assigned more than once: {tool}")
            assigned_tools.add(tool)
    return errors
