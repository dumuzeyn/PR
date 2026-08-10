from __future__ import annotations

import tkinter as tk
from types import SimpleNamespace
from tkinter import messagebox, ttk

from .scrollable_frame import ScrollableFrame
from .icons import SHORTCUTS, TOOL_GROUPS, action_icon, tool_icon
from .theme import TOKENS

ToolDefinition = tuple[str, str, str]


def normalize_tool_order(saved_order: list[str] | None, definitions: list[ToolDefinition]) -> list[str]:
    valid = [value for _label, value, _description in definitions]
    seen: set[str] = set()
    order: list[str] = []
    for value in saved_order or []:
        if value in valid and value not in seen:
            order.append(value)
            seen.add(value)
    for value in valid:
        if value not in seen:
            order.append(value)
    return order


def normalize_visible_tools(saved_visible: list[str] | None, order: list[str]) -> list[str]:
    visible = [value for value in saved_visible or [] if value in order]
    if not visible:
        visible = list(order)
    return visible

__all__ = [name for name in globals() if not name.startswith("__")]
