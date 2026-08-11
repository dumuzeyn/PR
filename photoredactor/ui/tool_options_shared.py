from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .theme import TOKENS
from ..brush_engine import BRUSH_BLEND_MODES


BRUSH_TOOLS = {"brush", "eraser", "blur_tool", "sharpen_tool", "dodge", "burn", "clone", "healing", "spot_healing"}
RETOUCH_TOOLS = {"blur_tool", "sharpen_tool", "dodge", "burn", "clone", "healing", "spot_healing"}
TOLERANCE_TOOLS = {"fill", "magic_wand", "color_range", "quick_selection"}
COLOR_TOOLS = {"brush", "fill", "gradient", "text"}
SHAPE_TOOLS = {"rect_shape", "ellipse_shape", "line_shape", "bezier_shape", "polygon_shape", "star_shape", "custom_shape"}
SELECTION_TOOLS = {"select", "ellipse_select", "lasso", "magnetic_lasso", "polygon_lasso", "quick_selection", "magic_wand", "color_range"}

__all__ = [name for name in globals() if not name.startswith("__")]
