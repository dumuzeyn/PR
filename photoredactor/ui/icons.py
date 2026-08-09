from __future__ import annotations

import math
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk

from .theme import TOKENS


SHORTCUTS = {
    "move": "V", "hand": "H", "brush": "B", "eraser": "E", "fill": "G", "gradient": "G",
    "text": "T", "eyedropper": "I", "crop": "C", "select": "M", "ellipse_select": "M",
    "lasso": "L", "magnetic_lasso": "L", "polygon_lasso": "L", "quick_selection": "W",
    "magic_wand": "W", "color_range": "W", "clone": "S", "healing": "J", "spot_healing": "J",
    "rect_shape": "U", "ellipse_shape": "U", "line_shape": "U", "bezier_shape": "P",
    "polygon_shape": "U", "star_shape": "U", "custom_shape": "U",
}


TOOL_GROUPS = {
    "move": "navigation", "hand": "navigation",
    "select": "selection", "ellipse_select": "selection", "lasso": "selection", "magnetic_lasso": "selection",
    "polygon_lasso": "selection", "quick_selection": "selection", "magic_wand": "selection", "color_range": "selection",
    "crop": "crop",
    "brush": "paint", "eraser": "paint", "fill": "paint", "gradient": "paint", "eyedropper": "paint",
    "blur_tool": "retouch", "sharpen_tool": "retouch", "dodge": "retouch", "burn": "retouch",
    "clone": "retouch", "healing": "retouch", "spot_healing": "retouch", "patch": "retouch",
    "text": "text",
    "rect_shape": "shape", "ellipse_shape": "shape", "line_shape": "shape", "bezier_shape": "shape",
    "polygon_shape": "shape", "star_shape": "shape", "custom_shape": "shape",
}


def tool_icon(master: tk.Misc, tool: str, size: int = 20, color: str = TOKENS.TEXT_PRIMARY) -> ImageTk.PhotoImage:
    scale = 3
    image = Image.new("RGBA", (size * scale, size * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    c = color
    w = 2 * scale
    p = lambda value: int(value * scale)
    if tool in {"rect_shape", "select", "crop"}:
        draw.rectangle((p(3), p(4), p(size - 3), p(size - 4)), outline=c, width=w)
    elif tool in {"ellipse_shape", "ellipse_select"}:
        draw.ellipse((p(3), p(4), p(size - 3), p(size - 4)), outline=c, width=w)
    elif tool in {"line_shape", "bezier_shape"}:
        draw.line((p(3), p(size - 4), p(size - 3), p(4)), fill=c, width=w)
        if tool == "bezier_shape":
            draw.ellipse((p(2), p(size - 5), p(5), p(size - 2)), fill=c)
            draw.ellipse((p(size - 5), p(2), p(size - 2), p(5)), fill=c)
    elif tool in {"polygon_shape", "star_shape", "custom_shape"}:
        points = [(p(size / 2), p(2)), (p(size - 2), p(size - 6)), (p(size - 5), p(size - 2)), (p(5), p(size - 2)), (p(2), p(size - 6))]
        draw.line(points + [points[0]], fill=c, width=w, joint="curve")
    elif tool == "move":
        draw.line((p(size / 2), p(2), p(size / 2), p(size - 2)), fill=c, width=w)
        draw.line((p(2), p(size / 2), p(size - 2), p(size / 2)), fill=c, width=w)
        for x, y in ((size / 2, 2), (size / 2, size - 2), (2, size / 2), (size - 2, size / 2)):
            draw.ellipse((p(x - 1.5), p(y - 1.5), p(x + 1.5), p(y + 1.5)), fill=c)
    elif tool == "hand":
        draw.rounded_rectangle((p(5), p(5), p(size - 4), p(size - 2)), radius=p(3), outline=c, width=w)
        draw.line((p(8), p(2), p(8), p(11)), fill=c, width=w)
    elif tool == "brush":
        draw.line((p(4), p(size - 3), p(size - 4), p(4)), fill=c, width=p(3))
        draw.rectangle((p(size - 8), p(2), p(size - 3), p(7)), outline=c, width=w)
    elif tool == "eraser":
        draw.polygon([(p(3), p(size - 6)), (p(size - 8), p(3)), (p(size - 2), p(9)), (p(8), p(size - 2))], outline=c)
        draw.line((p(5), p(size - 4), p(size - 2), p(size - 4)), fill=c, width=w)
    elif tool in {"blur_tool", "sharpen_tool", "dodge", "burn"}:
        draw.ellipse((p(3), p(3), p(size - 3), p(size - 3)), outline=c, width=w)
        if tool in {"sharpen_tool", "dodge"}:
            draw.line((p(6), p(size / 2), p(size - 6), p(size / 2)), fill=c, width=w)
        if tool in {"blur_tool", "dodge"}:
            draw.line((p(size / 2), p(6), p(size / 2), p(size - 6)), fill=c, width=w)
        if tool == "burn":
            draw.arc((p(6), p(4), p(size - 5), p(size - 4)), 80, 285, fill=c, width=w)
    elif tool in {"clone", "healing", "spot_healing", "patch"}:
        draw.ellipse((p(4), p(4), p(size - 4), p(size - 4)), outline=c, width=w)
        draw.line((p(size / 2), p(7), p(size / 2), p(size - 7)), fill=c, width=w)
        draw.line((p(7), p(size / 2), p(size - 7), p(size / 2)), fill=c, width=w)
    elif tool == "text":
        draw.line((p(4), p(3), p(size - 4), p(3)), fill=c, width=w)
        draw.line((p(size / 2), p(3), p(size / 2), p(size - 3)), fill=c, width=w)
    elif tool in {"fill", "gradient"}:
        draw.polygon([(p(4), p(8)), (p(9), p(3)), (p(size - 3), p(size - 7)), (p(size - 8), p(size - 2))], outline=c)
        draw.line((p(4), p(size - 3), p(size - 2), p(size - 3)), fill=c, width=w)
    elif tool == "eyedropper":
        draw.line((p(4), p(size - 3), p(size - 4), p(3)), fill=c, width=p(3))
        draw.ellipse((p(size - 8), p(2), p(size - 2), p(8)), outline=c, width=w)
    elif tool in {"lasso", "magnetic_lasso", "polygon_lasso"}:
        draw.arc((p(2), p(3), p(size - 2), p(size - 3)), 15, 330, fill=c, width=w)
        draw.line((p(size - 6), p(size - 5), p(size - 2), p(size - 2)), fill=c, width=w)
    elif tool in {"quick_selection", "magic_wand", "color_range"}:
        draw.line((p(4), p(size - 3), p(size - 5), p(4)), fill=c, width=w)
        for x, y in ((3, 4), (size - 3, 3), (size - 2, 10)):
            draw.line((p(x - 1), p(y), p(x + 1), p(y)), fill=c, width=w)
    else:
        draw.ellipse((p(4), p(4), p(size - 4), p(size - 4)), outline=c, width=w)
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(image, master=master)


def action_icon(master: tk.Misc, action: str, size: int = 15, color: str = TOKENS.TEXT_PRIMARY) -> ImageTk.PhotoImage:
    scale = 3
    image = Image.new("RGBA", (size * scale, size * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    p = lambda value: int(value * scale)
    width = max(2, p(1.4))
    center = p(size / 2)
    if action in {"add", "delete"}:
        draw.line((p(3), center, p(size - 3), center), fill=color, width=width)
        if action == "add":
            draw.line((center, p(3), center, p(size - 3)), fill=color, width=width)
    elif action == "duplicate":
        draw.rectangle((p(3), p(5), p(size - 5), p(size - 3)), outline=color, width=width)
        draw.rectangle((p(6), p(2), p(size - 2), p(size - 6)), outline=color, width=width)
    elif action in {"up", "down"}:
        if action == "up":
            points = [(p(3), p(size - 4)), (center, p(4)), (p(size - 3), p(size - 4))]
        else:
            points = [(p(3), p(4)), (center, p(size - 4)), (p(size - 3), p(4))]
        draw.line(points, fill=color, width=width, joint="curve")
    elif action == "settings":
        radius = p(size * 0.28)
        inner = p(size * 0.12)
        for angle in range(0, 360, 45):
            radians = math.radians(angle)
            x1 = center + int(radius * 0.9 * math.cos(radians))
            y1 = center + int(radius * 0.9 * math.sin(radians))
            x2 = center + int(radius * 1.35 * math.cos(radians))
            y2 = center + int(radius * 1.35 * math.sin(radians))
            draw.line((x1, y1, x2, y2), fill=color, width=width)
        draw.ellipse((center - radius, center - radius, center + radius, center + radius), outline=color, width=width)
        draw.ellipse((center - inner, center - inner, center + inner, center + inner), outline=color, width=width)
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(image, master=master)
