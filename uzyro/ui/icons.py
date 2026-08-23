from __future__ import annotations

from functools import lru_cache
import math
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk

from .theme import TOKENS


from .shortcuts import TOOL_SHORTCUT_GROUPS


SHORTCUTS = {tool: key.upper() for key, tools in TOOL_SHORTCUT_GROUPS.items() for tool in tools}


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
    "path_select": "path", "direct_select": "path", "add_anchor": "path",
    "delete_anchor": "path", "convert_anchor": "path",
}


@lru_cache(maxsize=512)
def tool_icon_bitmap(tool: str, size: int = 20, color: str = TOKENS.TEXT_PRIMARY) -> Image.Image:
    """Draw one coherent, dependency-free outline icon set for the tool palette."""
    scale = 4
    image = Image.new("RGBA", (20 * scale, 20 * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    p = lambda value: round(value * scale)
    width = p(1.45)
    thin = p(1.05)

    def line(points, *, stroke=width, fill=color) -> None:
        draw.line([(p(x), p(y)) for x, y in points], fill=fill, width=stroke, joint="curve")

    def ellipse(box, *, stroke=width, fill=None) -> None:
        draw.ellipse(tuple(p(value) for value in box), outline=color, fill=fill, width=stroke)

    def rectangle(box, *, stroke=width, fill=None) -> None:
        draw.rectangle(tuple(p(value) for value in box), outline=color, fill=fill, width=stroke)

    def dashed_segment(start, end, dash=2.0, gap=1.5) -> None:
        x1, y1 = start
        x2, y2 = end
        length = math.hypot(x2 - x1, y2 - y1)
        if length <= 0:
            return
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        position = 0.0
        while position < length:
            stop = min(length, position + dash)
            line([(x1 + ux * position, y1 + uy * position), (x1 + ux * stop, y1 + uy * stop)], stroke=thin)
            position += dash + gap

    def dashed_rectangle(box) -> None:
        x1, y1, x2, y2 = box
        dashed_segment((x1, y1), (x2, y1))
        dashed_segment((x2, y1), (x2, y2))
        dashed_segment((x2, y2), (x1, y2))
        dashed_segment((x1, y2), (x1, y1))

    def sparkle(cx, cy, radius=2.5) -> None:
        line([(cx - radius, cy), (cx + radius, cy)], stroke=thin)
        line([(cx, cy - radius), (cx, cy + radius)], stroke=thin)

    if tool == "hand":
        points = [(5, 10), (5, 6), (7, 6), (7, 10), (7, 3), (9, 3), (9, 9), (9, 2), (11, 2), (11, 9), (11, 4), (13, 4), (13, 11), (15, 9), (17, 10), (15, 15), (12, 18), (8, 17), (5, 13)]
        draw.polygon([(p(x), p(y)) for x, y in points], outline=color, width=width)
        line([(7, 10), (11, 10), (13, 12)], stroke=thin)
    elif tool == "move":
        line([(10, 2), (10, 18)])
        line([(2, 10), (18, 10)])
        line([(10, 2), (8, 4)], stroke=thin); line([(10, 2), (12, 4)], stroke=thin)
        line([(18, 10), (16, 8)], stroke=thin); line([(18, 10), (16, 12)], stroke=thin)
        line([(10, 18), (8, 16)], stroke=thin); line([(10, 18), (12, 16)], stroke=thin)
        line([(2, 10), (4, 8)], stroke=thin); line([(2, 10), (4, 12)], stroke=thin)
    elif tool == "brush":
        line([(5, 15), (14, 6)], stroke=p(2.6))
        line([(12.5, 4.5), (15.5, 7.5)], stroke=width)
        draw.polygon([(p(3), p(17)), (p(5), p(12.5)), (p(8), p(15.5))], fill=color)
    elif tool == "eraser":
        points = [(3, 13), (11, 4), (17, 10), (9, 17)]
        draw.polygon([(p(x), p(y)) for x, y in points], outline=color, width=width)
        line([(6.5, 9), (12.5, 15)], stroke=thin)
        line([(4, 18), (17, 18)], stroke=thin)
    elif tool == "blur_tool":
        points = [(10, 2), (5, 9), (5, 13), (7, 16), (10, 18), (13, 16), (15, 13), (15, 9), (10, 2)]
        line(points)
        line([(8, 14), (9.5, 15.5), (12.5, 14.5)], stroke=thin)
    elif tool == "sharpen_tool":
        line([(10, 2), (18, 17), (2, 17), (10, 2)])
        line([(10, 7), (10, 13)], stroke=thin)
        ellipse((9.3, 14.5, 10.7, 15.9), stroke=thin, fill=color)
    elif tool == "dodge":
        ellipse((6, 6, 14, 14))
        for start, end in [((10, 2), (10, 4)), ((10, 16), (10, 18)), ((2, 10), (4, 10)), ((16, 10), (18, 10)), ((4, 4), (5.5, 5.5)), ((14.5, 14.5), (16, 16)), ((16, 4), (14.5, 5.5)), ((5.5, 14.5), (4, 16))]:
            line([start, end], stroke=thin)
    elif tool == "burn":
        line([(11, 2), (6, 8), (7, 12), (4, 10), (4, 14), (7, 18), (12, 18), (16, 14), (15, 9), (12, 12), (13, 7), (11, 2)])
        line([(10, 11), (8, 15), (10, 17), (12, 15), (10, 11)], stroke=thin)
    elif tool == "clone":
        line([(7, 3), (13, 3), (14, 7), (16, 9), (16, 12), (4, 12), (4, 9), (6, 7), (7, 3)])
        rectangle((3, 12, 17, 17), stroke=thin)
        line([(7, 6), (13, 6)], stroke=thin)
    elif tool == "healing":
        bandage = [(2.5, 13), (13, 2.5), (17.5, 7), (7, 17.5)]
        draw.polygon([(p(x), p(y)) for x, y in bandage], outline=color, width=width)
        line([(8, 8), (12, 12)], stroke=thin)
        line([(9.5, 6.5), (13.5, 10.5)], stroke=thin)
        ellipse((5, 13, 6.5, 14.5), stroke=thin, fill=color)
        ellipse((13.5, 5, 15, 6.5), stroke=thin, fill=color)
    elif tool == "spot_healing":
        bandage = [(2.5, 13), (13, 2.5), (17.5, 7), (7, 17.5)]
        draw.polygon([(p(x), p(y)) for x, y in bandage], outline=color, width=width)
        line([(8, 8), (12, 12)], stroke=thin)
        sparkle(15.5, 14.5, 2.5)
        ellipse((5, 13, 6.5, 14.5), stroke=thin, fill=color)
    elif tool == "patch":
        dashed_rectangle((3, 4, 14, 15))
        line([(11, 17), (17, 17), (17, 11)], stroke=thin)
        line([(17, 11), (15, 13)], stroke=thin)
    elif tool == "fill":
        points = [(4, 8), (9, 3), (16, 10), (10, 16), (3, 9)]
        draw.polygon([(p(x), p(y)) for x, y in points], outline=color, width=width)
        line([(4, 8), (13, 8)], stroke=thin)
        line([(16, 13), (14.5, 16), (16, 18), (17.5, 16), (16, 13)], stroke=thin)
    elif tool == "gradient":
        rectangle((2.5, 4, 17.5, 16))
        for x, stroke in ((6, thin), (10, width), (14, p(2.2))):
            line([(x, 5), (x, 15)], stroke=stroke)
    elif tool == "text":
        line([(3, 3), (17, 3)])
        line([(10, 3), (10, 17)])
        line([(7, 17), (13, 17)], stroke=thin)
    elif tool == "eyedropper":
        line([(4, 16), (14, 6)], stroke=p(2.3))
        ellipse((12, 2, 18, 8))
        line([(3, 17), (6, 17)], stroke=thin)
    elif tool == "rect_shape":
        rectangle((3, 4, 17, 16))
    elif tool == "ellipse_shape":
        ellipse((3, 4, 17, 16))
    elif tool == "line_shape":
        line([(3, 16), (17, 4)])
        ellipse((2, 15, 4, 17), stroke=thin, fill=color); ellipse((16, 3, 18, 5), stroke=thin, fill=color)
    elif tool == "bezier_shape":
        line([(3, 15), (7, 3), (14, 17), (17, 5)], stroke=thin)
        line([(3, 15), (7, 3)], stroke=thin); line([(14, 17), (17, 5)], stroke=thin)
        for x, y in ((3, 15), (7, 3), (14, 17), (17, 5)):
            ellipse((x - 1, y - 1, x + 1, y + 1), stroke=thin, fill=color if (x, y) in {(3, 15), (17, 5)} else TOKENS.SURFACE)
    elif tool in {"path_select", "direct_select", "add_anchor", "delete_anchor", "convert_anchor"}:
        line([(3, 15), (7, 4), (14, 16), (17, 5)], stroke=thin)
        line([(3, 15), (7, 4)], stroke=thin)
        line([(14, 16), (17, 5)], stroke=thin)
        for x, y in ((3, 15), (7, 4), (14, 16), (17, 5)):
            ellipse((x - 1, y - 1, x + 1, y + 1), stroke=thin, fill=TOKENS.SURFACE)
        if tool == "path_select":
            line([(5, 5), (5, 12), (8, 10), (10, 15), (12, 14), (9, 9)], stroke=thin, fill=TOKENS.ACCENT)
        elif tool == "direct_select":
            rectangle((1.5, 12.5, 5.5, 16.5), fill=TOKENS.ACCENT)
        elif tool == "add_anchor":
            line([(8, 9), (8, 15)], fill=TOKENS.ACCENT)
            line([(5, 12), (11, 12)], fill=TOKENS.ACCENT)
        elif tool == "delete_anchor":
            line([(5, 12), (11, 12)], fill=TOKENS.ACCENT)
        else:
            line([(6, 9), (10, 12), (6, 15)], fill=TOKENS.ACCENT)
    elif tool == "polygon_shape":
        line([(10, 2), (18, 8), (15, 17), (5, 17), (2, 8), (10, 2)])
    elif tool == "star_shape":
        points = [(10, 2), (12, 7), (18, 7), (13.5, 11), (15.5, 17), (10, 13.5), (4.5, 17), (6.5, 11), (2, 7), (8, 7), (10, 2)]
        line(points)
    elif tool == "custom_shape":
        line([(10, 2), (13, 7), (18, 10), (13, 13), (10, 18), (7, 13), (2, 10), (7, 7), (10, 2)])
    elif tool == "select":
        dashed_rectangle((3, 4, 17, 16))
    elif tool == "ellipse_select":
        for start in range(0, 360, 45):
            draw.arc((p(3), p(4), p(17), p(16)), start=start, end=start + 24, fill=color, width=thin)
    elif tool == "lasso":
        draw.arc((p(2), p(3), p(17), p(15)), 15, 335, fill=color, width=width)
        line([(14, 13), (17, 17), (12, 16)], stroke=thin)
    elif tool == "magnetic_lasso":
        draw.arc((p(2), p(3), p(15), p(15)), 20, 330, fill=color, width=width)
        line([(13, 12), (16, 16)], stroke=thin)
        line([(14, 4), (18, 4), (18, 10), (14, 10)], stroke=thin)
        line([(14, 5), (14, 3)], stroke=thin); line([(18, 5), (18, 3)], stroke=thin)
    elif tool == "polygon_lasso":
        points = [(3, 6), (9, 3), (17, 7), (14, 15), (7, 17), (3, 12), (3, 6)]
        for first, second in zip(points, points[1:]):
            dashed_segment(first, second, 1.8, 1.2)
        ellipse((2, 5, 4, 7), stroke=thin, fill=color)
    elif tool == "quick_selection":
        draw.arc((p(2), p(3), p(14), p(15)), 20, 325, fill=color, width=thin)
        line([(10, 15), (17, 8)], stroke=p(2.2))
        sparkle(16, 4, 2)
    elif tool == "magic_wand":
        line([(4, 17), (13, 8)], stroke=p(2.1))
        sparkle(14.5, 5, 3)
        sparkle(6, 5, 1.6)
    elif tool == "color_range":
        line([(3, 16), (11, 8)], stroke=p(2.1))
        ellipse((10, 3, 13, 6), stroke=thin, fill=color)
        ellipse((15, 6, 18, 9), stroke=thin, fill=color)
        ellipse((12, 12, 15, 15), stroke=thin, fill=color)
    elif tool == "crop":
        line([(6, 2), (6, 14), (18, 14)])
        line([(2, 6), (14, 6), (14, 18)])
    else:
        ellipse((4, 4, 16, 16))

    return image.resize((size, size), Image.Resampling.LANCZOS)


def _dpi_scale(master: tk.Misc) -> float:
    try:
        # Tk defines 1.0 scaling as 72 DPI; Windows logical controls use 96 DPI.
        return max(1.0, min(4.0, float(master.tk.call("tk", "scaling")) / (96.0 / 72.0)))
    except (AttributeError, TypeError, ValueError, tk.TclError):
        return 1.0


def _window_icon_cache(master: tk.Misc) -> dict[tuple[object, ...], ImageTk.PhotoImage]:
    try:
        owner = master.winfo_toplevel()
    except (AttributeError, tk.TclError):
        owner = master
    cache = getattr(owner, "_uzyro_icon_cache", None)
    if cache is None:
        cache = {}
        setattr(owner, "_uzyro_icon_cache", cache)
    return cache


def tool_icon(master: tk.Misc, tool: str, size: int = 20, color: str = TOKENS.TEXT_PRIMARY) -> ImageTk.PhotoImage:
    scale = _dpi_scale(master)
    physical_size = max(1, round(size * scale))
    key = ("tool", tool, size, round(scale, 3), color)
    cache = _window_icon_cache(master)
    if key not in cache:
        cache[key] = ImageTk.PhotoImage(tool_icon_bitmap(tool, physical_size, color), master=master)
    return cache[key]


@lru_cache(maxsize=256)
def action_icon_bitmap(action: str, size: int = 15, color: str = TOKENS.TEXT_PRIMARY) -> Image.Image:
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
    elif action == "group":
        draw.rectangle((p(2), p(5), p(size - 2), p(size - 3)), outline=color, width=width)
        draw.line((p(3), p(5), p(6), p(2), p(10), p(2), p(12), p(5)), fill=color, width=width)
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
    return image


def action_icon(master: tk.Misc, action: str, size: int = 15, color: str = TOKENS.TEXT_PRIMARY) -> ImageTk.PhotoImage:
    scale = _dpi_scale(master)
    physical_size = max(1, round(size * scale))
    key = ("action", action, size, round(scale, 3), color)
    cache = _window_icon_cache(master)
    if key not in cache:
        cache[key] = ImageTk.PhotoImage(action_icon_bitmap(action, physical_size, color), master=master)
    return cache[key]
