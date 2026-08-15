from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from uzyro.brush_engine import BrushSettings, PixelBrushStroke
from uzyro.core import Document
from uzyro.patch_retouch import build_patch_edit
from uzyro.retouch_ops import RetouchStroke, apply_gradient, flood_fill, spot_heal
from uzyro.shape_ops import editable_bezier_nodes, render_shape_layer
from uzyro.source_retouch import CloneHealingStroke
from uzyro.vector_geometry import split_cubic_bezier


SIZE = (288, 162)
PURPLE = (132, 92, 246, 255)
CYAN = (41, 190, 206, 255)


@dataclass
class DemoFrame:
    image: Image.Image
    cursor: tuple[int, int]
    selection: np.ndarray | None = None
    guide: list[tuple[int, int]] = field(default_factory=list)
    guide_closed: bool = False
    nodes: list[dict[str, object]] = field(default_factory=list)
    active_node: int | None = None
    source: tuple[int, int] | None = None
    swatch: tuple[int, int, int, int] | None = None


def document_from(image: Image.Image) -> Document:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    document = Document.new(rgba.shape[1], rgba.shape[0], (0, 0, 0, 0))
    document.layer.pixels = rgba
    document.layer.touch_pixels()
    return document


def output(document: Document, checker: bool = False) -> Image.Image:
    return Image.fromarray(document.composite(checker), mode="RGBA").convert("RGB")


def points_between(start: tuple[int, int], end: tuple[int, int], amount: float, step: int = 5) -> list[tuple[int, int]]:
    distance = max(1, round(np.hypot(end[0] - start[0], end[1] - start[1])))
    count = max(1, distance // max(1, step))
    limit = max(1, round(count * amount))
    return [
        (
            round(start[0] + (end[0] - start[0]) * index / count),
            round(start[1] + (end[1] - start[1]) * index / count),
        )
        for index in range(limit + 1)
    ]


def paint_stroke(document: Document, amount: float, *, erase: bool = False) -> tuple[int, int]:
    start, end = (55, 122), (226, 42)
    settings = BrushSettings(radius=9, hardness=1.0, opacity=1.0, flow=1.0, spacing=0.0)
    stroke = PixelBrushStroke(document.layer, settings, PURPLE, erase=erase)
    points = points_between(start, end, amount, 4)
    for point in points:
        stroke.dab(*point)
    return points[-1]


def retouch_frame(tool: str, amount: float, base: Image.Image) -> DemoFrame:
    document = document_from(base)
    start, end = (82, 102), (208, 86)
    mode = {"blur_tool": "blur", "sharpen_tool": "sharpen", "dodge": "dodge", "burn": "burn"}[tool]
    strength = 0.75 if tool in {"blur_tool", "sharpen_tool"} else 0.55
    stroke = RetouchStroke(document.layer, mode, 18, 1.0, strength, "midtones")
    points = points_between(start, end, amount, 7)
    for point in points:
        stroke.dab(*point)
    return DemoFrame(output(document), points[-1])


def clone_frame(tool: str, amount: float, base: Image.Image) -> DemoFrame:
    document = document_from(base)
    source_pixels = document.layer.pixels.copy()
    source = (42, 108)
    target_start, target_end = (205, 108), (242, 108)
    settings = BrushSettings(radius=22, hardness=0.85, opacity=1.0, flow=1.0, spacing=0.0)
    stroke = CloneHealingStroke(document.layer, settings, source_pixels, heal=tool == "healing")
    points = points_between(target_start, target_end, amount, 8)
    for index, point in enumerate(points):
        stroke.dab(point[0], point[1], source[0] + index * 4, source[1])
    return DemoFrame(output(document), points[-1], source=source)


def blemished(base: Image.Image) -> Document:
    document = document_from(base)
    stroke = PixelBrushStroke(
        document.layer,
        BrushSettings(radius=7, hardness=0.7, opacity=1.0, flow=1.0, spacing=0.0),
        (45, 35, 30, 255),
    )
    stroke.dab(171, 107)
    return document


def healing_frame(tool: str, amount: float, base: Image.Image) -> DemoFrame:
    document = blemished(base)
    target = (171, 107)
    if tool == "spot_healing":
        if amount > 0:
            spot_heal(document.layer, *target, 11, amount, hardness=1.0, mode="proximity")
        return DemoFrame(output(document), target)
    mask = np.zeros((SIZE[1], SIZE[0]), dtype=np.uint8)
    cv2.rectangle(mask, (159, 95), (183, 119), 255, -1)
    source_x = round(119 + 25 * amount)
    edit = build_patch_edit(
        document.layer.pixels,
        (0, 0),
        mask,
        source_x,
        95,
        source_pixels=document.layer.pixels.copy(),
        source_origin=(0, 0),
        structure=5,
        color_adaptation=8.0,
    )
    if edit is not None and amount >= 0.45:
        (x1, y1, x2, y2), pixels = edit
        document.layer.pixels[y1:y2, x1:x2] = pixels
    guide = [(159, 95), (183, 95), (183, 119), (159, 119)]
    return DemoFrame(output(document), (source_x, 107), selection=mask, guide=guide, guide_closed=True)


def fill_gradient_text_frame(tool: str, amount: float, base: Image.Image) -> DemoFrame:
    document = document_from(base)
    if tool == "fill":
        point = (145, 27)
        if amount > 0.2:
            flood_fill(document.layer, *point, PURPLE, round(8 + 45 * amount))
        return DemoFrame(output(document), point)
    if tool == "gradient":
        start, end = (28, 132), (258, 30)
        current = (
            round(start[0] + (end[0] - start[0]) * amount),
            round(start[1] + (end[1] - start[1]) * amount),
        )
        if amount > 0:
            apply_gradient(
                document.layer,
                (*start, *current),
                (25, 95, 240, 255),
                (238, 93, 190, 255),
                kind="linear",
                options={"dither": True},
            )
        return DemoFrame(output(document), current, guide=[start, current])
    text = "UZYRO"
    visible = text[: max(0, round(len(text) * amount))]
    if visible:
        document.add_text_layer(visible, 74, 61, PURPLE, 38, "segoeuib.ttf")
    return DemoFrame(output(document), (76 + round(119 * amount), 101))


def shape_frame(tool: str, amount: float, base: Image.Image) -> DemoFrame:
    document = document_from(base)
    start, end = (61, 38), (229, 131)
    current = (
        round(start[0] + (end[0] - start[0]) * amount),
        round(start[1] + (end[1] - start[1]) * amount),
    )
    shape = {
        "rect_shape": "rectangle", "ellipse_shape": "ellipse", "line_shape": "line",
        "bezier_shape": "bezier", "polygon_shape": "polygon", "star_shape": "star",
        "custom_shape": "custom",
    }[tool]
    if amount > 0.02:
        fill = (132, 92, 246, 105) if shape != "line" else (0, 0, 0, 0)
        custom = None
        if shape == "custom":
            custom = [(0.0, 0.45), (0.35, 0.45), (0.5, 0.0), (0.65, 0.45), (1.0, 0.45), (0.72, 0.68), (0.82, 1.0), (0.5, 0.8), (0.18, 1.0), (0.28, 0.68)]
        document.add_shape_layer(shape, (*start, *current), fill, (255, 255, 255, 255), 3, custom_points=custom)
    return DemoFrame(output(document), current, guide=[start, current])


def base_path(document: Document) -> tuple[object, list[dict[str, object]]]:
    layer = document.add_shape_layer("bezier", (35, 34, 251, 131), (0, 0, 0, 0), PURPLE, 4)
    nodes = editable_bezier_nodes(layer.shape_data or {})
    layer.shape_data["path_nodes"] = deepcopy(nodes)
    render_shape_layer(layer)
    return layer, layer.shape_data["path_nodes"]


def split_middle(nodes: list[dict[str, object]]) -> None:
    first, second = nodes[0], nodes[1]
    control = [first["anchor"], first["out"], second["in"], second["anchor"]]
    left, right = split_cubic_bezier(control, 0.5)
    first["out"] = list(left[1])
    second["in"] = list(right[2])
    nodes.insert(1, {"anchor": list(left[3]), "in": list(left[2]), "out": list(right[1]), "linked": True})


def path_frame(tool: str, amount: float, base: Image.Image) -> DemoFrame:
    document = document_from(base)
    layer, nodes = base_path(document)
    active = 0
    if tool in {"delete_anchor", "convert_anchor"}:
        split_middle(nodes)
        active = 1
    if tool == "direct_select":
        active = 1
        node = nodes[active]
        dy = -round(45 * amount)
        for key in ("anchor", "in", "out"):
            node[key][1] += dy
    elif tool == "add_anchor" and amount >= 0.45:
        split_middle(nodes)
        active = 1
    elif tool == "delete_anchor" and amount >= 0.45:
        nodes.pop(1)
        active = 0
    elif tool == "convert_anchor":
        node = nodes[1]
        anchor = node["anchor"]
        distance = round(34 * amount)
        node["in"] = [anchor[0] - distance, anchor[1]]
        node["out"] = [anchor[0] + distance, anchor[1]]
        node["linked"] = amount > 0
    render_shape_layer(layer)
    cursor_point = nodes[min(active, len(nodes) - 1)]["anchor"]
    shown_nodes = deepcopy(nodes)
    if tool == "path_select" and amount < 0.4:
        shown_nodes = []
        cursor_point = [round(266 - 231 * amount), round(143 - 13 * amount)]
    return DemoFrame(
        output(document),
        (round(cursor_point[0]), round(cursor_point[1])),
        nodes=shown_nodes,
        active_node=min(active, len(nodes) - 1),
    )


def progressive_polygon(points: list[tuple[int, int]], amount: float) -> list[tuple[int, int]]:
    count = max(2, min(len(points), round(2 + (len(points) - 2) * amount)))
    return points[:count]


def selection_frame(tool: str, amount: float, base: Image.Image) -> DemoFrame:
    document = document_from(base)
    guide: list[tuple[int, int]] = []
    closed = False
    cursor_point = (72, 42)
    if tool == "select":
        end = (round(72 + 155 * amount), round(42 + 91 * amount))
        document.set_rect_selection((72, 42, *end))
        guide, closed, cursor_point = [(72, 42), (end[0], 42), end, (72, end[1])], True, end
    elif tool == "ellipse_select":
        end = (round(72 + 155 * amount), round(42 + 91 * amount))
        document.set_ellipse_selection((72, 42, *end))
        cursor_point = end
    elif tool in {"lasso", "polygon_lasso", "magnetic_lasso"}:
        intended = [(31, 122), (43, 78), (76, 64), (110, 83), (101, 128), (58, 140)]
        guide = progressive_polygon(intended, amount)
        if tool == "magnetic_lasso":
            edges = document.magnetic_edge_map()
            guide = [document.snap_point_to_edge(point, edges, 13) for point in guide]
        cursor_point = guide[-1]
        if amount >= 0.95:
            document.set_polygon_selection(guide)
            closed = True
    elif tool == "quick_selection":
        all_points = points_between((39, 108), (103, 93), amount, 9)
        document.quick_selection_brush(document.layer, all_points, 12, 28, smooth=1, edge_radius=2, edge_strength=0.5)
        guide, cursor_point = all_points, all_points[-1]
    elif tool == "magic_wand":
        cursor_point = (43, 108)
        if amount > 0.2:
            document.magic_wand_selection(document.layer, *cursor_point, round(12 + amount * 34), antialias=True)
    else:
        cursor_point = (43, 108)
        if amount > 0.2:
            document.color_range_selection(document.layer, *cursor_point, round(10 + amount * 42), antialias=True)
    return DemoFrame(output(document), cursor_point, document.selection_mask, guide, closed)


def move_hand_crop_frame(tool: str, amount: float, base: Image.Image) -> DemoFrame:
    if tool == "hand":
        canvas = Image.new("RGB", SIZE, "#292b30")
        offset = (round(-36 + 36 * amount), round(-18 + 18 * amount))
        canvas.paste(base, offset)
        return DemoFrame(canvas, (144, 81), guide=[(108, 81), (180, 81)])
    document = document_from(base)
    if tool == "move":
        layer = document.add_shape_layer("star", (35, 50, 98, 116), PURPLE, (255, 255, 255, 255), 2, sides=5, inner_ratio=0.45)
        layer.x = round(130 * amount)
        layer.y = -round(15 * amount)
        return DemoFrame(output(document), (round(67 + 130 * amount), round(83 - 15 * amount)))
    margin_x, margin_y = round(40 * amount), round(22 * amount)
    box = (margin_x, margin_y, SIZE[0] - margin_x, SIZE[1] - margin_y)
    if amount >= 0.9:
        document.crop(box)
        cropped = output(document).resize(SIZE, Image.Resampling.LANCZOS)
        return DemoFrame(cropped, (box[2], box[3]))
    guide = [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]
    return DemoFrame(output(document), (box[2], box[3]), guide=guide, guide_closed=True)


def actual_frame(tool: str, amount: float, still: Image.Image, landscape: Image.Image) -> DemoFrame:
    amount = float(np.clip(amount, 0.0, 1.0))
    if tool in {"hand", "move", "crop"}:
        return move_hand_crop_frame(tool, amount, landscape)
    if tool in {"brush", "eraser"}:
        document = document_from(still)
        point = paint_stroke(document, amount, erase=tool == "eraser")
        return DemoFrame(output(document, checker=tool == "eraser"), point)
    if tool in {"blur_tool", "sharpen_tool", "dodge", "burn"}:
        return retouch_frame(tool, amount, still)
    if tool in {"clone", "healing"}:
        return clone_frame(tool, amount, still)
    if tool in {"spot_healing", "patch"}:
        return healing_frame(tool, amount, still)
    if tool in {"fill", "gradient", "text"}:
        return fill_gradient_text_frame(tool, amount, landscape)
    if tool == "eyedropper":
        point = (round(42 + 150 * amount), round(109 - 24 * amount))
        pixels = np.asarray(still.convert("RGBA"), dtype=np.uint8)
        return DemoFrame(still.copy(), point, swatch=tuple(int(value) for value in pixels[point[1], point[0]]))
    if tool.endswith("_shape"):
        return shape_frame(tool, amount, landscape)
    if tool in {"path_select", "direct_select", "add_anchor", "delete_anchor", "convert_anchor"}:
        return path_frame(tool, amount, landscape)
    return selection_frame(tool, amount, still)


def selection_outline(mask: np.ndarray) -> Iterable[list[tuple[int, int]]]:
    contours, _hierarchy = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if len(contour) >= 2:
            yield [(int(point[0][0]), int(point[0][1])) for point in contour]
