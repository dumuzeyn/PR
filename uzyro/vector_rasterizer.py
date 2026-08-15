from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .vector_geometry import rotate_points


Point = tuple[float, float]


def _dash_paths(points: Sequence[Point], pattern: Sequence[float] | None, offset: float = 0.0) -> list[list[Point]]:
    clean = [max(0.1, float(value)) for value in (pattern or []) if float(value) > 0]
    if not clean:
        return [list(points)] if len(points) >= 2 else []
    if len(clean) % 2:
        clean *= 2
    period = sum(clean)
    phase = float(offset) % period
    pattern_index = 0
    while phase >= clean[pattern_index]:
        phase -= clean[pattern_index]
        pattern_index = (pattern_index + 1) % len(clean)
    remaining = clean[pattern_index] - phase
    drawing = pattern_index % 2 == 0
    paths: list[list[Point]] = []
    active: list[Point] = []
    for first, second in zip(points, points[1:]):
        dx, dy = second[0] - first[0], second[1] - first[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            continue
        consumed = 0.0
        while consumed < length - 1e-9:
            step = min(remaining, length - consumed)
            start = (first[0] + dx * consumed / length, first[1] + dy * consumed / length)
            consumed += step
            end = (first[0] + dx * consumed / length, first[1] + dy * consumed / length)
            if drawing:
                if not active:
                    active = [start]
                active.append(end)
            if remaining <= step + 1e-9:
                if drawing and len(active) >= 2:
                    paths.append(active)
                    active = []
                pattern_index = (pattern_index + 1) % len(clean)
                drawing = pattern_index % 2 == 0
                remaining = clean[pattern_index]
            else:
                remaining -= step
    if drawing and len(active) >= 2:
        paths.append(active)
    return paths


def _line_intersection(first: tuple[Point, Point], second: tuple[Point, Point]) -> Point | None:
    (x1, y1), (x2, y2) = first
    (x3, y3), (x4, y4) = second
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) <= 1e-9:
        return None
    cross1, cross2 = x1 * y2 - y1 * x2, x3 * y4 - y3 * x4
    return ((cross1 * (x3 - x4) - (x1 - x2) * cross2) / denominator,
            (cross1 * (y3 - y4) - (y1 - y2) * cross2) / denominator)


def _stroke_path(draw: ImageDraw.ImageDraw, points: list[Point], width: float, cap: str, join: str, miter_limit: float) -> None:
    if len(points) < 2:
        return
    radius = width * 0.5
    segments: list[tuple[Point, Point, Point]] = []
    for first, second in zip(points, points[1:]):
        dx, dy = second[0] - first[0], second[1] - first[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            continue
        normal = (-dy / length * radius, dx / length * radius)
        segments.append((first, second, normal))
        draw.polygon([
            (first[0] + normal[0], first[1] + normal[1]),
            (second[0] + normal[0], second[1] + normal[1]),
            (second[0] - normal[0], second[1] - normal[1]),
            (first[0] - normal[0], first[1] - normal[1]),
        ], fill=255)
    if not segments:
        return
    for index in range(len(segments) - 1):
        previous, following = segments[index], segments[index + 1]
        anchor = previous[1]
        if join == "round":
            draw.ellipse((anchor[0] - radius, anchor[1] - radius, anchor[0] + radius, anchor[1] + radius), fill=255)
            continue
        for sign in (-1.0, 1.0):
            previous_outer = (anchor[0] + previous[2][0] * sign, anchor[1] + previous[2][1] * sign)
            following_outer = (anchor[0] + following[2][0] * sign, anchor[1] + following[2][1] * sign)
            polygon = [anchor, previous_outer, following_outer]
            if join == "miter":
                intersection = _line_intersection(
                    ((previous[0][0] + previous[2][0] * sign, previous[0][1] + previous[2][1] * sign), previous_outer),
                    (following_outer, (following[1][0] + following[2][0] * sign, following[1][1] + following[2][1] * sign)),
                )
                if intersection is not None and math.hypot(intersection[0] - anchor[0], intersection[1] - anchor[1]) <= radius * max(1.0, miter_limit):
                    polygon = [anchor, previous_outer, intersection, following_outer]
            draw.polygon(polygon, fill=255)
    start, end = segments[0][0], segments[-1][1]
    if cap == "round":
        for x, y in (start, end):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    elif cap == "square":
        for segment, endpoint, direction in ((segments[0], start, -1.0), (segments[-1], end, 1.0)):
            dx, dy = segment[1][0] - segment[0][0], segment[1][1] - segment[0][1]
            length = max(1e-9, math.hypot(dx, dy))
            extension = (dx / length * radius * direction, dy / length * radius * direction)
            normal = segment[2]
            draw.polygon([
                (endpoint[0] + normal[0], endpoint[1] + normal[1]),
                (endpoint[0] - normal[0], endpoint[1] - normal[1]),
                (endpoint[0] + extension[0] - normal[0], endpoint[1] + extension[1] - normal[1]),
                (endpoint[0] + extension[0] + normal[0], endpoint[1] + extension[1] + normal[1]),
            ], fill=255)


def stroke_polyline_mask(
    canvas_shape: tuple[int, int], points: Sequence[Sequence[float]], width: float,
    cap: str = "round", join: str = "round", miter_limit: float = 4.0,
    dash_pattern: Sequence[float] | None = None, dash_offset: float = 0.0, supersample: int = 4,
) -> np.ndarray:
    height, canvas_width = canvas_shape
    result = np.zeros((height, canvas_width), dtype=np.uint8)
    if len(points) < 2 or width <= 0:
        return result
    scale = max(2, int(supersample))
    margin = max(3.0, float(width) * max(1.0, float(miter_limit)) * 0.5 + 2.0)
    xs, ys = [float(value[0]) for value in points], [float(value[1]) for value in points]
    x1, y1 = max(0, math.floor(min(xs) - margin)), max(0, math.floor(min(ys) - margin))
    x2, y2 = min(canvas_width, math.ceil(max(xs) + margin + 1)), min(height, math.ceil(max(ys) + margin + 1))
    if x1 >= x2 or y1 >= y2:
        return result
    local = [((float(value[0]) - x1) * scale, (float(value[1]) - y1) * scale) for value in points]
    image = Image.new("L", ((x2 - x1) * scale, (y2 - y1) * scale), 0)
    draw = ImageDraw.Draw(image)
    scaled_pattern = None if not dash_pattern else [float(value) * scale for value in dash_pattern]
    for path in _dash_paths(local, scaled_pattern, float(dash_offset) * scale):
        _stroke_path(draw, path, float(width) * scale, str(cap).lower(), str(join).lower(), float(miter_limit))
    reduced = cv2.resize(np.asarray(image, dtype=np.uint8), (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)
    result[y1:y2, x1:x2] = reduced
    return result


def closed_shape_mask(
    canvas_shape: tuple[int, int], kind: str, box: Sequence[float], points: Sequence[Sequence[float]] | None = None,
    corner_radius: float = 0.0, rotation: float = 0.0, supersample: int = 4,
) -> np.ndarray:
    height, width = canvas_shape
    x1, y1, x2, y2 = [float(value) for value in box]
    center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
    original_corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    corners = original_corners + rotate_points(original_corners, center, rotation)
    margin = 2
    left, top = max(0, math.floor(min(value[0] for value in corners) - margin)), max(0, math.floor(min(value[1] for value in corners) - margin))
    right, bottom = min(width, math.ceil(max(value[0] for value in corners) + margin + 1)), min(height, math.ceil(max(value[1] for value in corners) + margin + 1))
    result = np.zeros((height, width), dtype=np.uint8)
    if left >= right or top >= bottom:
        return result
    region_area = (right - left) * (bottom - top)
    requested_scale = max(2, int(supersample))
    scale = min(requested_scale, 2 if region_area > 2_000_000 else 3 if region_area > 750_000 else requested_scale)
    image = Image.new("L", ((right - left) * scale, (bottom - top) * scale), 0)
    draw = ImageDraw.Draw(image)
    local_box = tuple((value - (left if index % 2 == 0 else top)) * scale for index, value in enumerate((x1, y1, x2, y2)))
    if kind == "ellipse":
        draw.ellipse(local_box, fill=255)
    elif points:
        draw.polygon([((float(value[0]) - left) * scale, (float(value[1]) - top) * scale) for value in points], fill=255)
    else:
        draw.rounded_rectangle(local_box, radius=max(0.0, float(corner_radius)) * scale, fill=255)
    if abs(float(rotation)) > 1e-9:
        array = np.asarray(image, dtype=np.uint8)
        local_center = ((center[0] - left) * scale, (center[1] - top) * scale)
        matrix = cv2.getRotationMatrix2D(local_center, -float(rotation), 1.0)
        array = cv2.warpAffine(array, matrix, (array.shape[1], array.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        array = np.asarray(image, dtype=np.uint8)
    result[top:bottom, left:right] = cv2.resize(array, (right - left, bottom - top), interpolation=cv2.INTER_AREA)
    return result


def aligned_stroke_mask(fill_mask: np.ndarray, width: float, alignment: str = "center") -> np.ndarray:
    if width <= 0 or not np.any(fill_mask):
        return np.zeros_like(fill_mask)
    result = np.zeros_like(fill_mask)
    x, y, source_width, source_height = cv2.boundingRect(fill_mask)
    margin = max(2, math.ceil(float(width)) + 2)
    x1, y1 = max(0, x - margin), max(0, y - margin)
    x2, y2 = min(fill_mask.shape[1], x + source_width + margin), min(fill_mask.shape[0], y + source_height + margin)
    binary = (fill_mask[y1:y2, x1:x2] >= 128).astype(np.uint8)
    inside = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
    signed = inside - outside
    mode = str(alignment).lower()
    if mode in {"inside", "внутри"}:
        low, high = 0.0, float(width)
    elif mode in {"outside", "снаружи"}:
        low, high = -float(width), 0.0
    else:
        low, high = -float(width) * 0.5, float(width) * 0.5
    coverage = np.minimum(signed - low + 0.5, high - signed + 0.5)
    result[y1:y2, x1:x2] = np.rint(np.clip(coverage, 0.0, 1.0) * 255.0).astype(np.uint8)
    return result


__all__ = ["aligned_stroke_mask", "closed_shape_mask", "stroke_polyline_mask"]
