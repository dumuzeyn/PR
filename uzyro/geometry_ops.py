from __future__ import annotations

from .core_shared import *
from .layer import Layer


def normalized_box(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

def shape_box_from_drag(
    start: tuple[int, int],
    end: tuple[int, int],
    shape: str,
    *,
    keep_proportions: bool = False,
    from_center: bool = False,
) -> tuple[int, int, int, int]:
    sx, sy = int(start[0]), int(start[1])
    ex, ey = int(end[0]), int(end[1])
    dx, dy = ex - sx, ey - sy
    if shape == "line" and keep_proportions and (dx or dy):
        length = math.hypot(dx, dy)
        angle = round(math.atan2(dy, dx) / (math.pi / 4.0)) * (math.pi / 4.0)
        dx, dy = round(math.cos(angle) * length), round(math.sin(angle) * length)
    elif shape != "line" and keep_proportions:
        size = max(abs(dx), abs(dy))
        dx = size if dx >= 0 else -size
        dy = size if dy >= 0 else -size
    if from_center:
        return sx - dx, sy - dy, sx + dx, sy + dy
    return sx, sy, sx + dx, sy + dy

def shape_geometry_from_drag(
    tool: str,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    shift: bool = False,
    alt: bool = False,
    sides: int = 5,
    inner_ratio: float = 0.5,
    custom_points: Any = None,
) -> dict[str, Any]:
    shape_by_tool = {
        "rect_shape": "rectangle",
        "ellipse_shape": "ellipse",
        "line_shape": "line",
        "bezier_shape": "bezier",
        "polygon_shape": "polygon",
        "star_shape": "star",
        "custom_shape": "custom",
    }
    shape = shape_by_tool.get(tool, tool.removesuffix("_shape"))
    raw_box = shape_box_from_drag(start, end, shape, keep_proportions=shift, from_center=alt)
    box = normalized_box(raw_box)
    geometry: dict[str, Any] = {"shape": shape, "box": box}
    if shape == "line":
        geometry["line"] = raw_box
    elif shape == "bezier":
        geometry["points"] = bezier_curve_points(None, box, 64)
    elif shape == "polygon":
        geometry["points"] = regular_polygon_points(box, max(3, int(sides)))
    elif shape == "star":
        geometry["points"] = star_points(box, max(3, int(sides)), inner_ratio)
    elif shape == "custom":
        geometry["points"] = custom_shape_points(custom_points, box)
    return geometry

def shape_drag_is_meaningful(geometry: dict[str, Any], minimum: int = 3) -> bool:
    x1, y1, x2, y2 = geometry["box"]
    if geometry.get("shape") in {"line", "bezier"}:
        return math.hypot(x2 - x1, y2 - y1) >= minimum
    return x2 - x1 >= minimum and y2 - y1 >= minimum

def layer_contains_point(layer: Layer, point: tuple[int, int], tolerance: int = 5) -> bool:
    """Hit-test the rendered object alpha, including thin lines and Bezier curves."""
    if not layer.visible or layer.kind == "adjustment" or layer.pixels.size == 0:
        return False
    local_x = int(point[0]) - int(layer.x)
    local_y = int(point[1]) - int(layer.y)
    height, width = layer.pixels.shape[:2]
    radius = max(0, int(tolerance))
    x1, x2 = max(0, local_x - radius), min(width, local_x + radius + 1)
    y1, y2 = max(0, local_y - radius), min(height, local_y + radius + 1)
    if x1 >= x2 or y1 >= y2:
        return False
    alpha = layer.pixels[y1:y2, x1:x2, 3]
    if layer.kind == "shape" and layer.shape_data is not None:
        kind = str(layer.shape_data.get("shape", "rectangle")).lower()
        if kind not in {"line", "bezier"} and 0 <= local_x < width and 0 <= local_y < height:
            return bool(layer.pixels[local_y, local_x, 3] > 8)
    return bool(np.any(alpha > 8))

def topmost_layer_at(
    document: Document,
    point: tuple[int, int],
    *,
    kinds: tuple[str, ...] = ("shape", "text"),
    tolerance: int = 5,
) -> int | None:
    for index in range(len(document.layers) - 1, -1, -1):
        layer = document.layers[index]
        if layer.kind in kinds and layer_contains_point(layer, point, tolerance):
            return index
    return None

def resize_box_from_handle(
    box: tuple[int, int, int, int],
    handle: str,
    point: tuple[int, int],
    *,
    keep_proportions: bool = False,
    from_center: bool = False,
    minimum: int = 2,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = normalized_box(box)
    px, py = int(point[0]), int(point[1])
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    original_ratio = max(1e-6, (x2 - x1) / max(1, y2 - y1))
    if "w" in handle:
        x1 = px
        if from_center:
            x2 = round(cx + (cx - px))
    if "e" in handle:
        x2 = px
        if from_center:
            x1 = round(cx - (px - cx))
    if "n" in handle:
        y1 = py
        if from_center:
            y2 = round(cy + (cy - py))
    if "s" in handle:
        y2 = py
        if from_center:
            y1 = round(cy - (py - cy))
    x1, y1, x2, y2 = normalized_box((x1, y1, x2, y2))
    if keep_proportions and len(handle) == 2:
        width, height = max(minimum, x2 - x1), max(minimum, y2 - y1)
        if width / height > original_ratio:
            height = max(minimum, round(width / original_ratio))
        else:
            width = max(minimum, round(height * original_ratio))
        if "w" in handle:
            x1 = x2 - width
        else:
            x2 = x1 + width
        if "n" in handle:
            y1 = y2 - height
        else:
            y2 = y1 + height
        if from_center:
            x1, x2 = round(cx - width / 2), round(cx + width / 2)
            y1, y2 = round(cy - height / 2), round(cy + height / 2)
    if x2 - x1 < minimum:
        x2 = x1 + minimum
    if y2 - y1 < minimum:
        y2 = y1 + minimum
    return int(x1), int(y1), int(x2), int(y2)

def selection_contour_points(mask: np.ndarray | None, threshold: int = 128) -> list[np.ndarray]:
    if mask is None or mask.ndim != 2 or not np.any(mask >= threshold):
        return []
    binary = (mask >= threshold).astype(np.uint8)
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return [contour[:, 0, :].copy() for contour in contours if len(contour) >= 2]

def union_rect(a: tuple[int, int, int, int] | None, b: tuple[int, int, int, int] | None) -> tuple[int, int, int, int] | None:
    if a is None:
        return b
    if b is None:
        return a
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])

def checker_background(width: int, height: int, tile: int = 16) -> np.ndarray:
    key = (width, height, tile)
    cached = _checker_cache.get(key)
    if cached is not None:
        return cached
    tile_arr = np.zeros((tile * 2, tile * 2, 4), dtype=np.uint8)
    tile_arr[:, :, 3] = 255
    tile_arr[:tile, :tile, :3] = 224
    tile_arr[tile:, tile:, :3] = 224
    tile_arr[:tile, tile:, :3] = 192
    tile_arr[tile:, :tile, :3] = 192
    reps_y = height // tile_arr.shape[0] + 1
    reps_x = width // tile_arr.shape[1] + 1
    bg = np.tile(tile_arr, (reps_y, reps_x, 1))[:height, :width].copy()
    if len(_checker_cache) > 8:
        _checker_cache.clear()
    _checker_cache[key] = bg
    return bg


def checker_region(rect: tuple[int, int, int, int], tile: int = 16) -> np.ndarray:
    x1, y1, x2, y2 = (int(value) for value in rect)
    width, height = max(0, x2 - x1), max(0, y2 - y1)
    yy, xx = np.indices((height, width), dtype=np.int32)
    light = (((xx + x1) // tile + (yy + y1) // tile) & 1) == 0
    result = np.empty((height, width, 4), dtype=np.uint8)
    result[:, :, :3] = np.where(light[:, :, None], 224, 192)
    result[:, :, 3] = 255
    return result

def bezier_curve_points(raw_points: Any, box: tuple[int, int, int, int], steps: int = 64) -> list[tuple[float, float]]:
    if isinstance(raw_points, list) and len(raw_points) == 4:
        try:
            p0, p1, p2, p3 = [tuple(float(v) for v in point[:2]) for point in raw_points]
        except (TypeError, ValueError):
            p0 = p1 = p2 = p3 = None
    else:
        p0 = p1 = p2 = p3 = None
    if p0 is None:
        x1, y1, x2, y2 = normalized_box(box)
        p0, p1, p2, p3 = (x1, y2), (x1, y1), (x2, y1), (x2, y2)
    coords: list[tuple[float, float]] = []
    for i in range(max(2, steps) + 1):
        t = i / max(2, steps)
        mt = 1.0 - t
        x = mt**3 * p0[0] + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t**3 * p3[1]
        coords.append((x, y))
    return coords

def custom_shape_points(raw_points: Any, box: tuple[int, int, int, int]) -> list[tuple[float, float]]:
    if not isinstance(raw_points, list) or len(raw_points) < 3:
        raw_points = [[0.5, 0.0], [1.0, 0.5], [0.5, 1.0], [0.0, 0.5]]
    x1, y1, x2, y2 = normalized_box(box)
    width, height = x2 - x1, y2 - y1
    points: list[tuple[float, float]] = []
    for point in raw_points:
        try:
            px, py = float(point[0]), float(point[1])
        except (TypeError, ValueError, IndexError):
            continue
        points.append((x1 + np.clip(px, 0.0, 1.0) * width, y1 + np.clip(py, 0.0, 1.0) * height))
    return points if len(points) >= 3 else [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

def regular_polygon_points(box: tuple[int, int, int, int], sides: int) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = normalized_box(box)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    rx, ry = abs(x2 - x1) / 2.0, abs(y2 - y1) / 2.0
    start = -math.pi / 2.0
    return [(cx + math.cos(start + i * 2.0 * math.pi / sides) * rx, cy + math.sin(start + i * 2.0 * math.pi / sides) * ry) for i in range(sides)]

def star_points(box: tuple[int, int, int, int], points: int, inner_ratio: float) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = normalized_box(box)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    rx, ry = abs(x2 - x1) / 2.0, abs(y2 - y1) / 2.0
    inner_ratio = float(np.clip(inner_ratio, 0.05, 0.95))
    start = -math.pi / 2.0
    coords: list[tuple[float, float]] = []
    for i in range(points * 2):
        scale = 1.0 if i % 2 == 0 else inner_ratio
        angle = start + i * math.pi / points
        coords.append((cx + math.cos(angle) * rx * scale, cy + math.sin(angle) * ry * scale))
    return coords

__all__ = [name for name in globals() if not name.startswith("__")]
