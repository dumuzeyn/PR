from __future__ import annotations

from .core_shared import *
from .layer import Layer
from .geometry_ops import *
from .selection_ops import *
from .render_ops import *
from .text_ops import *
from .vector_geometry import adaptive_bezier_path, rotate_points
from .vector_rasterizer import aligned_stroke_mask, closed_shape_mask, stroke_polyline_mask


def render_shape_layer(layer: Layer) -> None:
    if layer.shape_data is None:
        return
    if layer.transform_data is not None:
        data = layer.transform_data
        layer.transform_data = None
        layer.x = int(data.get("render_base_x", data.get("base_x", layer.x)))
        layer.y = int(data.get("render_base_y", data.get("base_y", layer.y)))
        layer.pixels = blank_rgba(int(data.get("render_width", layer.pixels.shape[1])), int(data.get("render_height", layer.pixels.shape[0])), (0, 0, 0, 0))
        render_shape_layer(layer)
        crop = [int(value) for value in data.get("source_crop", [0, 0, layer.pixels.shape[1], layer.pixels.shape[0]])]
        source = layer.pixels[crop[1]:crop[3], crop[0]:crop[2]].copy()
        layer.transform_data = data
        layer.transform_source = source.copy()
        apply_saved_layer_transform(layer, source, layer.transform_mask_source)
        return
    layer.pixels[:] = 0
    data = layer.shape_data
    if str(data.get("shape", "rectangle")).lower() == "boolean":
        render_boolean_shape_layer(layer)
        apply_shape_postprocess(layer)
        return
    gradient = data.get("gradient")
    if isinstance(gradient, dict):
        height, width = layer.pixels.shape[:2]
        region = shape_render_region(data, width, height)
        if region is None:
            return
        x1, y1, x2, y2 = region
        local_data = translated_shape_data(data, -x1, -y1)
        mask = shape_data_to_mask(local_data, (y2 - y1, x2 - x1), apply_rotation=False)
        start, end = shape_gradient_vector(data, gradient)
        pixels = GradientEngine.render(
            x2 - x1,
            y2 - y1,
            (start[0] - x1, start[1] - y1),
            (end[0] - x1, end[1] - y1),
            gradient.get("stops"),
            str(gradient.get("type", "linear")),
            opacity_stops=gradient.get("opacity_stops"),
            reverse=bool(gradient.get("reverse", False)),
            dither=bool(gradient.get("dither", False)),
            transparency=bool(gradient.get("transparency", True)),
        )
        opacity = float(np.clip(gradient.get("opacity", 1.0), 0.0, 1.0))
        pixels[:, :, 3] = np.clip(pixels[:, :, 3].astype(np.float32) * (mask.astype(np.float32) / 255.0) * opacity, 0, 255).astype(np.uint8)
        stroke = data.get("stroke")
        stroke_width = max(0, int(data.get("stroke_width", 0)))
        if stroke is not None and stroke_width > 0:
            stroke_color = np.array(stroke, dtype=np.uint8)
            edge = shape_stroke_mask(mask, stroke_width, str(data.get("stroke_alignment", "center")))
            _paint_solid(pixels, edge, stroke_color)
        layer.pixels[y1:y2, x1:x2] = pixels
        apply_shape_postprocess(layer)
        return
    texture = data.get("texture")
    if isinstance(texture, dict):
        height, width = layer.pixels.shape[:2]
        region = shape_render_region(data, width, height)
        if region is None:
            return
        x1, y1, x2, y2 = region
        local_data = translated_shape_data(data, -x1, -y1)
        mask = shape_data_to_mask(local_data, (y2 - y1, x2 - x1), apply_rotation=False)
        yy, xx = np.mgrid[y1:y2, x1:x2]
        size = max(2, int(texture.get("size", 18)))
        kind = str(texture.get("type", "checker"))
        if kind == "dots":
            cx = np.mod(xx, size) - size / 2.0
            cy = np.mod(yy, size) - size / 2.0
            selector = (cx * cx + cy * cy) <= (size * 0.24) ** 2
        elif kind == "stripes":
            selector = np.mod((xx + yy) // size, 2) == 0
        else:
            selector = np.mod(xx // size + yy // size, 2) == 0
        first = np.array(texture.get("color_a", data.get("fill", [255, 255, 255, 255])), dtype=np.uint8)
        second = np.array(texture.get("color_b", data.get("stroke", [0, 0, 0, 255]) or [0, 0, 0, 255]), dtype=np.uint8)
        pixels = np.where(selector[:, :, None], first, second).astype(np.uint8)
        pixels[:, :, 3] = np.clip(pixels[:, :, 3].astype(np.float32) * (mask.astype(np.float32) / 255.0), 0, 255).astype(np.uint8)
        stroke = data.get("stroke")
        stroke_width = max(0, int(data.get("stroke_width", 0)))
        if stroke is not None and stroke_width > 0:
            stroke_color = np.array(stroke, dtype=np.uint8)
            edge = shape_stroke_mask(mask, stroke_width, str(data.get("stroke_alignment", "center")))
            _paint_solid(pixels, edge, stroke_color)
        layer.pixels[y1:y2, x1:x2] = pixels
        apply_shape_postprocess(layer)
        return
    box = tuple(float(v) for v in data.get("box", [0, 0, 1, 1]))
    fill = tuple(int(v) for v in data.get("fill", [255, 255, 255, 255]))
    stroke = data.get("stroke")
    outline = None if stroke is None else tuple(int(v) for v in stroke)
    stroke_width = max(0, int(data.get("stroke_width", 0)))
    shape = str(data.get("shape", "rectangle")).lower()
    angle = float(data.get("rotation", 0.0))
    if shape in {"line", "bezier"}:
        points = [box[:2], box[2:4]] if shape == "line" else editable_bezier_path_points(data, tolerance=0.2)
        center = ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)
        mask = stroke_polyline_mask(
            layer.pixels.shape[:2], rotate_points(points, center, angle),
            max(1, stroke_width or (2 if shape == "bezier" else 1)),
            str(data.get("stroke_cap", "round")), str(data.get("stroke_join", "round")),
            float(data.get("miter_limit", 4.0)), data.get("dash_pattern"), float(data.get("dash_offset", 0.0)),
        )
        _paint_solid(layer.pixels, mask, outline or fill)
    else:
        points = None
        if shape == "polygon":
            points = regular_polygon_points(box, max(3, int(data.get("sides", 5))))
        elif shape == "star":
            points = star_points(box, max(3, int(data.get("sides", 5))), float(data.get("inner_ratio", 0.5)))
        elif shape == "custom":
            points = custom_shape_points(data.get("custom_points"), box)
        mask = closed_shape_mask(layer.pixels.shape[:2], shape, box, points, float(data.get("corner_radius", 0)), angle)
        _paint_solid(layer.pixels, mask, fill)
        if outline is not None and stroke_width > 0:
            _paint_solid(layer.pixels, shape_stroke_mask(mask, stroke_width, str(data.get("stroke_alignment", "center"))), outline)
    apply_shape_postprocess(layer, rotation_applied=True)


def _paint_solid(pixels: np.ndarray, mask: np.ndarray, color: np.ndarray | tuple[int, ...]) -> None:
    bounds = cv2.boundingRect(mask)
    if bounds[2] <= 0 or bounds[3] <= 0:
        return
    x, y, width, height = bounds
    target = pixels[y:y + height, x:x + width]
    local_mask = mask[y:y + height, x:x + width]
    rgba = np.asarray(color, dtype=np.float32)
    source_alpha = local_mask.astype(np.float32) / 255.0 * (rgba[3] / 255.0)
    if not np.any(source_alpha > 0):
        return
    destination_alpha = target[:, :, 3].astype(np.float32) / 255.0
    output_alpha = source_alpha + destination_alpha * (1.0 - source_alpha)
    numerator = rgba[:3][None, None, :] * source_alpha[:, :, None]
    numerator += target[:, :, :3].astype(np.float32) * destination_alpha[:, :, None] * (1.0 - source_alpha[:, :, None])
    target[:, :, :3] = np.where(
        output_alpha[:, :, None] > 1e-8,
        numerator / np.maximum(output_alpha[:, :, None], 1e-8),
        0,
    ).astype(np.uint8)
    target[:, :, 3] = np.rint(output_alpha * 255.0).astype(np.uint8)

def shape_stroke_edge(mask: np.ndarray, width: int, alignment: str = "center") -> np.ndarray:
    return shape_stroke_mask(mask, width, alignment) > 0


def shape_stroke_mask(mask: np.ndarray, width: int, alignment: str = "center") -> np.ndarray:
    return aligned_stroke_mask(mask, max(1, int(width)), alignment)

def shape_gradient_vector(data: dict[str, Any], gradient: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    if "angle" not in gradient and "scale" not in gradient:
        start = tuple(float(value) for value in gradient.get("start", data.get("box", [0, 0])[:2]))
        end = tuple(float(value) for value in gradient.get("end", data.get("box", [0, 0, 1, 1])[2:4]))
        return start, end
    box = normalized_box(tuple(int(round(value)) for value in data.get("box", [0, 0, 1, 1])))
    center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
    length = max(1.0, max(box[2] - box[0], box[3] - box[1]) * float(np.clip(gradient.get("scale", 1.0), 0.01, 10.0)))
    angle = math.radians(float(gradient.get("angle", 0.0)))
    offset = (math.cos(angle) * length / 2.0, math.sin(angle) * length / 2.0)
    return (center[0] - offset[0], center[1] - offset[1]), (center[0] + offset[0], center[1] + offset[1])

def editable_bezier_nodes(data: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = data.get("path_nodes")
    if isinstance(nodes, list) and len(nodes) >= 2:
        return json.loads(json.dumps(nodes))
    box = tuple(int(value) for value in data.get("box", [0, 0, 1, 1]))
    points = data.get("control_points")
    if not isinstance(points, list) or len(points) != 4:
        x1, y1, x2, y2 = box
        points = [[x1, y2], [x1, y1], [x2, y1], [x2, y2]]
    return [
        {"anchor": list(points[0]), "in": list(points[0]), "out": list(points[1]), "linked": True},
        {"anchor": list(points[3]), "in": list(points[2]), "out": list(points[3]), "linked": True},
    ]

def editable_bezier_path_points(
    data: dict[str, Any], box: tuple[int, int, int, int] | None = None,
    samples: int | None = None, tolerance: float | None = None, zoom: float = 1.0,
) -> list[tuple[float, float]]:
    nodes = editable_bezier_nodes(data)
    screen_tolerance = max(0.05, float(tolerance if tolerance is not None else 0.35) / max(0.01, float(zoom)))
    return adaptive_bezier_path(nodes, screen_tolerance, bool(data.get("path_closed", False)))


def apply_shape_postprocess(layer: Layer, rotation_applied: bool = False) -> None:
    data = layer.shape_data or {}
    opacity = float(np.clip(data.get("opacity", 1.0), 0.0, 1.0))
    if opacity < 1.0:
        layer.pixels[:, :, 3] = np.clip(layer.pixels[:, :, 3].astype(np.float32) * opacity, 0, 255).astype(np.uint8)
    angle = 0.0 if rotation_applied else float(data.get("rotation", 0.0))
    if abs(angle) <= 1e-6 or not np.any(layer.pixels[:, :, 3]):
        return
    box = normalized_box(tuple(int(round(value)) for value in data.get("box", [0, 0, 1, 1])))
    center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    layer.pixels = cv2.warpAffine(layer.pixels, matrix, (layer.pixels.shape[1], layer.pixels.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

def shape_render_region(data: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int] | None:
    box = shape_data_bounds(data) or (0, 0, 1, 1)
    margin = max(1, int(data.get("stroke_width", 0)) * 2 + 1)
    x1 = max(0, box[0] - margin)
    y1 = max(0, box[1] - margin)
    x2 = min(width, box[2] + margin + 1)
    y2 = min(height, box[3] + margin + 1)
    return None if x1 >= x2 or y1 >= y2 else (x1, y1, x2, y2)

def translated_shape_data(data: dict[str, Any], dx: int, dy: int) -> dict[str, Any]:
    translated = json.loads(json.dumps(data))
    if str(data.get("shape", "rectangle")).lower() == "boolean":
        translated["children"] = [
            translated_shape_data(child, dx, dy) if isinstance(child, dict) else child
            for child in data.get("children", [])
        ]
    else:
        box = list(data.get("box", [0, 0, 1, 1]))
        translated["box"] = [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]
        points = data.get("control_points")
        if isinstance(points, list):
            translated["control_points"] = [[float(point[0]) + dx, float(point[1]) + dy] for point in points]
        path_nodes = data.get("path_nodes")
        if isinstance(path_nodes, list):
            translated["path_nodes"] = [
                {
                    **node,
                    **{
                        key: [float(node[key][0]) + dx, float(node[key][1]) + dy]
                        for key in ("anchor", "in", "out") if isinstance(node.get(key), list)
                    },
                }
                for node in path_nodes if isinstance(node, dict)
            ]
        gradient = data.get("gradient")
        if isinstance(gradient, dict):
            for key in ("start", "end"):
                point = gradient.get(key)
                if isinstance(point, list) and len(point) >= 2:
                    gradient[key] = [float(point[0]) + dx, float(point[1]) + dy]
    if str(data.get("shape", "rectangle")).lower() == "boolean":
        translated["box"] = list(shape_data_bounds(translated) or (0, 0, 1, 1))
    return translated

def shape_data_bounds(data: dict[str, Any]) -> tuple[int, int, int, int] | None:
    if str(data.get("shape", "rectangle")).lower() == "boolean":
        bounds = [
            shape_data_bounds(child)
            for child in data.get("children", [])
            if isinstance(child, dict) and bool(child.get("_enabled", True))
        ]
        bounds = [box for box in bounds if box is not None]
        if not bounds:
            return None
        return (
            min(box[0] for box in bounds), min(box[1] for box in bounds),
            max(box[2] for box in bounds), max(box[3] for box in bounds),
        )
    try:
        box = normalized_box(tuple(int(round(value)) for value in data.get("box", [0, 0, 1, 1])))
        angle = math.radians(float(data.get("rotation", 0.0)))
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        corners = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]], dtype=np.float64)
        if abs(angle) <= 1e-8:
            rotated = corners
        else:
            rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            rotated = (corners - (cx, cy)) @ rotation.T + (cx, cy)
        margin = max(0.0, float(data.get("stroke_width", 0)) * (1.0 if str(data.get("stroke_alignment", "center")) == "outside" else 0.5))
        return (
            math.floor(rotated[:, 0].min() - margin), math.floor(rotated[:, 1].min() - margin),
            math.ceil(rotated[:, 0].max() + margin), math.ceil(rotated[:, 1].max() + margin),
        )
    except (TypeError, ValueError):
        return None

def transform_shape_data_to_box(data: dict[str, Any], target: tuple[int, int, int, int]) -> dict[str, Any]:
    """Scale all editable geometry, including nested boolean operands, into target."""
    kind = str(data.get("shape", "rectangle")).lower()
    source = shape_data_bounds(data) if kind == "boolean" else normalized_box(tuple(int(round(value)) for value in data.get("box", [0, 0, 1, 1])))
    if source is None:
        return json.loads(json.dumps(data))
    tx1, ty1, tx2, ty2 = normalized_box(target)
    if kind != "boolean" and abs(float(data.get("rotation", 0.0))) > 1e-8:
        angle = math.radians(float(data.get("rotation", 0.0)))
        cosine, sine = abs(math.cos(angle)), abs(math.sin(angle))
        visual_width, visual_height = tx2 - tx1, ty2 - ty1
        determinant = cosine * cosine - sine * sine
        if abs(determinant) > 0.05:
            base_width = max(1.0, (visual_width * cosine - visual_height * sine) / determinant)
            base_height = max(1.0, (visual_height * cosine - visual_width * sine) / determinant)
        else:
            old_visual = shape_data_bounds(data) or source
            scale = min(visual_width / max(1, old_visual[2] - old_visual[0]), visual_height / max(1, old_visual[3] - old_visual[1]))
            base_width = (source[2] - source[0]) * scale
            base_height = (source[3] - source[1]) * scale
        center_x, center_y = (tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0
        tx1, ty1, tx2, ty2 = center_x - base_width / 2.0, center_y - base_height / 2.0, center_x + base_width / 2.0, center_y + base_height / 2.0
    sx1, sy1, sx2, sy2 = source
    scale_x = (tx2 - tx1) / max(1, sx2 - sx1)
    scale_y = (ty2 - ty1) / max(1, sy2 - sy1)

    def point(value: Any) -> list[float]:
        return [tx1 + (float(value[0]) - sx1) * scale_x, ty1 + (float(value[1]) - sy1) * scale_y]

    def transform(current: dict[str, Any]) -> dict[str, Any]:
        result = json.loads(json.dumps(current))
        if str(current.get("shape", "rectangle")).lower() == "boolean":
            result["children"] = [transform(child) if isinstance(child, dict) else child for child in current.get("children", [])]
        else:
            raw_box = current.get("box", [0, 0, 1, 1])
            first, second = point(raw_box[:2]), point(raw_box[2:4])
            result["box"] = [first[0], first[1], second[0], second[1]]
            points = current.get("control_points")
            if isinstance(points, list):
                result["control_points"] = [point(value) for value in points]
            path_nodes = current.get("path_nodes")
            if isinstance(path_nodes, list):
                result["path_nodes"] = [
                    {
                        **node,
                        **{key: point(node[key]) for key in ("anchor", "in", "out") if isinstance(node.get(key), list)},
                    }
                    for node in path_nodes
                    if isinstance(node, dict)
                ]
            gradient = result.get("gradient")
            if isinstance(gradient, dict):
                for key in ("start", "end"):
                    value = current.get("gradient", {}).get(key)
                    if isinstance(value, list) and len(value) >= 2:
                        gradient[key] = point(value)
        if str(current.get("shape", "rectangle")).lower() == "boolean":
            result["box"] = list(shape_data_bounds(result) or (tx1, ty1, tx2, ty2))
        return result

    transformed = transform(data)
    if kind == "boolean":
        transformed["box"] = list(shape_data_bounds(transformed) or (tx1, ty1, tx2, ty2))
    return transformed

def render_boolean_shape_layer(layer: Layer) -> None:
    if layer.shape_data is None:
        return
    data = layer.shape_data
    mask = boolean_shape_mask(data, layer.pixels.shape[:2])
    fill = tuple(int(v) for v in data.get("fill", [255, 255, 255, 255]))
    out = np.zeros_like(layer.pixels)
    out[:, :, :3] = fill[:3]
    out[:, :, 3] = np.clip(mask.astype(np.float32) * (fill[3] / 255.0), 0, 255).astype(np.uint8)
    stroke = data.get("stroke")
    stroke_width = max(0, int(data.get("stroke_width", 0)))
    if stroke is not None and stroke_width > 0 and np.any(mask):
        stroke_color = tuple(int(v) for v in stroke)
        kernel = np.ones((stroke_width * 2 + 1, stroke_width * 2 + 1), dtype=np.uint8)
        outer = cv2.dilate(mask, kernel)
        inner = cv2.erode(mask, kernel)
        edge = ((outer > 0) & (inner == 0))
        out[edge, :3] = stroke_color[:3]
        out[edge, 3] = stroke_color[3]
    layer.pixels = out

def boolean_shape_mask(data: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    children = data.get("children", [])
    if not isinstance(children, list) or not children:
        return np.zeros(shape, dtype=np.uint8)
    masks = [
        shape_data_to_mask(child, shape)
        for child in children
        if isinstance(child, dict) and bool(child.get("_enabled", True))
    ]
    if not masks:
        return np.zeros(shape, dtype=np.uint8)
    mode = str(data.get("boolean_mode", "union")).lower()
    result = masks[0] > 0
    for mask in masks[1:]:
        other = mask > 0
        if mode == "subtract":
            result = result & ~other
        elif mode == "intersect":
            result = result & other
        elif mode == "xor":
            result = result ^ other
        else:
            result = result | other
    return (result.astype(np.uint8) * 255)

def shape_data_to_mask(data: dict[str, Any], shape: tuple[int, int], apply_rotation: bool = True) -> np.ndarray:
    if str(data.get("shape", "rectangle")).lower() == "boolean":
        return boolean_shape_mask(data, shape)
    box = tuple(float(v) for v in data.get("box", [0, 0, 1, 1]))
    stroke_width = max(1, int(data.get("stroke_width", 1)))
    kind = str(data.get("shape", "rectangle")).lower()
    angle = float(data.get("rotation", 0.0)) if apply_rotation else 0.0
    if kind in {"line", "bezier"}:
        points = [box[:2], box[2:4]] if kind == "line" else editable_bezier_path_points(data, tolerance=0.2)
        center = ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)
        return stroke_polyline_mask(
            shape, rotate_points(points, center, angle), stroke_width,
            str(data.get("stroke_cap", "round")), str(data.get("stroke_join", "round")),
            float(data.get("miter_limit", 4.0)), data.get("dash_pattern"), float(data.get("dash_offset", 0.0)),
        )
    points = None
    if kind == "polygon":
        points = regular_polygon_points(box, max(3, int(data.get("sides", 5))))
    elif kind == "star":
        points = star_points(box, max(3, int(data.get("sides", 5))), float(data.get("inner_ratio", 0.5)))
    elif kind == "custom":
        points = custom_shape_points(data.get("custom_points"), box)
    return closed_shape_mask(shape, kind, box, points, float(data.get("corner_radius", 0)), angle)

__all__ = [name for name in globals() if not name.startswith("__")]
