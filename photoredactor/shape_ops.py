from __future__ import annotations

from .core_shared import *
from .layer import Layer
from .geometry_ops import *
from .selection_ops import *
from .render_ops import *
from .text_ops import *


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
        return
    gradient = data.get("gradient")
    if isinstance(gradient, dict):
        height, width = layer.pixels.shape[:2]
        region = shape_render_region(data, width, height)
        if region is None:
            return
        x1, y1, x2, y2 = region
        local_data = translated_shape_data(data, -x1, -y1)
        mask = shape_data_to_mask(local_data, (y2 - y1, x2 - x1))
        start = tuple(float(value) for value in gradient.get("start", data.get("box", [0, 0])[:2]))
        end = tuple(float(value) for value in gradient.get("end", data.get("box", [0, 0, 1, 1])[2:4]))
        pixels = GradientEngine.render(
            x2 - x1,
            y2 - y1,
            (start[0] - x1, start[1] - y1),
            (end[0] - x1, end[1] - y1),
            gradient.get("stops"),
            str(gradient.get("type", "linear")),
        )
        opacity = float(np.clip(gradient.get("opacity", 1.0), 0.0, 1.0))
        pixels[:, :, 3] = np.clip(pixels[:, :, 3].astype(np.float32) * (mask.astype(np.float32) / 255.0) * opacity, 0, 255).astype(np.uint8)
        stroke = data.get("stroke")
        stroke_width = max(0, int(data.get("stroke_width", 0)))
        if stroke is not None and stroke_width > 0:
            stroke_color = np.array(stroke, dtype=np.uint8)
            kernel = np.ones((stroke_width * 2 + 1, stroke_width * 2 + 1), dtype=np.uint8)
            edge = (cv2.dilate(mask, kernel) > 0) & (cv2.erode(mask, kernel) == 0)
            pixels[edge] = stroke_color
        layer.pixels[y1:y2, x1:x2] = pixels
        return
    texture = data.get("texture")
    if isinstance(texture, dict):
        height, width = layer.pixels.shape[:2]
        region = shape_render_region(data, width, height)
        if region is None:
            return
        x1, y1, x2, y2 = region
        local_data = translated_shape_data(data, -x1, -y1)
        mask = shape_data_to_mask(local_data, (y2 - y1, x2 - x1))
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
            kernel = np.ones((stroke_width * 2 + 1, stroke_width * 2 + 1), dtype=np.uint8)
            edge = (cv2.dilate(mask, kernel) > 0) & (cv2.erode(mask, kernel) == 0)
            pixels[edge] = stroke_color
        layer.pixels[y1:y2, x1:x2] = pixels
        return
    pil = rgba_array_to_pil(layer.pixels)
    draw = ImageDraw.Draw(pil)
    box = tuple(int(v) for v in data.get("box", [0, 0, 1, 1]))
    fill = tuple(int(v) for v in data.get("fill", [255, 255, 255, 255]))
    stroke = data.get("stroke")
    outline = None if stroke is None else tuple(int(v) for v in stroke)
    stroke_width = max(0, int(data.get("stroke_width", 0)))
    shape = str(data.get("shape", "rectangle")).lower()
    if shape == "ellipse":
        draw.ellipse(box, fill=fill, outline=outline, width=stroke_width)
    elif shape == "line":
        draw.line((box[0], box[1], box[2], box[3]), fill=outline or fill, width=max(1, stroke_width or 1))
    elif shape == "bezier":
        points = bezier_curve_points(data.get("control_points"), box)
        draw.line(points, fill=outline or fill, width=max(1, stroke_width or 2), joint="curve")
    elif shape == "polygon":
        points = regular_polygon_points(box, max(3, int(data.get("sides", 5))))
        draw.polygon(points, fill=fill)
        if outline is not None and stroke_width > 0:
            draw.line(points + [points[0]], fill=outline, width=stroke_width)
    elif shape == "star":
        points = star_points(box, max(3, int(data.get("sides", 5))), float(data.get("inner_ratio", 0.5)))
        draw.polygon(points, fill=fill)
        if outline is not None and stroke_width > 0:
            draw.line(points + [points[0]], fill=outline, width=stroke_width)
    elif shape == "custom":
        points = custom_shape_points(data.get("custom_points"), box)
        draw.polygon(points, fill=fill)
        if outline is not None and stroke_width > 0 and points:
            draw.line(points + [points[0]], fill=outline, width=stroke_width)
    else:
        draw.rectangle(box, fill=fill, outline=outline, width=stroke_width)
    layer.pixels = pil_to_rgba_array(pil)

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
        gradient = data.get("gradient")
        if isinstance(gradient, dict):
            for key in ("start", "end"):
                point = gradient.get(key)
                if isinstance(point, list) and len(point) >= 2:
                    gradient[key] = [float(point[0]) + dx, float(point[1]) + dy]
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
        return normalized_box(tuple(int(round(value)) for value in data.get("box", [0, 0, 1, 1])))
    except (TypeError, ValueError):
        return None

def transform_shape_data_to_box(data: dict[str, Any], target: tuple[int, int, int, int]) -> dict[str, Any]:
    """Scale all editable geometry, including nested boolean operands, into target."""
    source = shape_data_bounds(data)
    if source is None:
        return json.loads(json.dumps(data))
    tx1, ty1, tx2, ty2 = normalized_box(target)
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
            gradient = result.get("gradient")
            if isinstance(gradient, dict):
                for key in ("start", "end"):
                    value = current.get("gradient", {}).get(key)
                    if isinstance(value, list) and len(value) >= 2:
                        gradient[key] = point(value)
        result["box"] = list(shape_data_bounds(result) or (tx1, ty1, tx2, ty2))
        return result

    transformed = transform(data)
    transformed["box"] = [tx1, ty1, tx2, ty2]
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

def shape_data_to_mask(data: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    if str(data.get("shape", "rectangle")).lower() == "boolean":
        return boolean_shape_mask(data, shape)
    pil = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(pil)
    box = tuple(int(v) for v in data.get("box", [0, 0, 1, 1]))
    stroke_width = max(1, int(data.get("stroke_width", 1)))
    kind = str(data.get("shape", "rectangle")).lower()
    if kind == "ellipse":
        draw.ellipse(box, fill=255)
    elif kind == "line":
        draw.line((box[0], box[1], box[2], box[3]), fill=255, width=stroke_width)
    elif kind == "bezier":
        draw.line(bezier_curve_points(data.get("control_points"), box), fill=255, width=max(1, stroke_width), joint="curve")
    elif kind == "polygon":
        draw.polygon(regular_polygon_points(box, max(3, int(data.get("sides", 5)))), fill=255)
    elif kind == "star":
        draw.polygon(star_points(box, max(3, int(data.get("sides", 5))), float(data.get("inner_ratio", 0.5))), fill=255)
    elif kind == "custom":
        draw.polygon(custom_shape_points(data.get("custom_points"), box), fill=255)
    else:
        draw.rectangle(box, fill=255)
    return np.array(pil, dtype=np.uint8)

__all__ = [name for name in globals() if not name.startswith("__")]
