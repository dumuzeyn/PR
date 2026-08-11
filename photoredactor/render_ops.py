from __future__ import annotations

from .core_shared import *
from .layer import Layer
from .geometry_ops import *
from .selection_ops import *
from .filter_ops import *
from .gpu_acceleration import accelerated_alpha_blend, accelerated_gaussian_blur, accelerated_resize


def layer_alpha_canvas(document: Document, layer: Layer, pixels: np.ndarray | None = None) -> np.ndarray:
    canvas = np.zeros((document.height, document.width), dtype=np.uint8)
    source = pixels if pixels is not None else render_layer_pixels(layer)
    alpha = source[:, :, 3].copy()
    if layer.mask is not None and layer.mask_enabled:
        mask = effective_layer_mask(layer)
        mask_alpha = ((1.0 - float(layer.mask_density)) + (mask.astype(np.float32) / 255.0) * float(layer.mask_density)).clip(0, 1)
        alpha = np.clip(alpha.astype(np.float32) * mask_alpha, 0, 255).astype(np.uint8)
    paste_mask(canvas, alpha, layer.x, layer.y)
    return canvas

def document_alpha_to_layer_mask(alpha_canvas: np.ndarray, layer: Layer) -> np.ndarray:
    mask = np.zeros(layer.pixels.shape[:2], dtype=np.uint8)
    x1, y1 = max(0, layer.x), max(0, layer.y)
    x2 = min(alpha_canvas.shape[1], layer.x + layer.pixels.shape[1])
    y2 = min(alpha_canvas.shape[0], layer.y + layer.pixels.shape[0])
    if x1 >= x2 or y1 >= y2:
        return mask
    lx1, ly1 = x1 - layer.x, y1 - layer.y
    mask[ly1 : ly1 + (y2 - y1), lx1 : lx1 + (x2 - x1)] = alpha_canvas[y1:y2, x1:x2]
    return mask

def render_layer_pixels(layer: Layer) -> np.ndarray:
    if not layer.filters:
        return layer.pixels
    source = layer.working_pixels if layer.working_pixels is not None and layer.working_model == "RGBA" else layer.pixels
    return display_rgba(apply_filter_stack(source, layer.filters))

def render_layer_effects(layer: Layer, pixels: np.ndarray | None = None) -> list[tuple[np.ndarray, int, int, float, str]]:
    effects: list[tuple[np.ndarray, int, int, float, str]] = []
    if not layer.effects:
        return effects
    source = pixels if pixels is not None else render_layer_pixels(layer)
    alpha = source[:, :, 3]
    if not np.any(alpha):
        return effects
    shadow = layer.effects.get("drop_shadow")
    if shadow and shadow.get("enabled", True):
        blur_radius = max(0, int(shadow.get("blur", 12)))
        offset_x = int(shadow.get("x", 10))
        offset_y = int(shadow.get("y", 10))
        opacity = float(shadow.get("opacity", 0.55))
        color = tuple(int(v) for v in shadow.get("color", [0, 0, 0, 255]))
        mask = effect_mask(alpha, blur_radius)
        effects.append((solid_from_alpha(mask, color), layer.x + offset_x, layer.y + offset_y, opacity, "Normal"))
    glow = layer.effects.get("outer_glow")
    if glow and glow.get("enabled", True):
        blur_radius = max(1, int(glow.get("blur", 18)))
        opacity = float(glow.get("opacity", 0.5))
        color = tuple(int(v) for v in glow.get("color", [255, 220, 80, 255]))
        mask = effect_mask(alpha, blur_radius)
        effects.append((solid_from_alpha(mask, color), layer.x, layer.y, opacity, "Screen"))
    stroke = layer.effects.get("stroke")
    if stroke and stroke.get("enabled", True):
        size = max(1, int(stroke.get("size", 4)))
        opacity = float(stroke.get("opacity", 1.0))
        color = tuple(int(v) for v in stroke.get("color", [255, 255, 255, 255]))
        kernel = np.ones((size * 2 + 1, size * 2 + 1), dtype=np.uint8)
        expanded = cv2.dilate((alpha > 0).astype(np.uint8) * 255, kernel)
        outline = np.where((expanded > 0) & (alpha == 0), 255, 0).astype(np.uint8)
        effects.append((solid_from_alpha(outline, color), layer.x, layer.y, opacity, "Normal"))
    return effects

def effect_mask(alpha: np.ndarray, blur_radius: int) -> np.ndarray:
    mask = alpha.copy()
    if blur_radius > 0:
        kernel = np.ones((max(1, blur_radius // 2) * 2 + 1, max(1, blur_radius // 2) * 2 + 1), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel)
        k = blur_radius * 2 + 1
        mask = cv2.GaussianBlur(mask, (k, k), blur_radius)
    return mask

def solid_from_alpha(alpha: np.ndarray, color: tuple[int, int, int, int]) -> np.ndarray:
    arr = np.zeros((alpha.shape[0], alpha.shape[1], 4), dtype=np.uint8)
    arr[:, :, :3] = color[:3]
    arr[:, :, 3] = np.clip(alpha.astype(np.float32) * (color[3] / 255.0), 0, 255).astype(np.uint8)
    return arr

def rotate_bound(arr: np.ndarray, angle: float, interpolation: int) -> np.ndarray:
    h, w = arr.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = max(1, int(h * sin + w * cos))
    new_h = max(1, int(h * cos + w * sin))
    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]
    border = 0 if arr.ndim == 2 else (0, 0, 0, 0)
    return cv2.warpAffine(arr, matrix, (new_w, new_h), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=border)

def render_smart_object(layer: Layer) -> np.ndarray:
    """Render a smart layer from its immutable source and stored transform."""
    if layer.smart_source is None:
        return layer.pixels
    source = layer.smart_source
    transform = (layer.smart_data or {}).get("transform") or {}
    target_w = max(1, int(transform.get("width", source.shape[1])))
    target_h = max(1, int(transform.get("height", source.shape[0])))
    shrinking = target_w < source.shape[1] or target_h < source.shape[0]
    rendered = accelerated_resize(source, (target_w, target_h), cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC)
    if bool(transform.get("flip_horizontal", False)):
        rendered = cv2.flip(rendered, 1)
    if bool(transform.get("flip_vertical", False)):
        rendered = cv2.flip(rendered, 0)
    angle = float(transform.get("angle", 0.0)) % 360.0
    if abs(angle) > 0.001:
        rendered = rotate_bound(rendered, angle, cv2.INTER_CUBIC)
    rendered = np.ascontiguousarray(rendered)
    if layer.transform_data is not None:
        crop = [int(value) for value in layer.transform_data.get("source_crop", [0, 0, rendered.shape[1], rendered.shape[0]])]
        source = rendered[crop[1]:crop[3], crop[0]:crop[2]].copy()
        layer.transform_source = source.copy()
        apply_saved_layer_transform(layer, source, layer.transform_mask_source)
    else:
        layer.pixels = rendered
    return layer.pixels

def warp_pixels(arr: np.ndarray, mode: str, amount: float = 0.35, wavelength: float = 96.0, interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
    h, w = arr.shape[:2]
    if h <= 1 or w <= 1:
        return arr.copy()
    amount = float(np.clip(amount, -2.0, 2.0))
    wavelength = max(4.0, float(wavelength))
    yy, xx = np.indices((h, w), dtype=np.float32)
    src_x = xx.copy()
    src_y = yy.copy()
    mode = str(mode).lower().strip()
    if mode == "arc":
        nx = (xx / max(1.0, w - 1.0)) * 2.0 - 1.0
        src_y = yy - amount * h * 0.25 * (1.0 - nx * nx)
    elif mode == "arc_vertical":
        ny = (yy / max(1.0, h - 1.0)) * 2.0 - 1.0
        src_x = xx - amount * w * 0.25 * (1.0 - ny * ny)
    elif mode in {"bulge", "pinch"}:
        cx = (w - 1.0) / 2.0
        cy = (h - 1.0) / 2.0
        radius = max(1.0, min(w, h) / 2.0)
        dx = (xx - cx) / radius
        dy = (yy - cy) / radius
        r = np.sqrt(dx * dx + dy * dy)
        influence = np.clip(1.0 - r, 0.0, 1.0) ** 2
        direction = -1.0 if mode == "bulge" else 1.0
        factor = np.clip(1.0 + direction * amount * 0.75 * influence, 0.05, 4.0)
        src_x = cx + (xx - cx) * factor
        src_y = cy + (yy - cy) * factor
    elif mode == "wave_x":
        src_x = xx - np.sin(yy / wavelength * math.tau) * amount * w * 0.08
    elif mode == "wave_y":
        src_y = yy - np.sin(xx / wavelength * math.tau) * amount * h * 0.08
    else:
        return arr.copy()
    border = 0 if arr.ndim == 2 else (0, 0, 0, 0)
    return cv2.remap(
        arr,
        src_x.astype(np.float32),
        src_y.astype(np.float32),
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )

def perspective_warp_pixels(
    arr: np.ndarray,
    corners: list[tuple[float, float]] | list[list[float]],
    interpolation: int = cv2.INTER_CUBIC,
) -> tuple[np.ndarray, tuple[int, int]]:
    if len(corners) != 4:
        raise ValueError("Perspective transform needs four corners")
    height, width = arr.shape[:2]
    destination = np.asarray(corners, dtype=np.float32)
    min_x = math.floor(float(destination[:, 0].min()))
    min_y = math.floor(float(destination[:, 1].min()))
    max_x = math.ceil(float(destination[:, 0].max()))
    max_y = math.ceil(float(destination[:, 1].max()))
    output_width = max(1, max_x - min_x + 1)
    output_height = max(1, max_y - min_y + 1)
    source = np.array(
        [[0, 0], [max(0, width - 1), 0], [max(0, width - 1), max(0, height - 1)], [0, max(0, height - 1)]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination - np.array([min_x, min_y], dtype=np.float32))
    border = 0 if arr.ndim == 2 else tuple(0 for _ in range(arr.shape[2]))
    output = cv2.warpPerspective(
        arr,
        matrix,
        (output_width, output_height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )
    return np.ascontiguousarray(output), (min_x, min_y)

def mesh_warp_pixels(
    arr: np.ndarray,
    points: list[tuple[float, float]] | list[list[float]],
    rows: int = 4,
    columns: int = 4,
    interpolation: int = cv2.INTER_CUBIC,
) -> tuple[np.ndarray, tuple[int, int]]:
    rows, columns = max(2, int(rows)), max(2, int(columns))
    if len(points) != rows * columns:
        raise ValueError(f"Mesh transform needs {rows * columns} points")
    destination = np.asarray(points, dtype=np.float32).reshape(rows, columns, 2)
    min_x = math.floor(float(destination[:, :, 0].min()))
    min_y = math.floor(float(destination[:, :, 1].min()))
    max_x = math.ceil(float(destination[:, :, 0].max()))
    max_y = math.ceil(float(destination[:, :, 1].max()))
    output_width = max(1, max_x - min_x + 1)
    output_height = max(1, max_y - min_y + 1)
    destination = destination - np.array([min_x, min_y], dtype=np.float32)
    height, width = arr.shape[:2]
    source_x = np.linspace(0.0, max(0, width - 1), columns, dtype=np.float32)
    source_y = np.linspace(0.0, max(0, height - 1), rows, dtype=np.float32)
    output_shape = (output_height, output_width) if arr.ndim == 2 else (output_height, output_width, arr.shape[2])
    output = np.zeros(output_shape, dtype=arr.dtype)
    coverage = np.zeros((output_height, output_width), dtype=np.uint8)
    triangle_indices = ((0, 1, 3), (0, 3, 2))
    for row in range(rows - 1):
        for column in range(columns - 1):
            source_quad = np.array(
                [
                    [source_x[column], source_y[row]],
                    [source_x[column + 1], source_y[row]],
                    [source_x[column], source_y[row + 1]],
                    [source_x[column + 1], source_y[row + 1]],
                ],
                dtype=np.float32,
            )
            destination_quad = np.array(
                [
                    destination[row, column],
                    destination[row, column + 1],
                    destination[row + 1, column],
                    destination[row + 1, column + 1],
                ],
                dtype=np.float32,
            )
            for indices in triangle_indices:
                src_triangle = source_quad[list(indices)]
                dst_triangle = destination_quad[list(indices)]
                matrix = cv2.getAffineTransform(src_triangle, dst_triangle)
                border = 0 if arr.ndim == 2 else tuple(0 for _ in range(arr.shape[2]))
                warped = cv2.warpAffine(
                    arr,
                    matrix,
                    (output_width, output_height),
                    flags=interpolation,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=border,
                )
                triangle_mask = np.zeros_like(coverage)
                cv2.fillConvexPoly(triangle_mask, np.round(dst_triangle).astype(np.int32), 255, lineType=cv2.LINE_AA)
                active = triangle_mask > 0
                output[active] = warped[active]
                coverage[active] = 255
    return np.ascontiguousarray(output), (min_x, min_y)

def apply_saved_layer_transform(layer: Layer, source_pixels: np.ndarray | None = None, source_mask: np.ndarray | None = None) -> None:
    data = layer.transform_data
    if not data:
        return
    pixels = source_pixels if source_pixels is not None else layer.transform_source
    if pixels is None:
        return
    mode = str(data.get("mode", "perspective"))
    points = data.get("points") or []
    if mode == "mesh":
        transformed, offset = mesh_warp_pixels(pixels, points, int(data.get("rows", 4)), int(data.get("columns", 4)), cv2.INTER_CUBIC)
    else:
        transformed, offset = perspective_warp_pixels(pixels, points, cv2.INTER_CUBIC)
    layer.pixels = transformed
    base_x = int(data.get("base_x", 0))
    base_y = int(data.get("base_y", 0))
    layer.x = base_x + offset[0]
    layer.y = base_y + offset[1]
    mask = source_mask if source_mask is not None else layer.transform_mask_source
    if mask is not None:
        if mode == "mesh":
            layer.mask, _ = mesh_warp_pixels(mask, points, int(data.get("rows", 4)), int(data.get("columns", 4)), cv2.INTER_LINEAR)
        else:
            layer.mask, _ = perspective_warp_pixels(mask, points, cv2.INTER_LINEAR)

def alpha_blend(dst: np.ndarray, src: np.ndarray, x: int, y: int, opacity: float) -> np.ndarray:
    out = dst.copy()
    alpha_blend_inplace(out, src, x, y, opacity)
    return out

def alpha_blend_inplace(dst: np.ndarray, src: np.ndarray, x: int, y: int, opacity: float, alpha_mask: np.ndarray | None = None, mask_density: float = 1.0, blend_mode: str = "Normal") -> None:
    h, w = src.shape[:2]
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(dst.shape[1], int(x) + w), min(dst.shape[0], int(y) + h)
    if x1 >= x2 or y1 >= y2:
        return
    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
    s = src[sy1:sy2, sx1:sx2].astype(np.float32)
    target = dst[y1:y2, x1:x2]
    if blend_mode == "Normal":
        accelerated = accelerated_alpha_blend(
            src[sy1:sy2, sx1:sx2],
            target,
            opacity,
            None if alpha_mask is None else alpha_mask[sy1:sy2, sx1:sx2],
            mask_density,
        )
        if accelerated is not None:
            target[:] = accelerated
            return
    d = target.astype(np.float32)
    sa = (s[:, :, 3:4] / 255.0) * float(opacity)
    if alpha_mask is not None:
        ma = alpha_mask[sy1:sy2, sx1:sx2].astype(np.float32) / 255.0
        sa *= (1.0 - float(mask_density)) + ma[:, :, None] * float(mask_density)
    da = d[:, :, 3:4] / 255.0
    oa = sa + da * (1.0 - sa)
    blended = blend_rgb(s[:, :, :3], d[:, :, :3], blend_mode)
    rgb = np.where(oa > 0, (blended * sa + d[:, :, :3] * da * (1.0 - sa)) / np.maximum(oa, 1e-6), 0)
    target[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    target[:, :, 3] = np.clip(oa[:, :, 0] * 255, 0, 255).astype(np.uint8)

def brush_mask(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    cached = _brush_mask_cache.get(radius)
    if cached is not None:
        return cached
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = (xx * xx + yy * yy) <= radius * radius
    if len(_brush_mask_cache) > 64:
        _brush_mask_cache.clear()
    _brush_mask_cache[radius] = mask
    return mask

def retouch_falloff_mask(radius: int, hardness: float = 0.5) -> np.ndarray:
    radius = max(1, int(radius))
    hardness = float(hardness)
    if hardness > 1.0:
        hardness /= 100.0
    hardness = float(np.clip(hardness, 0.0, 1.0))
    key = radius, int(round(hardness * 100))
    cached = _retouch_mask_cache.get(key)
    if cached is not None:
        return cached
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    distance = np.sqrt(xx * xx + yy * yy).astype(np.float32) / float(radius)
    solid_radius = hardness * 0.96
    falloff = np.clip((1.0 - distance) / max(0.04, 1.0 - solid_radius), 0.0, 1.0)
    if len(_retouch_mask_cache) > 128:
        _retouch_mask_cache.clear()
    _retouch_mask_cache[key] = falloff
    return falloff

def retouch_effect_halo(mode: str, strength: float) -> int:
    if mode == "blur":
        sigma = 0.65 + 1.85 * float(np.clip(strength, 0.0, 1.0))
        return max(3, int(math.ceil(sigma * 3.0)))
    if mode == "sharpen":
        return 4
    return 0

def retouch_effect_rgb(source: np.ndarray, mode: str, strength: float, tonal_range: str = "midtones") -> np.ndarray:
    rgb = source[:, :, :3].astype(np.float32)
    strength = float(np.clip(strength, 0.0, 1.0))
    if mode == "blur":
        sigma = 0.65 + 1.85 * strength
        alpha = source[:, :, 3].astype(np.float32) / 255.0
        weight = accelerated_gaussian_blur(alpha, (0, 0), sigma, cv2.BORDER_REFLECT_101)
        premultiplied = accelerated_gaussian_blur(rgb * alpha[:, :, None], (0, 0), sigma, cv2.BORDER_REFLECT_101)
        return np.where(weight[:, :, None] > 1e-4, premultiplied / np.maximum(weight[:, :, None], 1e-4), rgb)
    if mode == "sharpen":
        blurred = accelerated_gaussian_blur(rgb, (0, 0), 1.0, cv2.BORDER_REFLECT_101)
        detail = rgb - blurred
        detail_luma = np.abs(detail[:, :, 0] * 0.2126 + detail[:, :, 1] * 0.7152 + detail[:, :, 2] * 0.0722)
        detail[detail_luma < 2.0] = 0.0
        return np.clip(rgb + detail * 0.85, 0.0, 255.0)
    if mode not in {"dodge", "burn"}:
        return rgb
    hls = cv2.cvtColor(source[:, :, :3], cv2.COLOR_RGB2HLS).astype(np.float32)
    luminance = hls[:, :, 1] / 255.0
    range_name = str(tonal_range).lower()
    if range_name in {"shadows", "тени"}:
        tonal_weight = np.clip(1.0 - luminance, 0.0, 1.0)
    elif range_name in {"highlights", "света"}:
        tonal_weight = np.clip(luminance, 0.0, 1.0)
    else:
        tonal_weight = np.clip(1.0 - np.abs(luminance - 0.5) * 2.0, 0.08, 1.0)
    if mode == "dodge":
        luminance = luminance + (1.0 - luminance) * 0.38 * tonal_weight
    else:
        luminance = luminance - luminance * 0.38 * tonal_weight
    hls[:, :, 1] = np.clip(luminance * 255.0, 0.0, 255.0)
    return cv2.cvtColor(hls.astype(np.uint8), cv2.COLOR_HLS2RGB).astype(np.float32)

__all__ = [name for name in globals() if not name.startswith("__")]
