from __future__ import annotations

import copy

from .core_shared import *
from .layer import Layer
from .geometry_ops import *
from .selection_ops import *
from .render_ops import *
from .retouch_ops import *
from .text_shaper import DEFAULT_TEXT_SHAPER, TextStyle, grapheme_clusters, style_from_data
from .text_layout import DEFAULT_TEXT_LAYOUT


def render_text_in_local_region(layer: Layer) -> bool:
    data = layer.text_data
    if data is None or data.get("_local_render") or str(data.get("path_mode", "none")) == "bezier":
        return False
    height, width = layer.pixels.shape[:2]
    size = max(4, int(data.get("size", 48)))
    text = str(data.get("text", ""))
    box_width = max(0, int(data.get("box_width", 0) or 0))
    tracking = abs(int(data.get("tracking", 0)))
    longest = max((len(line) for line in text.splitlines()), default=1)
    estimated_width = box_width or max(size * 2, longest * (size + tracking))
    if int(data.get("box_height", 0)) > 0:
        estimated_height = int(data.get("box_height", 0))
        estimated_lines = max(1, len(text.splitlines()))
    elif box_width:
        estimated_lines = sum(max(1, math.ceil(max(1, len(line)) * size * 0.7 / max(1, box_width))) for line in text.splitlines() or [""])
        estimated_height = 0
    else:
        estimated_lines = max(1, len(text.splitlines()))
        estimated_height = 0
    if bool(data.get("vertical", False)):
        estimated_width = max(1, len(text.splitlines())) * round(size * 1.4)
        estimated_height = max((len(line) for line in text.splitlines()), default=1) * (size + int(data.get("line_spacing", 0)))
    else:
        estimated_height = estimated_height or estimated_lines * (round(size * 1.7) + int(data.get("line_spacing", 0)))
    estimated_height += estimated_lines * (int(data.get("spacing_before", 0)) + int(data.get("spacing_after", 0)))
    padding = max(size * 3, abs(int(data.get("baseline_shift", 0))) + size, abs(int(data.get("first_line_indent", 0))) + size)
    if abs(float(data.get("rotation", 0.0))) > 0.001:
        diagonal = math.ceil(math.hypot(estimated_width, estimated_height))
        estimated_width = estimated_height = diagonal
    local_width = max(1, min(width + padding * 2, estimated_width + padding * 2))
    local_height = max(1, min(height + padding * 2, estimated_height + padding * 2))
    if local_width * local_height >= width * height * 0.85:
        return False
    original_data = data
    local_data = copy.deepcopy(data)
    local_data.update({"x": padding, "y": padding, "_local_render": True})
    layer.text_data = local_data
    layer.pixels = blank_rgba(local_width, local_height, (0, 0, 0, 0))
    render_text_layer(layer)
    local_pixels = layer.pixels
    layer.text_data = original_data
    output = blank_rgba(width, height, (0, 0, 0, 0))
    if np.any(local_pixels[:, :, 3]):
        ys, xs = np.where(local_pixels[:, :, 3] > 0)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
        target_x = int(original_data.get("x", 0)) - padding + x1
        target_y = int(original_data.get("y", 0)) - padding + y1
        alpha_blend_inplace(output, local_pixels[y1:y2, x1:x2], target_x, target_y, 1.0)
    layer.pixels = output
    return True


def render_text_layer(layer: Layer) -> None:
    if layer.text_data is None:
        return
    if layer.transform_data is not None:
        data = layer.transform_data
        layer.transform_data = None
        layer.x = int(data.get("render_base_x", data.get("base_x", layer.x)))
        layer.y = int(data.get("render_base_y", data.get("base_y", layer.y)))
        layer.pixels = blank_rgba(int(data.get("render_width", layer.pixels.shape[1])), int(data.get("render_height", layer.pixels.shape[0])), (0, 0, 0, 0))
        render_text_layer(layer)
        crop = [int(value) for value in data.get("source_crop", [0, 0, layer.pixels.shape[1], layer.pixels.shape[0]])]
        source = layer.pixels[crop[1]:crop[3], crop[0]:crop[2]].copy()
        layer.transform_data = data
        layer.transform_source = source.copy()
        apply_saved_layer_transform(layer, source, layer.transform_mask_source)
        return
    if render_text_in_local_region(layer):
        return
    layer.pixels[:] = 0
    pil = rgba_array_to_pil(layer.pixels)
    draw = ImageDraw.Draw(pil)
    data = layer.text_data
    bold = bool(data.get("bold", False))
    italic = bool(data.get("italic", False))
    underline = bool(data.get("underline", False))
    strike_through = bool(data.get("strike_through", False))
    text_style = style_from_data(data)
    font = DEFAULT_TEXT_SHAPER.shape(" ", text_style).font
    color = tuple(int(v) for v in data.get("color", [255, 255, 255, 255]))
    x = int(data.get("x", 0))
    y = int(data.get("y", 0))
    size = int(data.get("size", 48))
    box_width = max(0, int(data.get("box_width", 0) or 0))
    spacing = max(0, int(data.get("line_spacing", max(2, size // 5))))
    tracking = int(data.get("tracking", 0))
    align = str(data.get("align", "left")).lower()
    path_mode = str(data.get("path_mode", "none")).lower()
    path_amount = int(data.get("path_amount", 0))
    baseline_shift = int(data.get("baseline_shift", 0))
    if path_mode == "bezier":
        draw_text_on_bezier_path(pil, data, font, color, tracking, bold, underline)
    elif bool(data.get("vertical", False)):
        line_x, line_y = x, y - baseline_shift
        advance_x = max(1, round(size * 1.25))
        for cluster in grapheme_clusters(str(data.get("text", ""))):
            if cluster == "\n":
                line_x += advance_x
                line_y = y - baseline_shift
                continue
            run = DEFAULT_TEXT_SHAPER.shape(cluster, TextStyle(**{**text_style.__dict__, "tracking": 0.0}))
            draw_text_styled(
                draw, (line_x, line_y), cluster, fill=color, font=font, tracking=0,
                bold=bold, underline=underline, strike_through=strike_through,
                style=TextStyle(**{**text_style.__dict__, "tracking": 0.0}),
            )
            line_y += max(1, run.bbox[3] - run.bbox[1]) + spacing
    else:
        layout = DEFAULT_TEXT_LAYOUT.layout(data)
        for line in layout.lines:
            if line.justify:
                words = line.text.split(" ")
                words_width = sum(DEFAULT_TEXT_LAYOUT.measure(word, text_style) for word in words)
                gap = max(0.0, (line.available - words_width) / max(1, len(words) - 1))
                cursor = float(line.x)
                for word in words:
                    draw_text_styled(draw, (round(cursor), round(line.y)), word, fill=color, font=font, tracking=tracking, bold=bold, underline=underline, strike_through=strike_through, path_mode=path_mode, path_amount=path_amount, style=text_style)
                    cursor += DEFAULT_TEXT_LAYOUT.measure(word, text_style) + gap
            else:
                draw_text_styled(draw, (round(line.x), round(line.y)), line.text, fill=color, font=font, tracking=tracking, bold=bold, underline=underline, strike_through=strike_through, path_mode=path_mode, path_amount=path_amount, style=text_style)
    rendered = pil_to_rgba_array(pil)
    if str(data.get("text_mode", "paragraph" if box_width else "point")) == "paragraph" and int(data.get("box_height", 0)) > 0:
        box_height = max(1, int(data.get("box_height", 0)))
        clip = np.zeros(rendered.shape[:2], dtype=np.uint8)
        clip[max(0, y):min(rendered.shape[0], y + box_height), max(0, x):min(rendered.shape[1], x + max(1, box_width))] = 255
        rendered[:, :, 3] = np.minimum(rendered[:, :, 3], clip)
    horizontal_scale = max(1, int(data.get("horizontal_scale", 100))) / 100.0
    vertical_scale = max(1, int(data.get("vertical_scale", 100))) / 100.0
    if (abs(horizontal_scale - 1.0) > 0.001 or abs(vertical_scale - 1.0) > 0.001) and np.any(rendered[:, :, 3]):
        ys, xs = np.where(rendered[:, :, 3] > 0)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
        patch = rendered[y1:y2, x1:x2]
        scaled = cv2.resize(patch, (max(1, round(patch.shape[1] * horizontal_scale)), max(1, round(patch.shape[0] * vertical_scale))), interpolation=cv2.INTER_CUBIC)
        resized = np.zeros_like(rendered)
        alpha_blend_inplace(resized, scaled, x1, y1, 1.0)
        rendered = resized
    rotation = float(data.get("rotation", 0.0))
    flip_horizontal = bool(data.get("flip_horizontal", False))
    flip_vertical = bool(data.get("flip_vertical", False))
    if (abs(rotation) > 0.001 or flip_horizontal or flip_vertical) and np.any(rendered[:, :, 3]):
        ys, xs = np.where(rendered[:, :, 3] > 0)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
        patch = rendered[y1:y2, x1:x2]
        if flip_horizontal:
            patch = cv2.flip(patch, 1)
        if flip_vertical:
            patch = cv2.flip(patch, 0)
        if abs(rotation) > 0.001:
            patch = rotate_bound(patch, rotation, cv2.INTER_CUBIC)
        rotated = np.zeros_like(rendered)
        px = round((x1 + x2 - patch.shape[1]) / 2)
        py = round((y1 + y2 - patch.shape[0]) / 2)
        alpha_blend_inplace(rotated, patch, px, py, 1.0)
        rendered = rotated
    layer.pixels = rendered

def normalize_text_path_points(
    points: list[tuple[float, float]] | list[list[float]] | None,
    x: int,
    baseline_y: int,
    width: int,
) -> list[list[float]]:
    if isinstance(points, list) and len(points) == 4:
        normalized: list[list[float]] = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                break
            try:
                normalized.append([float(point[0]), float(point[1])])
            except (TypeError, ValueError):
                break
        if len(normalized) == 4:
            return normalized
    path_width = max(40.0, float(width))
    return [
        [float(x), float(baseline_y)],
        [float(x) + path_width / 3.0, float(baseline_y)],
        [float(x) + path_width * 2.0 / 3.0, float(baseline_y)],
        [float(x) + path_width, float(baseline_y)],
    ]

def normalize_text_path_range(start: float, end: float) -> tuple[float, float]:
    normalized_start = max(0.0, min(0.99, float(start)))
    normalized_end = max(0.01, min(1.0, float(end)))
    if normalized_end - normalized_start < 0.01:
        if normalized_start <= 0.99:
            normalized_end = min(1.0, normalized_start + 0.01)
        else:
            normalized_start = max(0.0, normalized_end - 0.01)
    return normalized_start, normalized_end

def text_path_samples(
    points: list[tuple[float, float]] | list[list[float]],
    steps: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = normalize_text_path_points(points, 0, 0, 100)
    control = np.asarray(normalized, dtype=np.float64)
    count = max(16, int(steps))
    t = np.linspace(0.0, 1.0, count + 1, dtype=np.float64)
    omt = 1.0 - t
    positions = (
        (omt ** 3)[:, None] * control[0]
        + (3.0 * omt * omt * t)[:, None] * control[1]
        + (3.0 * omt * t * t)[:, None] * control[2]
        + (t ** 3)[:, None] * control[3]
    )
    tangents = (
        (3.0 * omt * omt)[:, None] * (control[1] - control[0])
        + (6.0 * omt * t)[:, None] * (control[2] - control[1])
        + (3.0 * t * t)[:, None] * (control[3] - control[2])
    )
    lengths = np.linalg.norm(tangents, axis=1)
    invalid = lengths < 1e-8
    if np.any(invalid):
        differences = np.gradient(positions, axis=0)
        tangents[invalid] = differences[invalid]
        lengths = np.linalg.norm(tangents, axis=1)
    tangents /= np.maximum(lengths[:, None], 1e-8)
    segments = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segments)))
    return positions, tangents, cumulative

def text_path_point_at_distance(
    positions: np.ndarray,
    tangents: np.ndarray,
    cumulative: np.ndarray,
    distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(cumulative) < 2 or cumulative[-1] <= 1e-8:
        return positions[0].copy(), tangents[0].copy()
    target = max(0.0, min(float(cumulative[-1]), float(distance)))
    upper = min(len(cumulative) - 1, max(1, int(np.searchsorted(cumulative, target, side="right"))))
    lower = upper - 1
    span = max(1e-8, float(cumulative[upper] - cumulative[lower]))
    ratio = (target - float(cumulative[lower])) / span
    point = positions[lower] * (1.0 - ratio) + positions[upper] * ratio
    tangent = tangents[lower] * (1.0 - ratio) + tangents[upper] * ratio
    tangent /= max(1e-8, float(np.linalg.norm(tangent)))
    return point, tangent

def draw_text_on_bezier_path(
    image: Image.Image,
    data: dict[str, Any],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    tracking: int = 0,
    bold: bool = False,
    underline: bool = False,
) -> None:
    size = max(4, int(data.get("size", 48)))
    points = normalize_text_path_points(
        data.get("path_points"),
        int(data.get("x", 0)),
        int(data.get("y", 0)) + size,
        max(int(data.get("box_width", 0) or 0), size * 8),
    )
    sample_count = min(4096, max(256, int(cumulative_path_estimate(points) * 1.5)))
    positions, tangents, cumulative = text_path_samples(points, sample_count)
    path_reversed = bool(data.get("path_reverse", False))
    if path_reversed:
        positions = positions[::-1].copy()
        tangents = (-tangents[::-1]).copy()
        segments = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segments)))
    total_path = float(cumulative[-1])
    start_fraction = max(0.0, min(1.0, float(data.get("path_start", 0.0))))
    end_fraction = max(0.0, min(1.0, float(data.get("path_end", 1.0))))
    start_distance, end_distance = sorted((start_fraction * total_path, end_fraction * total_path))
    available = end_distance - start_distance
    if available <= 1.0:
        return
    text = " ".join(str(data.get("text", "")).splitlines())
    if not text:
        return
    style = style_from_data(data)
    clusters = grapheme_clusters(text)
    advances: list[float] = []
    prefix = ""
    previous_advance = 0.0
    for cluster in clusters:
        prefix += cluster
        shaped_advance = DEFAULT_TEXT_SHAPER.shape(prefix, style).advance
        advances.append(max(1.0, shaped_advance - previous_advance + int(tracking)))
        previous_advance = shaped_advance
    total_text = max(0.0, sum(advances) - int(tracking))
    align = str(data.get("align", "left")).lower()
    cursor = start_distance
    if align == "center":
        cursor += max(0.0, (available - total_text) / 2.0)
    elif align == "right":
        cursor += max(0.0, available - total_text)
    baseline_shift = float(data.get("baseline_shift", 0))
    side = -1 if int(data.get("path_side", 1)) < 0 else 1
    first_distance = cursor
    last_distance = cursor
    for cluster, advance in zip(clusters, advances):
        center_distance = cursor + advance / 2.0
        if center_distance > end_distance:
            break
        point, tangent = text_path_point_at_distance(positions, tangents, cumulative, center_distance)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        if path_reversed:
            normal = -normal
        run = DEFAULT_TEXT_SHAPER.shape(cluster, TextStyle(**{**style.__dict__, "tracking": 0.0}))
        bbox = run.bbox
        glyph_width = max(1, bbox[2] - bbox[0])
        glyph_height = max(1, bbox[3] - bbox[1])
        padding = max(4, size // 4)
        glyph = Image.new("RGBA", (glyph_width + padding * 2, glyph_height + padding * 2), (0, 0, 0, 0))
        glyph_draw = ImageDraw.Draw(glyph)
        DEFAULT_TEXT_SHAPER.draw(glyph_draw, (padding - bbox[0], padding - bbox[1]), run, fill, 1 if bold else 0)
        angle = math.degrees(math.atan2(float(tangent[1]), float(tangent[0])))
        rotated = glyph.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=True)
        center = point - normal * side * (glyph_height / 2.0 + baseline_shift)
        left = round(float(center[0]) - rotated.width / 2.0)
        top = round(float(center[1]) - rotated.height / 2.0)
        image.paste(rotated, (left, top), rotated)
        cursor += advance
        last_distance = min(cursor, end_distance)
    decorations = []
    if underline:
        decorations.append(max(1.0, size * 0.08))
    if bool(data.get("strike_through", False)):
        decorations.append(-max(1.0, size * 0.42))
    for decoration_offset in decorations:
        if last_distance <= first_distance:
            continue
        path_draw = ImageDraw.Draw(image)
        samples = max(8, round((last_distance - first_distance) / 4.0))
        line_points: list[tuple[float, float]] = []
        for distance in np.linspace(first_distance, last_distance, samples):
            point, tangent = text_path_point_at_distance(positions, tangents, cumulative, float(distance))
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
            if path_reversed:
                normal = -normal
            decoration_point = point + normal * side * (decoration_offset - baseline_shift)
            line_points.append((float(decoration_point[0]), float(decoration_point[1])))
        path_draw.line(line_points, fill=fill, width=max(1, size // 16), joint="curve")

def cumulative_path_estimate(points: list[list[float]]) -> float:
    return max(16.0, sum(math.hypot(points[index + 1][0] - points[index][0], points[index + 1][1] - points[index][1]) for index in range(3)))

def wrapped_text_lines(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, box_width: int,
    tracking: int = 0, style: TextStyle | None = None,
) -> list[str]:
    if box_width <= 0:
        return text.splitlines() or [""]
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for word in paragraph.split(" "):
            candidate = word if not current else f"{current} {word}"
            if text_line_width(draw, candidate, font, tracking, style) <= box_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
            if current and text_line_width(draw, current, font, tracking, style) > box_width:
                clusters = grapheme_clusters(current)
                current = ""
                chunk = ""
                for cluster in clusters:
                    candidate_cluster = chunk + cluster
                    if chunk and text_line_width(draw, candidate_cluster, font, tracking, style) > box_width:
                        lines.append(chunk)
                        chunk = cluster
                    else:
                        chunk = candidate_cluster
                current = chunk
        lines.append(current)
    return lines

def text_line_width(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont,
    tracking: int = 0, style: TextStyle | None = None,
) -> int:
    if not text:
        return 0
    if style is not None:
        return max(0, round(DEFAULT_TEXT_SHAPER.shape(text, style).advance))
    bbox = draw.textbbox((0, 0), text, font=font)
    base_width = bbox[2] - bbox[0]
    return max(0, base_width + max(0, len(grapheme_clusters(text)) - 1) * int(tracking))

def draw_text_with_tracking(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int, int], font: ImageFont.ImageFont, tracking: int = 0) -> None:
    if tracking == 0 or len(text) <= 1:
        draw.text(xy, text, fill=fill, font=font)
        return
    x, y = xy
    for char in text:
        draw.text((x, y), char, fill=fill, font=font)
        bbox = draw.textbbox((0, 0), char, font=font)
        x += max(0, bbox[2] - bbox[0]) + int(tracking)

def draw_text_styled(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fill: tuple[int, int, int, int],
    font: ImageFont.ImageFont,
    tracking: int = 0,
    bold: bool = False,
    underline: bool = False,
    strike_through: bool = False,
    path_mode: str = "none",
    path_amount: int = 0,
    style: TextStyle | None = None,
) -> None:
    x, y = xy
    active_style = style or TextStyle(size=int(getattr(font, "size", 12)), tracking=tracking)
    total_width = max(1, text_line_width(draw, text or " ", font, tracking, active_style))
    if path_mode not in {"arc", "wave"}:
        run = DEFAULT_TEXT_SHAPER.shape(text, active_style)
        DEFAULT_TEXT_SHAPER.draw(draw, (x, y), run, fill, 1 if bold else 0)
        if underline and text:
            underline_y = y + max(1, int(getattr(font, "size", 12) * 1.05))
            draw.line((x, underline_y, x + total_width, underline_y), fill=fill, width=max(1, int(getattr(font, "size", 12) / 16)))
        if strike_through and text:
            strike_y = y + max(1, int(getattr(font, "size", 12) * 0.55))
            draw.line((x, strike_y, x + total_width, strike_y), fill=fill, width=max(1, int(getattr(font, "size", 12) / 16)))
        return
    cursor = 0.0
    for cluster in grapheme_clusters(text):
        run = DEFAULT_TEXT_SHAPER.shape(cluster, TextStyle(**{**active_style.__dict__, "tracking": 0.0}))
        char_width = max(1.0, run.advance)
        center = cursor + char_width / 2.0
        offset = 0.0
        if path_mode == "arc":
            normalized = center / total_width * 2.0 - 1.0
            offset = -float(path_amount) * (1.0 - normalized * normalized)
        elif path_mode == "wave":
            offset = float(path_amount) * math.sin(center / total_width * math.pi * 2.0)
        DEFAULT_TEXT_SHAPER.draw(draw, (x + cursor, y + offset), run, fill, 1 if bold else 0)
        cursor += char_width + int(tracking)
    if underline and text:
        underline_y = y + max(1, int(getattr(font, "size", 12) * 1.05))
        draw.line((x, underline_y, x + total_width, underline_y), fill=fill, width=max(1, int(getattr(font, "size", 12) / 16)))
    if strike_through and text:
        strike_y = y + max(1, int(getattr(font, "size", 12) * 0.55))
        draw.line((x, strike_y, x + total_width, strike_y), fill=fill, width=max(1, int(getattr(font, "size", 12) / 16)))

def load_text_font(font_family: str, size: int, bold: bool = False, italic: bool = False) -> ImageFont.ImageFont:
    return DEFAULT_TEXT_SHAPER.shape(" ", TextStyle(font_family, size, bold, italic)).font

__all__ = [name for name in globals() if not name.startswith("__")]
