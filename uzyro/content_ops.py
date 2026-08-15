from __future__ import annotations

from .core_shared import *
from .layer import Layer
from .geometry_ops import *
from .selection_ops import *
from .filter_ops import *
from .render_ops import *


def _nearest_source_texture(pixels: np.ndarray, source_mask: np.ndarray) -> np.ndarray:
    """Expand allowed source pixels over the image using exact nearest-source labels."""
    allowed = np.asarray(source_mask, dtype=np.uint8) > 0
    if not np.any(allowed):
        return pixels.copy()
    distance_input = np.where(allowed, 0, 255).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(
        distance_input,
        cv2.DIST_L2,
        3,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    source_y, source_x = np.where(allowed)
    source_labels = labels[source_y, source_x]
    table_size = int(labels.max()) + 1
    map_y = np.zeros(table_size, dtype=np.int32)
    map_x = np.zeros(table_size, dtype=np.int32)
    map_y[source_labels] = source_y
    map_x[source_labels] = source_x
    return pixels[map_y[labels], map_x[labels]]

def _adapt_fill_colour(sampled: np.ndarray, reference: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return sampled
    binary = (mask > 0).astype(np.uint8)
    ring = cv2.dilate(binary, np.ones((7, 7), np.uint8), iterations=1).astype(bool) & ~binary.astype(bool)
    inside = binary.astype(bool)
    if np.count_nonzero(ring) < 4 or np.count_nonzero(inside) < 4:
        return sampled
    sample_lab = cv2.cvtColor(sampled[:, :, :3], cv2.COLOR_RGB2LAB).astype(np.float32)
    reference_lab = cv2.cvtColor(reference[:, :, :3], cv2.COLOR_RGB2LAB).astype(np.float32)
    source_values = sample_lab[inside]
    target_values = reference_lab[ring]
    source_mean = source_values.mean(axis=0)
    target_mean = target_values.mean(axis=0)
    source_std = np.maximum(source_values.std(axis=0), 5.0)
    target_std = np.maximum(target_values.std(axis=0), 5.0)
    matched = (sample_lab - source_mean) * (target_std / source_std) + target_mean
    mixed = sample_lab * (1.0 - strength) + matched * strength
    result = sampled.copy()
    result[:, :, :3] = cv2.cvtColor(np.clip(mixed, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return result

def content_aware_fill_variants(
    arr: np.ndarray,
    selection_mask: np.ndarray | None,
    source_mask: np.ndarray | None = None,
    radius: int = 5,
    color_adaptation: float = 0.65,
    rotation_adaptation: bool = True,
    scale_adaptation: bool = True,
    count: int = 3,
    seed: int = 0,
) -> list[np.ndarray]:
    """Create deterministic source-guided fill candidates without changing pixels outside the selection."""
    source = np.ascontiguousarray(arr, dtype=np.uint8)
    if selection_mask is None or not np.any(selection_mask):
        return [source.copy()]
    if source.shape[2] != 4:
        raise ValueError("Content-Aware Fill expects RGBA pixels")
    target_alpha = np.asarray(selection_mask, dtype=np.uint8)
    if target_alpha.shape != source.shape[:2]:
        target_alpha = cv2.resize(target_alpha, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_LINEAR)
    target_binary = target_alpha > 0
    radius = max(1, min(64, int(radius)))
    exclusion = cv2.dilate(target_binary.astype(np.uint8), np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8), iterations=1) > 0
    if source_mask is None:
        allowed = (source[:, :, 3] > 0) & ~exclusion
    else:
        candidate = np.asarray(source_mask, dtype=np.uint8)
        if candidate.shape != source.shape[:2]:
            candidate = cv2.resize(candidate, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_NEAREST)
        allowed = (candidate > 0) & (source[:, :, 3] > 0) & ~target_binary
    if not np.any(allowed):
        allowed = (source[:, :, 3] > 0) & ~target_binary
    if not np.any(allowed):
        return [source.copy()]

    height, width = source.shape[:2]
    center = (width / 2.0, height / 2.0)
    rng = np.random.default_rng(int(seed))
    variants: list[np.ndarray] = []
    count = max(1, min(6, int(count)))
    for index in range(count):
        if index == 0:
            angle = 0.0
            scale = 1.0
        else:
            direction = -1.0 if index % 2 == 0 else 1.0
            angle = direction * (3.5 + 2.0 * index) if rotation_adaptation else 0.0
            scale = 1.0 + direction * (0.025 + 0.012 * index) if scale_adaptation else 1.0
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        matrix[:, 2] += rng.uniform(-radius * 0.35, radius * 0.35, size=2)
        transformed = cv2.warpAffine(
            source,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        transformed_allowed = cv2.warpAffine(
            allowed.astype(np.uint8) * 255,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        sampled = _nearest_source_texture(transformed, transformed_allowed)
        sampled = _adapt_fill_colour(sampled, source, target_binary.astype(np.uint8) * 255, color_adaptation)

        hard_mask = target_binary.astype(np.uint8) * 255
        method = cv2.INPAINT_TELEA if index % 2 == 0 else cv2.INPAINT_NS
        base = source.copy()
        base[:, :, :3] = cv2.inpaint(source[:, :, :3], hard_mask, float(radius), method)
        base[:, :, 3] = cv2.inpaint(source[:, :, 3], hard_mask, float(radius), method)
        distance = cv2.distanceTransform(target_binary.astype(np.uint8), cv2.DIST_L2, 3)
        texture_weight = np.clip(distance / max(1.0, radius * 1.5), 0.0, 1.0) * (0.42 + min(index, 3) * 0.08)
        texture_weight = texture_weight[:, :, None].astype(np.float32)
        candidate = base.astype(np.float32)
        candidate[:, :, :3] = (
            base[:, :, :3].astype(np.float32) * (1.0 - texture_weight)
            + sampled[:, :, :3].astype(np.float32) * texture_weight
        )
        candidate[:, :, 3] = np.maximum(base[:, :, 3], sampled[:, :, 3]).astype(np.float32)
        alpha = (target_alpha.astype(np.float32) / 255.0)[:, :, None]
        output = np.clip(source.astype(np.float32) * (1.0 - alpha) + candidate * alpha, 0, 255).astype(np.uint8)
        output[~target_binary] = source[~target_binary]
        variants.append(np.ascontiguousarray(output))
    return variants

def content_aware_fill(
    arr: np.ndarray,
    selection_mask: np.ndarray | None,
    radius: int = 3,
    source_mask: np.ndarray | None = None,
    color_adaptation: float = 0.65,
    rotation_adaptation: bool = True,
    scale_adaptation: bool = True,
    variant: int = 0,
) -> np.ndarray:
    variants = content_aware_fill_variants(
        arr,
        selection_mask,
        source_mask,
        radius,
        color_adaptation,
        rotation_adaptation,
        scale_adaptation,
        max(1, int(variant) + 1),
    )
    return variants[min(max(0, int(variant)), len(variants) - 1)]

def _expanded_mask(mask: np.ndarray, left: int, top: int, width: int, height: int) -> np.ndarray:
    result = np.zeros((height, width), dtype=np.uint8)
    result[top : top + mask.shape[0], left : left + mask.shape[1]] = mask
    return result

def generative_expand_pixels(
    arr: np.ndarray,
    left: int,
    top: int,
    right: int,
    bottom: int,
    method: str = "content-aware",
) -> np.ndarray:
    """Expand an image with deterministic local edge synthesis and preserve its center exactly."""
    margins = tuple(max(0, int(value)) for value in (top, bottom, left, right))
    if not any(margins):
        return arr.copy()
    mode = str(method).lower().strip()
    border_mode = cv2.BORDER_REPLICATE if mode in {"edge", "extend"} else cv2.BORDER_REFLECT_101
    expanded = cv2.copyMakeBorder(arr, *margins, borderType=border_mode)
    if mode in {"content-aware", "generative", "texture"}:
        outside = np.full(expanded.shape[:2], 255, dtype=np.uint8)
        outside[top : top + arr.shape[0], left : left + arr.shape[1]] = 0
        # Multi-scale local synthesis removes obvious mirrored repetitions while
        # retaining edge colour and texture. The original image is restored below.
        small = cv2.resize(expanded[:, :, :3], None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        texture = cv2.resize(small, (expanded.shape[1], expanded.shape[0]), interpolation=cv2.INTER_CUBIC)
        detail = expanded[:, :, :3].astype(np.float32) - cv2.GaussianBlur(expanded[:, :, :3], (0, 0), 3.0).astype(np.float32)
        synthesized = np.clip(texture.astype(np.float32) + detail * 0.65, 0, 255).astype(np.uint8)
        feather = cv2.GaussianBlur(outside, (0, 0), 6.0).astype(np.float32)[:, :, None] / 255.0
        expanded[:, :, :3] = np.clip(
            expanded[:, :, :3].astype(np.float32) * (1.0 - feather) + synthesized.astype(np.float32) * feather,
            0,
            255,
        ).astype(np.uint8)
        if np.any(arr[:, :, 3]):
            expanded[:, :, 3] = np.where(outside > 0, np.maximum(expanded[:, :, 3], 255), expanded[:, :, 3])
    expanded[top : top + arr.shape[0], left : left + arr.shape[1]] = arr
    return np.ascontiguousarray(expanded)

def frequency_separation(arr: np.ndarray, radius: float = 8.0, texture_strength: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError("Frequency separation expects RGBA pixels")
    radius = max(0.5, float(radius))
    texture_strength = max(0.0, float(texture_strength))
    low = arr.copy()
    low_rgb = cv2.GaussianBlur(arr[:, :, :3], (0, 0), radius).astype(np.float32)
    low[:, :, :3] = np.clip(low_rgb, 0, 255).astype(np.uint8)
    detail = (arr[:, :, :3].astype(np.float32) - low_rgb) * texture_strength
    high = arr.copy()
    high[:, :, :3] = np.clip(127.5 + detail * 0.5, 0, 255).astype(np.uint8)
    return low, high

def portrait_cleanup(
    arr: np.ndarray,
    smoothing: float = 0.35,
    texture: float = 0.7,
    even_tone: float = 0.2,
    redness: float = 0.2,
    selection_mask: np.ndarray | None = None,
    detail_protection: float = 0.75,
) -> np.ndarray:
    smoothing = float(np.clip(smoothing, 0.0, 1.0))
    texture = float(np.clip(texture, 0.0, 1.5))
    even_tone = float(np.clip(even_tone, 0.0, 1.0))
    redness = float(np.clip(redness, 0.0, 1.0))
    detail_protection = float(np.clip(detail_protection, 0.0, 1.0))
    if max(smoothing, even_tone, redness) <= 0.0:
        return arr.copy()

    rgb = arr[:, :, :3]
    height, width = rgb.shape[:2]
    scale = max(1.0, min(height, width) / 320.0)
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    luminance, cr, cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]
    chroma_distance = np.sqrt(((cr - 154.0) / 34.0) ** 2 + ((cb - 108.0) / 27.0) ** 2)
    skin = np.clip((1.55 - chroma_distance) / 0.75, 0.0, 1.0)
    skin *= np.clip((luminance - 8.0) / 24.0, 0.0, 1.0)
    skin *= (arr[:, :, 3].astype(np.float32) / 255.0)
    binary = (skin >= 0.12).astype(np.uint8) * 255
    if not np.any(binary):
        return arr.copy()
    kernel_size = max(3, min(11, int(round(scale * 3.0)) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    skin *= cv2.GaussianBlur(binary, (0, 0), max(0.8, scale * 1.15)).astype(np.float32) / 255.0

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.GaussianBlur(cv2.magnitude(gx, gy), (0, 0), max(0.6, scale * 0.45))
    edge_protection = 1.0 - np.clip(edge / 150.0, 0.0, 0.92) * detail_protection
    skin *= edge_protection
    if selection_mask is not None:
        selected = np.asarray(selection_mask, dtype=np.uint8)
        if selected.shape != (height, width):
            selected = cv2.resize(selected, (width, height), interpolation=cv2.INTER_LINEAR)
        skin *= selected.astype(np.float32) / 255.0

    source = rgb.astype(np.float32)
    diameter = max(5, min(25, int(round(scale * 7.0)) | 1))
    smooth = cv2.bilateralFilter(
        rgb,
        diameter,
        24.0 + smoothing * 58.0,
        max(7.0, scale * (9.0 + smoothing * 13.0)),
    ).astype(np.float32)
    base = cv2.GaussianBlur(source, (0, 0), max(0.7, scale * 0.85))
    detail = source - base
    smoothed = smooth + detail * texture
    cleaned = source * (1.0 - smoothing) + smoothed * smoothing

    if even_tone > 0.0:
        tone = cv2.GaussianBlur(cleaned, (0, 0), scale * (3.5 + even_tone * 6.5))
        cleaned = cleaned * (1.0 - even_tone * 0.45) + tone * (even_tone * 0.45)

    if redness > 0.0:
        green_blue = (cleaned[:, :, 1] + cleaned[:, :, 2]) * 0.5
        excess = np.maximum(0.0, cleaned[:, :, 0] - green_blue)
        cleaned[:, :, 0] -= excess * redness * 0.65

    amount = skin[:, :, None]
    out = arr.copy()
    out[:, :, :3] = np.clip(source * (1.0 - amount) + cleaned * amount, 0, 255).astype(np.uint8)
    return out

def edge_aware_cleanup(arr: np.ndarray, selection_mask: np.ndarray | None, radius: int = 3, strength: float = 0.65) -> np.ndarray:
    if selection_mask is None or not np.any(selection_mask):
        return arr.copy()
    radius = max(1, int(radius))
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return arr.copy()
    mask = (selection_mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    outer = cv2.dilate(mask, kernel)
    inner = cv2.erode(mask, kernel)
    edge_band = cv2.subtract(outer, inner)
    if not np.any(edge_band):
        return arr.copy()
    feather = cv2.GaussianBlur(edge_band, (radius * 2 + 1, radius * 2 + 1), radius).astype(np.float32) / 255.0
    feather = np.clip(feather * strength, 0.0, 1.0)
    out = arr.copy().astype(np.float32)
    diameter = max(3, radius * 2 + 1)
    smooth_rgb = cv2.bilateralFilter(arr[:, :, :3], diameter, 24 + radius * 8, 12 + radius * 4).astype(np.float32)
    smooth_alpha = cv2.bilateralFilter(arr[:, :, 3], diameter, 24 + radius * 8, 12 + radius * 4).astype(np.float32)
    alpha = feather[:, :, None]
    out[:, :, :3] = out[:, :, :3] * (1.0 - alpha) + smooth_rgb * alpha
    out[:, :, 3] = out[:, :, 3] * (1.0 - feather) + smooth_alpha * feather
    return np.clip(out, 0, 255).astype(np.uint8)

def reduce_red_eye(
    arr: np.ndarray,
    selection_mask: np.ndarray | None = None,
    strength: float = 0.85,
    threshold: float = 0.35,
    darken: float = 0.18,
    feather: float = 2.0,
) -> np.ndarray:
    """Neutralize compact red pupils while preserving highlights, alpha, and unselected pixels."""
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError("Red-eye correction expects RGBA pixels")
    strength = float(np.clip(strength, 0.0, 1.0))
    threshold = float(np.clip(threshold, 0.0, 1.0))
    darken = float(np.clip(darken, 0.0, 1.0))
    feather = max(0.0, float(feather))
    if strength <= 0.0:
        return arr.copy()

    source = arr.astype(np.float32)
    r, g, b = source[:, :, 0], source[:, :, 1], source[:, :, 2]
    comparison = np.maximum(g, b)
    dominance = r - comparison
    ratio = r / np.maximum(comparison, 1.0)
    red_fraction = r / np.maximum(r + g + b, 1.0)
    minimum_dominance = 12.0 + threshold * 48.0
    minimum_ratio = 1.18 + threshold * 0.92
    minimum_fraction = 0.43 + threshold * 0.17
    candidate = (
        (r > 55.0)
        & (dominance >= minimum_dominance)
        & (ratio >= minimum_ratio)
        & (red_fraction >= minimum_fraction)
        & (arr[:, :, 3] > 0)
    )
    selection_alpha: np.ndarray | None = None
    if selection_mask is not None:
        selection_alpha = np.asarray(selection_mask, dtype=np.uint8)
        if selection_alpha.shape != arr.shape[:2]:
            selection_alpha = cv2.resize(selection_alpha, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_LINEAR)
        candidate &= selection_alpha > 0
    if not np.any(candidate):
        return arr.copy()

    binary = candidate.astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    accepted = np.zeros_like(binary)
    image_area = float(binary.size)
    minimum_area = max(3, int(round(image_area * 0.00002)))
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        aspect = width / max(1.0, float(height))
        fill = area / max(1.0, float(width * height))
        if area < minimum_area or not 0.24 <= aspect <= 4.2 or fill < 0.08:
            continue
        if selection_mask is None and area > image_area * 0.015:
            continue
        accepted[labels == index] = 255
    if not np.any(accepted):
        return arr.copy()

    radius = max(1, min(9, int(round(min(arr.shape[:2]) / 360.0))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    accepted = cv2.morphologyEx(accepted, cv2.MORPH_CLOSE, kernel)
    soft = accepted.astype(np.float32) / 255.0
    if feather > 0.0:
        soft = cv2.GaussianBlur(soft, (0, 0), feather)
    if selection_alpha is not None:
        soft *= selection_alpha.astype(np.float32) / 255.0
    dominance_weight = np.clip((dominance - minimum_dominance) / max(8.0, 72.0 - minimum_dominance), 0.0, 1.0)
    fraction_weight = np.clip((red_fraction - minimum_fraction) / 0.18, 0.0, 1.0)
    amount_2d = np.clip(soft * dominance_weight * fraction_weight * strength, 0.0, 1.0)
    amount_2d[amount_2d < 0.02] = 0.0
    amount = amount_2d[:, :, None]

    neutral = (g + b) * 0.5
    corrected = source[:, :, :3].copy()
    corrected[:, :, 0] = neutral
    corrected *= 1.0 - darken * soft[:, :, None]
    output = source.copy()
    output[:, :, :3] = source[:, :, :3] * (1.0 - amount) + corrected * amount
    output[:, :, 3] = source[:, :, 3]
    return np.clip(output, 0, 255).astype(np.uint8)

def image_statistics(arr: np.ndarray) -> dict[str, Any]:
    rgb = arr[:, :, :3].astype(np.float32)
    alpha = arr[:, :, 3]
    stats: dict[str, Any] = {
        "width": int(arr.shape[1]),
        "height": int(arr.shape[0]),
        "opaque_pixels": int(np.count_nonzero(alpha)),
        "transparent_pixels": int(alpha.size - np.count_nonzero(alpha)),
        "channels": {},
        "histogram": {},
    }
    for index, name in enumerate(["red", "green", "blue"]):
        channel = rgb[:, :, index]
        stats["channels"][name] = {
            "min": float(channel.min()),
            "max": float(channel.max()),
            "mean": float(channel.mean()),
            "std": float(channel.std()),
        }
        hist, _ = np.histogram(channel, bins=16, range=(0, 255))
        stats["histogram"][name] = [int(v) for v in hist]
    stats["channels"]["alpha"] = {
        "min": int(alpha.min()),
        "max": int(alpha.max()),
        "mean": float(alpha.mean()),
        "std": float(alpha.std()),
    }
    return stats

__all__ = [name for name in globals() if not name.startswith("__")]
