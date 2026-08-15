from __future__ import annotations

from .core_shared import *
from .layer import Layer
from .geometry_ops import *
from .selection_refinement import refine_soft_selection, signed_distance_field


def paste_mask(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    h, w = src.shape[:2]
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(dst.shape[1], int(x) + w), min(dst.shape[0], int(y) + h)
    if x1 >= x2 or y1 >= y2:
        return
    sx1, sy1 = x1 - x, y1 - y
    dst[y1:y2, x1:x2] = src[sy1 : sy1 + (y2 - y1), sx1 : sx1 + (x2 - x1)]

def shifted_mask(mask: np.ndarray, old_width: int, old_height: int, new_width: int, new_height: int, dx: int, dy: int) -> np.ndarray:
    out = np.zeros((new_height, new_width), dtype=np.uint8)
    x1, y1 = max(0, dx), max(0, dy)
    x2, y2 = min(new_width, dx + old_width), min(new_height, dy + old_height)
    if x1 >= x2 or y1 >= y2:
        return out
    sx1, sy1 = x1 - dx, y1 - dy
    out[y1:y2, x1:x2] = mask[sy1 : sy1 + (y2 - y1), sx1 : sx1 + (x2 - x1)]
    return out

def refine_selection_mask(mask: np.ndarray, smooth: int = 0, feather: int = 0, contrast: float = 1.0, shift: int = 0) -> np.ndarray:
    return refine_soft_selection(mask, smooth, feather, contrast, shift)

def refine_layer_mask(
    mask: np.ndarray,
    image: np.ndarray,
    smooth: int = 0,
    feather: int = 0,
    contrast: float = 1.0,
    shift: int = 0,
    edge_radius: int = 0,
    edge_strength: float = 0.0,
    confidence_threshold: int = 96,
) -> np.ndarray:
    out = refine_selection_mask(mask, max(0, int(smooth)), 0, 1.0, int(shift))
    radius = max(0, int(edge_radius))
    strength = float(np.clip(edge_strength, 0.0, 1.0))
    if radius > 0 and strength > 0.0 and np.any(out):
        out = correct_selection_edges(out, image, radius, strength, confidence_threshold)
    return refine_selection_mask(out, 0, max(0, int(feather)), max(0.0, float(contrast)), 0)

def refine_selection_brush(
    mask: np.ndarray,
    image: np.ndarray,
    brush: np.ndarray,
    mode: str = "refine",
    radius: int = 5,
    strength: float = 0.75,
) -> np.ndarray:
    if mask is None or mask.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    if brush.shape != mask.shape:
        raise ValueError("Brush and selection mask must have the same size")
    radius = max(1, int(radius))
    strength = float(np.clip(strength, 0.0, 1.0))
    coverage = brush.astype(np.float32) / 255.0 * strength
    source = mask.astype(np.float32)
    if mode == "add":
        return np.maximum(source, coverage * 255.0).astype(np.uint8)
    if mode == "subtract":
        return np.clip(source * (1.0 - coverage), 0, 255).astype(np.uint8)
    kernel_size = radius * 2 + 1
    if mode == "smooth":
        smoothed = cv2.GaussianBlur(mask, (kernel_size, kernel_size), radius)
        return np.clip(source * (1.0 - coverage) + smoothed.astype(np.float32) * coverage, 0, 255).astype(np.uint8)

    binary = np.where(mask >= 128, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    definite_foreground = cv2.erode(binary, kernel) > 0
    definite_background = cv2.dilate(binary, kernel) == 0
    visible = image[:, :, 3] > 0 if image.shape[2] > 3 else np.ones(mask.shape, dtype=bool)
    definite_foreground &= visible
    definite_background &= visible
    if np.count_nonzero(definite_foreground) < 4 or np.count_nonzero(definite_background) < 4:
        fallback = cv2.GaussianBlur(mask, (kernel_size, kernel_size), radius)
        return np.clip(source * (1.0 - coverage) + fallback.astype(np.float32) * coverage, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(image[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    foreground_color = np.median(lab[definite_foreground], axis=0)
    background_color = np.median(lab[definite_background], axis=0)
    foreground_distance = np.linalg.norm(lab - foreground_color, axis=2)
    background_distance = np.linalg.norm(lab - background_color, axis=2)
    matte = background_distance / np.maximum(1e-6, foreground_distance + background_distance)
    matte = cv2.bilateralFilter((matte * 255.0).astype(np.uint8), 5, 28, 28).astype(np.float32)
    matte[definite_foreground] = 255.0
    matte[definite_background] = 0.0
    boundary = cv2.subtract(cv2.dilate(binary, kernel), cv2.erode(binary, kernel)).astype(np.float32) / 255.0
    blend = coverage * np.maximum(boundary, (brush.astype(np.float32) / 255.0) * 0.35)
    return np.clip(source * (1.0 - blend) + matte * blend, 0, 255).astype(np.uint8)

def decontaminate_edge_colors(pixels: np.ndarray, mask: np.ndarray, strength: float = 0.5, radius: int = 3) -> np.ndarray:
    if pixels.size == 0 or mask.shape != pixels.shape[:2]:
        return pixels.copy()
    strength = float(np.clip(strength, 0.0, 1.0))
    radius = max(1, int(radius))
    out = pixels.copy()
    soft = mask.astype(np.float32) / 255.0
    edge = (soft > 0.02) & (soft < 0.98)
    if not np.any(edge):
        return out
    solid = mask >= 245
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    solid_rgb = np.where(solid[:, :, None], pixels[:, :, :3], 0).astype(np.uint8)
    expanded = np.empty_like(solid_rgb)
    for channel in range(3):
        expanded[:, :, channel] = cv2.dilate(solid_rgb[:, :, channel], kernel)
    edge &= np.any(expanded > 0, axis=2)
    if not np.any(edge):
        return out
    mix = np.clip((1.0 - np.abs(soft - 0.5) * 2.0) * strength, 0.0, 1.0)
    rgb = pixels[:, :, :3].astype(np.float32)
    rgb[edge] = rgb[edge] * (1.0 - mix[edge, None]) + expanded[edge].astype(np.float32) * mix[edge, None]
    out[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return out

def cleanup_selection_edges(mask: np.ndarray, image: np.ndarray, radius: int = 3, strength: float = 0.7) -> np.ndarray:
    if mask is None:
        return np.zeros((0, 0), dtype=np.uint8)
    if not np.any(mask):
        return np.zeros_like(mask, dtype=np.uint8)
    radius = max(1, int(radius))
    strength = float(np.clip(strength, 0.0, 1.0))
    source = np.asarray(mask, dtype=np.uint8)
    binary = np.where(source >= 128, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    soft = refine_soft_selection(cleaned, max(1, radius // 2), max(1, radius // 2), 1.0 + strength, 0).astype(np.float32)

    gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
    gray = np.where(image[:, :, 3] > 0, gray, 0).astype(np.uint8)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 120)
    edge_band = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)))
    boundary = cv2.subtract(cv2.dilate(binary, kernel), cv2.erode(binary, kernel))
    preserve = (edge_band.astype(np.float32) / 255.0) * (boundary.astype(np.float32) / 255.0) * strength

    target = soft * (1.0 - preserve) + source.astype(np.float32) * preserve
    mixed = source.astype(np.float32) * (1.0 - strength) + target * strength
    return np.rint(np.clip(mixed, 0, 255)).astype(np.uint8)

def selection_edge_confidence(mask: np.ndarray, image: np.ndarray, radius: int = 3) -> np.ndarray:
    if mask is None or mask.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    if not np.any(mask):
        return np.zeros_like(mask, dtype=np.uint8)
    radius = max(1, int(radius))
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    boundary = cv2.subtract(cv2.dilate(binary, kernel), cv2.erode(binary, kernel))
    if not np.any(boundary):
        return np.zeros_like(binary, dtype=np.uint8)

    gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
    if image.shape[2] > 3:
        gray = np.where(image[:, :, 3] > 0, gray, 0).astype(np.uint8)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 40, 130)
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    if float(magnitude.max()) > 0.0:
        magnitude = magnitude / float(magnitude.max())
    edge_band = cv2.dilate(edges, kernel).astype(np.float32) / 255.0
    confidence = np.maximum(edge_band, magnitude)
    confidence = cv2.GaussianBlur(confidence, (radius * 2 + 1, radius * 2 + 1), radius)
    return np.where(boundary > 0, np.clip(confidence * 255.0, 0, 255), 0).astype(np.uint8)

def correct_selection_edges(mask: np.ndarray, image: np.ndarray, radius: int = 3, strength: float = 0.65, threshold: int = 96) -> np.ndarray:
    if mask is None or mask.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    if not np.any(mask):
        return np.zeros_like(mask, dtype=np.uint8)
    radius = max(1, int(radius))
    strength = float(np.clip(strength, 0.0, 1.0))
    threshold = int(np.clip(threshold, 0, 255))
    source = np.asarray(mask, dtype=np.uint8)
    binary = np.where(source >= 128, 255, 0).astype(np.uint8)
    confidence = selection_edge_confidence(binary, image, radius)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    outer = cv2.dilate(binary, kernel)
    inner = cv2.erode(binary, kernel)
    boundary = cv2.subtract(outer, inner)
    if not np.any(boundary):
        return source.copy()

    trusted = (confidence >= threshold) & (boundary > 0)
    weak = (confidence < threshold) & (boundary > 0)
    candidate = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
    relaxed = refine_soft_selection(candidate, max(1, radius // 2), max(1, radius // 3), 1.0, 0)
    target = np.where(weak, relaxed, source).astype(np.float32)
    target[trusted] = source[trusted]
    blend = source.astype(np.float32) * (1.0 - strength) + target * strength
    return np.rint(np.clip(blend, 0, 255)).astype(np.uint8)

def smart_radius_refine(mask: np.ndarray, image: np.ndarray, radius: int = 5, strength: float = 0.7) -> np.ndarray:
    """Refine uncertain boundaries while retaining the mask's fractional alpha."""
    radius = max(1, int(radius))
    strength = float(np.clip(strength, 0.0, 1.0))
    source = np.asarray(mask, dtype=np.uint8)
    binary = np.where(source >= 128, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    boundary = cv2.subtract(cv2.dilate(binary, kernel), cv2.erode(binary, kernel))
    if not np.any(boundary):
        return source.copy()
    confidence = selection_edge_confidence(binary, image, radius).astype(np.float32) / 255.0
    adaptive = cv2.bilateralFilter(source, max(3, radius * 2 + 1), 24 + radius * 2, 24 + radius * 2)
    uncertain = (boundary.astype(np.float32) / 255.0) * (1.0 - confidence) * strength
    result = source.astype(np.float32) * (1.0 - uncertain) + adaptive.astype(np.float32) * uncertain
    return np.clip(result, 0, 255).astype(np.uint8)

def subject_selection_mask(pixels: np.ndarray, sensitivity: float = 0.5) -> np.ndarray:
    if pixels.size == 0:
        return np.zeros(pixels.shape[:2], dtype=np.uint8)
    alpha = pixels[:, :, 3]
    visible = alpha > 0
    if not np.any(visible):
        return np.zeros(alpha.shape, dtype=np.uint8)
    coverage = float(np.count_nonzero(visible)) / float(visible.size)
    if coverage < 0.92:
        return alpha.copy()

    sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
    original_height, original_width = alpha.shape
    scale = min(1.0, 900.0 / max(original_height, original_width))
    work_size = (max(2, round(original_width * scale)), max(2, round(original_height * scale)))
    rgb = pixels[:, :, :3].astype(np.uint8)
    work_rgb = rgb if work_size == (original_width, original_height) else cv2.resize(rgb, work_size, interpolation=cv2.INTER_AREA)
    work_visible = visible if work_size == (original_width, original_height) else cv2.resize(visible.astype(np.uint8), work_size, interpolation=cv2.INTER_NEAREST) > 0
    height, width = work_visible.shape
    border_width = max(1, min(height, width) // 80)
    border = np.zeros((height, width), dtype=bool)
    border[:border_width, :] = True
    border[-border_width:, :] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True
    border &= work_visible
    if np.count_nonzero(border) < 8:
        border = work_visible

    lab = cv2.cvtColor(work_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    background_color = np.median(lab[border], axis=0)
    distance = np.linalg.norm(lab - background_color, axis=2)
    border_noise = float(np.percentile(distance[border], 90)) if np.any(border) else 0.0
    threshold = max(7.0, border_noise + 5.0 + (1.0 - sensitivity) * 20.0)
    foreground_seed = (distance >= threshold) & work_visible
    radius = max(1, min(height, width) // 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    strong_foreground = cv2.erode(foreground_seed.astype(np.uint8) * 255, kernel) > 0
    grab_mask = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)
    grab_mask[~work_visible] = cv2.GC_BGD
    grab_mask[border] = cv2.GC_BGD
    grab_mask[foreground_seed] = cv2.GC_PR_FGD
    grab_mask[strong_foreground] = cv2.GC_FGD
    if np.any(strong_foreground) and np.any(grab_mask == cv2.GC_BGD):
        try:
            background_model = np.zeros((1, 65), dtype=np.float64)
            foreground_model = np.zeros((1, 65), dtype=np.float64)
            cv2.grabCut(work_rgb, grab_mask, None, background_model, foreground_model, 4, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            pass
    binary = np.where((grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    if not np.any(binary):
        binary = foreground_seed.astype(np.uint8) * 255

    inner = cv2.erode(binary, kernel) > 0
    outer = cv2.dilate(binary, kernel) == 0
    if np.count_nonzero(inner) >= 4 and np.count_nonzero(outer & work_visible) >= 4:
        foreground_color = np.median(lab[inner], axis=0)
        background_color = np.median(lab[outer & work_visible], axis=0)
        foreground_distance = np.linalg.norm(lab - foreground_color, axis=2)
        background_distance = np.linalg.norm(lab - background_color, axis=2)
        matte = background_distance / np.maximum(1e-6, foreground_distance + background_distance)
        matte = cv2.bilateralFilter((matte * 255.0).astype(np.uint8), 5, 24, 24)
        boundary = cv2.subtract(cv2.dilate(binary, kernel), cv2.erode(binary, kernel)) > 0
        soft = binary.copy()
        soft[boundary] = matte[boundary]
    else:
        soft = binary
    soft = np.minimum(soft, work_visible.astype(np.uint8) * 255)
    if work_size != (original_width, original_height):
        soft = cv2.resize(soft, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    return np.minimum(soft, alpha).astype(np.uint8)

def background_selection_mask(pixels: np.ndarray, sensitivity: float = 0.5) -> np.ndarray:
    if pixels.size == 0:
        return np.zeros(pixels.shape[:2], dtype=np.uint8)
    alpha = pixels[:, :, 3]
    visible = alpha > 0
    if not np.any(visible):
        return np.full(alpha.shape, 255, dtype=np.uint8)
    subject = subject_selection_mask(pixels, sensitivity)
    return np.where(visible, 255 - subject, 255).astype(np.uint8)

def sky_selection_mask(pixels: np.ndarray, sensitivity: float = 0.5) -> np.ndarray:
    if pixels.size == 0:
        return np.zeros(pixels.shape[:2], dtype=np.uint8)
    alpha = pixels[:, :, 3]
    visible = alpha > 0
    if not np.any(visible):
        return np.zeros(alpha.shape, dtype=np.uint8)
    rgb = pixels[:, :, :3].astype(np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    h, w = alpha.shape
    yy = np.arange(h)[:, None]
    upper_weight = yy < max(1, int(h * 0.72))
    sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
    blue_sky = (hue >= 78 + round((1.0 - sensitivity) * 12)) & (hue <= 138 - round((1.0 - sensitivity) * 6)) & (saturation >= max(8, 38 - round(sensitivity * 28))) & (value >= max(45, 95 - round(sensitivity * 45)))
    pale_sky = (saturation < 40 + round(sensitivity * 35)) & (value >= 195 - round(sensitivity * 48)) & (rgb[:, :, 2] >= rgb[:, :, 0] - round(2 + sensitivity * 16))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    top_height = max(2, h // 18)
    top_visible = visible[:top_height]
    continuity = np.zeros((h, w), dtype=bool)
    if np.count_nonzero(top_visible) >= max(8, w // 6):
        top_values = lab[:top_height][top_visible]
        top_color = np.median(top_values, axis=0)
        top_spread = float(np.percentile(np.linalg.norm(top_values - top_color, axis=1), 75))
        if top_spread <= 38.0:
            distance = np.linalg.norm(lab - top_color, axis=2)
            # A clean top strip should produce a strict colour model.  Widen it
            # only when the strip itself contains real variation (clouds/noise).
            continuity_limit = 6.0 + sensitivity * 8.0 + min(20.0, top_spread * 1.6)
            continuity = distance <= continuity_limit
            continuity &= value >= max(8, round(28 - sensitivity * 18))
    candidates = (blue_sky | pale_sky | continuity) & upper_weight & visible
    connected = top_connected_mask(candidates)
    if not np.any(connected):
        return np.zeros(alpha.shape, dtype=np.uint8)
    radius = max(1, min(h, w) // 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, kernel)
    connected = cv2.morphologyEx(connected, cv2.MORPH_OPEN, kernel)
    soft = cv2.GaussianBlur(connected, (radius * 2 + 1, radius * 2 + 1), radius)
    soft[connected == 255] = np.maximum(soft[connected == 255], 220)
    return np.minimum(soft, alpha).astype(np.uint8)

def automatic_selection_mask(pixels: np.ndarray, target: str, sensitivity: float = 0.5) -> np.ndarray:
    normalized = str(target).lower()
    if normalized in {"subject", "object", "объект"}:
        return subject_selection_mask(pixels, sensitivity)
    if normalized in {"background", "фон"}:
        return background_selection_mask(pixels, sensitivity)
    if normalized in {"sky", "небо"}:
        return sky_selection_mask(pixels, sensitivity)
    raise ValueError(f"Unsupported automatic selection target: {target}")

def border_connected_mask(candidates: np.ndarray) -> np.ndarray:
    binary = candidates.astype(np.uint8)
    if binary.size == 0 or not np.any(binary):
        return np.zeros(binary.shape, dtype=np.uint8)
    num, labels, _, _ = cv2.connectedComponentsWithStats(binary, 8)
    border_labels = set(np.unique(labels[0, binary[0, :] > 0]).tolist())
    border_labels.update(np.unique(labels[-1, binary[-1, :] > 0]).tolist())
    border_labels.update(np.unique(labels[binary[:, 0] > 0, 0]).tolist())
    border_labels.update(np.unique(labels[binary[:, -1] > 0, -1]).tolist())
    border_labels.discard(0)
    if num <= 1 or not border_labels:
        return np.zeros(binary.shape, dtype=np.uint8)
    return np.where(np.isin(labels, list(border_labels)), 255, 0).astype(np.uint8)

def top_connected_mask(candidates: np.ndarray) -> np.ndarray:
    binary = candidates.astype(np.uint8)
    if binary.size == 0 or not np.any(binary):
        return np.zeros(binary.shape, dtype=np.uint8)
    num, labels, _, _ = cv2.connectedComponentsWithStats(binary, 8)
    top_labels = set(np.unique(labels[0, binary[0, :] > 0]).tolist())
    top_labels.discard(0)
    if num <= 1 or not top_labels:
        return np.zeros(binary.shape, dtype=np.uint8)
    return np.where(np.isin(labels, list(top_labels)), 255, 0).astype(np.uint8)

def effective_layer_mask(layer: Layer) -> np.ndarray | None:
    if layer.mask is None:
        return None
    mask = layer.mask
    radius = float(getattr(layer, "mask_feather", 0.0))
    if radius <= 0:
        return mask
    k = max(3, int(round(radius)) * 2 + 1)
    return cv2.GaussianBlur(mask, (k, k), radius).astype(np.uint8)

__all__ = [name for name in globals() if not name.startswith("__")]
