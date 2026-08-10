from __future__ import annotations

from .core_shared import *
from .layer import Layer
from .geometry_ops import *
from .selection_ops import *
from .render_ops import *


class RetouchStroke:
    def __init__(
        self,
        layer: Layer,
        mode: str,
        radius: int,
        hardness: float,
        strength: float,
        tonal_range: str = "midtones",
        selection_mask: np.ndarray | None = None,
        tile_size: int = 128,
    ) -> None:
        self.layer = layer
        self.mode = mode
        self.radius = max(1, int(radius))
        self.hardness = float(np.clip(hardness, 0.0, 1.0))
        self.strength = float(np.clip(strength, 0.0, 1.0))
        self.tonal_range = tonal_range
        self.selection_mask = selection_mask
        self.tile_size = max(32, int(tile_size))
        self.before_tiles: dict[tuple[int, int], tuple[tuple[int, int, int, int], np.ndarray]] = {}
        self.coverage_tiles: dict[tuple[int, int], np.ndarray] = {}

    def tile_keys(self, rect: tuple[int, int, int, int]):
        x1, y1, x2, y2 = rect
        if x1 >= x2 or y1 >= y2:
            return
        for ty in range(y1 // self.tile_size, (y2 - 1) // self.tile_size + 1):
            for tx in range(x1 // self.tile_size, (x2 - 1) // self.tile_size + 1):
                yield tx, ty

    def capture_before(self, rect: tuple[int, int, int, int]) -> None:
        height, width = self.layer.pixels.shape[:2]
        for key in self.tile_keys(rect):
            if key in self.before_tiles:
                continue
            tx, ty = key
            x1, y1 = tx * self.tile_size, ty * self.tile_size
            x2, y2 = min(width, x1 + self.tile_size), min(height, y1 + self.tile_size)
            tile_rect = x1, y1, x2, y2
            self.before_tiles[key] = tile_rect, self.layer.pixels[y1:y2, x1:x2].copy()

    def original_region(self, rect: tuple[int, int, int, int]) -> np.ndarray:
        self.capture_before(rect)
        x1, y1, x2, y2 = rect
        result = self.layer.pixels[y1:y2, x1:x2].copy()
        for key in self.tile_keys(rect):
            tile_rect, before = self.before_tiles[key]
            tx1, ty1, tx2, ty2 = tile_rect
            ix1, iy1 = max(x1, tx1), max(y1, ty1)
            ix2, iy2 = min(x2, tx2), min(y2, ty2)
            result[iy1 - y1 : iy2 - y1, ix1 - x1 : ix2 - x1] = before[iy1 - ty1 : iy2 - ty1, ix1 - tx1 : ix2 - tx1]
        return result

    def merge_coverage(self, rect: tuple[int, int, int, int], dab: np.ndarray) -> None:
        x1, y1, x2, y2 = rect
        for key in self.tile_keys(rect):
            tile_rect, _before = self.before_tiles[key]
            tx1, ty1, tx2, ty2 = tile_rect
            coverage = self.coverage_tiles.get(key)
            if coverage is None:
                coverage = np.zeros((ty2 - ty1, tx2 - tx1), dtype=np.float32)
                self.coverage_tiles[key] = coverage
            ix1, iy1 = max(x1, tx1), max(y1, ty1)
            ix2, iy2 = min(x2, tx2), min(y2, ty2)
            target = coverage[iy1 - ty1 : iy2 - ty1, ix1 - tx1 : ix2 - tx1]
            source = dab[iy1 - y1 : iy2 - y1, ix1 - x1 : ix2 - x1]
            np.maximum(target, source, out=target)

    def coverage_region(self, rect: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = rect
        result = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
        for key in self.tile_keys(rect):
            coverage = self.coverage_tiles.get(key)
            if coverage is None:
                continue
            tile_rect, _before = self.before_tiles[key]
            tx1, ty1, tx2, ty2 = tile_rect
            ix1, iy1 = max(x1, tx1), max(y1, ty1)
            ix2, iy2 = min(x2, tx2), min(y2, ty2)
            result[iy1 - y1 : iy2 - y1, ix1 - x1 : ix2 - x1] = coverage[iy1 - ty1 : iy2 - ty1, ix1 - tx1 : ix2 - tx1]
        return result

    @profiled("retouch.stroke_dab")
    def dab(self, x: int, y: int) -> tuple[int, int, int, int] | None:
        if self.layer.locked or self.strength <= 0.0:
            return None
        lx, ly = int(x) - self.layer.x, int(y) - self.layer.y
        height, width = self.layer.pixels.shape[:2]
        radius = self.radius
        x1, y1 = max(0, lx - radius), max(0, ly - radius)
        x2, y2 = min(width, lx + radius + 1), min(height, ly + radius + 1)
        if x1 >= x2 or y1 >= y2:
            return None
        full_mask = retouch_falloff_mask(radius, self.hardness)
        mx1, my1 = x1 - (lx - radius), y1 - (ly - radius)
        dab = full_mask[my1 : my1 + (y2 - y1), mx1 : mx1 + (x2 - x1)].copy()
        if self.selection_mask is not None:
            dab *= self.selection_mask[y1:y2, x1:x2].astype(np.float32) / 255.0
        if not np.any(dab > 0.0):
            return None
        halo = retouch_effect_halo(self.mode, self.strength)
        source_rect = max(0, x1 - halo), max(0, y1 - halo), min(width, x2 + halo), min(height, y2 + halo)
        self.capture_before(source_rect)
        self.merge_coverage((x1, y1, x2, y2), dab)
        source = self.original_region(source_rect)
        edited = retouch_effect_rgb(source, self.mode, self.strength, self.tonal_range)
        sx, sy = x1 - source_rect[0], y1 - source_rect[1]
        edited = edited[sy : sy + (y2 - y1), sx : sx + (x2 - x1)]
        original = self.original_region((x1, y1, x2, y2))
        mix = self.coverage_region((x1, y1, x2, y2)) * self.strength
        output = original.copy()
        output[:, :, :3] = np.clip(original[:, :, :3].astype(np.float32) * (1.0 - mix[:, :, None]) + edited * mix[:, :, None], 0, 255).astype(np.uint8)
        self.layer.pixels[y1:y2, x1:x2] = output
        return x1, y1, x2, y2

@profiled("stroke.brush_dab")
def draw_brush(layer: Layer, x: int, y: int, radius: int, color: tuple[int, int, int, int], opacity: float = 1.0, erase=False, selection_mask: np.ndarray | None = None) -> tuple[int, int, int, int] | None:
    if layer.locked:
        return None
    lx, ly = x - layer.x, y - layer.y
    if lx < -radius or ly < -radius or lx >= layer.pixels.shape[1] + radius or ly >= layer.pixels.shape[0] + radius:
        return None
    x1 = max(0, lx - radius)
    y1 = max(0, ly - radius)
    x2 = min(layer.pixels.shape[1], lx + radius + 1)
    y2 = min(layer.pixels.shape[0], ly + radius + 1)
    full_mask = brush_mask(radius)
    mx1 = x1 - (lx - radius)
    my1 = y1 - (ly - radius)
    mask = full_mask[my1 : my1 + (y2 - y1), mx1 : mx1 + (x2 - x1)]
    coverage = mask.astype(np.float32)
    if selection_mask is not None:
        coverage *= selection_mask[y1:y2, x1:x2].astype(np.float32) / 255.0
        if not np.any(coverage > 0.0):
            return None
    target = layer.pixels[y1:y2, x1:x2]
    if erase:
        target[:, :, 3] = np.clip(target[:, :, 3].astype(np.float32) * (1.0 - float(opacity) * coverage), 0, 255).astype(np.uint8)
    else:
        paint = np.array(color, dtype=np.float32)
        dst = target.astype(np.float32)
        original = dst.copy()
        sa = (paint[3] / 255.0) * float(opacity) * coverage
        da = dst[:, :, 3] / 255.0
        oa = sa + da * (1.0 - sa)
        dst[:, :, :3] = np.where(
            oa[:, :, None] > 0,
            (paint[:3] * sa[:, :, None] + dst[:, :, :3] * da[:, :, None] * (1.0 - sa[:, :, None])) / np.maximum(oa[:, :, None], 1e-6),
            0,
        )
        dst[:, :, 3] = oa * 255
        dst = np.where(coverage[:, :, None] > 0.0, dst, original)
        target[:] = np.clip(dst, 0, 255).astype(np.uint8)
    return x1, y1, x2, y2

@profiled("stroke.mask_dab")
def draw_mask_brush(layer: Layer, x: int, y: int, radius: int, value: int, opacity: float = 1.0, selection_mask: np.ndarray | None = None) -> tuple[int, int, int, int] | None:
    if layer.locked:
        return None
    if layer.mask is None:
        layer.mask = np.full(layer.pixels.shape[:2], 255, dtype=np.uint8)
        layer.mask_enabled = True
    lx, ly = x - layer.x, y - layer.y
    if lx < -radius or ly < -radius or lx >= layer.mask.shape[1] + radius or ly >= layer.mask.shape[0] + radius:
        return None
    x1 = max(0, lx - radius)
    y1 = max(0, ly - radius)
    x2 = min(layer.mask.shape[1], lx + radius + 1)
    y2 = min(layer.mask.shape[0], ly + radius + 1)
    full_mask = brush_mask(radius)
    mx1 = x1 - (lx - radius)
    my1 = y1 - (ly - radius)
    mask = full_mask[my1 : my1 + (y2 - y1), mx1 : mx1 + (x2 - x1)]
    coverage = mask.astype(np.float32)
    if selection_mask is not None:
        coverage *= selection_mask[y1:y2, x1:x2].astype(np.float32) / 255.0
        if not np.any(coverage > 0.0):
            return None
    target = layer.mask[y1:y2, x1:x2].astype(np.float32)
    mix = coverage * float(opacity)
    target = target * (1.0 - mix) + int(value) * mix
    layer.mask[y1:y2, x1:x2] = np.clip(target, 0, 255).astype(np.uint8)
    return x1, y1, x2, y2

@profiled("retouch.local_dab")
def local_retouch(
    layer: Layer,
    x: int,
    y: int,
    radius: int,
    mode: str,
    strength: float = 0.25,
    selection_mask: np.ndarray | None = None,
    hardness: float = 0.5,
    tonal_range: str = "midtones",
) -> tuple[int, int, int, int] | None:
    stroke = RetouchStroke(layer, mode, radius, hardness, strength, tonal_range, selection_mask)
    return stroke.dab(x, y)

@profiled("retouch.clone_heal_dab")
def clone_or_heal(
    layer: Layer,
    source_x: int,
    source_y: int,
    target_x: int,
    target_y: int,
    radius: int,
    opacity: float = 1.0,
    heal: bool = False,
    selection_mask: np.ndarray | None = None,
    hardness: float = 0.5,
    source_pixels: np.ndarray | None = None,
    source_origin: tuple[int, int] | None = None,
) -> tuple[int, int, int, int] | None:
    if layer.locked:
        return None
    radius = max(1, int(radius))
    source_image = layer.pixels if source_pixels is None else source_pixels
    source_offset = (layer.x, layer.y) if source_origin is None else source_origin
    sx, sy = int(source_x) - int(source_offset[0]), int(source_y) - int(source_offset[1])
    tx, ty = int(target_x) - layer.x, int(target_y) - layer.y
    target_h, target_w = layer.pixels.shape[:2]
    source_h, source_w = source_image.shape[:2]
    ox1 = max(-radius, -tx, -sx)
    oy1 = max(-radius, -ty, -sy)
    ox2 = min(radius + 1, target_w - tx, source_w - sx)
    oy2 = min(radius + 1, target_h - ty, source_h - sy)
    x1, y1, x2, y2 = tx + ox1, ty + oy1, tx + ox2, ty + oy2
    if x1 >= x2 or y1 >= y2:
        return None
    sx1 = sx + ox1
    sy1 = sy + oy1
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)
    full_mask = retouch_falloff_mask(radius, hardness)
    mx1 = ox1 + radius
    my1 = oy1 + radius
    mask = full_mask[my1 : my1 + (y2 - y1), mx1 : mx1 + (x2 - x1)].copy()
    if selection_mask is not None:
        mask *= selection_mask[y1:y2, x1:x2].astype(np.float32) / 255.0
        if not np.any(mask > 0.0):
            return None
    src = source_image[sy1:sy2, sx1:sx2].astype(np.float32)
    dst = layer.pixels[y1:y2, x1:x2].astype(np.float32)
    edited = src.copy()
    if heal:
        sigma = max(1.0, min(8.0, radius * 0.22))
        source_low = cv2.GaussianBlur(src[:, :, :3], (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
        target_low = cv2.GaussianBlur(dst[:, :, :3], (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
        edited[:, :, :3] = np.clip(target_low + (src[:, :, :3] - source_low), 0, 255)
        edited[:, :, 3] = dst[:, :, 3]
    mix = mask * float(np.clip(opacity, 0.0, 1.0))
    dst = dst * (1.0 - mix[:, :, None]) + edited * mix[:, :, None]
    layer.pixels[y1:y2, x1:x2] = np.clip(dst, 0, 255).astype(np.uint8)
    return x1, y1, x2, y2

@profiled("retouch.spot_heal_dab")
def spot_heal(
    layer: Layer,
    x: int,
    y: int,
    radius: int,
    strength: float = 1.0,
    selection_mask: np.ndarray | None = None,
    hardness: float = 0.45,
) -> tuple[int, int, int, int] | None:
    if layer.locked:
        return None
    radius = max(2, int(radius))
    lx, ly = int(x) - layer.x, int(y) - layer.y
    margin = max(5, radius)
    x1 = max(0, lx - radius - margin)
    y1 = max(0, ly - radius - margin)
    x2 = min(layer.pixels.shape[1], lx + radius + margin + 1)
    y2 = min(layer.pixels.shape[0], ly + radius + margin + 1)
    if x1 >= x2 or y1 >= y2:
        return None

    falloff_full = retouch_falloff_mask(radius, hardness)
    target_left, target_top = lx - radius, ly - radius
    fx1, fy1 = max(x1, target_left), max(y1, target_top)
    fx2, fy2 = min(x2, lx + radius + 1), min(y2, ly + radius + 1)
    falloff = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
    if fx1 < fx2 and fy1 < fy2:
        falloff[fy1 - y1 : fy2 - y1, fx1 - x1 : fx2 - x1] = falloff_full[
            fy1 - target_top : fy2 - target_top,
            fx1 - target_left : fx2 - target_left,
        ]
    if selection_mask is not None:
        falloff *= selection_mask[y1:y2, x1:x2].astype(np.float32) / 255.0
    target_mask = (falloff > 0.04).astype(np.uint8) * 255
    if not np.any(target_mask > 0):
        return None

    patch = layer.pixels[y1:y2, x1:x2].copy()
    inpaint_radius = max(2.0, min(12.0, radius * 0.55))
    healed_rgb = cv2.inpaint(patch[:, :, :3], target_mask, inpaint_radius, cv2.INPAINT_TELEA)
    feather = falloff * float(np.clip(strength, 0.0, 1.0))
    mixed = patch[:, :, :3].astype(np.float32) * (1.0 - feather[:, :, None]) + healed_rgb.astype(np.float32) * feather[:, :, None]
    layer.pixels[y1:y2, x1:x2, :3] = np.clip(mixed, 0, 255).astype(np.uint8)
    return x1, y1, x2, y2

def flood_fill(layer: Layer, x: int, y: int, color: tuple[int, int, int, int], tolerance: int, selection_mask: np.ndarray | None = None) -> None:
    if layer.locked:
        return
    lx, ly = x - layer.x, y - layer.y
    if lx < 0 or ly < 0 or lx >= layer.pixels.shape[1] or ly >= layer.pixels.shape[0]:
        return
    img = layer.pixels.copy()
    seed = img[ly, lx].astype(np.int16)
    diff = np.abs(img.astype(np.int16) - seed).max(axis=2)
    mask = (diff <= tolerance).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 4)
    label = labels[ly, lx]
    if label == 0 and mask[ly, lx] == 0:
        return
    region = labels == label
    if selection_mask is not None:
        coverage = (selection_mask.astype(np.float32) / 255.0) * region.astype(np.float32)
        target = layer.pixels.astype(np.float32)
        paint = np.array(color, dtype=np.float32)
        layer.pixels[:] = np.clip(target * (1.0 - coverage[:, :, None]) + paint * coverage[:, :, None], 0, 255).astype(np.uint8)
    else:
        layer.pixels[region] = np.array(color, dtype=np.uint8)

def apply_gradient(
    layer: Layer,
    vector: tuple[int, int, int, int],
    start: tuple[int, int, int, int],
    end: tuple[int, int, int, int],
    selection_mask: np.ndarray | None = None,
    kind: str = "linear",
    stops: list[Any] | None = None,
) -> None:
    if layer.locked:
        return
    x1, y1, x2, y2 = (int(value) for value in vector)
    gradient_stops = stops or [(0.0, start), (1.0, end)]
    height, width = layer.pixels.shape[:2]
    patch = GradientEngine.render(width, height, (x1, y1), (x2, y2), gradient_stops, kind, (layer.x, layer.y))
    if selection_mask is None:
        layer.pixels[:] = patch
    else:
        coverage = selection_mask.astype(np.float32) / 255.0
        target = layer.pixels.astype(np.float32)
        layer.pixels[:] = np.clip(
            target * (1.0 - coverage[:, :, None]) + patch.astype(np.float32) * coverage[:, :, None],
            0,
            255,
        ).astype(np.uint8)

def add_text(layer: Layer, x: int, y: int, text: str, color: tuple[int, int, int, int], size: int, selection_mask: np.ndarray | None = None) -> None:
    if layer.locked:
        return
    pil = rgba_array_to_pil(layer.pixels)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    draw.text((x - layer.x, y - layer.y), text, fill=color, font=font)
    rendered = pil_to_rgba_array(pil)
    if selection_mask is None:
        layer.pixels = rendered
    else:
        coverage = selection_mask.astype(np.float32) / 255.0
        layer.pixels[:] = np.clip(
            layer.pixels.astype(np.float32) * (1.0 - coverage[:, :, None]) + rendered.astype(np.float32) * coverage[:, :, None],
            0,
            255,
        ).astype(np.uint8)

__all__ = [name for name in globals() if not name.startswith("__")]
