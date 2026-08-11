from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .core_shared import blend_rgb
from .layer import Layer
from .render_ops import retouch_falloff_mask


BRUSH_BLEND_MODES = ("Normal", "Multiply", "Screen", "Overlay", "Darken", "Lighten")


@dataclass(frozen=True)
class BrushSettings:
    radius: float
    hardness: float = 0.5
    opacity: float = 1.0
    flow: float = 1.0
    spacing: float = 0.25
    smoothing: float = 0.0
    blend_mode: str = "Normal"
    pressure_size: bool = False
    pressure_opacity: bool = False
    pressure_flow: bool = False

    def normalized(self) -> "BrushSettings":
        return BrushSettings(
            radius=max(1.0, float(self.radius)),
            hardness=float(np.clip(self.hardness, 0.0, 1.0)),
            opacity=float(np.clip(self.opacity, 0.0, 1.0)),
            flow=float(np.clip(self.flow, 0.0, 1.0)),
            spacing=float(np.clip(self.spacing, 0.01, 1.0)),
            smoothing=float(np.clip(self.smoothing, 0.0, 1.0)),
            blend_mode=self.blend_mode if self.blend_mode in BRUSH_BLEND_MODES else "Normal",
            pressure_size=bool(self.pressure_size),
            pressure_opacity=bool(self.pressure_opacity),
            pressure_flow=bool(self.pressure_flow),
        )

    def for_pressure(self, pressure: float) -> tuple[int, float, float]:
        value = float(np.clip(pressure, 0.01, 1.0))
        radius = self.radius * (value if self.pressure_size else 1.0)
        opacity = self.opacity * (value if self.pressure_opacity else 1.0)
        flow = self.flow * (value if self.pressure_flow else 1.0)
        return max(1, round(radius)), opacity, flow


@dataclass(frozen=True)
class BrushDab:
    x: int
    y: int
    pressure: float = 1.0


def pointer_pressure(event) -> float:
    """Read tablet pressure when Tk exposes it; ordinary pointers return 1."""
    raw = None
    for name in ("pressure", "force"):
        candidate = getattr(event, name, None)
        if candidate is not None:
            raw = candidate
            break
    if raw is None:
        return 1.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 1.0
    if value > 1.0:
        value /= 65535.0 if value > 1024.0 else 1024.0
    return float(np.clip(value, 0.01, 1.0))


class BrushPathSampler:
    """Turn pointer segments into distance-spaced dabs independent of event rate."""

    def __init__(self, settings: BrushSettings) -> None:
        self.settings = settings.normalized()
        self.step = max(1.0, self.settings.radius * 2.0 * self.settings.spacing)
        self._raw: tuple[float, float] | None = None
        self._filtered: tuple[float, float] | None = None
        self._pressure = 1.0
        self._remaining = self.step

    def begin(self, point: tuple[int, int], pressure: float = 1.0) -> list[BrushDab]:
        self._raw = float(point[0]), float(point[1])
        self._filtered = self._raw
        self._pressure = float(np.clip(pressure, 0.01, 1.0))
        self._remaining = self.step
        return [BrushDab(round(self._raw[0]), round(self._raw[1]), self._pressure)]

    def extend(self, point: tuple[int, int], pressure: float = 1.0) -> list[BrushDab]:
        end = float(point[0]), float(point[1])
        end_pressure = float(np.clip(pressure, 0.01, 1.0))
        if self._raw is None:
            return self.begin(point, end_pressure)
        start = self._raw
        start_pressure = self._pressure
        dx, dy = end[0] - start[0], end[1] - start[1]
        distance = math.hypot(dx, dy)
        if distance <= 1e-9:
            self._pressure = end_pressure
            return []
        dabs: list[BrushDab] = []
        travelled = self._remaining
        while travelled <= distance + 1e-9:
            ratio = travelled / distance
            raw = start[0] + dx * ratio, start[1] + dy * ratio
            filtered = self._smooth(raw)
            dab_pressure = start_pressure + (end_pressure - start_pressure) * ratio
            dabs.append(BrushDab(round(filtered[0]), round(filtered[1]), dab_pressure))
            travelled += self.step
        self._remaining = travelled - distance
        self._raw = end
        self._pressure = end_pressure
        return dabs

    def _smooth(self, point: tuple[float, float]) -> tuple[float, float]:
        if self._filtered is None or self.settings.smoothing <= 0.0:
            self._filtered = point
            return point
        follow = max(0.15, 1.0 - self.settings.smoothing * 0.82)
        self._filtered = (
            self._filtered[0] + (point[0] - self._filtered[0]) * follow,
            self._filtered[1] + (point[1] - self._filtered[1]) * follow,
        )
        return self._filtered


class StrokeBuffer:
    """Tile-backed original pixels and cumulative coverage for one stroke."""

    def __init__(self, target: np.ndarray, selection_mask: np.ndarray | None = None, tile_size: int = 128) -> None:
        self.target = target
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
        height, width = self.target.shape[:2]
        for key in self.tile_keys(rect):
            if key in self.before_tiles:
                continue
            tx, ty = key
            x1, y1 = tx * self.tile_size, ty * self.tile_size
            x2, y2 = min(width, x1 + self.tile_size), min(height, y1 + self.tile_size)
            tile_rect = x1, y1, x2, y2
            self.before_tiles[key] = tile_rect, self.target[y1:y2, x1:x2].copy()

    def original_region(self, rect: tuple[int, int, int, int]) -> np.ndarray:
        self.capture_before(rect)
        x1, y1, x2, y2 = rect
        result = self.target[y1:y2, x1:x2].copy()
        for key in self.tile_keys(rect):
            tile_rect, before = self.before_tiles[key]
            tx1, ty1, tx2, ty2 = tile_rect
            ix1, iy1 = max(x1, tx1), max(y1, ty1)
            ix2, iy2 = min(x2, tx2), min(y2, ty2)
            result[iy1 - y1 : iy2 - y1, ix1 - x1 : ix2 - x1] = before[iy1 - ty1 : iy2 - ty1, ix1 - tx1 : ix2 - tx1]
        return result

    def add_coverage(self, rect: tuple[int, int, int, int], dab: np.ndarray, opacity: float, flow: float) -> None:
        self.capture_before(rect)
        x1, y1, x2, y2 = rect
        cap = float(np.clip(opacity, 0.0, 1.0))
        flow_value = float(np.clip(flow, 0.0, 1.0))
        amount = np.clip(dab.astype(np.float32) * flow_value, 0.0, 1.0)
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
            source = amount[iy1 - y1 : iy2 - y1, ix1 - x1 : ix2 - x1]
            if flow_value >= 0.999:
                np.maximum(target, source * cap, out=target)
            else:
                np.maximum(target, target + (cap - target) * source, out=target)

    def coverage_region(self, rect: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = rect
        result = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
        for key in self.tile_keys(rect):
            coverage = self.coverage_tiles.get(key)
            if coverage is None:
                continue
            tx1, ty1, tx2, ty2 = self.before_tiles[key][0]
            ix1, iy1 = max(x1, tx1), max(y1, ty1)
            ix2, iy2 = min(x2, tx2), min(y2, ty2)
            result[iy1 - y1 : iy2 - y1, ix1 - x1 : ix2 - x1] = coverage[iy1 - ty1 : iy2 - ty1, ix1 - tx1 : ix2 - tx1]
        return result


class PixelBrushStroke:
    def __init__(
        self,
        layer: Layer,
        settings: BrushSettings,
        color: tuple[int, int, int, int],
        *,
        erase: bool = False,
        selection_mask: np.ndarray | None = None,
    ) -> None:
        self.layer = layer
        self.settings = settings.normalized()
        self.color = tuple(int(np.clip(value, 0, 255)) for value in color)
        self.erase = bool(erase)
        self.buffer = StrokeBuffer(layer.pixels, selection_mask)
        self.before_tiles = self.buffer.before_tiles

    def dab(self, x: int, y: int, pressure: float = 1.0) -> tuple[int, int, int, int] | None:
        if self.layer.locked:
            return None
        radius, opacity, flow = self.settings.for_pressure(pressure)
        lx, ly = int(x) - self.layer.x, int(y) - self.layer.y
        height, width = self.layer.pixels.shape[:2]
        x1, y1 = max(0, lx - radius), max(0, ly - radius)
        x2, y2 = min(width, lx + radius + 1), min(height, ly + radius + 1)
        if x1 >= x2 or y1 >= y2:
            return None
        full_mask = retouch_falloff_mask(radius, self.settings.hardness)
        mx1, my1 = x1 - (lx - radius), y1 - (ly - radius)
        dab = full_mask[my1 : my1 + y2 - y1, mx1 : mx1 + x2 - x1].copy()
        if self.buffer.selection_mask is not None:
            dab *= self.buffer.selection_mask[y1:y2, x1:x2].astype(np.float32) / 255.0
        if not np.any(dab > 0.0) or opacity <= 0.0 or flow <= 0.0:
            return None
        rect = x1, y1, x2, y2
        self.buffer.add_coverage(rect, dab, opacity, flow)
        coverage = self.buffer.coverage_region(rect)
        original = self.buffer.original_region(rect)
        output = original.copy()
        if self.erase:
            output[:, :, 3] = np.rint(original[:, :, 3].astype(np.float32) * (1.0 - coverage)).astype(np.uint8)
        else:
            dst = original.astype(np.float32)
            paint = np.empty_like(dst)
            paint[:] = np.asarray(self.color, dtype=np.float32)
            blended_rgb = blend_rgb(paint[:, :, :3], dst[:, :, :3], self.settings.blend_mode)
            source_alpha = (self.color[3] / 255.0) * coverage
            dest_alpha = dst[:, :, 3] / 255.0
            out_alpha = source_alpha + dest_alpha * (1.0 - source_alpha)
            output[:, :, :3] = np.clip(
                np.where(
                    out_alpha[:, :, None] > 1e-6,
                    (blended_rgb * source_alpha[:, :, None] + dst[:, :, :3] * dest_alpha[:, :, None] * (1.0 - source_alpha[:, :, None]))
                    / np.maximum(out_alpha[:, :, None], 1e-6),
                    0.0,
                ),
                0,
                255,
            ).astype(np.uint8)
            output[:, :, 3] = np.rint(np.clip(out_alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
        self.layer.pixels[y1:y2, x1:x2] = output
        return rect


class MaskBrushStroke:
    def __init__(self, layer: Layer, settings: BrushSettings, value: int, selection_mask: np.ndarray | None = None) -> None:
        self.layer = layer
        self.settings = settings.normalized()
        self.value = int(np.clip(value, 0, 255))
        if layer.mask is None:
            layer.mask = np.full(layer.pixels.shape[:2], 255, dtype=np.uint8)
            layer.mask_enabled = True
        self.buffer = StrokeBuffer(layer.mask, selection_mask)
        self.before_tiles = self.buffer.before_tiles

    def dab(self, x: int, y: int, pressure: float = 1.0) -> tuple[int, int, int, int] | None:
        radius, opacity, flow = self.settings.for_pressure(pressure)
        lx, ly = int(x) - self.layer.x, int(y) - self.layer.y
        height, width = self.layer.mask.shape[:2]
        x1, y1 = max(0, lx - radius), max(0, ly - radius)
        x2, y2 = min(width, lx + radius + 1), min(height, ly + radius + 1)
        if self.layer.locked or x1 >= x2 or y1 >= y2:
            return None
        full = retouch_falloff_mask(radius, self.settings.hardness)
        mx1, my1 = x1 - (lx - radius), y1 - (ly - radius)
        dab = full[my1 : my1 + y2 - y1, mx1 : mx1 + x2 - x1].copy()
        if self.buffer.selection_mask is not None:
            dab *= self.buffer.selection_mask[y1:y2, x1:x2].astype(np.float32) / 255.0
        if not np.any(dab > 0.0):
            return None
        rect = x1, y1, x2, y2
        self.buffer.add_coverage(rect, dab, opacity, flow)
        coverage = self.buffer.coverage_region(rect)
        original = self.buffer.original_region(rect).astype(np.float32)
        self.layer.mask[y1:y2, x1:x2] = np.rint(original * (1.0 - coverage) + self.value * coverage).astype(np.uint8)
        return rect


__all__ = [name for name in globals() if not name.startswith("__")]
