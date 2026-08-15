from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import colorsys
import math

import numpy as np

from .core_shared import blend_rgb
from .brush_dynamics import BrushDab, as_tablet_sample, dynamic_dabs
from .brush_tip import BRUSH_TEXTURE, BRUSH_TIP_CACHE
from .layer import Layer
from .tablet_input import TabletSample, pointer_input, pointer_pressure


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
    angle: float = 0.0
    roundness: float = 1.0
    flip_x: bool = False
    flip_y: bool = False
    custom_tip_path: str = ""
    size_jitter: float = 0.0
    minimum_diameter: float = 0.01
    size_control: str = "Off"
    angle_jitter: float = 0.0
    angle_control: str = "Off"
    roundness_jitter: float = 0.0
    minimum_roundness: float = 0.01
    roundness_control: str = "Off"
    scatter: float = 0.0
    scatter_both_axes: bool = False
    scatter_count: int = 1
    count_jitter: float = 0.0
    opacity_jitter: float = 0.0
    flow_jitter: float = 0.0
    minimum_opacity: float = 0.0
    minimum_flow: float = 0.0
    foreground_background_jitter: float = 0.0
    hue_jitter: float = 0.0
    saturation_jitter: float = 0.0
    brightness_jitter: float = 0.0
    texture_path: str = ""
    texture_scale: float = 1.0
    texture_depth: float = 0.0
    texture_invert: bool = False
    texture_canvas_space: bool = True
    dual_tip_path: str = ""
    smoothing_mode: str = "basic"
    stabilizer_strength: float = 0.5
    stabilizer_window: int = 8
    pulled_string_radius: float = 30.0
    random_seed: int = 0

    def normalized(self) -> "BrushSettings":
        return BrushSettings(
            radius=max(1.0, float(self.radius)),
            hardness=float(np.clip(self.hardness, 0.0, 1.0)),
            opacity=float(np.clip(self.opacity, 0.0, 1.0)),
            flow=float(np.clip(self.flow, 0.0, 1.0)),
            spacing=float(np.clip(self.spacing, 0.0, 1.0)),
            smoothing=float(np.clip(self.smoothing, 0.0, 1.0)),
            blend_mode=self.blend_mode if self.blend_mode in BRUSH_BLEND_MODES else "Normal",
            pressure_size=bool(self.pressure_size),
            pressure_opacity=bool(self.pressure_opacity),
            pressure_flow=bool(self.pressure_flow),
            angle=float(self.angle) % 360.0,
            roundness=float(np.clip(self.roundness, 0.01, 1.0)),
            flip_x=bool(self.flip_x),
            flip_y=bool(self.flip_y),
            custom_tip_path=str(self.custom_tip_path or ""),
            size_jitter=float(np.clip(self.size_jitter, 0.0, 1.0)),
            minimum_diameter=float(np.clip(self.minimum_diameter, 0.01, 1.0)),
            size_control=str(self.size_control),
            angle_jitter=float(np.clip(self.angle_jitter, 0.0, 1.0)),
            angle_control=str(self.angle_control),
            roundness_jitter=float(np.clip(self.roundness_jitter, 0.0, 1.0)),
            minimum_roundness=float(np.clip(self.minimum_roundness, 0.01, 1.0)),
            roundness_control=str(self.roundness_control),
            scatter=float(np.clip(self.scatter, 0.0, 10.0)),
            scatter_both_axes=bool(self.scatter_both_axes),
            scatter_count=max(1, min(16, int(self.scatter_count))),
            count_jitter=float(np.clip(self.count_jitter, 0.0, 1.0)),
            opacity_jitter=float(np.clip(self.opacity_jitter, 0.0, 1.0)),
            flow_jitter=float(np.clip(self.flow_jitter, 0.0, 1.0)),
            minimum_opacity=float(np.clip(self.minimum_opacity, 0.0, 1.0)),
            minimum_flow=float(np.clip(self.minimum_flow, 0.0, 1.0)),
            foreground_background_jitter=float(np.clip(self.foreground_background_jitter, 0.0, 1.0)),
            hue_jitter=float(np.clip(self.hue_jitter, 0.0, 1.0)),
            saturation_jitter=float(np.clip(self.saturation_jitter, 0.0, 1.0)),
            brightness_jitter=float(np.clip(self.brightness_jitter, 0.0, 1.0)),
            texture_path=str(self.texture_path or ""),
            texture_scale=max(0.05, float(self.texture_scale)),
            texture_depth=float(np.clip(self.texture_depth, 0.0, 1.0)),
            texture_invert=bool(self.texture_invert),
            texture_canvas_space=bool(self.texture_canvas_space),
            dual_tip_path=str(self.dual_tip_path or ""),
            smoothing_mode=str(self.smoothing_mode) if str(self.smoothing_mode) in {"basic", "stabilizer", "pulled_string"} else "basic",
            stabilizer_strength=float(np.clip(self.stabilizer_strength, 0.0, 1.0)),
            stabilizer_window=max(2, min(64, int(self.stabilizer_window))),
            pulled_string_radius=max(0.0, float(self.pulled_string_radius)),
            random_seed=int(self.random_seed),
        )

    def for_pressure(self, pressure: float | TabletSample) -> tuple[int, float, float]:
        value = as_tablet_sample(pressure).pressure
        radius = self.radius * (value if self.pressure_size else 1.0)
        opacity = self.opacity * (value if self.pressure_opacity else 1.0)
        flow = self.flow * (value if self.pressure_flow else 1.0)
        return max(1, round(radius)), opacity, flow

class BrushPathSampler:
    """Turn pointer segments into distance-spaced dabs independent of event rate."""

    def __init__(self, settings: BrushSettings) -> None:
        self.settings = settings.normalized()
        requested_step = self.settings.radius * 2.0 * self.settings.spacing
        continuous_step = self.settings.radius * 0.2
        self.step = max(1.0, continuous_step if self.settings.spacing <= 0.0 else requested_step)
        self._raw: tuple[float, float] | None = None
        self._filtered: tuple[float, float] | None = None
        self._sample = TabletSample()
        self._remaining = self.step
        self._stabilizer: deque[tuple[float, float]] = deque(maxlen=self.settings.stabilizer_window)
        self._pulled: tuple[float, float] | None = None
        self._dab_index = 0

    def begin(self, point: tuple[int, int], pressure: float | TabletSample = 1.0) -> list[BrushDab]:
        self._raw = float(point[0]), float(point[1])
        self._filtered = self._raw
        self._sample = as_tablet_sample(pressure)
        self._remaining = self.step
        self._stabilizer.clear()
        self._stabilizer.append(self._raw)
        self._pulled = self._raw
        self._dab_index = 0
        return self._decorate(self._raw, self._sample, (1.0, 0.0))

    def extend(self, point: tuple[int, int], pressure: float | TabletSample = 1.0) -> list[BrushDab]:
        end = float(point[0]), float(point[1])
        end_sample = as_tablet_sample(pressure)
        if self._raw is None:
            return self.begin(point, end_sample)
        start = self._raw
        start_sample = self._sample
        dx, dy = end[0] - start[0], end[1] - start[1]
        distance = math.hypot(dx, dy)
        if distance <= 1e-9:
            self._sample = end_sample
            return []
        dabs: list[BrushDab] = []
        travelled = self._remaining
        while travelled <= distance + 1e-9:
            ratio = travelled / distance
            raw = start[0] + dx * ratio, start[1] + dy * ratio
            filtered = self._smooth(raw)
            if filtered is not None:
                sample = TabletSample(
                    pressure=start_sample.pressure + (end_sample.pressure - start_sample.pressure) * ratio,
                    tilt_x=start_sample.tilt_x + (end_sample.tilt_x - start_sample.tilt_x) * ratio,
                    tilt_y=start_sample.tilt_y + (end_sample.tilt_y - start_sample.tilt_y) * ratio,
                    rotation=start_sample.rotation + (end_sample.rotation - start_sample.rotation) * ratio,
                    eraser=end_sample.eraser,
                    pressure_available=start_sample.pressure_available or end_sample.pressure_available,
                    tilt_available=start_sample.tilt_available or end_sample.tilt_available,
                    rotation_available=start_sample.rotation_available or end_sample.rotation_available,
                ).normalized()
                dabs.extend(self._decorate(filtered, sample, (dx, dy)))
            travelled += self.step
        self._remaining = travelled - distance
        self._raw = end
        self._sample = end_sample
        return dabs

    def _decorate(self, point: tuple[float, float], sample: TabletSample, tangent: tuple[float, float]) -> list[BrushDab]:
        result = dynamic_dabs(self.settings, point, sample, self._dab_index, tangent)
        self._dab_index += 1
        return result

    def _smooth(self, point: tuple[float, float]) -> tuple[float, float] | None:
        mode = self.settings.smoothing_mode
        if mode == "pulled_string":
            if self._pulled is None:
                self._pulled = point
            dx, dy = point[0] - self._pulled[0], point[1] - self._pulled[1]
            distance = math.hypot(dx, dy)
            radius = self.settings.pulled_string_radius
            if distance <= radius:
                return None
            self._pulled = point[0] - dx / distance * radius, point[1] - dy / distance * radius
            self._filtered = self._pulled
            return self._pulled
        if mode == "stabilizer":
            self._stabilizer.append(point)
            average = (
                sum(value[0] for value in self._stabilizer) / len(self._stabilizer),
                sum(value[1] for value in self._stabilizer) / len(self._stabilizer),
            )
            strength = self.settings.stabilizer_strength
            self._filtered = point[0] * (1.0 - strength) + average[0] * strength, point[1] * (1.0 - strength) + average[1] * strength
            return self._filtered
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
        background: tuple[int, int, int, int] | None = None,
        erase: bool = False,
        selection_mask: np.ndarray | None = None,
    ) -> None:
        self.layer = layer
        self.settings = settings.normalized()
        self.color = tuple(int(np.clip(value, 0, 255)) for value in color)
        self.background = tuple(int(np.clip(value, 0, 255)) for value in (background or color))
        self.erase = bool(erase)
        self.buffer = StrokeBuffer(layer.pixels, selection_mask)
        self.before_tiles = self.buffer.before_tiles
        self._dab_index = 0
        self._stroke_origin: tuple[int, int] | None = None

    def dab(
        self,
        x: int,
        y: int,
        pressure: float | TabletSample = 1.0,
        dab_info: BrushDab | None = None,
    ) -> tuple[int, int, int, int] | None:
        if self.layer.locked:
            return None
        if dab_info is None:
            dab_info = dynamic_dabs(self.settings, (x, y), as_tablet_sample(pressure), self._dab_index)[0]
        self._dab_index += 1
        radius = dab_info.radius or max(1, round(self.settings.radius))
        opacity = self.settings.opacity * dab_info.opacity_factor
        flow = self.settings.flow * dab_info.flow_factor
        lx, ly = int(x) - self.layer.x, int(y) - self.layer.y
        height, width = self.layer.pixels.shape[:2]
        x1, y1 = max(0, lx - radius), max(0, ly - radius)
        x2, y2 = min(width, lx + radius + 1), min(height, ly + radius + 1)
        if x1 >= x2 or y1 >= y2:
            return None
        full_mask = BRUSH_TIP_CACHE.stamp(
            radius, self.settings.hardness, dab_info.angle, dab_info.roundness,
            self.settings.flip_x, self.settings.flip_y, self.settings.custom_tip_path, self.settings.dual_tip_path,
        )
        mx1, my1 = x1 - (lx - radius), y1 - (ly - radius)
        dab = full_mask[my1 : my1 + y2 - y1, mx1 : mx1 + x2 - x1].copy()
        if self._stroke_origin is None:
            self._stroke_origin = int(x), int(y)
        dab = BRUSH_TEXTURE.apply(
            dab,
            self.settings.texture_path,
            (x1 + self.layer.x, y1 + self.layer.y, x2 + self.layer.x, y2 + self.layer.y),
            scale=self.settings.texture_scale,
            depth=self.settings.texture_depth,
            invert=self.settings.texture_invert,
            stroke_origin=self._stroke_origin,
            canvas_space=self.settings.texture_canvas_space,
        )
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
            paint_color = self._dynamic_color(dab_info)
            paint[:] = np.asarray(paint_color, dtype=np.float32)
            blended_rgb = blend_rgb(paint[:, :, :3], dst[:, :, :3], self.settings.blend_mode)
            source_alpha = (paint_color[3] / 255.0) * coverage
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

    def _dynamic_color(self, dab: BrushDab) -> tuple[int, int, int, int]:
        mix = float(np.clip(dab.color_mix, 0.0, 1.0))
        rgba = np.asarray(self.color, dtype=np.float32) * (1.0 - mix) + np.asarray(self.background, dtype=np.float32) * mix
        red, green, blue = np.clip(rgba[:3] / 255.0, 0.0, 1.0)
        hue, saturation, value = colorsys.rgb_to_hsv(float(red), float(green), float(blue))
        hue = (hue + dab.hue_shift) % 1.0
        saturation = float(np.clip(saturation * dab.saturation_factor, 0.0, 1.0))
        value = float(np.clip(value * dab.brightness_factor, 0.0, 1.0))
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        return *(round(channel * 255.0) for channel in rgb), round(float(rgba[3]))


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
        self._dab_index = 0
        self._stroke_origin: tuple[int, int] | None = None

    def dab(
        self,
        x: int,
        y: int,
        pressure: float | TabletSample = 1.0,
        dab_info: BrushDab | None = None,
    ) -> tuple[int, int, int, int] | None:
        if dab_info is None:
            dab_info = dynamic_dabs(self.settings, (x, y), as_tablet_sample(pressure), self._dab_index)[0]
        self._dab_index += 1
        radius = dab_info.radius or max(1, round(self.settings.radius))
        opacity = self.settings.opacity * dab_info.opacity_factor
        flow = self.settings.flow * dab_info.flow_factor
        lx, ly = int(x) - self.layer.x, int(y) - self.layer.y
        height, width = self.layer.mask.shape[:2]
        x1, y1 = max(0, lx - radius), max(0, ly - radius)
        x2, y2 = min(width, lx + radius + 1), min(height, ly + radius + 1)
        if self.layer.locked or x1 >= x2 or y1 >= y2:
            return None
        full = BRUSH_TIP_CACHE.stamp(
            radius, self.settings.hardness, dab_info.angle, dab_info.roundness,
            self.settings.flip_x, self.settings.flip_y, self.settings.custom_tip_path, self.settings.dual_tip_path,
        )
        mx1, my1 = x1 - (lx - radius), y1 - (ly - radius)
        dab = full[my1 : my1 + y2 - y1, mx1 : mx1 + x2 - x1].copy()
        if self._stroke_origin is None:
            self._stroke_origin = int(x), int(y)
        dab = BRUSH_TEXTURE.apply(
            dab,
            self.settings.texture_path,
            (x1 + self.layer.x, y1 + self.layer.y, x2 + self.layer.x, y2 + self.layer.y),
            scale=self.settings.texture_scale,
            depth=self.settings.texture_depth,
            invert=self.settings.texture_invert,
            stroke_origin=self._stroke_origin,
            canvas_space=self.settings.texture_canvas_space,
        )
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
