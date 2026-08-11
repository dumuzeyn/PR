from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .brush_engine import BrushSettings, StrokeBuffer
from .layer import Layer
from .render_ops import retouch_falloff_mask


@dataclass(frozen=True)
class SourceTransform:
    scale_x: float = 1.0
    scale_y: float = 1.0
    rotation: float = 0.0
    flip_horizontal: bool = False
    flip_vertical: bool = False

    def normalized(self) -> "SourceTransform":
        return SourceTransform(
            max(0.05, min(20.0, abs(float(self.scale_x)))),
            max(0.05, min(20.0, abs(float(self.scale_y)))),
            float(self.rotation) % 360.0,
            bool(self.flip_horizontal),
            bool(self.flip_vertical),
        )

    def inverse_matrix(self) -> np.ndarray:
        value = self.normalized()
        angle = math.radians(value.rotation)
        cosine, sine = math.cos(angle), math.sin(angle)
        flip_x = -1.0 if value.flip_horizontal else 1.0
        flip_y = -1.0 if value.flip_vertical else 1.0
        return np.asarray(
            [
                [cosine * flip_x / value.scale_x, sine * flip_x / value.scale_x],
                [-sine * flip_y / value.scale_y, cosine * flip_y / value.scale_y],
            ],
            dtype=np.float32,
        )


def sample_source_patch(
    source_pixels: np.ndarray,
    source_origin: tuple[int, int],
    source_center: tuple[int, int],
    width: int,
    height: int,
    transform: SourceTransform | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a transformed source rectangle and return pixels plus valid coverage."""
    transform = (transform or SourceTransform()).normalized()
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    offsets = np.stack((xx - (width - 1) * 0.5, yy - (height - 1) * 0.5), axis=-1)
    source_offsets = offsets @ transform.inverse_matrix().T
    map_x = source_offsets[:, :, 0] + float(source_center[0] - source_origin[0])
    map_y = source_offsets[:, :, 1] + float(source_center[1] - source_origin[1])
    valid = (
        (map_x >= 0.0)
        & (map_y >= 0.0)
        & (map_x <= source_pixels.shape[1] - 1)
        & (map_y <= source_pixels.shape[0] - 1)
    ).astype(np.float32)
    sampled = cv2.remap(
        source_pixels,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return sampled, valid


class CloneHealingStroke:
    """Tile-backed clone/healing stroke using one immutable sampling surface."""

    def __init__(
        self,
        layer: Layer,
        settings: BrushSettings,
        source_pixels: np.ndarray,
        source_origin: tuple[int, int] = (0, 0),
        *,
        heal: bool = False,
        selection_mask: np.ndarray | None = None,
        transform: SourceTransform | None = None,
    ) -> None:
        self.layer = layer
        self.settings = settings.normalized()
        self.source_pixels = source_pixels
        self.source_origin = source_origin
        self.heal = bool(heal)
        self.transform = (transform or SourceTransform()).normalized()
        self.buffer = StrokeBuffer(layer.pixels, selection_mask)
        self.before_tiles = self.buffer.before_tiles

    def dab(
        self,
        target_x: int,
        target_y: int,
        source_x: int,
        source_y: int,
        pressure: float = 1.0,
    ) -> tuple[int, int, int, int] | None:
        if self.layer.locked:
            return None
        radius, opacity, flow = self.settings.for_pressure(pressure)
        lx, ly = int(target_x) - self.layer.x, int(target_y) - self.layer.y
        height, width = self.layer.pixels.shape[:2]
        x1, y1 = max(0, lx - radius), max(0, ly - radius)
        x2, y2 = min(width, lx + radius + 1), min(height, ly + radius + 1)
        if x1 >= x2 or y1 >= y2 or opacity <= 0.0 or flow <= 0.0:
            return None
        rect = x1, y1, x2, y2
        full_mask = retouch_falloff_mask(radius, self.settings.hardness)
        mx1, my1 = x1 - (lx - radius), y1 - (ly - radius)
        dab_mask = full_mask[my1 : my1 + y2 - y1, mx1 : mx1 + x2 - x1].copy()
        source_center = (
            int(source_x) + x1 + (x2 - x1 - 1) * 0.5 - lx,
            int(source_y) + y1 + (y2 - y1 - 1) * 0.5 - ly,
        )
        sampled, valid = sample_source_patch(
            self.source_pixels,
            self.source_origin,
            source_center,
            x2 - x1,
            y2 - y1,
            self.transform,
        )
        dab_mask *= valid
        if self.buffer.selection_mask is not None:
            dab_mask *= self.buffer.selection_mask[y1:y2, x1:x2].astype(np.float32) / 255.0
        if not np.any(dab_mask > 0.0):
            return None

        self.buffer.capture_before(rect)
        previous = self.buffer.coverage_region(rect)
        self.buffer.add_coverage(rect, dab_mask, opacity, flow)
        coverage = self.buffer.coverage_region(rect)
        incremental = np.divide(
            coverage - previous,
            np.maximum(1.0 - previous, 1e-6),
            out=np.zeros_like(coverage),
            where=coverage > previous,
        )
        if not np.any(incremental > 0.0):
            return None
        current = self.layer.pixels[y1:y2, x1:x2].astype(np.float32)
        edited = sampled.astype(np.float32)
        if self.heal:
            edited = self._heal_patch(edited, current, radius)
        mixed = current * (1.0 - incremental[:, :, None]) + edited * incremental[:, :, None]
        self.layer.pixels[y1:y2, x1:x2] = np.clip(mixed, 0, 255).astype(np.uint8)
        return rect

    @staticmethod
    def _heal_patch(source: np.ndarray, target: np.ndarray, radius: int) -> np.ndarray:
        sigma = max(1.0, min(12.0, radius * 0.22))
        source_rgb = source[:, :, :3]
        target_rgb = target[:, :, :3]
        source_low = cv2.GaussianBlur(source_rgb, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
        target_low = cv2.GaussianBlur(target_rgb, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
        result = source.copy()
        result[:, :, :3] = np.clip(target_low + source_rgb - source_low, 0, 255)
        result[:, :, 3] = target[:, :, 3]
        return result


__all__ = ["CloneHealingStroke", "SourceTransform", "sample_source_patch"]
