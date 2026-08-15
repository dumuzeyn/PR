from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .tablet_input import TabletSample


CONTROL_SOURCES = ("Off", "Pen Pressure", "Pen Tilt", "Pen Rotation")


@dataclass(frozen=True)
class BrushDab:
    x: int
    y: int
    pressure: float = 1.0
    radius: int | None = None
    angle: float = 0.0
    roundness: float = 1.0
    opacity_factor: float = 1.0
    flow_factor: float = 1.0
    color_mix: float = 0.0
    hue_shift: float = 0.0
    saturation_factor: float = 1.0
    brightness_factor: float = 1.0


def as_tablet_sample(value: float | TabletSample) -> TabletSample:
    if isinstance(value, TabletSample):
        return value.normalized()
    return TabletSample(pressure=float(value), pressure_available=True).normalized()


def control_value(source: str, sample: TabletSample) -> float:
    if source == "Pen Pressure" and sample.pressure_available:
        return sample.pressure
    if source == "Pen Tilt" and sample.tilt_available:
        return min(1.0, math.hypot(sample.tilt_x, sample.tilt_y))
    if source == "Pen Rotation" and sample.rotation_available:
        return (sample.rotation % 360.0) / 360.0
    return 1.0


def _jitter(rng: np.random.Generator, amount: float, control: float) -> float:
    return (float(rng.random()) * 2.0 - 1.0) * float(np.clip(amount, 0.0, 1.0)) * control


def dynamic_dabs(
    settings: Any,
    point: tuple[float, float],
    sample: TabletSample,
    index: int,
    tangent: tuple[float, float] = (1.0, 0.0),
) -> list[BrushDab]:
    rng = np.random.default_rng(int(getattr(settings, "random_seed", 0)) + int(index) * 104729)
    size_control = control_value(getattr(settings, "size_control", "Off"), sample)
    size_scale = 1.0 - float(rng.random()) * float(getattr(settings, "size_jitter", 0.0)) * size_control
    size_scale = max(float(getattr(settings, "minimum_diameter", 0.01)), size_scale)
    if bool(getattr(settings, "pressure_size", False)):
        size_scale *= sample.pressure
    radius = max(1, round(float(settings.radius) * size_scale))

    angle_control = control_value(getattr(settings, "angle_control", "Off"), sample)
    angle = float(getattr(settings, "angle", 0.0)) + _jitter(rng, getattr(settings, "angle_jitter", 0.0), angle_control) * 180.0
    if getattr(settings, "angle_control", "Off") == "Pen Rotation" and sample.rotation_available:
        angle += sample.rotation
    round_control = control_value(getattr(settings, "roundness_control", "Off"), sample)
    roundness = float(getattr(settings, "roundness", 1.0)) * (1.0 - float(rng.random()) * float(getattr(settings, "roundness_jitter", 0.0)) * round_control)
    roundness = float(np.clip(roundness, getattr(settings, "minimum_roundness", 0.01), 1.0))

    opacity_control = sample.pressure if bool(getattr(settings, "pressure_opacity", False)) else 1.0
    flow_control = sample.pressure if bool(getattr(settings, "pressure_flow", False)) else 1.0
    opacity = opacity_control * (1.0 + _jitter(rng, getattr(settings, "opacity_jitter", 0.0), 1.0))
    flow = flow_control * (1.0 + _jitter(rng, getattr(settings, "flow_jitter", 0.0), 1.0))
    opacity = float(np.clip(opacity, getattr(settings, "minimum_opacity", 0.0), 1.0))
    flow = float(np.clip(flow, getattr(settings, "minimum_flow", 0.0), 1.0))

    count = max(1, int(getattr(settings, "scatter_count", 1)))
    count_jitter = float(np.clip(getattr(settings, "count_jitter", 0.0), 0.0, 1.0))
    count = max(1, round(count * (1.0 - float(rng.random()) * count_jitter)))
    scatter = max(0.0, float(getattr(settings, "scatter", 0.0))) * radius * 2.0
    length = max(1e-9, math.hypot(*tangent))
    tx, ty = tangent[0] / length, tangent[1] / length
    nx, ny = -ty, tx
    result: list[BrushDab] = []
    for _ in range(count):
        across = _jitter(rng, 1.0, 1.0) * scatter
        along = _jitter(rng, 1.0, 1.0) * scatter if bool(getattr(settings, "scatter_both_axes", False)) else 0.0
        color_mix = float(np.clip(float(rng.random()) * getattr(settings, "foreground_background_jitter", 0.0), 0.0, 1.0))
        result.append(BrushDab(
            round(point[0] + nx * across + tx * along),
            round(point[1] + ny * across + ty * along),
            sample.pressure,
            radius,
            angle,
            roundness,
            opacity,
            flow,
            color_mix,
            _jitter(rng, getattr(settings, "hue_jitter", 0.0), 1.0),
            max(0.0, 1.0 + _jitter(rng, getattr(settings, "saturation_jitter", 0.0), 1.0)),
            max(0.0, 1.0 + _jitter(rng, getattr(settings, "brightness_jitter", 0.0), 1.0)),
        ))
    return result


__all__ = ["BrushDab", "CONTROL_SOURCES", "as_tablet_sample", "control_value", "dynamic_dabs"]
