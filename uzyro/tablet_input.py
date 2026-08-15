from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class TabletSample:
    pressure: float = 1.0
    tilt_x: float = 0.0
    tilt_y: float = 0.0
    rotation: float = 0.0
    eraser: bool = False
    pressure_available: bool = False
    tilt_available: bool = False
    rotation_available: bool = False

    def normalized(self) -> "TabletSample":
        return TabletSample(
            pressure=float(np.clip(self.pressure, 0.01, 1.0)),
            tilt_x=float(np.clip(self.tilt_x, -1.0, 1.0)),
            tilt_y=float(np.clip(self.tilt_y, -1.0, 1.0)),
            rotation=float(self.rotation) % 360.0,
            eraser=bool(self.eraser),
            pressure_available=bool(self.pressure_available),
            tilt_available=bool(self.tilt_available),
            rotation_available=bool(self.rotation_available),
        )


class TabletBackend(Protocol):
    def sample(self, event: object) -> TabletSample: ...


def _normalized_pressure(raw: object) -> tuple[float, bool]:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 1.0, False
    if value > 1.0:
        value /= 65535.0 if value > 1024.0 else 1024.0
    return float(np.clip(value, 0.01, 1.0)), True


def _normalized_tilt(raw: object) -> tuple[float, bool]:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0, False
    if abs(value) > 1.0:
        value /= 90.0 if abs(value) <= 90.0 else 32767.0
    return float(np.clip(value, -1.0, 1.0)), True


class EventTabletBackend:
    """Fallback adapter for GUI events; native Ink/Wintab backends can replace it."""

    def sample(self, event: object) -> TabletSample:
        pressure_raw = next((getattr(event, name) for name in ("pressure", "force") if getattr(event, name, None) is not None), None)
        pressure, pressure_available = _normalized_pressure(pressure_raw)
        tilt_x, tilt_x_available = _normalized_tilt(getattr(event, "tilt_x", getattr(event, "tiltX", None)))
        tilt_y, tilt_y_available = _normalized_tilt(getattr(event, "tilt_y", getattr(event, "tiltY", None)))
        rotation_raw = getattr(event, "rotation", getattr(event, "barrel_rotation", None))
        try:
            rotation = float(rotation_raw)
            rotation_available = True
        except (TypeError, ValueError):
            rotation, rotation_available = 0.0, False
        eraser = bool(getattr(event, "eraser", False) or getattr(event, "is_eraser", False))
        return TabletSample(
            pressure, tilt_x, tilt_y, rotation, eraser,
            pressure_available, tilt_x_available or tilt_y_available, rotation_available,
        ).normalized()


class TabletInputRouter:
    def __init__(self, backend: TabletBackend | None = None) -> None:
        self.backend: TabletBackend = backend or EventTabletBackend()

    def set_backend(self, backend: TabletBackend | None) -> None:
        self.backend = backend or EventTabletBackend()

    def sample(self, event: object) -> TabletSample:
        try:
            return self.backend.sample(event).normalized()
        except Exception:
            return EventTabletBackend().sample(event)


DEFAULT_TABLET_INPUT = TabletInputRouter()


def pointer_input(event: object) -> TabletSample:
    return DEFAULT_TABLET_INPUT.sample(event)


def pointer_pressure(event: object) -> float:
    return pointer_input(event).pressure


__all__ = [
    "DEFAULT_TABLET_INPUT", "EventTabletBackend", "TabletBackend", "TabletInputRouter",
    "TabletSample", "pointer_input", "pointer_pressure",
]
