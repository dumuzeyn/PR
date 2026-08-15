from __future__ import annotations

from time import perf_counter
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uzyro.core import BrushSettings, CloneHealingStroke, Layer, SourceTransform, build_patch_edit, spot_heal


WIDTH, HEIGHT = 3840, 2160


def scene() -> np.ndarray:
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    pixels = np.empty((HEIGHT, WIDTH, 4), dtype=np.uint8)
    pixels[:, :, 0] = (xx // 5 + yy // 7) % 256
    pixels[:, :, 1] = (xx // 9 + 70) % 256
    pixels[:, :, 2] = (yy // 6 + 40) % 256
    pixels[:, :, 3] = 255
    return pixels


def timed(label: str, callback) -> None:
    started = perf_counter()
    value = callback()
    elapsed = (perf_counter() - started) * 1000
    suffix = f", tiles={value}" if isinstance(value, int) else ""
    print(f"{label}: {elapsed:.0f} ms{suffix}")


def source_stroke(heal: bool) -> int:
    source = scene()
    layer = Layer("target", source.copy())
    settings = BrushSettings(64, 0.45, 0.75, 0.35, 0.2, 0.2)
    stroke = CloneHealingStroke(
        layer,
        settings,
        source,
        heal=heal,
        transform=SourceTransform(1.2, 0.85, 17, True, False),
    )
    for x in range(900, 2900, 22):
        stroke.dab(x, 1100, x - 500, 720)
    return len(stroke.before_tiles)


def spot_pass() -> None:
    pixels = scene()
    cv2.line(pixels, (1500, 1080), (2300, 1080), (3, 3, 3, 255), 15)
    layer = Layer("spot", pixels)
    for x in range(1550, 2251, 40):
        spot_heal(layer, x, 1080, 28, 0.8, hardness=0.45, mode="content_aware")


def patch_preview() -> None:
    pixels = scene()
    selection = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.ellipse(selection, (2600, 1200), (300, 220), 0, 0, 360, 255, -1)
    result = build_patch_edit(
        pixels,
        (0, 0),
        selection,
        500,
        400,
        source_pixels=pixels,
        structure=6,
        color_adaptation=8,
    )
    assert result is not None


if __name__ == "__main__":
    timed("Clone Stamp 4K", lambda: source_stroke(False))
    timed("Healing Brush 4K", lambda: source_stroke(True))
    timed("Spot Healing 4K", spot_pass)
    timed("Patch preview 600x440 on 4K", patch_preview)
