from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photoredactor.core import BrushPathSampler, BrushSettings, Layer, PixelBrushStroke, RetouchStroke


def benchmark_stroke(mode: str, points: list[tuple[int, int]], pixels: np.ndarray) -> tuple[float, int]:
    layer = Layer(mode, pixels.copy())
    settings = BrushSettings(radius=42, hardness=0.45, opacity=0.8, flow=0.3, spacing=0.2, smoothing=0.2)
    sampler = BrushPathSampler(settings)
    dabs = sampler.begin(points[0])
    for point in points[1:]:
        dabs.extend(sampler.extend(point))
    stroke = (
        RetouchStroke(layer, mode, 42, 0.45, 0.25, flow=0.3)
        if mode in {"blur", "sharpen"}
        else PixelBrushStroke(layer, settings, (230, 80, 45, 255), erase=mode == "eraser")
    )
    started = time.perf_counter()
    for dab in dabs:
        stroke.dab(dab.x, dab.y, dab.pressure)
    elapsed = (time.perf_counter() - started) * 1000.0
    return elapsed, len(stroke.before_tiles)


def main() -> None:
    height, width = 2160, 3840
    rng = np.random.default_rng(21)
    pixels = rng.integers(20, 235, (height, width, 4), dtype=np.uint8)
    pixels[:, :, 3] = 255
    points = [(300 + index * 32, 1080 + round(120 * np.sin(index / 5))) for index in range(80)]
    for mode in ("brush", "eraser", "blur", "sharpen"):
        elapsed, tiles = benchmark_stroke(mode, points, pixels)
        print(f"{mode:8s}: {elapsed:8.1f} ms, undo tiles={tiles}")


if __name__ == "__main__":
    main()
