from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photoredactor.core import Document, Layer, draw_brush
from photoredactor.rendering import RenderEngine


def elapsed_ms(callback, repeats: int = 1) -> float:
    started = time.perf_counter()
    for _ in range(repeats):
        callback()
    return (time.perf_counter() - started) * 1000.0 / repeats


def make_document(width: int, height: int, layer_count: int) -> Document:
    document = Document(width, height, background=(0, 0, 0, 0))
    base = np.zeros((height, width, 4), dtype=np.uint8)
    base[:, :, 3] = 255
    document.layers = [Layer("Layer 1", base)]
    for index in range(1, layer_count):
        pixels = np.zeros((height, width, 4), dtype=np.uint8)
        pixels[:, :, :3] = ((index * 47) % 255, (index * 83) % 255, (index * 131) % 255)
        pixels[:, :, 3] = 20
        document.layers.append(Layer(f"Layer {index + 1}", pixels, opacity=0.4))
    return document


def run_case(width: int, height: int, layer_count: int) -> None:
    document = make_document(width, height, layer_count)
    reference_ms = elapsed_ms(lambda: document.composite(False))
    engine = RenderEngine(tile_size=256)
    initial_ms = elapsed_ms(lambda: engine.render(document, False))
    layer = document.layers[-1]
    local = draw_brush(layer, width // 2, height // 2, 24, (255, 0, 0, 255), 1.0)
    assert local is not None
    x1, y1, x2, y2 = local
    engine.invalidate_region(document, (x1 + layer.x, y1 + layer.y, x2 + layer.x, y2 + layer.y), layer)
    dirty_ms = elapsed_ms(lambda: engine.render(document, False))
    print(
        f"{width}x{height}, layers={layer_count}: "
        f"reference full={reference_ms:.1f} ms, cached full={initial_ms:.1f} ms, "
        f"one dirty tile={dirty_ms:.1f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PhotoRedactor rendering benchmark")
    parser.add_argument("--include-4k", action="store_true", help="also run memory-intensive 4K cases")
    args = parser.parse_args()
    for layer_count in (1, 10):
        run_case(1920, 1080, layer_count)
    if args.include_4k:
        for layer_count in (1, 10):
            run_case(3840, 2160, layer_count)


if __name__ == "__main__":
    main()
