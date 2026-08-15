from __future__ import annotations

import gc
import time

import numpy as np

from .core import Document, GradientEngine, Layer, draw_brush
from .preview_ops import transform_preview_pixels
from .rendering import RenderEngine


INTERACTIVE_TARGETS_MS = {
    "brush_dab_ms": 16.7,
    "gradient_preview_ms": 60.0,
    "transform_preview_ms": 50.0,
    "many_layers_tile_ms": 50.0,
}


def _elapsed_ms(operation) -> float:
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        started = time.perf_counter()
        operation()
        return (time.perf_counter() - started) * 1000.0
    finally:
        if gc_was_enabled:
            gc.enable()


def _best_ms(operation, repeats: int = 3) -> float:
    """Reduce scheduler noise while still timing complete interactive operations."""
    operation()
    return min(_elapsed_ms(operation) for _ in range(repeats))


def benchmark_interactive_paths() -> dict[str, float | bool]:
    """Exercise representative interactive workloads without changing a document."""
    opaque = np.full((1080, 1920, 4), 255, dtype=np.uint8)
    brush_layer = Layer("Кисть", opaque)
    positions = [(320 + index * 110, 540 + (index % 2) * 40) for index in range(10)]
    brush_ms = _best_ms(
        lambda: [draw_brush(brush_layer, x, y, 192, (35, 145, 235, 255), 0.45) for x, y in positions]
    ) / len(positions)

    gradient_ms = _best_ms(
        lambda: GradientEngine.render(
            1200, 800, (80, 120), (1080, 690),
            [(0.0, (30, 90, 220, 255)), (0.55, (245, 180, 40, 255)), (1.0, (250, 250, 250, 255))],
            "radial",
        )
    )

    source = np.full((1800, 2400, 4), (55, 125, 210, 255), dtype=np.uint8)
    points = [
        [x + (35 if row == 1 else -25 if row == 2 else 0), y]
        for row, y in enumerate(np.linspace(0, 1799, 4))
        for x in np.linspace(0, 2399, 4)
    ]
    transform_ms = _best_ms(
        lambda: transform_preview_pixels(
            source, {}, "Сетка", 0.32, (0, 0, 2400, 1800, 0), points, 4, 4,
        )
    )

    document = Document.new(800, 600, (0, 0, 0, 0))
    document.layers = [
        Layer(
            f"Слой {index + 1}",
            np.full((600, 800, 4), (20 + index * 3, 80 + index * 2, 160, 18), dtype=np.uint8),
        )
        for index in range(60)
    ]
    document.active_layer = len(document.layers) - 1
    engine = RenderEngine(tile_size=128)
    engine.render(document, False)
    layer = document.layer
    rect = draw_brush(layer, 400, 300, 48, (240, 70, 45, 255), 0.7)
    assert rect is not None
    engine.invalidate_region(document, rect, layer)
    def render_changed_tile() -> None:
        engine.invalidate_region(document, rect, layer)
        engine.render(document, False)

    layers_ms = _best_ms(render_changed_tile)

    report: dict[str, float | bool] = {
        "brush_dab_ms": brush_ms,
        "gradient_preview_ms": gradient_ms,
        "transform_preview_ms": transform_ms,
        "many_layers_tile_ms": layers_ms,
    }
    report["passed"] = all(float(report[name]) <= target for name, target in INTERACTIVE_TARGETS_MS.items())
    return report
