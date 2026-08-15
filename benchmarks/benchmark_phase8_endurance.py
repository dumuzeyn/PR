from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photoredactor.brush_engine import BrushSettings, PixelBrushStroke
from photoredactor.core import Document
from photoredactor.history import History, PixelTilePatchCommand, TilePatch
from photoredactor.layer_effects import EFFECT_ORDER, LayerEffectsStack


def run_brush_endurance() -> tuple[float, float, int]:
    document = Document.new(1920, 1080, (0, 0, 0, 0))
    layer = document.layer
    history = History(memory_limit_bytes=128 * 1024 * 1024)
    settings = BrushSettings(radius=18, hardness=0.65, opacity=0.85, flow=0.4)
    started = perf_counter()
    for index in range(100):
        color = (235, 70, 45, 255) if index % 2 else (35, 130, 225, 255)
        stroke = PixelBrushStroke(layer, settings, color)
        x = 80 + (index % 14) * 128
        y = 80 + ((index // 14) % 8) * 128
        stroke.dab(x, y)
        patches = [
            TilePatch(rect, before, layer.pixels[rect[1]:rect[3], rect[0]:rect[2]].copy())
            for rect, before in stroke.before_tiles.values()
        ]
        history.push(PixelTilePatchCommand("Мазок", layer.id, patches))
    return (perf_counter() - started) * 1000.0, history.memory_bytes / 1024 / 1024, len(history.undo_stack)


def run_layer_styles() -> float:
    document = Document.new(1600, 1000, (0, 0, 0, 0))
    document.layer.pixels[180:820, 260:1340] = (55, 135, 220, 255)
    document.layer.effects = {
        kind: {**LayerEffectsStack.item(kind), "enabled": True}
        for kind in EFFECT_ORDER
    }
    started = perf_counter()
    LayerEffectsStack.render(document.layer, document.layer.pixels)
    return (perf_counter() - started) * 1000.0


if __name__ == "__main__":
    brush_ms, memory_mb, commands = run_brush_endurance()
    print(f"100 brush strokes: {brush_ms:.0f} ms, history {memory_mb:.1f} MB, commands {commands}")
    print(f"10 layer styles at 1600x1000: {run_layer_styles():.0f} ms")
