from __future__ import annotations

import math

import numpy as np

from photoredactor.core import Document, GradientEngine, Layer, draw_brush, mesh_warp_pixels
from photoredactor.interactive_performance import INTERACTIVE_TARGETS_MS, benchmark_interactive_paths
from photoredactor.performance import PerformanceProfiler
from photoredactor.preview_ops import transform_preview_pixels
from photoredactor.rendering import RenderEngine


def test_fast_opaque_brush_matches_reference_with_rounding_tolerance() -> None:
    rng = np.random.default_rng(9)
    pixels = rng.integers(0, 256, (420, 620, 4), dtype=np.uint8)
    pixels[:, :, 3] = 255
    original = pixels.copy()
    layer = Layer("Кисть", pixels)
    draw_brush(layer, 310, 210, 140, (25, 175, 240, 255), 0.37)

    yy, xx = np.ogrid[-140:141, -140:141]
    mask = xx * xx + yy * yy <= 140 * 140
    expected = original[70:351, 170:451].astype(np.float32)
    amount = 0.37 * mask.astype(np.float32)
    expected[:, :, :3] = expected[:, :, :3] * (1.0 - amount[:, :, None]) + np.array((25, 175, 240)) * amount[:, :, None]
    expected = np.clip(expected, 0, 255).astype(np.uint8)
    difference = np.abs(layer.pixels[70:351, 170:451].astype(np.int16) - expected.astype(np.int16))
    assert int(difference.max()) <= 1


def test_gradient_lookup_matches_direct_interpolation() -> None:
    stops = [(0.0, (10, 30, 220, 255)), (0.42, (245, 180, 20, 210)), (1.0, (250, 250, 250, 0))]
    values = GradientEngine.coordinates(900, 600, (40, 60), (810, 530), "radial")
    actual = GradientEngine.render(900, 600, (40, 60), (810, 530), stops, "radial")
    expected = np.empty_like(actual)
    positions = np.array([stop[0] for stop in stops], dtype=np.float32)
    colors = np.array([stop[1] for stop in stops], dtype=np.float32)
    for channel in range(4):
        expected[:, :, channel] = np.interp(values, positions, colors[:, channel]).astype(np.uint8)
    assert int(np.abs(actual.astype(np.int16) - expected.astype(np.int16)).max()) <= 1


def test_transform_preview_is_bounded_to_screen_resolution() -> None:
    source = np.full((3000, 4000, 4), (40, 120, 210, 255), dtype=np.uint8)
    points = [[x, y] for y in np.linspace(0, 2999, 4) for x in np.linspace(0, 3999, 4)]
    output, _x, _y, scale = transform_preview_pixels(
        source, {}, "Сетка", 0.2, (0, 0, 4000, 3000, 0), points, 4, 4,
    )
    assert math.isclose(scale, 0.2)
    assert output.shape[0] <= 605
    assert output.shape[1] <= 805


def test_active_layer_base_keeps_many_layer_partial_render_exact() -> None:
    profiler = PerformanceProfiler(enabled=True)
    document = Document.new(384, 256, (0, 0, 0, 0))
    document.layers = [
        Layer(str(index), np.full((256, 384, 4), (index * 5, 80, 180, 24), dtype=np.uint8))
        for index in range(40)
    ]
    document.layers[26].clipping = True
    document.layers.append(
        Layer("Коррекция", np.zeros((1, 1, 4), dtype=np.uint8), kind="adjustment", adjustment={"type": "invert"})
    )
    document.active_layer = 24
    engine = RenderEngine(tile_size=128, performance=profiler)
    engine.render(document, True)
    layer = document.layer
    rect = draw_brush(layer, 180, 120, 24, (240, 30, 60, 255), 0.7)
    assert rect is not None
    engine.invalidate_region(document, rect, layer)
    actual = engine.render(document, True)
    np.testing.assert_array_equal(actual, document.composite(True))
    assert profiler.counter("render.active_base_hit") > 0
    assert len(engine._active_bases) == 1


def test_active_layer_base_is_discarded_after_lower_layer_edit() -> None:
    document = Document.new(220, 160, (0, 0, 0, 0))
    document.layers = [Layer(str(index), np.full((160, 220, 4), (index * 30, 80, 160, 80), dtype=np.uint8)) for index in range(5)]
    document.active_layer = 4
    engine = RenderEngine(tile_size=64)
    engine.render(document, False)
    lower = document.layers[1]
    rect = draw_brush(lower, 90, 70, 18, (250, 220, 20, 255), 1.0)
    assert rect is not None
    engine.invalidate_region(document, rect, lower)
    assert not engine._active_bases
    np.testing.assert_array_equal(engine.render(document, False), document.composite(False))


def test_active_layer_base_respects_large_document_memory_limit() -> None:
    document = Document.new(300, 200, (255, 255, 255, 255))
    engine = RenderEngine(tile_size=64)
    engine.scratch.memory_limit = 1024
    engine.render(document, False)
    assert not engine._active_bases


def test_representative_interactive_benchmark_finishes_within_guardrails() -> None:
    report = benchmark_interactive_paths()
    assert set(INTERACTIVE_TARGETS_MS).issubset(report)
    for name, target in INTERACTIVE_TARGETS_MS.items():
        assert float(report[name]) < max(2000.0, target * 20)
