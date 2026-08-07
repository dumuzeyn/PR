import numpy as np

from photoredactor.core import Document, Layer, draw_brush, draw_mask_brush
from photoredactor.performance import PerformanceProfiler
from photoredactor.rendering import RenderEngine


def sample_document() -> Document:
    rng = np.random.default_rng(42)
    document = Document.new(321, 239, (0, 0, 0, 0))
    document.layers = []
    for index in range(3):
        pixels = rng.integers(0, 256, (170, 210, 4), dtype=np.uint8)
        pixels[:, :, 3] = rng.integers(0, 256, (170, 210), dtype=np.uint8)
        layer = Layer(
            str(index),
            pixels,
            x=-20 + index * 31,
            y=10 + index * 17,
            opacity=0.35 + index * 0.2,
            blend_mode=["Normal", "Multiply", "Screen"][index],
        )
        if index == 1:
            layer.mask = rng.integers(0, 256, (170, 210), dtype=np.uint8)
            layer.mask_feather = 2
        if index == 2:
            layer.effects = {"stroke": {"enabled": True, "size": 2, "color": [255, 0, 0, 255]}}
        document.layers.append(layer)
    return document


def test_render_engine_matches_reference_composite() -> None:
    document = sample_document()
    engine = RenderEngine(tile_size=128)

    for checker in (False, True):
        expected = document.composite(checker)
        actual = engine.render(document, checker)
        np.testing.assert_array_equal(actual, expected)


def test_local_edit_recomposites_only_dirty_tiles() -> None:
    document = sample_document()
    engine = RenderEngine(tile_size=128)
    engine.render(document, False)
    layer = document.layers[0]
    local_rect = draw_brush(layer, 80, 80, 12, (240, 20, 90, 220), 0.8)
    assert local_rect is not None
    x1, y1, x2, y2 = local_rect
    engine.invalidate_region(document, (x1 + layer.x, y1 + layer.y, x2 + layer.x, y2 + layer.y), layer)

    actual = engine.render(document, False)

    assert len(engine.last_changed_tiles) <= 4
    np.testing.assert_array_equal(actual, document.composite(False))


def test_filter_cache_survives_opacity_change() -> None:
    profiler = PerformanceProfiler(enabled=True)
    document = sample_document()
    document.layers[0].filters = [{"name": "Gaussian Blur", "radius": 2}]
    engine = RenderEngine(tile_size=128, performance=profiler)
    engine.render(document, False)
    misses = profiler.counter("render.filter_cache_miss")

    document.layers[0].opacity = 0.4
    engine.invalidate_full(document, clear_layer_caches=False)
    engine.render(document, False)

    assert profiler.counter("render.filter_cache_miss") == misses
    assert profiler.counter("render.filter_cache_hit") > 0


def test_adjustment_and_clipping_match_reference_after_partial_edit() -> None:
    document = sample_document()
    document.layers[1].clipping = True
    adjustment_pixels = np.zeros((1, 1, 4), dtype=np.uint8)
    document.layers.append(
        Layer(
            "adjustment",
            adjustment_pixels,
            kind="adjustment",
            adjustment={"type": "brightness_contrast", "brightness": 18, "contrast": 1.1},
            opacity=0.65,
        )
    )
    engine = RenderEngine(tile_size=128)
    engine.render(document, False)
    layer = document.layers[0]
    local_rect = draw_brush(layer, 130, 120, 18, (10, 240, 70, 255), 0.7)
    assert local_rect is not None
    x1, y1, x2, y2 = local_rect
    engine.invalidate_region(document, (x1 + layer.x, y1 + layer.y, x2 + layer.x, y2 + layer.y), layer)

    np.testing.assert_array_equal(engine.render(document, False), document.composite(False))


def test_local_blur_filter_update_matches_full_filter_stack() -> None:
    profiler = PerformanceProfiler(enabled=True)
    document = sample_document()
    layer = document.layers[0]
    layer.filters = [{"type": "blur", "radius": 4, "enabled": True, "opacity": 1.0, "blend_mode": "Normal"}]
    engine = RenderEngine(tile_size=128, performance=profiler)
    engine.render(document, False)
    local_rect = draw_brush(layer, 100, 90, 10, (230, 30, 80, 255), 1.0)
    assert local_rect is not None
    x1, y1, x2, y2 = local_rect
    engine.invalidate_region(document, (x1 + layer.x, y1 + layer.y, x2 + layer.x, y2 + layer.y), layer)

    actual = engine.render(document, False)

    assert profiler.counter("render.filter_partial") == 1
    np.testing.assert_array_equal(actual, document.composite(False))


def test_feathered_mask_stroke_updates_mask_cache_locally() -> None:
    profiler = PerformanceProfiler(enabled=True)
    document = sample_document()
    layer = document.layers[1]
    layer.mask_feather = 5
    engine = RenderEngine(tile_size=128, performance=profiler)
    engine.render(document, False)
    local_rect = draw_mask_brush(layer, 120, 100, 12, 0, 1.0)
    assert local_rect is not None
    x1, y1, x2, y2 = local_rect
    engine.invalidate_region(document, (x1 + layer.x, y1 + layer.y, x2 + layer.x, y2 + layer.y), layer, "mask")

    actual = engine.render(document, False)

    assert profiler.counter("render.mask_partial") == 1
    np.testing.assert_array_equal(actual, document.composite(False))


def test_repeated_view_access_does_not_recompose_document() -> None:
    profiler = PerformanceProfiler(enabled=True)
    document = sample_document()
    engine = RenderEngine(tile_size=128, performance=profiler)
    first = engine.render(document, False)
    full_count = profiler.counter("render.full")

    second = engine.render(document, False)

    assert second is first
    assert profiler.counter("render.full") == full_count
    assert profiler.counter("render.partial") == 0
