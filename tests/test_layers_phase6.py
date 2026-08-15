from __future__ import annotations

from pathlib import Path

import numpy as np

from uzyro.adjustment_ops import apply_adjustment
from uzyro.core import Document, Layer, apply_filter_stack
from uzyro.filter_ops import deterministic_noise
from uzyro.layer_effects import EFFECT_ORDER, LayerEffectsStack


def styled_source() -> np.ndarray:
    pixels = np.zeros((80, 96, 4), dtype=np.uint8)
    pixels[16:66, 20:76] = (75, 145, 220, 255)
    pixels[29:52, 34:62] = (235, 175, 45, 255)
    return pixels


def test_every_layer_effect_is_real_deterministic_and_non_destructive() -> None:
    source = styled_source()
    for kind in EFFECT_ORDER:
        layer = Layer("Style", source.copy(), x=4, y=6, effects={kind: {"enabled": True, "opacity": 0.8}})
        before = layer.pixels.copy()
        underlays_a, styled_a = LayerEffectsStack.render(layer, layer.pixels)
        underlays_b, styled_b = LayerEffectsStack.render(layer, layer.pixels)
        assert np.array_equal(layer.pixels, before), kind
        assert len(underlays_a) == len(underlays_b)
        assert np.array_equal(styled_a, styled_b), kind
        if kind in {"drop_shadow", "outer_glow", "stroke"}:
            assert underlays_a and np.count_nonzero(underlays_a[0][0][:, :, 3]) > 0, kind
        else:
            assert not np.array_equal(styled_a[:, :, :3], source[:, :, :3]), kind


def test_layer_effect_stack_roundtrips_all_effects(tmp_path: Path) -> None:
    document = Document.new(96, 80, (0, 0, 0, 0))
    document.layer.pixels = styled_source()
    document.layer.effects = {
        kind: {**LayerEffectsStack.item(kind), "enabled": True, "opacity": 0.45 + index * 0.03}
        for index, kind in enumerate(EFFECT_ORDER)
    }
    rendered = document.composite(False)
    project = tmp_path / "layer-effects.prdx"
    document.save_project(project)
    restored = Document.open_project(project)
    assert tuple(restored.layer.effects) == EFFECT_ORDER
    assert np.array_equal(restored.composite(False), rendered)


def test_required_phase6_filters_have_distinct_results_and_preserve_alpha() -> None:
    source = styled_source()
    rng = np.random.default_rng(24)
    source[:, :, :3] = np.clip(source[:, :, :3].astype(np.int16) + rng.integers(-18, 19, source[:, :, :3].shape), 0, 255).astype(np.uint8)
    filters = [
        {"type": "blur", "radius": 3},
        {"type": "motion_blur", "distance": 10, "angle": 35},
        {"type": "unsharp_mask", "amount": 1.4, "radius": 1.8, "threshold": 2},
        {"type": "smart_sharpen", "amount": 1.5, "radius": 1.2},
        {"type": "noise", "amount": 0.04, "seed": 9},
        {"type": "reduce_noise", "strength": 0.5},
        {"type": "high_pass", "radius": 4.0},
    ]
    outputs = [apply_filter_stack(source, [item]) for item in filters]
    assert all(np.array_equal(output[:, :, 3], source[:, :, 3]) for output in outputs)
    assert all(not np.array_equal(output[:, :, :3], source[:, :, :3]) for output in outputs)
    signatures = {output[:, :, :3].tobytes() for output in outputs}
    assert len(signatures) == len(filters)


def test_black_white_adjustment_has_editable_channel_mix() -> None:
    source = np.array([[[240, 30, 20, 255], [20, 220, 35, 180], [20, 40, 235, 90]]], dtype=np.uint8)
    red = apply_adjustment(source, {"type": "black_white", "red": 1.0, "green": 0.0, "blue": 0.0})
    blue = apply_adjustment(source, {"type": "black_white", "red": 0.0, "green": 0.0, "blue": 1.0})
    assert np.array_equal(red[:, :, 0], red[:, :, 1])
    assert np.array_equal(blue[:, :, 1], blue[:, :, 2])
    assert not np.array_equal(red[:, :, :3], blue[:, :, :3])
    assert np.array_equal(red[:, :, 3], source[:, :, 3])


def test_noise_filter_is_deterministic() -> None:
    source = styled_source()
    assert np.array_equal(deterministic_noise(source, 0.15), deterministic_noise(source, 0.15))


def test_disabled_layer_effects_leave_pixels_unchanged() -> None:
    source = styled_source()
    layer = Layer("Style", source.copy(), effects={kind: {"enabled": False} for kind in EFFECT_ORDER})
    underlays, styled = LayerEffectsStack.render(layer, layer.pixels)
    assert underlays == []
    assert np.array_equal(styled, source)
