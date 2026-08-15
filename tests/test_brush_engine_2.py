from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from photoredactor.brush_dynamics import dynamic_dabs
from photoredactor.brush_tip import BRUSH_TEXTURE, BrushTipCache
from photoredactor.core import BrushPathSampler, BrushSettings, Layer, PixelBrushStroke
from photoredactor.tablet_input import EventTabletBackend, TabletSample


def save_mask(path: Path, mask: np.ndarray) -> str:
    ok, payload = cv2.imencode(".png", mask)
    assert ok
    path.write_bytes(payload.tobytes())
    return str(path)


def spread(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask > 0.15)
    return float(xs.std()), float(ys.std())


def test_roundness_and_angle_change_actual_tip_geometry() -> None:
    cache = BrushTipCache()
    horizontal = cache.stamp(30, 0.8, angle=0, roundness=0.3)
    vertical = cache.stamp(30, 0.8, angle=90, roundness=0.3)
    hx, hy = spread(horizontal)
    vx, vy = spread(vertical)
    assert hx > hy * 2.0
    assert vy > vx * 2.0


def test_custom_tip_flip_dual_mask_and_cache(tmp_path: Path) -> None:
    tip = np.zeros((30, 30), dtype=np.uint8)
    tip[4:25, 3:11] = 255
    dual = np.zeros((30, 30), dtype=np.uint8)
    dual[:, ::2] = 255
    tip_path = save_mask(tmp_path / "tip.png", tip)
    dual_path = save_mask(tmp_path / "dual.png", dual)
    cache = BrushTipCache()
    original = cache.stamp(20, 1.0, custom_path=tip_path)
    flipped = cache.stamp(20, 1.0, flip_x=True, custom_path=tip_path)
    combined = cache.stamp(20, 1.0, custom_path=tip_path, dual_path=dual_path)
    assert original is cache.stamp(20, 1.0, custom_path=tip_path)
    assert np.allclose(original, np.fliplr(flipped), atol=0.02)
    assert float(combined.sum()) < float(original.sum()) * 0.75


def test_canvas_space_texture_does_not_slide_between_dabs(tmp_path: Path) -> None:
    texture = np.tile(np.array([[0, 255], [255, 0]], dtype=np.uint8), (6, 6))
    path = save_mask(tmp_path / "texture.png", texture)
    mask = np.ones((9, 9), dtype=np.float32)
    first = BRUSH_TEXTURE.apply(mask, path, (10, 10, 19, 19), depth=1.0, canvas_space=True)
    second = BRUSH_TEXTURE.apply(mask, path, (10, 10, 19, 19), depth=1.0, stroke_origin=(9, 9), canvas_space=True)
    shifted = BRUSH_TEXTURE.apply(mask, path, (10, 10, 19, 19), depth=1.0, stroke_origin=(9, 10), canvas_space=False)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, shifted)


def test_shape_scatter_and_transfer_are_seeded_and_pressure_aware() -> None:
    settings = BrushSettings(
        radius=20, size_jitter=0.6, minimum_diameter=0.2, size_control="Pen Pressure",
        angle_jitter=0.5, roundness=0.8, roundness_jitter=0.5,
        scatter=1.5, scatter_both_axes=True, scatter_count=4, count_jitter=0.2,
        opacity_jitter=0.5, flow_jitter=0.5, minimum_opacity=0.2, minimum_flow=0.3,
        random_seed=73,
    ).normalized()
    sample = TabletSample(0.35, pressure_available=True)
    first = dynamic_dabs(settings, (50, 50), sample, 4, (1, 0))
    second = dynamic_dabs(settings, (50, 50), sample, 4, (1, 0))
    assert first == second
    assert 1 <= len(first) <= 4
    assert any(dab.y != 50 for dab in first)
    assert all(dab.radius >= 4 and dab.opacity_factor >= 0.2 and dab.flow_factor >= 0.3 for dab in first)


def sampled(settings: BrushSettings, count: int) -> list[tuple[int, int, int | None]]:
    sampler = BrushPathSampler(settings)
    result = sampler.begin((10, 50))
    for x in np.linspace(10, 210, count)[1:]:
        result.extend(sampler.extend((round(float(x)), 50)))
    return [(dab.x, dab.y, dab.radius) for dab in result]


def test_scattered_stroke_is_independent_of_pointer_event_rate() -> None:
    settings = BrushSettings(radius=8, spacing=0.25, scatter=0.8, scatter_count=3, random_seed=44)
    assert sampled(settings, 6) == sampled(settings, 101)


def test_stabilizer_and_pulled_string_have_distinct_behavior() -> None:
    stabilizer = BrushPathSampler(BrushSettings(5, spacing=0.2, smoothing_mode="stabilizer", stabilizer_strength=1.0, stabilizer_window=8))
    stabilizer.begin((0, 0))
    stable = stabilizer.extend((40, 20))
    assert stable and stable[-1].x < 40
    pulled = BrushPathSampler(BrushSettings(5, spacing=0.2, smoothing_mode="pulled_string", pulled_string_radius=12))
    pulled.begin((0, 0))
    assert pulled.extend((8, 0)) == []
    moved = pulled.extend((30, 0))
    assert moved and max(dab.x for dab in moved) <= 18
    for mode in ("stabilizer", "pulled_string"):
        settings = BrushSettings(
            8, spacing=0.2, smoothing_mode=mode,
            stabilizer_strength=0.8, stabilizer_window=8, pulled_string_radius=12,
        )
        assert sampled(settings, 6) == sampled(settings, 101)


def test_zero_spacing_uses_bounded_continuous_sampling() -> None:
    settings = BrushSettings(28, hardness=1.0, spacing=0.0, smoothing=0.0)
    sampler = BrushPathSampler(settings)
    dabs = sampler.begin((100, 80)) + sampler.extend((1600, 80))
    assert 250 <= len(dabs) <= 300
    assert abs(sampler.step - 5.6) < 1e-9
    layer = Layer("Непрерывная линия", np.zeros((160, 1700, 4), dtype=np.uint8))
    stroke = PixelBrushStroke(layer, settings, (25, 120, 240, 255))
    for dab in dabs:
        stroke.dab(dab.x, dab.y, dab.pressure, dab)
    assert np.all(layer.pixels[80, 100:1601, 3] == 255)


def test_tablet_adapter_normalizes_optional_channels() -> None:
    sample = EventTabletBackend().sample(SimpleNamespace(pressure=32768, tilt_x=45, tilt_y=-90, rotation=450, eraser=True))
    assert 0.49 < sample.pressure < 0.51
    assert sample.tilt_x == 0.5 and sample.tilt_y == -1.0
    assert sample.rotation == 90.0 and sample.eraser
    fallback = EventTabletBackend().sample(object())
    assert fallback.pressure == 1.0 and not fallback.pressure_available


def test_color_dynamics_mix_foreground_and_background() -> None:
    pixels = np.zeros((50, 80, 4), dtype=np.uint8)
    pixels[:, :, 3] = 255
    layer = Layer("color", pixels)
    settings = BrushSettings(8, hardness=1.0, foreground_background_jitter=1.0, random_seed=8)
    stroke = PixelBrushStroke(layer, settings, (255, 0, 0, 255), background=(0, 0, 255, 255))
    stroke.dab(30, 25)
    center = layer.pixels[25, 30]
    assert center[0] > 0 and center[2] > 0
