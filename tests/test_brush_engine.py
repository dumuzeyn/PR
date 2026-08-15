from __future__ import annotations

import numpy as np

from photoredactor.core import (
    BrushPathSampler,
    BrushSettings,
    Document,
    Layer,
    PixelBrushStroke,
    RetouchStroke,
    pointer_pressure,
)


def opaque_layer(width: int = 128, height: int = 96, value: int = 128) -> Layer:
    pixels = np.full((height, width, 4), value, dtype=np.uint8)
    pixels[:, :, 3] = 255
    return Layer("paint", pixels)


def sampled_line(event_count: int, settings: BrushSettings) -> list[tuple[int, int]]:
    sampler = BrushPathSampler(settings)
    points = sampler.begin((10, 30))
    for x in np.linspace(10, 110, event_count)[1:]:
        points.extend(sampler.extend((round(float(x)), 30)))
    return [(dab.x, dab.y) for dab in points]


def test_path_sampling_is_independent_of_pointer_event_rate() -> None:
    settings = BrushSettings(radius=10, spacing=0.25)
    assert sampled_line(7, settings) == sampled_line(121, settings)


def test_spacing_is_relative_to_brush_diameter() -> None:
    dense = sampled_line(20, BrushSettings(radius=10, spacing=0.1))
    sparse = sampled_line(20, BrushSettings(radius=10, spacing=1.0))
    assert len(dense) == 51
    assert len(sparse) == 6


def test_flow_accumulates_but_opacity_caps_one_stroke() -> None:
    layer = opaque_layer(value=0)
    stroke = PixelBrushStroke(
        layer,
        BrushSettings(radius=10, hardness=1.0, opacity=0.5, flow=0.2),
        (255, 255, 255, 255),
    )
    first = None
    for index in range(6):
        stroke.dab(64, 48)
        value = int(layer.pixels[48, 64, 0])
        if index == 0:
            first = value
    assert first is not None and 20 <= first <= 30
    assert first < int(layer.pixels[48, 64, 0]) <= 128

    capped = int(layer.pixels[48, 64, 0])
    for _ in range(30):
        stroke.dab(64, 48)
    assert capped < int(layer.pixels[48, 64, 0]) <= 128


def test_eraser_changes_alpha_without_blackening_rgb() -> None:
    layer = opaque_layer(value=173)
    before_rgb = layer.pixels[:, :, :3].copy()
    stroke = PixelBrushStroke(layer, BrushSettings(radius=9, hardness=0.5, opacity=0.8, flow=1.0), (0, 0, 0, 255), erase=True)
    stroke.dab(64, 48)
    np.testing.assert_array_equal(layer.pixels[:, :, :3], before_rgb)
    assert 0 < int(layer.pixels[48, 64, 3]) < 255


def test_hard_eraser_fully_clears_alpha_and_hidden_rgb() -> None:
    layer = opaque_layer(value=173)
    stroke = PixelBrushStroke(
        layer,
        BrushSettings(radius=9, hardness=1.0, opacity=1.0, flow=1.0, spacing=0.0),
        (0, 0, 0, 255),
        erase=True,
    )
    stroke.dab(64, 48)
    np.testing.assert_array_equal(layer.pixels[48, 64], np.zeros(4, dtype=np.uint8))


def test_brush_respects_soft_selection_coverage() -> None:
    layer = opaque_layer(value=0)
    selection = np.zeros((96, 128), dtype=np.uint8)
    selection[:, :64] = 128
    stroke = PixelBrushStroke(
        layer,
        BrushSettings(radius=14, hardness=1.0, opacity=1.0, flow=1.0),
        (255, 0, 0, 255),
        selection_mask=selection,
    )
    stroke.dab(64, 48)
    assert 110 <= int(layer.pixels[48, 60, 0]) <= 140
    assert int(layer.pixels[48, 68, 0]) == 0


def test_brush_blend_modes_use_existing_blending_rules() -> None:
    multiply = opaque_layer(value=128)
    screen = opaque_layer(value=128)
    PixelBrushStroke(multiply, BrushSettings(5, hardness=1.0, blend_mode="Multiply"), (128, 128, 128, 255)).dab(32, 32)
    PixelBrushStroke(screen, BrushSettings(5, hardness=1.0, blend_mode="Screen"), (128, 128, 128, 255)).dab(32, 32)
    assert int(multiply.pixels[32, 32, 0]) < 128
    assert int(screen.pixels[32, 32, 0]) > 128


def test_pressure_can_change_size_opacity_and_flow() -> None:
    settings = BrushSettings(
        radius=20,
        opacity=0.8,
        flow=0.6,
        pressure_size=True,
        pressure_opacity=True,
        pressure_flow=True,
    ).normalized()
    low = settings.for_pressure(0.25)
    high = settings.for_pressure(1.0)
    assert low[0] < high[0]
    assert low[1] < high[1]
    assert low[2] < high[2]


def test_missing_pointer_pressure_has_safe_fallback() -> None:
    assert pointer_pressure(object()) == 1.0
    event = type("Event", (), {"pressure": 512})()
    assert 0.49 < pointer_pressure(event) < 0.51


def textured_pixels() -> np.ndarray:
    rng = np.random.default_rng(12)
    pixels = rng.integers(20, 235, (96, 128, 4), dtype=np.uint8)
    pixels[:, :, 3] = 255
    return pixels


def retouch_amount(mode: str, strength: float, passes: int = 1) -> tuple[float, np.ndarray, np.ndarray]:
    source = textured_pixels()
    layer = Layer(mode, source.copy())
    for _ in range(passes):
        stroke = RetouchStroke(layer, mode, 14, 0.45, strength, flow=0.3)
        stroke.dab(64, 48)
    delta = np.abs(layer.pixels[:, :, :3].astype(np.int16) - source[:, :, :3].astype(np.int16))
    return float(delta.mean()), source, layer.pixels


def test_blur_strength_and_multiple_passes_are_gradual() -> None:
    weak, _, _ = retouch_amount("blur", 0.1)
    medium, _, _ = retouch_amount("blur", 0.5)
    strong, _, _ = retouch_amount("blur", 1.0)
    repeated, _, _ = retouch_amount("blur", 0.5, passes=3)
    assert 0 < weak < medium < strong
    assert repeated > medium


def test_blur_and_sharpen_do_not_touch_pixels_outside_brush() -> None:
    for mode in ("blur", "sharpen"):
        _, source, edited = retouch_amount(mode, 1.0)
        np.testing.assert_array_equal(edited[:25], source[:25])
        np.testing.assert_array_equal(edited[72:], source[72:])
        assert edited.dtype == np.uint8
        assert int(edited.min()) >= 0 and int(edited.max()) <= 255


def test_blur_and_sharpen_preserve_working_depth_and_untouched_precision() -> None:
    rng = np.random.default_rng(33)
    source = rng.random((48, 64, 4), dtype=np.float32)
    source[:, :, 3] = 1.0
    for depth, dtype in ((16, np.uint16), (32, np.float32)):
        for mode in ("blur", "sharpen"):
            document = Document.new(64, 48)
            document.set_bit_depth(depth)
            document.layer.set_working_rgba(source, depth, "RGBA")
            before = document.layer.working_pixels.copy()
            stroke = RetouchStroke(document.layer, mode, 8, 0.5, 0.4, flow=0.35)
            stroke.dab(32, 24)
            document.layer.touch_pixels()
            assert document.layer.working_pixels.dtype == dtype
            np.testing.assert_array_equal(document.layer.working_pixels[:8], before[:8])
