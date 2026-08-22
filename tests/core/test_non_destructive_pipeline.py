from pathlib import Path

import numpy as np

from uzyro.color_management import display_rgba, quantize_rgba
from uzyro.core import (
    BLEND_MODES,
    Document,
    adjust_brightness_contrast,
    apply_adjustment,
    apply_filter_stack,
    encode_png,
    render_layer_pixels,
)
from uzyro.history import History, LayerPropertyCommand


FILTERS = [
    {"type": "blur", "radius": 2},
    {"type": "sharpen", "amount": 0.8},
    {"type": "noise", "amount": 0.05, "seed": 7},
    {"type": "median", "size": 7},
    {"type": "edge", "strength": 0.6},
    {"type": "emboss", "strength": 0.5},
]

ADJUSTMENTS = [
    {"type": "brightness_contrast", "brightness": 15, "contrast": 1.1},
    {"type": "saturation", "saturation": 1.2},
    {"type": "vibrance", "vibrance": 0.3, "saturation": 1.1},
    {"type": "temperature_tint", "temperature": 18, "tint": -8},
    {"type": "hue_saturation", "hue": 20, "saturation": 1.1, "lightness": 5},
    {"type": "exposure", "exposure": 0.25, "offset": 0.01, "gamma": 1.1},
    {"type": "color_balance", "red": 8, "green": -4, "blue": 5},
    {"type": "levels", "black": 5, "white": 245, "gamma": 1.1},
    {"type": "curves", "shadows": 58, "midtones": 138, "highlights": 200},
    {"type": "threshold", "threshold": 120},
    {"type": "posterize", "levels": 8},
    {"type": "invert"},
    {"type": "grayscale"},
]


def sample(dtype=np.uint8) -> np.ndarray:
    yy, xx = np.mgrid[:24, :32]
    normalized = np.dstack(
        (
            xx / 31.0,
            yy / 23.0,
            (xx + yy) / 54.0,
            0.2 + 0.8 * xx / 31.0,
        )
    ).astype(np.float32)
    return quantize_rgba(normalized, 8 if dtype == np.uint8 else 16 if dtype == np.uint16 else 32)


def test_every_filter_preserves_8_16_and_32_bit_precision() -> None:
    for dtype in (np.uint8, np.uint16, np.float32):
        source = sample(dtype)
        for item in FILTERS:
            result = apply_filter_stack(source, [{**item, "channel": "RGB"}])
            assert result.dtype == dtype
            assert result.shape == source.shape
            assert np.isfinite(result).all()
            assert np.array_equal(result[:, :, 3], source[:, :, 3])


def test_filter_channels_and_alpha_are_isolated() -> None:
    source = sample(np.float32)
    red = apply_filter_stack(source, [{"type": "noise", "amount": 0.2, "seed": 4, "channel": "Red"}])
    assert not np.array_equal(red[:, :, 0], source[:, :, 0])
    np.testing.assert_array_equal(red[:, :, 1:], source[:, :, 1:])

    alpha = apply_filter_stack(source, [{"type": "noise", "amount": 0.2, "seed": 4, "channel": "Alpha"}])
    np.testing.assert_array_equal(alpha[:, :, :3], source[:, :, :3])
    assert not np.array_equal(alpha[:, :, 3], source[:, :, 3])


def test_filter_mask_inversion_density_and_feather() -> None:
    source = sample(np.float32)
    zero_mask = np.zeros(source.shape[:2], dtype=np.uint8)
    encoded = encode_png(np.dstack([zero_mask] * 4))
    base = {"type": "noise", "amount": 0.2, "seed": 11, "mask": encoded}
    blocked = apply_filter_stack(source, [base])
    np.testing.assert_allclose(blocked, source)
    inverted = apply_filter_stack(source, [{**base, "mask_inverted": True}])
    assert not np.array_equal(inverted, source)
    no_density = apply_filter_stack(source, [{**base, "mask_density": 0.0}])
    np.testing.assert_allclose(no_density, inverted)

    half_mask = np.zeros(source.shape[:2], dtype=np.uint8)
    half_mask[:, source.shape[1] // 2 :] = 255
    feathered = apply_filter_stack(source, [{**base, "mask": encode_png(np.dstack([half_mask] * 4)), "mask_feather": 3.0}])
    assert np.any(np.abs(feathered[:, source.shape[1] // 2 - 2] - source[:, source.shape[1] // 2 - 2]) > 1e-5)


def test_every_filter_channel_and_blend_mode_combination_is_bounded_and_deterministic() -> None:
    source = sample(np.float32)
    for template in FILTERS:
        for mode in BLEND_MODES:
            for channel in ("RGB", "Red", "Green", "Blue", "Alpha"):
                item = {**template, "opacity": 0.65, "blend_mode": mode, "channel": channel}
                first = apply_filter_stack(source, [item])
                second = apply_filter_stack(source, [item])
                np.testing.assert_array_equal(first, second)
                assert float(first.min()) >= 0.0
                assert float(first.max()) <= 1.0
                if channel == "Alpha":
                    np.testing.assert_array_equal(first[:, :, :3], source[:, :, :3])
                elif channel != "RGB":
                    changed = {"Red": 0, "Green": 1, "Blue": 2}[channel]
                    for index in {0, 1, 2} - {changed}:
                        np.testing.assert_array_equal(first[:, :, index], source[:, :, index])


def test_every_adjustment_supports_precision_and_channels() -> None:
    for dtype in (np.uint8, np.uint16, np.float32):
        source = sample(dtype)
        for adjustment in ADJUSTMENTS:
            for channel in ("RGB", "Red", "Green", "Blue", "Alpha"):
                result = apply_adjustment(source, {**adjustment, "channel": channel})
                assert result.dtype == dtype
                assert result.shape == source.shape
                assert np.isfinite(result).all()


def test_adjustment_channel_isolation() -> None:
    source = sample(np.uint16)
    red = apply_adjustment(source, {"type": "invert", "channel": "Red"})
    np.testing.assert_array_equal(red[:, :, 1:], source[:, :, 1:])
    np.testing.assert_array_equal(red[:, :, 0], 65535 - source[:, :, 0])
    alpha = apply_adjustment(source, {"type": "invert", "channel": "Alpha"})
    np.testing.assert_array_equal(alpha[:, :, :3], source[:, :, :3])
    np.testing.assert_array_equal(alpha[:, :, 3], 65535 - source[:, :, 3])


def test_every_adjustment_and_layer_blend_mode_combination_renders() -> None:
    for adjustment in ADJUSTMENTS:
        for mode in BLEND_MODES:
            document = Document.new(32, 24, (25, 65, 105, 255))
            document.layers[0].pixels = sample(np.uint8)
            document.add_adjustment_layer("Проверка", {**adjustment, "channel": "RGB"})
            document.layer.blend_mode = mode
            document.layer.opacity = 0.7
            document.layer.mask = np.tile(np.linspace(0, 255, 32, dtype=np.uint8), (24, 1))
            first = document.composite()
            second = document.composite()
            np.testing.assert_array_equal(first, second)
            assert first.dtype == np.uint8
            assert first.shape == (24, 32, 4)


def test_high_precision_source_drives_non_destructive_layer_render() -> None:
    document = Document.new(32, 24, (0, 0, 0, 0))
    document.set_bit_depth(16)
    layer = document.layer
    layer.working_pixels = sample(np.uint16)
    layer.pixels = display_rgba(layer.working_pixels)
    layer.filters = [{"type": "blur", "radius": 2, "channel": "RGB"}]
    expected = display_rgba(apply_filter_stack(layer.working_pixels, layer.filters))
    np.testing.assert_array_equal(render_layer_pixels(layer), expected)
    assert layer.working_pixels.dtype == np.uint16


def test_filter_and_adjustment_metadata_roundtrip_and_undo(tmp_path: Path) -> None:
    document = Document.new(32, 24, (40, 80, 120, 255))
    filters = [{
        "type": "blur",
        "radius": 3,
        "channel": "Blue",
        "mask_inverted": True,
        "mask_density": 0.65,
        "mask_feather": 4.0,
    }]
    command = LayerPropertyCommand("Фильтры", document.layer.id, "filters", [], filters)
    command.redo(document)
    history = History()
    history.push(command)
    history.undo(document)
    assert document.layer.filters == []
    history.redo(document)
    assert document.layer.filters == filters

    document.add_adjustment_layer("Красный канал", {"type": "brightness_contrast", "brightness": 20, "contrast": 1.0, "channel": "Red"})
    document.layer.mask = np.full((24, 32), 180, dtype=np.uint8)
    path = tmp_path / "filters-and-adjustments.prdx"
    document.save_project(path)
    restored = Document.open_project(path)
    assert restored.layers[0].filters == filters
    assert restored.layer.adjustment["channel"] == "Red"
    np.testing.assert_array_equal(restored.layer.mask, document.layer.mask)
    assert restored.composite().shape == (24, 32, 4)


def test_adjustment_functions_no_longer_reduce_high_precision_to_uint8() -> None:
    source = sample(np.float32)
    result = adjust_brightness_contrast(source, 7, 1.03)
    assert result.dtype == np.float32
    assert np.unique(result[:, :, 0]).size > 16
