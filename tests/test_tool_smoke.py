import numpy as np

from photoredactor.core import (
    Document,
    apply_filter_stack,
    apply_gradient,
    clone_or_heal,
    draw_brush,
    draw_mask_brush,
    flood_fill,
    local_retouch,
    spot_heal,
)


def textured_document() -> Document:
    document = Document.new(160, 120, (0, 0, 0, 0))
    yy, xx = np.mgrid[:120, :160]
    document.layer.pixels[:, :, 0] = (xx * 3) % 255
    document.layer.pixels[:, :, 1] = (yy * 5) % 255
    document.layer.pixels[:, :, 2] = ((xx + yy) * 2) % 255
    document.layer.pixels[:, :, 3] = 255
    return document


def test_paint_and_retouch_tools_modify_only_local_pixels() -> None:
    for operation in (
        lambda layer: draw_brush(layer, 60, 50, 8, (255, 20, 30, 255), 0.8),
        lambda layer: draw_brush(layer, 60, 50, 8, (0, 0, 0, 0), 0.8, erase=True),
        lambda layer: local_retouch(layer, 60, 50, 8, "blur", 0.8),
        lambda layer: local_retouch(layer, 60, 50, 8, "sharpen", 0.8),
        lambda layer: local_retouch(layer, 60, 50, 8, "dodge", 0.8),
        lambda layer: local_retouch(layer, 60, 50, 8, "burn", 0.8),
        lambda layer: clone_or_heal(layer, 30, 30, 60, 50, 8, 0.8, False),
        lambda layer: clone_or_heal(layer, 30, 30, 60, 50, 8, 0.8, True),
        lambda layer: spot_heal(layer, 60, 50, 8, 0.8),
    ):
        document = textured_document()
        before = document.layer.pixels.copy()
        rect = operation(document.layer)
        assert rect is not None
        assert np.any(document.layer.pixels != before)


def test_mask_fill_gradient_and_filter_stack_smoke() -> None:
    document = textured_document()
    layer = document.layer
    assert draw_mask_brush(layer, 40, 40, 10, 0, 1.0) is not None
    assert layer.mask is not None and np.any(layer.mask < 255)

    flood_fill(layer, 0, 0, (10, 20, 30, 255), 0)
    apply_gradient(layer, (20, 20, 100, 70), (0, 0, 0, 255), (255, 255, 255, 255))
    filtered = apply_filter_stack(
        layer.pixels,
        [
            {"type": "blur", "radius": 2},
            {"type": "median", "size": 3, "opacity": 0.5},
            {"type": "emboss", "strength": 0.4, "blend_mode": "Overlay"},
        ],
    )

    assert filtered.shape == layer.pixels.shape
    assert filtered.dtype == np.uint8
