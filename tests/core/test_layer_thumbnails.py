from __future__ import annotations

import numpy as np

from uzyro.core import Document
from uzyro.layer_thumbnails import build_layer_thumbnail


def _color_bounds(image, color_index: int, minimum: int) -> tuple[int, int, int, int]:
    pixels = np.asarray(image)
    mask = pixels[:, :, color_index] >= minimum
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def test_small_shape_fills_layer_thumbnail_instead_of_shrinking_with_document() -> None:
    document = Document.new(1200, 800, (255, 255, 255, 255))
    shape = document.add_shape_layer("rectangle", (570, 380, 630, 420), (240, 30, 40, 255))

    thumbnail = build_layer_thumbnail(shape, 32)

    left, top, right, bottom = _color_bounds(thumbnail, 0, 225)
    assert thumbnail.size == (32, 32)
    assert right - left >= 24
    assert bottom - top >= 14


def test_layer_thumbnail_renders_text_and_respects_mask() -> None:
    document = Document.new(640, 480, (255, 255, 255, 255))
    text = document.add_text_layer("UZYRO", 250, 210, (20, 90, 230, 255), 48)
    text.mask = np.zeros(text.pixels.shape[:2], dtype=np.uint8)
    text.touch_mask()

    hidden = np.asarray(build_layer_thumbnail(text, 32))
    text.mask.fill(255)
    text.touch_mask()
    visible = np.asarray(build_layer_thumbnail(text, 32))

    hidden_rgb = hidden[:, :, :3].astype(np.int16)
    visible_rgb = visible[:, :, :3].astype(np.int16)
    assert not np.any((hidden_rgb[:, :, 2] > hidden_rgb[:, :, 0] + 20) & (hidden_rgb[:, :, 2] > 210))
    assert np.any((visible_rgb[:, :, 2] > visible_rgb[:, :, 0] + 20) & (visible_rgb[:, :, 2] > 210))
