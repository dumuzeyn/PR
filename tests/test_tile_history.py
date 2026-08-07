import numpy as np

from photoredactor.core import Document
from photoredactor.history import History, LayerFieldsCommand, MaskTilePatchCommand, PixelTilePatchCommand, TilePatch


def test_pixel_tile_command_undo_redo() -> None:
    document = Document.new(300, 200, (0, 0, 0, 0))
    layer = document.layer
    rects = [(0, 0, 64, 64), (192, 128, 256, 192)]
    patches = []
    for index, rect in enumerate(rects):
        x1, y1, x2, y2 = rect
        before = layer.pixels[y1:y2, x1:x2].copy()
        layer.pixels[y1:y2, x1:x2] = (20 + index * 100, 40, 60, 255)
        patches.append(TilePatch(rect, before, layer.pixels[y1:y2, x1:x2].copy()))
    expected = layer.pixels.copy()
    history = History()
    history.push(PixelTilePatchCommand("stroke", layer.id, patches))

    history.undo(document)
    assert not np.any(layer.pixels[:, :, 3])
    history.redo(document)
    np.testing.assert_array_equal(layer.pixels, expected)


def test_mask_tile_command_creates_missing_mask_on_redo() -> None:
    document = Document.new(64, 64)
    layer = document.layer
    rect = (0, 0, 32, 32)
    before = np.full((32, 32), 255, dtype=np.uint8)
    after = np.zeros((32, 32), dtype=np.uint8)
    command = MaskTilePatchCommand("mask stroke", layer.id, [TilePatch(rect, before, after)])

    command.redo(document)

    assert layer.mask is not None
    assert not np.any(layer.mask[:32, :32])
    command.undo(document)
    assert np.all(layer.mask[:32, :32] == 255)


def test_sparse_tile_history_uses_less_memory_than_bounding_rectangle() -> None:
    tile = np.zeros((64, 64, 4), dtype=np.uint8)
    patches = [
        TilePatch((0, 0, 64, 64), tile.copy(), tile.copy()),
        TilePatch((960, 960, 1024, 1024), tile.copy(), tile.copy()),
    ]
    command = PixelTilePatchCommand("long stroke", "layer", patches)

    full_bounding_snapshot_bytes = 1024 * 1024 * 4 * 2
    assert command.memory_bytes < full_bounding_snapshot_bytes // 100


def test_layer_fields_command_restores_only_changed_values() -> None:
    document = Document.new(80, 60)
    layer = document.layer
    original_pixels = layer.pixels
    command = LayerFieldsCommand(
        "properties",
        layer.id,
        {"name": "Background", "opacity": 1.0},
        {"name": "Retouched", "opacity": 0.35},
    )

    command.redo(document)
    assert (layer.name, layer.opacity) == ("Retouched", 0.35)
    assert layer.pixels is original_pixels
    command.undo(document)
    assert (layer.name, layer.opacity) == ("Background", 1.0)
