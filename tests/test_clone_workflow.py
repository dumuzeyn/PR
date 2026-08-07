import numpy as np

from photoredactor.core import Layer, SourceAnchor, clone_or_heal, spot_heal


def test_source_anchor_keeps_aligned_offset_between_strokes() -> None:
    anchor = SourceAnchor(aligned=True)
    anchor.set_source((10, 20))
    assert anchor.begin_stroke((40, 50))
    assert anchor.source_for((44, 55)) == (14, 25)
    anchor.end_stroke()
    assert anchor.begin_stroke((70, 80))
    assert anchor.source_for((70, 80)) == (40, 50)


def test_source_anchor_resets_non_aligned_source_each_stroke() -> None:
    anchor = SourceAnchor(aligned=False)
    anchor.set_source((10, 20))
    anchor.begin_stroke((40, 50))
    anchor.end_stroke()
    anchor.begin_stroke((70, 80))
    assert anchor.source_for((70, 80)) == (10, 20)


def test_clone_clips_at_source_and_target_edges() -> None:
    pixels = np.zeros((24, 24, 4), dtype=np.uint8)
    pixels[:, :, 3] = 255
    pixels[:8, :8, :3] = (220, 30, 20)
    layer = Layer("Edge", pixels)
    changed = clone_or_heal(layer, 1, 1, 22, 22, 6, 1.0, False, hardness=1.0, source_pixels=pixels.copy())
    assert changed is not None
    assert layer.pixels[22, 22, 0] > 150


def test_healing_transfers_texture_but_not_clone_brightness() -> None:
    yy, xx = np.mgrid[0:64, 0:96]
    source_texture = (((xx + yy) % 2) * 80 + 40).astype(np.uint8)
    pixels = np.zeros((64, 96, 4), dtype=np.uint8)
    pixels[:, :, :3] = 180
    pixels[:, :40, :3] = source_texture[:, :40, None]
    pixels[:, :, 3] = 255
    clone = Layer("Clone", pixels.copy())
    healing = Layer("Healing", pixels.copy())
    baseline = pixels.copy()
    clone_or_heal(clone, 22, 32, 70, 32, 14, 1.0, False, hardness=0.8, source_pixels=baseline)
    clone_or_heal(healing, 22, 32, 70, 32, 14, 1.0, True, hardness=0.8, source_pixels=baseline)
    clone_mean = float(clone.pixels[25:40, 63:78, :3].mean())
    healing_mean = float(healing.pixels[25:40, 63:78, :3].mean())
    assert abs(healing_mean - 180.0) < abs(clone_mean - 180.0)
    assert not np.array_equal(clone.pixels, healing.pixels)


def test_spot_heal_reduces_a_short_scratch() -> None:
    pixels = np.full((64, 64, 4), (150, 150, 150, 255), dtype=np.uint8)
    pixels[30:34, 20:44, :3] = 5
    layer = Layer("Scratch", pixels)
    before_error = np.abs(layer.pixels[30:34, 24:40, :3].astype(np.int16) - 150).mean()
    for x in range(24, 41, 5):
        spot_heal(layer, x, 32, 7, 1.0, hardness=0.55)
    after_error = np.abs(layer.pixels[30:34, 24:40, :3].astype(np.int16) - 150).mean()
    assert after_error < before_error
