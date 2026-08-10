from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from photoredactor.core import Document, blend_rgb, frequency_separation, portrait_cleanup, reduce_red_eye


def portrait_scene(scale: int = 1, skin_tone: tuple[int, int, int] = (185, 126, 102)) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    height, width = 180 * scale, 240 * scale
    pixels = np.full((height, width, 4), (32, 66, 104, 255), dtype=np.uint8)
    yy, xx = np.mgrid[:height, :width]
    face = ((xx - width * 0.5) / (width * 0.26)) ** 2 + ((yy - height * 0.55) / (height * 0.38)) ** 2 <= 1.0
    noise = np.random.default_rng(400 + scale).normal(0, 9, (height, width, 1))
    pixels[:, :, :3][face] = np.clip(np.asarray(skin_tone) + noise[face], 0, 255)
    hair = face & (yy < height * 0.31 + np.abs(xx - width * 0.5) * 0.18)
    pixels[:, :, :3][hair] = (24, 20, 18)
    eyes = np.zeros((height, width), dtype=bool)
    highlights = np.zeros((height, width), dtype=bool)
    for center_x in (0.42, 0.58):
        eye = ((xx - width * center_x) / (width * 0.055)) ** 2 + ((yy - height * 0.48) / (height * 0.035)) ** 2 <= 1.0
        pupil = ((xx - width * center_x) / (width * 0.022)) ** 2 + ((yy - height * 0.48) / (height * 0.023)) ** 2 <= 1.0
        highlight = (xx - int(width * (center_x + 0.006))) ** 2 + (yy - int(height * 0.475)) ** 2 <= max(1, int(width * 0.004)) ** 2
        pixels[:, :, :3][eye] = (230, 225, 215)
        pixels[:, :, :3][pupil] = (215, 35, 30)
        pixels[:, :, :3][highlight] = (245, 245, 240)
        eyes |= pupil
        highlights |= highlight
    red_object = np.zeros((height, width), dtype=bool)
    red_object[int(height * 0.72) : int(height * 0.88), int(width * 0.04) : int(width * 0.25)] = True
    pixels[:, :, :3][red_object] = (210, 25, 30)
    cheek = face & (yy > height * 0.57) & (yy < height * 0.70) & (xx > width * 0.35) & (xx < width * 0.65)
    return pixels, {"face": face, "hair": hair, "eyes": eyes, "highlights": highlights, "red_object": red_object, "cheek": cheek}


def test_frequency_separation_reconstructs_multiple_image_sizes_and_alpha() -> None:
    for scale in (1, 2, 4):
        source, _regions = portrait_scene(scale)
        source[:, : 5 * scale, 3] = np.linspace(0, 255, 5 * scale, dtype=np.uint8)[None, :]
        low, high = frequency_separation(source, 4.0 * scale, 1.0)
        rebuilt = blend_rgb(high[:, :, :3], low[:, :, :3], "Linear Light")
        assert int(np.abs(rebuilt.astype(np.int16) - source[:, :, :3].astype(np.int16)).max()) <= 2
        assert np.array_equal(low[:, :, 3], source[:, :, 3])
        assert np.array_equal(high[:, :, 3], source[:, :, 3])


def test_portrait_cleanup_is_scale_stable_and_preserves_hair_edges() -> None:
    changed_amounts: list[float] = []
    for scale in (1, 2, 4):
        source, regions = portrait_scene(scale)
        result = portrait_cleanup(source, 0.65, 0.6, 0.35, 0.4, None, 0.8)
        delta = np.abs(result[:, :, :3].astype(np.int16) - source[:, :, :3].astype(np.int16))
        changed_amounts.append(float(delta[regions["face"]].mean()))
        texture_ratio = float(result[:, :, :3][regions["cheek"]].std() / source[:, :, :3][regions["cheek"]].std())
        assert 0.65 < texture_ratio < 0.95
        assert float(delta[regions["hair"]].mean()) < 1.5
        assert int(delta[~regions["face"]].max()) == 0
        assert np.array_equal(result[:, :, 3], source[:, :, 3])
    assert max(changed_amounts) - min(changed_amounts) < 0.8


def test_portrait_cleanup_obeys_soft_selection() -> None:
    source, regions = portrait_scene(2, (126, 82, 66))
    selection = np.zeros(source.shape[:2], dtype=np.uint8)
    selection[:, : source.shape[1] // 2] = 255
    selection[:, source.shape[1] // 2] = 128
    result = portrait_cleanup(source, 0.8, 0.55, 0.4, 0.35, selection, 0.9)
    delta = np.abs(result[:, :, :3].astype(np.int16) - source[:, :, :3].astype(np.int16))
    assert float(delta[:, : source.shape[1] // 2][regions["face"][:, : source.shape[1] // 2]].mean()) > 1.0
    assert not np.any(delta[:, source.shape[1] // 2 + 1 :])


def test_red_eye_removal_preserves_selection_highlights_and_red_objects() -> None:
    for scale in (1, 2, 4):
        source, regions = portrait_scene(scale)
        selection = np.zeros(source.shape[:2], dtype=np.uint8)
        selection[int(source.shape[0] * 0.38) : int(source.shape[0] * 0.58), int(source.shape[1] * 0.32) : int(source.shape[1] * 0.68)] = 255
        result = reduce_red_eye(source, selection, 0.9, 0.3, 0.2, 2.0 * scale)
        before_dominance = source[:, :, 0].astype(np.float32) - source[:, :, 1:3].mean(axis=2)
        after_dominance = result[:, :, 0].astype(np.float32) - result[:, :, 1:3].mean(axis=2)
        assert float(after_dominance[regions["eyes"]].mean()) < float(before_dominance[regions["eyes"]].mean()) * 0.55
        assert np.array_equal(result[regions["red_object"]], source[regions["red_object"]])
        assert np.array_equal(result[regions["highlights"]], source[regions["highlights"]])
        protected_skin = selection.astype(bool) & ~regions["eyes"] & ~regions["highlights"]
        assert np.array_equal(result[protected_skin], source[protected_skin])
        assert np.array_equal(result[:, :, 3], source[:, :, 3])


def test_automatic_red_eye_detection_rejects_large_red_surface() -> None:
    source, regions = portrait_scene(2)
    result = reduce_red_eye(source, None, 0.9, 0.3, 0.2, 3.0)
    assert not np.array_equal(result[regions["eyes"]], source[regions["eyes"]])
    assert np.array_equal(result[regions["red_object"]], source[regions["red_object"]])


def test_patch_healing_matches_target_tone_and_keeps_texture_after_project_roundtrip() -> None:
    document = Document.new(140, 90, (172, 142, 116, 255))
    layer = document.layer
    checker = ((np.indices((24, 28)).sum(axis=0) % 2) * 28 - 14).astype(np.int16)
    layer.pixels[12:36, 10:38, :3] = np.clip(72 + checker[:, :, None], 0, 255)
    layer.pixels[42:66, 86:114, :3] = 8
    selection = np.zeros((document.height, document.width), dtype=np.uint8)
    cv2.ellipse(selection, (100, 54), (14, 12), 0, 0, 360, 255, -1)
    document.selection_mask = selection
    assert document.patch_active_selection(10, 12, True, 0.9, 1.0)
    active = selection > 0
    healed = layer.pixels[:, :, :3][active]
    assert abs(float(healed.mean()) - 143.0) < 35.0
    assert float(healed.std()) > 5.0
    assert np.all(layer.pixels[:, :, 3] == 255)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "advanced-retouch.prdx"
        document.save_project(path)
        restored = Document.open_project(path)
    assert np.array_equal(restored.layer.pixels, layer.pixels)
