from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
import uzyro.document_mixins.project_io as project_io

from uzyro.color_management import (
    builtin_srgb_bytes,
    convert_icc,
    profile_details,
    rgba_to_working,
    working_to_rgba,
)
from uzyro.core import Document
from uzyro.history import History, PixelPatchCommand
from uzyro.print_pipeline import export_cmyk_tiff, print_preflight, proof_document


def system_cmyk_profile() -> Path | None:
    color_dir = Path("C:/Windows/System32/spool/drivers/color")
    preferred = color_dir / "RSWOP.icm"
    candidates = [preferred] if preferred.exists() else []
    candidates.extend(color_dir.glob("*.ic[cm]"))
    for path in candidates:
        try:
            if profile_details(path)["color_space"].upper() == "CMYK":
                return path
        except Exception:
            continue
    return None


def high_precision_gradient(width: int = 1024) -> np.ndarray:
    values = np.linspace(0.12, 0.88, width, dtype=np.float32)
    rgb = np.stack((values, values ** 1.1, np.sqrt(values)), axis=1)[None, :, :]
    return np.dstack((rgb, np.ones((1, width), dtype=np.float32)))


def test_working_models_roundtrip_and_display_edit_preserves_unmodified_precision() -> None:
    source = np.tile(high_precision_gradient(300), (16, 1, 1))
    document = Document.new(300, 16)
    document.set_bit_depth(32)
    layer = document.layer
    layer.set_working_rgba(source, 32, "RGBA")
    before = layer.working_rgba().copy()
    layer.pixels[4, 20, 0] = 240
    layer.touch_pixels()
    after = layer.working_rgba()
    np.testing.assert_array_equal(after[:, 21:], before[:, 21:])
    assert after[4, 20, 0] == pytest.approx(240 / 255.0)
    for model in ("Lab", "CMYK", "RGBA"):
        working = rgba_to_working(source, model, 32)
        restored = working_to_rgba(working, model)
        assert np.max(np.abs(restored - source)) < (0.03 if model == "Lab" else 0.002)


def test_precision_composite_keeps_more_than_eight_bit_tonal_steps() -> None:
    source = high_precision_gradient(1024)
    document = Document.new(1024, 1, (0, 0, 0, 0))
    document.set_bit_depth(32)
    document.layer.set_working_rgba(source, 32, "RGBA")
    precise = document.composite_precision(False)
    display = document.composite(False)
    assert precise.dtype == np.float32
    assert np.unique(precise[:, :, 0]).size > 900
    assert np.unique(display[:, :, 0]).size <= 256


def test_high_precision_survives_resize_rotate_flip_crop_and_warp() -> None:
    source = np.tile(high_precision_gradient(320), (24, 1, 1))
    document = Document.new(320, 24)
    document.set_bit_depth(32)
    document.layer.set_working_rgba(source, 32, "RGBA")
    document.resize_image(480, 36)
    assert np.unique(document.layer.working_rgba()[:, :, 0]).size > 256
    document.transform_active_layer(angle=7.5, flip_horizontal=True)
    assert document.layer.working_pixels is not None
    assert document.layer.working_pixels.dtype == np.float32
    document.warp_active_layer("wave", 0.08, 80.0)
    document.crop((20, 4, document.width - 20, document.height - 4))
    assert np.unique(document.layer.working_rgba()[:, :, 0]).size > 256


def test_selected_pixel_transform_does_not_collapse_precision() -> None:
    document = Document.new(320, 24)
    document.set_bit_depth(32)
    document.layer.set_working_rgba(np.tile(high_precision_gradient(320), (24, 1, 1)), 32, "RGBA")
    document.selection_mask = np.zeros((24, 320), dtype=np.uint8)
    document.selection_mask[2:22, 20:300] = 255
    assert document.transform_selected_pixels(x=20, y=2, width=290, height=20, angle=1.5)
    precise = document.layer.working_rgba()
    assert document.layer.working_pixels.dtype == np.float32
    assert np.unique(precise[:, :, 0]).size > 256


def test_icc_identity_preserves_float_values_exactly() -> None:
    source = high_precision_gradient(513)
    profile = builtin_srgb_bytes()
    converted = convert_icc(source, profile, profile, "relative", True)
    np.testing.assert_array_equal(converted, source)
    details = profile_details(profile)
    assert details["color_space"].upper() == "RGB"


def test_assign_profile_keeps_pixels_while_convert_recalculates(monkeypatch: pytest.MonkeyPatch) -> None:
    document = Document.new(128, 4)
    document.set_bit_depth(32)
    source = np.tile(high_precision_gradient(128), (4, 1, 1))
    document.layer.set_working_rgba(source, 32, "RGBA")
    profile = builtin_srgb_bytes()
    before = document.layer.working_rgba().copy()
    document.assign_color_profile(profile)
    np.testing.assert_array_equal(document.layer.working_rgba(), before)

    monkeypatch.setattr(project_io, "convert_icc", lambda pixels, *_args, **_kwargs: np.clip(pixels * 0.8, 0.0, 1.0))
    document.convert_color_profile(profile)
    assert not np.array_equal(document.layer.working_rgba(), before)


def test_precision_patch_undo_restores_exact_sub_eight_bit_values() -> None:
    document = Document.new(64, 8)
    document.set_bit_depth(32)
    document.layer.set_working_rgba(np.tile(high_precision_gradient(64), (8, 1, 1)), 32, "RGBA")
    layer = document.layer
    rect = (10, 2, 20, 6)
    before_precise = layer.working_rgba()[2:6, 10:20].copy()
    before_display = layer.pixels[2:6, 10:20].copy()
    edited = layer.working_rgba()
    edited[2:6, 10:20, :3] = 0.75
    layer.set_working_rgba(edited, 32, "RGBA")
    after_precise = layer.working_rgba()[2:6, 10:20].copy()
    command = PixelPatchCommand(
        "precision edit",
        layer.id,
        rect,
        before_display,
        layer.pixels[2:6, 10:20].copy(),
        before_precise,
        after_precise,
    )
    history = History()
    history.push(command)
    history.undo(document)
    np.testing.assert_array_equal(layer.working_rgba()[2:6, 10:20], before_precise)
    history.redo(document)
    np.testing.assert_array_equal(layer.working_rgba()[2:6, 10:20], after_precise)


def test_high_precision_tiff_roundtrip_contains_icc_and_dpi(tmp_path: Path) -> None:
    document = Document.new(512, 4)
    document.dpi = 360
    document.set_bit_depth(32)
    source = np.tile(high_precision_gradient(512), (4, 1, 1))
    document.layer.set_working_rgba(source, 32, "RGBA")
    document.assign_color_profile(builtin_srgb_bytes())
    path = tmp_path / "float-color.tif"
    document.export_flat(path)
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        restored = page.asarray()
        assert restored.dtype == np.float32
        assert page.tags["InterColorProfile"].count > 100
        assert page.tags["XResolution"].value == (360, 1)
    np.testing.assert_allclose(restored, source, atol=1e-6)


def test_real_cmyk_profile_softproof_preflight_and_export(tmp_path: Path) -> None:
    profile = system_cmyk_profile()
    if profile is None:
        pytest.skip("No real CMYK printer profile is installed")
    document = Document.new(96, 64, (190, 80, 35, 255))
    document.dpi = 300
    document.set_bit_depth(16)
    proofed, warning = proof_document(document, profile, "relative", True, True)
    report = print_preflight(document, profile, "relative", True, 300.0)
    assert proofed.shape == (64, 96, 4)
    assert warning.shape == (64, 96)
    assert report["profile"]["color_space"].upper() == "CMYK"
    assert report["maximum_ink"] > 0
    path = tmp_path / "press-ready.tif"
    export_cmyk_tiff(document, path, profile)
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        assert page.photometric.name == "SEPARATED"
        assert page.shape == (64, 96, 4)
        assert page.tags["InterColorProfile"].count == profile.stat().st_size


def test_raw_decoder_keeps_native_sixteen_bit_payload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rgb16 = np.arange(12 * 18 * 3, dtype=np.uint16).reshape(12, 18, 3) * 7

    class FakeRaw:
        metadata = SimpleNamespace(make="Test", model="Sensor", iso_speed=100, shutter=0.01, aperture=4.0, focal_len=35, timestamp=None)
        camera_whitebalance = [2.0, 1.0, 1.5, 1.0]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def postprocess(self, **_kwargs):
            return rgb16.copy()

    monkeypatch.setitem(sys.modules, "rawpy", SimpleNamespace(imread=lambda _path: FakeRaw()))
    raw_path = tmp_path / "sample.dng"
    raw_path.write_bytes(b"not read by the fake decoder")
    document = Document.from_raw(raw_path)
    assert document.bit_depth == 16
    assert document.layer.working_depth == 16
    assert document.layer.working_pixels.dtype == np.uint16
    np.testing.assert_array_equal(document.layer.working_pixels[:, :, :3], rgb16)
