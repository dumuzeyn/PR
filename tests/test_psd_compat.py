from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from psd_tools import PSDImage
from psd_tools.constants import Tag

from photoredactor.color_management import builtin_srgb_bytes, color_settings
from photoredactor.core import Document, Layer
from photoredactor.psd_compat import _compatibility_risks, export_psd, load_psd


def test_layered_psd_roundtrip_preserves_editable_raster_properties(tmp_path: Path) -> None:
    document = Document.new(30, 22, (12, 24, 36, 255))
    document.dpi = 144
    document.assign_color_profile(builtin_srgb_bytes())
    document.layer.name = "Нижний слой"
    document.layer.locked = True
    pixels = np.full((9, 13, 4), (210, 50, 70, 225), dtype=np.uint8)
    top = Layer("Верхний слой", pixels, x=5, y=4, opacity=0.65, blend_mode="Multiply", clipping=True)
    top.visible = False
    top.mask = np.tile(np.linspace(0, 255, 13, dtype=np.uint8), (9, 1))
    document.layers.append(top)
    document.active_layer = 1
    path = tmp_path / "layered.psd"

    report = export_psd(document, path)
    restored = load_psd(path, Document)
    assert report == {"path": str(path), "format": "PSD", "warnings": []}
    assert (restored.width, restored.height, restored.dpi, restored.bit_depth) == (30, 22, 144, 8)
    assert [layer.name for layer in restored.layers] == ["Нижний слой", "Верхний слой"]
    assert restored.layers[0].locked
    loaded = restored.layers[1]
    assert (loaded.x, loaded.y) == (5, 4)
    assert abs(loaded.opacity - 0.65) < 0.005
    assert loaded.blend_mode == "Multiply"
    assert not loaded.visible
    assert loaded.clipping
    np.testing.assert_array_equal(loaded.mask, top.mask)
    assert color_settings(restored.metadata)["icc_base64"] == color_settings(document.metadata)["icc_base64"]


def test_small_psb_is_written_with_psb_header_and_reopens(tmp_path: Path) -> None:
    document = Document.new(18, 12, (40, 60, 80, 255))
    path = tmp_path / "small.psb"
    export_psd(document, path)
    assert PSDImage.open(path).version == 2
    restored = Document.from_image(path)
    assert restored.metadata["psd_compatibility"]["format"] == "PSB"
    assert restored.layers[0].name == "Background"


def test_16_bit_psd_restores_real_high_precision_pixels(tmp_path: Path) -> None:
    document = Document.new(11, 7, (10, 20, 30, 255))
    document.set_bit_depth(16)
    path = tmp_path / "depth16.psd"
    export_psd(document, path)
    restored = load_psd(path, Document)
    assert restored.bit_depth == 16
    assert restored.layer.working_pixels is not None
    assert restored.layer.working_pixels.dtype == np.uint16
    np.testing.assert_array_equal(restored.layer.working_pixels[0, 0], np.array((2570, 5140, 7710, 65535), dtype=np.uint16))


def test_psd_groups_are_kept_in_leaf_names_and_order(tmp_path: Path) -> None:
    psd = PSDImage.new("RGBA", (20, 16), color=(0, 0, 0, 0))
    group = psd.create_group(name="Products")
    group.name = "Товары"
    child = psd.create_pixel_layer(Image.new("RGBA", (7, 5), (220, 80, 30, 255)), name="Cup", top=3, left=4)
    child.name = "Чашка"
    group.append(child)
    path = tmp_path / "group.psd"
    psd.save(path)
    restored = load_psd(path, Document)
    assert [layer.name for layer in restored.layers] == ["Товары / Чашка"]
    assert (restored.layer.x, restored.layer.y) == (4, 3)


def test_complex_layers_are_reported_in_compatible_export(tmp_path: Path) -> None:
    document = Document.new(24, 18, (0, 0, 0, 0))
    text = document.add_text_layer("Текст", 2, 3, (255, 255, 255, 255), 14)
    text.filters = [{"type": "blur", "radius": 1}]
    text.effects = {"drop_shadow": {"enabled": True, "x": 2, "y": 2, "blur": 2, "opacity": 0.5}}
    path = tmp_path / "compatible.psd"
    report = export_psd(document, path)
    assert any("пиксельный" in warning for warning in report["warnings"])
    reopened = PSDImage.open(path)
    assert [layer.name for layer in reopened] == ["Background", f"{text.name} · эффект 1", text.name]


def test_adjustment_export_has_one_visible_merged_preview(tmp_path: Path) -> None:
    document = Document.new(12, 9, (30, 60, 90, 255))
    document.layers.append(Layer("Инверсия", np.zeros((9, 12, 4), dtype=np.uint8), kind="adjustment", adjustment={"type": "invert"}))
    path = tmp_path / "adjustment.psd"
    report = export_psd(document, path)
    reopened = PSDImage.open(path)
    visible = [layer.name for layer in reopened if layer.visible]
    assert visible == ["Сведённая визуальная копия"]
    assert any("Корректирующие" in warning for warning in report["warnings"])


def test_only_real_photoshop_editability_risks_are_reported() -> None:
    plain = SimpleNamespace(name="Фото", kind="pixel", tagged_blocks={})
    assert _compatibility_risks(plain) == []
    complex_layer = SimpleNamespace(
        name="Логотип", kind="smartobject",
        tagged_blocks={Tag.OBJECT_BASED_EFFECTS_LAYER_INFO: object(), Tag.VECTOR_MASK_SETTING1: object()},
    )
    risks = _compatibility_risks(complex_layer)
    assert len(risks) == 3
    assert any("Smart Object" in item for item in risks)
    assert any("Эффекты" in item for item in risks)
    assert any("Векторные" in item for item in risks)
