from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from photoredactor.core import Document, render_layer_pixels


def make_image(path: Path, color: tuple[int, int, int, int], size: tuple[int, int] = (48, 36)) -> np.ndarray:
    pixels = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    pixels[:, :] = color
    Image.fromarray(pixels, "RGBA").save(path)
    return pixels


def test_layers_convert_to_nested_document_and_remain_editable(tmp_path: Path) -> None:
    document = Document.new(180, 130, (0, 0, 0, 0))
    first = document.layer
    first.name = "Base"
    first.pixels[25:90, 30:110] = (210, 55, 42, 255)
    document.add_shape_layer("ellipse", (75, 35, 155, 115), (45, 155, 225, 230))
    selected = {layer.id for layer in document.layers}
    smart = document.convert_layers_to_smart_object(selected)
    assert smart is not None
    assert len(document.layers) == 1
    assert smart.kind == "embedded"
    nested = document.active_smart_document()
    assert nested is not None
    assert len(nested.layers) == 2
    assert nested.layers[0].name == "Base"
    assert nested.layers[1].kind == "shape"

    before = smart.pixels.copy()
    nested.layers[0].pixels[25:90, 30:110] = (40, 215, 95, 255)
    nested.layers[0].touch_pixels()
    assert document.update_active_smart_document(nested)
    assert not np.array_equal(smart.pixels, before)
    assert smart.smart_data["nested_document"]["layers"][1]["kind"] == "shape"

    project = tmp_path / "nested.prdx"
    document.save_project(project)
    restored = Document.open_project(project)
    restored_nested = restored.active_smart_document()
    assert restored_nested is not None
    assert len(restored_nested.layers) == 2
    assert restored_nested.layers[1].shape_data is not None
    assert np.array_equal(restored.layer.smart_source, smart.smart_source)


def test_nested_project_can_be_placed_linked_and_updated(tmp_path: Path) -> None:
    nested_path = tmp_path / "source.prdx"
    nested = Document.new(72, 54, (30, 80, 160, 255))
    nested.save_project(nested_path)
    host = Document.new(120, 90, (0, 0, 0, 0))
    layer = host.place_project(nested_path, linked=True)
    assert layer.kind == "linked"
    assert layer.smart_data["content_type"] == "document"
    assert host.linked_layer_status()["status"] == "current"
    original = layer.pixels.copy()

    changed = Document.open_project(nested_path)
    changed.layer.pixels[:, :] = (210, 65, 45, 255)
    changed.layer.touch_pixels()
    changed.save_project(nested_path)
    assert host.linked_layer_status()["status"] == "modified"
    assert host.resolve_linked_conflict("update")
    assert not np.array_equal(layer.pixels, original)
    assert host.active_smart_document() is not None


def test_link_conflict_uses_hash_even_if_size_and_timestamp_match(tmp_path: Path) -> None:
    source = tmp_path / "linked.bmp"
    make_image(source, (220, 50, 40, 255), (32, 24))
    document = Document.new(60, 50)
    layer = document.place_image(source, linked=True)
    saved_stat = source.stat()
    make_image(source, (40, 80, 220, 255), (32, 24))
    os.utime(source, ns=(saved_stat.st_atime_ns, saved_stat.st_mtime_ns))
    assert source.stat().st_size == layer.smart_data["fingerprint"]["size"]
    assert document.linked_layer_status()["status"] == "modified"
    cached = layer.smart_source.copy()
    assert document.resolve_linked_conflict("embed")
    assert layer.kind == "embedded"
    assert np.array_equal(layer.smart_source, cached)


def test_smart_filters_and_advanced_transform_stay_non_destructive(tmp_path: Path) -> None:
    source_path = tmp_path / "source.png"
    source = make_image(source_path, (70, 120, 190, 255), (80, 60))
    document = Document.new(140, 110, (0, 0, 0, 0))
    layer = document.place_image(source_path)
    layer.smart_source[18:42, 25:55, :3] = (210, 55, 40)
    render_before = layer.smart_source.copy()
    layer.smart_source = render_before
    from photoredactor.core import render_smart_object
    render_smart_object(layer)
    immutable = layer.smart_source.copy()
    layer.filters = [{"type": "blur", "radius": 3, "opacity": 1.0}]
    filtered = render_layer_pixels(layer)
    assert not np.array_equal(filtered, layer.pixels)
    document.set_active_layer_advanced_transform("perspective", [(8, 5), (91, 12), (84, 72), (2, 65)])
    transformed = layer.pixels.copy()
    assert np.array_equal(layer.smart_source, immutable)
    assert layer.transform_data is not None
    assert document.reset_active_layer_advanced_transform()
    assert np.array_equal(layer.smart_source, immutable)
    assert layer.filters
    assert not np.array_equal(transformed, layer.pixels)


def test_replacing_smart_contents_preserves_advanced_transform(tmp_path: Path) -> None:
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    make_image(first_path, (210, 55, 40, 255), (70, 50))
    make_image(second_path, (40, 190, 95, 255), (110, 80))
    document = Document.new(180, 130, (0, 0, 0, 0))
    layer = document.place_image(first_path)
    corners = [(18, 12), (112, 4), (105, 91), (9, 78)]
    document.set_active_layer_advanced_transform("perspective", corners)
    points_before = [list(point) for point in layer.transform_data["points"]]
    base_before = (layer.transform_data["base_x"], layer.transform_data["base_y"])
    assert document.replace_active_smart_contents(second_path, linked=False)
    assert layer.transform_data is not None
    restored_points = [
        [point[0] + layer.transform_data["base_x"], point[1] + layer.transform_data["base_y"]]
        for point in layer.transform_data["points"]
    ]
    original_document_points = [[point[0] + base_before[0], point[1] + base_before[1]] for point in points_before]
    np.testing.assert_allclose(restored_points, original_document_points)
    assert np.array_equal(layer.smart_source[:, :, :3], np.full((80, 110, 3), (40, 190, 95), dtype=np.uint8))


def test_nested_document_preserves_high_precision_layer_payload() -> None:
    document = Document.new(32, 24, (20, 40, 80, 255))
    document.set_bit_depth(16)
    working = document.layer.working_pixels.copy()
    layer_id = document.layer.id
    smart = document.convert_layers_to_smart_object({layer_id})
    assert smart is not None
    nested = document.active_smart_document()
    assert nested is not None
    assert nested.bit_depth == 16
    assert nested.layer.id == layer_id
    assert nested.layer.working_pixels is not None
    assert nested.layer.working_pixels.dtype == np.uint16
    assert np.array_equal(nested.layer.working_pixels, working)
