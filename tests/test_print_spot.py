from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile

from photoredactor.core import Document, Layer
from photoredactor.app_mixins.shortcuts import ShortcutsMixin
from photoredactor.print_pipeline import export_color_separations, spot_separations
from photoredactor.spot_colors import (
    SpotColor,
    assign_spot_color,
    assigned_spot_color,
    document_spot_colors,
    load_library,
    replace_document_spot_colors,
    save_library,
)
from photoredactor.windows_print import PrintPlacement, PrinterPage, WindowsPrinter, calculate_placement, rgba_to_bgra_on_paper


def test_ase_and_native_library_roundtrip_preserves_named_spots(tmp_path: Path) -> None:
    colors = [
        SpotColor("Фирменный зелёный", (52.4, -41.0, 23.5), (24, 143, 91), "Моя типография", "green"),
        SpotColor("Тёплый серый", (70.0, 2.5, 4.0), (177, 170, 166), "Моя типография", "gray"),
    ]
    ase_path = tmp_path / "library.ase"
    save_library(ase_path, colors)
    ase_colors = load_library(ase_path)
    assert [color.name for color in ase_colors] == [color.name for color in colors]
    assert all(color.source == "PhotoRedactor" for color in ase_colors)
    np.testing.assert_allclose([color.lab for color in ase_colors], [color.lab for color in colors], atol=1e-4)

    native_path = tmp_path / "library.prswatches"
    save_library(native_path, colors)
    native_colors = load_library(native_path)
    assert native_colors == colors


def test_spot_assignment_survives_project_and_builds_layer_plate(tmp_path: Path) -> None:
    document = Document.new(8, 7, (0, 0, 0, 0))
    pixels = np.zeros((3, 4, 4), dtype=np.uint8)
    pixels[:, :, :3] = (10, 20, 30)
    pixels[:, :, 3] = 255
    layer = Layer("Плашка", pixels, x=2, y=1, opacity=0.5)
    document.layers.append(layer)
    document.active_layer = 1
    color = SpotColor("Special Orange", (65.0, 40.0, 65.0), (230, 115, 35), "Licensed Book", "orange")
    replace_document_spot_colors(document, [color])
    assign_spot_color(document, layer.id, color.id)

    plates = spot_separations(document)
    mask = plates[color.id]["mask"]
    assert mask.shape == (7, 8)
    assert np.all(mask[1:4, 2:6] == 127)
    assert np.count_nonzero(mask) == 12

    path = tmp_path / "spots.prdx"
    document.save_project(path)
    restored = Document.open_project(path)
    assert document_spot_colors(restored)[0].name == color.name
    assert assigned_spot_color(restored, restored.layer.id).id == color.id


def test_export_color_separations_writes_process_and_spot_plates(monkeypatch, tmp_path: Path) -> None:
    document = Document.new(5, 4, (0, 0, 0, 0))
    color = SpotColor("Spot / Blue", (45.0, 12.0, -50.0), (55, 95, 180), "Press", "blue")
    replace_document_spot_colors(document, [color])
    assign_spot_color(document, document.layer.id, color.id)
    fake_cmyk = np.full((4, 5, 4), 64, dtype=np.uint8)
    monkeypatch.setattr("photoredactor.print_pipeline.cmyk_separation", lambda *_args, **_kwargs: fake_cmyk)
    monkeypatch.setattr("photoredactor.print_pipeline.profile_details", lambda _profile: {"name": "Test CMYK", "color_space": "CMYK"})

    manifest = export_color_separations(document, tmp_path / "plates", b"profile")
    paths = sorted(manifest.parent.glob("*.tif"))
    assert len(paths) == 5
    spot_paths = [path for path in paths if "_Spot_" in path.name]
    assert len(spot_paths) == 1 and "/" not in spot_paths[0].name
    assert all(tifffile.imread(path).shape == (4, 5) for path in paths)
    assert '"type": "spot"' in manifest.read_text(encoding="utf-8")


def test_print_placement_and_alpha_compositing_are_deterministic() -> None:
    page = PrinterPage(2400, 3300, 90, 120, 300, 300)
    assert calculate_placement(1200, 1800, 300, page, True) == PrintPlacement(190, 120, 2200, 3300)
    assert calculate_placement(1200, 1800, 300, page, False) == PrintPlacement(690, 870, 1200, 1800)
    pixels = np.array([[[255, 0, 0, 255], [0, 0, 0, 0], [0, 0, 255, 128]]], dtype=np.uint8)
    bgra = rgba_to_bgra_on_paper(pixels)
    np.testing.assert_array_equal(bgra[0, 0], [0, 0, 255, 255])
    np.testing.assert_array_equal(bgra[0, 1], [255, 255, 255, 255])
    np.testing.assert_allclose(bgra[0, 2], [255, 127, 127, 255], atol=1)


def test_windows_print_driver_completes_gdi_document() -> None:
    calls: list[str] = []

    def call(name, result=1):
        def invoke(*_args):
            calls.append(name)
            return result

        return invoke

    printer = WindowsPrinter.__new__(WindowsPrinter)
    printer.choose_printer = lambda _owner: SimpleNamespace(hDC=123, hDevMode=None, hDevNames=None)
    printer.page_details = lambda _hdc: PrinterPage(1000, 1000, 0, 0, 300, 300)
    printer.gdi32 = SimpleNamespace(
        StartDocW=call("StartDoc"), StartPage=call("StartPage"), SetStretchBltMode=call("Mode"),
        StretchDIBits=call("Pixels"), EndPage=call("EndPage"), EndDoc=call("EndDoc"),
        AbortDoc=call("AbortDoc"), DeleteDC=call("DeleteDC"),
    )
    printer.kernel32 = SimpleNamespace(GlobalFree=call("GlobalFree"))
    pixels = np.full((10, 12, 4), 255, dtype=np.uint8)
    assert printer.print_rgba(pixels, "Test", 300, fit_to_page=True)
    assert calls == ["StartDoc", "StartPage", "Mode", "Pixels", "EndPage", "EndDoc", "DeleteDC"]


def test_control_p_reaches_system_print_command() -> None:
    calls: list[str] = []
    app = SimpleNamespace(shortcut_print=lambda _event=None: calls.append("print") or "break")
    event = SimpleNamespace(keycode=80, keysym="p", state=0x0004)
    assert ShortcutsMixin.shortcut_control_key(app, event) == "break"
    assert calls == ["print"]
