from __future__ import annotations

import tkinter as tk
import unittest

import pytest

import numpy as np

from uzyro.app import UZYROApp
from uzyro.core import Document, GradientEngine


RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)


def test_interpolation_spaces_are_real_and_preserve_endpoints() -> None:
    stops = [(0.0, RED), (1.0, BLUE)]
    rendered = {
        space: GradientEngine.render(3, 1, (0, 0), (2, 0), stops, interpolation_space=space)
        for space in GradientEngine.INTERPOLATION_SPACES
    }
    for image in rendered.values():
        assert tuple(image[0, 0]) == RED
        assert tuple(image[0, -1]) == BLUE
    assert tuple(rendered["srgb"][0, 1, :3]) == (128, 0, 128)
    assert rendered["linear_rgb"][0, 1, 0] > rendered["srgb"][0, 1, 0] + 50
    assert rendered["oklab"][0, 1, 1] > 50


def test_all_gradient_geometries_have_expected_axes() -> None:
    stops = [(0.0, (0, 0, 0, 255)), (1.0, (255, 255, 255, 255))]
    linear = GradientEngine.render(9, 9, (4, 4), (8, 4), stops, "linear")
    radial = GradientEngine.render(9, 9, (4, 4), (8, 4), stops, "radial")
    reflected = GradientEngine.render(9, 9, (4, 4), (8, 4), stops, "reflected")
    diamond = GradientEngine.render(9, 9, (4, 4), (8, 4), stops, "diamond")
    angular = GradientEngine.render(9, 9, (4, 4), (8, 4), stops, "angular")
    assert linear[4, 0, 0] == 0 and linear[4, 8, 0] == 255
    assert radial[4, 4, 0] == 0 and radial[0, 4, 0] == 255
    assert reflected[4, 0, 0] == 255 and reflected[4, 8, 0] == 255
    assert diamond[4, 4, 0] == 0 and diamond[0, 4, 0] == 255
    assert angular[4, 8, 0] == 0 and 60 <= angular[8, 4, 0] <= 68


def test_noise_gradient_is_seeded_and_honors_models_and_channels() -> None:
    stops = [(0.0, RED), (1.0, BLUE)]
    options = {
        "enabled": True,
        "roughness": 0.75,
        "color_model": "rgb",
        "seed": 41,
        "channels": [[0.2, 0.4], [0.5, 0.7], [0.8, 1.0]],
    }
    first = GradientEngine.render(512, 2, (0, 0), (511, 0), stops, noise=options)
    second = GradientEngine.render(512, 2, (0, 0), (511, 0), stops, noise=options)
    changed = GradientEngine.render(512, 2, (0, 0), (511, 0), stops, noise={**options, "seed": 42})
    assert np.array_equal(first, second)
    assert not np.array_equal(first, changed)
    assert 50 <= int(first[:, :, 0].min()) <= int(first[:, :, 0].max()) <= 103
    assert 127 <= int(first[:, :, 1].min()) <= int(first[:, :, 1].max()) <= 180
    assert 203 <= int(first[:, :, 2].min()) <= int(first[:, :, 2].max()) <= 255

    gray = GradientEngine.render(
        256, 1, (0, 0), (255, 0), stops,
        noise={**options, "color_model": "grayscale", "channels": [[0.25, 0.75]], "seed": 8},
    )
    assert np.array_equal(gray[:, :, 0], gray[:, :, 1])
    assert np.array_equal(gray[:, :, 1], gray[:, :, 2])


def test_dither_is_deterministic_subtle_and_origin_stable() -> None:
    stops = [(0.0, (64, 64, 64, 255)), (1.0, (192, 192, 192, 255))]
    plain = GradientEngine.render(256, 16, (0, 0), (255, 0), stops)
    dithered = GradientEngine.render(256, 16, (0, 0), (255, 0), stops, dither=True)
    repeated = GradientEngine.render(256, 16, (0, 0), (255, 0), stops, dither=True)
    assert np.array_equal(dithered, repeated)
    assert not np.array_equal(plain, dithered)
    difference = np.abs(dithered[:, :, :3].astype(np.int16) - plain[:, :, :3].astype(np.int16))
    assert int(difference.max()) <= 1

    whole = GradientEngine.render(64, 8, (0, 0), (63, 0), stops, dither=True)
    right = GradientEngine.render(32, 8, (0, 0), (63, 0), stops, origin=(32, 0), dither=True)
    assert np.array_equal(whole[:, 32:], right)


def test_professional_gradient_definition_survives_project_round_trip(tmp_path) -> None:
    document = Document.new(80, 60, (0, 0, 0, 0))
    definition = {
        "type": "diamond",
        "start": [8, 10],
        "end": [68, 48],
        "stops": [{"position": 0.0, "color": list(RED)}, {"position": 1.0, "color": list(BLUE)}],
        "interpolation_space": "oklab",
        "dither": True,
        "noise": {
            "enabled": True,
            "roughness": 0.35,
            "color_model": "hsv",
            "seed": 123,
            "restrict_colors": True,
            "channels": [[0.0, 0.7], [0.4, 1.0], [0.3, 0.9]],
        },
    }
    layer = document.add_shape_layer("rectangle", (8, 10, 68, 48), RED, BLUE, 0, gradient=definition)
    path = tmp_path / "gradient.prdx"
    document.save_project(path)
    restored = Document.open_project(path)
    assert restored.layers[-1].shape_data["gradient"] == layer.shape_data["gradient"]
    assert np.array_equal(restored.layers[-1].pixels, layer.pixels)


@pytest.mark.ui
@pytest.mark.ui_isolated
class GradientEditorUiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_clipboard_reader = UZYROApp.read_clipboard_image
        UZYROApp.read_clipboard_image = staticmethod(lambda: None)
        self.app = UZYROApp()
        self.app.update()

    def tearDown(self) -> None:
        self.app.destroy()
        UZYROApp.read_clipboard_image = staticmethod(self.original_clipboard_reader)

    def test_editor_changes_visible_settings_with_generation_mode(self) -> None:
        self.app.gradient_editor_dialog(self.app.current_gradient_definition())
        self.app.update()
        mode = self.app._gradient_editor_mode
        self.assertEqual(mode.get(), "Обычный")
        self.assertTrue(self.app._gradient_editor_standard_settings.winfo_ismapped())
        self.assertFalse(self.app._gradient_editor_noise_settings.winfo_ismapped())
        mode.set("Шумовой")
        self.app.update()
        self.assertFalse(self.app._gradient_editor_standard_settings.winfo_ismapped())
        self.assertTrue(self.app._gradient_editor_noise_settings.winfo_ismapped())
        self.assertTrue(self.app._gradient_editor_channel_bar.winfo_ismapped())


if __name__ == "__main__":
    unittest.main()
