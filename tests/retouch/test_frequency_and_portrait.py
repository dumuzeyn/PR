from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from uzyro.core import (
    Document,
    Layer,
    blend_rgb,
    frequency_separation,
    portrait_cleanup,
    spot_heal,
)


class RetouchTests(unittest.TestCase):
    def test_frequency_separation_reconstructs_source(self) -> None:
        rng = np.random.default_rng(12)
        source = rng.integers(0, 256, (96, 128, 4), dtype=np.uint8)
        source[:, :, 3] = 255
        low, high = frequency_separation(source, 7.0, 1.0)
        rebuilt = blend_rgb(high[:, :, :3], low[:, :, :3], "Linear Light")
        difference = np.abs(rebuilt.astype(np.int16) - source[:, :, :3].astype(np.int16))
        self.assertLessEqual(int(difference.max()), 2)

    def test_frequency_layers_survive_project_roundtrip(self) -> None:
        document = Document.new(64, 48, (90, 120, 150, 255))
        before = document.composite()
        self.assertTrue(document.frequency_separate_active(5.0, 1.0))
        self.assertFalse(document.layers[0].visible)
        self.assertEqual(document.layers[-1].blend_mode, "Linear Light")
        self.assertLessEqual(int(np.abs(document.composite().astype(np.int16) - before.astype(np.int16)).max()), 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retouch.prdx"
            document.save_project(path)
            restored = Document.open_project(path)
        self.assertEqual(restored.layers[-1].blend_mode, "Linear Light")
        self.assertEqual(len(restored.layers), 3)

    def test_spot_heal_removes_compact_defect(self) -> None:
        pixels = np.full((80, 80, 4), (176, 132, 112, 255), dtype=np.uint8)
        pixels[37:44, 37:44, :3] = 12
        layer = Layer("portrait", pixels.copy())
        self.assertIsNotNone(spot_heal(layer, 40, 40, 8, 1.0))
        self.assertGreater(float(layer.pixels[40, 40, :3].mean()), 100.0)
        self.assertTrue(np.all(layer.pixels[:, :, 3] == 255))

    def test_portrait_cleanup_preserves_non_skin_and_alpha(self) -> None:
        source = np.full((72, 96, 4), (190, 140, 118, 255), dtype=np.uint8)
        noise = np.random.default_rng(4).integers(-20, 21, (72, 96, 1))
        source[:, :, :3] = np.clip(source[:, :, :3].astype(np.int16) + noise, 0, 255)
        source[:10, :10, :3] = (20, 70, 210)
        cleaned = portrait_cleanup(source, 0.65, 0.55, 0.35, 0.4)
        self.assertFalse(np.array_equal(cleaned, source))
        self.assertTrue(np.array_equal(cleaned[4, 4], source[4, 4]))
        self.assertTrue(np.array_equal(cleaned[:, :, 3], source[:, :, 3]))


if __name__ == "__main__":
    unittest.main()
