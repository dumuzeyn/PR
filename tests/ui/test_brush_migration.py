from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

from uzyro.app import UZYROApp


class BrushSettingsMigrationTests(unittest.TestCase):
    def test_previous_factory_brush_is_updated_to_new_defaults(self) -> None:
        settings_path = Path(os.environ["LOCALAPPDATA"]) / "UZYRO" / "settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(json.dumps({
            "tool_settings": {"brush": {
                "size": 28, "hardness": 0.5, "opacity": 1.0, "flow": 1.0,
                "spacing": 0.25, "smoothing": 0.15, "blend_mode": "Normal",
            }},
            "brush_presets": {"Круглая кисть": {
                "size": 28, "hardness": 0.8, "opacity": 1.0, "flow": 1.0,
                "spacing": 0.25, "smoothing": 0.15, "blend_mode": "Normal",
            }},
        }), encoding="utf-8")
        original = UZYROApp.read_clipboard_image
        UZYROApp.read_clipboard_image = staticmethod(lambda: None)
        app = UZYROApp()
        try:
            self.assertAlmostEqual(app.hardness.get(), 1.0)
            self.assertAlmostEqual(app.brush_spacing.get(), 0.0)
            self.assertAlmostEqual(app.brush_presets["Круглая кисть"]["hardness"], 1.0)
            self.assertAlmostEqual(app.brush_presets["Круглая кисть"]["spacing"], 0.0)
        finally:
            app.destroy()
            UZYROApp.read_clipboard_image = staticmethod(original)

    def test_polluted_ui_test_profile_is_replaced_with_writable_defaults(self) -> None:
        settings_path = Path(os.environ["LOCALAPPDATA"]) / "UZYRO" / "settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(json.dumps({
            "tool_settings": {"brush": {
                "size": 18, "hardness": 0.17, "opacity": 0.83, "flow": 0.3,
                "spacing": 0.2, "smoothing": 0.15, "blend_mode": "Overlay",
            }},
            "brush_advanced": {"angle": 37.0, "roundness": 0.42, "scatter": 1.25},
        }), encoding="utf-8")
        original = UZYROApp.read_clipboard_image
        UZYROApp.read_clipboard_image = staticmethod(lambda: None)
        app = UZYROApp()
        try:
            self.assertEqual(app.brush_size.get(), 28)
            self.assertEqual(app.brush_blend_mode.get(), "Normal")
            self.assertAlmostEqual(app.hardness.get(), 1.0)
            self.assertAlmostEqual(app.brush_spacing.get(), 0.0)
            self.assertAlmostEqual(app.opacity.get(), 1.0)
            self.assertAlmostEqual(app.brush_flow.get(), 1.0)
            self.assertEqual(app.brush_advanced["angle"], 0.0)
            self.assertEqual(app.brush_advanced["scatter"], 0.0)
        finally:
            app.destroy()
            UZYROApp.read_clipboard_image = staticmethod(original)


if __name__ == "__main__":
    unittest.main()
