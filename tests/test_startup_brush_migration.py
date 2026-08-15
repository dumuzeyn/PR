from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

from photoredactor.app import PhotoRedactorApp


class BrushSettingsMigrationTests(unittest.TestCase):
    def test_polluted_ui_test_profile_is_replaced_with_writable_defaults(self) -> None:
        settings_path = Path(os.environ["LOCALAPPDATA"]) / "PhotoRedactor" / "settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(json.dumps({
            "tool_settings": {"brush": {
                "size": 18, "hardness": 0.17, "opacity": 0.83, "flow": 0.3,
                "spacing": 0.2, "smoothing": 0.15, "blend_mode": "Overlay",
            }},
            "brush_advanced": {"angle": 37.0, "roundness": 0.42, "scatter": 1.25},
        }), encoding="utf-8")
        original = PhotoRedactorApp.read_clipboard_image
        PhotoRedactorApp.read_clipboard_image = staticmethod(lambda: None)
        app = PhotoRedactorApp()
        try:
            self.assertEqual(app.brush_size.get(), 28)
            self.assertEqual(app.brush_blend_mode.get(), "Normal")
            self.assertAlmostEqual(app.opacity.get(), 1.0)
            self.assertAlmostEqual(app.brush_flow.get(), 1.0)
            self.assertEqual(app.brush_advanced["angle"], 0.0)
            self.assertEqual(app.brush_advanced["scatter"], 0.0)
        finally:
            app.destroy()
            PhotoRedactorApp.read_clipboard_image = staticmethod(original)


if __name__ == "__main__":
    unittest.main()
