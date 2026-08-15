from __future__ import annotations

import unittest

import numpy as np

from photoredactor.app import PhotoRedactorApp


class CloneSourcesPanelTests(unittest.TestCase):
    def test_five_sources_keep_independent_points_and_transforms(self) -> None:
        original = PhotoRedactorApp.read_clipboard_image
        PhotoRedactorApp.read_clipboard_image = staticmethod(lambda: None)
        app = PhotoRedactorApp()
        try:
            app.ensure_clone_sources()
            self.assertEqual(tuple(app._clone_sources), ("A", "B", "C", "D", "E"))
            app.switch_clone_source("A")
            app.set_clone_source((18, 22))
            app.clone_scale_x.set(135)
            app.clone_rotation.set(17)
            app.clone_overlay_invert.set(True)
            app.switch_clone_source("B")
            app.set_clone_source((72, 64))
            app.clone_scale_x.set(80)
            app.clone_rotation.set(-28)
            app.clone_overlay_clipped.set(False)
            app.switch_clone_source("A")
            self.assertEqual(app._source_anchor.point, (18, 22))
            self.assertEqual(app.clone_scale_x.get(), 135)
            self.assertEqual(app.clone_rotation.get(), 17)
            self.assertTrue(app.clone_overlay_invert.get())
            app.switch_clone_source("B")
            self.assertEqual(app._source_anchor.point, (72, 64))
            self.assertEqual(app.clone_scale_x.get(), 80)
            self.assertEqual(app.clone_rotation.get(), -28)
            self.assertFalse(app.clone_overlay_clipped.get())
            payload = app.source_retouch_settings_payload()
            self.assertEqual(payload["active_source"], "B")
            self.assertEqual(len(payload["sources"]), 5)
            self.assertTrue(payload["sources"]["A"]["layer_id"])
            self.assertNotIn("_sample_pixels", payload["sources"]["A"])
            self.assertIsInstance(app._clone_sources["A"].get("_sample_pixels"), np.ndarray)
        finally:
            app.destroy()
            PhotoRedactorApp.read_clipboard_image = staticmethod(original)


if __name__ == "__main__":
    unittest.main()
