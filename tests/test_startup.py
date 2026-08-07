from __future__ import annotations

import tkinter as tk
import time
import unittest

import numpy as np
from PIL import Image

from photoredactor.app import PhotoRedactorApp


class StartupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_clipboard_reader = PhotoRedactorApp.read_clipboard_image
        PhotoRedactorApp.read_clipboard_image = staticmethod(lambda: None)
        self.app = PhotoRedactorApp()
        self.app.update()

    def tearDown(self) -> None:
        self.app.destroy()
        PhotoRedactorApp.read_clipboard_image = staticmethod(self.original_clipboard_reader)

    def test_editor_is_hidden_on_startup(self) -> None:
        self.assertIsNotNone(self.app.startup_frame)
        self.assertTrue(self.app.startup_frame.winfo_ismapped())
        self.assertFalse(self.app.editor_root.winfo_ismapped())
        self.assertEqual(str(self.app.cget("menu")), "")
        self.assertEqual(self.app.state(), "normal")
        expected_x = (self.app.winfo_screenwidth() - self.app.winfo_width()) // 2
        expected_y = (self.app.winfo_screenheight() - self.app.winfo_height()) // 2
        self.assertAlmostEqual(self.app.winfo_x(), expected_x, delta=3)
        self.assertAlmostEqual(self.app.winfo_y(), expected_y, delta=3)

    def test_clipboard_preset_is_first(self) -> None:
        clipboard = Image.new("RGBA", (321, 187), (20, 120, 220, 180))
        presets = self.app.available_document_presets(clipboard)
        self.assertEqual((presets[0]["width"], presets[0]["height"]), (321, 187))
        self.assertTrue(presets[0]["clipboard"])

    def test_clipboard_document_keeps_exact_size_and_pixels(self) -> None:
        clipboard = Image.new("RGBA", (321, 187), (20, 120, 220, 180))
        self.app.create_document_from_settings(
            {
                "width": 321,
                "height": 187,
                "dpi": 72,
                "background": (0, 0, 0, 0),
                "include_clipboard": True,
            },
            clipboard,
        )
        self.app.update()
        self.assertEqual((self.app.doc.width, self.app.doc.height), (321, 187))
        self.assertEqual(self.app.doc.metadata["source"], "clipboard")
        self.assertTrue(np.array_equal(self.app.doc.layer.pixels[20, 20], np.array((20, 120, 220, 180), dtype=np.uint8)))
        self.assertTrue(self.app.editor_root.winfo_ismapped())
        self.assertNotEqual(str(self.app.cget("menu")), "")
        self.assertEqual(self.app.state(), "zoomed")

    def test_new_document_dialog_builds_clipboard_preview(self) -> None:
        clipboard = Image.new("RGBA", (640, 360), (40, 100, 180, 255))
        original_wait = tk.Toplevel.wait_window
        shown_size: list[str] = []

        def inspect_dialog(window, target=None) -> None:
            shown_size.append(str(self.app._new_document_size_label.cget("text")))
            window.destroy()

        tk.Toplevel.wait_window = inspect_dialog
        try:
            self.assertIsNone(self.app.new_document_dialog(clipboard))
            self.assertIsNotNone(self.app._new_document_preview)
            self.assertEqual(shown_size, ["640 x 360 px"])
        finally:
            tk.Toplevel.wait_window = original_wait

    def test_custom_canvas_size_is_reused(self) -> None:
        self.app.save_settings = lambda: None
        self.app.remember_custom_canvas(777, 555, 144, "Прозрачный")
        custom = next(preset for preset in self.app.available_document_presets() if preset["name"] == "Свой размер")
        self.assertEqual((custom["width"], custom["height"]), (777, 555))
        self.assertEqual(custom["dpi"], 144)
        self.assertEqual(custom["background"], "Прозрачный")

    def test_zoom_preserves_viewport_center(self) -> None:
        self.app.create_document_from_settings(
            {
                "width": 1600,
                "height": 1000,
                "dpi": 72,
                "background": (255, 255, 255, 255),
                "include_clipboard": False,
            }
        )
        self.app.update()
        self.app.fit_to_screen()
        self.app.update()

        def viewport_center() -> tuple[float, float]:
            origin_x, origin_y = self.app._canvas_origin
            return (
                (self.app.canvas.canvasx(self.app.canvas.winfo_width() / 2) - origin_x) / self.app.zoom.get(),
                (self.app.canvas.canvasy(self.app.canvas.winfo_height() / 2) - origin_y) / self.app.zoom.get(),
            )

        fitted = viewport_center()
        self.assertAlmostEqual(fitted[0], 800.0, delta=5.0)
        self.assertAlmostEqual(fitted[1], 500.0, delta=5.0)
        self.app.center_canvas_on_doc(620.0, 410.0)
        self.app.update()
        before = viewport_center()
        self.app.set_zoom(self.app.zoom.get() * 1.8)
        self.app.update()
        after = viewport_center()
        self.assertAlmostEqual(after[0], before[0], delta=5.0)
        self.assertAlmostEqual(after[1], before[1], delta=5.0)

    def test_new_document_recenters_after_previous_view(self) -> None:
        settings = {
            "width": 1200,
            "height": 800,
            "dpi": 72,
            "background": (255, 255, 255, 255),
            "include_clipboard": False,
        }
        self.app.create_document_from_settings(settings)
        self.app.update()
        self.app.center_canvas_on_doc(80.0, 60.0)
        self.app.create_document_from_settings({**settings, "width": 900, "height": 600})
        deadline = time.monotonic() + 0.4
        while time.monotonic() < deadline:
            self.app.update()
            time.sleep(0.01)
        origin_x, origin_y = self.app._canvas_origin
        viewport_x = (self.app.canvas.canvasx(self.app.canvas.winfo_width() / 2) - origin_x) / self.app.zoom.get()
        viewport_y = (self.app.canvas.canvasy(self.app.canvas.winfo_height() / 2) - origin_y) / self.app.zoom.get()
        self.assertAlmostEqual(viewport_x, 450.0, delta=5.0)
        self.assertAlmostEqual(viewport_y, 300.0, delta=5.0)


if __name__ == "__main__":
    unittest.main()
