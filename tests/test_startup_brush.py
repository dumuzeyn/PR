from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from photoredactor.app import PhotoRedactorApp
from photoredactor.history import PixelTilePatchCommand


class BrushStartupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_clipboard_reader = PhotoRedactorApp.read_clipboard_image
        PhotoRedactorApp.read_clipboard_image = staticmethod(lambda: None)
        self.app = PhotoRedactorApp()
        self.app.create_document_from_settings(
            {"width": 240, "height": 140, "dpi": 72, "background": (255, 255, 255, 255), "include_clipboard": False}
        )
        self.app.update()

    def tearDown(self) -> None:
        self.app.destroy()
        PhotoRedactorApp.read_clipboard_image = staticmethod(self.original_clipboard_reader)

    def pointer(self, point: tuple[int, int], pressure: float = 1.0):
        x, y = self.app.doc_to_canvas(*point)
        return SimpleNamespace(x=x, y=y, state=0, pressure=pressure)

    def test_one_brush_drag_is_one_tile_undo_operation(self) -> None:
        self.app.tool.set("brush")
        self.app.brush_size.set(12)
        self.app.brush_flow.set(0.3)
        self.app.brush_spacing.set(0.2)
        self.app.history.clear()
        before = self.app.doc.layer.pixels.copy()

        self.app.pointer_down(self.pointer((30, 70)))
        for x in range(35, 211, 5):
            self.app.pointer_drag(self.pointer((x, 70)))
        self.app.pointer_up(self.pointer((210, 70)))

        self.assertEqual(len(self.app.history.undo_stack), 1)
        self.assertIsInstance(self.app.history.undo_stack[0], PixelTilePatchCommand)
        self.assertFalse(np.array_equal(before, self.app.doc.layer.pixels))
        self.app.undo()
        np.testing.assert_array_equal(self.app.doc.layer.pixels, before)

    def test_brush_settings_remain_independent_between_tools(self) -> None:
        self.app.tool.set("brush")
        self.app.hardness.set(0.17)
        self.app.brush_flow.set(0.23)
        self.app.brush_spacing.set(0.11)
        self.app.tool.set("eraser")
        self.app.hardness.set(0.82)
        self.app.brush_flow.set(0.74)
        self.app.tool.set("brush")
        self.assertAlmostEqual(self.app.hardness.get(), 0.17)
        self.assertAlmostEqual(self.app.brush_flow.get(), 0.23)
        self.assertAlmostEqual(self.app.brush_spacing.get(), 0.11)

    def test_custom_brush_preset_roundtrip(self) -> None:
        self.app.brush_preset_name.set("Тестовый")
        self.app.brush_size.set(47)
        self.app.brush_flow.set(0.31)
        self.app.brush_blend_mode.set("Overlay")
        self.app.save_brush_preset()
        self.app.brush_size.set(5)
        self.app.brush_flow.set(1.0)
        self.app.brush_blend_mode.set("Normal")
        self.app.apply_brush_preset()
        self.assertEqual(self.app.brush_size.get(), 47)
        self.assertAlmostEqual(self.app.brush_flow.get(), 0.31)
        self.assertEqual(self.app.brush_blend_mode.get(), "Overlay")
        self.app.delete_brush_preset()
        self.assertNotIn("Тестовый", self.app.brush_presets)

    def test_eraser_blur_and_sharpen_each_commit_one_undo(self) -> None:
        rng = np.random.default_rng(18)
        source = rng.integers(20, 235, self.app.doc.layer.pixels.shape, dtype=np.uint8)
        source[:, :, 3] = 255
        for tool in ("eraser", "blur_tool", "sharpen_tool"):
            self.app.doc.layer.pixels[:] = source
            self.app.doc.layer.touch_pixels()
            self.app.history.clear()
            self.app.tool.set(tool)
            self.app.brush_size.set(10)
            self.app.brush_spacing.set(0.2)
            self.app.pointer_down(self.pointer((50, 70)))
            self.app.pointer_drag(self.pointer((150, 70)))
            self.app.pointer_up(self.pointer((150, 70)))
            self.assertEqual(len(self.app.history.undo_stack), 1, tool)
            self.assertIsInstance(self.app.history.undo_stack[0], PixelTilePatchCommand)
            self.assertFalse(np.array_equal(source, self.app.doc.layer.pixels), tool)
            self.app.undo()
            np.testing.assert_array_equal(self.app.doc.layer.pixels, source)


if __name__ == "__main__":
    unittest.main()
