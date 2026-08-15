from __future__ import annotations

from types import SimpleNamespace
import tkinter as tk
import unittest

import numpy as np

from photoredactor.app import PhotoRedactorApp
from photoredactor.core import draw_brush


class Phase8UiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_clipboard_reader = PhotoRedactorApp.read_clipboard_image
        PhotoRedactorApp.read_clipboard_image = staticmethod(lambda: None)
        self.app = PhotoRedactorApp()
        self.app.update()

    def tearDown(self) -> None:
        self.app.destroy()
        PhotoRedactorApp.read_clipboard_image = staticmethod(self.original_clipboard_reader)

    def test_color_picker_supports_hex_alpha_rgb_hsv_and_hsl(self) -> None:
        original_wait = tk.Toplevel.wait_window

        def choose(window, target=None) -> None:
            window.update()
            variables = self.app._color_picker_vars
            self.assertEqual(set(variables), {"rgb", "hsv", "hsl", "hex", "alpha"})
            for variable, value in zip(variables["hsv"], (0, 100, 100)):
                variable.set(value)
            self.app._color_picker_apply_hsv()
            self.assertEqual(tuple(variable.get() for variable in variables["rgb"]), (255, 0, 0))
            for variable, value in zip(variables["hsl"], (120, 100, 50)):
                variable.set(value)
            self.app._color_picker_apply_hsl()
            self.assertEqual(tuple(variable.get() for variable in variables["rgb"]), (0, 255, 0))
            variables["hex"].set("#33669980")
            self.app._color_picker_apply_hex()
            self.app._color_picker_accept()

        tk.Toplevel.wait_window = choose
        try:
            color = self.app.color_picker_dialog((10, 20, 30, 255), "Проверка цвета")
        finally:
            tk.Toplevel.wait_window = original_wait
        self.assertEqual(color, (51, 102, 153, 128))

    def test_color_control_swaps_and_resets_without_text_buttons(self) -> None:
        self.app.show_editor()
        self.app.update()
        self.app.foreground = (12, 34, 56, 255)
        self.app.background = (210, 190, 170, 128)
        self.app.refresh_color_control()
        self.assertTrue(self.app.color_control_canvas.find_withtag("swap"))
        self.assertTrue(self.app.color_control_canvas.find_withtag("reset"))

        self.app.color_control_click(SimpleNamespace(x=49, y=7))
        self.assertEqual(self.app.foreground, (210, 190, 170, 128))
        self.assertEqual(self.app.background, (12, 34, 56, 255))
        self.app.color_control_click(SimpleNamespace(x=8, y=34))
        self.assertEqual(self.app.foreground, (0, 0, 0, 255))
        self.assertEqual(self.app.background, (255, 255, 255, 255))

    def test_brush_cursor_matches_real_dab_at_current_zoom(self) -> None:
        self.app.doc = self.app.doc.new(240, 180, (0, 0, 0, 0))
        self.app.brush_size.set(18)
        self.app.zoom.set(1.75)
        self.app.tool.set("brush")
        self.app.canvas_to_doc = lambda _event: (120, 90)
        self.app.update_brush_preview(SimpleNamespace(x=0, y=0, state=0))
        ring = self.app._brush_preview_ids[1]
        preview_ids = tuple(self.app._brush_preview_ids)
        x1, _y1, x2, _y2 = self.app.canvas.coords(ring)
        for _index in range(100):
            self.app.update_brush_preview(SimpleNamespace(x=0, y=0, state=0))
        self.assertEqual(tuple(self.app._brush_preview_ids), preview_ids)

        before = self.app.doc.layer.pixels.copy()
        changed = draw_brush(self.app.doc.layer, 120, 90, 18, (20, 100, 220, 255), 1.0)
        self.assertIsNotNone(changed)
        changed_width = changed[2] - changed[0]
        self.assertAlmostEqual(x2 - x1, changed_width * self.app.zoom.get(), delta=4.0)
        self.assertFalse(np.array_equal(before, self.app.doc.layer.pixels))

    def test_clone_and_healing_show_source_selection_mode(self) -> None:
        self.app.canvas_to_doc = lambda _event: (40, 30)
        for tool in ("clone", "healing"):
            self.app.tool.set(tool)
            self.app.pointer_motion(SimpleNamespace(x=0, y=0, state=0x20000))
            self.assertEqual(str(self.app.canvas.cget("cursor")), "target")
            label = self.app._brush_preview_ids[-1]
            self.assertEqual(str(self.app.canvas.itemcget(label, "text")), "Источник")
            self.assertEqual(str(self.app.canvas.itemcget(label, "state")), "normal")
            self.app.pointer_motion(SimpleNamespace(x=0, y=0, state=0))
            self.assertEqual(str(self.app.canvas.cget("cursor")), "crosshair")

    def test_delete_is_contextual_for_raster_selection_and_vector_object(self) -> None:
        self.app.show_editor()
        self.app.update()
        self.app.shortcut_context = lambda: "canvas"
        raster = self.app.doc.layer
        raster.pixels[:] = (40, 90, 160, 255)
        before = raster.pixels.copy()
        self.app.doc.selection_mask = None
        self.assertEqual(self.app.shortcut_delete(), "break")
        self.assertTrue(np.array_equal(raster.pixels, before))
        self.assertEqual(len(self.app.doc.layers), 1)

        shape = self.app.doc.add_shape_layer("ellipse", (10, 10, 70, 60), (220, 60, 40, 255))
        self.app.selected_layer_ids = {shape.id}
        self.app.refresh()
        self.assertEqual(self.app.shortcut_delete(), "break")
        self.assertIsNone(self.app.doc.get_layer(shape.id))
        self.app.undo()
        self.assertIsNotNone(self.app.doc.get_layer(shape.id))


if __name__ == "__main__":
    unittest.main()
