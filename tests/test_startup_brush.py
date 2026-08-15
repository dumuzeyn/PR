from __future__ import annotations

from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk
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

    def test_fresh_install_default_brush_makes_a_visible_mark(self) -> None:
        self.assertEqual(self.app.brush_blend_mode.get(), "Normal")
        self.assertAlmostEqual(self.app.hardness.get(), 1.0)
        self.assertAlmostEqual(self.app.brush_spacing.get(), 0.0)
        self.assertAlmostEqual(self.app.opacity.get(), 1.0)
        self.assertAlmostEqual(self.app.brush_flow.get(), 1.0)
        self.assertFalse(self.app.pressure_opacity.get())
        self.assertFalse(self.app.pressure_flow.get())
        before = self.app.doc.layer.pixels.copy()

        self.app.tool.set("brush")
        center_x = self.app.canvas.winfo_width() // 2
        center_y = self.app.canvas.winfo_height() // 2
        self.app.pointer_down(SimpleNamespace(x=center_x - 30, y=center_y, state=0))
        self.app.pointer_drag(SimpleNamespace(x=center_x + 30, y=center_y, state=0))
        self.app.pointer_up(SimpleNamespace(x=center_x + 30, y=center_y, state=0))

        self.assertFalse(np.array_equal(before, self.app.doc.layer.pixels))
        self.assertTrue(np.any(self.app.doc.layer.pixels[:, :, :3] != 255))
        self.assertEqual(len(self.app.history.undo_stack), 1)

    def test_ctrl_z_undoes_brush_when_option_field_keeps_focus(self) -> None:
        before = self.app.doc.layer.pixels.copy()
        center_x = self.app.canvas.winfo_width() // 2
        center_y = self.app.canvas.winfo_height() // 2
        self.app.pointer_down(SimpleNamespace(x=center_x - 30, y=center_y, state=0))
        self.app.pointer_drag(SimpleNamespace(x=center_x + 30, y=center_y, state=0))
        self.app.pointer_up(SimpleNamespace(x=center_x + 30, y=center_y, state=0))
        self.app.update()
        self.assertEqual(len(self.app.history.undo_stack), 1)

        option = next(
            widget for widget in self._descendants(self.app.tool_options_panel)
            if isinstance(widget, (ttk.Entry, ttk.Spinbox))
        )
        option.focus_force()
        self.app.update()
        option.event_generate("<Control-KeyPress-z>", state=0x0004, keycode=90)
        self.app.update()

        np.testing.assert_array_equal(self.app.doc.layer.pixels, before)
        self.assertEqual(len(self.app.history.undo_stack), 0)
        self.assertEqual(len(self.app.history.redo_stack), 1)

    def test_ctrl_z_stays_local_inside_canvas_text_editor(self) -> None:
        self.app.begin_text_editor((20, 25), (180, 90))
        editor = self.app._text_editor
        self.assertIsNotNone(editor)
        editor.focus_force()
        editor.insert("1.0", "тест")
        editor.edit_separator()
        editor.insert(tk.END, " ввод")
        self.app.update()

        calls: list[str] = []
        self.app.undo = lambda: calls.append("document undo")
        self.assertIsNone(self.app.shortcut_undo())
        editor.edit_undo()

        self.assertEqual(editor.get("1.0", "end-1c"), "тест")
        self.assertEqual(calls, [])

    @staticmethod
    def _descendants(widget):
        for child in widget.winfo_children():
            yield child
            yield from BrushStartupTests._descendants(child)

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
