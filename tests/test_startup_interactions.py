from __future__ import annotations

import copy
import tkinter as tk
import time
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
from PIL import Image

from photoredactor.app import PhotoRedactorApp
from photoredactor.history import DocumentStateCommand, LayerFieldsCommand, LayerInsertCommand, LayerMoveCommand, PixelPatchCommand, SelectionMaskCommand, ShapeDataCommand, TextDataCommand
from photoredactor.ui.shortcuts import COMMAND_SHORTCUTS

class StartupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_clipboard_reader = PhotoRedactorApp.read_clipboard_image
        PhotoRedactorApp.read_clipboard_image = staticmethod(lambda: None)
        self.app = PhotoRedactorApp()
        self.app.update()

    def tearDown(self) -> None:
        self.app.destroy()
        PhotoRedactorApp.read_clipboard_image = staticmethod(self.original_clipboard_reader)

    def test_gradient_options_follow_selected_mode(self) -> None:
        def labels(widget) -> set[str]:
            result: set[str] = set()
            for child in widget.winfo_children():
                try:
                    value = str(child.cget("text"))
                except tk.TclError:
                    value = ""
                if value:
                    result.add(value)
                result.update(labels(child))
            return result

        self.app.tool.set("gradient")
        self.app.gradient_mode.set("Заливка")
        self.app.update()
        fill_labels = labels(self.app.tool_options_panel)
        self.assertIn("Тип", fill_labels)
        self.assertNotIn("Фигура", fill_labels)
        self.assertNotIn("Текстура", fill_labels)

        self.app.gradient_mode.set("Объект")
        self.app.gradient_object_fill.set("Текстура")
        self.app.update()
        texture_labels = labels(self.app.tool_options_panel)
        self.assertIn("Фигура", texture_labels)
        self.assertIn("Заливка", texture_labels)
        self.assertIn("Текстура", texture_labels)
        self.assertNotIn("Тип", texture_labels)

    def test_large_gradient_preview_is_limited_to_visible_canvas(self) -> None:
        self.app.create_document_from_settings(
            {"width": 2400, "height": 1800, "dpi": 72, "background": (255, 255, 255, 255), "include_clipboard": False}
        )
        self.app.tool.set("gradient")
        self.app.gradient_mode.set("Объект")
        self.app.zoom.set(1.0)
        self.app.refresh()
        self.app.update()
        self.app.update_gradient_preview((0, 0), (2400, 1800))
        deadline = time.monotonic() + 0.3
        while self.app._gradient_preview_image is None and time.monotonic() < deadline:
            self.app.update()
            time.sleep(0.01)
        preview = self.app._gradient_preview_image
        self.assertIsNotNone(preview)
        self.assertLessEqual(preview.width(), self.app.canvas.winfo_width() + 8)
        self.assertLessEqual(preview.height(), self.app.canvas.winfo_height() + 8)
        self.app.clear_gradient_preview()

    def test_colors_live_only_in_contextual_toolbar(self) -> None:
        self.app.tool.set("gradient")
        self.app.update()
        self.assertFalse(hasattr(self.app, "color_control_canvas"))
        primary = self.app.tool_options_panel.body.winfo_children()[0]
        canvases = [child for child in primary.winfo_children() if isinstance(child, tk.Canvas)]
        self.assertEqual(len(canvases), 1)

    def test_move_drag_uses_regional_transform_and_one_history_item(self) -> None:
        self.app.doc = self.app.doc.new(900, 700, (0, 0, 0, 0))
        self.app.doc.layers.clear()
        layer = self.app.doc.add_shape_layer("ellipse", (60, 50, 180, 140), (40, 120, 220, 255))
        self.app.history.clear()
        self.app.tool.set("move")
        self.app.refresh()
        start_x, start_y = self.app.doc_to_canvas(100, 90)
        captured: list[tuple[tuple[int, int, int, int] | None, str]] = []
        original_refresh = self.app.request_canvas_refresh
        self.app.request_canvas_refresh = lambda rect=None, _layer=None, kind="pixels", **_kwargs: captured.append((rect, kind))
        try:
            self.app.pointer_down(SimpleNamespace(x=start_x, y=start_y, state=0))
            self.app.pointer_drag(SimpleNamespace(x=start_x + 24, y=start_y + 18, state=0))
            self.app.pointer_up(SimpleNamespace(x=start_x + 24, y=start_y + 18, state=0))
        finally:
            self.app.request_canvas_refresh = original_refresh
        self.assertEqual((layer.x, layer.y), (24, 18))
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0][1], "transform")
        dirty = captured[0][0]
        self.assertIsNotNone(dirty)
        self.assertLess((dirty[2] - dirty[0]) * (dirty[3] - dirty[1]), self.app.doc.width * self.app.doc.height)
        self.assertEqual(len(self.app.history.undo_stack), 1)
        self.assertIsInstance(self.app.history.undo_stack[-1], LayerMoveCommand)
        self.assertFalse(any(isinstance(command, DocumentStateCommand) for command in self.app.history.undo_stack))

    def test_right_click_outside_selection_clears_it_for_selection_tools(self) -> None:
        self.app.doc = self.app.doc.new(180, 120, (0, 0, 0, 0))
        mask = np.zeros((120, 180), dtype=np.uint8)
        mask[20:70, 30:90] = 255
        self.app.doc.selection_mask = mask
        self.app.tool.set("ellipse_select")
        self.app.history.clear()
        self.app.refresh()

        inside_x, inside_y = self.app.doc_to_canvas(50, 40)
        self.assertEqual(self.app.selection_right_click(SimpleNamespace(x=inside_x, y=inside_y)), "break")
        self.assertIsNotNone(self.app.doc.selection_mask)
        self.assertEqual(len(self.app.history.undo_stack), 0)

        outside_x, outside_y = self.app.doc_to_canvas(130, 95)
        self.assertEqual(self.app.selection_right_click(SimpleNamespace(x=outside_x, y=outside_y)), "break")
        self.assertIsNone(self.app.doc.selection_mask)
        self.assertEqual(len(self.app.history.undo_stack), 1)
        self.assertIsInstance(self.app.history.undo_stack[-1], SelectionMaskCommand)

    def test_shape_resize_uses_regional_updates_and_compact_history(self) -> None:
        self.app.doc = self.app.doc.new(900, 700, (0, 0, 0, 0))
        self.app.doc.layers.clear()
        layer = self.app.doc.add_shape_layer("rectangle", (40, 30, 180, 120), (220, 40, 40, 255))
        self.app.selected_layer_ids = {layer.id}
        self.app.history.clear()
        self.app.tool.set("move")
        self.app.refresh()
        captured: list[tuple[tuple[int, int, int, int] | None, str]] = []
        original_refresh = self.app.request_canvas_refresh
        self.app.request_canvas_refresh = lambda rect=None, _layer=None, kind="pixels", **_kwargs: captured.append((rect, kind))
        try:
            self.app.begin_object_resize("se")
            self.app.resize_selected_object_live((250, 170), 0)
            self.app.finish_object_resize()
        finally:
            self.app.request_canvas_refresh = original_refresh
        self.assertEqual(tuple(layer.shape_data["box"]), (40, 30, 250, 170))
        self.assertTrue(captured)
        self.assertTrue(all(rect is not None and kind == "pixels" for rect, kind in captured))
        self.assertEqual(len(self.app.history.undo_stack), 1)
        self.assertIsInstance(self.app.history.undo_stack[-1], ShapeDataCommand)
        self.assertFalse(any(isinstance(command, DocumentStateCommand) for command in self.app.history.undo_stack))

    def test_fifty_shape_insertions_use_layer_commands(self) -> None:
        self.app.doc = self.app.doc.new(320, 240, (0, 0, 0, 0))
        self.app.doc.layers.clear()
        self.app.history.clear()
        for index in range(50):
            x = 4 + (index % 10) * 28
            y = 4 + (index // 10) * 40
            geometry = self.app.shape_geometry_for_drag("ellipse_shape", (x, y), (x + 20, y + 24), 0)
            self.app.create_shape_from_drag("ellipse_shape", geometry)
        self.assertEqual(len(self.app.doc.layers), 50)
        self.assertEqual(len(self.app.history.undo_stack), 50)
        self.assertTrue(all(isinstance(command, LayerInsertCommand) for command in self.app.history.undo_stack))
        self.assertFalse(any(isinstance(command, DocumentStateCommand) for command in self.app.history.undo_stack))

    def test_editor_layout_remains_useful_at_1280_by_720(self) -> None:
        self.app.create_document_from_settings(
            {"width": 900, "height": 600, "dpi": 72, "background": (255, 255, 255, 255), "include_clipboard": False}
        )
        self.app.state("normal")
        self.app.geometry("1280x720+0+0")
        self.app.update()
        minimum_canvas_width = 700 if self.app.winfo_width() >= 1200 else 520
        self.assertGreaterEqual(self.app.canvas.winfo_width(), minimum_canvas_width)
        self.assertGreaterEqual(self.app.canvas.winfo_height(), 520)
        self.assertLessEqual(self.app.right_tabs.winfo_width(), 360)

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

    def test_retouch_tools_keep_independent_settings(self) -> None:
        self.app.opacity.set(0.83)
        self.app.tool.set("dodge")
        self.app.exposure.set(0.12)
        self.app.tool.set("brush")
        self.assertAlmostEqual(self.app.opacity.get(), 0.83)
        self.app.tool.set("dodge")
        self.assertAlmostEqual(self.app.exposure.get(), 0.12)
        self.assertNotAlmostEqual(self.app.opacity.get(), self.app.exposure.get())

    def test_layer_eye_click_does_not_change_active_layer(self) -> None:
        self.app.create_document_from_settings(
            {"width": 80, "height": 60, "dpi": 72, "background": (255, 255, 255, 255), "include_clipboard": False}
        )
        self.app.doc.add_layer("Верхний")
        self.app.doc.add_layer("Активный")
        self.app.refresh_layers()
        self.app.update()
        active = self.app.doc.active_layer
        row = 2
        bounds = self.app.layer_list.bbox(row)
        self.assertIsNotNone(bounds)
        self.app.layer_list_click(SimpleNamespace(x=8, y=bounds[1] + bounds[3] // 2))
        self.assertFalse(self.app.doc.layers[0].visible)
        self.assertEqual(self.app.doc.active_layer, active)

    def test_shape_preview_reuses_one_canvas_item(self) -> None:
        self.app.create_document_from_settings(
            {"width": 160, "height": 100, "dpi": 72, "background": (255, 255, 255, 255), "include_clipboard": False}
        )
        self.app.tool.set("line_shape")
        self.app.draw_selection((10, 10), (60, 30))
        first = list(self.app._drag_preview_ids)
        self.app.draw_selection((10, 10), (120, 70))
        self.assertEqual(self.app._drag_preview_ids, first)
        self.assertEqual(len(first), 1)

    def test_ctrl_a_in_layers_selects_every_layer(self) -> None:
        self.app.create_document_from_settings(
            {"width": 80, "height": 60, "dpi": 72, "background": (255, 255, 255, 255), "include_clipboard": False}
        )
        self.app.doc.add_layer("Two")
        self.app.doc.add_layer("Three")
        self.app.refresh_layers()
        self.app.layer_list.focus_set()
        self.app.update()
        self.app.shortcut_select_all()
        self.assertEqual(len(self.app.layer_list.curselection()), 3)
        self.assertEqual(len(self.app.selected_layer_ids), 3)

    def test_delete_on_canvas_uses_exact_selection_and_undo(self) -> None:
        self.app.create_document_from_settings(
            {"width": 20, "height": 16, "dpi": 72, "background": (255, 255, 255, 255), "include_clipboard": False}
        )
        self.app.doc.set_rect_selection((4, 3, 12, 10))
        self.app.delete_selected_pixels()
        self.assertEqual(int(self.app.doc.layer.pixels[5, 6, 3]), 0)
        self.assertEqual(int(self.app.doc.layer.pixels[1, 1, 3]), 255)
        self.app.undo()
        self.assertEqual(int(self.app.doc.layer.pixels[5, 6, 3]), 255)

    def test_crop_waits_for_explicit_apply(self) -> None:
        self.app.create_document_from_settings(
            {"width": 100, "height": 80, "dpi": 72, "background": (255, 255, 255, 255), "include_clipboard": False}
        )
        self.app._crop_box = self.app.crop_box_for_drag((10, 12), (70, 62))
        self.app.draw_crop_overlay(self.app._crop_box)
        self.assertEqual((self.app.doc.width, self.app.doc.height), (100, 80))
        self.app.apply_crop_overlay()
        self.assertEqual((self.app.doc.width, self.app.doc.height), (60, 50))

    def test_text_is_created_from_in_canvas_editor(self) -> None:
        self.app.create_document_from_settings(
            {"width": 240, "height": 140, "dpi": 72, "background": (255, 255, 255, 255), "include_clipboard": False}
        )
        self.app.tool.set("text")
        self.app.begin_text_editor((20, 25), (180, 90))
        self.assertIsNotNone(self.app._text_editor)
        self.app._text_editor.insert("1.0", "Текст на холсте")
        self.app.finish_text_edit()
        self.assertEqual(self.app.doc.layer.kind, "text")
        self.assertEqual(self.app.doc.layer.text_data["box_width"], 160)
        self.assertIn("Текст на холсте", self.app.doc.layer.text_data["text"])


if __name__ == "__main__":
    unittest.main()
