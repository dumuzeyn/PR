from __future__ import annotations

import tkinter as tk
import unittest

import numpy as np

from uzyro.app import UZYROApp
from uzyro.core import Document
from uzyro.document_manager import DocumentManager
from uzyro.history import History, LayerMoveCommand


class DocumentWorkspaceUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_clipboard_reader = UZYROApp.read_clipboard_image
        UZYROApp.read_clipboard_image = staticmethod(lambda: None)
        cls.app = UZYROApp()
        cls.app.update()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.app.destroy()
        UZYROApp.read_clipboard_image = staticmethod(cls.original_clipboard_reader)

    def setUp(self) -> None:
        document = Document.new()
        self.app.doc = document
        self.app.history = History()
        self.app.document_manager = DocumentManager(document, self.app.history)
        self.app.selected_layer_ids = {document.layer.id}
        self.app._editor_active = False
        self.app.show_start_screen()
        self.app.update_idletasks()

    def test_tabs_switch_documents_with_independent_history_and_view_state(self) -> None:
        first = Document.new(240, 160, (255, 255, 255, 255))
        self.app.open_document_session(first, replace_startup=True)
        first_session = self.app.document_manager.active
        self.app.zoom.set(0.75)
        self.app.history.push(LayerMoveCommand("Документ A", first.layer.id, (0, 0), (5, 0)))

        second = Document.new(480, 300, (0, 0, 0, 0))
        self.app.open_document_session(second)
        second_session = self.app.document_manager.active
        self.assertEqual(len(self.app.document_manager.documents), 2)
        self.assertEqual(len(self.app._document_tab_buttons), 2)
        self.assertFalse(self.app.history.undo_stack)

        self.app.switch_document(first_session.id)
        self.assertIs(self.app.doc, first)
        self.assertEqual(self.app.history.undo_stack[-1].label, "Документ A")
        self.assertAlmostEqual(self.app.zoom.get(), 0.75)
        self.app.switch_document(second_session.id)
        self.assertIs(self.app.doc, second)
        self.app.cycle_document(-1)
        self.assertIs(self.app.doc, first)

    def test_selected_shapes_have_common_bounds_and_group_nudge(self) -> None:
        document = Document.new(320, 220, (0, 0, 0, 0))
        document.layers.clear()
        first = document.add_shape_layer("rectangle", (20, 30, 90, 100), (240, 80, 80, 255))
        second = document.add_shape_layer("ellipse", (150, 80, 250, 180), (80, 120, 240, 255))
        self.app.open_document_session(document, replace_startup=True)
        self.app.tool.set("move")
        self.app.selected_layer_ids = {first.id, second.id}
        bounds = self.app.selected_object_bounds()
        self.assertIsNotNone(bounds)
        self.assertLessEqual(bounds[0], 20)
        self.assertGreaterEqual(bounds[2], 250)

        event = type("Event", (), {"state": 0, "keysym": "Right"})()
        self.app.nudge_selected_object(event)
        self.assertEqual((first.x, second.x), (1, 1))
        self.assertEqual(len(self.app.history.undo_stack), 1)
        self.app.undo()
        self.assertEqual((first.x, second.x), (0, 0))

    def test_group_resize_and_rotation_each_create_one_history_step(self) -> None:
        document = Document.new(360, 260, (0, 0, 0, 0))
        document.layers.clear()
        first = document.add_shape_layer("rectangle", (20, 30, 90, 100), (240, 80, 80, 255))
        second = document.add_shape_layer("ellipse", (150, 80, 250, 180), (80, 120, 240, 255))
        self.app.open_document_session(document, replace_startup=True)
        self.app.tool.set("move")
        self.app.selected_layer_ids = {first.id, second.id}

        original = self.app.selected_object_bounds()
        self.app.begin_object_resize("se", original[2:])
        self.app.resize_selected_object_live((original[2] + 55, original[3] + 35), 0)
        self.app.finish_object_resize()
        enlarged = self.app.selected_object_bounds()
        self.assertGreater(enlarged[2] - enlarged[0], original[2] - original[0])
        self.assertEqual(len(self.app.history.undo_stack), 1)

        geometry, _angle = self.app._selected_object_geometry()
        rotate_point = tuple(round(value) for value in geometry["rotate"])
        self.app.begin_object_resize("rotate", rotate_point)
        center = self.app._object_rotation_center
        self.app.resize_selected_object_live((round(center[0] + 80), round(center[1])), 0)
        self.app.finish_object_resize()
        self.assertEqual(len(self.app.history.undo_stack), 2)
        self.assertTrue(any(abs(float((layer.shape_data or {}).get("rotation", 0.0))) > 0.01 for layer in (first, second)))

    def test_move_one_hundred_shapes_is_one_history_step(self) -> None:
        document = Document.new(900, 700, (0, 0, 0, 0))
        document.layers.clear()
        for index in range(100):
            x, y = (index % 10) * 80, (index // 10) * 60
            document.add_shape_layer("rectangle", (x, y, x + 40, y + 30), (40, 120, 220, 255))
        self.app.open_document_session(document, replace_startup=True)
        self.app.tool.set("move")
        self.app.selected_layer_ids = {layer.id for layer in document.layers}
        self.app._move_layer_id = document.layer.id
        self.app._move_group_starts = {layer.id: (layer.x, layer.y) for layer in document.layers}
        self.app._move_group_masks = {layer.id: None for layer in document.layers}
        self.app._move_last_bounds = self.app.selected_object_bounds()
        self.app.begin_layer_move_preview((0, 0))
        self.app.move_selected_layers_live((12, 7))
        self.app.update_idletasks()
        self.app.finish_layer_move_preview()
        self.app.end_move_layer()

        self.assertTrue(all((layer.x, layer.y) == (12, 7) for layer in document.layers))
        self.assertEqual(len(self.app.history.undo_stack), 1)

    def test_pixel_selection_preview_hides_source_and_cancel_restores_it(self) -> None:
        document = Document.new(80, 60, (0, 0, 0, 0))
        document.layer.pixels[10:30, 15:40] = (220, 45, 70, 255)
        document.layer.touch_pixels()
        selection = np.zeros((60, 80), dtype=np.uint8)
        selection[10:30, 15:40] = 255
        document.selection_mask = selection
        document.dirty = False
        original = document.layer.pixels.copy()
        self.app.open_document_session(document, replace_startup=True)

        self.app.begin_move_selection_preview((15, 10, 40, 30))
        self.assertFalse(document.layer.pixels[10:30, 15:40, 3].any())
        self.assertIsNotNone(self.app._move_selection_preview_id)
        self.app.cancel_move_selection_preview()

        self.assertTrue(np.array_equal(document.layer.pixels, original))
        self.assertFalse(document.dirty)
        self.assertIsNone(self.app._move_selection_preview_id)

    def test_pixel_selection_move_commits_content_and_outline_as_one_action(self) -> None:
        document = Document.new(90, 65, (0, 0, 0, 0))
        document.layer.pixels[10:30, 15:40] = (220, 45, 70, 255)
        document.layer.touch_pixels()
        selection = np.zeros((65, 90), dtype=np.uint8)
        selection[10:30, 15:40] = 255
        document.selection_mask = selection
        original = document.layer.pixels.copy()
        self.app.open_document_session(document, replace_startup=True)
        self.app._move_selection_bounds = (15, 10, 40, 30)
        self.app._move_selection_delta = (30, 5)
        self.app.begin_move_selection_preview(self.app._move_selection_bounds)

        self.app.end_move_selection()

        self.assertFalse(document.layer.pixels[10:30, 15:40, 3].any())
        self.assertTrue(document.layer.pixels[15:35, 45:70, 3].all())
        self.assertEqual(document.selection_bounds(), (45, 15, 70, 35))
        self.assertEqual(len(self.app.history.undo_stack), 1)
        self.app.undo()
        self.assertTrue(np.array_equal(document.layer.pixels, original))
        self.assertEqual(document.selection_bounds(), (15, 10, 40, 30))


if __name__ == "__main__":
    unittest.main()
