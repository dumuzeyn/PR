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
        preset_size_labels: list[str] = []
        centered: list[bool] = []

        def inspect_dialog(window, target=None) -> None:
            shown_size.append(str(self.app._new_document_size_label.cget("text")))
            texts = [
                str(self.app._new_document_preset_canvas.itemcget(item, "text"))
                for item in self.app._new_document_preset_canvas.find_all()
                if self.app._new_document_preset_canvas.type(item) == "text"
            ]
            preset_size_labels.extend(text for text in texts if text.startswith("Размер холста:"))
            expected_x = (window.winfo_screenwidth() - window.winfo_width()) // 2
            expected_y = (window.winfo_screenheight() - window.winfo_height()) // 2
            centered.append(abs(window.winfo_x() - expected_x) <= 3 and abs(window.winfo_y() - expected_y) <= 3)
            window.destroy()

        tk.Toplevel.wait_window = inspect_dialog
        try:
            self.assertIsNone(self.app.new_document_dialog(clipboard))
            self.assertIsNotNone(self.app._new_document_preview)
            self.assertEqual(shown_size, ["640 x 360 px"])
            self.assertEqual(len(preset_size_labels), len(self.app.available_document_presets(clipboard)))
            self.assertTrue(all(" x " in label and label.endswith(" px") for label in preset_size_labels))
            self.assertEqual(centered, [True])
        finally:
            tk.Toplevel.wait_window = original_wait

    def test_new_document_preset_double_click_creates_selected_canvas(self) -> None:
        original_wait = tk.Toplevel.wait_window

        def choose_preset(_window, target=None) -> None:
            binding = self.app._new_document_preset_canvas.tag_bind("preset-3", "<Double-Button-1>")
            self.assertTrue(binding)
            self.app._new_document_accept_preset(3)

        tk.Toplevel.wait_window = choose_preset
        try:
            settings = self.app.new_document_dialog()
        finally:
            tk.Toplevel.wait_window = original_wait
        preset = self.app.available_document_presets()[3]
        self.assertIsNotNone(settings)
        self.assertEqual((settings["width"], settings["height"]), (preset["width"], preset["height"]))

    def test_text_path_editor_exposes_visual_controls_and_compact_undo(self) -> None:
        self.app.doc = self.app.doc.new(520, 320, (245, 245, 245, 255))
        self.app.doc.layers.clear()
        layer = self.app.doc.add_text_layer("Текст по контуру", 40, 80, (20, 30, 40, 255), 34)
        self.app.selected_layer_ids = {layer.id}
        self.app.history.clear()
        original_wait = tk.Toplevel.wait_window
        centered: list[bool] = []

        def edit_and_accept(window, target=None) -> None:
            window.update()
            self.assertGreaterEqual(self.app._text_path_canvas.winfo_width(), 500)
            self.assertEqual(len(self.app._text_path_points), 4)
            self.app._text_path_points[1][1] -= 75
            self.app._text_path_points[2][1] += 45
            self.app._text_path_start_var.set(0.12)
            self.app._text_path_end_var.set(0.82)
            self.app._text_path_side_var.set(-1)
            self.app._text_path_reverse_var.set(True)
            expected_x = (window.winfo_screenwidth() - window.winfo_width()) // 2
            expected_y = (window.winfo_screenheight() - window.winfo_height()) // 2
            centered.append(abs(window.winfo_x() - expected_x) <= 3 and abs(window.winfo_y() - expected_y) <= 3)
            self.app._text_path_accept()

        tk.Toplevel.wait_window = edit_and_accept
        try:
            self.app.edit_text_path()
        finally:
            tk.Toplevel.wait_window = original_wait
        self.assertEqual(layer.text_data["path_mode"], "bezier")
        self.assertAlmostEqual(layer.text_data["path_start"], 0.12)
        self.assertAlmostEqual(layer.text_data["path_end"], 0.82)
        self.assertEqual(layer.text_data["path_side"], -1)
        self.assertTrue(layer.text_data["path_reverse"])
        self.assertTrue(centered[0])
        self.assertEqual(len(self.app.history.undo_stack), 1)
        self.assertIsInstance(self.app.history.undo_stack[-1], TextDataCommand)
        self.app.undo()
        self.assertEqual(layer.text_data.get("path_mode"), "none")
        self.app.redo()
        self.assertEqual(layer.text_data.get("path_mode"), "bezier")

    def test_select_and_mask_brush_outputs_new_layer_with_compact_undo(self) -> None:
        self.app.doc = self.app.doc.new(320, 240, (30, 80, 160, 255))
        layer = self.app.doc.layer
        layer.pixels[55:190, 70:220, :3] = (190, 65, 45)
        selection = np.zeros((240, 320), dtype=np.uint8)
        selection[70:180, 85:205] = 255
        self.app.doc.selection_mask = selection
        self.app.history.clear()
        original_wait = tk.Toplevel.wait_window

        def paint_and_accept(window, target=None) -> None:
            window.update()
            self.assertGreaterEqual(self.app._select_mask_canvas.winfo_width(), 500)
            self.app._select_mask_brush_mode.set("Добавить")
            self.app._select_mask_brush_size.set(30)
            self.app._select_mask_apply_stroke((220, 120), (260, 120))
            self.app._select_mask_output.set("Новый слой")
            self.app._select_mask_decontaminate.set(True)
            self.app._select_mask_accept()

        tk.Toplevel.wait_window = paint_and_accept
        try:
            self.app.select_and_mask_workspace()
        finally:
            tk.Toplevel.wait_window = original_wait
        self.assertEqual(len(self.app.doc.layers), 2)
        result_layer = self.app.doc.layer
        self.assertEqual(result_layer.kind, "raster")
        self.assertGreater(int(result_layer.pixels[120, 250, 3]), 0)
        self.assertEqual(len(self.app.history.undo_stack), 1)
        self.assertIsInstance(self.app.history.undo_stack[-1], LayerInsertCommand)
        self.app.undo()
        self.assertEqual(len(self.app.doc.layers), 1)
        self.app.redo()
        self.assertEqual(len(self.app.doc.layers), 2)

    def test_select_and_mask_layer_mask_output_uses_layer_fields_history(self) -> None:
        self.app.doc = self.app.doc.new(180, 120, (255, 255, 255, 255))
        selection = np.zeros((120, 180), dtype=np.uint8)
        selection[20:100, 35:145] = 180
        self.app.doc.selection_mask = np.where(selection > 0, 255, 0).astype(np.uint8)
        self.app.history.clear()
        original_dialog = self.app.select_and_mask_dialog
        self.app.select_and_mask_dialog = lambda: {"mask": selection, "output": "Маска слоя", "decontaminate": False}
        try:
            self.app.select_and_mask_workspace()
        finally:
            self.app.select_and_mask_dialog = original_dialog
        self.assertTrue(np.array_equal(self.app.doc.layer.mask, selection))
        self.assertIsInstance(self.app.history.undo_stack[-1], LayerFieldsCommand)
        self.app.undo()
        self.assertIsNone(self.app.doc.layer.mask)
        self.app.redo()
        self.assertTrue(np.array_equal(self.app.doc.layer.mask, selection))

    def test_select_and_mask_selection_and_new_layer_mask_outputs(self) -> None:
        self.app.doc = self.app.doc.new(160, 110, (220, 225, 230, 255))
        original = np.zeros((110, 160), dtype=np.uint8)
        original[25:85, 45:115] = 255
        refined = cv2.GaussianBlur(original, (9, 9), 3)
        self.app.doc.selection_mask = original.copy()
        original_dialog = self.app.select_and_mask_dialog
        self.app.select_and_mask_dialog = lambda: {"mask": refined, "output": "Выделение", "decontaminate": False}
        try:
            self.app.select_and_mask_workspace()
        finally:
            self.app.select_and_mask_dialog = original_dialog
        self.assertTrue(np.array_equal(self.app.doc.selection_mask, refined))
        self.assertIsInstance(self.app.history.undo_stack[-1], SelectionMaskCommand)
        self.app.undo()
        self.assertTrue(np.array_equal(self.app.doc.selection_mask, original))

        self.app.history.clear()
        self.app.select_and_mask_dialog = lambda: {"mask": refined, "output": "Новый слой с маской", "decontaminate": False}
        try:
            self.app.select_and_mask_workspace()
        finally:
            self.app.select_and_mask_dialog = original_dialog
        self.assertEqual(len(self.app.doc.layers), 2)
        self.assertTrue(np.array_equal(self.app.doc.layer.mask, refined))
        self.assertIsInstance(self.app.history.undo_stack[-1], LayerInsertCommand)

    def test_automatic_selection_preview_and_refine_handoff(self) -> None:
        self.app.doc = self.app.doc.new(320, 240, (55, 105, 165, 255))
        layer = self.app.doc.layer
        cv2.ellipse(layer.pixels, (160, 135), (58, 70), 0, 0, 360, (190, 60, 45, 255), -1)
        cv2.circle(layer.pixels, (235, 165), 9, (190, 60, 45, 255), -1)
        self.app.history.clear()
        original_wait = tk.Toplevel.wait_window
        original_refine = self.app.select_and_mask_workspace
        refine_calls: list[bool] = []
        centered: list[bool] = []
        self.app.select_and_mask_workspace = lambda: refine_calls.append(True)

        def configure_and_accept(window, target=None) -> None:
            window.update()
            self.assertTrue(self.app._automatic_selection_preview.cget("image"))
            self.app._automatic_selection_target.set("Объект")
            self.app._automatic_selection_sensitivity.set(0.65)
            self.app._automatic_selection_output.set("Уточнить в «Выделить и маска»")
            expected_x = (window.winfo_screenwidth() - window.winfo_width()) // 2
            expected_y = (window.winfo_screenheight() - window.winfo_height()) // 2
            centered.append(abs(window.winfo_x() - expected_x) <= 3 and abs(window.winfo_y() - expected_y) <= 3)
            self.app._automatic_selection_accept()

        tk.Toplevel.wait_window = configure_and_accept
        try:
            self.app.automatic_selection_workspace()
        finally:
            tk.Toplevel.wait_window = original_wait
            self.app.select_and_mask_workspace = original_refine
        self.assertIsNotNone(self.app.doc.selection_mask)
        self.assertEqual(int(self.app.doc.selection_mask[135, 160]), 255)
        self.assertEqual(int(self.app.doc.selection_mask[10, 10]), 0)
        self.assertTrue(np.any((self.app.doc.selection_mask > 0) & (self.app.doc.selection_mask < 255)))
        self.assertIsInstance(self.app.history.undo_stack[-1], SelectionMaskCommand)
        self.assertEqual(refine_calls, [True])
        self.assertTrue(centered[0])

    def test_content_aware_workspace_edits_source_and_returns_real_variant(self) -> None:
        self.app.doc = self.app.doc.new(260, 180, (48, 102, 190, 255))
        layer = self.app.doc.layer
        layer.pixels[:, :90, :3] = (205, 75, 42)
        cv2.rectangle(layer.pixels, (105, 58), (155, 122), (12, 12, 12, 255), -1)
        selection = np.zeros((180, 260), dtype=np.uint8)
        selection[58:123, 105:156] = 255
        self.app.doc.selection_mask = selection
        original_wait = tk.Toplevel.wait_window
        centered: list[bool] = []

        def paint_and_accept(window, target=None) -> None:
            window.update()
            self.assertGreaterEqual(self.app._content_aware_source_canvas.winfo_width(), 300)
            before = self.app._content_aware_source_mask()
            self.app._content_aware_brush_mode.set("Исключить")
            self.app._content_aware_brush_size.set(36)
            self.app._content_aware_paint((35, 90), (70, 90))
            after = self.app._content_aware_source_mask()
            self.assertLess(np.count_nonzero(after), np.count_nonzero(before))
            self.app._content_aware_variant.set(2)
            expected_x = (window.winfo_screenwidth() - window.winfo_width()) // 2
            expected_y = (window.winfo_screenheight() - window.winfo_height()) // 2
            centered.append(abs(window.winfo_x() - expected_x) <= 3 and abs(window.winfo_y() - expected_y) <= 3)
            self.app._content_aware_accept()

        tk.Toplevel.wait_window = paint_and_accept
        try:
            settings = self.app.content_aware_fill_dialog(selection)
        finally:
            tk.Toplevel.wait_window = original_wait
        self.assertIsNotNone(settings)
        self.assertEqual(settings["variant"], 2)
        self.assertEqual(settings["source_mask"].shape, selection.shape)
        self.assertTrue(settings["rotation_adaptation"])
        self.assertTrue(settings["scale_adaptation"])
        self.assertTrue(centered[0])

    def test_content_aware_apply_uses_compact_pixel_history(self) -> None:
        self.app.doc = self.app.doc.new(180, 120, (70, 135, 95, 255))
        layer = self.app.doc.layer
        cv2.rectangle(layer.pixels, (70, 38), (110, 82), (230, 35, 35, 255), -1)
        selection = np.zeros((120, 180), dtype=np.uint8)
        selection[38:83, 70:111] = 255
        self.app.doc.selection_mask = selection
        source = np.zeros_like(selection)
        source[:, :60] = 255
        original = layer.pixels.copy()
        original_dialog = self.app.content_aware_fill_dialog
        original_background = self.app.run_background
        self.app.content_aware_fill_dialog = lambda _mask: {
            "source_mask": source,
            "radius": 5,
            "color_adaptation": 0.25,
            "rotation_adaptation": True,
            "scale_adaptation": True,
            "variant": 1,
        }
        self.app.run_background = lambda _label, worker, done, _valid: done(worker())
        try:
            self.app.filter_content_aware_fill()
        finally:
            self.app.content_aware_fill_dialog = original_dialog
            self.app.run_background = original_background
        self.assertFalse(np.array_equal(layer.pixels[45:75, 78:102], original[45:75, 78:102]))
        self.assertTrue(np.array_equal(layer.pixels[:25], original[:25]))
        self.assertEqual(len(self.app.history.undo_stack), 1)
        self.assertIsInstance(self.app.history.undo_stack[-1], PixelPatchCommand)
        self.app.undo()
        self.assertTrue(np.array_equal(layer.pixels, original))
        self.app.redo()
        self.assertFalse(np.array_equal(layer.pixels, original))

    def test_transform_workspace_switches_controls_and_edits_mesh_node(self) -> None:
        self.app.doc = self.app.doc.new(360, 240, (0, 0, 0, 0))
        self.app.doc.layers.clear()
        self.app.doc.add_shape_layer("ellipse", (70, 55, 280, 190), (215, 65, 45, 255))
        original_wait = tk.Toplevel.wait_window
        centered: list[bool] = []

        def edit_and_accept(window, target=None) -> None:
            window.update()
            self.assertGreaterEqual(self.app._transform_workspace_canvas.winfo_width(), 600)
            self.app._transform_workspace_mode.set("Сетка")
            window.update()
            self.assertEqual(len(self.app._transform_workspace_points()), 16)
            self.app._transform_workspace_topology.set("3 x 3")
            window.update()
            self.assertEqual(self.app._transform_workspace_grid_shape(), (3, 3))
            self.app._transform_workspace_selected_node.set(4)
            self.app._transform_workspace_split_grid("cross")
            window.update()
            self.assertEqual(self.app._transform_workspace_grid_shape(), (4, 4))
            self.assertEqual(self.app._transform_workspace_topology.get(), "Пользовательская")
            self.app._transform_workspace_select_grid_line("row", 2)
            self.app._transform_workspace_delete_grid_line()
            self.assertEqual(self.app._transform_workspace_grid_shape(), (3, 4))
            self.app._transform_workspace_selected_node.set(5)
            old_point = self.app._transform_workspace_points()[5]
            self.app._transform_workspace_node_x.set(old_point[0] + 24)
            self.app._transform_workspace_node_y.set(old_point[1] - 18)
            self.assertNotEqual(self.app._transform_workspace_points()[5], old_point)
            expected_x = (window.winfo_screenwidth() - window.winfo_width()) // 2
            expected_y = (window.winfo_screenheight() - window.winfo_height()) // 2
            centered.append(abs(window.winfo_x() - expected_x) <= 3 and abs(window.winfo_y() - expected_y) <= 3)
            self.app._transform_workspace_accept()

        tk.Toplevel.wait_window = edit_and_accept
        try:
            data = self.app.transform_workspace_dialog(self.app.doc.layer, "Свободная")
        finally:
            tk.Toplevel.wait_window = original_wait
        self.assertEqual(data["mode"], "Сетка")
        self.assertEqual(len(data["points"]), 12)
        self.assertEqual(data["row_positions"], [0.0, 0.5, 1.0])
        self.assertEqual(data["column_positions"], [0.0, 0.5, 0.75, 1.0])
        self.assertTrue(centered[0])

    def test_transform_workspace_applies_once_and_preserves_shape_editability(self) -> None:
        self.app.doc = self.app.doc.new(320, 220, (0, 0, 0, 0))
        self.app.doc.layers.clear()
        layer = self.app.doc.add_shape_layer("rectangle", (65, 50, 245, 170), (220, 55, 40, 255))
        self.app.history.clear()
        before = copy.deepcopy(layer.shape_data)
        data = {"mode": "Перспектива", "points": [[55, 62], [255, 42], [232, 182], [74, 166]], "rows": 4, "columns": 4}
        self.app.run_document_command("Трансформация слоя", lambda: self.app.apply_transform_workspace_data(data))
        self.assertEqual(layer.kind, "shape")
        self.assertEqual(layer.shape_data, before)
        self.assertEqual(layer.transform_data["mode"], "perspective")
        self.assertEqual(len(self.app.history.undo_stack), 1)
        self.assertIsInstance(self.app.history.undo_stack[-1], LayerFieldsCommand)
        transformed = layer.pixels.copy()
        self.app.undo()
        restored = self.app.doc.layer
        self.assertIsNone(restored.transform_data)
        self.assertEqual(restored.shape_data, before)
        self.app.redo()
        self.assertTrue(np.array_equal(self.app.doc.layer.pixels, transformed))

    def test_convert_and_edit_nested_smart_object_in_full_editor(self) -> None:
        self.app.doc = self.app.doc.new(180, 120, (0, 0, 0, 0))
        self.app.doc.layer.pixels[25:85, 35:115] = (215, 55, 42, 255)
        self.app.doc.add_shape_layer("ellipse", (75, 30, 155, 105), (45, 150, 225, 230))
        self.app.selected_layer_ids = {layer.id for layer in self.app.doc.layers}
        self.app.history.clear()
        self.app.convert_to_smart_object()
        self.assertEqual(len(self.app.doc.layers), 1)
        self.assertEqual(self.app.doc.layer.kind, "embedded")
        self.assertEqual(len(self.app.history.undo_stack), 1)
        before = self.app.doc.layer.pixels.copy()
        original_launcher = self.app.launch_smart_document_editor
        original_background = self.app.run_background

        class FakeProcess:
            def __init__(self, path):
                self.path = path

            def wait(self):
                edited = self.app.doc.open_project(self.path)
                self.assertEqual(len(edited.layers), 2)
                edited.layers[0].pixels[25:85, 35:115] = (35, 215, 90, 255)
                edited.layers[0].touch_pixels()
                edited.save_project(self.path)
                return 0

        self.app.launch_smart_document_editor = lambda path: FakeProcess(path)
        FakeProcess.app = self.app
        FakeProcess.assertEqual = self.assertEqual
        self.app.run_background = lambda _label, worker, done, *_args: done(worker())
        try:
            self.app.edit_smart_object_contents()
        finally:
            self.app.launch_smart_document_editor = original_launcher
            self.app.run_background = original_background
        self.assertFalse(np.array_equal(self.app.doc.layer.pixels, before))
        self.assertIsNotNone(self.app.doc.active_smart_document())

    def test_link_conflict_dialog_keeps_cached_version_as_embedded(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "linked.png"
            Image.new("RGBA", (36, 28), (210, 55, 40, 255)).save(source)
            self.app.doc = self.app.doc.new(80, 60, (0, 0, 0, 0))
            layer = self.app.doc.place_image(source, linked=True)
            cached = layer.smart_source.copy()
            Image.new("RGBA", (36, 28), (40, 90, 220, 255)).save(source)
            original_wait = tk.Toplevel.wait_window

            def choose_embed(window, target=None) -> None:
                window.update()
                self.app._linked_conflict_choice("embed")

            tk.Toplevel.wait_window = choose_embed
            try:
                self.app.resolve_linked_conflict_dialog()
            finally:
                tk.Toplevel.wait_window = original_wait
            self.assertEqual(layer.kind, "embedded")
            self.assertTrue(np.array_equal(layer.smart_source, cached))


if __name__ == "__main__":
    unittest.main()
