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

    def test_filter_and_adjustment_dialogs_edit_channels_and_filter_mask(self) -> None:
        original_wait = tk.Toplevel.wait_window

        def accept_filter(window, target=None) -> None:
            window.update()
            self.app._filter_dialog_channel.set("Красный")
            self.app._filter_dialog_mask_inverted.set(True)
            self.app._filter_dialog_mask_density.set(65)
            self.app._filter_dialog_mask_feather.set(4)
            self.app._filter_dialog_accept()

        tk.Toplevel.wait_window = accept_filter
        try:
            filters = self.app.layer_filters_dialog(
                [{"type": "blur", "radius": 3}],
                self.app.doc.layer.pixels,
                np.full(self.app.doc.layer.pixels.shape[:2], 255, dtype=np.uint8),
            )
        finally:
            tk.Toplevel.wait_window = original_wait
        self.assertIsNotNone(filters)
        self.assertEqual(filters[0]["channel"], "Red")
        self.assertTrue(filters[0]["mask_inverted"])
        self.assertAlmostEqual(filters[0]["mask_density"], 0.65)
        self.assertEqual(filters[0]["mask_feather"], 4.0)

        def accept_adjustment(window, target=None) -> None:
            window.update()
            self.app._adjustment_dialog_channel.set("Синий")
            self.app._adjustment_dialog_accept()

        tk.Toplevel.wait_window = accept_adjustment
        try:
            adjustment = self.app.adjustment_layer_dialog({"type": "invert", "channel": "RGB"})
        finally:
            tk.Toplevel.wait_window = original_wait
        self.assertIsNotNone(adjustment)
        self.assertEqual(adjustment["adjustment"]["channel"], "Blue")

        def accept_mask(window, target=None) -> None:
            window.update()
            self.app._filter_mask_editor_invert()
            self.app._filter_mask_editor_accept()

        tk.Toplevel.wait_window = accept_mask
        try:
            edited_mask = self.app.filter_mask_editor(np.full((40, 60), 255, dtype=np.uint8))
        finally:
            tk.Toplevel.wait_window = original_wait
        self.assertIsNotNone(edited_mask)
        self.assertFalse(np.any(edited_mask))

    def test_editor_shows_tool_names_and_explicit_options_title(self) -> None:
        move_button = self.app.tool_palette.buttons["move"]
        self.assertEqual(str(move_button.cget("text")), "Перемещение")
        self.assertEqual(str(move_button.cget("compound")), "left")
        self.app.tool.set("blur_tool")
        self.app.update()
        self.assertEqual(str(self.app.tool_options_panel.title.cget("text")), "Параметры: Размытие")

    def test_ctrl_z_uses_physical_key_with_russian_layout(self) -> None:
        calls: list[str] = []
        self.app._editor_active = True
        self.app.undo = lambda: calls.append("undo")
        event = SimpleNamespace(keycode=90, keysym="Cyrillic_ya", state=0x0004)
        self.assertEqual(self.app.shortcut_control_key(event), "break")
        self.assertEqual(calls, ["undo"])

    def test_control_shortcuts_support_shift_variants_and_russian_layout(self) -> None:
        calls: list[str] = []
        self.app._editor_active = True
        self.app.redo = lambda: calls.append("redo")
        self.app.save_as_project = lambda: calls.append("save_as")
        self.app.invert_selection = lambda: calls.append("invert")
        self.assertEqual(
            self.app.shortcut_control_key(SimpleNamespace(keycode=90, keysym="Cyrillic_ya", state=0x0005)),
            "break",
        )
        self.assertEqual(
            self.app.shortcut_control_key(SimpleNamespace(keycode=83, keysym="Cyrillic_yeru", state=0x0005)),
            "break",
        )
        self.assertEqual(
            self.app.shortcut_control_key(SimpleNamespace(keycode=73, keysym="Cyrillic_sha", state=0x0005)),
            "break",
        )
        self.assertEqual(calls, ["redo", "save_as", "invert"])

    def test_every_registered_control_command_reaches_its_callback(self) -> None:
        methods = {
            "undo": "shortcut_undo", "redo": "shortcut_redo", "save": "shortcut_save",
            "save_as": "shortcut_save_as", "open": "shortcut_open", "new_document": "shortcut_new",
            "select_all": "shortcut_select_all", "deselect": "shortcut_deselect",
            "invert_selection": "shortcut_invert_selection", "copy": "shortcut_copy",
            "cut": "shortcut_cut", "paste": "shortcut_paste", "new_layer": "shortcut_new_layer",
            "duplicate_layer": "shortcut_duplicate_layer", "merge_down": "shortcut_merge_down",
            "flatten": "shortcut_flatten", "free_transform": "shortcut_free_transform",
            "fit_to_screen": "shortcut_fit_to_screen", "actual_size": "shortcut_actual_size",
        }
        calls: list[str] = []
        for command, method_name in methods.items():
            setattr(self.app, method_name, lambda _event=None, value=command: calls.append(value) or "break")
        for shortcut in COMMAND_SHORTCUTS:
            state = 0x0004 | (0x0001 if shortcut.shift else 0)
            result = self.app.shortcut_control_key(
                SimpleNamespace(keycode=ord(shortcut.key.upper()), keysym=shortcut.key, state=state)
            )
            self.assertEqual(result, "break")
            self.assertEqual(calls[-1], shortcut.command)

    def test_global_key_dispatchers_are_registered(self) -> None:
        self.assertTrue(self.app.bind_all("<Control-KeyPress>"))
        self.assertTrue(self.app.bind_all("<KeyPress>"))

    def test_tool_shortcuts_cycle_groups_and_brackets_change_size(self) -> None:
        self.app._editor_active = True
        self.app.tool.set("select")
        key_m = SimpleNamespace(keycode=77, keysym="Cyrillic_softsign", state=0)
        self.assertEqual(self.app.shortcut_plain_key(key_m), "break")
        self.assertEqual(self.app.tool.get(), "ellipse_select")
        self.assertEqual(self.app.shortcut_plain_key(key_m), "break")
        self.assertEqual(self.app.tool.get(), "select")

        self.app.tool.set("brush")
        self.app.brush_size.set(20)
        self.assertEqual(
            self.app.shortcut_plain_key(SimpleNamespace(keycode=221, keysym="Cyrillic_hardsign", state=0)),
            "break",
        )
        self.assertEqual(self.app.brush_size.get(), 22)
        self.assertEqual(
            self.app.shortcut_plain_key(SimpleNamespace(keycode=219, keysym="Cyrillic_ha", state=0)),
            "break",
        )
        self.assertEqual(self.app.brush_size.get(), 20)

    def test_plain_shortcuts_do_not_modify_text_fields(self) -> None:
        entry = tk.Entry(self.app.startup_frame)
        entry.pack()
        entry.focus_force()
        self.app.update()
        self.app._editor_active = True
        self.app.tool.set("move")
        self.app.brush_size.set(20)
        self.assertIsNone(self.app.shortcut_plain_key(SimpleNamespace(keycode=66, keysym="b", state=0)))
        self.assertIsNone(self.app.shortcut_plain_key(SimpleNamespace(keycode=221, keysym="bracketright", state=0)))
        self.assertEqual(self.app.tool.get(), "move")
        self.assertEqual(self.app.brush_size.get(), 20)

    def test_menu_accelerators_come_from_working_shortcuts(self) -> None:
        def accelerator_for(menu: tk.Menu, label: str) -> str:
            end = menu.index(tk.END)
            self.assertIsNotNone(end)
            for index in range(int(end) + 1):
                if menu.type(index) != "separator" and menu.entrycget(index, "label") == label:
                    return str(menu.entrycget(index, "accelerator"))
            self.fail(f"Menu entry not found: {label}")

        self.assertEqual(accelerator_for(self.app.file_menu, "Сохранить проект как"), "Ctrl+Shift+S")
        self.assertEqual(accelerator_for(self.app.edit_menu, "Копировать"), "Ctrl+C")
        self.assertEqual(accelerator_for(self.app.select_menu, "Инвертировать выделение"), "Ctrl+Shift+I")
        self.assertEqual(accelerator_for(self.app.layer_menu, "Новый слой"), "Ctrl+Shift+N")
        self.assertEqual(accelerator_for(self.app.layer_menu, "Свободная трансформация"), "Ctrl+T")
        self.assertEqual(accelerator_for(self.app.view_menu, "По размеру окна"), "Ctrl+0")
        self.assertEqual(accelerator_for(self.app.tools_menu, "Размытие"), "R")

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

    def test_canvas_object_selection_syncs_active_layer_and_layers_panel(self) -> None:
        self.app.doc = self.app.doc.new(320, 220, (0, 0, 0, 0))
        self.app.doc.layers.clear()
        first = self.app.doc.add_shape_layer("rectangle", (20, 20, 130, 100), (220, 40, 40, 255))
        second = self.app.doc.add_shape_layer("ellipse", (80, 50, 220, 160), (40, 120, 220, 255))
        self.app.selected_layer_ids = {second.id}
        self.app.tool.set("move")
        self.app.refresh()
        selected = self.app.select_object_at((40, 40))
        self.assertIs(selected, first)
        self.assertEqual(self.app.doc.layer.id, first.id)
        selected_rows = self.app.layer_list.curselection()
        self.assertEqual(len(selected_rows), 1)
        self.assertEqual(len(self.app.doc.layers) - 1 - selected_rows[0], self.app.doc.active_layer)

    def test_selected_shape_has_eight_stable_handles(self) -> None:
        self.app.doc = self.app.doc.new(320, 220, (0, 0, 0, 0))
        self.app.doc.layers.clear()
        layer = self.app.doc.add_shape_layer("ellipse", (40, 30, 180, 120), (40, 120, 220, 255))
        self.app.selected_layer_ids = {layer.id}
        self.app.tool.set("move")
        self.app.refresh()
        self.app.update_object_bounds()
        self.assertEqual(len(self.app._object_bounds_ids), 9)
        item_ids = tuple(self.app._object_bounds_ids)
        self.app.update_object_bounds()
        self.assertEqual(tuple(self.app._object_bounds_ids), item_ids)

    def test_boolean_shape_creation_is_undoable_and_keeps_sources(self) -> None:
        self.app.doc = self.app.doc.new(180, 140, (0, 0, 0, 0))
        self.app.doc.layers.clear()
        lower = self.app.doc.add_shape_layer("rectangle", (15, 20, 100, 100), (220, 40, 40, 255))
        upper = self.app.doc.add_shape_layer("ellipse", (65, 35, 145, 115), (40, 120, 220, 255))
        self.app.history.clear()
        self.app.boolean_shape_editor = lambda initial, _title: {**initial, "boolean_mode": "subtract"}
        self.app.boolean_shape_layers()
        self.assertEqual(len(self.app.doc.layers), 1)
        self.assertEqual(self.app.doc.layer.shape_data["boolean_mode"], "subtract")
        self.assertEqual(len(self.app.doc.layer.shape_data["children"]), 2)
        self.app.undo()
        self.assertEqual(len(self.app.doc.layers), 2)
        self.assertEqual([layer.id for layer in self.app.doc.layers], [lower.id, upper.id])
        self.app.redo()
        self.assertEqual(len(self.app.doc.layers), 1)

    def test_boolean_shape_editor_has_contours_and_live_preview(self) -> None:
        self.app.doc = self.app.doc.new(180, 140, (0, 0, 0, 0))
        self.app.doc.layers.clear()
        self.app.doc.add_shape_layer("rectangle", (15, 20, 100, 100), (220, 40, 40, 255))
        self.app.doc.add_shape_layer("ellipse", (65, 35, 145, 115), (40, 120, 220, 255))
        initial = self.app.doc.boolean_shape_data_with_lower("union")
        self.assertIsNotNone(initial)
        original_wait = tk.Toplevel.wait_window
        captured: list[tuple[int, int]] = []

        def inspect(window, target=None) -> None:
            self.app.update()
            captured.append((self.app._boolean_editor_list.size(), len(self.app._boolean_editor_preview.find_all())))
            window.destroy()

        tk.Toplevel.wait_window = inspect
        try:
            self.assertIsNone(self.app.boolean_shape_editor(initial, "Проверка"))
        finally:
            tk.Toplevel.wait_window = original_wait
        self.assertEqual(captured[0][0], 2)
        self.assertGreaterEqual(captured[0][1], 1)

    def test_boolean_contour_edit_is_undoable(self) -> None:
        self.app.doc = self.app.doc.new(180, 140, (0, 0, 0, 0))
        self.app.doc.layers.clear()
        self.app.doc.add_shape_layer("rectangle", (15, 20, 100, 100), (220, 40, 40, 255))
        self.app.doc.add_shape_layer("ellipse", (65, 35, 145, 115), (40, 120, 220, 255))
        self.assertTrue(self.app.doc.boolean_active_shape_with_lower("union"))
        self.app.history.clear()
        before = copy.deepcopy(self.app.doc.layer.shape_data)
        edited = copy.deepcopy(before)
        edited["children"][1]["_enabled"] = False
        self.app.boolean_shape_editor = lambda _initial, _title: edited
        self.app.edit_boolean_shape()
        self.assertFalse(self.app.doc.layer.shape_data["children"][1]["_enabled"])
        self.assertIsInstance(self.app.history.undo_stack[-1], ShapeDataCommand)
        self.app.undo()
        self.assertEqual(self.app.doc.layer.shape_data, before)


if __name__ == "__main__":
    unittest.main()
