from __future__ import annotations

import tkinter as tk
import time
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from photoredactor.app import PhotoRedactorApp
from photoredactor.history import DocumentStateCommand, LayerInsertCommand, LayerMoveCommand, SelectionMaskCommand, ShapeDataCommand
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
