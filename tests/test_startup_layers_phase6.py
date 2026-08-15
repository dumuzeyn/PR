from __future__ import annotations

from types import SimpleNamespace
import tkinter as tk
import unittest

from uzyro.app import UZYROApp


class Phase6LayerUiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_clipboard_reader = UZYROApp.read_clipboard_image
        UZYROApp.read_clipboard_image = staticmethod(lambda: None)
        self.app = UZYROApp(); self.app.update()

    def tearDown(self) -> None:
        self.app.destroy()
        UZYROApp.read_clipboard_image = staticmethod(self.original_clipboard_reader)

    def test_layer_styles_dialog_has_live_preview_and_returns_enabled_effect(self) -> None:
        self.app.doc = self.app.doc.new(280, 190, (0, 0, 0, 0))
        self.app.doc.layers.clear()
        layer = self.app.doc.add_shape_layer("star", (55, 35, 225, 165), (45, 125, 210, 255), sides=5)
        original_wait = tk.Toplevel.wait_window

        def edit(window, target=None) -> None:
            window.update()
            self.assertIsNotNone(self.app._layer_styles_preview_image)
            for index in range(10):
                self.app._layer_styles_listbox.selection_clear(0, tk.END)
                self.app._layer_styles_listbox.selection_set(index)
                self.app._layer_styles_listbox.event_generate("<<ListboxSelect>>")
                window.update()
                self.app._layer_styles_enabled.set(True)
            self.app._layer_styles_accept()

        tk.Toplevel.wait_window = edit
        try:
            result = self.app.layer_styles_dialog(layer)
        finally:
            tk.Toplevel.wait_window = original_wait
        self.assertEqual(sum(bool(item["enabled"]) for item in result.values()), 10)
        self.assertEqual(result["drop_shadow"]["blend_mode"], "Multiply")

    def test_destructive_filter_dialog_returns_edited_values(self) -> None:
        self.app.doc = self.app.doc.new(120, 90, (30, 80, 140, 255))
        original_wait = tk.Toplevel.wait_window

        def edit(window, target=None) -> None:
            window.update()
            self.app._destructive_filter_values["radius"].set(7.5)
            self.app._destructive_filter_accept()

        tk.Toplevel.wait_window = edit
        try:
            result = self.app.destructive_filter_dialog(
                "Проверка", [("radius", "Радиус", 3, 1, 20, 0.5)], lambda pixels, values: pixels
            )
        finally:
            tk.Toplevel.wait_window = original_wait
        self.assertEqual(result, {"radius": 7.5})

    def test_smart_filter_dialog_edits_contextual_parameters(self) -> None:
        self.app.doc = self.app.doc.new(120, 90, (30, 80, 140, 255))
        original_wait = tk.Toplevel.wait_window

        def edit(window, target=None) -> None:
            window.update()
            extras = self.app._filter_dialog_extra_values
            extras["radius"].set(4.5)
            extras["threshold"].set(12)
            self.app._filter_dialog_accept()

        tk.Toplevel.wait_window = edit
        try:
            result = self.app.layer_filters_dialog(
                [{"type": "unsharp_mask", "amount": 1.2, "radius": 2.0, "threshold": 0.0}],
                self.app.doc.layer.pixels,
            )
        finally:
            tk.Toplevel.wait_window = original_wait
        self.assertEqual(result[0]["radius"], 4.5)
        self.assertEqual(result[0]["threshold"], 12.0)

    def test_alt_visibility_click_solos_and_restores_layers(self) -> None:
        self.app.doc = self.app.doc.new(180, 120, (10, 20, 30, 255))
        self.app.doc.add_layer("Middle"); self.app.doc.add_layer("Top")
        self.app.doc.layers[1].visible = False
        expected = [True, False, True]
        self.app.show_editor(); self.app.refresh(); self.app.update()
        bounds = self.app.layer_list.bbox(0)
        event = SimpleNamespace(x=8, y=bounds[1] + bounds[3] // 2, state=0x20000)
        self.assertEqual(self.app.layer_list_click(event), "break")
        self.assertEqual([layer.visible for layer in self.app.doc.layers], [False, False, True])
        self.app.layer_list_click(event)
        self.assertEqual([layer.visible for layer in self.app.doc.layers], expected)


if __name__ == "__main__":
    unittest.main()
