from __future__ import annotations

import tkinter as tk
from tkinter import ttk
import unittest

from photoredactor.app import PhotoRedactorApp


class BrushEnginePanelTests(unittest.TestCase):
    def test_panel_opens_and_advanced_settings_roundtrip_in_preset(self) -> None:
        original = PhotoRedactorApp.read_clipboard_image
        PhotoRedactorApp.read_clipboard_image = staticmethod(lambda: None)
        app = PhotoRedactorApp()
        try:
            app.brush_advanced.update({"angle": 37.0, "roundness": 0.42, "scatter": 1.25})
            app.brush_preset_name.set("Расширенный тест")
            app.save_brush_preset()
            app.brush_advanced.update({"angle": 0.0, "roundness": 1.0, "scatter": 0.0})
            app.apply_brush_preset()
            self.assertEqual(app.brush_advanced["angle"], 37.0)
            self.assertEqual(app.brush_advanced["roundness"], 0.42)
            self.assertEqual(app.brush_advanced["scatter"], 1.25)

            app.open_brush_engine_panel()
            app.update()
            dialogs = [widget for widget in app.winfo_children() if isinstance(widget, tk.Toplevel) and widget.title() == "Движок кисти"]
            self.assertEqual(len(dialogs), 1)
            notebooks = [widget for widget in dialogs[0].winfo_children() if isinstance(widget, ttk.Notebook)]
            self.assertEqual(len(notebooks), 1)
            self.assertEqual(len(notebooks[0].tabs()), 6)
            dialogs[0].destroy()
        finally:
            app.brush_presets.pop("Расширенный тест", None)
            app.destroy()
            PhotoRedactorApp.read_clipboard_image = staticmethod(original)


if __name__ == "__main__":
    unittest.main()
