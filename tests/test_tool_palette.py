from __future__ import annotations

import tkinter as tk
import unittest

from photoredactor.ui.tool_palette import ToolPaletteDialog


DEFINITIONS = [
    ("Первый", "first", "Первый инструмент"),
    ("Второй", "second", "Второй инструмент"),
    ("Третий", "third", "Третий инструмент"),
]


class ToolPaletteDialogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tk.Tk()
        self.dialog = ToolPaletteDialog(
            self.root,
            definitions=DEFINITIONS,
            order=["first", "second", "third"],
            visible=["first", "second", "third"],
        )
        self.dialog.update()

    def tearDown(self) -> None:
        if self.dialog.winfo_exists():
            self.dialog.destroy()
        self.root.destroy()

    def row_center(self, index: int) -> int:
        bounds = self.dialog.listbox.bbox(index)
        self.assertIsNotNone(bounds)
        return int(bounds[1] + bounds[3] / 2)

    def test_visibility_icon_toggles_tool_with_one_click(self) -> None:
        y = self.row_center(1)
        self.dialog.listbox.event_generate("<ButtonPress-1>", x=10, y=y)
        self.dialog.listbox.event_generate("<ButtonRelease-1>", x=10, y=y)
        self.dialog.update()
        self.assertNotIn("second", self.dialog.visible)
        self.assertTrue(str(self.dialog.listbox.get(1)).startswith("☐"))

    def test_dragging_reorders_tools(self) -> None:
        start_y = self.row_center(0)
        target_y = self.row_center(2)
        self.dialog.listbox.event_generate("<ButtonPress-1>", x=80, y=start_y)
        self.dialog.listbox.event_generate("<B1-Motion>", x=80, y=target_y)
        self.dialog.listbox.event_generate("<ButtonRelease-1>", x=80, y=target_y)
        self.dialog.update()
        self.assertEqual(self.dialog.order, ["second", "third", "first"])


if __name__ == "__main__":
    unittest.main()
