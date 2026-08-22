from __future__ import annotations

import tkinter as tk
from types import SimpleNamespace
import unittest

from uzyro.ui.icons import TOOL_GROUPS, tool_icon_bitmap
from uzyro.ui.tool_palette import ToolPaletteDialog


DEFINITIONS = [
    ("Первый", "first", "Первый инструмент"),
    ("Второй", "second", "Второй инструмент"),
    ("Третий", "third", "Третий инструмент"),
]


class ToolPaletteDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = tk.Tk()
        cls.root.geometry("1x1+0+0")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.root.destroy()

    def setUp(self) -> None:
        self.dialog = ToolPaletteDialog(
            self.root,
            definitions=DEFINITIONS,
            order=["first", "second", "third"],
            visible=["first", "second", "third"],
        )
        self.dialog.update()
        self.dialog.deiconify()
        self.dialog.update_idletasks()

    def tearDown(self) -> None:
        if self.dialog.winfo_exists():
            self.dialog.destroy()
        self.root.update_idletasks()

    def row_center(self, index: int) -> int:
        bounds = self.dialog.listbox.bbox(index)
        self.assertIsNotNone(bounds)
        return int(bounds[1] + bounds[3] / 2)

    def test_tool_name_toggles_visibility_with_one_click(self) -> None:
        y = self.row_center(1)
        self.dialog.listbox.event_generate("<ButtonPress-1>", x=120, y=y)
        self.dialog.listbox.event_generate("<ButtonRelease-1>", x=120, y=y)
        self.dialog.update()
        self.assertNotIn("second", self.dialog.visible)
        self.assertIn("Второй", str(self.dialog.listbox.get(1)))
        self.assertIn("скрыт", str(self.dialog.listbox.get(1)))
        self.assertEqual(str(self.dialog.visibility_status.cget("text")), "Скрыт")

    def test_selected_tool_has_visible_drag_handle(self) -> None:
        self.assertTrue(self.dialog._drag_handle.place_info())
        self.assertEqual(str(self.dialog._drag_handle.cget("text")), "≡")
        self.assertEqual(str(self.dialog._drag_handle.cget("cursor")), "fleur")

    def test_drag_handle_reorders_without_toggling_visibility(self) -> None:
        target_y = self.row_center(2)
        self.dialog.begin_handle_drag(SimpleNamespace())
        root_y = self.dialog.listbox.winfo_rooty() + target_y
        root_x = self.dialog.listbox.winfo_rootx() + 20
        event = SimpleNamespace(x_root=root_x, y_root=root_y)
        self.dialog.drag_from_handle(event)
        self.dialog.update()
        self.assertIsNotNone(self.dialog._drag_ghost)
        self.assertIn("≡", str(self.dialog._drag_ghost.winfo_children()[0].cget("text")))
        self.dialog.end_handle_drag(event)
        self.dialog.update()
        self.assertEqual(self.dialog.order, ["second", "third", "first"])
        self.assertIn("first", self.dialog.visible)

    def test_dragging_reorders_tools(self) -> None:
        start_y = self.row_center(0)
        target_y = self.row_center(2)
        self.dialog.listbox.event_generate("<ButtonPress-1>", x=80, y=start_y)
        self.dialog.listbox.event_generate("<B1-Motion>", x=80, y=target_y)
        self.dialog.update()
        self.assertEqual(self.dialog.order, ["first", "second", "third"])
        self.assertIsNotNone(self.dialog._drag_ghost)
        self.assertTrue(self.dialog._drop_indicator.place_info())
        self.dialog.listbox.event_generate("<ButtonRelease-1>", x=80, y=target_y)
        self.dialog.update()
        self.assertEqual(self.dialog.order, ["second", "third", "first"])
        self.assertIsNone(self.dialog._drag_ghost)

    def test_every_tool_has_a_nonempty_distinct_icon(self) -> None:
        bitmaps = {tool: tool_icon_bitmap(tool, 24) for tool in TOOL_GROUPS}
        signatures = {tool: bitmap.tobytes() for tool, bitmap in bitmaps.items()}
        self.assertTrue(all(bitmap.getbbox() is not None for bitmap in bitmaps.values()))
        self.assertEqual(len(set(signatures.values())), len(signatures))


if __name__ == "__main__":
    unittest.main()
