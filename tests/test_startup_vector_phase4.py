from __future__ import annotations

import tkinter as tk
import unittest

from photoredactor.app import PhotoRedactorApp


def widget_texts(widget: tk.Misc) -> set[str]:
    result: set[str] = set()
    for child in widget.winfo_children():
        try:
            value = str(child.cget("text"))
            if value:
                result.add(value)
        except tk.TclError:
            pass
        result.update(widget_texts(child))
    return result


class VectorPhase4UITests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_clipboard_reader = PhotoRedactorApp.read_clipboard_image
        PhotoRedactorApp.read_clipboard_image = staticmethod(lambda: None)
        self.app = PhotoRedactorApp()
        self.app.update()

    def tearDown(self) -> None:
        self.app.destroy()
        PhotoRedactorApp.read_clipboard_image = staticmethod(self.original_clipboard_reader)

    def test_vector_tools_properties_gradient_editor_and_path_drag(self) -> None:
        for tool_id in ("path_select", "direct_select", "add_anchor", "delete_anchor", "convert_anchor"):
            self.assertIn(tool_id, self.app.tool_palette.buttons)
            self.assertTrue(str(self.app.tool_palette.buttons[tool_id].cget("text")))

        self.app.doc = self.app.doc.new(420, 280, (0, 0, 0, 0))
        self.app.doc.layers.clear()
        text = self.app.doc.add_text_layer("Расширенный текст", 25, 30, (20, 30, 40, 255), 32, box_width=300)
        self.app.selected_layer_ids = {text.id}
        self.app.refresh()
        self.app.refresh_properties()
        labels = widget_texts(self.app.object_properties)
        for expected in (
            "Символ", "Начертание", "Кернинг", "Стандартные лигатуры", "Масштаб по горизонтали, %",
            "Зачеркнуть", "Абзац", "Режим текста", "Ширина блока", "Высота блока", "Интервал после",
        ):
            self.assertIn(expected, labels)

        self.app.open_gradient_editor()
        self.app.update()
        editors = [child for child in self.app.winfo_children() if isinstance(child, tk.Toplevel) and child.title() == "Редактор градиента"]
        self.assertEqual(len(editors), 1)
        editor_labels = widget_texts(editors[0])
        self.assertIn("Цветовые точки", editor_labels)
        self.assertIn("Точки прозрачности", editor_labels)
        editors[0].destroy()

        layer = self.app.doc.add_shape_layer("bezier", (30, 40, 350, 220), (0, 0, 0, 0), (240, 240, 240, 255), 4)
        self.app.selected_layer_ids = {layer.id}
        self.app.refresh_properties()
        shape_labels = widget_texts(self.app.object_properties)
        for expected in ("Концы линии", "Соединения", "Предел острого угла", "Штриховка", "Смещение штрихов"):
            self.assertIn(expected, shape_labels)
        self.app.tool.set("direct_select")
        self.app.history.clear()
        nodes = self.app.ensure_editable_path(layer)
        anchor = tuple(round(value) for value in nodes[0]["anchor"])
        self.app.path_pointer_down("direct_select", anchor, 0)
        self.app.path_pointer_drag((anchor[0] + 24, anchor[1] - 12), 0)
        self.app.finish_path_drag()
        self.assertEqual(len(self.app.history.undo_stack), 1)
        self.assertNotEqual(tuple(nodes[0]["anchor"]), anchor)
        self.app.undo()
        self.assertEqual(tuple(round(value) for value in self.app.ensure_editable_path(layer)[0]["anchor"]), anchor)


if __name__ == "__main__":
    unittest.main()
