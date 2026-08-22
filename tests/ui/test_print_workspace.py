from __future__ import annotations

from unittest.mock import patch

from uzyro.spot_colors import SpotColor, assign_spot_color, replace_document_spot_colors
from tests.ui.support import ApplicationUITestCase


class PrintWorkspaceUITests(ApplicationUITestCase):

    def test_spot_workspace_and_system_print_entrypoints(self) -> None:
        color = SpotColor("Пробная плашка", (55.0, 30.0, -20.0), (175, 115, 165), "Тест", "test-spot")
        replace_document_spot_colors(self.app.doc, [color])
        assign_spot_color(self.app.doc, self.app.doc.layer.id, color.id)
        self.app.spot_colors_workspace()
        self.app.update()
        tree = self.app._spot_color_tree
        self.assertEqual(len(tree.get_children()), 1)
        self.assertIn("Пробная плашка", tree.item(tree.get_children()[0], "text"))
        tree.winfo_toplevel().destroy()

        with patch("uzyro.app_mixins.print_spot_workspace.print_document", return_value=True) as printer:
            self.app.system_print_document()
        printer.assert_called_once()


if __name__ == "__main__":
    unittest.main()
