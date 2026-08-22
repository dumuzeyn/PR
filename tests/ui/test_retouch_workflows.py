from __future__ import annotations

from types import SimpleNamespace
from tkinter import ttk

import numpy as np

from uzyro.history import PixelTilePatchCommand
from tests.ui.support import ApplicationUITestCase


class RetouchWorkflowUITests(ApplicationUITestCase):
    def setUp(self) -> None:
        super().setUp()
        self.app.create_document_from_settings(
            {"width": 220, "height": 140, "dpi": 72, "background": (180, 170, 160, 255), "include_clipboard": False}
        )
        self.app.update()

    def pointer(self, point: tuple[int, int], state: int = 0):
        x, y = self.app.doc_to_canvas(*point)
        return SimpleNamespace(
            x=x - self.app.canvas.canvasx(0),
            y=y - self.app.canvas.canvasy(0),
            state=state,
            pressure=1.0,
        )

    def test_clone_drag_uses_source_transform_and_one_undo(self) -> None:
        layer = self.app.doc.layer
        layer.pixels[25:70, 20:65, :3] = (225, 35, 30)
        before = layer.pixels.copy()
        self.app.tool.set("clone")
        self.app.brush_size.set(12)
        self.app.brush_flow.set(0.35)
        self.app.clone_scale_x.set(140)
        self.app.clone_rotation.set(20)
        self.app.set_clone_source((40, 45))
        self.app.history.clear()
        self.app.pointer_down(self.pointer((140, 70)))
        for x in range(145, 181, 5):
            self.app.pointer_drag(self.pointer((x, 70)))
        self.app.pointer_up(self.pointer((180, 70)))
        self.assertEqual(len(self.app.history.undo_stack), 1)
        self.assertIsInstance(self.app.history.undo_stack[0], PixelTilePatchCommand)
        self.assertFalse(np.array_equal(before, layer.pixels))
        self.app.undo()
        np.testing.assert_array_equal(layer.pixels, before)

    def test_clone_overlay_exists_before_painting_and_can_be_hidden(self) -> None:
        self.app.tool.set("healing")
        self.app.clone_overlay_visible.set(True)

    def test_clone_source_panel_exposes_transform_controls(self) -> None:
        self.app.tool.set("clone")
        self.app.open_clone_source_panel()
        self.app.update()
        dialog = self.app._clone_source_dialog
        self.assertIsNotNone(dialog)
        self.assertEqual(dialog.title(), "Источник клонирования")

        def descendants(widget):
            result = []
            for child in widget.winfo_children():
                result.append(child)
                result.extend(descendants(child))
            return result

        widgets = descendants(dialog)
        self.assertGreaterEqual(sum(isinstance(widget, ttk.Spinbox) for widget in widgets), 5)
        texts = {str(widget.cget("text")) for widget in widgets if "text" in widget.keys()}
        self.assertIn("Отразить X", texts)
        self.assertIn("Показывать до рисования", texts)
        dialog.destroy()
        self.app.set_clone_source((45, 50))
        self.app.update_brush_preview(self.pointer((145, 80)))
        self.assertIsNotNone(self.app._clone_overlay_id)
        self.assertIsNotNone(self.app._clone_overlay_image)
        self.app.clone_overlay_visible.set(False)
        self.app.update_brush_preview(self.pointer((145, 80)))
        self.assertIsNone(self.app._clone_overlay_id)
        self.app.clone_overlay_visible.set(True)

    def test_healing_and_spot_healing_each_create_one_tile_undo(self) -> None:
        rng = np.random.default_rng(22)
        source = rng.integers(60, 210, self.app.doc.layer.pixels.shape, dtype=np.uint8)
        source[:, :, 3] = 255
        for tool in ("healing", "spot_healing"):
            self.app.doc.layer.pixels[:] = source
            self.app.history.clear()
            self.app.tool.set(tool)
            self.app.brush_size.set(10)
            self.app.brush_flow.set(0.5)
            if tool == "healing":
                self.app.set_clone_source((45, 45))
            self.app.pointer_down(self.pointer((130, 75)))
            self.app.pointer_drag(self.pointer((170, 75)))
            self.app.pointer_up(self.pointer((170, 75)))
            self.assertEqual(len(self.app.history.undo_stack), 1, tool)
            self.assertIsInstance(self.app.history.undo_stack[0], PixelTilePatchCommand)
            self.app.undo()
            np.testing.assert_array_equal(self.app.doc.layer.pixels, source)

    def test_patch_preview_waits_for_enter_and_escape_cancels(self) -> None:
        layer = self.app.doc.layer
        layer.pixels[20:60, 15:55, :3] = (35, 70, 210)
        layer.pixels[70:100, 125:165, :3] = 8
        mask = np.zeros((140, 220), dtype=np.uint8)
        mask[70:100, 125:165] = 255
        self.app.doc.selection_mask = mask
        self.app.tool.set("patch")
        self.app.history.clear()
        before = layer.pixels.copy()

        self.app.pointer_down(self.pointer((140, 80)))
        self.app.pointer_drag(self.pointer((30, 30)))
        self.app.pointer_up(self.pointer((30, 30)))
        np.testing.assert_array_equal(layer.pixels, before)
        self.assertIsNotNone(self.app._patch_pending_bounds)
        self.assertEqual(len(self.app.history.undo_stack), 0)
        self.app.cancel_incomplete_interaction()
        np.testing.assert_array_equal(layer.pixels, before)
        self.assertIsNone(self.app._patch_pending_bounds)

        self.app.pointer_down(self.pointer((140, 80)))
        self.app.pointer_drag(self.pointer((30, 30)))
        self.app.pointer_up(self.pointer((30, 30)))
        self.assertEqual(self.app.shortcut_enter(), "break")
        self.assertFalse(np.array_equal(layer.pixels, before))
        self.assertEqual(len(self.app.history.undo_stack), 1)
        self.assertIsInstance(self.app.history.undo_stack[0], PixelTilePatchCommand)
        self.app.undo()
        np.testing.assert_array_equal(layer.pixels, before)

    def test_clone_sources_keep_independent_points_and_transforms(self) -> None:
        self.app.ensure_clone_sources()
        self.assertEqual(tuple(self.app._clone_sources), ("A", "B", "C", "D", "E"))
        self.app.switch_clone_source("A")
        self.app.set_clone_source((18, 22))
        self.app.clone_scale_x.set(135)
        self.app.clone_rotation.set(17)
        self.app.clone_overlay_invert.set(True)
        self.app.switch_clone_source("B")
        self.app.set_clone_source((72, 64))
        self.app.clone_scale_x.set(80)
        self.app.clone_rotation.set(-28)
        self.app.clone_overlay_clipped.set(False)
        self.app.switch_clone_source("A")
        self.assertEqual(self.app._source_anchor.point, (18, 22))
        self.assertEqual(self.app.clone_scale_x.get(), 135)
        self.assertEqual(self.app.clone_rotation.get(), 17)
        self.assertTrue(self.app.clone_overlay_invert.get())
        self.app.switch_clone_source("B")
        self.assertEqual(self.app._source_anchor.point, (72, 64))
        self.assertEqual(self.app.clone_scale_x.get(), 80)
        self.assertEqual(self.app.clone_rotation.get(), -28)
        self.assertFalse(self.app.clone_overlay_clipped.get())
        payload = self.app.source_retouch_settings_payload()
        self.assertEqual(payload["active_source"], "B")
        self.assertEqual(len(payload["sources"]), 5)
        self.assertTrue(payload["sources"]["A"]["layer_id"])
        self.assertNotIn("_sample_pixels", payload["sources"]["A"])
        self.assertIsInstance(self.app._clone_sources["A"].get("_sample_pixels"), np.ndarray)


if __name__ == "__main__":
    unittest.main()
