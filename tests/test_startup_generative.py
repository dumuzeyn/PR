from __future__ import annotations

import os
import tempfile
import tkinter as tk
import unittest

import numpy as np

from photoredactor.app import PhotoRedactorApp
from photoredactor.core import Document
from photoredactor.generative_api import GeneratedVariant, variant_seeds
import photoredactor.app_mixins.generative_workspace as workspace_module


class _FakeStabilityClient:
    def __init__(self, api_key: str) -> None:
        if api_key != "test-key":
            raise AssertionError("UI did not pass the configured API key")

    def variants(self, operation, seed: int, count: int) -> list[GeneratedVariant]:
        return [GeneratedVariant(operation(value), value) for value in variant_seeds(seed, count)]

    def inpaint(self, image, mask, prompt, negative_prompt, seed, style):
        assert prompt and negative_prompt == "надпись" and style == "photographic"
        return np.full_like(image, (seed % 255, 45, 210, 255))

    def outpaint(self, image, margins, prompt, seed, creativity, style):
        left, top, right, bottom = margins
        result = np.full(
            (image.shape[0] + top + bottom, image.shape[1] + left + right, 4),
            (30, seed % 255, 180, 255),
            dtype=np.uint8,
        )
        result[top:top + image.shape[0], left:left + image.shape[1]] = image
        return result


class StartupGenerativeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.original_local_app_data = os.environ.get("LOCALAPPDATA")
        self.original_api_key = os.environ.get("STABILITY_API_KEY")
        self.original_client = workspace_module.StabilityImageClient
        self.original_clipboard_reader = PhotoRedactorApp.read_clipboard_image
        os.environ["LOCALAPPDATA"] = self.temp.name
        os.environ["STABILITY_API_KEY"] = "test-key"
        workspace_module.StabilityImageClient = _FakeStabilityClient
        PhotoRedactorApp.read_clipboard_image = staticmethod(lambda: None)
        self.app = PhotoRedactorApp()
        self.app.run_background = lambda _label, worker, done=None, is_current=None: done(worker()) if done else worker()
        self.app.update()

    def tearDown(self) -> None:
        self.app.destroy()
        PhotoRedactorApp.read_clipboard_image = staticmethod(self.original_clipboard_reader)
        workspace_module.StabilityImageClient = self.original_client
        if self.original_local_app_data is None:
            os.environ.pop("LOCALAPPDATA", None)
        else:
            os.environ["LOCALAPPDATA"] = self.original_local_app_data
        if self.original_api_key is None:
            os.environ.pop("STABILITY_API_KEY", None)
        else:
            os.environ["STABILITY_API_KEY"] = self.original_api_key
        self.temp.cleanup()

    def test_fill_variants_apply_as_layer_and_undo_redo(self) -> None:
        self.app.doc = Document.new(24, 18, (12, 22, 32, 255))
        self.app.doc.selection_mask = np.zeros((18, 24), dtype=np.uint8)
        self.app.doc.selection_mask[4:13, 6:19] = 255
        self.app.doc.selection_mask[4, 6:19] = 128
        original = self.app.doc.composite(False).copy()
        self.app.generative_fill_dialog()
        self.app._generative_workspace_prompt.set("красная чашка")
        self.app._generative_workspace_negative.set("надпись")
        self.app._generative_workspace_seed.set(100)
        self.app._generative_workspace_variants.set(2)
        self.app._generative_workspace_generate()
        self.app.update()

        self.assertEqual([item[0].seed for item in self.app._generative_workspace_history], [100, 101])
        self.app._generative_workspace_apply()
        self.app.update()
        self.assertEqual(len(self.app.doc.layers), 2)
        generated = self.app.doc.layer
        self.assertEqual(generated.generation_data["prompt"], "красная чашка")
        self.assertEqual(generated.generation_data["variant_seeds"], [100, 101])
        self.assertFalse(np.any(generated.pixels[self.app.doc.selection_mask == 0]))
        self.assertTrue(np.all(generated.pixels[4, 6:19, 3] == 128))
        composite = self.app.doc.composite(False)
        np.testing.assert_array_equal(composite[self.app.doc.selection_mask == 0], original[self.app.doc.selection_mask == 0])

        self.app.undo()
        self.assertEqual(len(self.app.doc.layers), 1)
        self.app.redo()
        self.assertEqual(len(self.app.doc.layers), 2)
        self.assertEqual(self.app.doc.layer.generation_data["seed"], 100)

    def test_models_menu_follows_view_and_manager_lists_downloads(self) -> None:
        labels = [self.app.editor_menu.entrycget(index, "label") for index in range(self.app.editor_menu.index(tk.END) + 1)]
        self.assertEqual(labels[labels.index("Вид") + 1], "Модели")
        self.app.model_manager_dialog()
        self.app.update()
        self.assertEqual(
            self.app._model_manager_tree.get_children(),
            ("realistic-vision-51-inpaint", "sd15-inpaint"),
        )

    def test_local_performance_profiles_update_generation_controls(self) -> None:
        self.app.doc = Document.new(24, 18, (12, 22, 32, 255))
        self.app.doc.selection_mask = np.full((18, 24), 255, dtype=np.uint8)
        self.app.generative_fill_dialog()

        self.app._generative_workspace_performance.set("Быстро")
        self.app._generative_workspace_profile_box.event_generate("<<ComboboxSelected>>")
        self.app.update()

        self.assertEqual(self.app._generative_workspace_steps.get(), 4)
        self.assertAlmostEqual(self.app._generative_workspace_cfg.get(), 1.2)
        self.assertEqual(self.app._generative_workspace_sampler.get(), "LCM")
        self.app._generative_workspace_steps.set(9)
        self.assertEqual(self.app._generative_workspace_performance.get(), "Вручную")

    def test_expand_applies_new_canvas_and_has_exact_undo(self) -> None:
        self.app.doc = Document.new(20, 14, (70, 80, 90, 255))
        self.app.generative_expand_dialog()
        self.app._generative_workspace_prompt.set("лес за границами кадра")
        self.app._generative_workspace_seed.set(230)
        for variable, value in zip(self.app._generative_workspace_margins, (3, 2, 5, 4)):
            variable.set(value)
        self.app._generative_workspace_variants.set(1)
        self.app._generative_workspace_generate()
        self.app._generative_workspace_apply()
        self.app.update()

        self.assertEqual((self.app.doc.width, self.app.doc.height), (28, 20))
        self.assertEqual(len(self.app.doc.layers), 2)
        self.assertEqual((self.app.doc.layers[1].x, self.app.doc.layers[1].y), (3, 2))
        self.assertEqual(self.app.doc.layers[0].generation_data["margins"], [3, 2, 5, 4])
        self.app.undo()
        self.assertEqual((self.app.doc.width, self.app.doc.height), (20, 14))
        self.assertEqual(len(self.app.doc.layers), 1)
        self.assertEqual((self.app.doc.layer.x, self.app.doc.layer.y), (0, 0))
        self.app.redo()
        self.assertEqual((self.app.doc.width, self.app.doc.height), (28, 20))
        self.assertEqual(len(self.app.doc.layers), 2)


if __name__ == "__main__":
    unittest.main()
