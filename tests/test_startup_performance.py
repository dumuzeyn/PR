from __future__ import annotations

import unittest

from uzyro.app import UZYROApp


class StartupPerformanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_clipboard_reader = UZYROApp.read_clipboard_image
        UZYROApp.read_clipboard_image = staticmethod(lambda: None)
        self.app = UZYROApp()
        self.app.update()

    def tearDown(self) -> None:
        self.app.destroy()
        UZYROApp.read_clipboard_image = staticmethod(self.original_clipboard_reader)

    def test_interactive_performance_dialog_runs_benchmark_and_shows_result(self) -> None:
        report = {
            "brush_dab_ms": 1.2,
            "gradient_preview_ms": 22.0,
            "transform_preview_ms": 27.0,
            "many_layers_tile_ms": 5.0,
            "passed": True,
        }
        original = self.app.run_background
        self.app.run_background = lambda _status, _operation, done: done(report)
        try:
            self.app.interactive_performance_dialog()
            self.app._interactive_performance_run()
            self.app.update()
            text = str(self.app._interactive_performance_result.cget("text"))
            self.assertIn("Целевые задержки соблюдены", text)
            self.assertIn("60 слоёв", text)
        finally:
            self.app.run_background = original
            self.app._interactive_performance_dialog.destroy()


if __name__ == "__main__":
    unittest.main()
