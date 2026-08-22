from __future__ import annotations

import unittest

from uzyro.app import UZYROApp


class ApplicationUITestCase(unittest.TestCase):
    """A fresh application per test with deterministic clipboard and cleanup."""

    def setUp(self) -> None:
        super().setUp()
        self.original_clipboard_reader = UZYROApp.read_clipboard_image
        UZYROApp.read_clipboard_image = staticmethod(lambda: None)
        self.app = UZYROApp()
        self.app.update()

    def tearDown(self) -> None:
        try:
            self.app.destroy()
        finally:
            UZYROApp.read_clipboard_image = staticmethod(self.original_clipboard_reader)
            super().tearDown()
