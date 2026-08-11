from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from photoredactor.app import PhotoRedactorApp
from photoredactor.core import Document
from photoredactor.psd_compat import export_psd
import photoredactor.app_mixins.psd_files as psd_files_module


class StartupPSDTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.original_local_app_data = os.environ.get("LOCALAPPDATA")
        self.original_clipboard_reader = PhotoRedactorApp.read_clipboard_image
        os.environ["LOCALAPPDATA"] = self.temp.name
        PhotoRedactorApp.read_clipboard_image = staticmethod(lambda: None)
        self.app = PhotoRedactorApp()
        self.app.run_background = lambda _label, worker, done=None, is_current=None: done(worker()) if done else worker()
        self.app.update()

    def tearDown(self) -> None:
        self.app.destroy()
        PhotoRedactorApp.read_clipboard_image = staticmethod(self.original_clipboard_reader)
        if self.original_local_app_data is None:
            os.environ.pop("LOCALAPPDATA", None)
        else:
            os.environ["LOCALAPPDATA"] = self.original_local_app_data
        self.temp.cleanup()

    def test_open_psd_and_export_psb_from_application(self) -> None:
        source = Path(self.temp.name) / "source.psd"
        export_psd(Document.new(26, 19, (25, 50, 75, 255)), source)
        self.app.open_path(str(source))
        self.app.update()
        self.assertEqual((self.app.doc.width, self.app.doc.height), (26, 19))
        self.assertEqual(self.app.doc.metadata["psd_compatibility"]["format"], "PSD")
        self.assertTrue(self.app.editor_root.winfo_ismapped())

        target = Path(self.temp.name) / "export.psb"
        original_dialog = psd_files_module.filedialog.asksaveasfilename
        original_info = psd_files_module.messagebox.showinfo
        messages: list[str] = []
        psd_files_module.filedialog.asksaveasfilename = lambda **_kwargs: str(target)
        psd_files_module.messagebox.showinfo = lambda _title, text: messages.append(str(text))
        try:
            self.app.export_psd_compatible()
        finally:
            psd_files_module.filedialog.asksaveasfilename = original_dialog
            psd_files_module.messagebox.showinfo = original_info
        self.assertTrue(target.exists())
        self.assertIn("PSB сохранён", messages[0])


if __name__ == "__main__":
    unittest.main()
