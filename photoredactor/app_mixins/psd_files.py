from __future__ import annotations

from ..app_shared import *


class PSDFileMixin:
    def export_psd_compatible(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".psd",
            filetypes=[("Photoshop PSD", "*.psd"), ("Photoshop Large Document PSB", "*.psb")],
        )
        if not path:
            return
        snapshot = self.document_copy()

        def done(report: dict[str, object]) -> None:
            warnings = list(report.get("warnings", []))
            text = f"Файл {report['format']} сохранён."
            if warnings:
                text += "\n\n" + "\n".join(str(item) for item in warnings[:8])
            messagebox.showinfo("Экспорт PSD/PSB", text)

        self.run_background("Экспорт PSD/PSB", lambda: export_psd(snapshot, path), done)
