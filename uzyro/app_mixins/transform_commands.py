from __future__ import annotations

from ..app_shared import *


class TransformCommandsMixin:
    def transform_selected_pixels(self) -> None:
        layer = self.doc.layer
        if layer.locked:
            self.status_text("Слой заблокирован")
            return
        selection = self.doc.layer_selection_mask(layer)
        if selection is None or not np.any(selection):
            messagebox.showinfo("Трансформация", "Сначала создайте выделение на активном слое.")
            return
        ys, xs = np.where(selection > 0)
        lx1, ly1, lx2, ly2 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
        patch = layer.pixels[ly1:ly2, lx1:lx2].copy()
        patch[:, :, 3] = np.clip(
            patch[:, :, 3].astype(np.float32) * (selection[ly1:ly2, lx1:lx2].astype(np.float32) / 255.0), 0, 255
        ).astype(np.uint8)
        preview = Layer("Выделенные пиксели", patch, x=layer.x + lx1, y=layer.y + ly1)
        data = self.transform_workspace_dialog(preview, "Свободная")
        if data is None:
            return

        def edit() -> None:
            if data["mode"] == "Свободная":
                self.doc.transform_selected_pixels(
                    int(data["x"]), int(data["y"]), int(data["width"]), int(data["height"]),
                    float(data["angle"]), bool(data["flip_horizontal"]), bool(data["flip_vertical"]),
                )
            else:
                self.doc.transform_selected_pixels_advanced(
                    "perspective" if data["mode"] == "Перспектива" else "mesh",
                    data["points"], int(data.get("rows", 4)), int(data.get("columns", 4)),
                    data.get("row_positions"), data.get("column_positions"),
                )

        self.run_document_command("Трансформация выделенных пикселей", edit)
        self.refresh()

    def perspective_transform_layer(self) -> None:
        layer = self.doc.layer
        if layer.locked:
            self.status_text("Слой заблокирован")
            return
        data = self.transform_workspace_dialog(layer, "Перспектива")
        if data is not None:
            self.run_document_command("Перспективная трансформация", lambda: self.apply_transform_workspace_data(data))
            self.refresh()

    def warp_layer(self) -> None:
        layer = self.doc.layer
        if layer.locked:
            self.status_text("Слой заблокирован")
            return
        data = self.transform_workspace_dialog(layer, "Сетка")
        if data is not None:
            self.run_document_command("Деформация слоя", lambda: self.apply_transform_workspace_data(data))
            self.refresh()
