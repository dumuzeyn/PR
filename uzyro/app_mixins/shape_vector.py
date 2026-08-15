from __future__ import annotations

from ..app_shared import *


class ShapeVectorMixin:
    def edit_shape_layer(self) -> None:
        layer = self.doc.layer
        if layer.kind != "shape" or layer.shape_data is None:
            messagebox.showinfo("Свойства фигуры", "Сначала выберите слой-фигуру.")
            return
        self.refresh_properties()
        self.right_tabs.select(1)
        self.status_text("Свойства фигуры открыты в правой панели")

    def edit_bezier_points(self) -> None:
        layer = self.doc.layer
        if layer.kind != "shape" or layer.shape_data is None or str(layer.shape_data.get("shape")) != "bezier":
            messagebox.showinfo("Точки Безье", "Выберите слой с кривой Безье.")
            return
        raw_points = layer.shape_data.get("control_points")
        if not isinstance(raw_points, list) or len(raw_points) != 4:
            x1, y1, x2, y2 = [float(v) for v in layer.shape_data.get("box", [0, 0, 1, 1])]
            raw_points = [[x1, y2], [x1, y1], [x2, y1], [x2, y2]]
        points = [[float(point[0]), float(point[1])] for point in raw_points]
        dialog = tk.Toplevel(self)
        dialog.title("Редактор точек Безье")
        dialog.transient(self)
        dialog.resizable(False, False)
        dialog.grab_set()
        canvas = tk.Canvas(dialog, width=620, height=440, background="#22252b", highlightthickness=0, cursor="crosshair")
        canvas.pack(padx=12, pady=(12, 6))
        buttons = ttk.Frame(dialog)
        buttons.pack(fill=tk.X, padx=12, pady=(0, 12))
        active_point: list[int | None] = [None]
        scale = min(580 / max(1, self.doc.width), 400 / max(1, self.doc.height))
        ox, oy = (620 - self.doc.width * scale) / 2, (440 - self.doc.height * scale) / 2
        background = rgba_array_to_pil(self.render_engine.render(self.doc, checker=True)).resize((max(1, round(self.doc.width * scale)), max(1, round(self.doc.height * scale))), Image.Resampling.BILINEAR)
        self._bezier_preview_image = ImageTk.PhotoImage(background)

        def to_canvas(point: list[float] | tuple[float, float]) -> tuple[float, float]:
            return ox + float(point[0]) * scale, oy + float(point[1]) * scale

        def redraw() -> None:
            canvas.delete("all")
            canvas.create_image(ox, oy, image=self._bezier_preview_image, anchor=tk.NW)
            p = [to_canvas(point) for point in points]
            canvas.create_line(*p[0], *p[1], fill="#ffd166", dash=(4, 3), width=1)
            canvas.create_line(*p[2], *p[3], fill="#ffd166", dash=(4, 3), width=1)
            curve = bezier_curve_points(points, tuple(int(v) for v in layer.shape_data.get("box", [0, 0, 1, 1])), 96)
            coords = [value for point in curve for value in to_canvas(point)]
            canvas.create_line(*coords, fill="#50e3ff", width=3, smooth=True)
            for index, (px, py) in enumerate(p):
                color = "#50e3ff" if index in {0, 3} else "#ffd166"
                if index in {0, 3}:
                    canvas.create_oval(px - 8, py - 8, px + 8, py + 8, fill=color, outline="#111318", width=2)
                else:
                    canvas.create_rectangle(px - 8, py - 8, px + 8, py + 8, fill=color, outline="#111318", width=2)
                label = ("P0 начало", "P1 ручка", "P2 ручка", "P3 конец")[index]
                canvas.create_text(px, py - 17, text=label, fill=color, font=("Segoe UI", 9, "bold"))

        def press(event) -> None:
            active_point[0] = None
            best = 14.0
            for index, point in enumerate(points):
                px, py = to_canvas(point)
                distance = ((event.x - px) ** 2 + (event.y - py) ** 2) ** 0.5
                if distance < best:
                    active_point[0], best = index, distance

        def drag(event) -> None:
            if active_point[0] is None:
                return
            index = int(active_point[0])
            points[index] = [(event.x - ox) / scale, (event.y - oy) / scale]
            redraw()

        def accept() -> None:
            self.run_shape_data_command("Edit Bezier points", lambda: self.doc.edit_shape_layer(control_points=[tuple(point) for point in points]))
            dialog.destroy()
            self.refresh()

        ttk.Label(buttons, text="Круги - концы, квадраты - управляющие ручки.").pack(side=tk.LEFT)
        ttk.Button(buttons, text="ОК", command=accept).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT)
        ToolTip(canvas, "Перетаскивайте конечные P0/P3 и управляющие P1/P2.")
        canvas.bind("<ButtonPress-1>", press)
        canvas.bind("<B1-Motion>", drag)
        redraw()
        dialog.wait_window()

    def boolean_shape_layers(self) -> None:
        initial = self.doc.boolean_shape_data_with_lower("union")
        if initial is None:
            messagebox.showinfo("Булева операция фигур", "Выберите незаблокированную фигуру, расположенную прямо над другой фигурой.")
            return
        edited = self.boolean_shape_editor(initial, "Создать булеву фигуру")
        if edited is None:
            return
        mode = str(edited.get("boolean_mode", "union"))
        self.run_document_command("Создать булеву фигуру", lambda: self.doc.boolean_active_shape_with_lower(mode, edited))
        self.selected_layer_ids = {self.doc.layer.id}
        self.refresh()

    def edit_boolean_shape(self) -> None:
        layer = self.doc.layer
        if layer.kind != "shape" or layer.shape_data is None or str(layer.shape_data.get("shape")) != "boolean":
            return
        edited = self.boolean_shape_editor(layer.shape_data, "Редактировать булеву фигуру")
        if edited is None or edited == layer.shape_data:
            return

        def apply() -> None:
            layer.shape_data = copy.deepcopy(edited)
            layer.name = f"Булева фигура: {edited.get('boolean_mode', 'union')}"
            render_shape_layer(layer)
            layer.touch_pixels()
            self.doc.dirty = True

        self.run_shape_data_command("Редактировать булеву фигуру", apply)
        self.refresh()
