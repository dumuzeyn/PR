from __future__ import annotations

from ..app_shared import *


class FreeTransformMixin:
    def free_transform_dialog(self, layer) -> dict[str, object] | None:
        dialog = tk.Toplevel(self)
        dialog.title("Свободная трансформация")
        dialog.transient(self)
        dialog.resizable(False, False)
        dialog.grab_set()
        result: dict[str, object] | None = None
        x_var = tk.IntVar(value=int(layer.x))
        y_var = tk.IntVar(value=int(layer.y))
        width_var = tk.IntVar(value=int(layer.pixels.shape[1]))
        height_var = tk.IntVar(value=int(layer.pixels.shape[0]))
        angle_var = tk.DoubleVar(value=0.0)
        flip_h_var = tk.BooleanVar(value=False)
        flip_v_var = tk.BooleanVar(value=False)
        keep_ratio = tk.BooleanVar(value=True)
        original_ratio = layer.pixels.shape[1] / max(1, layer.pixels.shape[0])

        canvas = tk.Canvas(dialog, width=500, height=340, background="#22252b", highlightthickness=0, cursor="crosshair")
        canvas.grid(row=0, column=0, rowspan=9, padx=12, pady=12)
        source = rgba_array_to_pil(layer.pixels)
        handle_positions: dict[str, tuple[float, float]] = {}
        drag_state: dict[str, object] = {"handle": None, "last": (0, 0)}

        def safe_values() -> tuple[int, int, int, int, float]:
            try:
                return int(x_var.get()), int(y_var.get()), max(1, int(width_var.get())), max(1, int(height_var.get())), float(angle_var.get())
            except (tk.TclError, ValueError):
                return layer.x, layer.y, layer.pixels.shape[1], layer.pixels.shape[0], 0.0

        def preview_geometry() -> tuple[float, float, float]:
            scale = min(460 / max(1, self.doc.width), 300 / max(1, self.doc.height))
            return scale, (500 - self.doc.width * scale) / 2, (340 - self.doc.height * scale) / 2

        def redraw(*_args) -> None:
            x, y, width, height, angle = safe_values()
            scale, ox, oy = preview_geometry()
            canvas.delete("all")
            canvas.create_rectangle(ox, oy, ox + self.doc.width * scale, oy + self.doc.height * scale, fill="#343840", outline="#707680")
            preview = source.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BILINEAR)
            if flip_h_var.get():
                preview = preview.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if flip_v_var.get():
                preview = preview.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            if abs(angle) > 0.001:
                preview = preview.rotate(-angle, expand=True, resample=Image.Resampling.BICUBIC)
            self._transform_preview_image = ImageTk.PhotoImage(preview)
            canvas.create_image(ox + (x + width / 2) * scale, oy + (y + height / 2) * scale, image=self._transform_preview_image)
            x1, y1 = ox + x * scale, oy + y * scale
            x2, y2 = ox + (x + width) * scale, oy + (y + height) * scale
            canvas.create_rectangle(x1, y1, x2, y2, outline="#50e3ff", width=2, dash=(5, 3))
            positions = {
                "nw": (x1, y1), "n": ((x1 + x2) / 2, y1), "ne": (x2, y1), "e": (x2, (y1 + y2) / 2),
                "se": (x2, y2), "s": ((x1 + x2) / 2, y2), "sw": (x1, y2), "w": (x1, (y1 + y2) / 2),
            }
            handle_positions.clear(); handle_positions.update(positions)
            for hx, hy in positions.values():
                canvas.create_rectangle(hx - 5, hy - 5, hx + 5, hy + 5, fill="#f7f9fb", outline="#167d96")

        def press(event) -> None:
            nearest = None
            distance = 12.0
            for name, (hx, hy) in handle_positions.items():
                current = ((event.x - hx) ** 2 + (event.y - hy) ** 2) ** 0.5
                if current < distance:
                    nearest, distance = name, current
            if nearest is None:
                x, y, width, height, _angle = safe_values()
                scale, ox, oy = preview_geometry()
                if ox + x * scale <= event.x <= ox + (x + width) * scale and oy + y * scale <= event.y <= oy + (y + height) * scale:
                    nearest = "move"
            drag_state["handle"] = nearest
            drag_state["last"] = (event.x, event.y)

        def drag(event) -> None:
            handle = drag_state.get("handle")
            if not handle:
                return
            last_x, last_y = drag_state["last"]
            scale, _ox, _oy = preview_geometry()
            dx, dy = (event.x - last_x) / scale, (event.y - last_y) / scale
            x, y, width, height, _angle = safe_values()
            if handle == "move":
                x += round(dx); y += round(dy)
            else:
                if "w" in str(handle): x += round(dx); width -= round(dx)
                if "e" in str(handle): width += round(dx)
                if "n" in str(handle): y += round(dy); height -= round(dy)
                if "s" in str(handle): height += round(dy)
                width, height = max(1, width), max(1, height)
                if keep_ratio.get() and str(handle) in {"nw", "ne", "se", "sw"}:
                    if abs(dx) >= abs(dy): height = max(1, round(width / original_ratio))
                    else: width = max(1, round(height * original_ratio))
            x_var.set(x); y_var.set(y); width_var.set(width); height_var.set(height)
            drag_state["last"] = (event.x, event.y)

        for row, (label, variable) in enumerate([("X", x_var), ("Y", y_var), ("Ширина", width_var), ("Высота", height_var), ("Поворот", angle_var)]):
            ttk.Label(dialog, text=label).grid(row=row, column=1, sticky="w", padx=(0, 8), pady=(12 if row == 0 else 4, 0))
            ttk.Spinbox(dialog, textvariable=variable, from_=-100000 if row < 2 else 1, to=100000, width=12).grid(row=row, column=2, sticky="ew", padx=(0, 12), pady=(12 if row == 0 else 4, 0))
        ttk.Checkbutton(dialog, text="Сохранять пропорции", variable=keep_ratio).grid(row=5, column=1, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(dialog, text="Отразить по горизонтали", variable=flip_h_var, command=redraw).grid(row=6, column=1, columnspan=2, sticky="w")
        ttk.Checkbutton(dialog, text="Отразить по вертикали", variable=flip_v_var, command=redraw).grid(row=7, column=1, columnspan=2, sticky="w")
        buttons = ttk.Frame(dialog)
        buttons.grid(row=8, column=1, columnspan=2, sticky="e", padx=12, pady=12)

        def accept() -> None:
            nonlocal result
            x, y, width, height, angle = safe_values()
            result = {"x": x, "y": y, "width": width, "height": height, "angle": angle, "flip_horizontal": flip_h_var.get(), "flip_vertical": flip_v_var.get()}
            dialog.destroy()

        ttk.Button(buttons, text="ОК", command=accept).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT)
        canvas.bind("<ButtonPress-1>", press)
        canvas.bind("<B1-Motion>", drag)
        for variable in [x_var, y_var, width_var, height_var, angle_var]: variable.trace_add("write", redraw)
        redraw()
        dialog.wait_window()
        return result
