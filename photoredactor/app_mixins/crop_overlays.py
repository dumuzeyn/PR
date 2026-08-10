from __future__ import annotations

from ..app_shared import *


class CropOverlaysMixin:
    def crop_box_for_drag(self, start: tuple[int, int], end: tuple[int, int]) -> tuple[int, int, int, int]:
        sx, sy = start
        ex, ey = end
        ratios = {
            "1:1": 1.0,
            "4:3": 4.0 / 3.0,
            "3:2": 3.0 / 2.0,
            "16:9": 16.0 / 9.0,
            "Исходное": self.doc.width / max(1, self.doc.height),
            "Свое": max(1, int(self.crop_custom_width.get())) / max(1, int(self.crop_custom_height.get())),
        }
        ratio = ratios.get(self.crop_aspect.get())
        if ratio is not None:
            dx, dy = ex - sx, ey - sy
            sign_x = -1 if dx < 0 else 1
            sign_y = -1 if dy < 0 else 1
            width, height = abs(dx), abs(dy)
            if height == 0 or width / max(1, height) > ratio:
                height = round(width / ratio)
            else:
                width = round(height * ratio)
            ex, ey = sx + sign_x * width, sy + sign_y * height
        x1, x2 = sorted((max(0, min(self.doc.width, sx)), max(0, min(self.doc.width, ex))))
        y1, y2 = sorted((max(0, min(self.doc.height, sy)), max(0, min(self.doc.height, ey))))
        return int(x1), int(y1), int(x2), int(y2)

    def crop_handle_at(self, point: tuple[int, int]) -> str | None:
        if self._crop_box is None:
            return None
        x1, y1, x2, y2 = self._crop_box
        handles = {
            "nw": (x1, y1), "n": ((x1 + x2) / 2, y1), "ne": (x2, y1),
            "w": (x1, (y1 + y2) / 2), "e": (x2, (y1 + y2) / 2),
            "sw": (x1, y2), "s": ((x1 + x2) / 2, y2), "se": (x2, y2),
        }
        threshold = max(3.0, 9.0 / max(0.05, float(self.zoom.get())))
        nearest = None
        best = threshold
        for name, (hx, hy) in handles.items():
            distance = math.hypot(point[0] - hx, point[1] - hy)
            if distance <= best:
                nearest, best = name, distance
        return nearest

    def resize_crop_box(
        self,
        box: tuple[int, int, int, int],
        handle: str,
        point: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        px = max(0, min(self.doc.width, int(point[0])))
        py = max(0, min(self.doc.height, int(point[1])))
        if "w" in handle:
            x1 = px
        if "e" in handle:
            x2 = px
        if "n" in handle:
            y1 = py
        if "s" in handle:
            y2 = py
        if len(handle) == 2 and self.crop_aspect.get() != "Свободно":
            anchor_x = x2 if "w" in handle else x1
            anchor_y = y2 if "n" in handle else y1
            return self.crop_box_for_drag((anchor_x, anchor_y), (px, py))
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        return int(left), int(top), int(right), int(bottom)

    def draw_crop_overlay(self, box: tuple[int, int, int, int]) -> None:
        self.clear_crop_overlay()
        x1, y1, x2, y2 = box
        left, top = self.doc_to_canvas(0, 0)
        right, bottom = self.doc_to_canvas(self.doc.width, self.doc.height)
        cx1, cy1 = self.doc_to_canvas(x1, y1)
        cx2, cy2 = self.doc_to_canvas(x2, y2)
        shade = {"fill": "#111111", "outline": "", "stipple": "gray50"}
        self._crop_overlay_ids.extend([
            self.canvas.create_rectangle(left, top, right, cy1, **shade),
            self.canvas.create_rectangle(left, cy2, right, bottom, **shade),
            self.canvas.create_rectangle(left, cy1, cx1, cy2, **shade),
            self.canvas.create_rectangle(cx2, cy1, right, cy2, **shade),
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="#f5f5f5", width=2),
        ])
        for fraction in (1 / 3, 2 / 3):
            gx = cx1 + (cx2 - cx1) * fraction
            gy = cy1 + (cy2 - cy1) * fraction
            self._crop_overlay_ids.append(self.canvas.create_line(gx, cy1, gx, cy2, fill="#e0e0e0", dash=(4, 4)))
            self._crop_overlay_ids.append(self.canvas.create_line(cx1, gy, cx2, gy, fill="#e0e0e0", dash=(4, 4)))
        handles = [
            (cx1, cy1), ((cx1 + cx2) / 2, cy1), (cx2, cy1),
            (cx1, (cy1 + cy2) / 2), (cx2, (cy1 + cy2) / 2),
            (cx1, cy2), ((cx1 + cx2) / 2, cy2), (cx2, cy2),
        ]
        for hx, hy in handles:
            self._crop_overlay_ids.append(self.canvas.create_rectangle(hx - 4, hy - 4, hx + 4, hy + 4, fill="#ffffff", outline="#1976d2"))
        for item_id in self._crop_overlay_ids:
            self.canvas.tag_raise(item_id)

    def clear_crop_overlay(self) -> None:
        if not hasattr(self, "canvas"):
            return
        for item_id in self._crop_overlay_ids:
            self.canvas.delete(item_id)
        self._crop_overlay_ids.clear()

    def apply_crop_overlay(self) -> None:
        if self._crop_box is None:
            self.status_text("Сначала протяните рамку кадрирования")
            return
        x1, y1, x2, y2 = self._crop_box
        if x2 - x1 < 2 or y2 - y1 < 2:
            self.status_text("Область кадрирования слишком мала")
            return
        self.run_document_command("Crop", lambda: self.doc.crop(self._crop_box))
        self.doc.clear_selection()
        self.selection_box = None
        self._crop_box = None
        self.clear_crop_overlay()
        self.refresh()

    def draw_lasso(self) -> None:
        self.delete_lasso_overlay()
        if len(self._lasso_points) < 2:
            return
        coords = [coord for point in self._lasso_points for xy in [self.doc_to_canvas(point[0], point[1])] for coord in xy]
        self._polygon_ids.append(self.canvas.create_line(*coords, fill="#50e3ff", dash=(4, 3), width=2, smooth=True))

    def draw_polygon_lasso(self, hover: tuple[int, int] | None = None) -> None:
        self.delete_lasso_overlay()
        preview_points = list(self._polygon_points)
        if hover is not None and preview_points:
            preview_points.append(hover)
        if len(preview_points) >= 2:
            coords = [coord for point in preview_points for xy in [self.doc_to_canvas(point[0], point[1])] for coord in xy]
            self._polygon_ids.append(self.canvas.create_line(*coords, fill="#50e3ff", dash=(4, 3), width=2))
        for x, y in self._polygon_points:
            cx, cy = self.doc_to_canvas(x, y)
            self._polygon_ids.append(self.canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill="#50e3ff", outline=""))

    def clear_lasso_overlay(self) -> None:
        self.delete_lasso_overlay()
        self._lasso_points.clear()
        self._polygon_points.clear()
        self._magnetic_edges = None

    def delete_lasso_overlay(self) -> None:
        for item_id in self._polygon_ids:
            self.canvas.delete(item_id)
        self._polygon_ids.clear()

    def magnetic_lasso_point(self, point: tuple[int, int]) -> tuple[int, int]:
        if self._magnetic_edges is None:
            self._magnetic_edges = self.doc.magnetic_edge_map(self.render_engine.render(self.doc, False))
        return self.doc.snap_point_to_edge(point, self._magnetic_edges, max(8, int(self.tolerance.get())))

    def clear_selection_overlay(self) -> None:
        if self.selection_id is not None:
            self.canvas.delete(self.selection_id)
            self.selection_id = None
        for item_id in self._selection_overlay_ids:
            self.canvas.delete(item_id)
        self._selection_overlay_ids.clear()
        if self._selection_animation_id is not None:
            try:
                self.after_cancel(self._selection_animation_id)
            except tk.TclError:
                pass
            self._selection_animation_id = None

    def selection_contours(self) -> list[np.ndarray]:
        mask = self.doc.selection_mask
        if mask is None or not np.any(mask >= 128):
            self._selection_contours = []
            self._selection_contour_signature = None
            return []
        signature = (id(mask), mask.shape[0], mask.shape[1])
        if signature != self._selection_contour_signature:
            self._selection_contours = selection_contour_points(mask)
            self._selection_contour_signature = signature
        return self._selection_contours

    def update_selection_overlay(self) -> None:
        self.clear_selection_overlay()
        contours = self.selection_contours()
        if not contours:
            return
        epsilon = max(0.5, 0.8 / max(0.05, float(self.zoom.get())))
        for contour in contours:
            simplified = cv2.approxPolyDP(contour.reshape(-1, 1, 2), epsilon, True)[:, 0, :]
            if len(simplified) < 2:
                continue
            closed = np.vstack([simplified, simplified[0]])
            coords = [value for x, y in closed for value in self.doc_to_canvas(float(x), float(y))]
            dark = self.canvas.create_line(*coords, fill="#111111", width=2, dash=(5, 5), dashoffset=self._selection_dash_phase)
            light = self.canvas.create_line(*coords, fill="#f4f4f4", width=1, dash=(5, 5), dashoffset=self._selection_dash_phase + 5)
            self._selection_overlay_ids.extend([dark, light])
        for item_id in self._selection_overlay_ids:
            self.canvas.tag_raise(item_id)
        if self._selection_overlay_ids:
            self._selection_animation_id = self.after(240, self.animate_selection_overlay)
        self.update_grid_and_guides()

    def animate_selection_overlay(self) -> None:
        self._selection_animation_id = None
        if not self._selection_overlay_ids:
            return
        self._selection_dash_phase = (self._selection_dash_phase + 1) % 10
        for index, item_id in enumerate(self._selection_overlay_ids):
            try:
                self.canvas.itemconfigure(item_id, dashoffset=self._selection_dash_phase + (5 if index % 2 else 0))
            except tk.TclError:
                return
        self._selection_animation_id = self.after(240, self.animate_selection_overlay)

    def update_grid_and_guides(self) -> None:
        for item_id in self._overlay_ids:
            self.canvas.delete(item_id)
        self._overlay_ids.clear()
        ox, oy = self._canvas_origin
        right, bottom = self.doc_to_canvas(self.doc.width, self.doc.height)
        if self.grid_visible.get():
            spacing = max(4, int(self.grid_spacing.get()))
            for x in range(spacing, self.doc.width, spacing):
                cx, _ = self.doc_to_canvas(x, 0)
                self._overlay_ids.append(self.canvas.create_line(cx, oy, cx, bottom, fill="#3b3f48", dash=(2, 6)))
            for y in range(spacing, self.doc.height, spacing):
                _, cy = self.doc_to_canvas(0, y)
                self._overlay_ids.append(self.canvas.create_line(ox, cy, right, cy, fill="#3b3f48", dash=(2, 6)))
        for orientation, value in self._guide_doc_lines:
            if orientation == "h":
                _, cy = self.doc_to_canvas(0, value)
                self._overlay_ids.append(self.canvas.create_line(ox, cy, right, cy, fill="#ff4fd8", width=1))
            else:
                cx, _ = self.doc_to_canvas(value, 0)
                self._overlay_ids.append(self.canvas.create_line(cx, oy, cx, bottom, fill="#ff4fd8", width=1))
        for item_id in self._overlay_ids:
            self.canvas.tag_raise(item_id)

    def create_shape_from_drag(self, tool: str, geometry: dict) -> None:
        options = self._shape_drag_options or self.current_shape_options(tool)
        shape = str(geometry["shape"])
        box = tuple(int(v) for v in geometry.get("line", geometry["box"]))
        layer = self.doc.add_shape_layer(
            shape,
            box,
            options["fill"],
            options["stroke"],
            int(options["stroke_width"]),
            int(options["sides"]),
            float(options["inner_ratio"]),
            custom_points=options["custom_points"],
        )
        self.selected_layer_ids = {layer.id}
        self.push_command(LayerInsertCommand("Shape layer", self.doc.active_layer, copy.deepcopy(layer)))

    def run_shape_data_command(self, label: str, edit) -> None:
        layer = self.doc.layer
        if layer.kind != "shape" or layer.shape_data is None:
            return
        before = copy.deepcopy(layer.shape_data)
        before_name = layer.name
        edit()
        after = copy.deepcopy(layer.shape_data or {})
        if before != after or before_name != layer.name:
            self.push_command(ShapeDataCommand(label, layer.id, before, after, before_name, layer.name))

    def clear_selection(self) -> None:
        self.run_selection_command("Clear selection", self.doc.clear_selection)

    def copy_pixels(self) -> None:
        if not self._editor_active:
            return
        layer = self.doc.layer
        selection = self.doc.layer_selection_mask(layer)
        if selection is None:
            self._pixel_clipboard = layer.pixels.copy()
            self._pixel_clipboard_origin = (layer.x, layer.y)
        else:
            ys, xs = np.where(selection > 0)
            if len(xs) == 0:
                return
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            pixels = layer.pixels[y1:y2, x1:x2].copy()
            coverage = selection[y1:y2, x1:x2].astype(np.float32) / 255.0
            pixels[:, :, 3] = np.clip(pixels[:, :, 3].astype(np.float32) * coverage, 0, 255).astype(np.uint8)
            self._pixel_clipboard = pixels
            self._pixel_clipboard_origin = (layer.x + x1, layer.y + y1)
        self.status_text("Пиксели скопированы")

    def paste_pixels(self) -> None:
        if self._pixel_clipboard is None:
            self.status_text("Буфер пикселей пуст")
            return
        layer = Layer(
            "Вставленные пиксели",
            self._pixel_clipboard.copy(),
            x=int(self._pixel_clipboard_origin[0]),
            y=int(self._pixel_clipboard_origin[1]),
        )
        self.doc.layers.append(layer)
        self.doc.active_layer = len(self.doc.layers) - 1
        self.doc.dirty = True
        self.selected_layer_ids = {layer.id}
        self.push_command(LayerInsertCommand("Paste pixels", self.doc.active_layer, copy.deepcopy(layer)))
        self.refresh()

    def delete_selected_pixels(self) -> None:
        if not self._editor_active or self.doc.selection_mask is None:
            self.status_text("Удаление пикселей: сначала создайте выделение")
            return
        layer = self.doc.layer
        if layer.locked:
            self.status_text("Слой заблокирован")
            return
        selection = self.doc.layer_selection_mask(layer)
        if selection is None or not np.any(selection):
            self.status_text("Выделение не пересекает активный слой")
            return

        def erase() -> None:
            coverage = selection.astype(np.float32) / 255.0
            layer.pixels[:, :, 3] = np.clip(
                layer.pixels[:, :, 3].astype(np.float32) * (1.0 - coverage),
                0,
                255,
            ).astype(np.uint8)
            layer.touch_pixels()

        self.run_pixel_delta_command("Delete selected pixels", erase)

    def select_all(self) -> None:
        self.run_selection_command("Select all", self.doc.select_all)

    def invert_selection(self) -> None:
        self.run_selection_command("Invert selection", self.doc.invert_selection)

    def feather_selection(self) -> None:
        radius = simpledialog.askinteger("Feather", "Radius px:", initialvalue=8, minvalue=1, maxvalue=500)
        if radius:
            self.run_selection_command("Feather selection", lambda: self.doc.feather_selection(radius))

    def grow_selection(self) -> None:
        pixels = simpledialog.askinteger("Grow", "Pixels:", initialvalue=8, minvalue=1, maxvalue=500)
        if pixels:
            self.run_selection_command("Grow selection", lambda: self.doc.grow_selection(pixels))

    def shrink_selection(self) -> None:
        pixels = simpledialog.askinteger("Shrink", "Pixels:", initialvalue=8, minvalue=1, maxvalue=500)
        if pixels:
            self.run_selection_command("Shrink selection", lambda: self.doc.shrink_selection(pixels))

    def smooth_selection(self) -> None:
        radius = simpledialog.askinteger("Smooth", "Radius px:", initialvalue=4, minvalue=1, maxvalue=500)
        if radius:
            self.run_selection_command("Smooth selection", lambda: self.doc.smooth_selection(radius))

    def border_selection(self) -> None:
        width = simpledialog.askinteger("Border", "Width px:", initialvalue=8, minvalue=1, maxvalue=500)
        if width:
            self.run_selection_command("Border selection", lambda: self.doc.border_selection(width))

    def refine_selection(self) -> None:
        raw = simpledialog.askstring("Refine selection", "smooth,feather,contrast,shift:", initialvalue="2,2,1.25,0")
        if not raw:
            return
        try:
            parts = [part.strip() for part in raw.split(",")]
            if len(parts) != 4:
                raise ValueError
            smooth = max(0, int(float(parts[0])))
            feather = max(0, int(float(parts[1])))
            contrast = max(0.0, float(parts[2]))
            shift = int(float(parts[3]))
        except ValueError:
            messagebox.showerror("Refine selection", "Use: smooth,feather,contrast,shift")
            return
        self.run_selection_command("Refine selection", lambda: self.doc.refine_selection(smooth, feather, contrast, shift))

    def cleanup_selection_edges(self) -> None:
        if self.doc.selection_mask is None:
            messagebox.showinfo("Умная очистка края", "Сначала создайте выделение.")
            return
        radius = simpledialog.askinteger("Умная очистка края", "Радиус края:", initialvalue=3, minvalue=1, maxvalue=40)
        if radius is None:
            return
        strength = simpledialog.askfloat("Умная очистка края", "Сила 0..1:", initialvalue=0.7, minvalue=0.0, maxvalue=1.0)
        if strength is None:
            return
        self.run_selection_command("Умная очистка края", lambda: self.doc.cleanup_selection_edges(radius, strength))

    def correct_selection_edges(self) -> None:
        if self.doc.selection_mask is None:
            messagebox.showinfo("Коррекция края", "Сначала создайте выделение.")
            return
        radius = simpledialog.askinteger("Коррекция края", "Радиус анализа:", initialvalue=3, minvalue=1, maxvalue=40)
        if radius is None:
            return
        strength = simpledialog.askfloat("Коррекция края", "Сила 0..1:", initialvalue=0.65, minvalue=0.0, maxvalue=1.0)
        if strength is None:
            return
        threshold = simpledialog.askinteger("Коррекция края", "Порог уверенности 0..255:", initialvalue=96, minvalue=0, maxvalue=255)
        if threshold is None:
            return
        self.run_selection_command("Коррекция края по уверенности", lambda: self.doc.correct_selection_edges(radius, strength, threshold))
