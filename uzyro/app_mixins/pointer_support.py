from __future__ import annotations

from ..app_shared import *


class PointerSupportMixin:
    def canvas_to_doc(self, event) -> tuple[int, int]:
        ox, oy = self._canvas_origin
        x = int((self.canvas.canvasx(event.x) - ox) / self.zoom.get())
        y = int((self.canvas.canvasy(event.y) - oy) / self.zoom.get())
        return x, y

    def doc_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        ox, oy = self._canvas_origin
        scale = self.zoom.get()
        return ox + x * scale, oy + y * scale

    def selection_mode_from_event(self, event) -> str:
        shift = bool(event.state & 0x0001)
        ctrl = bool(event.state & 0x0004)
        if shift and ctrl:
            return "intersect"
        if shift:
            return "add"
        if ctrl:
            return "subtract"
        mode = self.selection_mode.get()
        return mode if mode in {"replace", "add", "subtract", "intersect"} else "replace"

    @staticmethod
    def brush_preview_tools() -> set[str]:
        return {"brush", "eraser", "blur_tool", "sharpen_tool", "dodge", "burn", "clone", "healing", "spot_healing", "quick_selection"}

    def pointer_motion(self, event) -> None:
        self._last_pointer_event = event
        point = self.canvas_to_doc(event)
        if hasattr(self, "status_coords"):
            self.status_coords.configure(text=f"{point[0]}, {point[1]}")
        tool = self.tool.get()
        if tool in {"clone", "healing"} and not self._panning and event_alt_down(event):
            self.canvas.configure(cursor="target")
        elif tool in {"clone", "healing"} and not self._panning:
            self.canvas.configure(cursor="crosshair")
        elif tool == "move" and not self._panning:
            if self._move_layer_id is not None:
                self.canvas.configure(cursor="fleur")
            else:
                handle = self.selection_transform_handle_at(point) or self.object_handle_at(point)
                cursor_by_handle = {
                    "nw": "size_nw_se", "se": "size_nw_se", "ne": "size_ne_sw", "sw": "size_ne_sw",
                    "n": "sb_v_double_arrow", "s": "sb_v_double_arrow", "e": "sb_h_double_arrow", "w": "sb_h_double_arrow",
                    "rotate": "exchange",
                }
                if handle:
                    self.canvas.configure(cursor=cursor_by_handle.get(handle, "arrow"))
                else:
                    hit = hit_test_document(self.doc, point, tolerance=max(2, round(5 / max(self.zoom.get(), 0.01))))
                    self.canvas.configure(cursor="fleur" if hit is not None else "arrow")
        if not self._panning:
            self.update_brush_preview(event)
        if self.tool.get() == "polygon_lasso" and self._polygon_points:
            self.draw_polygon_lasso(self.canvas_to_doc(event))

    def pointer_leave(self, _event) -> None:
        self._last_pointer_event = None
        if hasattr(self, "status_coords"):
            self.status_coords.configure(text="")
        self.clear_brush_preview()

    def object_document_bounds(self, layer: Layer | None = None) -> tuple[int, int, int, int] | None:
        layer = layer or self.doc.layer
        if layer.kind == "shape" and layer.shape_data is not None:
            x1, y1, x2, y2 = shape_data_bounds(layer.shape_data) or (0, 0, 1, 1)
            return x1 + layer.x, y1 + layer.y, x2 + layer.x, y2 + layer.y
        if layer.kind == "text" and layer.pixels.size:
            ys, xs = np.where(layer.pixels[:, :, 3] > 8)
            if len(xs):
                return int(xs.min()) + layer.x, int(ys.min()) + layer.y, int(xs.max() + 1) + layer.x, int(ys.max() + 1) + layer.y
        return None

    def layer_render_bounds(self, layer: Layer) -> tuple[int, int, int, int]:
        object_bounds = self.object_document_bounds(layer)
        if object_bounds is not None:
            return object_bounds
        height, width = layer.pixels.shape[:2]
        return layer.x, layer.y, layer.x + width, layer.y + height

    def object_handle_points(self, bounds: tuple[int, int, int, int]) -> dict[str, tuple[float, float]]:
        x1, y1, x2, y2 = bounds
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return {"nw": (x1, y1), "n": (cx, y1), "ne": (x2, y1), "e": (x2, cy), "se": (x2, y2), "s": (cx, y2), "sw": (x1, y2), "w": (x1, cy)}

    def object_handle_at(self, point: tuple[int, int]) -> str | None:
        if self.tool.get() != "move" or self.doc.layer.kind != "shape" or self.doc.layer.id not in self.selected_layer_ids:
            return None
        bounds = self.object_document_bounds()
        if bounds is None:
            return None
        tolerance = max(3.0, 7.0 / max(self.zoom.get(), 0.01))
        for name, (x, y) in self.object_handle_points(bounds).items():
            if abs(point[0] - x) <= tolerance and abs(point[1] - y) <= tolerance:
                return name
        return None

    def update_object_bounds(self) -> None:
        if not hasattr(self, "canvas"):
            return
        layer = self.doc.layer if self.doc.layers else None
        bounds = None if self.tool.get() != "move" or layer is None or layer.id not in self.selected_layer_ids or layer.kind not in {"shape", "text"} else self.object_document_bounds(layer)
        if bounds is None:
            for item_id in self._object_bounds_ids:
                self.canvas.delete(item_id)
            self._object_bounds_ids.clear()
            self.update_path_overlay()
            return
        x1, y1 = self.doc_to_canvas(bounds[0], bounds[1])
        x2, y2 = self.doc_to_canvas(bounds[2], bounds[3])
        points = self.object_handle_points((x1, y1, x2, y2))
        dpi_scale = max(1.0, float(self.tk.call("tk", "scaling")))
        radius = max(3, round(3 * dpi_scale))
        if len(self._object_bounds_ids) != 9:
            for item_id in self._object_bounds_ids:
                self.canvas.delete(item_id)
            outline = self.canvas.create_rectangle(x1, y1, x2, y2, outline=TOKENS.ACCENT, width=1, dash=(4, 3))
            handles = [self.canvas.create_rectangle(0, 0, 0, 0, fill=TOKENS.ACCENT, outline=TOKENS.TEXT_PRIMARY, width=1) for _ in points]
            self._object_bounds_ids = [outline, *handles]
        self.canvas.coords(self._object_bounds_ids[0], x1, y1, x2, y2)
        for item_id, (_name, (x, y)) in zip(self._object_bounds_ids[1:], points.items()):
            self.canvas.coords(item_id, x - radius, y - radius, x + radius, y + radius)
            self.canvas.tag_raise(item_id)
        self.canvas.tag_raise(self._object_bounds_ids[0])
        self.update_path_overlay()

    def select_object_at(self, point: tuple[int, int], add: bool = False, cycle: bool = False) -> Layer | None:
        tolerance = max(2, round(5 / max(self.zoom.get(), 0.01)))
        hits = hit_test_stack(self.doc, point, tolerance)
        hit = hits[0] if hits else None
        if cycle and hits:
            active_id = self.doc.layer.id
            active_index = next((index for index, item in enumerate(hits) if item.layer_id == active_id), -1)
            hit = hits[(active_index + 1) % len(hits)]
        if hit is None:
            if not add:
                self.selected_layer_ids.clear()
                self.refresh_layers()
                self.update_object_bounds()
            return None
        layer = self.doc.layers[hit.layer_index]
        self.doc.active_layer = hit.layer_index
        if add:
            if layer.id in self.selected_layer_ids:
                self.selected_layer_ids.remove(layer.id)
            else:
                self.selected_layer_ids.add(layer.id)
        else:
            self.selected_layer_ids = {layer.id}
        self.refresh_layers()
        self.refresh_properties()
        self.update_object_bounds()
        return layer

    def begin_object_resize(self, handle: str) -> None:
        layer = self.doc.layer
        if layer.kind != "shape" or layer.shape_data is None or layer.locked:
            return
        self._object_resize_handle = handle
        self._object_resize_layer_id = layer.id
        self._object_resize_before = copy.deepcopy(layer.shape_data)
        self._object_resize_rendered_bounds = self.object_document_bounds(layer)
        self._last_object_resize_render = 0.0

    def resize_selected_object_live(self, point: tuple[int, int], state: int) -> None:
        layer = self.doc.get_layer(self._object_resize_layer_id or "")
        if layer is None or layer.shape_data is None or self._object_resize_before is None or self._object_resize_handle is None:
            return
        old_box = shape_data_bounds(self._object_resize_before) or (0, 0, 1, 1)
        document_box = tuple(value + (layer.x if index % 2 == 0 else layer.y) for index, value in enumerate(old_box))
        resized = resize_box_from_handle(
            document_box,
            self._object_resize_handle,
            point,
            keep_proportions=bool(state & 0x0001),
            from_center=event_alt_down(state),
        )
        layer.shape_data = transform_shape_data_to_box(
            self._object_resize_before,
            (resized[0] - layer.x, resized[1] - layer.y, resized[2] - layer.x, resized[3] - layer.y),
        )
        self.doc.dirty = True
        self.update_object_bounds()
        now = time.perf_counter()
        if layer.pixels.shape[0] * layer.pixels.shape[1] > 2_000_000:
            return
        if now - self._last_object_resize_render >= 1 / 30:
            old_bounds = self._object_resize_rendered_bounds or document_box
            render_shape_layer(layer)
            new_bounds = self.object_document_bounds(layer) or resized
            self.request_canvas_refresh(union_rect(old_bounds, new_bounds), layer, "pixels")
            self._object_resize_rendered_bounds = new_bounds
            self._last_object_resize_render = now

    def finish_object_resize(self) -> None:
        layer = self.doc.get_layer(self._object_resize_layer_id or "")
        if layer is not None and layer.shape_data is not None and self._object_resize_before is not None and layer.shape_data != self._object_resize_before:
            old_bounds = self._object_resize_rendered_bounds or self.object_document_bounds(layer)
            render_shape_layer(layer)
            new_bounds = self.object_document_bounds(layer)
            if old_bounds is not None and new_bounds is not None:
                self.request_canvas_refresh(union_rect(old_bounds, new_bounds), layer, "pixels")
            self.push_command(ShapeDataCommand("Изменить размер фигуры", layer.id, self._object_resize_before, copy.deepcopy(layer.shape_data), layer.name, layer.name))
        self._object_resize_handle = None
        self._object_resize_layer_id = None
        self._object_resize_before = None
        self._object_resize_rendered_bounds = None
        self._last_object_resize_render = 0.0
        self.refresh_properties()

    def brush_size_changed(self, *_args) -> None:
        if self._last_pointer_event is not None:
            self.update_brush_preview(self._last_pointer_event)
        self.update_clone_source_marker()
        self.quick_preview_settings_changed()

    def quick_preview_settings_changed(self, *_args) -> None:
        if self.tool.get() == "quick_selection" and self._quick_points:
            self.reset_quick_preview_cache()
            self.update_quick_selection_preview(force=True)

    def selection_mode_changed(self, *_args) -> None:
        if self._last_pointer_event is not None and self.tool.get() in {
            "select",
            "ellipse_select",
            "lasso",
            "magnetic_lasso",
            "polygon_lasso",
            "quick_selection",
            "magic_wand",
            "color_range",
        }:
            self.update_brush_preview(self._last_pointer_event)

    def update_brush_preview(self, event) -> None:
        tool = self.tool.get()
        if tool not in self.brush_preview_tools():
            self.clear_brush_preview()
            return
        point = self.canvas_to_doc(event)
        if point[0] < 0 or point[1] < 0 or point[0] >= self.doc.width or point[1] >= self.doc.height:
            self.clear_brush_preview()
            return
        cx, cy = self.doc_to_canvas(point[0] + 0.5, point[1] + 0.5)
        radius = max(1.0, float(self.brush_size.get()) * float(self.zoom.get()))
        mode = self.selection_mode_from_event(event) if tool == "quick_selection" else "replace"
        color_by_mode = {"replace": "#50e3ff", "add": "#59f28a", "subtract": "#ff6262", "intersect": "#ffd166"}
        label_by_mode = {"replace": "new", "add": "+", "subtract": "-", "intersect": "x"}
        choosing_source = tool in {"clone", "healing"} and event_alt_down(event)
        outline = "#ffb000" if choosing_source else color_by_mode.get(mode, "#50e3ff")
        label = "Источник" if choosing_source else label_by_mode.get(mode, "")
        advanced = getattr(self, "brush_advanced", {}) if tool in {"brush", "eraser"} else {}
        roundness = float(np.clip(advanced.get("roundness", 1.0), 0.01, 1.0))
        angle = math.radians(float(advanced.get("angle", 0.0)))
        def ellipse_coords(current_radius: float) -> list[float]:
            points: list[float] = []
            for step in range(48):
                phase = math.tau * step / 48.0
                px, py = math.cos(phase) * current_radius, math.sin(phase) * current_radius * roundness
                points.extend((cx + px * math.cos(angle) - py * math.sin(angle), cy + px * math.sin(angle) + py * math.cos(angle)))
            return points
        shape_coords = ellipse_coords(radius)
        coords = [cx - radius, cy - radius, cx + radius, cy + radius]
        hardness_radius = max(1.0, radius * float(np.clip(self.hardness.get(), 0.0, 1.0)))
        hardness_coords = [cx - hardness_radius, cy - hardness_radius, cx + hardness_radius, cy + hardness_radius]
        if not self._brush_preview_ids:
            fill_id = self.canvas.create_oval(*coords, outline="", fill=outline, stipple="gray25")
            ring_id = self.canvas.create_oval(*coords, outline=outline, width=2)
            hardness_id = self.canvas.create_oval(*hardness_coords, outline=outline, dash=(2, 3), width=1)
            cross_h = self.canvas.create_line(cx - 6, cy, cx + 6, cy, fill=outline, width=1)
            cross_v = self.canvas.create_line(cx, cy - 6, cx, cy + 6, fill=outline, width=1)
            text_id = self.canvas.create_text(cx, cy + radius + 12, text=label, fill=outline, font=("Segoe UI", 9, "bold"))
            self._brush_preview_ids = [fill_id, ring_id, hardness_id, cross_h, cross_v, text_id]
        else:
            fill_id, ring_id, hardness_id, cross_h, cross_v, text_id = self._brush_preview_ids
            self.canvas.coords(fill_id, *coords)
            self.canvas.coords(ring_id, *coords)
            self.canvas.coords(hardness_id, *hardness_coords)
            self.canvas.coords(cross_h, cx - 6, cy, cx + 6, cy)
            self.canvas.coords(cross_v, cx, cy - 6, cx, cy + 6)
            self.canvas.coords(text_id, cx, cy + radius + 12)
        fill_id, ring_id, hardness_id, cross_h, cross_v, text_id = self._brush_preview_ids
        self.canvas.itemconfigure(fill_id, fill=outline)
        self.canvas.itemconfigure(ring_id, outline=outline)
        self.canvas.itemconfigure(hardness_id, outline=outline, state=tk.NORMAL if tool in self.brush_preview_tools() else tk.HIDDEN)
        self.canvas.itemconfigure(cross_h, fill=outline)
        self.canvas.itemconfigure(cross_v, fill=outline)
        self.canvas.itemconfigure(text_id, text=label, fill=outline, state=tk.NORMAL if tool == "quick_selection" or choosing_source else tk.HIDDEN)
        transformed_tip = tool in {"brush", "eraser"} and (roundness < 0.999 or abs(float(advanced.get("angle", 0.0))) > 0.01)
        tip_id = getattr(self, "_brush_tip_shape_id", None)
        if transformed_tip:
            if tip_id is None:
                tip_id = self.canvas.create_polygon(*shape_coords, fill="", outline=outline, dash=(3, 2), width=1, smooth=True)
                self._brush_tip_shape_id = tip_id
            else:
                self.canvas.coords(tip_id, *shape_coords)
                self.canvas.itemconfigure(tip_id, outline=outline, state=tk.NORMAL)
            self.canvas.tag_raise(tip_id)
        elif tip_id is not None:
            self.canvas.itemconfigure(tip_id, state=tk.HIDDEN)
        for item_id in self._brush_preview_ids:
            self.canvas.tag_raise(item_id)
        self.update_clone_overlay(point)

    def clear_brush_preview(self) -> None:
        for item_id in self._brush_preview_ids:
            self.canvas.delete(item_id)
        self._brush_preview_ids.clear()
        if getattr(self, "_brush_tip_shape_id", None) is not None:
            self.canvas.delete(self._brush_tip_shape_id)
            self._brush_tip_shape_id = None
        self.clear_clone_overlay()

    def update_quick_selection_preview(self, force: bool = False) -> None:
        if self.tool.get() != "quick_selection" or not self._quick_points:
            self.clear_quick_selection_preview()
            return
        now = time.perf_counter()
        if not force and now - self._last_quick_preview_time < 0.075:
            return
        self._last_quick_preview_time = now
        new_points = self._quick_points[self._quick_preview_processed :]
        if new_points:
            with self.render_engine.profiler.measure("selection.quick_preview"):
                partial = self.doc._quick_selection_mask(
                    self.doc.layer,
                    new_points,
                    max(2, int(self.brush_size.get())),
                    int(self.tolerance.get()),
                    int(self.quick_smooth.get()),
                    int(self.quick_edge_radius.get()),
                    float(self.quick_edge_strength.get()),
                )
            self._quick_preview_mask = partial if self._quick_preview_mask is None else np.maximum(self._quick_preview_mask, partial)
            self._quick_preview_processed = len(self._quick_points)
        mask = self._quick_preview_mask
        current = self._quick_base_selection
        if mask is not None and current is not None:
            if self._quick_mode == "add":
                mask = np.maximum(current, mask)
            elif self._quick_mode == "subtract":
                mask = np.clip(current.astype(np.float32) * (1.0 - mask.astype(np.float32) / 255.0), 0, 255).astype(np.uint8)
            elif self._quick_mode == "intersect":
                mask = np.minimum(current, mask)
        if mask is None:
            mask = current
        if mask is None or not np.any(mask):
            self.clear_quick_selection_preview()
            return
        bounds = self.mask_bounds(mask)
        if bounds is None:
            self.clear_quick_selection_preview()
            return
        x1, y1, x2, y2 = bounds
        mask_image = Image.fromarray(mask[y1:y2, x1:x2], mode="L")
        scale = self.zoom.get()
        if scale != 1.0:
            mask_image = mask_image.resize(
                (max(1, round((x2 - x1) * scale)), max(1, round((y2 - y1) * scale))),
                Image.Resampling.NEAREST,
            )
        color_by_mode = {
            "replace": (45, 205, 255),
            "add": (55, 225, 120),
            "subtract": (255, 80, 80),
            "intersect": (255, 195, 60),
        }
        color = color_by_mode.get(self._quick_mode, color_by_mode["replace"])
        alpha = mask_image.point(lambda value: 78 if value else 0)
        overlay = Image.new("RGBA", mask_image.size, (*color, 0))
        overlay.putalpha(alpha)
        self._quick_preview_image = ImageTk.PhotoImage(overlay)
        ox, oy = self._canvas_origin
        preview_x, preview_y = ox + round(x1 * scale), oy + round(y1 * scale)
        if self._quick_preview_id is None:
            self._quick_preview_id = self.canvas.create_image(preview_x, preview_y, image=self._quick_preview_image, anchor=tk.NW)
        else:
            self.canvas.itemconfigure(self._quick_preview_id, image=self._quick_preview_image)
            self.canvas.coords(self._quick_preview_id, preview_x, preview_y)
        self.canvas.tag_raise(self._quick_preview_id)
        for item_id in self._brush_preview_ids:
            self.canvas.tag_raise(item_id)

    def clear_quick_selection_preview(self) -> None:
        if self._quick_preview_id is not None:
            self.canvas.delete(self._quick_preview_id)
            self._quick_preview_id = None
        self._quick_preview_image = None
        self.reset_quick_preview_cache()

    def reset_quick_preview_cache(self) -> None:
        self._quick_preview_mask = None
        self._quick_preview_processed = 0

    def space_down(self, _event) -> None:
        self._space_down = True
        self.canvas.configure(cursor="fleur")

    def space_up(self, _event) -> None:
        self._space_down = False
        if not self._panning:
            self.canvas.configure(cursor="")

    def pan_down(self, event) -> None:
        self._panning = True
        self.canvas.scan_mark(event.x, event.y)
        self.canvas.configure(cursor="fleur")

    def pan_drag(self, event) -> None:
        if self._panning:
            self.canvas.scan_dragto(event.x, event.y, gain=1)
            if self._last_pointer_event is not None:
                self.update_brush_preview(self._last_pointer_event)

    def pan_up(self, _event) -> None:
        self._panning = False
        self.canvas.configure(cursor="fleur" if self._space_down else "")
