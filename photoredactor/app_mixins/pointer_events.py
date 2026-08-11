from __future__ import annotations

from ..app_shared import *


class PointerEventsMixin:
    def pointer_down(self, event) -> None:
        if self.tool.get() == "hand" or self._space_down:
            self.pan_down(event)
            return
        self.canvas.focus_set()
        point = self.canvas_to_doc(event)
        self.drag_start = point
        self.last_point = point
        tool = self.tool.get()
        if tool == "crop" and self._crop_box is not None:
            handle = self.crop_handle_at(point)
            if handle is not None:
                self._crop_drag_handle = handle
                self._crop_drag_origin_box = self._crop_box
                self.status_text("Перетащите маркер кадрирования")
            else:
                x1, y1, x2, y2 = self._crop_box
                if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
                    self.drag_start = None
                    return
        if tool.endswith("_shape"):
            self._shape_drag_options = self.current_shape_options(tool)
            self.clear_selection_overlay()
        if tool in ["brush", "eraser", "blur_tool", "sharpen_tool", "dodge", "burn", "clone", "healing", "spot_healing"]:
            if tool in ["clone", "healing"]:
                if event.state & 0x0008:
                    self.set_clone_source(point)
                    return
                if self._source_anchor.point is None:
                    self.status_text("Сначала задайте источник: Alt + левый клик")
                    self.drag_start = None
                    return
                self._source_anchor.aligned = bool(self.clone_aligned.get())
                self._source_anchor.sampling = self.clone_sampling.get()
                self._source_anchor.begin_stroke(point)
                self._clone_anchor_target = self._source_anchor.stroke_target
                self._clone_anchor_source = self._source_anchor.stroke_source
                self.prepare_clone_sample()
            kind = "mask" if tool in ["brush", "eraser"] and self.paint_target.get() == "mask" else "pixels"
            pressure = pointer_pressure(event)
            self.begin_stroke(kind, point, pressure)
            self.paint_at(point, pressure)
        elif tool == "fill":
            self.run_pixel_delta_command("Fill", lambda: flood_fill(self.doc.layer, point[0], point[1], self.foreground, int(self.tolerance.get()), self.doc.layer_selection_mask(self.doc.layer)))
        elif tool == "magic_wand":
            mode = self.selection_mode_from_event(event)
            contiguous = bool(self.magic_contiguous.get())
            self.run_selection_command("Magic wand selection", lambda: self.doc.magic_wand_selection(self.doc.layer, point[0], point[1], int(self.tolerance.get()), mode, contiguous))
        elif tool == "color_range":
            mode = self.selection_mode_from_event(event)
            lx, ly = point[0] - self.doc.layer.x, point[1] - self.doc.layer.y
            if 0 <= lx < self.doc.layer.pixels.shape[1] and 0 <= ly < self.doc.layer.pixels.shape[0]:
                sample = tuple(int(value) for value in self.doc.layer.pixels[ly, lx])
                self.color_range_sample_hex.set(self.color_hex(sample).upper())
                if hasattr(self, "tool_options_panel"):
                    self.tool_options_panel.render()
            self.run_selection_command("Color range selection", lambda: self.doc.color_range_selection(self.doc.layer, point[0], point[1], int(self.tolerance.get()), mode))
            self.status_text(f"Диапазон цвета {self.color_range_sample_hex.get()}, допуск {int(self.tolerance.get())}")
        elif tool == "quick_selection":
            self._quick_points = [point]
            self._quick_mode = self.selection_mode_from_event(event)
            self._quick_base_selection = None if self.doc.selection_mask is None else self.doc.selection_mask.copy()
            self.reset_quick_preview_cache()
            self.update_quick_selection_preview(force=True)
            self.status_text("Кисть быстрого выделения")
        elif tool == "patch":
            self.begin_patch_drag(point)
        elif tool == "eyedropper":
            self.pick_color_from_document(point)
        elif tool == "text":
            if self._text_editor is not None:
                self.finish_text_edit()
            self.status_text("Клик создает строку текста, перетаскивание создает текстовый блок")
        elif tool == "move":
            handle = self.object_handle_at(point)
            if handle is not None:
                self.begin_object_resize(handle)
                return
            layer = self.select_object_at(point, add=bool(event.state & 0x0001)) if self.auto_select.get() else self.doc.layer
            if layer is None and self.doc.layers and self.doc.layer.kind not in {"shape", "text", "adjustment"}:
                candidate = self.doc.layer
                if layer_contains_point(candidate, point, 0):
                    layer = candidate
            if layer is None or not layer_contains_point(layer, point, max(2, round(5 / max(self.zoom.get(), 0.01)))):
                self.drag_start = None
                self.status_text("На этом месте нет редактируемого объекта")
                return
            if layer.locked:
                self.status_text("Объект выбран, но слой заблокирован")
                self.drag_start = None
                return
            self._move_layer_id = layer.id
            self._move_start = (layer.x, layer.y)
            self._move_start_mask = None if layer.mask is None else layer.mask.copy()
            self._move_last_bounds = self.layer_render_bounds(layer)
        elif tool == "polygon_lasso":
            self._polygon_points.append(point)
            self.draw_polygon_lasso()
        elif tool == "lasso":
            self._lasso_points = [point]
        elif tool == "magnetic_lasso":
            self._magnetic_edges = self.doc.magnetic_edge_map(self.render_engine.render(self.doc, False))
            snapped = self.doc.snap_point_to_edge(point, self._magnetic_edges, max(8, int(self.tolerance.get())))
            self._lasso_points = [snapped]
            self.status_text(f"Магнитное лассо: {snapped[0]}, {snapped[1]}")

    def pointer_drag(self, event) -> None:
        self._last_pointer_event = event
        self.update_brush_preview(event)
        if self._panning:
            self.pan_drag(event)
            return
        point = self.canvas_to_doc(event)
        tool = self.tool.get()
        if tool in ["brush", "eraser", "blur_tool", "sharpen_tool", "dodge", "burn", "clone", "healing", "spot_healing"]:
            self.paint_line(self.last_point or point, point, pointer_pressure(event))
            self.last_point = point
        elif tool == "move" and self._object_resize_handle is not None:
            self.resize_selected_object_live(point, event.state)
        elif tool == "move" and self.drag_start:
            dx, dy = point[0] - self.drag_start[0], point[1] - self.drag_start[1]
            if dx or dy:
                layer = self.doc.get_layer(self._move_layer_id or "")
                if layer is not None:
                    old_bounds = self._move_last_bounds or self.layer_render_bounds(layer)
                    self.doc.move_active_layer(dx, dy)
                    new_bounds = self.layer_render_bounds(layer)
                    dirty = union_rect(old_bounds, new_bounds)
                    self._move_last_bounds = new_bounds
                    self.drag_start = point
                    self.request_canvas_refresh(dirty, layer, "transform")
                    self.update_object_bounds()
        elif tool in ["select", "ellipse_select", "crop", "gradient", "text", "rect_shape", "ellipse_shape", "line_shape", "bezier_shape", "polygon_shape", "star_shape", "custom_shape"]:
            self.draw_selection(self.drag_start, point, event.state)
            if tool == "gradient" and self.drag_start:
                self.update_gradient_preview(self.drag_start, point)
        elif tool == "lasso" and self.drag_start:
            self._lasso_points.append(point)
            self.draw_lasso()
        elif tool == "magnetic_lasso" and self.drag_start:
            snapped = self.magnetic_lasso_point(point)
            if not self._lasso_points or (snapped[0] - self._lasso_points[-1][0]) ** 2 + (snapped[1] - self._lasso_points[-1][1]) ** 2 >= 4:
                self._lasso_points.append(snapped)
                self.draw_lasso()
        elif tool == "quick_selection" and self.drag_start:
            spacing = max(1, int(self.brush_size.get()) // 2)
            previous = self._quick_points[-1]
            if (point[0] - previous[0]) ** 2 + (point[1] - previous[1]) ** 2 >= spacing ** 2:
                self._quick_points.append(point)
                self.update_quick_selection_preview()
            self.status_text(f"Точек быстрого выделения: {len(self._quick_points)}")
        elif tool == "patch" and self.drag_start:
            self.draw_patch_preview(point)

    def pointer_up(self, event) -> None:
        if self._panning:
            self.pan_up(event)
            return
        point = self.canvas_to_doc(event)
        tool = self.tool.get()
        if tool in ["brush", "eraser", "blur_tool", "sharpen_tool", "dodge", "burn", "clone", "healing", "spot_healing"]:
            self.end_stroke(f"{tool.title()} stroke")
            self._source_anchor.end_stroke()
            self._clone_anchor_target = None
            self._clone_anchor_source = None
            self._clone_sample_pixels = None
        elif tool == "move":
            if self._object_resize_handle is not None:
                self.finish_object_resize()
            else:
                self.end_move_layer()
        elif tool == "gradient" and self.drag_start:
            if self.gradient_mode.get() == "Объект":
                self.create_gradient_object(self.drag_start, point)
                self.refresh()
            else:
                kind = self.current_gradient_kind()
                stops = self.current_gradient_stops()
                self.run_pixel_delta_command(
                    "Gradient",
                    lambda: apply_gradient(
                        self.doc.layer,
                        (*self.drag_start, *point),
                        self.foreground,
                        self.background,
                        self.doc.layer_selection_mask(self.doc.layer),
                        kind,
                        stops,
                    ),
                )
            self.clear_gradient_preview()
        elif tool == "text" and self.drag_start:
            self.begin_text_editor(self.drag_start, point)
            self.clear_crop_overlay()
        elif tool in ["rect_shape", "ellipse_shape", "line_shape", "bezier_shape", "polygon_shape", "star_shape", "custom_shape"] and self.drag_start:
            geometry = self.shape_geometry_for_drag(tool, self.drag_start, point, event.state)
            if shape_drag_is_meaningful(geometry):
                self.create_shape_from_drag(tool, geometry)
                self.refresh()
            else:
                self.update_selection_overlay()
        elif tool in ["select", "ellipse_select", "crop"] and self.drag_start:
            self.selection_box = (*self.drag_start, *point)
            if tool == "select":
                mode = self.selection_mode_from_event(event)
                feather = int(self.selection_feather.get())
                self.run_selection_command("Rectangular selection", lambda: self.doc.set_rect_selection(self.selection_box, mode, feather))
            elif tool == "ellipse_select":
                mode = self.selection_mode_from_event(event)
                feather = int(self.selection_feather.get())
                antialias = bool(self.selection_antialias.get())
                self.run_selection_command("Elliptical selection", lambda: self.doc.set_ellipse_selection(self.selection_box, mode, feather, antialias))
            elif tool == "crop":
                if self._crop_drag_handle is None:
                    self._crop_box = self.crop_box_for_drag(self.drag_start, point)
                self.draw_crop_overlay(self._crop_box)
                self.status_text("Кадрирование готово: Enter или двойной клик применяет, Escape отменяет")
            self.draw_selection(self.drag_start, point, event.state)
        elif tool == "lasso" and len(self._lasso_points) >= 3:
            mode = self.selection_mode_from_event(event)
            points = list(self._lasso_points)
            feather = int(self.selection_feather.get())
            antialias = bool(self.selection_antialias.get())
            self.run_selection_command("Lasso selection", lambda: self.doc.set_polygon_selection(points, mode, feather, antialias))
            self.clear_lasso_overlay()
        elif tool == "magnetic_lasso":
            if self.drag_start:
                self._lasso_points.append(self.magnetic_lasso_point(point))
            if len(self._lasso_points) >= 3:
                mode = self.selection_mode_from_event(event)
                points = list(self._lasso_points)
                feather = int(self.selection_feather.get())
                antialias = bool(self.selection_antialias.get())
                self.run_selection_command("Magnetic lasso selection", lambda: self.doc.set_polygon_selection(points, mode, feather, antialias))
            self.clear_lasso_overlay()
            self._magnetic_edges = None
        elif tool == "quick_selection" and self._quick_points:
            points = list(self._quick_points)
            mode = self._quick_mode
            radius = max(2, int(self.brush_size.get()))
            tolerance = int(self.tolerance.get())
            smooth = int(self.quick_smooth.get())
            edge_radius = int(self.quick_edge_radius.get())
            edge_strength = float(self.quick_edge_strength.get())
            self._quick_points.clear()
            self.clear_quick_selection_preview()
            self.run_selection_command(
                "Quick selection",
                lambda: self.doc.quick_selection_brush(
                    self.doc.layer,
                    points,
                    radius,
                    tolerance,
                    mode,
                    smooth,
                    edge_radius,
                    edge_strength,
                ),
            )
        elif tool == "patch" and self.drag_start:
            self.finish_patch_drag(point)
        if tool != "crop":
            self.clear_drag_preview()
        self.drag_start = None
        self.last_point = None
        self._shape_drag_options = None
        self._crop_drag_handle = None
        self._crop_drag_origin_box = None

    def pointer_double_click(self, event) -> None:
        if self.tool.get() == "move":
            point = self.canvas_to_doc(event)
            layer = self.select_object_at(point)
            if layer is not None and layer.kind == "text":
                self.edit_active_text_on_canvas()
            return
        if self.tool.get() == "crop" and self._crop_box is not None:
            point = self.canvas_to_doc(event)
            x1, y1, x2, y2 = self._crop_box
            if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
                self.apply_crop_overlay()
            return
        if self.tool.get() == "polygon_lasso" and len(self._polygon_points) >= 3:
            mode = self.selection_mode_from_event(event)
            points = list(self._polygon_points)
            feather = int(self.selection_feather.get())
            antialias = bool(self.selection_antialias.get())
            self.run_selection_command("Polygon lasso selection", lambda: self.doc.set_polygon_selection(points, mode, feather, antialias))
            self.clear_lasso_overlay()
