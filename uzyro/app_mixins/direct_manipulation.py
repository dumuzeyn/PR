from __future__ import annotations

from ..app_shared import *


class DirectManipulationMixin:
    """Fast on-canvas move, resize and rotation controls."""

    def _shape_base_bounds(self, layer: Layer) -> tuple[float, float, float, float] | None:
        data = layer.shape_data
        if data is None:
            return None
        if str(data.get("shape", "rectangle")).lower() == "boolean":
            bounds = shape_data_bounds(data)
        else:
            try:
                values = [float(value) for value in data.get("box", [0, 0, 1, 1])]
                bounds = min(values[0], values[2]), min(values[1], values[3]), max(values[0], values[2]), max(values[1], values[3])
            except (TypeError, ValueError, IndexError):
                bounds = shape_data_bounds(data)
        if bounds is None:
            return None
        return bounds[0] + layer.x, bounds[1] + layer.y, bounds[2] + layer.x, bounds[3] + layer.y

    def object_document_bounds(self, layer: Layer | None = None) -> tuple[int, int, int, int] | None:
        layer = layer or self.doc.layer
        if layer.kind == "shape" and layer.shape_data is not None:
            bounds = shape_data_bounds(layer.shape_data) or (0, 0, 1, 1)
            return bounds[0] + layer.x, bounds[1] + layer.y, bounds[2] + layer.x, bounds[3] + layer.y
        if layer.kind == "text" and layer.pixels.size:
            cache = getattr(self, "_text_bounds_cache", {})
            key = layer.id, layer.pixels_revision, layer.pixels.shape
            local = cache.get(layer.id)
            if local is None or local[0] != key:
                ys, xs = np.where(layer.pixels[:, :, 3] > 8)
                bounds = None if not len(xs) else (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
                cache[layer.id] = (key, bounds)
                self._text_bounds_cache = cache
            else:
                bounds = local[1]
            if bounds is not None:
                return bounds[0] + layer.x, bounds[1] + layer.y, bounds[2] + layer.x, bounds[3] + layer.y
        return None

    def selected_object_layers(self) -> list[Layer]:
        selected = set(getattr(self, "selected_layer_ids", set()))
        return [
            layer for layer in self.doc.layers
            if layer.id in selected and layer.visible and layer.kind in {"shape", "text"}
        ]

    def selected_object_bounds(self) -> tuple[int, int, int, int] | None:
        bounds: tuple[int, int, int, int] | None = None
        for layer in self.selected_object_layers():
            geometry = self._object_handle_geometry(layer)
            if geometry is None:
                continue
            corners = [geometry[0][name] for name in ("nw", "ne", "se", "sw")]
            current = (
                math.floor(min(point[0] for point in corners)),
                math.floor(min(point[1] for point in corners)),
                math.ceil(max(point[0] for point in corners)),
                math.ceil(max(point[1] for point in corners)),
            )
            bounds = current if bounds is None else union_rect(bounds, current)
        return bounds

    def _selected_object_geometry(self) -> tuple[dict[str, tuple[float, float]], float] | None:
        layers = self.selected_object_layers()
        if len(layers) == 1:
            return self._object_handle_geometry(layers[0])
        bounds = self.selected_object_bounds()
        if bounds is None:
            return None
        return self._rotated_handle_points(bounds, 0.0, 28.0 / max(self.zoom.get(), 0.01)), 0.0

    @staticmethod
    def _rotated_handle_points(bounds: tuple[float, float, float, float], angle: float, rotate_gap: float) -> dict[str, tuple[float, float]]:
        x1, y1, x2, y2 = bounds
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        points = {
            "nw": (x1, y1), "n": (cx, y1), "ne": (x2, y1), "e": (x2, cy),
            "se": (x2, y2), "s": (cx, y2), "sw": (x1, y2), "w": (x1, cy),
            "rotate": (cx, y1 - rotate_gap),
        }
        if abs(angle) < 1e-8:
            return points
        radians = math.radians(angle)
        cosine, sine = math.cos(radians), math.sin(radians)
        return {
            name: (cx + (x - cx) * cosine - (y - cy) * sine, cy + (x - cx) * sine + (y - cy) * cosine)
            for name, (x, y) in points.items()
        }

    def _object_handle_geometry(self, layer: Layer | None = None) -> tuple[dict[str, tuple[float, float]], float] | None:
        layer = layer or self.doc.layer
        if layer.kind == "shape":
            bounds = self._shape_base_bounds(layer)
            angle = float((layer.shape_data or {}).get("rotation", 0.0))
        else:
            bounds = self.object_document_bounds(layer)
            angle = 0.0
        if bounds is None:
            return None
        gap = 28.0 / max(self.zoom.get(), 0.01)
        return self._rotated_handle_points(bounds, angle, gap), angle

    def object_handle_points(self, bounds: tuple[int, int, int, int]) -> dict[str, tuple[float, float]]:
        return self._rotated_handle_points(bounds, 0.0, 28.0 / max(self.zoom.get(), 0.01))

    def object_handle_at(self, point: tuple[int, int]) -> str | None:
        if self.tool.get() != "move" or not self.selected_object_layers():
            return None
        geometry = self._selected_object_geometry()
        if geometry is None:
            return None
        tolerance = max(3.0, 8.0 / max(self.zoom.get(), 0.01))
        for name, (x, y) in geometry[0].items():
            if math.hypot(point[0] - x, point[1] - y) <= tolerance:
                return name
        return None

    def _selection_handle_bounds(self) -> tuple[int, int, int, int] | None:
        mask = self.doc.selection_mask
        if mask is None:
            self._selection_bounds_cache = (None, None)
            return None
        active_bounds = getattr(self, "_selection_transform_bounds", None)
        if getattr(self, "_selection_transform_handle", None) is not None and active_bounds is not None:
            return active_bounds
        cached_key, cached_bounds = getattr(self, "_selection_bounds_cache", (None, None))
        key = id(mask)
        if cached_key != key:
            cached_bounds = self.doc.selection_bounds()
            self._selection_bounds_cache = (key, cached_bounds)
        return cached_bounds

    def selection_transform_handle_at(self, point: tuple[int, int]) -> str | None:
        if self.tool.get() != "move" or self.doc.selection_mask is None:
            return None
        bounds = self._selection_handle_bounds()
        if bounds is None:
            return None
        gap = 28.0 / max(self.zoom.get(), 0.01)
        rotate = self._rotated_handle_points(bounds, float(getattr(self, "_selection_transform_angle", 0.0)), gap)["rotate"]
        tolerance = max(4.0, 9.0 / max(self.zoom.get(), 0.01))
        return "rotate" if math.hypot(point[0] - rotate[0], point[1] - rotate[1]) <= tolerance else None

    def _clear_transform_bounds(self) -> None:
        for item_id in self._object_bounds_ids:
            self.canvas.delete(item_id)
        self._object_bounds_ids.clear()

    def update_object_bounds(self) -> None:
        if not hasattr(self, "canvas"):
            return
        selection_bounds = self._selection_handle_bounds() if self.tool.get() == "move" else None
        layer = self.doc.layer if self.doc.layers else None
        if selection_bounds is not None:
            angle = float(getattr(self, "_selection_transform_angle", 0.0))
            gap = 28.0 / max(self.zoom.get(), 0.01)
            points = self._rotated_handle_points(selection_bounds, angle, gap)
            shown_names = ("rotate",)
        elif self.tool.get() == "move" and layer is not None and self.selected_object_layers():
            geometry = self._selected_object_geometry()
            if geometry is None:
                self._clear_transform_bounds()
                return
            points = geometry[0]
            shown_names = tuple(points)
        else:
            self._clear_transform_bounds()
            self.update_path_overlay()
            return
        corners = [points[name] for name in ("nw", "ne", "se", "sw", "nw")]
        canvas_corners = [self.doc_to_canvas(*point) for point in corners]
        top = self.doc_to_canvas(*points["n"])
        rotate = self.doc_to_canvas(*points["rotate"])
        expected = 2 + len(shown_names)
        if len(self._object_bounds_ids) != expected:
            self._clear_transform_bounds()
            outline = self.canvas.create_line(*(value for pair in canvas_corners for value in pair), fill=TOKENS.ACCENT, width=2)
            connector = self.canvas.create_line(*top, *rotate, fill=TOKENS.ACCENT, width=1)
            handles = [self.canvas.create_oval(0, 0, 0, 0, fill="#ffffff", outline=TOKENS.ACCENT, width=2) for _ in shown_names]
            self._object_bounds_ids = [outline, connector, *handles]
        self.canvas.coords(self._object_bounds_ids[0], *(value for pair in canvas_corners for value in pair))
        self.canvas.coords(self._object_bounds_ids[1], *top, *rotate)
        dpi_scale = max(1.0, float(self.tk.call("tk", "scaling")))
        for item_id in self._object_bounds_ids[:2]:
            self.canvas.tag_raise(item_id)
        for item_id, name in zip(self._object_bounds_ids[2:], shown_names):
            x, y = self.doc_to_canvas(*points[name])
            radius = max(5, round((6 if name == "rotate" else 4) * dpi_scale))
            self.canvas.coords(item_id, x - radius, y - radius, x + radius, y + radius)
            self.canvas.tag_raise(item_id)
        self.update_path_overlay()

    @staticmethod
    def _pointer_angle(point: tuple[int, int], center: tuple[float, float]) -> float:
        return math.degrees(math.atan2(point[1] - center[1], point[0] - center[0]))

    def begin_object_resize(self, handle: str, point: tuple[int, int] | None = None) -> None:
        layer = self.doc.layer
        data = layer.shape_data if layer.kind == "shape" else layer.text_data if layer.kind == "text" else None
        if data is None or layer.locked:
            return
        selected = [item for item in self.selected_object_layers() if not item.locked]
        self._object_resize_group_bounds = self.selected_object_bounds() if len(selected) > 1 else None
        self._object_resize_group_before = {
            item.id: {
                "x": item.x,
                "y": item.y,
                "shape_data": copy.deepcopy(item.shape_data),
                "text_data": copy.deepcopy(item.text_data),
                "bounds": self.object_document_bounds(item),
            }
            for item in selected
        } if len(selected) > 1 else None
        self._object_resize_handle = handle
        self._object_resize_layer_id = layer.id
        self._object_resize_before = copy.deepcopy(data)
        self._object_resize_rendered_bounds = self.object_document_bounds(layer)
        self._last_object_resize_render = 0.0
        if handle == "rotate":
            bounds = self._object_resize_group_bounds or (self._shape_base_bounds(layer) if layer.kind == "shape" else self.object_document_bounds(layer))
            if bounds is not None:
                self._object_rotation_center = ((bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0)
                self._object_rotation_pointer = self._pointer_angle(point or (round(bounds[2]), round(bounds[1])), self._object_rotation_center)
                self._object_rotation_initial = float(data.get("rotation", 0.0))
                self.status_text("Потяните круглую ручку, чтобы повернуть объект; Shift фиксирует шаг 15°")

    def _render_transformed_object(self, layer: Layer, old_bounds: tuple[int, int, int, int] | None) -> None:
        if layer.kind == "shape":
            render_shape_layer(layer)
        else:
            render_text_layer(layer)
        layer.touch_pixels()
        new_bounds = self.object_document_bounds(layer)
        if old_bounds is not None and new_bounds is not None:
            self.request_canvas_refresh(union_rect(old_bounds, new_bounds), layer, "pixels")
        self._object_resize_rendered_bounds = new_bounds

    def resize_selected_object_live(self, point: tuple[int, int], state: int) -> None:
        group_before = getattr(self, "_object_resize_group_before", None)
        group_bounds = getattr(self, "_object_resize_group_bounds", None)
        if group_before and group_bounds is not None:
            self._resize_selected_group_live(point, state, group_before, group_bounds)
            return
        layer = self.doc.get_layer(self._object_resize_layer_id or "")
        before = self._object_resize_before
        handle = self._object_resize_handle
        if layer is None or before is None or handle is None:
            return
        data = layer.shape_data if layer.kind == "shape" else layer.text_data
        if data is None:
            return
        if handle == "rotate":
            center = getattr(self, "_object_rotation_center", point)
            angle = float(getattr(self, "_object_rotation_initial", 0.0)) + self._pointer_angle(point, center) - float(getattr(self, "_object_rotation_pointer", 0.0))
            if state & 0x0001:
                angle = round(angle / 15.0) * 15.0
            data["rotation"] = round(angle, 2)
        elif layer.kind == "shape" and layer.shape_data is not None:
            old_box = shape_data_bounds(before) or (0, 0, 1, 1)
            document_box = tuple(value + (layer.x if index % 2 == 0 else layer.y) for index, value in enumerate(old_box))
            resized = resize_box_from_handle(document_box, handle, point, keep_proportions=bool(state & 0x0001), from_center=event_alt_down(state))
            layer.shape_data = transform_shape_data_to_box(before, (resized[0] - layer.x, resized[1] - layer.y, resized[2] - layer.x, resized[3] - layer.y))
        else:
            return
        self.doc.dirty = True
        self.update_object_bounds()
        now = time.perf_counter()
        if now - self._last_object_resize_render >= 1 / 30 and layer.pixels.shape[0] * layer.pixels.shape[1] <= 2_000_000:
            self._render_transformed_object(layer, self._object_resize_rendered_bounds)
            self._last_object_resize_render = now

    def _resize_selected_group_live(self, point, state, before, group_bounds) -> None:
        handle = self._object_resize_handle
        if handle is None:
            return
        if handle == "rotate":
            center = getattr(self, "_object_rotation_center", ((group_bounds[0] + group_bounds[2]) / 2, (group_bounds[1] + group_bounds[3]) / 2))
            angle = self._pointer_angle(point, center) - float(getattr(self, "_object_rotation_pointer", 0.0))
            if state & 0x0001:
                angle = round(angle / 15.0) * 15.0
            radians = math.radians(angle)
            cosine, sine = math.cos(radians), math.sin(radians)
            for layer_id, initial in before.items():
                layer = self.doc.get_layer(layer_id)
                bounds = initial.get("bounds")
                if layer is None or bounds is None:
                    continue
                object_center = ((bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0)
                dx, dy = object_center[0] - center[0], object_center[1] - center[1]
                rotated = (center[0] + dx * cosine - dy * sine, center[1] + dx * sine + dy * cosine)
                layer.x = int(initial["x"] + round(rotated[0] - object_center[0]))
                layer.y = int(initial["y"] + round(rotated[1] - object_center[1]))
                data = copy.deepcopy(initial["shape_data"] if layer.kind == "shape" else initial["text_data"])
                if data is not None:
                    data["rotation"] = round(float(data.get("rotation", 0.0)) + angle, 2)
                    if layer.kind == "shape":
                        layer.shape_data = data
                    else:
                        layer.text_data = data
        else:
            target = resize_box_from_handle(group_bounds, handle, point, keep_proportions=bool(state & 0x0001), from_center=event_alt_down(state))
            old_width, old_height = max(1, group_bounds[2] - group_bounds[0]), max(1, group_bounds[3] - group_bounds[1])
            scale_x, scale_y = (target[2] - target[0]) / old_width, (target[3] - target[1]) / old_height
            for layer_id, initial in before.items():
                layer = self.doc.get_layer(layer_id)
                bounds = initial.get("bounds")
                if layer is None or bounds is None:
                    continue
                mapped = (
                    target[0] + (bounds[0] - group_bounds[0]) * scale_x,
                    target[1] + (bounds[1] - group_bounds[1]) * scale_y,
                    target[0] + (bounds[2] - group_bounds[0]) * scale_x,
                    target[1] + (bounds[3] - group_bounds[1]) * scale_y,
                )
                if layer.kind == "shape" and initial["shape_data"] is not None:
                    layer.x, layer.y = int(initial["x"]), int(initial["y"])
                    local = (mapped[0] - layer.x, mapped[1] - layer.y, mapped[2] - layer.x, mapped[3] - layer.y)
                    layer.shape_data = transform_shape_data_to_box(initial["shape_data"], local)
                elif layer.kind == "text" and initial["text_data"] is not None:
                    layer.x = int(initial["x"] + round(mapped[0] - bounds[0]))
                    layer.y = int(initial["y"] + round(mapped[1] - bounds[1]))
                    data = copy.deepcopy(initial["text_data"])
                    factor = max(0.05, min(abs(scale_x), abs(scale_y)))
                    data["size"] = max(4, round(float(data.get("size", 48)) * factor))
                    layer.text_data = data
        self.doc.dirty = True
        self.update_object_bounds()
        now = time.perf_counter()
        if now - self._last_object_resize_render >= 1 / 30:
            for layer_id in before:
                layer = self.doc.get_layer(layer_id)
                if layer is not None:
                    self._render_transformed_object(layer, self._object_resize_rendered_bounds)
            self._last_object_resize_render = now

    def finish_object_resize(self) -> None:
        group_before = getattr(self, "_object_resize_group_before", None)
        if group_before:
            after = {}
            before_states = {}
            for layer_id, initial in group_before.items():
                layer = self.doc.get_layer(layer_id)
                if layer is None:
                    continue
                before_states[layer_id] = {key: copy.deepcopy(initial[key]) for key in ("x", "y", "shape_data", "text_data")}
                after[layer_id] = {"x": layer.x, "y": layer.y, "shape_data": copy.deepcopy(layer.shape_data), "text_data": copy.deepcopy(layer.text_data)}
                self._render_transformed_object(layer, self._object_resize_rendered_bounds)
            if after != before_states:
                label = "Повернуть объекты" if self._object_resize_handle == "rotate" else "Изменить размер объектов"
                self.push_command(ObjectStatesCommand(label, before_states, after))
            self._object_resize_group_before = None
            self._object_resize_group_bounds = None
            self._object_resize_handle = None
            self._object_resize_layer_id = None
            self._object_resize_before = None
            self._object_resize_rendered_bounds = None
            self._last_object_resize_render = 0.0
            self.refresh_properties()
            self.update_object_bounds()
            return
        layer = self.doc.get_layer(self._object_resize_layer_id or "")
        before = self._object_resize_before
        handle = self._object_resize_handle
        data = None if layer is None else layer.shape_data if layer.kind == "shape" else layer.text_data
        if layer is not None and data is not None and before is not None and data != before:
            self._render_transformed_object(layer, self._object_resize_rendered_bounds)
            label = "Повернуть объект" if handle == "rotate" else "Изменить размер фигуры"
            command_type = ShapeDataCommand if layer.kind == "shape" else TextDataCommand
            self.push_command(command_type(label, layer.id, before, copy.deepcopy(data), layer.name, layer.name))
        self._object_resize_handle = None
        self._object_resize_layer_id = None
        self._object_resize_before = None
        self._object_resize_rendered_bounds = None
        self._last_object_resize_render = 0.0
        self.refresh_properties()
        self.update_object_bounds()

    def begin_selection_transform(self, handle: str, point: tuple[int, int]) -> None:
        bounds = self._selection_handle_bounds()
        if handle != "rotate" or bounds is None or self.doc.layer.locked:
            return
        self._selection_transform_handle = handle
        self._selection_transform_bounds = bounds
        self._selection_transform_center = ((bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0)
        self._selection_transform_pointer = self._pointer_angle(point, self._selection_transform_center)
        self._selection_transform_angle = 0.0
        self.status_text("Поворот выделенной области; Shift фиксирует шаг 15°")

    def update_selection_transform(self, point: tuple[int, int], state: int) -> None:
        center = getattr(self, "_selection_transform_center", point)
        angle = self._pointer_angle(point, center) - float(getattr(self, "_selection_transform_pointer", 0.0))
        if state & 0x0001:
            angle = round(angle / 15.0) * 15.0
        self._selection_transform_angle = round(angle, 2)
        self.update_object_bounds()
        self.status_text(f"Поворот выделенной области: {self._selection_transform_angle:.1f}°")

    def finish_selection_transform(self) -> None:
        bounds = getattr(self, "_selection_transform_bounds", None)
        angle = float(getattr(self, "_selection_transform_angle", 0.0))
        self._selection_transform_handle = None
        self._selection_transform_angle = 0.0
        if bounds is not None and abs(angle) > 0.01:
            x1, y1, x2, y2 = bounds
            self.run_document_command("Повернуть выделенные пиксели", lambda: self.doc.transform_selected_pixels(x1, y1, x2 - x1, y2 - y1, angle))
            self.refresh()
        else:
            self.update_object_bounds()

    def begin_layer_move_preview(self, point: tuple[int, int]) -> None:
        self._move_pointer_origin = point
        self._move_last_refresh_at = 0.0
        self._move_rendered_bounds = self._move_last_bounds
        self._move_pending_point = None

    def move_selected_layers_live(self, point: tuple[int, int]) -> None:
        self._move_pending_point = point
        if self._move_after_id is None:
            self._move_after_id = self.after_idle(self._flush_layer_move)

    def _flush_layer_move(self) -> None:
        self._move_after_id = None
        point = self._move_pending_point
        self._move_pending_point = None
        if point is None:
            return
        origin = getattr(self, "_move_pointer_origin", self.drag_start or point)
        total_dx, total_dy = point[0] - origin[0], point[1] - origin[1]
        moved = False
        active = self.doc.active_layer
        selected_layers: list[Layer] = []
        for layer_id, start in self._move_group_starts.items():
            layer = self.doc.get_layer(layer_id)
            if layer is None:
                continue
            selected_layers.append(layer)
            wanted = start[0] + total_dx, start[1] + total_dy
            if wanted != (layer.x, layer.y):
                layer.x, layer.y = wanted
                original_mask = self._move_group_masks.get(layer.id)
                if original_mask is not None and not layer.mask_linked:
                    height, width = original_mask.shape[:2]
                    layer.mask = shifted_mask(original_mask, width, height, width, height, -total_dx, -total_dy)
                self.doc.dirty = True
                moved = True
        self.doc.active_layer = active
        if not moved:
            return
        initial = self._move_last_bounds
        new_bounds = None if initial is None else (initial[0] + total_dx, initial[1] + total_dy, initial[2] + total_dx, initial[3] + total_dy)
        self.update_object_bounds()
        now = time.perf_counter()
        if now - float(getattr(self, "_move_last_refresh_at", 0.0)) >= 1 / 30 and selected_layers and new_bounds is not None:
            rendered = getattr(self, "_move_rendered_bounds", initial) or new_bounds
            self.request_canvas_refresh(union_rect(rendered, new_bounds), selected_layers[-1], "transform")
            self._move_rendered_bounds = new_bounds
            self._move_last_refresh_at = now

    def finish_layer_move_preview(self) -> None:
        if self._move_after_id is not None:
            self.after_cancel(self._move_after_id)
            self._move_after_id = None
        if self._move_pending_point is not None:
            self._flush_layer_move()
        rendered = getattr(self, "_move_rendered_bounds", None)
        if self._move_group_starts:
            bounds = [self.layer_render_bounds(layer) for layer_id in self._move_group_starts if (layer := self.doc.get_layer(layer_id)) is not None]
            current = None
            for item in bounds:
                current = item if current is None else union_rect(current, item)
            layer = self.doc.get_layer(self._move_layer_id or "")
            if current is not None and layer is not None and current != rendered:
                self.request_canvas_refresh(union_rect(rendered or current, current), layer, "transform")
        self._move_pointer_origin = None
        self._move_rendered_bounds = None
        self._move_pending_point = None

    def end_move_selection(self) -> None:
        bounds = self._move_selection_bounds
        dx, dy = self._move_selection_delta
        self.restore_move_selection_source(refresh=bounds is None or (dx == 0 and dy == 0))
        self._move_selection_start = None
        self._move_selection_bounds = None
        self._move_selection_delta = (0, 0)
        self.clear_move_selection_preview()
        self.clear_drag_preview()
        if bounds is None or (dx == 0 and dy == 0):
            self.update_object_bounds()
            return
        x1, y1, x2, y2 = bounds
        self.run_document_command("Переместить выделенные пиксели", lambda: self.doc.transform_selected_pixels(x1 + dx, y1 + dy, x2 - x1, y2 - y1))
        self.refresh()
