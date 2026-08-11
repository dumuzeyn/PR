from __future__ import annotations

from ..app_shared import *


class PaintingMixin:
    def begin_stroke(self, kind: str = "pixels") -> None:
        self._edit_generation += 1
        self._stroke_layer_id = self.doc.layer.id
        self._stroke_kind = kind
        if kind == "mask" and self.doc.layer.mask is None:
            self.doc.layer.mask = np.full(self.doc.layer.pixels.shape[:2], 255, dtype=np.uint8)
            self.doc.layer.mask_enabled = True
        self._stroke_rect = None
        self._stroke_before = None
        self._stroke_tiles = {}
        self._stroke_selection_mask = self.doc.layer_selection_mask(self.doc.layer)
        tool = self.tool.get()
        if kind == "pixels" and tool in {"blur_tool", "sharpen_tool", "dodge", "burn"}:
            mode = "blur" if tool == "blur_tool" else "sharpen" if tool == "sharpen_tool" else tool
            strength = float(self.exposure.get()) if tool in {"dodge", "burn"} else float(self.retouch_strength.get())
            self._retouch_stroke = RetouchStroke(
                self.doc.layer,
                mode,
                int(self.brush_size.get()),
                float(self.hardness.get()),
                strength,
                self.tonal_range.get(),
                self._stroke_selection_mask,
            )
        else:
            self._retouch_stroke = None

    def brush_local_rect(self, point: tuple[int, int]) -> tuple[int, int, int, int] | None:
        layer = self.doc.layer
        radius = int(self.brush_size.get())
        if self.tool.get() == "spot_healing":
            radius *= 2
        lx, ly = point[0] - layer.x, point[1] - layer.y
        if lx < -radius or ly < -radius or lx >= layer.pixels.shape[1] + radius or ly >= layer.pixels.shape[0] + radius:
            return None
        return (
            max(0, lx - radius),
            max(0, ly - radius),
            min(layer.pixels.shape[1], lx + radius + 1),
            min(layer.pixels.shape[0], ly + radius + 1),
        )

    def capture_stroke_before(self, rect: tuple[int, int, int, int] | None) -> None:
        if rect is None:
            return
        if self._retouch_stroke is not None:
            return
        layer = self.doc.layer
        target = layer.mask if self._stroke_kind == "mask" else layer.pixels
        if target is None:
            return
        x1, y1, x2, y2 = rect
        tile_size = 128
        self._stroke_rect = union_rect(self._stroke_rect, rect)
        for ty in range(y1 // tile_size, (y2 - 1) // tile_size + 1):
            for tx in range(x1 // tile_size, (x2 - 1) // tile_size + 1):
                key = tx, ty
                if key in self._stroke_tiles:
                    continue
                px1, py1 = tx * tile_size, ty * tile_size
                px2, py2 = min(target.shape[1], px1 + tile_size), min(target.shape[0], py1 + tile_size)
                tile_rect = px1, py1, px2, py2
                self._stroke_tiles[key] = tile_rect, target[py1:py2, px1:px2].copy()

    def end_stroke(self, label: str) -> None:
        if self._retouch_stroke is not None:
            self._stroke_tiles = self._retouch_stroke.before_tiles
        if self._stroke_layer_id and self._stroke_tiles:
            layer = self.doc.get_layer(self._stroke_layer_id)
            if layer is not None:
                target = layer.mask if self._stroke_kind == "mask" else layer.pixels
                patches: list[TilePatch] = []
                precision_before: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
                if self._stroke_kind == "pixels" and layer.working_pixels is not None:
                    original_working = layer.working_rgba()
                    precision_before = [
                        (rect, original_working[rect[1]:rect[3], rect[0]:rect[2]].copy())
                        for rect, _before in self._stroke_tiles.values()
                    ]
                if target is not None:
                    for rect, before in self._stroke_tiles.values():
                        x1, y1, x2, y2 = rect
                        after = target[y1:y2, x1:x2].copy()
                        if not np.array_equal(before, after):
                            patches.append(TilePatch(rect, before, after))
                if self._stroke_kind == "mask" and layer.mask is not None:
                    self.push_command(MaskTilePatchCommand(label, self._stroke_layer_id, patches))
                elif self._stroke_kind == "pixels":
                    layer.touch_pixels()
                    precision_patches = None
                    if precision_before:
                        edited_working = layer.working_rgba()
                        precision_patches = [
                            TilePatch(rect, before, edited_working[rect[1]:rect[3], rect[0]:rect[2]].copy())
                            for rect, before in precision_before
                        ]
                    self.push_command(PixelTilePatchCommand(label, self._stroke_layer_id, patches, precision_patches))
        self._stroke_layer_id = None
        self._stroke_kind = "pixels"
        self._stroke_rect = None
        self._stroke_before = None
        self._stroke_tiles = {}
        self._stroke_selection_mask = None
        self._retouch_stroke = None
        if self._editor_active:
            self.refresh_layer_previews()

    def end_move_layer(self) -> None:
        if self._move_layer_id and self._move_start:
            layer = self.doc.get_layer(self._move_layer_id)
            if layer is not None:
                end = (layer.x, layer.y)
                if end != self._move_start:
                    after_mask = None if layer.mask is None else layer.mask.copy()
                    self.push_command(LayerMoveCommand("Move layer", self._move_layer_id, self._move_start, end, self._move_start_mask, after_mask))
        self._move_layer_id = None
        self._move_start = None
        self._move_start_mask = None
        self._move_last_bounds = None

    def paint_at(self, point: tuple[int, int]) -> None:
        self.capture_stroke_before(self.brush_local_rect(point))
        tool = self.tool.get()
        selection_mask = self._stroke_selection_mask
        changed = None
        if tool == "spot_healing":
            changed = spot_heal(
                self.doc.layer,
                point[0],
                point[1],
                int(self.brush_size.get()),
                float(self.retouch_strength.get()),
                selection_mask,
                float(self.hardness.get()),
            )
        elif tool in ["clone", "healing"]:
            source = self.clone_source_for_point(point)
            if source is not None:
                amount = float(self.opacity.get()) if tool == "clone" else float(self.retouch_strength.get())
                changed = clone_or_heal(
                    self.doc.layer,
                    source[0],
                    source[1],
                    point[0],
                    point[1],
                    int(self.brush_size.get()),
                    amount,
                    tool == "healing",
                    selection_mask,
                    float(self.hardness.get()),
                    self._clone_sample_pixels,
                    self._clone_sample_origin,
                )
        elif self._stroke_kind == "mask":
            changed = draw_mask_brush(self.doc.layer, point[0], point[1], int(self.brush_size.get()), 0 if tool == "eraser" else 255, float(self.opacity.get()), selection_mask)
        elif tool in ["blur_tool", "sharpen_tool", "dodge", "burn"]:
            if self._retouch_stroke is not None:
                changed = self._retouch_stroke.dab(point[0], point[1])
        else:
            changed = draw_brush(
                self.doc.layer,
                point[0],
                point[1],
                int(self.brush_size.get()),
                self.foreground,
                float(self.opacity.get()),
                tool == "eraser",
                selection_mask,
            )
        self.doc.dirty = True
        if changed is not None:
            rect = self.local_to_document_rect(changed, self.doc.layer)
            self.request_canvas_refresh(rect, self.doc.layer, self._stroke_kind)
        elif tool in {"clone", "healing"}:
            self.status_text("Источник и цель не пересекают доступные пиксели")

    def paint_line(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        radius = max(1, int(self.brush_size.get()))
        distance = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
        tool = self.tool.get()
        spacing = radius * (0.9 if tool == "spot_healing" else 0.45)
        steps = max(1, int(np.ceil(distance / max(1.0, spacing))))
        selection_mask = self._stroke_selection_mask
        opacity = float(self.opacity.get())
        changed_rect = None
        for i in range(1, steps + 1):
            t = i / steps
            x = round(start[0] * (1 - t) + end[0] * t)
            y = round(start[1] * (1 - t) + end[1] * t)
            self.capture_stroke_before(self.brush_local_rect((x, y)))
            changed = None
            if tool == "spot_healing":
                changed = spot_heal(self.doc.layer, x, y, radius, float(self.retouch_strength.get()), selection_mask, float(self.hardness.get()))
            elif tool in ["clone", "healing"]:
                source = self.clone_source_for_point((x, y))
                if source is not None:
                    amount = opacity if tool == "clone" else float(self.retouch_strength.get())
                    changed = clone_or_heal(
                        self.doc.layer, source[0], source[1], x, y, radius, amount,
                        tool == "healing", selection_mask, float(self.hardness.get()),
                        self._clone_sample_pixels, self._clone_sample_origin,
                    )
            elif self._stroke_kind == "mask":
                changed = draw_mask_brush(self.doc.layer, x, y, radius, 0 if tool == "eraser" else 255, opacity, selection_mask)
            elif tool in ["blur_tool", "sharpen_tool", "dodge", "burn"]:
                if self._retouch_stroke is not None:
                    changed = self._retouch_stroke.dab(x, y)
            else:
                changed = draw_brush(
                    self.doc.layer,
                    x,
                    y,
                    radius,
                    self.foreground,
                    opacity,
                    tool == "eraser",
                    selection_mask,
                )
            changed_rect = union_rect(changed_rect, changed)
        self.doc.dirty = True
        if changed_rect is not None:
            rect = self.local_to_document_rect(changed_rect, self.doc.layer)
            self.request_canvas_refresh(rect, self.doc.layer, self._stroke_kind)
        elif tool in {"clone", "healing"}:
            self.status_text("Источник и цель не пересекают доступные пиксели")

    @staticmethod
    def local_to_document_rect(rect: tuple[int, int, int, int], layer) -> tuple[int, int, int, int]:
        return rect[0] + layer.x, rect[1] + layer.y, rect[2] + layer.x, rect[3] + layer.y

    def clone_source_for_point(self, point: tuple[int, int]) -> tuple[int, int] | None:
        return self._source_anchor.source_for(point)

    def clone_source_click(self, event) -> str:
        if self.tool.get() not in {"clone", "healing"}:
            return "break"
        self.canvas.focus_set()
        self.set_clone_source(self.canvas_to_doc(event))
        return "break"

    def set_clone_source(self, point: tuple[int, int]) -> None:
        if point[0] < 0 or point[1] < 0 or point[0] >= self.doc.width or point[1] >= self.doc.height:
            self.status_text("Источник должен находиться внутри холста")
            return
        self._source_anchor.set_source(point)
        self._clone_source = point
        self.drag_start = None
        self.last_point = None
        self.update_clone_source_marker()
        self.status_text(f"Источник выбран: {point[0]}, {point[1]}. Теперь проведите кистью по цели.")

    def prepare_clone_sample(self) -> None:
        mode = self.clone_sampling.get()
        if mode == "Текущий слой":
            self._clone_sample_pixels = self.doc.layer.pixels.copy()
            self._clone_sample_origin = (self.doc.layer.x, self.doc.layer.y)
            return
        temporary = copy.copy(self.doc)
        if mode == "Текущий и ниже":
            temporary.layers = list(self.doc.layers[: self.doc.active_layer + 1])
            temporary.active_layer = len(temporary.layers) - 1
        else:
            temporary.layers = list(self.doc.layers)
        self._clone_sample_pixels = temporary.composite(False).copy()
        self._clone_sample_origin = (0, 0)

    def update_clone_source_marker(self) -> None:
        for item_id in self._clone_source_marker_ids:
            self.canvas.delete(item_id)
        self._clone_source_marker_ids.clear()
        if self._source_anchor.point is None or self.tool.get() not in {"clone", "healing"}:
            return
        cx, cy = self.doc_to_canvas(*self._source_anchor.point)
        radius = max(4.0, float(self.brush_size.get()) * float(self.zoom.get()))
        self._clone_source_marker_ids = [
            self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, outline="#ffb000", dash=(5, 3), width=2),
            self.canvas.create_line(cx - 7, cy, cx + 7, cy, fill="#ffb000", width=2),
            self.canvas.create_line(cx, cy - 7, cx, cy + 7, fill="#ffb000", width=2),
        ]
        for item_id in self._clone_source_marker_ids:
            self.canvas.tag_raise(item_id)

    def begin_patch_drag(self, point: tuple[int, int]) -> None:
        if self.doc.layer.locked:
            self.status_text("Слой заблокирован")
            self.drag_start = None
            return
        if self.doc.selection_mask is None or not np.any(self.doc.selection_mask):
            self.status_text("Сначала создайте выделение для инструмента Заплатка")
            self.drag_start = None
            return
        x, y = point
        if x < 0 or y < 0 or x >= self.doc.width or y >= self.doc.height or self.doc.selection_mask[y, x] == 0:
            self.status_text("Начните перетаскивание внутри активного выделения")
            self.drag_start = None
            return
        self._patch_start_bounds = self.doc.selection_bounds()
        self.draw_patch_preview(point)
        self.status_text("Перетащите выделение на область-источник")

    def patch_source_bounds_for_point(self, point: tuple[int, int]) -> tuple[int, int, int, int] | None:
        if self.drag_start is None or self._patch_start_bounds is None:
            return None
        dx = point[0] - self.drag_start[0]
        dy = point[1] - self.drag_start[1]
        x1, y1, x2, y2 = self._patch_start_bounds
        return x1 + dx, y1 + dy, x2 + dx, y2 + dy

    def patch_source_in_active_layer(self, bounds: tuple[int, int, int, int]) -> bool:
        layer = self.doc.layer
        x1, y1, x2, y2 = bounds
        return x1 >= layer.x and y1 >= layer.y and x2 <= layer.x + layer.pixels.shape[1] and y2 <= layer.y + layer.pixels.shape[0]

    def draw_patch_preview(self, point: tuple[int, int]) -> None:
        bounds = self.patch_source_bounds_for_point(point)
        if bounds is None:
            return
        x1, y1 = self.doc_to_canvas(bounds[0], bounds[1])
        x2, y2 = self.doc_to_canvas(bounds[2], bounds[3])
        coords = [x1, y1, x2, y2]
        valid = self.patch_source_in_active_layer(bounds)
        color = "#ffb000" if valid else "#ff4a4a"
        if self._patch_preview_id is None:
            self._patch_preview_id = self.canvas.create_rectangle(*coords, outline=color, dash=(6, 3), width=2)
        else:
            self.canvas.coords(self._patch_preview_id, *coords)
            self.canvas.itemconfigure(self._patch_preview_id, outline=color)
        self.canvas.tag_raise(self._patch_preview_id)

    def clear_patch_preview(self) -> None:
        if self._patch_preview_id is not None:
            self.canvas.delete(self._patch_preview_id)
            self._patch_preview_id = None
        self._patch_start_bounds = None

    def finish_patch_drag(self, point: tuple[int, int]) -> None:
        bounds = self.patch_source_bounds_for_point(point)
        if bounds is None:
            self.clear_patch_preview()
            return
        if not self.patch_source_in_active_layer(bounds):
            self.status_text("Источник заплатки должен полностью попадать в активный слой")
            self.clear_patch_preview()
            return
        source_x, source_y = bounds[0], bounds[1]
        self.run_document_command("Интерактивная заплатка", lambda: self.doc.patch_active_selection(source_x, source_y, True))
        self.clear_patch_preview()
        self.refresh()

    def current_shape_options(self, tool: str) -> dict:
        custom_name = self.custom_shape_preset.get()
        custom_points = CUSTOM_SHAPE_PRESETS.get(custom_name, next(iter(CUSTOM_SHAPE_PRESETS.values())))
        line_shape = tool in {"line_shape", "bezier_shape"}
        return {
            "fill": self.foreground,
            "stroke": self.foreground if line_shape else self.background,
            "stroke_width": max(1 if line_shape else 0, int(self.shape_stroke_width.get())),
            "sides": max(3, min(64, int(self.polygon_sides.get() if tool == "polygon_shape" else self.star_points_count.get()))),
            "inner_ratio": float(np.clip(self.star_inner_ratio.get(), 0.05, 0.95)),
            "custom_points": custom_points,
        }

    def shape_geometry_for_drag(self, tool: str, start: tuple[int, int], end: tuple[int, int], state: int = 0) -> dict:
        options = self._shape_drag_options or self.current_shape_options(tool)
        return shape_geometry_from_drag(
            tool,
            start,
            end,
            shift=bool(state & 0x0001),
            alt=bool(state & 0x0008),
            sides=int(options["sides"]),
            inner_ratio=float(options["inner_ratio"]),
            custom_points=options["custom_points"],
        )

    @staticmethod
    def color_hex(color: tuple[int, int, int, int]) -> str:
        return "#{:02x}{:02x}{:02x}".format(*color[:3])

    def draw_selection(self, start: tuple[int, int] | None, end: tuple[int, int], state: int = 0) -> None:
        if not start:
            return
        tool = self.tool.get()
        if tool == "crop":
            if self._crop_drag_handle is not None and self._crop_drag_origin_box is not None:
                self._crop_box = self.resize_crop_box(self._crop_drag_origin_box, self._crop_drag_handle, end)
            else:
                self._crop_box = self.crop_box_for_drag(start, end)
            self.draw_crop_overlay(self._crop_box)
            return
        is_shape = tool.endswith("_shape")
        if is_shape:
            shape_options = self._shape_drag_options or self.current_shape_options(tool)
            geometry = self.shape_geometry_for_drag(tool, start, end, state)
            shape = str(geometry["shape"])
            fill = self.color_hex(shape_options["fill"])
            outline = self.color_hex(shape_options["stroke"])
            stroke_width = int(shape_options["stroke_width"])
            width = max(1, round(stroke_width * self.zoom.get()))
            visible_outline = outline if stroke_width > 0 else ""
            x1, y1, x2, y2 = geometry["box"]
            canvas_box = (*self.doc_to_canvas(x1, y1), *self.doc_to_canvas(x2, y2))
            if shape == "ellipse":
                self.update_drag_preview_item("oval", canvas_box, fill=fill, outline=visible_outline, width=width)
            elif shape == "line":
                line = geometry["line"]
                coords = (*self.doc_to_canvas(line[0], line[1]), *self.doc_to_canvas(line[2], line[3]))
                self.update_drag_preview_item("line", coords, fill=outline, width=width)
            elif shape == "bezier":
                curve = [value for point in geometry["points"] for value in self.doc_to_canvas(point[0], point[1])]
                self.update_drag_preview_item("line", curve, fill=outline, width=width, smooth=True)
            elif shape in {"polygon", "star", "custom"}:
                polygon = [value for point in geometry["points"] for value in self.doc_to_canvas(point[0], point[1])]
                self.update_drag_preview_item("polygon", polygon, fill=fill, outline=visible_outline, width=width)
            else:
                self.update_drag_preview_item("rectangle", canvas_box, fill=fill, outline=visible_outline, width=width)
        else:
            x1, y1 = self.doc_to_canvas(start[0], start[1])
            x2, y2 = self.doc_to_canvas(end[0], end[1])
            coords = [x1, y1, x2, y2]
            if tool == "ellipse_select":
                self.update_drag_preview_item("oval", coords, outline="#50e3ff", dash=(5, 4), width=2)
            elif tool == "gradient":
                self.update_drag_preview_item("line", coords, fill="#50e3ff", width=2, arrow=tk.LAST)
            else:
                self.update_drag_preview_item("rectangle", coords, outline="#50e3ff", dash=(5, 4), width=2)
        for item_id in self._drag_preview_ids:
            self.canvas.tag_raise(item_id)
