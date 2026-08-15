from __future__ import annotations

from ..app_shared import *


class PaintingMixin:
    def current_brush_settings(self, *, opacity: float | None = None) -> BrushSettings:
        advanced = getattr(self, "brush_advanced", {})
        return BrushSettings(
            radius=int(self.brush_size.get()),
            hardness=float(self.hardness.get()),
            opacity=float(self.opacity.get()) if opacity is None else float(opacity),
            flow=float(self.brush_flow.get()),
            spacing=float(self.brush_spacing.get()),
            smoothing=float(self.brush_smoothing.get()),
            blend_mode=self.brush_blend_mode.get(),
            pressure_size=bool(self.pressure_size.get()),
            pressure_opacity=bool(self.pressure_opacity.get()),
            pressure_flow=bool(self.pressure_flow.get()),
            **{key: value for key, value in advanced.items() if key in BrushSettings.__dataclass_fields__},
        ).normalized()

    def begin_stroke(self, kind: str = "pixels", point: tuple[int, int] | None = None, pressure: float | TabletSample = 1.0) -> None:
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
        opacity_override = 1.0 if tool in {"blur_tool", "sharpen_tool", "dodge", "burn"} else None
        if tool == "healing":
            opacity_override = float(self.retouch_strength.get())
        settings = self.current_brush_settings(opacity=opacity_override)
        self._brush_path = BrushPathSampler(settings)
        self._initial_brush_dabs = []
        if point is not None:
            self._initial_brush_dabs = self._brush_path.begin(point, pressure)
        self._active_brush_stroke = None
        self._source_retouch_stroke = None
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
                flow=settings.flow,
                opacity=settings.opacity,
                pressure_size=settings.pressure_size,
                pressure_opacity=settings.pressure_opacity,
                pressure_flow=settings.pressure_flow,
            )
        elif kind == "mask" and tool in {"brush", "eraser"}:
            self._active_brush_stroke = MaskBrushStroke(
                self.doc.layer,
                settings,
                0 if tool == "eraser" else 255,
                self._stroke_selection_mask,
            )
            self._retouch_stroke = None
        elif kind == "pixels" and tool in {"brush", "eraser"}:
            self._active_brush_stroke = PixelBrushStroke(
                self.doc.layer,
                settings,
                self.foreground,
                background=self.background,
                erase=tool == "eraser",
                selection_mask=self._stroke_selection_mask,
            )
            self._retouch_stroke = None
        elif kind == "pixels" and tool in {"clone", "healing"} and self._clone_sample_pixels is not None:
            self._source_retouch_stroke = CloneHealingStroke(
                self.doc.layer,
                settings,
                self._clone_sample_pixels,
                self._clone_sample_origin,
                heal=tool == "healing",
                selection_mask=self._stroke_selection_mask,
                transform=self.current_source_transform(),
            )
            self._retouch_stroke = None
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
        if self._retouch_stroke is not None or self._active_brush_stroke is not None or self._source_retouch_stroke is not None:
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
        if self._active_brush_stroke is not None:
            self._stroke_tiles = self._active_brush_stroke.before_tiles
        elif self._retouch_stroke is not None:
            self._stroke_tiles = self._retouch_stroke.before_tiles
        elif self._source_retouch_stroke is not None:
            self._stroke_tiles = self._source_retouch_stroke.before_tiles
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
        self._active_brush_stroke = None
        self._source_retouch_stroke = None
        self._brush_path = None
        if self._editor_active:
            self.refresh_layer_previews()

    def end_move_layer(self) -> None:
        if len(self._move_group_starts) > 1:
            after = {layer_id: (layer.x, layer.y) for layer_id in self._move_group_starts if (layer := self.doc.get_layer(layer_id)) is not None}
            if after != self._move_group_starts:
                after_masks = {layer_id: None if (layer := self.doc.get_layer(layer_id)) is None or layer.mask is None else layer.mask.copy() for layer_id in after}
                self.push_command(LayerGroupMoveCommand("Переместить группу", dict(self._move_group_starts), after, dict(self._move_group_masks), after_masks))
        elif self._move_layer_id and self._move_start:
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
        self._move_group_starts.clear()
        self._move_group_masks.clear()

    def paint_at(self, point: tuple[int, int], pressure: float | TabletSample = 1.0) -> None:
        self.capture_stroke_before(self.brush_local_rect(point))
        tool = self.tool.get()
        pressure_value = pressure.pressure if isinstance(pressure, TabletSample) else float(pressure)
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
                self.spot_healing_mode.get(),
            )
        elif tool in ["clone", "healing"]:
            source = self.clone_source_for_point(point)
            if source is not None and self._source_retouch_stroke is not None:
                changed = self._source_retouch_stroke.dab(point[0], point[1], source[0], source[1], pressure_value)
        elif self._stroke_kind == "mask":
            if self._active_brush_stroke is not None:
                initial = self._initial_brush_dabs or [None]
                for dab in initial:
                    current = self._active_brush_stroke.dab(
                        point[0] if dab is None else dab.x,
                        point[1] if dab is None else dab.y,
                        pressure if dab is None else dab.pressure,
                        dab,
                    )
                    changed = union_rect(changed, current)
                self._initial_brush_dabs = []
            else:
                changed = draw_mask_brush(self.doc.layer, point[0], point[1], int(self.brush_size.get()), 0 if tool == "eraser" else 255, float(self.opacity.get()), selection_mask)
        elif tool in ["blur_tool", "sharpen_tool", "dodge", "burn"]:
            if self._retouch_stroke is not None:
                changed = self._retouch_stroke.dab(point[0], point[1], pressure_value)
        elif self._active_brush_stroke is not None:
            initial = self._initial_brush_dabs or [None]
            for dab in initial:
                current = self._active_brush_stroke.dab(
                    point[0] if dab is None else dab.x,
                    point[1] if dab is None else dab.y,
                    pressure if dab is None else dab.pressure,
                    dab,
                )
                changed = union_rect(changed, current)
            self._initial_brush_dabs = []
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

    def paint_line(self, start: tuple[int, int], end: tuple[int, int], pressure: float | TabletSample = 1.0) -> None:
        radius = max(1, int(self.brush_size.get()))
        tool = self.tool.get()
        if self._brush_path is None:
            self._brush_path = BrushPathSampler(self.current_brush_settings())
            self._brush_path.begin(start, pressure)
        dabs = self._brush_path.extend(end, pressure)
        selection_mask = self._stroke_selection_mask
        opacity = float(self.opacity.get())
        changed_rect = None
        for dab in dabs:
            x, y = dab.x, dab.y
            self.capture_stroke_before(self.brush_local_rect((x, y)))
            changed = None
            if tool == "spot_healing":
                changed = spot_heal(
                    self.doc.layer, x, y, radius,
                    float(self.retouch_strength.get()) * float(self.brush_flow.get()),
                    selection_mask, float(self.hardness.get()), self.spot_healing_mode.get(),
                )
            elif tool in ["clone", "healing"]:
                source = self.clone_source_for_point((x, y))
                if source is not None and self._source_retouch_stroke is not None:
                    changed = self._source_retouch_stroke.dab(x, y, source[0], source[1], dab.pressure)
            elif self._stroke_kind == "mask":
                if self._active_brush_stroke is not None:
                    changed = self._active_brush_stroke.dab(x, y, dab.pressure, dab)
                else:
                    changed = draw_mask_brush(self.doc.layer, x, y, radius, 0 if tool == "eraser" else 255, opacity, selection_mask)
            elif tool in ["blur_tool", "sharpen_tool", "dodge", "burn"]:
                if self._retouch_stroke is not None:
                    changed = self._retouch_stroke.dab(x, y, dab.pressure)
            elif self._active_brush_stroke is not None:
                changed = self._active_brush_stroke.dab(x, y, dab.pressure, dab)
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
        source = self._source_anchor.source_for(point)
        if source is None:
            return None
        return source[0] + int(self.clone_offset_x.get()), source[1] + int(self.clone_offset_y.get())

    def current_source_transform(self) -> SourceTransform:
        return SourceTransform(
            float(self.clone_scale_x.get()) / 100.0,
            float(self.clone_scale_y.get()) / 100.0,
            float(self.clone_rotation.get()),
            bool(self.clone_flip_horizontal.get()),
            bool(self.clone_flip_vertical.get()),
        )

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
        self.clone_source_x.set(point[0])
        self.clone_source_y.set(point[1])
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
