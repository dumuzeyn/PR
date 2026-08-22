from __future__ import annotations

from ..core_shared import *
from ..layer import Layer
from ..geometry_ops import *
from ..selection_ops import *
from ..filter_ops import *
from ..render_ops import *
from ..retouch_ops import *
from ..text_ops import *
from ..shape_ops import *
from ..content_ops import *
from ..adjustment_ops import *


class VectorLayersDocumentMixin:
    def add_text_layer(
        self,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int, int],
        size: int,
        font_family: str = "arial.ttf",
        box_width: int = 0,
        align: str = "left",
        line_spacing: int | None = None,
        tracking: int = 0,
        bold: bool = False,
        italic: bool = False,
        underline: bool = False,
        path_mode: str = "none",
        path_amount: int = 0,
        path_points: list[tuple[float, float]] | list[list[float]] | None = None,
        path_start: float = 0.0,
        path_end: float = 1.0,
        path_side: int = 1,
        path_reverse: bool = False,
        baseline_shift: int = 0,
        rotation: float = 0.0,
        box_height: int = 0,
        text_mode: str | None = None,
        kerning_enabled: bool = True,
        standard_ligatures: bool = True,
        discretionary_ligatures: bool = False,
        stylistic_set: int = 0,
        direction: str = "auto",
        language: str = "",
    ) -> Layer:
        normalized_path_start, normalized_path_end = normalize_text_path_range(path_start, path_end)
        layer = Layer(
            name=f"Text: {text[:24]}",
            pixels=blank_rgba(self.width, self.height, (0, 0, 0, 0)),
            kind="text",
            text_data={
                "text": text,
                "x": int(x),
                "y": int(y),
                "color": list(color),
                "size": int(size),
                "font_family": font_family,
                "box_width": int(box_width),
                "box_height": max(0, int(box_height)),
                "text_mode": text_mode if text_mode in {"point", "paragraph"} else "paragraph" if int(box_width) > 0 else "point",
                "align": align,
                "line_spacing": int(line_spacing if line_spacing is not None else max(2, int(size) // 5)),
                "tracking": int(tracking),
                "kerning": 0,
                "kerning_enabled": bool(kerning_enabled),
                "standard_ligatures": bool(standard_ligatures),
                "discretionary_ligatures": bool(discretionary_ligatures),
                "stylistic_set": max(0, min(20, int(stylistic_set))),
                "direction": direction if direction in {"auto", "ltr", "rtl"} else "auto",
                "language": str(language),
                "horizontal_scale": 100,
                "vertical_scale": 100,
                "bold": bool(bold),
                "italic": bool(italic),
                "underline": bool(underline),
                "strike_through": False,
                "vertical": False,
                "indent_left": 0,
                "indent_right": 0,
                "first_line_indent": 0,
                "spacing_before": 0,
                "spacing_after": 0,
                "path_mode": path_mode if path_mode in {"none", "arc", "wave", "bezier"} else "none",
                "path_amount": int(path_amount),
                "path_points": normalize_text_path_points(path_points, int(x), int(y) + int(size), max(int(box_width), int(size) * 8)),
                "path_start": normalized_path_start,
                "path_end": normalized_path_end,
                "path_side": -1 if int(path_side) < 0 else 1,
                "path_reverse": bool(path_reverse),
                "baseline_shift": int(baseline_shift),
                "rotation": float(rotation),
            },
        )
        render_text_layer(layer)
        self.layers.append(layer)
        self.active_layer = len(self.layers) - 1
        self.dirty = True
        return layer

    def add_shape_layer(
        self,
        shape: str,
        box: tuple[int, int, int, int],
        fill: tuple[int, int, int, int],
        stroke: tuple[int, int, int, int] | None = None,
        stroke_width: int = 0,
        sides: int = 5,
        inner_ratio: float = 0.5,
        control_points: list[tuple[float, float]] | None = None,
        custom_points: list[tuple[float, float]] | None = None,
        gradient: dict[str, Any] | None = None,
        texture: dict[str, Any] | None = None,
    ) -> Layer:
        shape_box = normalized_box(box)
        if shape == "bezier" and control_points is None:
            x1, y1, x2, y2 = shape_box
            control_points = [(x1, y2), (x1, y1), (x2, y1), (x2, y2)]
        layer = Layer(
            name=f"{shape.title()} shape",
            pixels=blank_rgba(self.width, self.height, (0, 0, 0, 0)),
            kind="shape",
            shape_data={
                "shape": shape,
                "box": [int(v) for v in shape_box],
                "fill": list(fill),
                "fill_opacity": 1.0,
                "stroke": None if stroke is None else list(stroke),
                "stroke_enabled": bool(stroke is not None and stroke_width > 0),
                "stroke_opacity": 1.0,
                "stroke_width": int(stroke_width),
                "stroke_alignment": "center",
                "stroke_cap": "round",
                "stroke_join": "round",
                "miter_limit": 4.0,
                "dash_pattern": [],
                "dash_offset": 0.0,
                "opacity": 1.0,
                "rotation": 0.0,
                "corner_radius": 0,
                "sides": int(sides),
                "inner_ratio": float(inner_ratio),
                "control_points": None if control_points is None else [[float(x), float(y)] for x, y in control_points],
                "path_nodes": None,
                "custom_points": None if custom_points is None else [[float(x), float(y)] for x, y in custom_points],
                "gradient": None if gradient is None else json.loads(json.dumps(gradient)),
                "texture": None if texture is None else json.loads(json.dumps(texture)),
            },
        )
        render_shape_layer(layer)
        self.layers.append(layer)
        self.active_layer = len(self.layers) - 1
        self.dirty = True
        return layer

    def edit_shape_layer(
        self,
        shape: str | None = None,
        fill: tuple[int, int, int, int] | None = None,
        stroke: tuple[int, int, int, int] | None = None,
        stroke_width: int | None = None,
        sides: int | None = None,
        inner_ratio: float | None = None,
        control_points: list[tuple[float, float]] | None = None,
        custom_points: list[tuple[float, float]] | None = None,
    ) -> None:
        layer = self.layer
        if layer.locked or layer.kind != "shape" or layer.shape_data is None:
            return
        if shape is not None:
            layer.shape_data["shape"] = shape
            layer.name = f"{shape.title()} shape"
        if fill is not None:
            layer.shape_data["fill"] = list(fill)
        if stroke is not None:
            layer.shape_data["stroke"] = list(stroke)
        if stroke_width is not None:
            layer.shape_data["stroke_width"] = int(stroke_width)
        if sides is not None:
            layer.shape_data["sides"] = max(3, int(sides))
        if inner_ratio is not None:
            layer.shape_data["inner_ratio"] = float(np.clip(inner_ratio, 0.05, 0.95))
        if control_points is not None:
            layer.shape_data["control_points"] = [[float(x), float(y)] for x, y in control_points]
        if custom_points is not None:
            layer.shape_data["custom_points"] = [[float(x), float(y)] for x, y in custom_points]
        render_shape_layer(layer)
        self.dirty = True

    def boolean_shape_data_with_lower(self, mode: str) -> dict[str, Any] | None:
        if self.active_layer <= 0:
            return None
        upper = self.layer
        lower = self.layers[self.active_layer - 1]
        if upper.locked or lower.locked or upper.kind != "shape" or lower.kind != "shape" or upper.shape_data is None or lower.shape_data is None:
            return None
        mode = str(mode).lower().strip()
        if mode not in {"union", "subtract", "intersect", "xor"}:
            return None
        fill = upper.shape_data.get("fill", lower.shape_data.get("fill", [255, 255, 255, 255]))
        stroke = upper.shape_data.get("stroke")
        stroke_width = int(upper.shape_data.get("stroke_width", 0))
        lower_data = translated_shape_data(lower.shape_data, lower.x, lower.y)
        upper_data = translated_shape_data(upper.shape_data, upper.x, upper.y)
        lower_data.setdefault("_name", lower.name)
        upper_data.setdefault("_name", upper.name)
        lower_data.setdefault("_enabled", True)
        upper_data.setdefault("_enabled", True)
        result = {
            "shape": "boolean",
            "boolean_mode": mode,
            "children": [lower_data, upper_data],
            "fill": list(fill),
            "stroke": None if stroke is None else list(stroke),
            "stroke_width": max(0, stroke_width),
        }
        result["box"] = list(shape_data_bounds(result) or (0, 0, 1, 1))
        return result

    def boolean_active_shape_with_lower(self, mode: str, shape_data: dict[str, Any] | None = None) -> bool:
        data = self.boolean_shape_data_with_lower(mode) if shape_data is None else json.loads(json.dumps(shape_data))
        if data is None or self.active_layer <= 0:
            return False
        upper = self.layer
        lower = self.layers[self.active_layer - 1]
        if upper.locked or lower.locked or upper.kind != "shape" or lower.kind != "shape":
            return False
        data["shape"] = "boolean"
        data["boolean_mode"] = str(data.get("boolean_mode", mode)).lower()
        data["box"] = list(shape_data_bounds(data) or (0, 0, 1, 1))
        combined = Layer(
            name=f"Булева фигура: {data['boolean_mode']}",
            pixels=blank_rgba(self.width, self.height, (0, 0, 0, 0)),
            kind="shape",
            shape_data=data,
        )
        render_shape_layer(combined)
        self.layers[self.active_layer - 1] = combined
        del self.layers[self.active_layer]
        self.active_layer -= 1
        self.dirty = True
        return True

    def edit_text_layer(
        self,
        text: str | None = None,
        size: int | None = None,
        color: tuple[int, int, int, int] | None = None,
        font_family: str | None = None,
        box_width: int | None = None,
        align: str | None = None,
        line_spacing: int | None = None,
        tracking: int | None = None,
        kerning: int | None = None,
        horizontal_scale: int | None = None,
        vertical_scale: int | None = None,
        bold: bool | None = None,
        italic: bool | None = None,
        underline: bool | None = None,
        strike_through: bool | None = None,
        vertical: bool | None = None,
        indent_left: int | None = None,
        indent_right: int | None = None,
        first_line_indent: int | None = None,
        spacing_before: int | None = None,
        spacing_after: int | None = None,
        path_mode: str | None = None,
        path_amount: int | None = None,
        path_points: list[tuple[float, float]] | list[list[float]] | None = None,
        path_start: float | None = None,
        path_end: float | None = None,
        path_side: int | None = None,
        path_reverse: bool | None = None,
        baseline_shift: int | None = None,
        rotation: float | None = None,
        box_height: int | None = None,
        text_mode: str | None = None,
        kerning_enabled: bool | None = None,
        standard_ligatures: bool | None = None,
        discretionary_ligatures: bool | None = None,
        stylistic_set: int | None = None,
        direction: str | None = None,
        language: str | None = None,
    ) -> None:
        layer = self.layer
        if layer.locked or layer.kind != "text" or layer.text_data is None:
            return
        if text is not None:
            layer.text_data["text"] = text
            layer.name = f"Text: {text[:24]}"
        if size is not None:
            layer.text_data["size"] = int(size)
        if color is not None:
            layer.text_data["color"] = list(color)
        if font_family is not None:
            layer.text_data["font_family"] = font_family
        if box_width is not None:
            layer.text_data["box_width"] = max(0, int(box_width))
        if box_height is not None:
            layer.text_data["box_height"] = max(0, int(box_height))
        if text_mode is not None:
            layer.text_data["text_mode"] = text_mode if text_mode in {"point", "paragraph"} else "point"
        if align is not None:
            layer.text_data["align"] = align if align in {"left", "center", "right", "justify"} else "left"
        if line_spacing is not None:
            layer.text_data["line_spacing"] = max(0, int(line_spacing))
        if tracking is not None:
            layer.text_data["tracking"] = int(tracking)
        if kerning is not None:
            layer.text_data["kerning"] = int(kerning)
        if kerning_enabled is not None:
            layer.text_data["kerning_enabled"] = bool(kerning_enabled)
        if standard_ligatures is not None:
            layer.text_data["standard_ligatures"] = bool(standard_ligatures)
        if discretionary_ligatures is not None:
            layer.text_data["discretionary_ligatures"] = bool(discretionary_ligatures)
        if stylistic_set is not None:
            layer.text_data["stylistic_set"] = max(0, min(20, int(stylistic_set)))
        if direction is not None:
            layer.text_data["direction"] = direction if direction in {"auto", "ltr", "rtl"} else "auto"
        if language is not None:
            layer.text_data["language"] = str(language)
        if horizontal_scale is not None:
            layer.text_data["horizontal_scale"] = max(1, int(horizontal_scale))
        if vertical_scale is not None:
            layer.text_data["vertical_scale"] = max(1, int(vertical_scale))
        if bold is not None:
            layer.text_data["bold"] = bool(bold)
        if italic is not None:
            layer.text_data["italic"] = bool(italic)
        if underline is not None:
            layer.text_data["underline"] = bool(underline)
        if strike_through is not None:
            layer.text_data["strike_through"] = bool(strike_through)
        if vertical is not None:
            layer.text_data["vertical"] = bool(vertical)
        for key, value in {
            "indent_left": indent_left,
            "indent_right": indent_right,
            "first_line_indent": first_line_indent,
            "spacing_before": spacing_before,
            "spacing_after": spacing_after,
        }.items():
            if value is not None:
                layer.text_data[key] = int(value)
        if path_mode is not None:
            layer.text_data["path_mode"] = path_mode if path_mode in {"none", "arc", "wave", "bezier"} else "none"
        if path_amount is not None:
            layer.text_data["path_amount"] = int(path_amount)
        if path_points is not None:
            layer.text_data["path_points"] = normalize_text_path_points(
                path_points,
                int(layer.text_data.get("x", 0)),
                int(layer.text_data.get("y", 0)) + int(layer.text_data.get("size", 48)),
                max(int(layer.text_data.get("box_width", 0) or 0), int(layer.text_data.get("size", 48)) * 8),
            )
        if path_start is not None or path_end is not None:
            normalized_start, normalized_end = normalize_text_path_range(
                float(layer.text_data.get("path_start", 0.0)) if path_start is None else path_start,
                float(layer.text_data.get("path_end", 1.0)) if path_end is None else path_end,
            )
            layer.text_data["path_start"] = normalized_start
            layer.text_data["path_end"] = normalized_end
        if path_side is not None:
            layer.text_data["path_side"] = -1 if int(path_side) < 0 else 1
        if path_reverse is not None:
            layer.text_data["path_reverse"] = bool(path_reverse)
        if baseline_shift is not None:
            layer.text_data["baseline_shift"] = int(baseline_shift)
        if rotation is not None:
            layer.text_data["rotation"] = float(rotation)
        render_text_layer(layer)
        self.dirty = True

    def transform_active_text_box(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        angle: float = 0.0,
        flip_horizontal: bool = False,
        flip_vertical: bool = False,
    ) -> bool:
        layer = self.layer
        if layer.locked or layer.kind != "text" or layer.text_data is None or not np.any(layer.pixels[:, :, 3]):
            return False
        ys, xs = np.where(layer.pixels[:, :, 3] > 0)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
        old_width, old_height = max(1, x2 - x1), max(1, y2 - y1)
        scale_x = max(1, int(width)) / old_width
        scale_y = max(1, int(height)) / old_height
        data = layer.text_data
        data["x"] = int(data.get("x", 0)) + int(x) - x1
        data["y"] = int(data.get("y", 0)) + int(y) - y1
        data["size"] = max(4, round(int(data.get("size", 48)) * scale_y))
        if int(data.get("box_width", 0) or 0) > 0:
            data["box_width"] = max(1, round(int(data["box_width"]) * scale_x))
        if int(data.get("box_height", 0) or 0) > 0:
            data["box_height"] = max(1, round(int(data["box_height"]) * scale_y))
        data["line_spacing"] = max(0, round(int(data.get("line_spacing", 0)) * scale_y))
        data["tracking"] = round(int(data.get("tracking", 0)) * scale_x)
        raw_path = data.get("path_points")
        if isinstance(raw_path, list) and len(raw_path) == 4:
            data["path_points"] = [
                [
                    int(x) + (float(point[0]) - x1) * scale_x,
                    int(y) + (float(point[1]) - y1) * scale_y,
                ]
                for point in raw_path
                if isinstance(point, (list, tuple)) and len(point) >= 2
            ]
        data["rotation"] = float(data.get("rotation", 0.0)) + float(angle)
        if flip_horizontal:
            data["flip_horizontal"] = not bool(data.get("flip_horizontal", False))
        if flip_vertical:
            data["flip_vertical"] = not bool(data.get("flip_vertical", False))
        render_text_layer(layer)
        self.dirty = True
        return True

    def add_adjustment_layer(self, name: str, adjustment: dict[str, Any]) -> None:
        layer = Layer(
            name=name,
            pixels=blank_rgba(self.width, self.height, (0, 0, 0, 0)),
            kind="adjustment",
            adjustment=dict(adjustment),
        )
        self.layers.append(layer)
        self.active_layer = len(self.layers) - 1
        self.dirty = True

    def delete_active_layer(self) -> None:
        if len(self.layers) <= 1:
            return
        del self.layers[self.active_layer]
        self.active_layer = max(0, self.active_layer - 1)
        self.dirty = True

    def toggle_active_clipping(self) -> None:
        if self.active_layer <= 0:
            return
        self.layer.clipping = not self.layer.clipping
        self.dirty = True

    def set_active_layer_effects(self, effects: dict[str, Any]) -> None:
        self.layer.effects = json.loads(json.dumps(effects))
        self.dirty = True

    def set_active_layer_filters(self, filters: list[dict[str, Any]]) -> None:
        self.layer.filters = json.loads(json.dumps(filters))
        self.dirty = True

    def clear_active_layer_filters(self) -> None:
        self.layer.filters = []
        self.dirty = True
