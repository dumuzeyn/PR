from __future__ import annotations

from ..app_shared import *
from ..shape_ops import editable_bezier_nodes, editable_bezier_path_points
from ..vector_geometry import nearest_cubic_parameter, split_cubic_bezier


PATH_TOOLS = {"path_select", "direct_select", "add_anchor", "delete_anchor", "convert_anchor"}


class PathEditingMixin:
    def active_bezier_layer(self) -> Layer | None:
        if not self.doc.layers:
            return None
        layer = self.doc.layer
        if layer.kind != "shape" or layer.shape_data is None or str(layer.shape_data.get("shape")) != "bezier":
            return None
        return layer

    def ensure_editable_path(self, layer: Layer) -> list[dict[str, object]]:
        nodes = editable_bezier_nodes(layer.shape_data or {})
        if layer.shape_data is not None and not isinstance(layer.shape_data.get("path_nodes"), list):
            layer.shape_data["path_nodes"] = copy.deepcopy(nodes)
        return layer.shape_data["path_nodes"]  # type: ignore[return-value,index]

    def snap_path_point(self, point: tuple[float, float]) -> list[float]:
        if bool(self.grid_visible.get()):
            spacing = max(4, int(self.grid_spacing.get()))
            return [round(point[0] / spacing) * spacing, round(point[1] / spacing) * spacing]
        return [float(point[0]), float(point[1])]

    def path_point_hit(self, point: tuple[int, int]) -> tuple[int, str] | None:
        layer = self.active_bezier_layer()
        if layer is None:
            return None
        local = (point[0] - layer.x, point[1] - layer.y)
        tolerance = max(5.0, 9.0 / max(0.01, float(self.zoom.get())))
        best: tuple[int, str] | None = None
        best_distance = tolerance
        nodes = self.ensure_editable_path(layer)
        for kind in ("anchor", "in", "out"):
            for index, node in enumerate(nodes):
                value = node.get(kind, node["anchor"])
                distance = math.hypot(local[0] - float(value[0]), local[1] - float(value[1]))
                if distance < best_distance:
                    best, best_distance = (index, kind), distance
            if best is not None:
                return best
        return best

    def path_pointer_down(self, tool: str, point: tuple[int, int], state: int) -> None:
        if tool == "path_select":
            layer = self.select_object_at(point, add=bool(state & 0x0001))
            if layer is None or layer.kind != "shape" or layer.shape_data is None or str(layer.shape_data.get("shape")) != "bezier":
                self.status_text("Под курсором нет контура Безье")
            else:
                self.ensure_editable_path(layer)
                self.status_text("Контур выбран")
            self.update_path_overlay()
            return
        layer = self.active_bezier_layer()
        if layer is None:
            candidate = self.select_object_at(point)
            if candidate is None or candidate.kind != "shape" or candidate.shape_data is None or str(candidate.shape_data.get("shape")) != "bezier":
                self.status_text("Сначала выберите контур Безье")
                return
            layer = candidate
        hit = self.path_point_hit(point)
        if tool == "add_anchor":
            self.add_path_anchor(point)
            return
        if hit is None:
            if not state & 0x0001:
                self._path_selected_nodes.clear()
            self.update_path_overlay()
            return
        index, kind = hit
        if tool == "delete_anchor":
            self.delete_path_anchor(index)
            return
        if tool == "convert_anchor":
            self.convert_path_anchor(index)
            return
        if kind == "anchor":
            if state & 0x0001:
                if index in self._path_selected_nodes:
                    self._path_selected_nodes.remove(index)
                else:
                    self._path_selected_nodes.add(index)
            else:
                self._path_selected_nodes = {index}
        elif not state & 0x0001:
            self._path_selected_nodes = {index}
        self._path_drag_target = (index, kind)
        self._path_drag_before = copy.deepcopy(layer.shape_data)
        self._path_drag_origin = (point, copy.deepcopy(self.ensure_editable_path(layer)))
        self.update_path_overlay()

    def path_pointer_drag(self, point: tuple[int, int], state: int) -> None:
        layer = self.active_bezier_layer()
        if layer is None or self._path_drag_target is None or self._path_drag_origin is None:
            return
        index, kind = self._path_drag_target
        origin, original_nodes = self._path_drag_origin
        nodes = self.ensure_editable_path(layer)
        local = self.snap_path_point((point[0] - layer.x, point[1] - layer.y))
        if kind == "anchor":
            dx = local[0] - (origin[0] - layer.x)
            dy = local[1] - (origin[1] - layer.y)
            targets = self._path_selected_nodes or {index}
            for target in targets:
                if target >= len(nodes):
                    continue
                source = original_nodes[target]
                for key in ("anchor", "in", "out"):
                    value = source.get(key, source["anchor"])
                    nodes[target][key] = [float(value[0]) + dx, float(value[1]) + dy]
        else:
            node = nodes[index]
            node[kind] = local
            if event_alt_down(state):
                node["linked"] = False
            elif bool(node.get("linked", True)):
                opposite = "out" if kind == "in" else "in"
                anchor = node["anchor"]
                node[opposite] = [float(anchor[0]) * 2.0 - local[0], float(anchor[1]) * 2.0 - local[1]]
        self.render_path_edit(layer)

    def finish_path_drag(self) -> None:
        layer = self.active_bezier_layer()
        before = self._path_drag_before
        self._path_drag_target = None
        self._path_drag_origin = None
        self._path_drag_before = None
        if layer is None or before is None or layer.shape_data == before:
            return
        self.push_command(ShapeDataCommand("Изменить контур Безье", layer.id, before, copy.deepcopy(layer.shape_data), layer.name, layer.name))
        self.doc.dirty = True
        self.refresh_properties()

    def render_path_edit(self, layer: Layer) -> None:
        old_bounds = self.layer_render_bounds(layer)
        render_shape_layer(layer)
        layer.touch_pixels()
        self.doc.dirty = True
        new_bounds = self.layer_render_bounds(layer)
        dirty = new_bounds if old_bounds is None else old_bounds if new_bounds is None else union_rect(old_bounds, new_bounds)
        self.request_canvas_refresh(dirty, layer, "pixels")
        self.update_path_overlay()

    def add_path_anchor(self, point: tuple[int, int]) -> None:
        layer = self.active_bezier_layer()
        if layer is None or layer.shape_data is None:
            return
        before = copy.deepcopy(layer.shape_data)
        nodes = self.ensure_editable_path(layer)
        local = self.snap_path_point((point[0] - layer.x, point[1] - layer.y))
        nearest_segment = 0
        nearest_distance = float("inf")
        nearest_amount = 0.5
        for index in range(len(nodes) - 1):
            first, second = nodes[index], nodes[index + 1]
            control = [first["anchor"], first.get("out", first["anchor"]), second.get("in", second["anchor"]), second["anchor"]]
            amount, distance = nearest_cubic_parameter(control, local)
            if distance < nearest_distance:
                nearest_segment, nearest_amount, nearest_distance = index, amount, distance
        tolerance = max(5.0, 9.0 / max(0.01, float(self.zoom.get())))
        if nearest_distance > tolerance:
            self.status_text("Кликните ближе к участку контура")
            return
        previous, following = nodes[nearest_segment], nodes[nearest_segment + 1]
        control = [previous["anchor"], previous.get("out", previous["anchor"]), following.get("in", following["anchor"]), following["anchor"]]
        left, right = split_cubic_bezier(control, nearest_amount)
        previous["out"] = list(left[1])
        following["in"] = list(right[2])
        node = {
            "anchor": list(left[3]),
            "in": list(left[2]),
            "out": list(right[1]),
            "linked": True,
        }
        nodes.insert(nearest_segment + 1, node)
        self._path_selected_nodes = {nearest_segment + 1}
        self.render_path_edit(layer)
        self.push_command(ShapeDataCommand("Добавить узел контура", layer.id, before, copy.deepcopy(layer.shape_data), layer.name, layer.name))

    def delete_path_anchor(self, index: int | None = None) -> None:
        layer = self.active_bezier_layer()
        if layer is None or layer.shape_data is None:
            return
        nodes = self.ensure_editable_path(layer)
        targets = {index} if index is not None else set(self._path_selected_nodes)
        targets = {value for value in targets if value is not None and 0 <= value < len(nodes)}
        if not targets or len(nodes) - len(targets) < 2:
            self.status_text("В открытом контуре должно остаться минимум два узла")
            return
        before = copy.deepcopy(layer.shape_data)
        for target in sorted(targets, reverse=True):
            nodes.pop(target)
        self._path_selected_nodes.clear()
        self.render_path_edit(layer)
        self.push_command(ShapeDataCommand("Удалить узел контура", layer.id, before, copy.deepcopy(layer.shape_data), layer.name, layer.name))

    def delete_selected_anchors(self) -> bool:
        if self.tool.get() != "direct_select" or not self._path_selected_nodes:
            return False
        self.delete_path_anchor()
        return True

    def convert_path_anchor(self, index: int) -> None:
        layer = self.active_bezier_layer()
        if layer is None or layer.shape_data is None:
            return
        nodes = self.ensure_editable_path(layer)
        before = copy.deepcopy(layer.shape_data)
        node = nodes[index]
        anchor = [float(value) for value in node["anchor"]]
        collapsed = all(math.hypot(float(node[key][0]) - anchor[0], float(node[key][1]) - anchor[1]) < 1.0 for key in ("in", "out"))
        if collapsed:
            node["in"], node["out"], node["linked"] = [anchor[0] - 30, anchor[1]], [anchor[0] + 30, anchor[1]], True
        else:
            node["in"], node["out"], node["linked"] = anchor.copy(), anchor.copy(), False
        self._path_selected_nodes = {index}
        self.render_path_edit(layer)
        self.push_command(ShapeDataCommand("Преобразовать узел контура", layer.id, before, copy.deepcopy(layer.shape_data), layer.name, layer.name))

    def update_path_overlay(self) -> None:
        if not hasattr(self, "canvas"):
            return
        for item_id in self._path_overlay_ids:
            self.canvas.delete(item_id)
        self._path_overlay_ids.clear()
        if self.tool.get() not in PATH_TOOLS:
            return
        layer = self.active_bezier_layer()
        if layer is None:
            return
        for index, node in enumerate(self.ensure_editable_path(layer)):
            anchor = self.doc_to_canvas(float(node["anchor"][0]) + layer.x, float(node["anchor"][1]) + layer.y)
            incoming = self.doc_to_canvas(float(node["in"][0]) + layer.x, float(node["in"][1]) + layer.y)
            outgoing = self.doc_to_canvas(float(node["out"][0]) + layer.x, float(node["out"][1]) + layer.y)
            self._path_overlay_ids.append(self.canvas.create_line(*incoming, *anchor, *outgoing, fill="#f0b84f", dash=(4, 3), width=1))
            for kind, (x, y) in (("in", incoming), ("out", outgoing)):
                self._path_overlay_ids.append(self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#f0b84f", outline=TOKENS.WORKSPACE))
            x, y = anchor
            fill = TOKENS.ACCENT if index in self._path_selected_nodes else TOKENS.SURFACE
            self._path_overlay_ids.append(self.canvas.create_rectangle(x - 6, y - 6, x + 6, y + 6, fill=fill, outline=TOKENS.TEXT_PRIMARY, width=1))
        for item_id in self._path_overlay_ids:
            self.canvas.tag_raise(item_id)
