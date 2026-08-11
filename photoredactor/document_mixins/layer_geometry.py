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
from ..precision_pipeline import alpha_blend_precision


class LayerGeometryDocumentMixin:
    def duplicate_active_layer(self) -> None:
        self.layers.insert(self.active_layer + 1, self.layer.clone())
        self.active_layer += 1
        self.dirty = True

    def move_active_layer(self, dx: int, dy: int) -> None:
        layer = self.layer
        if layer.locked:
            return
        dx, dy = int(dx), int(dy)
        if dx == 0 and dy == 0:
            return
        layer.x += dx
        layer.y += dy
        if layer.mask is not None and not layer.mask_linked:
            h, w = layer.mask.shape[:2]
            layer.mask = shifted_mask(layer.mask, w, h, w, h, -dx, -dy)
        self.dirty = True

    def merge_down(self) -> None:
        if self.active_layer <= 0:
            return
        lower = self.layers[self.active_layer - 1]
        upper = self.layer
        min_x = min(lower.x, upper.x)
        min_y = min(lower.y, upper.y)
        max_x = max(lower.x + lower.pixels.shape[1], upper.x + upper.pixels.shape[1])
        max_y = max(lower.y + lower.pixels.shape[0], upper.y + upper.pixels.shape[0])
        precise = self.bit_depth > 8 or lower.working_pixels is not None or upper.working_pixels is not None
        if precise:
            merged = np.zeros((max_y - min_y, max_x - min_x, 4), dtype=np.float32)
            alpha_blend_precision(merged, lower.working_rgba(), lower.x - min_x, lower.y - min_y, lower.opacity, lower.mask if lower.mask_enabled else None, lower.mask_density, lower.blend_mode)
            alpha_blend_precision(merged, upper.working_rgba(), upper.x - min_x, upper.y - min_y, upper.opacity, upper.mask if upper.mask_enabled else None, upper.mask_density, upper.blend_mode)
            lower.set_working_rgba(merged, self.bit_depth, self.color_model)
        else:
            merged = blank_rgba(max_x - min_x, max_y - min_y, (0, 0, 0, 0))
            alpha_blend_inplace(merged, lower.pixels, lower.x - min_x, lower.y - min_y, lower.opacity, lower.mask if lower.mask_enabled else None, lower.mask_density, lower.blend_mode)
            alpha_blend_inplace(merged, upper.pixels, upper.x - min_x, upper.y - min_y, upper.opacity, upper.mask if upper.mask_enabled else None, upper.mask_density, upper.blend_mode)
            lower.replace_pixels(merged)
        lower.x = min_x
        lower.y = min_y
        lower.opacity = 1.0
        lower.mask = None
        del self.layers[self.active_layer]
        self.active_layer -= 1
        self.dirty = True

    def flatten(self) -> None:
        pixels = self.composite_precision(False) if self.bit_depth > 8 else self.composite(False)
        flattened = Layer("Flattened", display_rgba(pixels))
        if self.bit_depth > 8:
            flattened.set_working_rgba(pixels, self.bit_depth, self.color_model)
        self.layers = [flattened]
        self.active_layer = 0
        self.dirty = True

    def resize_image(self, width: int, height: int, interpolation=cv2.INTER_CUBIC) -> None:
        for layer in self.layers:
            new_w = max(1, round(layer.pixels.shape[1] * width / self.width))
            new_h = max(1, round(layer.pixels.shape[0] * height / self.height))
            if layer.kind in {"linked", "embedded"} and layer.smart_source is not None:
                data = dict(layer.smart_data or {})
                transform = dict(data.get("transform") or {})
                transform["width"] = max(1, round(int(transform.get("width", layer.smart_source.shape[1])) * width / self.width))
                transform["height"] = max(1, round(int(transform.get("height", layer.smart_source.shape[0])) * height / self.height))
                data["transform"] = transform
                layer.smart_data = data
                render_smart_object(layer)
            else:
                source = layer.working_rgba() if layer.working_pixels is not None else layer.pixels
                layer.replace_pixels(cv2.resize(source, (new_w, new_h), interpolation=interpolation))
            if layer.mask is not None:
                layer.mask = cv2.resize(layer.mask, (layer.pixels.shape[1], layer.pixels.shape[0]), interpolation=cv2.INTER_NEAREST)
            layer.x = round(layer.x * width / self.width)
            layer.y = round(layer.y * height / self.height)
            if layer.kind in {"linked", "embedded"}:
                layer.touch_pixels()
        self.width, self.height = width, height
        if self.selection_mask is not None:
            self.selection_mask = cv2.resize(self.selection_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        for name, mask in list(self.saved_selections.items()):
            self.saved_selections[name] = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        self.dirty = True

    def resize_canvas(self, width: int, height: int, anchor="center") -> None:
        dx = (width - self.width) // 2 if anchor == "center" else 0
        dy = (height - self.height) // 2 if anchor == "center" else 0
        for layer in self.layers:
            layer.x += dx
            layer.y += dy
        self.width, self.height = width, height
        if self.selection_mask is not None:
            new_mask = np.zeros((height, width), dtype=np.uint8)
            x1, y1 = max(0, dx), max(0, dy)
            x2, y2 = min(width, dx + self.selection_mask.shape[1]), min(height, dy + self.selection_mask.shape[0])
            if x1 < x2 and y1 < y2:
                sx1, sy1 = x1 - dx, y1 - dy
                new_mask[y1:y2, x1:x2] = self.selection_mask[sy1 : sy1 + (y2 - y1), sx1 : sx1 + (x2 - x1)]
            self.selection_mask = new_mask
        for name, mask in list(self.saved_selections.items()):
            new_mask = np.zeros((height, width), dtype=np.uint8)
            x1, y1 = max(0, dx), max(0, dy)
            x2, y2 = min(width, dx + mask.shape[1]), min(height, dy + mask.shape[0])
            if x1 < x2 and y1 < y2:
                sx1, sy1 = x1 - dx, y1 - dy
                new_mask[y1:y2, x1:x2] = mask[sy1 : sy1 + (y2 - y1), sx1 : sx1 + (x2 - x1)]
            self.saved_selections[name] = new_mask
        self.dirty = True

    def generative_expand(self, left: int, top: int, right: int, bottom: int, method: str = "content-aware") -> Layer:
        left, top, right, bottom = (max(0, int(value)) for value in (left, top, right, bottom))
        if left + top + right + bottom == 0:
            raise ValueError("At least one expansion margin must be greater than zero")
        original = self.composite(False)
        expanded = generative_expand_pixels(original, left, top, right, bottom, method)
        generated = expanded.copy()
        generated[top : top + self.height, left : left + self.width] = 0
        for layer in self.layers:
            layer.x += left
            layer.y += top
        generated_layer = Layer("Генеративное расширение", generated)
        self.layers.insert(0, generated_layer)
        self.active_layer += 1
        self.width += left + right
        self.height += top + bottom
        if self.selection_mask is not None:
            selection = np.zeros((self.height, self.width), dtype=np.uint8)
            selection[top : top + self.selection_mask.shape[0], left : left + self.selection_mask.shape[1]] = self.selection_mask
            self.selection_mask = selection
        self.saved_selections = {
            name: _expanded_mask(mask, left, top, self.width, self.height) for name, mask in self.saved_selections.items()
        }
        self.metadata["last_generative_expand"] = {
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "method": method,
        }
        self.dirty = True
        return generated_layer

    def crop(self, box: tuple[int, int, int, int]) -> None:
        x1, y1, x2, y2 = normalized_box(box)
        x1, x2 = max(0, min(self.width, x1)), max(0, min(self.width, x2))
        y1, y2 = max(0, min(self.height, y1)), max(0, min(self.height, y2))
        if x1 == x2 or y1 == y2:
            return
        new_w, new_h = max(1, x2 - x1), max(1, y2 - y1)
        for layer in self.layers:
            if layer.working_pixels is not None:
                canvas = np.zeros((self.height, self.width, 4), dtype=np.float32)
                alpha_blend_precision(canvas, layer.working_rgba(), layer.x, layer.y, 1.0)
            else:
                canvas = blank_rgba(self.width, self.height, (0, 0, 0, 0))
                alpha_blend_inplace(canvas, layer.pixels, layer.x, layer.y, 1.0)
            if layer.mask is not None:
                mask_canvas = np.zeros((self.height, self.width), dtype=np.uint8)
                paste_mask(mask_canvas, layer.mask, layer.x, layer.y)
                layer.mask = mask_canvas[y1:y2, x1:x2].copy()
            layer.replace_pixels(canvas[y1:y2, x1:x2].copy())
            layer.x = 0
            layer.y = 0
        self.width, self.height = new_w, new_h
        if self.selection_mask is not None:
            self.selection_mask = self.selection_mask[y1:y2, x1:x2].copy()
        for name, mask in list(self.saved_selections.items()):
            self.saved_selections[name] = mask[y1:y2, x1:x2].copy()
        self.dirty = True

    def trim_transparent(self) -> None:
        alpha = self.composite(False)[:, :, 3]
        if not np.any(alpha):
            return
        ys, xs = np.where(alpha > 0)
        self.crop((int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)))

    def reveal_all(self) -> None:
        min_x, min_y = 0, 0
        max_x, max_y = self.width, self.height
        for layer in self.layers:
            h, w = layer.pixels.shape[:2]
            min_x = min(min_x, layer.x)
            min_y = min(min_y, layer.y)
            max_x = max(max_x, layer.x + w)
            max_y = max(max_y, layer.y + h)
        if min_x == 0 and min_y == 0 and max_x == self.width and max_y == self.height:
            return
        dx, dy = -min_x, -min_y
        old_w, old_h = self.width, self.height
        new_w, new_h = max(1, max_x - min_x), max(1, max_y - min_y)
        for layer in self.layers:
            layer.x += dx
            layer.y += dy
        if self.selection_mask is not None:
            self.selection_mask = shifted_mask(self.selection_mask, old_w, old_h, new_w, new_h, dx, dy)
        for name, mask in list(self.saved_selections.items()):
            self.saved_selections[name] = shifted_mask(mask, old_w, old_h, new_w, new_h, dx, dy)
        self.width, self.height = new_w, new_h
        self.dirty = True
