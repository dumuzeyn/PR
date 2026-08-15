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
from ..selection_color import perceptual_color_mask, combine_sample_masks


class SelectionsDocumentMixin:
    def set_rect_selection(self, box: tuple[int, int, int, int], mode: str = "replace", feather: int = 0) -> None:
        x1, y1, x2, y2 = normalized_box(box)
        x1, x2 = max(0, min(self.width, x1)), max(0, min(self.width, x2))
        y1, y2 = max(0, min(self.height, y1)), max(0, min(self.height, y2))
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        if x1 < x2 and y1 < y2:
            mask[y1:y2, x1:x2] = 255
        if feather > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), max(0.1, float(feather)))
        self.apply_selection_mask(mask, mode)

    def set_ellipse_selection(self, box: tuple[int, int, int, int], mode: str = "replace", feather: int = 0, antialias: bool = True) -> None:
        x1, y1, x2, y2 = normalized_box(box)
        x1, x2 = max(0, min(self.width, x1)), max(0, min(self.width, x2))
        y1, y2 = max(0, min(self.height, y1)), max(0, min(self.height, y2))
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        if x1 < x2 and y1 < y2:
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1, lineType=cv2.LINE_AA if antialias else cv2.LINE_8)
        if feather > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), max(0.1, float(feather)))
        self.apply_selection_mask(mask, mode)

    def set_polygon_selection(self, points: list[tuple[int, int]], mode: str = "replace", feather: int = 0, antialias: bool = True) -> None:
        if len(points) < 3:
            return
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        pts = np.array([[(max(0, min(self.width - 1, int(x))), max(0, min(self.height - 1, int(y)))) for x, y in points]], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255, lineType=cv2.LINE_AA if antialias else cv2.LINE_8)
        if feather > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), max(0.1, float(feather)))
        self.apply_selection_mask(mask, mode)

    def set_single_row_selection(self, y: int, mode: str = "replace") -> None:
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        if 0 <= y < self.height:
            mask[y : y + 1, :] = 255
        self.apply_selection_mask(mask, mode)

    def set_single_column_selection(self, x: int, mode: str = "replace") -> None:
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        if 0 <= x < self.width:
            mask[:, x : x + 1] = 255
        self.apply_selection_mask(mask, mode)

    def magic_wand_selection(
        self, layer: Layer, x: int, y: int, tolerance: int, mode: str = "replace",
        contiguous: bool = True, antialias: bool = True, sample_all_layers: bool = False,
    ) -> None:
        source = self.composite(False) if sample_all_layers else layer.pixels
        origin_x, origin_y = (0, 0) if sample_all_layers else (layer.x, layer.y)
        lx, ly = int(x) - origin_x, int(y) - origin_y
        if lx < 0 or ly < 0 or lx >= source.shape[1] or ly >= source.shape[0]:
            return
        seed = tuple(int(value) for value in source[ly, lx])
        soft = perceptual_color_mask(source, [seed], tolerance, antialias)
        candidates = (soft > 0).astype(np.uint8)
        if contiguous:
            if candidates[ly, lx] == 0:
                local = np.zeros_like(soft)
            else:
                cv2.floodFill(candidates, None, (lx, ly), 2, flags=4)
                local = np.where(candidates == 2, soft, 0).astype(np.uint8)
        else:
            local = soft
        mask = local if sample_all_layers else self._layer_mask_to_document(layer, local)
        self.apply_selection_mask(mask, mode)

    def color_range_selection(
        self, layer: Layer, x: int, y: int, tolerance: int, mode: str = "replace",
        samples: list[tuple[int, int, int, int]] | None = None,
        excluded: list[tuple[int, int, int, int]] | None = None,
        antialias: bool = True, sample_all_layers: bool = False,
    ) -> None:
        source = self.composite(False) if sample_all_layers else layer.pixels
        origin_x, origin_y = (0, 0) if sample_all_layers else (layer.x, layer.y)
        lx, ly = int(x) - origin_x, int(y) - origin_y
        if lx < 0 or ly < 0 or lx >= source.shape[1] or ly >= source.shape[0]:
            return
        included = list(samples or [tuple(int(value) for value in source[ly, lx])])
        local = combine_sample_masks(source, included, list(excluded or []), tolerance, antialias)
        mask = local if sample_all_layers else self._layer_mask_to_document(layer, local)
        self.apply_selection_mask(mask, mode)

    def _quick_selection_mask(
        self,
        layer: Layer,
        points: list[tuple[int, int]],
        radius: int,
        tolerance: int,
        smooth: int = 0,
        edge_radius: int = 0,
        edge_strength: float = 0.0,
    ) -> np.ndarray:
        local_union = np.zeros(layer.pixels.shape[:2], dtype=np.uint8)
        radius = max(1, int(radius))
        tolerance = max(0, int(tolerance))
        cache_key = (layer.id, layer.pixels_revision, layer.pixels.shape[:2])
        cache = getattr(self, "_quick_selection_analysis", None)
        if cache is None or cache[0] != cache_key:
            rgb = layer.pixels[:, :, :3]
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            edge_map = cv2.convertScaleAbs(cv2.magnitude(gradient_x, gradient_y))
            self._quick_selection_analysis = (cache_key, lab, edge_map)
        else:
            _key, lab, edge_map = cache
        edge_limit = 110.0 - float(np.clip(edge_strength, 0.0, 1.0)) * 78.0
        for x, y in points:
            lx, ly = int(x) - layer.x, int(y) - layer.y
            if lx < 0 or ly < 0 or lx >= layer.pixels.shape[1] or ly >= layer.pixels.shape[0]:
                continue
            x1, y1 = max(0, lx - radius), max(0, ly - radius)
            x2, y2 = min(layer.pixels.shape[1], lx + radius + 1), min(layer.pixels.shape[0], ly + radius + 1)
            sample = layer.pixels[y1:y2, x1:x2]
            opaque = sample[:, :, 3] > 0
            if np.any(opaque):
                seed = np.median(lab[y1:y2, x1:x2][opaque], axis=0)
            else:
                seed = lab[ly, lx]
            gate_radius = radius * 3
            gx1, gy1 = max(0, lx - gate_radius), max(0, ly - gate_radius)
            gx2 = min(layer.pixels.shape[1], lx + gate_radius + 1)
            gy2 = min(layer.pixels.shape[0], ly + gate_radius + 1)
            region = layer.pixels[gy1:gy2, gx1:gx2]
            color_delta = lab[gy1:gy2, gx1:gx2].astype(np.int16) - seed.astype(np.int16)
            color_distance = np.sqrt(np.sum(color_delta.astype(np.float32) ** 2, axis=2))
            perceptual_limit = max(1.0, float(tolerance) * 1.8)
            local_edges = edge_map[gy1:gy2, gx1:gx2]
            candidates = ((color_distance <= perceptual_limit) & (local_edges <= edge_limit) & (region[:, :, 3] > 0)).astype(np.uint8)
            candidates[ly - gy1, lx - gx1] = 1
            _, labels, _, _ = cv2.connectedComponentsWithStats(candidates, 4)
            rx, ry = lx - gx1, ly - gy1
            label = labels[ry, rx]
            if label == 0 and candidates[ry, rx] == 0:
                continue
            component = (labels == label).astype(np.uint8) * 255
            brush_gate = np.zeros_like(component)
            cv2.circle(brush_gate, (rx, ry), gate_radius, 255, -1)
            gated = np.where(brush_gate > 0, component, 0).astype(np.uint8)
            local_union[gy1:gy2, gx1:gx2] = np.maximum(local_union[gy1:gy2, gx1:gx2], gated)
        if np.any(local_union):
            local_union = refine_selection_mask(local_union, max(0, int(smooth)), 0, 1.0, 0)
            edge_radius = max(0, int(edge_radius))
            edge_strength = float(np.clip(edge_strength, 0.0, 1.0))
            if edge_radius > 0 and edge_strength > 0.0:
                local_union = correct_selection_edges(local_union, layer.pixels, edge_radius, edge_strength, 96)
        return self._layer_mask_to_document(layer, local_union)

    def preview_quick_selection_brush(
        self,
        layer: Layer,
        points: list[tuple[int, int]],
        radius: int,
        tolerance: int,
        mode: str = "replace",
        smooth: int = 0,
        edge_radius: int = 0,
        edge_strength: float = 0.0,
    ) -> np.ndarray | None:
        mask = self._quick_selection_mask(layer, points, radius, tolerance, smooth, edge_radius, edge_strength)
        if not np.any(mask):
            return None if self.selection_mask is None else self.selection_mask.copy()
        current = self.selection_mask
        if mode == "add" and current is not None:
            return np.maximum(current, mask)
        if mode == "subtract" and current is not None:
            result = np.clip(current.astype(np.float32) * (1.0 - mask.astype(np.float32) / 255.0), 0, 255).astype(np.uint8)
            return result if np.any(result) else None
        if mode == "intersect" and current is not None:
            result = np.minimum(current, mask)
            return result if np.any(result) else None
        return mask

    def quick_selection_brush(
        self,
        layer: Layer,
        points: list[tuple[int, int]],
        radius: int,
        tolerance: int,
        mode: str = "replace",
        smooth: int = 0,
        edge_radius: int = 0,
        edge_strength: float = 0.0,
    ) -> None:
        mask = self._quick_selection_mask(layer, points, radius, tolerance, smooth, edge_radius, edge_strength)
        if np.any(mask):
            self.apply_selection_mask(mask, mode)

    def magnetic_edge_map(self, composite: np.ndarray | None = None) -> np.ndarray:
        composite = self.composite(False) if composite is None else composite
        gray = cv2.cvtColor(composite[:, :, :3], cv2.COLOR_RGB2GRAY)
        gray = np.where(composite[:, :, 3] > 0, gray, 0).astype(np.uint8)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 140)
        grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        if float(magnitude.max()) > 0.0:
            magnitude = (magnitude / magnitude.max() * 255.0).astype(np.uint8)
        else:
            magnitude = np.zeros_like(edges)
        return np.maximum(edges, magnitude)

    def snap_point_to_edge(self, point: tuple[int, int], edge_map: np.ndarray, radius: int = 14) -> tuple[int, int]:
        x = max(0, min(self.width - 1, int(point[0])))
        y = max(0, min(self.height - 1, int(point[1])))
        radius = max(1, int(radius))
        x1, y1 = max(0, x - radius), max(0, y - radius)
        x2, y2 = min(self.width, x + radius + 1), min(self.height, y + radius + 1)
        local = edge_map[y1:y2, x1:x2]
        if local.size == 0 or int(local.max()) <= 0:
            return x, y
        ys, xs = np.where(local > 0)
        if len(xs) == 0:
            return x, y
        doc_x = xs + x1
        doc_y = ys + y1
        distance = np.sqrt((doc_x - x) ** 2 + (doc_y - y) ** 2)
        score = local[ys, xs].astype(np.float32) - distance.astype(np.float32) * 4.0
        best = int(np.argmax(score))
        return int(doc_x[best]), int(doc_y[best])

    def select_opaque_pixels(self, layer: Layer, mode: str = "replace") -> None:
        local = (layer.pixels[:, :, 3] > 0).astype(np.uint8) * 255
        self.apply_selection_mask(self._layer_mask_to_document(layer, local), mode)

    def select_subject(self, layer: Layer, mode: str = "replace", sensitivity: float = 0.5) -> None:
        local = subject_selection_mask(layer.pixels, sensitivity)
        if np.any(local):
            self.apply_selection_mask(self._layer_mask_to_document(layer, local), mode)

    def select_background(self, layer: Layer, mode: str = "replace", sensitivity: float = 0.5) -> None:
        local = background_selection_mask(layer.pixels, sensitivity)
        if np.any(local):
            self.apply_selection_mask(self._layer_mask_to_document(layer, local), mode)

    def select_sky(self, layer: Layer, mode: str = "replace", sensitivity: float = 0.5) -> None:
        local = sky_selection_mask(layer.pixels, sensitivity)
        if np.any(local):
            self.apply_selection_mask(self._layer_mask_to_document(layer, local), mode)

    def save_selection(self, name: str) -> None:
        if self.selection_mask is None:
            return
        self.saved_selections[name] = self.selection_mask.copy()
        self.dirty = True

    def load_selection(self, name: str, mode: str = "replace") -> None:
        mask = self.saved_selections.get(name)
        if mask is not None:
            self.apply_selection_mask(mask.copy(), mode)

    def delete_saved_selection(self, name: str) -> None:
        if name in self.saved_selections:
            del self.saved_selections[name]
            self.dirty = True

    def _layer_mask_to_document(self, layer: Layer, local_mask: np.ndarray) -> np.ndarray:
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        paste_mask(mask, local_mask, layer.x, layer.y)
        return mask

    def apply_selection_mask(self, mask: np.ndarray, mode: str = "replace") -> None:
        mask = np.clip(mask, 0, 255).astype(np.uint8)
        if mode == "add" and self.selection_mask is not None:
            self.selection_mask = np.maximum(self.selection_mask, mask)
        elif mode == "subtract" and self.selection_mask is not None:
            self.selection_mask = np.clip(
                self.selection_mask.astype(np.float32) * (1.0 - mask.astype(np.float32) / 255.0),
                0,
                255,
            ).astype(np.uint8)
        elif mode == "intersect" and self.selection_mask is not None:
            self.selection_mask = np.minimum(self.selection_mask, mask)
        else:
            self.selection_mask = mask
        if self.selection_mask is not None and not np.any(self.selection_mask):
            self.selection_mask = None

    def clear_selection(self) -> None:
        self.selection_mask = None

    def select_all(self) -> None:
        self.selection_mask = np.full((self.height, self.width), 255, dtype=np.uint8)

    def invert_selection(self) -> None:
        if self.selection_mask is None:
            self.selection_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.selection_mask = 255 - self.selection_mask
        if not np.any(self.selection_mask):
            self.selection_mask = None

    def feather_selection(self, radius: int) -> None:
        if self.selection_mask is None:
            return
        radius = max(1, int(radius))
        k = radius * 2 + 1
        self.selection_mask = cv2.GaussianBlur(self.selection_mask, (k, k), radius)
        self.dirty = True

    def grow_selection(self, pixels: int) -> None:
        if self.selection_mask is None:
            return
        pixels = max(1, int(pixels))
        kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), dtype=np.uint8)
        self.selection_mask = cv2.dilate(self.selection_mask, kernel)
        self.dirty = True

    def shrink_selection(self, pixels: int) -> None:
        if self.selection_mask is None:
            return
        pixels = max(1, int(pixels))
        kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), dtype=np.uint8)
        self.selection_mask = cv2.erode(self.selection_mask, kernel)
        if not np.any(self.selection_mask):
            self.selection_mask = None
        self.dirty = True

    def smooth_selection(self, radius: int) -> None:
        if self.selection_mask is None:
            return
        radius = max(1, int(radius))
        k = radius * 2 + 1
        mask = cv2.GaussianBlur(self.selection_mask, (k, k), radius)
        self.selection_mask = np.where(mask >= 128, 255, 0).astype(np.uint8)
        if not np.any(self.selection_mask):
            self.selection_mask = None
        self.dirty = True

    def border_selection(self, width: int) -> None:
        if self.selection_mask is None:
            return
        width = max(1, int(width))
        kernel = np.ones((width * 2 + 1, width * 2 + 1), dtype=np.uint8)
        outer = cv2.dilate(self.selection_mask, kernel)
        inner = cv2.erode(self.selection_mask, kernel)
        border = np.clip(outer.astype(np.int16) - inner.astype(np.int16), 0, 255).astype(np.uint8)
        self.selection_mask = border if np.any(border) else None
        self.dirty = True

    def refine_selection(self, smooth: int = 0, feather: int = 0, contrast: float = 1.0, shift: int = 0) -> None:
        if self.selection_mask is None:
            return
        mask = refine_selection_mask(self.selection_mask, smooth, feather, contrast, shift)
        self.selection_mask = mask if np.any(mask) else None
        self.dirty = True

    def cleanup_selection_edges(self, radius: int = 3, strength: float = 0.7) -> None:
        if self.selection_mask is None:
            return
        mask = cleanup_selection_edges(self.selection_mask, self.composite(False), radius, strength)
        self.selection_mask = mask if np.any(mask) else None
        self.dirty = True

    def correct_selection_edges(self, radius: int = 3, strength: float = 0.65, threshold: int = 96) -> None:
        if self.selection_mask is None:
            return
        mask = correct_selection_edges(self.selection_mask, self.composite(False), radius, strength, threshold)
        self.selection_mask = mask if np.any(mask) else None
        self.dirty = True
