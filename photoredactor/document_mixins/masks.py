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


class MasksDocumentMixin:
    def selection_bounds(self) -> tuple[int, int, int, int] | None:
        if self.selection_mask is None or not np.any(self.selection_mask):
            return None
        ys, xs = np.where(self.selection_mask > 0)
        return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)

    def layer_selection_mask(self, layer: Layer) -> np.ndarray | None:
        if self.selection_mask is None:
            return None
        mask = np.zeros(layer.pixels.shape[:2], dtype=np.uint8)
        x1, y1 = max(0, layer.x), max(0, layer.y)
        x2 = min(self.width, layer.x + layer.pixels.shape[1])
        y2 = min(self.height, layer.y + layer.pixels.shape[0])
        if x1 >= x2 or y1 >= y2:
            return mask
        lx1, ly1 = x1 - layer.x, y1 - layer.y
        mask[ly1 : ly1 + (y2 - y1), lx1 : lx1 + (x2 - x1)] = self.selection_mask[y1:y2, x1:x2]
        return mask

    def add_reveal_all_mask(self) -> None:
        layer = self.layer
        layer.mask = np.full(layer.pixels.shape[:2], 255, dtype=np.uint8)
        layer.mask_enabled = True
        layer.mask_linked = True

    def add_hide_all_mask(self) -> None:
        layer = self.layer
        layer.mask = np.zeros(layer.pixels.shape[:2], dtype=np.uint8)
        layer.mask_enabled = True
        layer.mask_linked = True

    def add_mask_from_selection(self) -> None:
        layer = self.layer
        mask = self.layer_selection_mask(layer)
        if mask is None:
            self.add_reveal_all_mask()
        else:
            layer.mask = mask
            layer.mask_enabled = True
            layer.mask_linked = True

    def invert_active_mask(self) -> None:
        layer = self.layer
        if layer.mask is not None:
            layer.mask = 255 - layer.mask

    def toggle_active_mask(self) -> None:
        layer = self.layer
        if layer.mask is not None:
            layer.mask_enabled = not layer.mask_enabled

    def toggle_active_mask_link(self) -> None:
        layer = self.layer
        if layer.mask is not None:
            layer.mask_linked = not layer.mask_linked
            self.dirty = True

    def set_active_mask_density(self, density: float) -> None:
        layer = self.layer
        if layer.mask is not None:
            layer.mask_density = float(np.clip(density, 0.0, 1.0))
            self.dirty = True

    def set_active_mask_feather(self, radius: float) -> None:
        layer = self.layer
        if layer.mask is not None:
            layer.mask_feather = max(0.0, float(radius))
            self.dirty = True

    def preview_active_mask_refinement(
        self,
        smooth: int = 0,
        feather: int = 0,
        contrast: float = 1.0,
        shift: int = 0,
        edge_radius: int = 0,
        edge_strength: float = 0.0,
        confidence_threshold: int = 96,
    ) -> np.ndarray | None:
        layer = self.layer
        if layer.mask is None:
            return None
        return refine_layer_mask(
            layer.mask,
            layer.pixels,
            smooth,
            feather,
            contrast,
            shift,
            edge_radius,
            edge_strength,
            confidence_threshold,
        )

    def refine_active_mask(
        self,
        smooth: int = 0,
        feather: int = 0,
        contrast: float = 1.0,
        shift: int = 0,
        edge_radius: int = 0,
        edge_strength: float = 0.0,
        confidence_threshold: int = 96,
    ) -> None:
        mask = self.preview_active_mask_refinement(
            smooth,
            feather,
            contrast,
            shift,
            edge_radius,
            edge_strength,
            confidence_threshold,
        )
        if mask is not None:
            self.layer.mask = mask
            self.dirty = True

    def delete_active_mask(self) -> None:
        self.layer.mask = None
        self.layer.mask_linked = True

    def apply_active_mask(self) -> None:
        layer = self.layer
        if layer.mask is None:
            return
        mask = effective_layer_mask(layer)
        density = float(np.clip(layer.mask_density, 0.0, 1.0))
        alpha = ((1.0 - density) + (mask.astype(np.float32) / 255.0) * density).clip(0, 1)
        layer.pixels[:, :, 3] = np.clip(layer.pixels[:, :, 3].astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        layer.mask = None
        layer.mask_feather = 0.0

    def patch_active_selection(
        self,
        source_x: int,
        source_y: int,
        heal: bool = True,
        color_adaptation: float = 0.85,
        texture_strength: float = 1.0,
    ) -> bool:
        layer = self.layer
        if layer.locked:
            return False
        selection = self.layer_selection_mask(layer)
        if selection is None or not np.any(selection):
            return False
        ys, xs = np.where(selection > 0)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
        w, h = x2 - x1, y2 - y1
        sx1 = int(source_x) - layer.x
        sy1 = int(source_y) - layer.y
        sx2, sy2 = sx1 + w, sy1 + h
        if sx1 < 0 or sy1 < 0 or sx2 > layer.pixels.shape[1] or sy2 > layer.pixels.shape[0]:
            return False
        src = layer.pixels[sy1:sy2, sx1:sx2].astype(np.float32)
        dst = layer.pixels[y1:y2, x1:x2].astype(np.float32)
        mask = selection[y1:y2, x1:x2].astype(np.float32) / 255.0
        edited = src.copy()
        active = mask > 0
        if heal and np.any(active):
            adaptation = float(np.clip(color_adaptation, 0.0, 1.0))
            texture_strength = float(np.clip(texture_strength, 0.0, 2.0))
            binary = (selection > 0).astype(np.uint8)
            ring_radius = max(2, min(32, int(round(max(w, h) * 0.18))))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_radius * 2 + 1, ring_radius * 2 + 1))
            ring = (cv2.dilate(binary, kernel) > 0) & (binary == 0)
            ring_values = layer.pixels[ring, :3].astype(np.float32)
            source_values = src[active, :3]
            if ring_values.size:
                target_mean = np.median(ring_values, axis=0)
                target_std = np.maximum(np.std(ring_values, axis=0), 4.0)
            else:
                target_mean = dst[active, :3].mean(axis=0)
                target_std = np.maximum(dst[active, :3].std(axis=0), 4.0)
            source_mean = source_values.mean(axis=0)
            source_std = np.maximum(source_values.std(axis=0), 4.0)
            matched = (src[:, :, :3] - source_mean) * (target_std / source_std) * texture_strength + target_mean
            edited[:, :, :3] = src[:, :, :3] * (1.0 - adaptation) + matched * adaptation
            edited[:, :, 3] = dst[:, :, 3]
        mixed = edited * mask[:, :, None] + dst * (1.0 - mask[:, :, None])
        layer.pixels[y1:y2, x1:x2] = np.clip(mixed, 0, 255).astype(np.uint8)
        self.dirty = True
        return True
