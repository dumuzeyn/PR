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


class TransformsDocumentMixin:
    def transform_active_layer(
        self,
        x: int | None = None,
        y: int | None = None,
        width: int | None = None,
        height: int | None = None,
        angle: float = 0.0,
        flip_horizontal: bool = False,
        flip_vertical: bool = False,
    ) -> None:
        layer = self.layer
        if layer.locked:
            return
        if x is not None:
            layer.x = int(x)
        if y is not None:
            layer.y = int(y)
        if layer.kind in {"linked", "embedded"}:
            if layer.smart_source is None:
                layer.smart_source = layer.pixels.copy()
            data = dict(layer.smart_data or {})
            transform = dict(data.get("transform") or {})
            current_w = int(transform.get("width", layer.pixels.shape[1]))
            current_h = int(transform.get("height", layer.pixels.shape[0]))
            center_x = layer.x + layer.pixels.shape[1] / 2.0
            center_y = layer.y + layer.pixels.shape[0] / 2.0
            transform.update(
                {
                    "width": max(1, int(width if width is not None else current_w)),
                    "height": max(1, int(height if height is not None else current_h)),
                    "angle": float(transform.get("angle", 0.0)) + float(angle),
                    "flip_horizontal": bool(transform.get("flip_horizontal", False)) ^ bool(flip_horizontal),
                    "flip_vertical": bool(transform.get("flip_vertical", False)) ^ bool(flip_vertical),
                }
            )
            data["transform"] = transform
            layer.smart_data = data
            render_smart_object(layer)
            if x is None:
                layer.x = round(center_x - layer.pixels.shape[1] / 2.0)
            if y is None:
                layer.y = round(center_y - layer.pixels.shape[0] / 2.0)
            if layer.mask is not None and layer.mask.shape != layer.pixels.shape[:2]:
                layer.mask = cv2.resize(layer.mask, (layer.pixels.shape[1], layer.pixels.shape[0]), interpolation=cv2.INTER_NEAREST)
            layer.touch_pixels()
            self.dirty = True
            return
        target_w = max(1, int(width or layer.pixels.shape[1]))
        target_h = max(1, int(height or layer.pixels.shape[0]))
        if (target_w, target_h) != (layer.pixels.shape[1], layer.pixels.shape[0]):
            layer.pixels = cv2.resize(layer.pixels, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            if layer.mask is not None:
                layer.mask = cv2.resize(layer.mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        if flip_horizontal:
            layer.pixels = cv2.flip(layer.pixels, 1)
            if layer.mask is not None:
                layer.mask = cv2.flip(layer.mask, 1)
        if flip_vertical:
            layer.pixels = cv2.flip(layer.pixels, 0)
            if layer.mask is not None:
                layer.mask = cv2.flip(layer.mask, 0)
        if abs(float(angle)) > 0.001:
            center_x = layer.x + layer.pixels.shape[1] / 2.0
            center_y = layer.y + layer.pixels.shape[0] / 2.0
            layer.pixels = rotate_bound(layer.pixels, float(angle), cv2.INTER_CUBIC)
            if layer.mask is not None:
                layer.mask = rotate_bound(layer.mask, float(angle), cv2.INTER_LINEAR)
            layer.x = round(center_x - layer.pixels.shape[1] / 2.0)
            layer.y = round(center_y - layer.pixels.shape[0] / 2.0)
        self.dirty = True

    def transform_selected_pixels(
        self,
        x: int | None = None,
        y: int | None = None,
        width: int | None = None,
        height: int | None = None,
        angle: float = 0.0,
        flip_horizontal: bool = False,
        flip_vertical: bool = False,
    ) -> bool:
        layer = self.layer
        if layer.locked:
            return False
        selection = self.layer_selection_mask(layer)
        if selection is None or not np.any(selection):
            return False
        ys, xs = np.where(selection > 0)
        lx1, ly1, lx2, ly2 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
        patch = layer.pixels[ly1:ly2, lx1:lx2].copy()
        patch_mask = selection[ly1:ly2, lx1:lx2].astype(np.float32) / 255.0
        patch[:, :, 3] = np.clip(patch[:, :, 3].astype(np.float32) * patch_mask, 0, 255).astype(np.uint8)

        original_region = layer.pixels[ly1:ly2, lx1:lx2]
        keep_alpha = 1.0 - patch_mask
        original_region[:, :, 3] = np.clip(original_region[:, :, 3].astype(np.float32) * keep_alpha, 0, 255).astype(np.uint8)

        dest_x = int(x if x is not None else layer.x + lx1)
        dest_y = int(y if y is not None else layer.y + ly1)
        target_w = max(1, int(width or patch.shape[1]))
        target_h = max(1, int(height or patch.shape[0]))
        if (target_w, target_h) != (patch.shape[1], patch.shape[0]):
            patch = cv2.resize(patch, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        if flip_horizontal:
            patch = cv2.flip(patch, 1)
        if flip_vertical:
            patch = cv2.flip(patch, 0)
        if abs(float(angle)) > 0.001:
            center_x = dest_x + patch.shape[1] / 2.0
            center_y = dest_y + patch.shape[0] / 2.0
            patch = rotate_bound(patch, float(angle), cv2.INTER_CUBIC)
            dest_x = round(center_x - patch.shape[1] / 2.0)
            dest_y = round(center_y - patch.shape[0] / 2.0)

        old_x, old_y = layer.x, layer.y
        old_h, old_w = layer.pixels.shape[:2]
        new_x = min(old_x, dest_x)
        new_y = min(old_y, dest_y)
        new_right = max(old_x + old_w, dest_x + patch.shape[1])
        new_bottom = max(old_y + old_h, dest_y + patch.shape[0])
        new_pixels = blank_rgba(new_right - new_x, new_bottom - new_y, (0, 0, 0, 0))
        new_pixels[old_y - new_y : old_y - new_y + old_h, old_x - new_x : old_x - new_x + old_w] = layer.pixels
        alpha_blend_inplace(new_pixels, patch, dest_x - new_x, dest_y - new_y, 1.0)

        if layer.mask is not None:
            new_mask = np.zeros(new_pixels.shape[:2], dtype=np.uint8)
            paste_mask(new_mask, layer.mask, old_x - new_x, old_y - new_y)
            layer.mask = new_mask

        new_selection = np.zeros((self.height, self.width), dtype=np.uint8)
        transformed_alpha = np.where(patch[:, :, 3] > 0, 255, 0).astype(np.uint8)
        paste_mask(new_selection, transformed_alpha, dest_x, dest_y)
        self.selection_mask = new_selection if np.any(new_selection) else None
        layer.pixels = new_pixels
        layer.x = new_x
        layer.y = new_y
        self.dirty = True
        return True

    def transform_selected_pixels_advanced(
        self,
        mode: str,
        points: list[tuple[float, float]] | list[list[float]],
        rows: int = 4,
        columns: int = 4,
    ) -> bool:
        layer = self.layer
        if layer.locked:
            return False
        selection = self.layer_selection_mask(layer)
        if selection is None or not np.any(selection):
            return False
        ys, xs = np.where(selection > 0)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
        patch = layer.pixels[y1:y2, x1:x2].copy()
        patch_mask = selection[y1:y2, x1:x2].astype(np.float32) / 255.0
        patch[:, :, 3] = np.clip(patch[:, :, 3].astype(np.float32) * patch_mask, 0, 255).astype(np.uint8)
        original_region = layer.pixels[y1:y2, x1:x2]
        original_region[:, :, 3] = np.clip(original_region[:, :, 3].astype(np.float32) * (1.0 - patch_mask), 0, 255).astype(np.uint8)
        if str(mode).lower() == "mesh":
            transformed, offset = mesh_warp_pixels(patch, points, rows, columns, cv2.INTER_CUBIC)
        else:
            transformed, offset = perspective_warp_pixels(patch, points, cv2.INTER_CUBIC)
        destination_x, destination_y = offset
        old_x, old_y = layer.x, layer.y
        old_height, old_width = layer.pixels.shape[:2]
        new_x = min(old_x, destination_x)
        new_y = min(old_y, destination_y)
        new_right = max(old_x + old_width, destination_x + transformed.shape[1])
        new_bottom = max(old_y + old_height, destination_y + transformed.shape[0])
        new_pixels = blank_rgba(new_right - new_x, new_bottom - new_y, (0, 0, 0, 0))
        new_pixels[old_y - new_y:old_y - new_y + old_height, old_x - new_x:old_x - new_x + old_width] = layer.pixels
        alpha_blend_inplace(new_pixels, transformed, destination_x - new_x, destination_y - new_y, 1.0)
        if layer.mask is not None:
            new_mask = np.zeros(new_pixels.shape[:2], dtype=np.uint8)
            paste_mask(new_mask, layer.mask, old_x - new_x, old_y - new_y)
            layer.mask = new_mask
        layer.pixels = new_pixels
        layer.x, layer.y = new_x, new_y
        new_selection = np.zeros((self.height, self.width), dtype=np.uint8)
        paste_mask(new_selection, transformed[:, :, 3], destination_x, destination_y)
        self.selection_mask = new_selection if np.any(new_selection) else None
        layer.touch_pixels()
        self.dirty = True
        return True

    def perspective_transform_active_layer(self, corners: list[tuple[float, float]] | tuple[tuple[float, float], ...]) -> None:
        layer = self.layer
        if layer.locked:
            return
        if len(corners) != 4:
            raise ValueError("Perspective transform needs four destination corners.")
        h, w = layer.pixels.shape[:2]
        dst_doc = np.array(corners, dtype=np.float32)
        min_x = math.floor(float(dst_doc[:, 0].min()))
        min_y = math.floor(float(dst_doc[:, 1].min()))
        max_x = math.ceil(float(dst_doc[:, 0].max()))
        max_y = math.ceil(float(dst_doc[:, 1].max()))
        out_w = max(1, max_x - min_x)
        out_h = max(1, max_y - min_y)
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        dst = dst_doc - np.array([min_x, min_y], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)
        layer.pixels = cv2.warpPerspective(
            layer.pixels,
            matrix,
            (out_w, out_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        if layer.mask is not None:
            layer.mask = cv2.warpPerspective(
                layer.mask,
                matrix,
                (out_w, out_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        layer.x = min_x
        layer.y = min_y
        self.dirty = True

    def warp_active_layer(self, mode: str, amount: float = 0.35, wavelength: float = 96.0) -> None:
        layer = self.layer
        if layer.locked:
            return
        layer.pixels = warp_pixels(layer.pixels, mode, amount, wavelength, cv2.INTER_CUBIC)
        if layer.mask is not None:
            layer.mask = warp_pixels(layer.mask, mode, amount, wavelength, cv2.INTER_LINEAR)
        self.dirty = True

    def set_active_layer_advanced_transform(
        self,
        mode: str,
        points: list[tuple[float, float]] | list[list[float]],
        rows: int = 4,
        columns: int = 4,
    ) -> None:
        layer = self.layer
        if layer.locked:
            return
        mode = str(mode).lower().strip()
        if mode not in {"perspective", "mesh"}:
            raise ValueError("Advanced transform mode must be perspective or mesh")
        expected = 4 if mode == "perspective" else max(2, int(rows)) * max(2, int(columns))
        if len(points) != expected:
            raise ValueError(f"Advanced transform needs {expected} points")
        existing = layer.transform_data or {}
        if layer.transform_source is None or not existing:
            render_height, render_width = layer.pixels.shape[:2]
            visible = layer.pixels[:, :, 3] > 0
            if np.any(visible):
                ys, xs = np.where(visible)
                crop = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
            else:
                crop = (0, 0, render_width, render_height)
            x1, y1, x2, y2 = crop
            layer.transform_source = layer.pixels[y1:y2, x1:x2].copy()
            if layer.mask is None and layer.kind == "adjustment":
                layer.transform_mask_source = np.full((y2 - y1, x2 - x1), 255, dtype=np.uint8)
            else:
                layer.transform_mask_source = None if layer.mask is None else layer.mask[y1:y2, x1:x2].copy()
            base_x, base_y = int(layer.x + x1), int(layer.y + y1)
            render_base_x, render_base_y = int(layer.x), int(layer.y)
        else:
            base_x = int(existing.get("base_x", layer.x))
            base_y = int(existing.get("base_y", layer.y))
            render_width = int(existing.get("render_width", layer.transform_source.shape[1]))
            render_height = int(existing.get("render_height", layer.transform_source.shape[0]))
            crop = tuple(int(value) for value in existing.get("source_crop", [0, 0, render_width, render_height]))
            render_base_x = int(existing.get("render_base_x", base_x - crop[0]))
            render_base_y = int(existing.get("render_base_y", base_y - crop[1]))
        local_points = [[float(point[0]) - base_x, float(point[1]) - base_y] for point in points]
        layer.transform_data = {
            "mode": mode,
            "points": local_points,
            "rows": max(2, int(rows)),
            "columns": max(2, int(columns)),
            "base_x": base_x,
            "base_y": base_y,
            "source_width": int(layer.transform_source.shape[1]),
            "source_height": int(layer.transform_source.shape[0]),
            "render_width": render_width,
            "render_height": render_height,
            "source_crop": list(crop),
            "render_base_x": render_base_x,
            "render_base_y": render_base_y,
            "mask_was_none": bool(existing.get("mask_was_none", layer.mask is None)),
        }
        apply_saved_layer_transform(layer)
        layer.touch_pixels()
        self.dirty = True

    def reset_active_layer_advanced_transform(self) -> bool:
        layer = self.layer
        if layer.locked or layer.transform_data is None:
            return False
        data = layer.transform_data
        base_x = int(data.get("base_x", layer.x))
        base_y = int(data.get("base_y", layer.y))
        source = None if layer.transform_source is None else layer.transform_source.copy()
        mask_source = None if layer.transform_mask_source is None else layer.transform_mask_source.copy()
        layer.transform_data = None
        layer.transform_source = None
        layer.transform_mask_source = None
        render_base_x = int(data.get("render_base_x", base_x))
        render_base_y = int(data.get("render_base_y", base_y))
        layer.x, layer.y = render_base_x, render_base_y
        if layer.kind == "text" and layer.text_data is not None:
            layer.pixels = blank_rgba(int(data.get("render_width", self.width)), int(data.get("render_height", self.height)), (0, 0, 0, 0))
            render_text_layer(layer)
        elif layer.kind == "shape" and layer.shape_data is not None:
            layer.pixels = blank_rgba(int(data.get("render_width", self.width)), int(data.get("render_height", self.height)), (0, 0, 0, 0))
            render_shape_layer(layer)
        elif layer.kind in {"linked", "embedded"} and layer.smart_source is not None:
            render_smart_object(layer)
        elif source is not None:
            layer.pixels = source
            layer.mask = None if bool(data.get("mask_was_none", False)) else mask_source
            layer.x, layer.y = base_x, base_y
        layer.touch_pixels()
        self.dirty = True
        return True
