from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from typing import Iterable

import cv2
import numpy as np

from .core import (
    Document,
    Layer,
    alpha_blend_inplace,
    apply_adjustment_layer,
    apply_filter_stack,
    blank_rgba,
    checker_background,
    checker_region,
    effective_layer_mask,
    render_layer_effects,
)
from .performance import PerformanceProfiler, profiler
from .large_document import MipmapPyramid, ScratchCache, gpu_status


Rect = tuple[int, int, int, int]
Tile = tuple[int, int]


class RenderEngine:
    """Caches layer work and recomposites only tiles invalidated by local edits."""

    def __init__(self, tile_size: int = 256, performance: PerformanceProfiler | None = None) -> None:
        self.tile_size = max(64, int(tile_size))
        self.profiler = performance or profiler
        self._document: Document | None = None
        self._composites: dict[bool, np.ndarray] = {}
        self._reduced_composites: dict[tuple[bool, int], np.ndarray] = {}
        self._dirty_tiles: dict[bool, set[Tile]] = {False: set(), True: set()}
        self._reduced_dirty_tiles: dict[bool, set[Tile]] = {False: set(), True: set()}
        cache_mb = max(16, int(os.environ.get("PHOTO_REDACTOR_CACHE_MB", "256")))
        self.scratch = ScratchCache(memory_limit=cache_mb * 1024 * 1024)
        self.mipmaps = MipmapPyramid(self.scratch)
        self.gpu = gpu_status()
        self._filtered_cache: dict[str, tuple[object, ...]] = {}
        self._mask_cache: dict[str, tuple[tuple[object, ...], np.ndarray | None]] = {}
        self._effects_cache: dict[str, tuple[tuple[object, ...], list[tuple[np.ndarray, int, int, float, str]]]] = {}
        self._filter_dirty: dict[str, Rect] = {}
        self._mask_dirty: dict[str, Rect] = {}
        self._last_changed_tiles: set[Tile] = set()
        self._render_revision = 0

    def ensure_document(self, document: Document) -> None:
        if self._document is document:
            return
        self._document = document
        self._composites.clear()
        self._reduced_composites.clear()
        self._dirty_tiles = {False: set(), True: set()}
        self._reduced_dirty_tiles = {False: set(), True: set()}
        self.clear_layer_caches()
        self._last_changed_tiles.clear()

    def clear_layer_caches(self) -> None:
        self._filtered_cache.clear()
        self.scratch.clear()
        self._mask_cache.clear()
        self._effects_cache.clear()
        self._filter_dirty.clear()
        self._mask_dirty.clear()

    def invalidate_full(self, document: Document, clear_layer_caches: bool = True) -> None:
        self.ensure_document(document)
        self._composites.clear()
        self._reduced_composites.clear()
        self._dirty_tiles = {False: set(), True: set()}
        self._reduced_dirty_tiles = {False: set(), True: set()}
        self._last_changed_tiles.clear()
        if clear_layer_caches:
            self.clear_layer_caches()
        self.profiler.count("render.invalidate_full")

    def invalidate_region(self, document: Document, rect: Rect, layer: Layer | None = None, kind: str = "pixels") -> None:
        self.ensure_document(document)
        source_rect = self._clip_rect(document, rect)
        display_rect = rect
        if layer is not None and kind != "mask" and layer.filters:
            display_rect = self._expand_rect(display_rect, self._filter_halo(layer.filters))
        if layer is not None and kind == "mask" and layer.mask_feather > 0:
            display_rect = self._expand_rect(display_rect, max(1, int(round(layer.mask_feather)) * 2))
        clipped = self._clip_rect(document, self._expand_for_effects(display_rect, layer))
        if clipped is None:
            return
        if layer is not None:
            if kind == "transform":
                pass
            elif kind == "mask":
                layer.touch_mask()
                if source_rect is not None:
                    local = (
                        source_rect[0] - layer.x,
                        source_rect[1] - layer.y,
                        source_rect[2] - layer.x,
                        source_rect[3] - layer.y,
                    )
                    self._mask_dirty[layer.id] = self._union_rect(self._mask_dirty.get(layer.id), local)
            else:
                layer.touch_pixels()
                if source_rect is not None:
                    local = (
                        source_rect[0] - layer.x,
                        source_rect[1] - layer.y,
                        source_rect[2] - layer.x,
                        source_rect[3] - layer.y,
                    )
                    self._filter_dirty[layer.id] = self._union_rect(self._filter_dirty.get(layer.id), local)
            self._effects_cache.pop(layer.id, None)
        tiles = set(self.tiles_for_rect(clipped))
        self._dirty_tiles[False].update(tiles)
        self._dirty_tiles[True].update(tiles)
        self._reduced_dirty_tiles[False].update(tiles)
        self._reduced_dirty_tiles[True].update(tiles)
        self.profiler.count("render.invalidate_tile", len(tiles))

    def render(self, document: Document, checker: bool = False) -> np.ndarray:
        self.ensure_document(document)
        cached = self._composites.get(checker)
        if cached is None or cached.shape[:2] != (document.height, document.width):
            with self.profiler.measure("render.composite.full"):
                cached = self.composite_region(document, (0, 0, document.width, document.height), checker)
            self._composites[checker] = cached
            self._dirty_tiles[checker].clear()
            self._last_changed_tiles = set(self.all_tiles(document))
            self.profiler.count("render.full")
            self._render_revision += 1
            return cached

        dirty = self._dirty_tiles[checker]
        self._last_changed_tiles = set(dirty)
        if dirty:
            with self.profiler.measure("render.composite.dirty_tiles"):
                for tx, ty in sorted(dirty):
                    rect = self.tile_rect(document, tx, ty)
                    x1, y1, x2, y2 = rect
                    cached[y1:y2, x1:x2] = self.composite_region(document, rect, checker)
            self.profiler.count("render.partial")
            self.profiler.count("render.tiles_composited", len(dirty))
            dirty.clear()
            self._render_revision += 1
        return cached

    @property
    def last_changed_tiles(self) -> set[Tile]:
        return set(self._last_changed_tiles)

    @property
    def render_revision(self) -> int:
        return self._render_revision

    def composite_region(self, document: Document, rect: Rect, checker: bool = False) -> np.ndarray:
        x1, y1, x2, y2 = rect
        if checker:
            out = checker_region(rect)
        else:
            out = blank_rgba(x2 - x1, y2 - y1, (0, 0, 0, 0))
        previous_alpha: np.ndarray | None = None
        for layer in document.layers:
            if not layer.visible:
                continue
            if layer.kind == "adjustment" and layer.adjustment is not None:
                clipping = previous_alpha if layer.clipping and previous_alpha is not None else None
                shifted = replace(layer, x=layer.x - x1, y=layer.y - y1)
                apply_adjustment_layer(out, shifted, clipping)
                continue
            if not self._layer_intersects_region(layer, rect):
                previous_alpha = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                continue
            pixels = self.filtered_pixels(layer)
            alpha_mask = self.layer_mask(layer) if layer.mask_enabled else None
            if layer.clipping and previous_alpha is not None:
                clipping_mask = self._tile_alpha_to_layer_mask(previous_alpha, rect, layer)
                alpha_mask = clipping_mask if alpha_mask is None else np.minimum(alpha_mask, clipping_mask)
            for effect_pixels, ex, ey, opacity, blend_mode in self.layer_effects(layer, pixels):
                self._blend_region(out, rect, effect_pixels, ex, ey, opacity, None, 1.0, blend_mode)
            self._blend_region(out, rect, pixels, layer.x, layer.y, layer.opacity, alpha_mask, layer.mask_density, layer.blend_mode)
            previous_alpha = self._layer_alpha_region(rect, layer, pixels, alpha_mask)
        return out

    def _layer_intersects_region(self, layer: Layer, rect: Rect) -> bool:
        if layer.kind == "shape" and layer.shape_data is not None:
            raw_box = layer.shape_data.get("box", [0, 0, 1, 1])
            sx1, sx2 = sorted((int(raw_box[0]) + layer.x, int(raw_box[2]) + layer.x))
            sy1, sy2 = sorted((int(raw_box[1]) + layer.y, int(raw_box[3]) + layer.y))
            stroke = max(1, int(layer.shape_data.get("stroke_width", 0)))
            bounds = (sx1 - stroke, sy1 - stroke, sx2 + stroke + 1, sy2 + stroke + 1)
        else:
            height, width = layer.pixels.shape[:2]
            bounds = (layer.x, layer.y, layer.x + width, layer.y + height)
        if layer.filters:
            bounds = self._expand_rect(bounds, self._filter_halo(layer.filters))
        bounds = self._expand_for_effects(bounds, layer)
        return bounds[0] < rect[2] and bounds[2] > rect[0] and bounds[1] < rect[3] and bounds[3] > rect[1]

    def filtered_pixels(self, layer: Layer) -> np.ndarray:
        if not layer.filters:
            return layer.pixels
        signature = (layer.pixels_revision, self._json_signature(layer.filters))
        cached_signature = self._filtered_cache.get(layer.id)
        cache_key = ("filtered", layer.id, signature)
        old_cache_key = ("filtered", layer.id, cached_signature) if cached_signature is not None else None
        cached = self.scratch.get(old_cache_key) if old_cache_key is not None else None
        if cached is not None and cached_signature == signature:
            self.profiler.count("render.filter_cache_hit")
            return cached
        dirty = self._filter_dirty.pop(layer.id, None)
        if cached is not None and dirty is not None and layer.filters and self._supports_local_filters(layer.filters):
            with self.profiler.measure("render.filter_stack.partial"):
                pixels = cached
                self._update_filtered_region(layer, pixels, dirty)
            self._filtered_cache[layer.id] = signature
            self.scratch.put(cache_key, pixels)
            if old_cache_key != cache_key:
                self.scratch.delete(old_cache_key)
            self.profiler.count("render.filter_partial")
            return pixels
        with self.profiler.measure("render.filter_stack"):
            pixels = layer.pixels if not layer.filters else apply_filter_stack(layer.pixels, layer.filters)
        self._filtered_cache[layer.id] = signature
        self.scratch.put(cache_key, pixels)
        if old_cache_key is not None and old_cache_key != cache_key:
            self.scratch.delete(old_cache_key)
        self.profiler.count("render.filter_cache_miss")
        return pixels

    def render_for_zoom(self, document: Document, zoom: float, checker: bool = True) -> tuple[np.ndarray, int]:
        """Composite a zoomed-out document tile by tile without a full-size canvas."""
        self.ensure_document(document)
        level = self.mipmaps.level_for_zoom(zoom)
        if level == 0:
            return self.render(document, checker), 0
        factor = 2 ** level
        target_width = max(1, math.ceil(document.width / factor))
        target_height = max(1, math.ceil(document.height / factor))
        key = checker, level
        reduced = self._reduced_composites.get(key)
        if reduced is None or reduced.shape[:2] != (target_height, target_width):
            reduced = blank_rgba(target_width, target_height, (0, 0, 0, 0))
            self._reduced_composites[key] = reduced
            tiles = set(self.all_tiles(document))
        else:
            tiles = set(self._reduced_dirty_tiles[checker])
        self._last_changed_tiles = set(tiles)
        if tiles:
            with self.profiler.measure("render.composite.reduced_tiles"):
                for tx, ty in sorted(tiles):
                    source_rect = self.tile_rect(document, tx, ty)
                    x1, y1, x2, y2 = source_rect
                    dx1, dy1 = x1 // factor, y1 // factor
                    dx2, dy2 = math.ceil(x2 / factor), math.ceil(y2 / factor)
                    source = self.composite_region(document, source_rect, checker)
                    reduced[dy1:dy2, dx1:dx2] = cv2.resize(
                        source,
                        (dx2 - dx1, dy2 - dy1),
                        interpolation=cv2.INTER_AREA,
                    )
            self._reduced_dirty_tiles[checker].difference_update(tiles)
            self.profiler.count("render.reduced_tiles", len(tiles))
            self._render_revision += 1
        return reduced, level

    def cache_status(self) -> dict[str, object]:
        return {**self.scratch.stats, "gpu": dict(self.gpu)}

    def _update_filtered_region(self, layer: Layer, filtered: np.ndarray, dirty: Rect) -> None:
        halo = self._filter_halo(layer.filters)
        height, width = layer.pixels.shape[:2]
        dx1, dy1 = max(0, dirty[0] - halo), max(0, dirty[1] - halo)
        dx2, dy2 = min(width, dirty[2] + halo), min(height, dirty[3] + halo)
        x1 = max(0, dx1 - halo)
        y1 = max(0, dy1 - halo)
        x2 = min(width, dx2 + halo)
        y2 = min(height, dy2 + halo)
        if x1 >= x2 or y1 >= y2:
            return
        updated = apply_filter_stack(layer.pixels[y1:y2, x1:x2], layer.filters)
        filtered[dy1:dy2, dx1:dx2] = updated[dy1 - y1:dy2 - y1, dx1 - x1:dx2 - x1]

    @staticmethod
    def _supports_local_filters(filters: list[dict]) -> bool:
        supported = {"blur", "median", "edge", "emboss"}
        return all(
            not item.get("mask")
            and (not bool(item.get("enabled", True)) or str(item.get("type", "")).lower() in supported)
            for item in filters
        )

    @staticmethod
    def _filter_halo(filters: list[dict]) -> int:
        halo = 0
        for item in filters:
            if not bool(item.get("enabled", True)):
                continue
            kind = str(item.get("type", "")).lower()
            if kind == "blur":
                halo += max(1, int(item.get("radius", 3)))
            elif kind == "median":
                halo += max(3, int(item.get("size", 3)) | 1) // 2
            elif kind == "sharpen":
                halo += 5
            elif kind == "edge":
                halo += 3
            elif kind == "emboss":
                halo += 1
        return max(1, halo)

    @staticmethod
    def _expand_rect(rect: Rect, margin: int) -> Rect:
        return rect[0] - margin, rect[1] - margin, rect[2] + margin, rect[3] + margin

    def layer_mask(self, layer: Layer) -> np.ndarray | None:
        signature = (layer.mask_revision, layer.mask_feather, id(layer.mask))
        cached = self._mask_cache.get(layer.id)
        if cached is not None and cached[0] == signature:
            self.profiler.count("render.mask_cache_hit")
            return cached[1]
        dirty = self._mask_dirty.pop(layer.id, None)
        if cached is not None and dirty is not None and layer.mask is not None:
            if layer.mask_feather <= 0:
                mask = layer.mask
            else:
                mask = cached[1]
                if mask is not None:
                    self._update_feathered_mask_region(layer, mask, dirty)
            self._mask_cache[layer.id] = (signature, mask)
            self.profiler.count("render.mask_partial")
            return mask
        with self.profiler.measure("render.mask"):
            mask = effective_layer_mask(layer)
        self._mask_cache[layer.id] = (signature, mask)
        return mask

    @staticmethod
    def _update_feathered_mask_region(layer: Layer, filtered: np.ndarray, dirty: Rect) -> None:
        if layer.mask is None:
            return
        radius = max(1, int(round(layer.mask_feather)))
        halo = radius * 2
        height, width = layer.mask.shape[:2]
        dx1, dy1 = max(0, dirty[0] - halo), max(0, dirty[1] - halo)
        dx2, dy2 = min(width, dirty[2] + halo), min(height, dirty[3] + halo)
        x1, y1 = max(0, dx1 - halo), max(0, dy1 - halo)
        x2, y2 = min(width, dx2 + halo), min(height, dy2 + halo)
        if x1 >= x2 or y1 >= y2:
            return
        kernel = radius * 2 + 1
        updated = cv2.GaussianBlur(layer.mask[y1:y2, x1:x2], (kernel, kernel), float(layer.mask_feather)).astype(np.uint8)
        filtered[dy1:dy2, dx1:dx2] = updated[dy1 - y1:dy2 - y1, dx1 - x1:dx2 - x1]

    def layer_effects(self, layer: Layer, pixels: np.ndarray) -> list[tuple[np.ndarray, int, int, float, str]]:
        signature = (layer.pixels_revision, self._json_signature(layer.filters), self._json_signature(layer.effects))
        cached = self._effects_cache.get(layer.id)
        if cached is not None and cached[0] == signature:
            self.profiler.count("render.effects_cache_hit")
            return cached[1]
        with self.profiler.measure("render.effects"):
            effects = render_layer_effects(layer, pixels)
        self._effects_cache[layer.id] = (signature, effects)
        return effects

    def all_tiles(self, document: Document) -> Iterable[Tile]:
        for ty in range((document.height + self.tile_size - 1) // self.tile_size):
            for tx in range((document.width + self.tile_size - 1) // self.tile_size):
                yield tx, ty

    def tiles_for_rect(self, rect: Rect) -> Iterable[Tile]:
        x1, y1, x2, y2 = rect
        if x1 >= x2 or y1 >= y2:
            return
        for ty in range(y1 // self.tile_size, (y2 - 1) // self.tile_size + 1):
            for tx in range(x1 // self.tile_size, (x2 - 1) // self.tile_size + 1):
                yield tx, ty

    def tile_rect(self, document: Document, tx: int, ty: int) -> Rect:
        x1, y1 = tx * self.tile_size, ty * self.tile_size
        return x1, y1, min(document.width, x1 + self.tile_size), min(document.height, y1 + self.tile_size)

    @staticmethod
    def _json_signature(value: object) -> str:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _union_rect(left: Rect | None, right: Rect) -> Rect:
        if left is None:
            return right
        return min(left[0], right[0]), min(left[1], right[1]), max(left[2], right[2]), max(left[3], right[3])

    @staticmethod
    def _clip_rect(document: Document, rect: Rect) -> Rect | None:
        x1, y1, x2, y2 = rect
        clipped = max(0, x1), max(0, y1), min(document.width, x2), min(document.height, y2)
        return clipped if clipped[0] < clipped[2] and clipped[1] < clipped[3] else None

    @staticmethod
    def _expand_for_effects(rect: Rect, layer: Layer | None) -> Rect:
        if layer is None or not layer.effects:
            return rect
        margin = 0
        shadow = layer.effects.get("drop_shadow", {})
        glow = layer.effects.get("outer_glow", {})
        stroke = layer.effects.get("stroke", {})
        margin = max(margin, int(shadow.get("blur", 0)) * 2 + abs(int(shadow.get("x", 0))) + abs(int(shadow.get("y", 0))))
        margin = max(margin, int(glow.get("blur", 0)) * 2, int(stroke.get("size", 0)) * 2)
        return rect[0] - margin, rect[1] - margin, rect[2] + margin, rect[3] + margin

    @staticmethod
    def _blend_region(out: np.ndarray, rect: Rect, src: np.ndarray, x: int, y: int, opacity: float, alpha_mask: np.ndarray | None, mask_density: float, blend_mode: str) -> None:
        rx1, ry1, rx2, ry2 = rect
        ix1, iy1 = max(rx1, x), max(ry1, y)
        ix2, iy2 = min(rx2, x + src.shape[1]), min(ry2, y + src.shape[0])
        if ix1 >= ix2 or iy1 >= iy2:
            return
        sx1, sy1 = ix1 - x, iy1 - y
        sx2, sy2 = sx1 + ix2 - ix1, sy1 + iy2 - iy1
        source = src[sy1:sy2, sx1:sx2]
        mask = None if alpha_mask is None else alpha_mask[sy1:sy2, sx1:sx2]
        alpha_blend_inplace(out, source, ix1 - rx1, iy1 - ry1, opacity, mask, mask_density, blend_mode)

    @staticmethod
    def _tile_alpha_to_layer_mask(alpha: np.ndarray, rect: Rect, layer: Layer) -> np.ndarray:
        result = np.zeros(layer.pixels.shape[:2], dtype=np.uint8)
        rx1, ry1, rx2, ry2 = rect
        ix1, iy1 = max(rx1, layer.x), max(ry1, layer.y)
        ix2, iy2 = min(rx2, layer.x + layer.pixels.shape[1]), min(ry2, layer.y + layer.pixels.shape[0])
        if ix1 < ix2 and iy1 < iy2:
            lx1, ly1 = ix1 - layer.x, iy1 - layer.y
            result[ly1:ly1 + iy2 - iy1, lx1:lx1 + ix2 - ix1] = alpha[iy1 - ry1:iy2 - ry1, ix1 - rx1:ix2 - rx1]
        return result

    @staticmethod
    def _layer_alpha_region(rect: Rect, layer: Layer, pixels: np.ndarray, alpha_mask: np.ndarray | None) -> np.ndarray:
        rx1, ry1, rx2, ry2 = rect
        out = np.zeros((ry2 - ry1, rx2 - rx1), dtype=np.uint8)
        ix1, iy1 = max(rx1, layer.x), max(ry1, layer.y)
        ix2, iy2 = min(rx2, layer.x + pixels.shape[1]), min(ry2, layer.y + pixels.shape[0])
        if ix1 >= ix2 or iy1 >= iy2:
            return out
        sx1, sy1 = ix1 - layer.x, iy1 - layer.y
        alpha = pixels[sy1:sy1 + iy2 - iy1, sx1:sx1 + ix2 - ix1, 3].astype(np.float32)
        if alpha_mask is not None:
            mask = alpha_mask[sy1:sy1 + iy2 - iy1, sx1:sx1 + ix2 - ix1].astype(np.float32) / 255.0
            alpha *= (1.0 - float(layer.mask_density)) + mask * float(layer.mask_density)
        out[iy1 - ry1:iy2 - ry1, ix1 - rx1:ix2 - rx1] = np.clip(alpha, 0, 255).astype(np.uint8)
        return out
