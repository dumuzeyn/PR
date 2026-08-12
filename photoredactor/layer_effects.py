from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .core_shared import GradientEngine, blend_rgb

EffectLayer = tuple[np.ndarray, int, int, float, str]

EFFECT_ORDER = (
    "drop_shadow", "outer_glow", "stroke", "bevel_emboss", "inner_shadow",
    "inner_glow", "satin", "color_overlay", "gradient_overlay", "pattern_overlay",
)
EFFECT_LABELS = {
    "drop_shadow": "Тень", "outer_glow": "Внешнее свечение", "stroke": "Обводка",
    "bevel_emboss": "Тиснение", "inner_shadow": "Внутренняя тень",
    "inner_glow": "Внутреннее свечение", "satin": "Атлас",
    "color_overlay": "Наложение цвета", "gradient_overlay": "Наложение градиента",
    "pattern_overlay": "Наложение узора",
}


def _defaults(kind: str) -> dict[str, Any]:
    common: dict[str, Any] = {"enabled": False, "opacity": 0.65, "blend_mode": "Normal"}
    values = {
        "drop_shadow": {"x": 10, "y": 10, "blur": 12, "color": [0, 0, 0, 255], "blend_mode": "Multiply"},
        "outer_glow": {"blur": 18, "color": [255, 220, 80, 255], "blend_mode": "Screen"},
        "stroke": {"size": 4, "position": "outside", "color": [255, 255, 255, 255], "opacity": 1.0},
        "bevel_emboss": {"size": 6, "depth": 1.0, "angle": 135.0, "highlight": [255, 255, 255, 255], "shadow": [0, 0, 0, 255]},
        "inner_shadow": {"x": 6, "y": 6, "blur": 8, "color": [0, 0, 0, 255], "blend_mode": "Multiply"},
        "inner_glow": {"blur": 12, "color": [255, 245, 190, 255], "blend_mode": "Screen"},
        "satin": {"distance": 8, "size": 14, "angle": 20.0, "color": [30, 30, 45, 255], "blend_mode": "Multiply"},
        "color_overlay": {"color": [45, 130, 220, 255], "blend_mode": "Normal"},
        "gradient_overlay": {"angle": 0.0, "scale": 100.0, "color": [35, 100, 220, 255], "color2": [245, 185, 55, 255]},
        "pattern_overlay": {"scale": 16.0, "angle": 45.0, "color": [245, 245, 245, 255], "color2": [60, 65, 72, 255]},
    }
    return {**common, **values[kind]}


class LayerEffectsStack:
    """Normalize and render the deterministic, non-destructive layer-style stack."""

    @classmethod
    def normalize(cls, effects: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
        source = effects if isinstance(effects, dict) else {}
        result: dict[str, dict[str, Any]] = {}
        for kind in EFFECT_ORDER:
            if kind not in source:
                continue
            raw = source.get(kind)
            if not isinstance(raw, dict):
                continue
            item = {**_defaults(kind), **raw}
            item["enabled"] = bool(item.get("enabled", True))
            item["opacity"] = float(np.clip(item.get("opacity", 1.0), 0.0, 1.0))
            result[kind] = item
        return result

    @classmethod
    def item(cls, kind: str, raw: dict[str, Any] | None = None) -> dict[str, Any]:
        if kind not in EFFECT_ORDER:
            raise ValueError(f"Unsupported layer effect: {kind}")
        return {**_defaults(kind), **(raw or {})}

    @classmethod
    def render(cls, layer, pixels: np.ndarray) -> tuple[list[EffectLayer], np.ndarray]:
        stack = cls.normalize(layer.effects)
        alpha = pixels[:, :, 3]
        if not stack or not np.any(alpha):
            return [], pixels
        underlays: list[EffectLayer] = []
        for kind in ("drop_shadow", "outer_glow", "stroke"):
            item = stack.get(kind)
            if item and item["enabled"]:
                underlays.extend(cls._render_underlay(kind, item, layer, alpha))
        styled = pixels.copy()
        for kind in EFFECT_ORDER[3:]:
            item = stack.get(kind)
            if item and item["enabled"]:
                styled = cls._render_overlay(kind, item, styled, alpha)
        return underlays, styled

    @staticmethod
    def _render_underlay(kind: str, item: dict[str, Any], layer, alpha: np.ndarray) -> list[EffectLayer]:
        opacity = float(item["opacity"])
        color = tuple(int(value) for value in item.get("color", [0, 0, 0, 255]))
        if kind == "drop_shadow":
            mask = _expanded_blur(alpha, int(item.get("blur", 12)))
            return [(_solid(mask, color), layer.x + int(item.get("x", 10)), layer.y + int(item.get("y", 10)), opacity, str(item.get("blend_mode", "Multiply")))]
        if kind == "outer_glow":
            mask = np.clip(_expanded_blur(alpha, int(item.get("blur", 18))).astype(np.int16) - alpha.astype(np.int16), 0, 255).astype(np.uint8)
            return [(_solid(mask, color), layer.x, layer.y, opacity, str(item.get("blend_mode", "Screen")))]
        size = max(1, int(item.get("size", 4)))
        binary = np.where(alpha > 0, 255, 0).astype(np.uint8)
        kernel = np.ones((size * 2 + 1, size * 2 + 1), dtype=np.uint8)
        position = str(item.get("position", "outside"))
        outside = cv2.dilate(binary, kernel) - binary
        inside = binary - cv2.erode(binary, kernel)
        mask = inside if position == "inside" else np.maximum(outside, inside) if position == "center" else outside
        return [(_solid(mask, color), layer.x, layer.y, opacity, str(item.get("blend_mode", "Normal")))]

    @classmethod
    def _render_overlay(cls, kind: str, item: dict[str, Any], pixels: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        height, width = alpha.shape
        opacity = float(item["opacity"])
        mode = str(item.get("blend_mode", "Normal"))
        coverage = alpha.astype(np.float32) / 255.0
        if kind == "bevel_emboss":
            smooth = cv2.GaussianBlur(coverage, (0, 0), max(0.5, float(item.get("size", 6)) / 3.0))
            gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
            angle = np.deg2rad(float(item.get("angle", 135.0)))
            shade = np.clip((gx * np.cos(angle) + gy * np.sin(angle)) * float(item.get("depth", 1.0)) * 4.0, -1.0, 1.0)
            highlight = np.asarray(item.get("highlight", [255, 255, 255, 255])[:3], dtype=np.float32)
            shadow = np.asarray(item.get("shadow", [0, 0, 0, 255])[:3], dtype=np.float32)
            overlay = np.where((shade >= 0)[:, :, None], highlight, shadow)
            return _blend_inside(pixels, overlay, np.abs(shade) * coverage * opacity, "Screen" if float(shade.mean()) >= 0 else "Overlay")
        if kind in {"inner_shadow", "inner_glow"}:
            blur = max(1, int(item.get("blur", 10)))
            if kind == "inner_shadow":
                shifted = np.roll(coverage, (int(item.get("y", 5)), int(item.get("x", 5))), axis=(0, 1))
                edge = np.clip(coverage - cv2.GaussianBlur(shifted, (0, 0), blur / 2.0), 0.0, 1.0)
            else:
                edge = np.clip(coverage - cv2.erode(coverage, np.ones((blur * 2 + 1, blur * 2 + 1), np.uint8)), 0.0, 1.0)
                edge = cv2.GaussianBlur(edge, (0, 0), max(0.5, blur / 3.0))
            color = np.broadcast_to(np.asarray(item.get("color", [0, 0, 0, 255])[:3], dtype=np.float32), (height, width, 3))
            return _blend_inside(pixels, color, edge * coverage * opacity, mode)
        if kind == "satin":
            distance = int(item.get("distance", 8)); angle = np.deg2rad(float(item.get("angle", 20.0)))
            dx, dy = round(np.cos(angle) * distance), round(np.sin(angle) * distance)
            shifted_a = np.roll(coverage, (dy, dx), axis=(0, 1)); shifted_b = np.roll(coverage, (-dy, -dx), axis=(0, 1))
            satin = cv2.GaussianBlur(np.abs(shifted_a - shifted_b), (0, 0), max(0.5, float(item.get("size", 14)) / 4.0))
            color = np.broadcast_to(np.asarray(item.get("color", [30, 30, 45, 255])[:3], dtype=np.float32), (height, width, 3))
            return _blend_inside(pixels, color, satin * coverage * opacity, mode)
        if kind == "gradient_overlay":
            angle = np.deg2rad(float(item.get("angle", 0.0))); scale = max(1.0, float(item.get("scale", 100.0))) / 100.0
            center = (width / 2.0, height / 2.0); reach = max(width, height) * scale / 2.0
            start = (center[0] - np.cos(angle) * reach, center[1] - np.sin(angle) * reach)
            end = (center[0] + np.cos(angle) * reach, center[1] + np.sin(angle) * reach)
            overlay = GradientEngine.render(width, height, start, end, [[0.0, item.get("color", [35, 100, 220, 255])], [1.0, item.get("color2", [245, 185, 55, 255])]])[:, :, :3]
        elif kind == "pattern_overlay":
            scale = max(2, int(item.get("scale", 16))); yy, xx = np.indices((height, width))
            angle = np.deg2rad(float(item.get("angle", 45.0))); axis = xx * np.cos(angle) + yy * np.sin(angle)
            mask = ((np.floor(axis / scale) + np.floor(((-xx * np.sin(angle) + yy * np.cos(angle))) / scale)) % 2).astype(bool)
            first = np.asarray(item.get("color", [245, 245, 245, 255])[:3]); second = np.asarray(item.get("color2", [60, 65, 72, 255])[:3])
            overlay = np.where(mask[:, :, None], first, second).astype(np.float32)
        else:
            overlay = np.broadcast_to(np.asarray(item.get("color", [45, 130, 220, 255])[:3], dtype=np.float32), (height, width, 3))
        return _blend_inside(pixels, overlay, coverage * opacity, mode)


def _expanded_blur(alpha: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius == 0:
        return alpha.copy()
    kernel = np.ones((max(1, radius // 2) * 2 + 1,) * 2, dtype=np.uint8)
    return cv2.GaussianBlur(cv2.dilate(alpha, kernel), (radius * 2 + 1,) * 2, radius)


def _solid(alpha: np.ndarray, color: tuple[int, ...]) -> np.ndarray:
    output = np.zeros((*alpha.shape, 4), dtype=np.uint8)
    output[:, :, :3] = color[:3]
    output[:, :, 3] = np.clip(alpha.astype(np.float32) * (color[3] / 255.0), 0, 255).astype(np.uint8)
    return output


def _blend_inside(pixels: np.ndarray, overlay: np.ndarray, coverage: np.ndarray, mode: str) -> np.ndarray:
    output = pixels.copy()
    amount = np.clip(coverage, 0.0, 1.0)[:, :, None]
    blended = blend_rgb(overlay.astype(np.float32), output[:, :, :3], mode)
    output[:, :, :3] = np.clip(output[:, :, :3] * (1.0 - amount) + blended * amount, 0, 255).astype(np.uint8)
    return output
