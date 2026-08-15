from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from photoredactor.core import (
    BrushPathSampler,
    BrushSettings,
    Document,
    GradientEngine,
    Layer,
    PixelBrushStroke,
    clone_or_heal,
    refine_selection_mask,
    render_shape_layer,
    spot_heal,
)


WIDTH, HEIGHT = 192, 128
INK = (31, 120, 210, 255)


def _canvas(color=(246, 247, 249, 255)) -> np.ndarray:
    pixels = np.empty((HEIGHT, WIDTH, 4), dtype=np.uint8)
    pixels[:] = color
    return pixels


def _paint_path(settings: BrushSettings, points: list[tuple[int, int]]) -> np.ndarray:
    layer = Layer("Golden brush", _canvas())
    stroke = PixelBrushStroke(layer, settings, INK, background=(235, 65, 75, 255))
    sampler = BrushPathSampler(settings)
    dabs = sampler.begin(points[0])
    for point in points[1:]:
        dabs.extend(sampler.extend(point))
    for dab in dabs:
        stroke.dab(dab.x, dab.y, dab.pressure, dab)
    return layer.pixels


def _brush_case(**overrides) -> np.ndarray:
    values = {"radius": 16, "hardness": 0.5, "opacity": 0.92, "flow": 0.8, "spacing": 0.18, "random_seed": 1701}
    values.update(overrides)
    settings = BrushSettings(**values)
    points = [(18, 94), (46, 35), (78, 88), (112, 30), (174, 78)]
    return _paint_path(settings, points)


def brush_hardness_0() -> np.ndarray:
    return _brush_case(hardness=0.0)


def brush_hardness_50() -> np.ndarray:
    return _brush_case(hardness=0.5)


def brush_hardness_100() -> np.ndarray:
    return _brush_case(hardness=1.0)


def brush_flow() -> np.ndarray:
    return _brush_case(flow=0.16, spacing=0.08)


def brush_spacing() -> np.ndarray:
    return _brush_case(spacing=0.82, hardness=0.9)


def brush_angle() -> np.ndarray:
    return _brush_case(roundness=0.28, angle=38.0, hardness=0.9)


def brush_roundness() -> np.ndarray:
    return _brush_case(roundness=0.22, angle=90.0, hardness=0.85)


def brush_scatter() -> np.ndarray:
    return _brush_case(scatter=1.25, scatter_both_axes=True, scatter_count=3, size_jitter=0.35)


def brush_stabilizer() -> np.ndarray:
    settings = BrushSettings(
        radius=8, hardness=0.75, spacing=0.15, smoothing_mode="stabilizer",
        stabilizer_strength=0.85, stabilizer_window=8, random_seed=3,
    )
    points = [(10 + index * 4, 64 + (10 if index % 2 else -10)) for index in range(44)]
    return _paint_path(settings, points)


def _gradient(kind: str, interpolation_space: str = "srgb") -> np.ndarray:
    stops = [
        {"position": 0.0, "color": [240, 48, 65, 255], "midpoint": 0.42},
        {"position": 0.52, "color": [45, 210, 125, 255], "midpoint": 0.58},
        {"position": 1.0, "color": [35, 85, 235, 255]},
    ]
    return GradientEngine.render(
        WIDTH, HEIGHT, (24, 24), (168, 105), stops, kind,
        interpolation_space=interpolation_space, dither=True,
    )


def gradient_linear() -> np.ndarray:
    return _gradient("linear")


def gradient_radial() -> np.ndarray:
    return _gradient("radial")


def gradient_reflected() -> np.ndarray:
    return _gradient("reflected")


def gradient_oklab() -> np.ndarray:
    return _gradient("linear", "oklab")


def _shape(kind: str, **properties) -> np.ndarray:
    document = Document.new(WIDTH, HEIGHT, (0, 0, 0, 0))
    box = properties.pop("box", (28, 18, 164, 110))
    layer = document.add_shape_layer(kind, box, (45, 145, 220, 220), (245, 185, 45, 255), 6, sides=7, inner_ratio=0.42)
    layer.shape_data.update(properties)
    render_shape_layer(layer)
    return layer.pixels


def shape_rectangle() -> np.ndarray:
    return _shape("rectangle", corner_radius=14)


def shape_ellipse() -> np.ndarray:
    return _shape("ellipse")


def shape_star() -> np.ndarray:
    return _shape("star")


def shape_bezier() -> np.ndarray:
    return _shape("bezier", fill=[0, 0, 0, 0], stroke_width=8, stroke_cap="round")


def shape_rotated() -> np.ndarray:
    return _shape("rectangle", box=(48, 28, 144, 100), corner_radius=10, rotation=27.0)


def shape_stroke_joins() -> np.ndarray:
    document = Document.new(WIDTH, HEIGHT, (0, 0, 0, 0))
    layer = document.add_shape_layer("bezier", (20, 20, 172, 108), (0, 0, 0, 0), (40, 60, 85, 255), 12)
    layer.shape_data["path_nodes"] = [
        {"anchor": [22, 100], "in": [22, 100], "out": [22, 100], "linked": False},
        {"anchor": [94, 20], "in": [94, 20], "out": [94, 20], "linked": False},
        {"anchor": [170, 100], "in": [170, 100], "out": [170, 100], "linked": False},
    ]
    layer.shape_data.update({"stroke_join": "miter", "miter_limit": 8.0, "stroke_cap": "square"})
    render_shape_layer(layer)
    return layer.pixels


def _mask_image(mask: np.ndarray) -> np.ndarray:
    output = _canvas((35, 39, 46, 255))
    alpha = mask.astype(np.float32) / 255.0
    selected = np.array([70, 180, 240], dtype=np.float32)
    output[:, :, :3] = np.clip(output[:, :, :3] * (1.0 - alpha[:, :, None]) + selected * alpha[:, :, None], 0, 255)
    return output


def _base_ellipse() -> np.ndarray:
    document = Document.new(WIDTH, HEIGHT, (0, 0, 0, 0))
    document.set_ellipse_selection((35, 18, 157, 111))
    return document.selection_mask


def selection_ellipse() -> np.ndarray:
    return _mask_image(_base_ellipse())


def selection_feather() -> np.ndarray:
    return _mask_image(refine_selection_mask(_base_ellipse(), feather=7))


def selection_smooth() -> np.ndarray:
    mask = _base_ellipse().copy()
    mask[30:100:5, 28:38] = 255
    return _mask_image(refine_selection_mask(mask, smooth=6))


def selection_contrast() -> np.ndarray:
    return _mask_image(refine_selection_mask(_base_ellipse(), feather=7, contrast=2.4))


def selection_shift_edge() -> np.ndarray:
    return _mask_image(refine_selection_mask(_base_ellipse(), shift=-8))


def _selection_mode(mode: str) -> np.ndarray:
    document = Document.new(WIDTH, HEIGHT, (0, 0, 0, 0))
    document.set_ellipse_selection((20, 20, 125, 108))
    document.set_rect_selection((78, 12, 174, 96), mode)
    return _mask_image(document.selection_mask if document.selection_mask is not None else np.zeros((HEIGHT, WIDTH), np.uint8))


def selection_add() -> np.ndarray:
    return _selection_mode("add")


def selection_subtract() -> np.ndarray:
    return _selection_mode("subtract")


def selection_intersect() -> np.ndarray:
    return _selection_mode("intersect")


def _retouch_scene(include_target: bool = True) -> np.ndarray:
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    pixels = _canvas()
    pixels[:, :, 0] = 70 + (xx * 130 // WIDTH)
    pixels[:, :, 1] = 95 + (yy * 90 // HEIGHT)
    pixels[:, :, 2] = 125 + ((xx + yy) % 24)
    cv2.circle(pixels, (45, 64), 19, (235, 72, 52, 255), -1, cv2.LINE_AA)
    if include_target:
        cv2.circle(pixels, (145, 64), 13, (25, 25, 25, 255), -1, cv2.LINE_AA)
    return pixels


def healing_clone() -> np.ndarray:
    layer = Layer("Clone", _retouch_scene())
    clone_or_heal(layer, 45, 64, 145, 64, 20, hardness=0.8)
    return layer.pixels


def healing_diffusion() -> np.ndarray:
    pixels = _retouch_scene(False)
    cv2.circle(pixels, (145, 64), 7, (18, 18, 18, 255), -1, cv2.LINE_AA)
    layer = Layer("Healing", pixels)
    clone_or_heal(layer, 105, 64, 145, 64, 18, heal=True, hardness=0.85, diffusion=7)
    return layer.pixels


def healing_spot() -> np.ndarray:
    pixels = _retouch_scene(False)
    cv2.circle(pixels, (145, 64), 7, (20, 20, 20, 255), -1, cv2.LINE_AA)
    layer = Layer("Spot", pixels)
    spot_heal(layer, 145, 64, 12, hardness=0.9, mode="content_aware")
    return layer.pixels


def _text(text: str, **properties) -> np.ndarray:
    document = Document.new(WIDTH, HEIGHT, (0, 0, 0, 0))
    layer = document.add_text_layer(text, 12, 18, (32, 55, 78, 255), 28, "Arial", **properties)
    return layer.pixels


def text_kerning() -> np.ndarray:
    return _text("AV To WA AV", kerning_enabled=True)


def text_tracking() -> np.ndarray:
    return _text("TRACKING", tracking=5)


def text_paragraph() -> np.ndarray:
    document = Document.new(WIDTH, HEIGHT, (0, 0, 0, 0))
    return document.add_text_layer(
        "A compact multiline paragraph wraps inside its box.", 12, 14,
        (32, 55, 78, 255), 20, "Arial", 168, box_height=108,
        text_mode="paragraph", line_spacing=3,
    ).pixels


def text_cyrillic() -> np.ndarray:
    return _text("Русский текст\nи типографика", box_width=170, box_height=100, text_mode="paragraph")


def text_on_path() -> np.ndarray:
    points = [[12, 92], [48, 18], [142, 18], [180, 92]]
    return _text("Текст по кривой", box_width=170, path_mode="bezier", path_points=points)


GOLDEN_CASES: dict[str, Callable[[], np.ndarray]] = {
    name.removeprefix("case_"): value
    for name, value in {
        "case_brush_hardness_0": brush_hardness_0, "case_brush_hardness_50": brush_hardness_50,
        "case_brush_hardness_100": brush_hardness_100, "case_brush_flow": brush_flow,
        "case_brush_spacing": brush_spacing, "case_brush_angle": brush_angle,
        "case_brush_roundness": brush_roundness, "case_brush_scatter": brush_scatter,
        "case_brush_stabilizer": brush_stabilizer, "case_gradient_linear": gradient_linear,
        "case_gradient_radial": gradient_radial, "case_gradient_reflected": gradient_reflected,
        "case_gradient_oklab": gradient_oklab, "case_shape_rectangle": shape_rectangle,
        "case_shape_ellipse": shape_ellipse, "case_shape_star": shape_star,
        "case_shape_bezier": shape_bezier, "case_shape_rotated": shape_rotated,
        "case_shape_stroke_joins": shape_stroke_joins, "case_selection_ellipse": selection_ellipse,
        "case_selection_feather": selection_feather, "case_selection_smooth": selection_smooth,
        "case_selection_contrast": selection_contrast, "case_selection_shift_edge": selection_shift_edge,
        "case_selection_add": selection_add, "case_selection_subtract": selection_subtract,
        "case_selection_intersect": selection_intersect, "case_healing_clone": healing_clone,
        "case_healing_diffusion": healing_diffusion, "case_healing_spot": healing_spot,
        "case_text_kerning": text_kerning, "case_text_tracking": text_tracking,
        "case_text_paragraph": text_paragraph, "case_text_cyrillic": text_cyrillic,
        "case_text_on_path": text_on_path,
    }.items()
}


__all__ = ["GOLDEN_CASES", "HEIGHT", "WIDTH"]
