from __future__ import annotations

import copy


BRUSH_ADVANCED_DEFAULTS = {
    "angle": 0.0,
    "roundness": 1.0,
    "flip_x": False,
    "flip_y": False,
    "custom_tip_path": "",
    "size_jitter": 0.0,
    "minimum_diameter": 0.01,
    "size_control": "Off",
    "angle_jitter": 0.0,
    "angle_control": "Off",
    "roundness_jitter": 0.0,
    "minimum_roundness": 0.01,
    "roundness_control": "Off",
    "scatter": 0.0,
    "scatter_both_axes": False,
    "scatter_count": 1,
    "count_jitter": 0.0,
    "opacity_jitter": 0.0,
    "flow_jitter": 0.0,
    "minimum_opacity": 0.0,
    "minimum_flow": 0.0,
    "foreground_background_jitter": 0.0,
    "hue_jitter": 0.0,
    "saturation_jitter": 0.0,
    "brightness_jitter": 0.0,
    "texture_path": "",
    "texture_scale": 1.0,
    "texture_depth": 0.0,
    "texture_invert": False,
    "texture_canvas_space": True,
    "dual_tip_path": "",
    "smoothing_mode": "basic",
    "stabilizer_strength": 0.5,
    "stabilizer_window": 8,
    "pulled_string_radius": 30.0,
    "random_seed": 0,
}


def default_brush_config() -> dict[str, object]:
    return copy.deepcopy(BRUSH_ADVANCED_DEFAULTS)


def normalize_brush_config(value: object) -> dict[str, object]:
    result = default_brush_config()
    if isinstance(value, dict):
        result.update({key: value[key] for key in result if key in value})
    return result


__all__ = ["BRUSH_ADVANCED_DEFAULTS", "default_brush_config", "normalize_brush_config"]
