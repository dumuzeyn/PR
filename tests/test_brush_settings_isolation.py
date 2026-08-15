from __future__ import annotations

from uzyro.brush_config import is_test_polluted_brush_settings


def test_old_ui_test_profile_is_recognized_without_matching_normal_user_settings() -> None:
    polluted = {
        "size": 18, "hardness": 0.17, "opacity": 0.83, "flow": 0.3,
        "spacing": 0.2, "smoothing": 0.15, "blend_mode": "Overlay",
    }
    advanced = {"angle": 37.0, "roundness": 0.42, "scatter": 1.25}
    assert is_test_polluted_brush_settings(polluted, advanced)

    changed_size = {**polluted, "size": 19, "smoothing": 0.0}
    assert is_test_polluted_brush_settings(changed_size, advanced)

    user_overlay = {**polluted, "hardness": 0.18}
    assert not is_test_polluted_brush_settings(user_overlay, advanced)
    assert not is_test_polluted_brush_settings(polluted, {**advanced, "scatter": 1.0})
