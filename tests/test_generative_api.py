from __future__ import annotations

import numpy as np

from uzyro.generative_api import (
    GenerativeAPIError,
    inpaint_proxy,
    outpaint_proxy,
    strict_inpaint_result,
    strict_outpaint_result,
    validate_outpaint_dimensions,
    variant_seeds,
)


def rgba(width: int, height: int, color=(20, 40, 60, 255)) -> np.ndarray:
    return np.full((height, width, 4), color, dtype=np.uint8)


def test_variant_seeds_are_repeatable_and_bounded() -> None:
    assert variant_seeds(25, 4) == [25, 26, 27, 28]
    assert variant_seeds(4_294_967_294, 2) == [4_294_967_294, 1]
    random_values = variant_seeds(0, 4)
    assert len(random_values) == 4
    assert all(1 <= value <= 4_294_967_294 for value in random_values)


def test_inpaint_preserves_every_pixel_outside_mask() -> None:
    source = rgba(8, 6)
    generated = rgba(4, 3, (220, 10, 15, 255))
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[2:5, 3:7] = 255
    result = strict_inpaint_result(source, generated, mask)
    np.testing.assert_array_equal(result[mask == 0], source[mask == 0])
    assert np.all(result[mask == 255, 0] == 220)


def test_outpaint_preserves_original_rectangle_exactly() -> None:
    source = rgba(7, 5)
    generated = rgba(3, 3, (200, 30, 10, 255))
    result = strict_outpaint_result(source, generated, (2, 3, 4, 1))
    assert result.shape == (9, 13, 4)
    np.testing.assert_array_equal(result[3:8, 2:9], source)


def test_proxy_limits_pixels_and_provider_margins() -> None:
    image = rgba(5000, 3000)
    mask = np.full((3000, 5000), 255, dtype=np.uint8)
    proxy, proxy_mask, scale = inpaint_proxy(image, mask)
    assert proxy.shape[:2] == proxy_mask.shape
    assert proxy.shape[0] * proxy.shape[1] <= 8_600_000
    assert 0 < scale < 1

    outpaint, margins, outpaint_scale = outpaint_proxy(image, (5000, 100, 3000, 20))
    assert max(margins) <= 2000
    assert outpaint_scale < 1
    assert outpaint.shape[0] * outpaint.shape[1] <= 8_600_000


def test_outpaint_dimensions_are_validated_before_request() -> None:
    image = rgba(100, 100)
    assert validate_outpaint_dimensions(image, (20, 10, 30, 40)) == (150, 150)
    for margins in ((0, 0, 0, 0), (1000, 0, 0, 0)):
        try:
            validate_outpaint_dimensions(image, margins)
        except GenerativeAPIError:
            pass
        else:
            raise AssertionError("Недопустимые размеры должны быть отклонены")


def test_generative_module_has_no_cloud_client_or_credentials() -> None:
    import uzyro.generative_api as module

    assert not hasattr(module, "StabilityImageClient")
    assert "STABILITY_API_KEY" not in module.__dict__
