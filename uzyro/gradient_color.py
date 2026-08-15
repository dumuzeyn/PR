from __future__ import annotations

import numpy as np


INTERPOLATION_SPACES = ("srgb", "linear_rgb", "oklab")


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.power(rgb, 1.0 / 2.4) - 0.055)


def _linear_to_oklab(rgb: np.ndarray) -> np.ndarray:
    lms = rgb @ np.array(
        [
            [0.4122214708, 0.2119034982, 0.0883024619],
            [0.5363325363, 0.6806995451, 0.2817188376],
            [0.0514459929, 0.1073969566, 0.6299787005],
        ],
        dtype=np.float32,
    )
    lms = np.cbrt(lms)
    return lms @ np.array(
        [
            [0.2104542553, 1.9779984951, 0.0259040371],
            [0.7936177850, -2.4285922050, 0.7827717662],
            [-0.0040720468, 0.4505937099, -0.8086757660],
        ],
        dtype=np.float32,
    )


def _oklab_to_linear(lab: np.ndarray) -> np.ndarray:
    lms = lab @ np.array(
        [
            [1.0, 1.0, 1.0],
            [0.3963377774, -0.1055613458, -0.0894841775],
            [0.2158037573, -0.0638541728, -1.2914855480],
        ],
        dtype=np.float32,
    )
    lms = lms * lms * lms
    return lms @ np.array(
        [
            [4.0767416621, -1.2684380046, -0.0041960863],
            [-3.3077115913, 2.6097574011, -0.7034186147],
            [0.2309699292, -0.3413193965, 1.7076147010],
        ],
        dtype=np.float32,
    )


def color_lookup(
    axis: np.ndarray,
    positions: np.ndarray,
    colors: np.ndarray,
    interpolation_space: str = "srgb",
) -> np.ndarray:
    """Interpolate RGBA stops in the requested color space."""
    space = interpolation_space if interpolation_space in INTERPOLATION_SPACES else "srgb"
    rgb = np.asarray(colors[:, :3], dtype=np.float32)
    if space == "linear_rgb":
        working = _srgb_to_linear(rgb)
    elif space == "oklab":
        working = _linear_to_oklab(_srgb_to_linear(rgb))
    else:
        working = rgb

    result = np.empty((len(axis), 4), dtype=np.float32)
    interpolated = np.column_stack(
        [np.interp(axis, positions, working[:, channel]) for channel in range(3)]
    ).astype(np.float32)
    if space == "linear_rgb":
        result[:, :3] = _linear_to_srgb(interpolated)
    elif space == "oklab":
        result[:, :3] = _linear_to_srgb(_oklab_to_linear(interpolated))
    else:
        result[:, :3] = interpolated
    result[:, 3] = np.interp(axis, positions, colors[:, 3])
    return np.clip(result, 0.0, 1.0)


def ordered_dither(
    image: np.ndarray,
    output_depth: int,
    origin: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Apply a subtle deterministic threshold pattern without low-frequency noise."""
    matrix = np.array(
        [
            [0, 32, 8, 40, 2, 34, 10, 42],
            [48, 16, 56, 24, 50, 18, 58, 26],
            [12, 44, 4, 36, 14, 46, 6, 38],
            [60, 28, 52, 20, 62, 30, 54, 22],
            [3, 35, 11, 43, 1, 33, 9, 41],
            [51, 19, 59, 27, 49, 17, 57, 25],
            [15, 47, 7, 39, 13, 45, 5, 37],
            [63, 31, 55, 23, 61, 29, 53, 21],
        ],
        dtype=np.float32,
    )
    height, width = image.shape[:2]
    x0, y0 = int(round(origin[0])) % 8, int(round(origin[1])) % 8
    yy = (np.arange(height) + y0) % 8
    xx = (np.arange(width) + x0) % 8
    threshold = (matrix[np.ix_(yy, xx)] - 31.5) / 64.0
    levels = 255.0 if output_depth == 8 else 65535.0 if output_depth == 16 else 65535.0
    result = image.copy()
    result[:, :, :3] = np.clip(result[:, :, :3] + threshold[:, :, None] / levels, 0.0, 1.0)
    return result
