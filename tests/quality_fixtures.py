from __future__ import annotations

import cv2
import numpy as np


SIZE = (180, 240)


def _rgba(rgb: np.ndarray) -> np.ndarray:
    alpha = np.full(rgb.shape[:2] + (1,), 255, dtype=np.uint8)
    return np.concatenate((np.clip(rgb, 0, 255).astype(np.uint8), alpha), axis=2)


def hair_scene() -> tuple[np.ndarray, np.ndarray]:
    height, width = SIZE
    image = np.full((height, width, 3), (205, 212, 220), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(image, (120, 106), (48, 60), 0, 0, 360, (72, 60, 55), -1, cv2.LINE_AA)
    cv2.ellipse(mask, (120, 106), (48, 60), 0, 0, 360, 255, -1, cv2.LINE_AA)
    for index, x in enumerate(range(76, 165, 5)):
        end = (x + (index % 3 - 1) * 15, 18 + (index % 4) * 4)
        cv2.line(image, (x, 64), end, (68, 56, 52), 1, cv2.LINE_AA)
        cv2.line(mask, (x, 64), end, 235, 1, cv2.LINE_AA)
    return _rgba(image), mask


def similar_object_scene() -> tuple[np.ndarray, np.ndarray]:
    height, width = SIZE
    yy, xx = np.mgrid[:height, :width]
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = 104 + xx * 18 // width
    image[:, :, 1] = 122 + yy * 16 // height
    image[:, :, 2] = 135 + (xx + yy) * 12 // (width + height)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, (120, 98), (55, 63), -8, 0, 360, 255, -1, cv2.LINE_AA)
    texture = ((xx * 7 + yy * 3) % 19) - 9
    object_rgb = np.stack((118 + texture, 139 + texture, 151 + texture), axis=2)
    alpha = mask.astype(np.float32)[:, :, None] / 255.0
    image = image * (1.0 - alpha) + object_rgb * alpha
    return _rgba(image), mask


def sky_buildings_scene() -> tuple[np.ndarray, np.ndarray]:
    height, width = SIZE
    yy, xx = np.mgrid[:height, :width]
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = 95 + yy * 65 // height
    image[:, :, 1] = 155 + yy * 45 // height
    image[:, :, 2] = 225
    mask = np.full((height, width), 255, dtype=np.uint8)
    ground = 132 + (np.sin(np.arange(width) / 27.0) * 5).astype(np.int32)
    for x, top in enumerate(ground):
        image[top:, x] = (58, 66, 72)
        mask[top:, x] = 0
    for x1, y1, x2 in ((20, 62, 54), (78, 88, 112), (148, 48, 184), (196, 78, 229)):
        image[y1:, x1:x2] = (52, 58, 65)
        mask[y1:, x1:x2] = 0
    cv2.line(image, (64, 22), (64, 132), (45, 48, 52), 2, cv2.LINE_AA)
    cv2.line(mask, (64, 22), (64, 132), 0, 2, cv2.LINE_AA)
    return _rgba(image), mask


def repeated_texture_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = SIZE
    yy, xx = np.mgrid[:height, :width]
    clean = np.empty((height, width, 3), dtype=np.uint8)
    pattern = ((xx // 8 + yy // 8) % 2) * 80
    clean[:, :, 0] = 75 + pattern
    clean[:, :, 1] = 105 + pattern // 2
    clean[:, :, 2] = 145 + pattern // 3
    damaged = clean.copy()
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (98, 62), (142, 112), 255, -1)
    damaged[mask > 0] = (120, 120, 120)
    return _rgba(damaged), _rgba(clean), mask


def wire_texture_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = SIZE
    yy, xx = np.mgrid[:height, :width]
    clean = np.stack((80 + xx // 3, 105 + yy // 4, 125 + ((xx + yy) % 31)), axis=2).astype(np.uint8)
    damaged = clean.copy()
    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.array([[15, 142], [75, 88], [130, 118], [224, 34]], np.int32)
    cv2.polylines(damaged, [points], False, (20, 20, 24), 5, cv2.LINE_AA)
    cv2.polylines(mask, [points], False, 255, 9, cv2.LINE_AA)
    return _rgba(damaged), _rgba(clean), mask


def skin_blemish_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = SIZE
    yy, xx = np.mgrid[:height, :width]
    rng = np.random.default_rng(2026)
    clean = np.empty((height, width, 3), dtype=np.float32)
    clean[:, :, 0] = 194 + xx * 14 / width + rng.normal(0, 2, SIZE)
    clean[:, :, 1] = 139 + yy * 10 / height + rng.normal(0, 1.5, SIZE)
    clean[:, :, 2] = 118 + (xx + yy) * 8 / sum(SIZE) + rng.normal(0, 1.5, SIZE)
    damaged = np.clip(clean, 0, 255).astype(np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (120, 90), 9, 255, -1, cv2.LINE_AA)
    cv2.circle(damaged, (120, 90), 7, (145, 62, 62), -1, cv2.LINE_AA)
    return _rgba(damaged), _rgba(clean), mask


def sharp_edge_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = SIZE
    clean = np.full((height, width, 3), (55, 90, 170), dtype=np.uint8)
    clean[:, width // 2 :] = (225, 185, 75)
    damaged = clean.copy()
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (width // 2, 90), 10, 255, -1, cv2.LINE_AA)
    cv2.circle(damaged, (width // 2, 90), 8, (25, 25, 25), -1, cv2.LINE_AA)
    return _rgba(damaged), _rgba(clean), mask


QUALITY_FIXTURES = {
    "hair": hair_scene,
    "similar_object": similar_object_scene,
    "sky_buildings": sky_buildings_scene,
    "repeated_texture": repeated_texture_scene,
    "wire_texture": wire_texture_scene,
    "skin_blemish": skin_blemish_scene,
    "sharp_edge": sharp_edge_scene,
}
