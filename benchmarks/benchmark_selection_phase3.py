from __future__ import annotations

import time
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uzyro.core import Document
from uzyro.segmentation import OpenCvCpuSegmentationBackend


def timed(label: str, operation) -> None:
    started = time.perf_counter()
    result = operation()
    elapsed = (time.perf_counter() - started) * 1000.0
    selected = 0 if result is None else int(np.count_nonzero(result))
    print(f"{label}: {elapsed:.1f} ms, selected={selected}")


def main() -> None:
    width, height = 3840, 2160
    yy, xx = np.indices((height, width))
    pixels = np.empty((height, width, 4), dtype=np.uint8)
    pixels[:, :, 0] = np.clip(40 + xx // 24, 0, 255)
    pixels[:, :, 1] = np.clip(70 + yy // 18, 0, 255)
    pixels[:, :, 2] = np.clip(180 - xx // 42, 0, 255)
    pixels[:, :, 3] = 255
    cv2.ellipse(pixels, (1920, 1120), (620, 760), 0, 0, 360, (195, 58, 42, 255), -1)
    document = Document.new(width, height)
    document.layer.pixels = pixels

    timed("Magic Wand 4K", lambda: (document.magic_wand_selection(document.layer, 200, 180, 28, contiguous=True), document.selection_mask)[1])
    timed("Color Range 4K", lambda: (document.color_range_selection(document.layer, 1920, 1120, 26), document.selection_mask)[1])
    timed(
        "Quick Selection 4K",
        lambda: document._quick_selection_mask(document.layer, [(1920, 1120), (1840, 1050), (2010, 1200)], 48, 32, 2, 3, 0.7),
    )
    backend = OpenCvCpuSegmentationBackend()
    timed("Object ROI 4K", lambda: backend.select_object(pixels, (1100, 250, 2750, 2050), 0.65))


if __name__ == "__main__":
    main()
