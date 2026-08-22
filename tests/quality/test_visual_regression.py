from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from uzyro.quality_metrics import assert_similar_image
from tests.golden_cases import GOLDEN_CASES


GOLDEN_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "golden"


@pytest.mark.parametrize("name", sorted(GOLDEN_CASES))
def test_tool_render_matches_golden_image(name: str) -> None:
    path = GOLDEN_ROOT / f"{name}.png"
    assert path.is_file(), f"Missing golden image: {path}"
    expected = np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)
    actual = GOLDEN_CASES[name]()
    assert_similar_image(actual, expected, minimum_ssim=0.985, maximum_mae=1.5)
