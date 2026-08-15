from __future__ import annotations

from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.quality_fixtures import QUALITY_FIXTURES


def main() -> None:
    target = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "quality"
    target.mkdir(parents=True, exist_ok=True)
    for name, build in QUALITY_FIXTURES.items():
        values = build()
        suffixes = ("input", "mask") if len(values) == 2 else ("input", "expected", "mask")
        for suffix, value in zip(suffixes, values):
            mode = "L" if value.ndim == 2 else "RGBA"
            Image.fromarray(value, mode).save(target / f"{name}_{suffix}.png", optimize=True)
    print(f"Updated {len(QUALITY_FIXTURES)} quality fixtures in {target}")


if __name__ == "__main__":
    main()
