from __future__ import annotations

from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.golden_cases import GOLDEN_CASES


def main() -> None:
    target = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "golden"
    target.mkdir(parents=True, exist_ok=True)
    for name, render in sorted(GOLDEN_CASES.items()):
        Image.fromarray(render(), "RGBA").save(target / f"{name}.png", optimize=True)
    print(f"Updated {len(GOLDEN_CASES)} golden images in {target}")


if __name__ == "__main__":
    main()
