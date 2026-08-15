from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import tracemalloc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uzyro.core import GradientEngine
from uzyro.interactive_performance import benchmark_interactive_paths
from tests.golden_cases import GOLDEN_CASES


STOPS = [
    (0.0, (240, 45, 65, 255)),
    (0.5, (35, 210, 125, 255)),
    (1.0, (35, 80, 235, 255)),
]


def timed(operation) -> tuple[float, object]:
    started = time.perf_counter()
    result = operation()
    return (time.perf_counter() - started) * 1000.0, result


def gradient_4k(**options) -> dict[str, float]:
    tracemalloc.start()
    elapsed, image = timed(
        lambda: GradientEngine.render(
            3840, 2160, (120, 160), (3650, 1980), STOPS,
            "linear", **options,
        )
    )
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "milliseconds": elapsed,
        "peak_megabytes": peak / 1024 / 1024,
        "output_megabytes": image.nbytes / 1024 / 1024,
    }


def main() -> None:
    golden_ms, images = timed(lambda: [render() for render in GOLDEN_CASES.values()])
    del images
    report = {
        "interactive": benchmark_interactive_paths(),
        "gradient_4k_srgb": gradient_4k(),
        "gradient_4k_oklab": gradient_4k(interpolation_space="oklab"),
        "gradient_4k_noise": gradient_4k(
            noise={
                "enabled": True, "roughness": 0.65, "color_model": "hsv", "seed": 2026,
                "channels": [[0.0, 1.0], [0.35, 1.0], [0.35, 1.0]],
            }
        ),
        "golden_cases": len(GOLDEN_CASES),
        "golden_suite_ms": golden_ms,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
