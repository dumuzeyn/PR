from __future__ import annotations

import os
import time
import tracemalloc

from uzyro.core import GradientEngine
from uzyro.interactive_performance import INTERACTIVE_TARGETS_MS, benchmark_interactive_paths
from tests.golden_cases import GOLDEN_CASES


TIMING_TOLERANCE = 2.0 if os.environ.get("CI", "").lower() == "true" else 1.0


def test_interactive_paths_remain_within_declared_targets() -> None:
    report = benchmark_interactive_paths()
    assert all(
        float(report[name]) <= target * TIMING_TOLERANCE
        for name, target in INTERACTIVE_TARGETS_MS.items()
    ), report


def test_4k_gradient_has_bounded_temporary_memory() -> None:
    stops = [(0.0, (255, 0, 0, 255)), (0.5, (0, 255, 0, 255)), (1.0, (0, 0, 255, 255))]
    tracemalloc.start()
    started = time.perf_counter()
    image = GradientEngine.render(
        3840, 2160, (0, 0), (3839, 2159), stops,
        interpolation_space="oklab",
    )
    elapsed = time.perf_counter() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert image.nbytes < 40 * 1024 * 1024
    assert peak < 160 * 1024 * 1024
    assert elapsed < 1.5 * TIMING_TOLERANCE


def test_complete_golden_scene_rendering_is_fast_enough_for_ci() -> None:
    started = time.perf_counter()
    rendered = [render() for render in GOLDEN_CASES.values()]
    elapsed = time.perf_counter() - started
    assert len(rendered) == 35
    assert elapsed < 5.0 * TIMING_TOLERANCE
