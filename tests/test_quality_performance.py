from __future__ import annotations

import time
import tracemalloc

from photoredactor.core import GradientEngine
from photoredactor.interactive_performance import benchmark_interactive_paths
from tests.golden_cases import GOLDEN_CASES


def test_interactive_paths_remain_within_declared_targets() -> None:
    report = benchmark_interactive_paths()
    assert report["passed"], report


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
    assert elapsed < 1.5


def test_complete_golden_scene_rendering_is_fast_enough_for_ci() -> None:
    started = time.perf_counter()
    rendered = [render() for render in GOLDEN_CASES.values()]
    elapsed = time.perf_counter() - started
    assert len(rendered) == 35
    assert elapsed < 5.0
