from __future__ import annotations

from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
import os
import time
from typing import Iterator


@dataclass(frozen=True)
class TimingSummary:
    calls: int
    total_ms: float
    average_ms: float
    maximum_ms: float


class PerformanceProfiler:
    """Low-overhead timings that stay silent unless debug profiling is enabled."""

    def __init__(self, enabled: bool | None = None, sample_limit: int = 240) -> None:
        if enabled is None:
            value = os.environ.get("UZYRO_DEBUG_PERF", os.environ.get("UZYRO_DEBUG_PERF", ""))
            enabled = value.lower() in {"1", "true", "yes", "on"}
        self.enabled = bool(enabled)
        self.sample_limit = max(1, int(sample_limit))
        self._samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=self.sample_limit))
        self._counters: dict[str, int] = defaultdict(int)

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            self._samples[name].append((time.perf_counter() - started) * 1000.0)

    def count(self, name: str, amount: int = 1) -> None:
        if self.enabled:
            self._counters[name] += int(amount)

    def counter(self, name: str) -> int:
        return self._counters.get(name, 0)

    def summary(self) -> dict[str, TimingSummary]:
        result: dict[str, TimingSummary] = {}
        for name, samples in self._samples.items():
            if samples:
                total = sum(samples)
                result[name] = TimingSummary(len(samples), total, total / len(samples), max(samples))
        return result

    def reset(self) -> None:
        self._samples.clear()
        self._counters.clear()


profiler = PerformanceProfiler()


def profiled(name: str):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with profiler.measure(name):
                return function(*args, **kwargs)

        return wrapped

    return decorate
