from __future__ import annotations

from pathlib import Path
import copy
import statistics
import sys
import time
import tracemalloc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photoredactor.core import Document
from photoredactor.history import LayerInsertCommand


def run_case(legacy_snapshots: bool, count: int = 100) -> dict[str, float]:
    document = Document.new(256, 256, (0, 0, 0, 0))
    commands = []
    samples = []
    tracemalloc.start()
    started = time.perf_counter()
    for index in range(count):
        item_started = time.perf_counter()
        if legacy_snapshots:
            before = document.raw_state()
        shape = "line" if index % 2 == 0 else "ellipse"
        layer = document.add_shape_layer(shape, (8 + index, 8, 80 + index, 72), (40, 150, 240, 255), (10, 20, 30, 255), 2)
        if legacy_snapshots:
            commands.append((before, document.raw_state()))
        else:
            commands.append(LayerInsertCommand("Shape", document.active_layer, copy.deepcopy(layer)))
        samples.append((time.perf_counter() - item_started) * 1000.0)
    total_ms = (time.perf_counter() - started) * 1000.0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "total_ms": total_ms,
        "median_ms": statistics.median(samples),
        "second_ms": samples[1],
        "eightieth_ms": samples[79],
        "growth_ratio": samples[79] / max(0.0001, samples[1]),
        "peak_mb": peak / (1024.0 * 1024.0),
    }


def main() -> None:
    for label, legacy in (("before/full snapshots", True), ("after/insert command", False)):
        result = run_case(legacy)
        print(
            f"{label}: total={result['total_ms']:.1f} ms, median={result['median_ms']:.3f} ms, "
            f"2nd={result['second_ms']:.3f} ms, 80th={result['eightieth_ms']:.3f} ms, "
            f"growth={result['growth_ratio']:.2f}x, peak={result['peak_mb']:.1f} MB"
        )


if __name__ == "__main__":
    main()
