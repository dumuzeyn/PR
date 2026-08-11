from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def normalized_grid_positions(values: Sequence[float] | None, count: int) -> list[float]:
    """Return a valid monotonic source-grid axis in the 0..1 range."""
    count = max(2, int(count))
    if values is None or len(values) != count:
        return np.linspace(0.0, 1.0, count).tolist()
    positions = [float(np.clip(value, 0.0, 1.0)) for value in values]
    if positions[0] != 0.0 or positions[-1] != 1.0:
        return np.linspace(0.0, 1.0, count).tolist()
    if any(right - left < 1e-5 for left, right in zip(positions, positions[1:])):
        return np.linspace(0.0, 1.0, count).tolist()
    return positions


def regular_grid_points(
    bounds: tuple[float, float, float, float],
    row_positions: Sequence[float],
    column_positions: Sequence[float],
) -> list[list[float]]:
    x, y, width, height = bounds
    return [
        [x + width * column, y + height * row]
        for row in row_positions
        for column in column_positions
    ]


def insert_grid_line(
    points: Sequence[Sequence[float]],
    rows: int,
    columns: int,
    row_positions: Sequence[float],
    column_positions: Sequence[float],
    axis: str,
    interval: int,
) -> tuple[list[list[float]], int, int, list[float], list[float], int]:
    """Insert a source-grid line and interpolate its destination control points."""
    grid = np.asarray(points, dtype=np.float64).reshape(rows, columns, 2)
    row_values = normalized_grid_positions(row_positions, rows)
    column_values = normalized_grid_positions(column_positions, columns)
    if axis == "row":
        interval = min(max(0, int(interval)), rows - 2)
        position = (row_values[interval] + row_values[interval + 1]) / 2.0
        ratio = (position - row_values[interval]) / (row_values[interval + 1] - row_values[interval])
        inserted = grid[interval] * (1.0 - ratio) + grid[interval + 1] * ratio
        grid = np.insert(grid, interval + 1, inserted, axis=0)
        row_values.insert(interval + 1, position)
        return grid.reshape(-1, 2).tolist(), rows + 1, columns, row_values, column_values, (interval + 1) * columns
    if axis != "column":
        raise ValueError("Grid axis must be row or column")
    interval = min(max(0, int(interval)), columns - 2)
    position = (column_values[interval] + column_values[interval + 1]) / 2.0
    ratio = (position - column_values[interval]) / (column_values[interval + 1] - column_values[interval])
    inserted = grid[:, interval] * (1.0 - ratio) + grid[:, interval + 1] * ratio
    grid = np.insert(grid, interval + 1, inserted, axis=1)
    column_values.insert(interval + 1, position)
    return grid.reshape(-1, 2).tolist(), rows, columns + 1, row_values, column_values, interval + 1


def remove_grid_line(
    points: Sequence[Sequence[float]],
    rows: int,
    columns: int,
    row_positions: Sequence[float],
    column_positions: Sequence[float],
    axis: str,
    index: int,
) -> tuple[list[list[float]], int, int, list[float], list[float], int]:
    """Remove one interior line while keeping at least a 2x2 grid."""
    grid = np.asarray(points, dtype=np.float64).reshape(rows, columns, 2)
    row_values = normalized_grid_positions(row_positions, rows)
    column_values = normalized_grid_positions(column_positions, columns)
    if axis == "row":
        if rows <= 2 or not 0 < index < rows - 1:
            raise ValueError("Only an interior row can be removed")
        grid = np.delete(grid, index, axis=0)
        row_values.pop(index)
        selected = min(index, rows - 2) * columns
        return grid.reshape(-1, 2).tolist(), rows - 1, columns, row_values, column_values, selected
    if axis != "column":
        raise ValueError("Grid axis must be row or column")
    if columns <= 2 or not 0 < index < columns - 1:
        raise ValueError("Only an interior column can be removed")
    grid = np.delete(grid, index, axis=1)
    column_values.pop(index)
    selected = min(index, columns - 2)
    return grid.reshape(-1, 2).tolist(), rows, columns - 1, row_values, column_values, selected
