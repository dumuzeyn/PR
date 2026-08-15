from __future__ import annotations

import math
from collections.abc import Sequence


Point = tuple[float, float]


def _point(value: Sequence[float]) -> Point:
    return float(value[0]), float(value[1])


def _line_distance(point: Point, start: Point, end: Point) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 1e-12:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    return abs(dy * point[0] - dx * point[1] + end[0] * start[1] - end[1] * start[0]) / length


def _split_cubic(points: tuple[Point, Point, Point, Point]) -> tuple[tuple[Point, ...], tuple[Point, ...]]:
    p0, p1, p2, p3 = points

    def midpoint(first: Point, second: Point) -> Point:
        return (first[0] + second[0]) * 0.5, (first[1] + second[1]) * 0.5

    p01, p12, p23 = midpoint(p0, p1), midpoint(p1, p2), midpoint(p2, p3)
    p012, p123 = midpoint(p01, p12), midpoint(p12, p23)
    center = midpoint(p012, p123)
    return (p0, p01, p012, center), (center, p123, p23, p3)


def cubic_bezier_point(control_points: Sequence[Sequence[float]], amount: float) -> Point:
    if len(control_points) != 4:
        raise ValueError("A cubic Bezier requires four control points")
    points = [_point(value) for value in control_points]
    t = max(0.0, min(1.0, float(amount)))
    inverse = 1.0 - t
    weights = (inverse**3, 3.0 * inverse * inverse * t, 3.0 * inverse * t * t, t**3)
    return tuple(sum(points[index][axis] * weights[index] for index in range(4)) for axis in range(2))  # type: ignore[return-value]


def split_cubic_bezier(control_points: Sequence[Sequence[float]], amount: float) -> tuple[tuple[Point, ...], tuple[Point, ...]]:
    if len(control_points) != 4:
        raise ValueError("A cubic Bezier requires four control points")
    p0, p1, p2, p3 = (_point(value) for value in control_points)
    t = max(0.0, min(1.0, float(amount)))

    def mix(first: Point, second: Point) -> Point:
        return first[0] + (second[0] - first[0]) * t, first[1] + (second[1] - first[1]) * t

    p01, p12, p23 = mix(p0, p1), mix(p1, p2), mix(p2, p3)
    p012, p123 = mix(p01, p12), mix(p12, p23)
    center = mix(p012, p123)
    return (p0, p01, p012, center), (center, p123, p23, p3)


def nearest_cubic_parameter(control_points: Sequence[Sequence[float]], point: Sequence[float]) -> tuple[float, float]:
    target = _point(point)

    def distance_squared(amount: float) -> float:
        current = cubic_bezier_point(control_points, amount)
        return (current[0] - target[0]) ** 2 + (current[1] - target[1]) ** 2

    samples = 48
    best_index = min(range(samples + 1), key=lambda index: distance_squared(index / samples))
    left, right = max(0.0, (best_index - 1) / samples), min(1.0, (best_index + 1) / samples)
    for _ in range(18):
        first = left + (right - left) / 3.0
        second = right - (right - left) / 3.0
        if distance_squared(first) <= distance_squared(second):
            right = second
        else:
            left = first
    amount = (left + right) * 0.5
    return amount, math.sqrt(distance_squared(amount))


def adaptive_cubic_bezier(
    control_points: Sequence[Sequence[float]],
    tolerance: float = 0.25,
    max_depth: int = 16,
) -> list[Point]:
    """Flatten a cubic Bezier until its screen-space chord error is acceptable."""
    if len(control_points) != 4:
        raise ValueError("A cubic Bezier requires four control points")
    cubic = tuple(_point(value) for value in control_points)
    threshold = max(0.01, float(tolerance))
    output: list[Point] = [cubic[0]]

    def visit(points: tuple[Point, Point, Point, Point], depth: int) -> None:
        flatness = max(_line_distance(points[1], points[0], points[3]), _line_distance(points[2], points[0], points[3]))
        if flatness <= threshold or depth >= max_depth:
            output.append(points[3])
            return
        left, right = _split_cubic(points)
        visit(left, depth + 1)
        visit(right, depth + 1)

    visit(cubic, 0)
    return output


def adaptive_bezier_path(nodes: Sequence[dict], tolerance: float = 0.25, closed: bool = False) -> list[Point]:
    if len(nodes) < 2:
        return []
    pairs = [(index, index + 1) for index in range(len(nodes) - 1)]
    if closed:
        pairs.append((len(nodes) - 1, 0))
    output: list[Point] = []
    for first_index, second_index in pairs:
        first, second = nodes[first_index], nodes[second_index]
        first_anchor = first.get("anchor", (0.0, 0.0))
        second_anchor = second.get("anchor", first_anchor)
        points = adaptive_cubic_bezier(
            (first_anchor, first.get("out", first_anchor), second.get("in", second_anchor), second_anchor),
            tolerance,
        )
        output.extend(points if not output else points[1:])
    return output


def rotate_points(points: Sequence[Sequence[float]], center: Point, angle_degrees: float) -> list[Point]:
    if abs(float(angle_degrees)) <= 1e-9:
        return [_point(value) for value in points]
    angle = math.radians(float(angle_degrees))
    cosine, sine = math.cos(angle), math.sin(angle)
    cx, cy = center
    return [
        (cx + (float(value[0]) - cx) * cosine - (float(value[1]) - cy) * sine,
         cy + (float(value[0]) - cx) * sine + (float(value[1]) - cy) * cosine)
        for value in points
    ]


def point_to_polyline_distance(point: Point, points: Sequence[Sequence[float]]) -> float:
    if not points:
        return float("inf")
    if len(points) == 1:
        return math.hypot(point[0] - float(points[0][0]), point[1] - float(points[0][1]))
    best = float("inf")
    for first, second in zip(points, points[1:]):
        ax, ay = float(first[0]), float(first[1])
        bx, by = float(second[0]), float(second[1])
        dx, dy = bx - ax, by - ay
        length_squared = dx * dx + dy * dy
        amount = 0.0 if length_squared <= 1e-12 else max(0.0, min(1.0, ((point[0] - ax) * dx + (point[1] - ay) * dy) / length_squared))
        best = min(best, math.hypot(point[0] - (ax + amount * dx), point[1] - (ay + amount * dy)))
    return best


__all__ = [
    "Point", "adaptive_bezier_path", "adaptive_cubic_bezier", "cubic_bezier_point",
    "nearest_cubic_parameter", "point_to_polyline_distance", "rotate_points", "split_cubic_bezier",
]
