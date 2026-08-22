from __future__ import annotations

from dataclasses import dataclass

from .document import Document
from .geometry_ops import layer_contains_point
from .layer import Layer


@dataclass(frozen=True)
class ObjectHit:
    object_id: str
    layer_id: str
    layer_index: int
    kind: str
    local_position: tuple[int, int]
    locked: bool


def hit_test_document(
    document: Document,
    point: tuple[int, int],
    tolerance: int = 0,
    *,
    include_locked: bool = True,
) -> ObjectHit | None:
    """Return the topmost visible object under a document-space point."""
    hits = hit_test_stack(document, point, tolerance, include_locked=include_locked)
    return hits[0] if hits else None


def hit_test_stack(
    document: Document,
    point: tuple[int, int],
    tolerance: int = 0,
    *,
    include_locked: bool = True,
) -> list[ObjectHit]:
    """Return all visible hits from front to back using the shared object rules."""
    hits: list[ObjectHit] = []
    for index in range(len(document.layers) - 1, -1, -1):
        layer = document.layers[index]
        if not layer.visible or (layer.locked and not include_locked):
            continue
        if layer_contains_point(layer, point, tolerance):
            hits.append(ObjectHit(
                object_id=layer.id,
                layer_id=layer.id,
                layer_index=index,
                kind=layer.kind,
                local_position=(point[0] - layer.x, point[1] - layer.y),
                locked=bool(layer.locked),
            ))
    return hits


def layers_inside_box(document: Document, box: tuple[int, int, int, int]) -> list[Layer]:
    x1, y1, x2, y2 = box
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    selected: list[Layer] = []
    for layer in document.layers:
        if not layer.visible or layer.locked or layer.kind not in {"shape", "text"}:
            continue
        alpha = layer.pixels[:, :, 3]
        if not alpha.any():
            continue
        ys, xs = alpha.nonzero()
        bounds = (int(xs.min()) + layer.x, int(ys.min()) + layer.y, int(xs.max() + 1) + layer.x, int(ys.max() + 1) + layer.y)
        if bounds[0] >= left and bounds[1] >= top and bounds[2] <= right and bounds[3] <= bottom:
            selected.append(layer)
    return selected


__all__ = ["ObjectHit", "hit_test_document", "hit_test_stack", "layers_inside_box"]
