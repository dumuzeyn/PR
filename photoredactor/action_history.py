from __future__ import annotations

import copy
import uuid
from typing import Any

import numpy as np

from .action_codec import decode_value, encode_value, layer_from_data, layer_to_data
from .core import Document, render_shape_layer, render_text_layer
from .history import (
    DocumentFieldsCommand,
    DocumentStateCommand,
    LayerBlendModeCommand,
    LayerDeleteCommand,
    LayerFieldsCommand,
    LayerInsertCommand,
    LayerMoveCommand,
    LayerOpacityCommand,
    LayerPropertyCommand,
    LayerReorderCommand,
    LayerVisibilityCommand,
    LayersDeleteCommand,
    MaskPatchCommand,
    MaskTilePatchCommand,
    PixelPatchCommand,
    PixelTilePatchCommand,
    SelectionMaskCommand,
    ShapeDataCommand,
    TextDataCommand,
)


def _selector(document: Document, layer_id: str) -> dict[str, Any]:
    index = next((i for i, layer in enumerate(document.layers) if layer.id == layer_id), document.active_layer)
    return {"id": layer_id, "index": index}


def _patch_data(patch: Any) -> dict[str, Any]:
    return {"rect": list(patch.rect), "after": encode_value(patch.after)}


def history_command_to_payload(command: Any, document: Document) -> dict[str, Any]:
    payload: dict[str, Any] = {"kind": type(command).__name__}
    layer_id = getattr(command, "layer_id", None)
    if layer_id:
        payload["layer"] = _selector(document, layer_id)

    if isinstance(command, PixelPatchCommand):
        payload.update(rect=list(command.rect), value=encode_value(command.after))
        if command.precision_after is not None:
            payload["precision"] = encode_value(command.precision_after)
    elif isinstance(command, PixelTilePatchCommand):
        payload["patches"] = [_patch_data(item) for item in command.patches]
        payload["precision_patches"] = [_patch_data(item) for item in command.precision_patches or []]
    elif isinstance(command, MaskPatchCommand):
        payload.update(rect=list(command.rect), value=encode_value(command.after))
    elif isinstance(command, MaskTilePatchCommand):
        payload["patches"] = [_patch_data(item) for item in command.patches]
    elif isinstance(command, (LayerPropertyCommand,)):
        payload.update(attribute=command.attribute, value=encode_value(command.after))
    elif isinstance(command, LayerFieldsCommand):
        payload["values"] = encode_value(command.after)
    elif isinstance(command, TextDataCommand):
        payload.update(data=encode_value(command.after), name=getattr(command, "after_name", document.layer.name))
    elif isinstance(command, ShapeDataCommand):
        payload.update(data=encode_value(command.after), name=command.after_name)
    elif isinstance(command, DocumentFieldsCommand):
        payload["values"] = encode_value(command.after)
    elif isinstance(command, LayerInsertCommand):
        payload.update(index=command.index, layer_data=layer_to_data(command.layer))
    elif isinstance(command, LayerDeleteCommand):
        payload["index"] = command.index
    elif isinstance(command, LayersDeleteCommand):
        payload["indices"] = [index for index, _layer in command.layers]
    elif isinstance(command, LayerReorderCommand):
        payload["index"] = command.after
    elif isinstance(command, LayerMoveCommand):
        payload.update(position=list(command.after), mask=encode_value(command.after_mask))
    elif isinstance(command, LayerOpacityCommand):
        payload.update(attribute="opacity", value=command.after)
    elif isinstance(command, LayerVisibilityCommand):
        payload.update(attribute="visible", value=command.after)
    elif isinstance(command, LayerBlendModeCommand):
        payload.update(attribute="blend_mode", value=command.after)
    elif isinstance(command, DocumentStateCommand):
        payload["state"] = encode_value(command.after)
    elif isinstance(command, SelectionMaskCommand):
        payload["selection"] = encode_value(command.after)
    else:
        raise TypeError(f"Команда {type(command).__name__} пока не поддерживает запись")
    return payload


def _resolve_layer(document: Document, selector: dict[str, Any], context: dict[str, Any]):
    source_id = str(selector.get("id", ""))
    mapped_id = context.setdefault("layer_ids", {}).get(source_id, source_id)
    layer = document.get_layer(mapped_id)
    if layer is not None:
        return layer
    index = max(0, min(int(selector.get("index", document.active_layer)), len(document.layers) - 1))
    return document.layers[index]


def _write_patch(target: np.ndarray, rect: list[int], value: np.ndarray) -> None:
    x1, y1, x2, y2 = map(int, rect)
    if x1 < 0 or y1 < 0 or x2 > target.shape[1] or y2 > target.shape[0]:
        raise ValueError("Область записанного инструмента выходит за границы текущего слоя")
    if value.shape != target[y1:y2, x1:x2].shape:
        raise ValueError("Размер записанного штриха не совпадает с текущим слоем")
    target[y1:y2, x1:x2] = value


def _apply_layer_values(layer: Any, values: dict[str, Any]) -> None:
    for attribute, value in values.items():
        setattr(layer, attribute, copy.deepcopy(value))
    if "pixels" in values or "working_pixels" in values:
        layer.touch_pixels()
    if "mask" in values or "mask_feather" in values:
        layer.touch_mask()


def apply_history_payload(document: Document, payload: dict[str, Any], context: dict[str, Any] | None = None) -> None:
    context = context if context is not None else {}
    kind = str(payload.get("kind", ""))
    selector = dict(payload.get("layer") or {})
    layer = _resolve_layer(document, selector, context) if selector else None

    if kind in {"PixelPatchCommand", "MaskPatchCommand"}:
        target = layer.pixels if kind == "PixelPatchCommand" else layer.mask
        if target is None and kind == "MaskPatchCommand":
            layer.mask = np.full(layer.pixels.shape[:2], 255, dtype=np.uint8)
            target = layer.mask
        _write_patch(target, payload["rect"], decode_value(payload["value"]))
        if kind == "PixelPatchCommand" and payload.get("precision") is not None and layer.working_pixels is not None:
            precise = layer.working_rgba()
            _write_patch(precise, payload["rect"], decode_value(payload["precision"]))
            layer.set_working_rgba(precise, layer.working_depth, layer.working_model)
        elif kind == "PixelPatchCommand":
            layer.touch_pixels()
        else:
            layer.touch_mask()
    elif kind in {"PixelTilePatchCommand", "MaskTilePatchCommand"}:
        is_mask = kind == "MaskTilePatchCommand"
        if is_mask and layer.mask is None:
            layer.mask = np.full(layer.pixels.shape[:2], 255, dtype=np.uint8)
        target = layer.mask if is_mask else layer.pixels
        for patch in payload.get("patches", []):
            _write_patch(target, patch["rect"], decode_value(patch["after"]))
        if not is_mask and payload.get("precision_patches") and layer.working_pixels is not None:
            precise = layer.working_rgba()
            for patch in payload["precision_patches"]:
                _write_patch(precise, patch["rect"], decode_value(patch["after"]))
            layer.set_working_rgba(precise, layer.working_depth, layer.working_model)
        elif is_mask:
            layer.touch_mask()
        else:
            layer.touch_pixels()
    elif kind in {"LayerPropertyCommand", "LayerOpacityCommand", "LayerVisibilityCommand", "LayerBlendModeCommand"}:
        setattr(layer, str(payload["attribute"]), decode_value(payload["value"]))
        if payload["attribute"] == "mask":
            layer.touch_mask()
    elif kind == "LayerFieldsCommand":
        _apply_layer_values(layer, decode_value(payload["values"]))
    elif kind in {"TextDataCommand", "ShapeDataCommand"}:
        data = decode_value(payload["data"])
        if kind == "TextDataCommand":
            layer.text_data = data
            render_text_layer(layer)
        else:
            layer.shape_data = data
            render_shape_layer(layer)
        layer.name = str(payload.get("name", layer.name))
        layer.touch_pixels()
    elif kind == "DocumentFieldsCommand":
        for attribute, value in decode_value(payload["values"]).items():
            setattr(document, attribute, copy.deepcopy(value))
    elif kind == "LayerInsertCommand":
        original_id = str(decode_value(payload["layer_data"]).get("id", ""))
        new_id = original_id if document.get_layer(original_id) is None else uuid.uuid4().hex
        inserted = layer_from_data(payload["layer_data"], layer_id=new_id)
        index = max(0, min(int(payload.get("index", len(document.layers))), len(document.layers)))
        document.layers.insert(index, inserted)
        document.active_layer = index
        context.setdefault("layer_ids", {})[original_id] = new_id
    elif kind in {"LayerDeleteCommand", "LayersDeleteCommand"}:
        indices = payload.get("indices", [payload.get("index", document.active_layer)])
        for index in sorted({int(item) for item in indices}, reverse=True):
            if len(document.layers) > 1 and 0 <= index < len(document.layers):
                document.layers.pop(index)
        document.active_layer = min(document.active_layer, len(document.layers) - 1)
    elif kind == "LayerReorderCommand":
        current = document.layers.index(layer)
        item = document.layers.pop(current)
        index = max(0, min(int(payload["index"]), len(document.layers)))
        document.layers.insert(index, item)
        document.active_layer = index
    elif kind == "LayerMoveCommand":
        layer.x, layer.y = map(int, payload["position"])
        mask = decode_value(payload.get("mask"))
        if mask is not None:
            layer.mask = mask
            layer.touch_mask()
    elif kind == "DocumentStateCommand":
        document.restore_raw_state(decode_value(payload["state"]))
    elif kind == "SelectionMaskCommand":
        selection = decode_value(payload.get("selection"))
        document.selection_mask = None if selection is None else selection.copy()
    else:
        raise ValueError(f"Неизвестный тип записанной операции: {kind}")
    document.dirty = True


__all__ = ["apply_history_payload", "history_command_to_payload"]
