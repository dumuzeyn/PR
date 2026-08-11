from __future__ import annotations

import base64
import io
import zlib
from dataclasses import fields
from typing import Any

import numpy as np

from .layer import Layer


def encode_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False)
        payload = zlib.compress(buffer.getvalue(), level=6)
        return {
            "__type__": "ndarray",
            "data": base64.b64encode(payload).decode("ascii"),
        }
    if isinstance(value, tuple):
        return {"__type__": "tuple", "items": [encode_value(item) for item in value]}
    if isinstance(value, list):
        return [encode_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): encode_value(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Неподдерживаемое значение действия: {type(value).__name__}")


def decode_value(value: Any) -> Any:
    if isinstance(value, list):
        return [decode_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    kind = value.get("__type__")
    if kind == "tuple":
        return tuple(decode_value(item) for item in value.get("items", []))
    if kind == "ndarray":
        raw = zlib.decompress(base64.b64decode(str(value["data"]), validate=True))
        return np.load(io.BytesIO(raw), allow_pickle=False)
    return {key: decode_value(item) for key, item in value.items()}


def layer_to_data(layer: Layer) -> dict[str, Any]:
    ignored = {"pixels_revision", "mask_revision"}
    return {
        item.name: encode_value(getattr(layer, item.name))
        for item in fields(Layer)
        if item.name not in ignored
    }


def layer_from_data(data: dict[str, Any], *, layer_id: str | None = None) -> Layer:
    values = {key: decode_value(value) for key, value in data.items()}
    if layer_id is not None:
        values["id"] = layer_id
    return Layer(**values)


__all__ = ["decode_value", "encode_value", "layer_from_data", "layer_to_data"]
