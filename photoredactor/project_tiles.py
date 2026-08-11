from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Callable
import zipfile

import numpy as np
from PIL import Image


PROJECT_FORMAT_VERSION = 3
PROJECT_TILE_SIZE = 512
TileProgress = Callable[[int, int], None]


def _encode_tile(tile: np.ndarray, codec: str) -> bytes:
    buffer = io.BytesIO()
    if codec == "png-l":
        Image.fromarray(tile, "L").save(buffer, "PNG", compress_level=3)
    elif codec == "png-rgba":
        Image.fromarray(tile, "RGBA").save(buffer, "PNG", compress_level=3)
    elif codec == "png-rgb":
        Image.fromarray(tile, "RGB").save(buffer, "PNG", compress_level=3)
    else:
        np.save(buffer, tile, allow_pickle=False)
    return buffer.getvalue()


def _decode_tile(payload: bytes, codec: str) -> np.ndarray:
    if codec.startswith("png-"):
        mode = {"png-l": "L", "png-rgb": "RGB", "png-rgba": "RGBA"}[codec]
        with Image.open(io.BytesIO(payload)) as image:
            return np.asarray(image.convert(mode)).copy()
    return np.load(io.BytesIO(payload), allow_pickle=False)


def _codec_for(array: np.ndarray) -> str:
    if array.dtype == np.uint8 and array.ndim == 2:
        return "png-l"
    if array.dtype == np.uint8 and array.ndim == 3 and array.shape[2] == 4:
        return "png-rgba"
    if array.dtype == np.uint8 and array.ndim == 3 and array.shape[2] == 3:
        return "png-rgb"
    return "npy"


def write_tiled_array(
    archive: zipfile.ZipFile,
    prefix: str,
    value: np.ndarray,
    tile_size: int = PROJECT_TILE_SIZE,
) -> dict[str, Any]:
    array = np.asarray(value)
    if array.ndim not in {2, 3}:
        raise ValueError("Project tile arrays must have two or three dimensions")
    height, width = array.shape[:2]
    tile_size = max(64, int(tile_size))
    codec = _codec_for(array)
    extension = "png" if codec.startswith("png-") else "npy"
    tiles: list[dict[str, Any]] = []
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile = np.ascontiguousarray(array[y : y + tile_size, x : x + tile_size])
            payload = _encode_tile(tile, codec)
            tile_path = f"{prefix}/{y // tile_size:05d}_{x // tile_size:05d}.{extension}"
            archive.writestr(tile_path, payload)
            tiles.append(
                {
                    "x": x,
                    "y": y,
                    "width": int(tile.shape[1]),
                    "height": int(tile.shape[0]),
                    "path": tile_path,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    return {
        "format": "tiles-v1",
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "tile_size": tile_size,
        "codec": codec,
        "tiles": tiles,
    }


def is_tiled_array(value: object) -> bool:
    return (
        isinstance(value, dict)
        and value.get("format") == "tiles-v1"
        and isinstance(value.get("shape"), list)
        and isinstance(value.get("tiles"), list)
        and "dtype" in value
    )


def read_tiled_array(
    archive: zipfile.ZipFile,
    descriptor: dict[str, Any],
    progress: TileProgress | None = None,
) -> np.ndarray:
    if not is_tiled_array(descriptor):
        raise ValueError("Unsupported project tile descriptor")
    shape = tuple(int(item) for item in descriptor.get("shape", []))
    if len(shape) not in {2, 3} or any(item <= 0 for item in shape) or shape[0] > 100000 or shape[1] > 100000:
        raise ValueError("Invalid project tile dimensions")
    try:
        dtype = np.dtype(str(descriptor["dtype"]))
    except (KeyError, TypeError) as exc:
        raise ValueError("Invalid project tile data type") from exc
    codec = str(descriptor.get("codec", "npy"))
    tiles = descriptor.get("tiles")
    if not isinstance(tiles, list) or not tiles:
        raise ValueError("Project tile list is empty")
    result = np.zeros(shape, dtype=dtype)
    names = set(archive.namelist())
    total = len(tiles)
    covered = np.zeros(shape[:2], dtype=np.bool_)
    for index, raw in enumerate(tiles, 1):
        path = str(raw.get("path", ""))
        if not path or path not in names:
            raise ValueError(f"Project tile is missing: {path or '<unknown>'}")
        payload = archive.read(path)
        if hashlib.sha256(payload).hexdigest() != str(raw.get("sha256", "")):
            raise ValueError(f"Project tile checksum mismatch: {path}")
        x, y = int(raw.get("x", -1)), int(raw.get("y", -1))
        width, height = int(raw.get("width", 0)), int(raw.get("height", 0))
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > shape[1] or y + height > shape[0]:
            raise ValueError(f"Project tile bounds are invalid: {path}")
        tile = _decode_tile(payload, codec)
        expected = (height, width) + shape[2:]
        if tile.shape != expected or tile.dtype != dtype:
            raise ValueError(f"Project tile payload does not match its descriptor: {path}")
        if np.any(covered[y : y + height, x : x + width]):
            raise ValueError(f"Project tiles overlap: {path}")
        result[y : y + height, x : x + width] = tile
        covered[y : y + height, x : x + width] = True
        if progress is not None:
            progress(index, total)
    if not np.all(covered):
        raise ValueError("Project tile set does not cover the complete image")
    return np.ascontiguousarray(result)


def tiled_payload_stats(descriptor: object) -> dict[str, int]:
    if not is_tiled_array(descriptor):
        return {"tiles": 0, "bytes": 0}
    shape = tuple(int(item) for item in descriptor.get("shape", []))
    byte_count = int(np.prod(shape, dtype=np.int64)) * np.dtype(str(descriptor.get("dtype", "uint8"))).itemsize
    return {"tiles": len(descriptor.get("tiles", [])), "bytes": byte_count}


def temporary_project_path(path: str | Path) -> Path:
    target = Path(path)
    return target.with_name(f".{target.name}.saving")


__all__ = [name for name in globals() if not name.startswith("__")]
