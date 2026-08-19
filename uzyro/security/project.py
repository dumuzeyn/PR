from __future__ import annotations

import math
from typing import Any
import zipfile

import numpy as np

from .errors import ResourceLimitError, SecurityValidationError
from .files import validate_dimensions, validate_zip_archive
from .limits import LIMITS
from .paths import validate_archive_member
from .validation import bounded_int, bounded_string, finite_float, validate_json_tree


PROJECT_VERSIONS = {1, 2, 3}
PROJECT_DTYPES = {"uint8", "uint16", "float16", "float32"}
LAYER_KINDS = {"raster", "text", "shape", "adjustment", "linked", "embedded"}


def validate_project_manifest(
    manifest: Any, archive: zipfile.ZipFile, *, infos: dict[str, zipfile.ZipInfo] | None = None,
) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise SecurityValidationError("Манифест проекта должен быть JSON-объектом")
    validate_json_tree(manifest)
    version = bounded_int(manifest.get("format_version", 1), "Версия проекта", 1, max(PROJECT_VERSIONS))
    if version not in PROJECT_VERSIONS:
        raise SecurityValidationError("Версия проекта не поддерживается")
    width = bounded_int(manifest.get("width"), "Ширина проекта", 1, LIMITS.max_dimension)
    height = bounded_int(manifest.get("height"), "Высота проекта", 1, LIMITS.max_dimension)
    validate_dimensions(width, height, copies=2)
    bounded_int(manifest.get("dpi", 300), "DPI", 1, 2400)
    if int(manifest.get("bit_depth", 8)) not in {8, 16, 32}:
        raise SecurityValidationError("Некорректная глубина цвета проекта")
    layers = manifest.get("layers")
    if not isinstance(layers, list) or not layers:
        raise SecurityValidationError("Проект не содержит слоёв")
    if len(layers) > LIMITS.max_layers:
        raise ResourceLimitError("Проект содержит слишком много слоёв")
    infos = validate_zip_archive(archive) if infos is None else infos
    total_arrays = 0
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise SecurityValidationError(f"Слой {index + 1} имеет некорректную структуру")
        bounded_string(layer.get("name", "Слой"), f"Имя слоя {index + 1}", LIMITS.max_name_chars)
        kind = bounded_string(layer.get("kind", "raster"), f"Тип слоя {index + 1}", 32)
        if kind not in LAYER_KINDS:
            raise SecurityValidationError(f"Неизвестный тип слоя: {kind}")
        finite_float(layer.get("opacity", 1.0), "Непрозрачность слоя", 0.0, 1.0)
        finite_float(layer.get("mask_density", 1.0), "Плотность маски", 0.0, 1.0)
        finite_float(layer.get("mask_feather", 0.0), "Растушёвка маски", 0.0, LIMITS.max_dimension)
        for key in ("pixels", "mask", "smart_source", "transform_source", "transform_mask_source", "working_pixels"):
            value = layer.get(key)
            if value is None:
                continue
            if _is_tiled(value):
                total_arrays += validate_tile_descriptor(value, infos)
            elif isinstance(value, str):
                member = validate_archive_member(value).as_posix()
                if member not in infos:
                    raise SecurityValidationError(f"В проекте отсутствуют данные слоя: {member}")
            else:
                raise SecurityValidationError(f"Поле {key} слоя имеет некорректный формат")
    for value in [manifest.get("selection"), *(manifest.get("saved_selections") or {}).values()]:
        if value is None:
            continue
        if _is_tiled(value):
            total_arrays += validate_tile_descriptor(value, infos)
        elif isinstance(value, str) and validate_archive_member(value).as_posix() not in infos:
            raise SecurityValidationError("В проекте отсутствуют данные выделения")
    if total_arrays > LIMITS.memory_budget_bytes():
        raise ResourceLimitError("Проект требует больше памяти, чем разрешено безопасным бюджетом")
    return manifest


def validate_legacy_project(snapshot: Any) -> dict[str, Any]:
    if not isinstance(snapshot, dict):
        raise SecurityValidationError("Старый проект должен содержать JSON-объект")
    validate_json_tree(snapshot)
    width = bounded_int(snapshot.get("width"), "Ширина проекта", 1, LIMITS.max_dimension)
    height = bounded_int(snapshot.get("height"), "Высота проекта", 1, LIMITS.max_dimension)
    validate_dimensions(width, height, copies=2)
    layers = snapshot.get("layers")
    if not isinstance(layers, list) or not layers or len(layers) > LIMITS.max_layers:
        raise ResourceLimitError("Некорректное количество слоёв проекта")
    return snapshot


def validate_tile_descriptor(descriptor: dict[str, Any], infos: dict[str, zipfile.ZipInfo]) -> int:
    shape_raw = descriptor.get("shape")
    if not isinstance(shape_raw, list) or len(shape_raw) not in {2, 3}:
        raise SecurityValidationError("Некорректная форма массива проекта")
    shape = tuple(bounded_int(value, "Размер массива", 1, LIMITS.max_dimension) for value in shape_raw)
    channels = 1 if len(shape) == 2 else shape[2]
    if channels not in {1, 2, 3, 4}:
        raise SecurityValidationError("Некорректное количество каналов массива проекта")
    dtype_name = str(np.dtype(str(descriptor.get("dtype", ""))))
    if dtype_name not in PROJECT_DTYPES:
        raise SecurityValidationError("Тип данных массива проекта не разрешён")
    byte_count = math.prod(shape) * np.dtype(dtype_name).itemsize
    if shape[0] * shape[1] > LIMITS.max_image_pixels or byte_count > LIMITS.memory_budget_bytes():
        raise ResourceLimitError("Массив проекта превышает безопасный лимит памяти")
    tiles = descriptor.get("tiles")
    if not isinstance(tiles, list) or not tiles or len(tiles) > LIMITS.max_tiles_per_array:
        raise ResourceLimitError("Некорректное количество плиток проекта")
    seen_paths: set[str] = set()
    for raw in tiles:
        if not isinstance(raw, dict):
            raise SecurityValidationError("Некорректное описание плитки проекта")
        path = validate_archive_member(str(raw.get("path", ""))).as_posix()
        if path in seen_paths or path not in infos:
            raise SecurityValidationError("Плитка проекта отсутствует или повторяется")
        seen_paths.add(path)
        info = infos[path]
        if info.file_size > LIMITS.max_tile_bytes:
            raise ResourceLimitError("Сжатая плитка проекта превышает безопасный размер")
        digest = str(raw.get("sha256", ""))
        if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
            raise SecurityValidationError("Некорректная контрольная сумма плитки")
        x = bounded_int(raw.get("x"), "X плитки", 0, shape[1] - 1)
        y = bounded_int(raw.get("y"), "Y плитки", 0, shape[0] - 1)
        width = bounded_int(raw.get("width"), "Ширина плитки", 1, shape[1])
        height = bounded_int(raw.get("height"), "Высота плитки", 1, shape[0])
        if x + width > shape[1] or y + height > shape[0]:
            raise SecurityValidationError("Плитка проекта выходит за границы массива")
    return byte_count


def _is_tiled(value: Any) -> bool:
    return isinstance(value, dict) and value.get("format") == "tiles-v1"


__all__ = ["validate_legacy_project", "validate_project_manifest", "validate_tile_descriptor"]
