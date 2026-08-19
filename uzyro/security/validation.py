from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any

from .errors import ResourceLimitError, SecurityValidationError
from .limits import LIMITS


IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?$")


def bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise SecurityValidationError(f"{name}: ожидалось целое число")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SecurityValidationError(f"{name}: некорректное целое число") from exc
    if result < minimum or result > maximum:
        raise ResourceLimitError(f"{name}: значение вне безопасного диапазона")
    return result


def finite_float(value: Any, name: str, minimum: float, maximum: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SecurityValidationError(f"{name}: некорректное число") from exc
    if not math.isfinite(result):
        raise SecurityValidationError(f"{name}: NaN и Infinity запрещены")
    if result < minimum or result > maximum:
        raise ResourceLimitError(f"{name}: значение вне безопасного диапазона")
    return result


def bounded_string(value: Any, name: str, maximum: int = LIMITS.max_string_chars) -> str:
    if not isinstance(value, str):
        raise SecurityValidationError(f"{name}: ожидалась строка")
    if len(value) > maximum:
        raise ResourceLimitError(f"{name}: строка слишком длинная")
    if "\x00" in value:
        raise SecurityValidationError(f"{name}: нулевой символ запрещён")
    return value


def validate_identifier(value: Any, name: str = "Идентификатор") -> str:
    result = bounded_string(value, name, 128)
    if not IDENTIFIER_RE.fullmatch(result):
        raise SecurityValidationError(f"{name}: разрешены латинские буквы, цифры, '.', '_' и '-'")
    return result


def validate_json_tree(value: Any) -> Any:
    nodes = 0
    stack: list[tuple[Any, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > LIMITS.max_json_nodes:
            raise ResourceLimitError("JSON содержит слишком много элементов")
        if depth > LIMITS.max_json_depth:
            raise ResourceLimitError("JSON имеет слишком большую вложенность")
        if isinstance(current, dict):
            for key, child in current.items():
                bounded_string(key, "Ключ JSON", LIMITS.max_name_chars)
                stack.append((child, depth + 1))
        elif isinstance(current, list):
            stack.extend((child, depth + 1) for child in current)
        elif isinstance(current, str):
            bounded_string(current, "Строка JSON")
        elif isinstance(current, float) and not math.isfinite(current):
            raise SecurityValidationError("JSON содержит NaN или Infinity")
        elif current is not None and not isinstance(current, (bool, int, float)):
            raise SecurityValidationError(f"JSON содержит неподдерживаемый тип: {type(current).__name__}")
    return value


def loads_bounded_json(payload: bytes | str, *, maximum: int = LIMITS.max_json_bytes) -> Any:
    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    if len(raw) > maximum:
        raise ResourceLimitError("JSON-файл превышает безопасный размер")
    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=lambda item: (_ for _ in ()).throw(
            SecurityValidationError(f"Недопустимое числовое значение: {item}")
        ))
    except UnicodeDecodeError as exc:
        raise SecurityValidationError("JSON должен быть в UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise SecurityValidationError(f"Повреждённый JSON: {exc.msg}") from exc
    return validate_json_tree(value)


def load_bounded_json(path: str | Path, *, maximum: int = LIMITS.max_json_bytes) -> Any:
    source = Path(path)
    if source.stat().st_size > maximum:
        raise ResourceLimitError("JSON-файл превышает безопасный размер")
    return loads_bounded_json(source.read_bytes(), maximum=maximum)


__all__ = [
    "bounded_int", "bounded_string", "finite_float", "load_bounded_json", "loads_bounded_json",
    "validate_identifier", "validate_json_tree",
]
