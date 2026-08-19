from __future__ import annotations

import json
import os
from pathlib import Path
import uuid
from typing import Any

from .validation import load_bounded_json, validate_json_tree


def load_settings_file(path: str | Path) -> dict[str, Any]:
    data = load_bounded_json(path, maximum=8 * 1024 * 1024)
    if not isinstance(data, dict):
        raise ValueError("Настройки должны содержать JSON-объект")
    return data


def save_settings_file(path: str | Path, payload: dict[str, Any]) -> None:
    validate_json_tree(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = ["load_settings_file", "save_settings_file"]
