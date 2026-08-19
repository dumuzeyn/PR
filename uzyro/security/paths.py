from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import re
from typing import Iterable

from .errors import ResourceLimitError, SecurityValidationError


WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL", *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
DANGEROUS_INPUT_SUFFIXES = {
    ".bat", ".cmd", ".com", ".cpl", ".dll", ".exe", ".hta", ".js", ".jse", ".lnk",
    ".msi", ".msp", ".ps1", ".py", ".pyw", ".reg", ".scr", ".url", ".vbe", ".vbs", ".wsf",
}
DEVICE_PREFIXES = ("\\\\.\\", "\\\\?\\", "\\??\\")


def canonical_path(path: str | os.PathLike[str], *, must_exist: bool = False) -> Path:
    raw = os.fspath(path)
    if not raw or "\x00" in raw:
        raise SecurityValidationError("Пустой или повреждённый путь")
    if raw.startswith(DEVICE_PREFIXES):
        raise SecurityValidationError("Пути устройств Windows запрещены")
    candidate = Path(raw).expanduser()
    try:
        resolved = candidate.resolve(strict=must_exist)
    except (OSError, RuntimeError) as exc:
        raise SecurityValidationError("Не удалось безопасно разрешить путь") from exc
    if must_exist and not resolved.exists():
        raise SecurityValidationError("Файл не найден")
    return resolved


def is_within(path: str | Path, root: str | Path) -> bool:
    candidate = canonical_path(path)
    base = canonical_path(root)
    return candidate == base or base in candidate.parents


def ensure_within(path: str | Path, root: str | Path, *, must_exist: bool = False) -> Path:
    candidate = canonical_path(path, must_exist=must_exist)
    base = canonical_path(root, must_exist=must_exist and Path(root).exists())
    if candidate != base and base not in candidate.parents:
        raise SecurityValidationError("Путь выходит за пределы разрешённой папки")
    return candidate


def validate_archive_member(name: str) -> PurePosixPath:
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise SecurityValidationError("Архив содержит некорректный путь")
    if name.startswith(("/", "//")) or re.match(r"^[A-Za-z]:", name):
        raise SecurityValidationError("Архив содержит абсолютный путь")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SecurityValidationError("Архив содержит path traversal")
    return path


def validate_local_input(path: str | Path, *, allowed_suffixes: Iterable[str] | None = None) -> Path:
    resolved = canonical_path(path, must_exist=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise SecurityValidationError("Ожидался обычный локальный файл")
    suffix = resolved.suffix.lower()
    if suffix in DANGEROUS_INPUT_SUFFIXES:
        raise SecurityValidationError("Этот тип файла нельзя открывать как документ")
    if allowed_suffixes is not None and suffix not in {str(item).lower() for item in allowed_suffixes}:
        raise SecurityValidationError("Тип файла не поддерживается этой операцией")
    return resolved


def safe_filename(value: str, *, fallback: str = "file", maximum: int = 180) -> str:
    name = Path(str(value)).name.strip().rstrip(". ")
    if not name or name in {".", ".."}:
        name = fallback
    if any(char in name for char in '<>:"/\\|?*') or "\x00" in name:
        raise SecurityValidationError("Имя файла содержит запрещённые символы")
    if name.split(".", 1)[0].upper() in WINDOWS_RESERVED:
        raise SecurityValidationError("Зарезервированное имя Windows запрещено")
    if len(name) > maximum:
        raise ResourceLimitError("Имя файла слишком длинное")
    return name


def secure_output_path(root: str | Path, filename: str) -> Path:
    base = canonical_path(root)
    target = base / safe_filename(filename)
    return ensure_within(target, base)


__all__ = [
    "DANGEROUS_INPUT_SUFFIXES", "canonical_path", "ensure_within", "is_within", "safe_filename",
    "secure_output_path", "validate_archive_member", "validate_local_input",
]
