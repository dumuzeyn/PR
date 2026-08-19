from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from typing import Iterator

from .paths import canonical_path


def temporary_root() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "UZYRO" / "Temp"
    base.mkdir(parents=True, exist_ok=True)
    return canonical_path(base, must_exist=True)


@contextmanager
def secure_temporary_directory(prefix: str) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix=prefix, dir=temporary_root()) as directory:
        yield Path(directory).resolve()


__all__ = ["secure_temporary_directory", "temporary_root"]
