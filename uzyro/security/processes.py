from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Mapping, Sequence

from .errors import ResourceLimitError, SecurityValidationError
from .limits import LIMITS
from .paths import canonical_path


@dataclass(frozen=True)
class ProcessResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def validate_executable(path: str | Path, *, allow_python: bool = False) -> Path:
    executable = canonical_path(path, must_exist=True)
    if not executable.is_file() or executable.is_symlink():
        raise SecurityValidationError("Исполняемый файл отсутствует или является ссылкой")
    suffix = executable.suffix.lower()
    allowed = {".exe"} if os.name == "nt" else {"", ".bin", ".run"}
    if allow_python and executable == canonical_path(sys.executable, must_exist=True):
        return executable
    if suffix not in allowed:
        raise SecurityValidationError("Разрешён только прямой запуск исполняемого файла")
    return executable


def run_checked(
    args: Sequence[str | os.PathLike[str]], *, timeout: float, input_data: str | bytes | None = None,
    maximum_output: int = LIMITS.max_process_output_bytes, env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None, allow_python: bool = False,
) -> ProcessResult:
    if not args:
        raise SecurityValidationError("Команда запуска пуста")
    executable = validate_executable(args[0], allow_python=allow_python)
    command = [str(executable), *(os.fspath(item) for item in args[1:])]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE if input_data is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        cwd=None if cwd is None else str(canonical_path(cwd, must_exist=True)),
        env=None if env is None else dict(env),
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    overflow = threading.Event()
    outputs = {"stdout": bytearray(), "stderr": bytearray()}

    def read_stream(name: str, stream) -> None:
        while chunk := stream.read(65_536):
            remaining = maximum_output - len(outputs[name])
            if remaining > 0:
                outputs[name].extend(chunk[:remaining])
            if len(chunk) > remaining:
                overflow.set()
                return

    threads = [
        threading.Thread(target=read_stream, args=("stdout", process.stdout), daemon=True),
        threading.Thread(target=read_stream, args=("stderr", process.stderr), daemon=True),
    ]
    for thread in threads:
        thread.start()
    if input_data is not None and process.stdin is not None:
        payload = input_data.encode("utf-8") if isinstance(input_data, str) else input_data
        process.stdin.write(payload)
        process.stdin.close()
    deadline = time.monotonic() + max(0.1, float(timeout))
    while process.poll() is None:
        if overflow.is_set() or time.monotonic() >= deadline:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
            if overflow.is_set():
                raise ResourceLimitError("Дочерний процесс превысил лимит вывода")
            raise TimeoutError("Дочерний процесс превысил время выполнения")
        time.sleep(0.03)
    for thread in threads:
        thread.join(timeout=2)
    return ProcessResult(
        tuple(command), int(process.returncode),
        bytes(outputs["stdout"]).decode("utf-8", errors="replace"),
        bytes(outputs["stderr"]).decode("utf-8", errors="replace"),
    )


__all__ = ["ProcessResult", "run_checked", "validate_executable"]
