from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .core import Document
from .plugins import PluginAPI


RESULT_PREFIX = "UZYRO_PLUGIN_RESULT="


def _inside(path: Path, roots: list[Path]) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return any(resolved == root or root in resolved.parents for root in roots)


def _explicit_path(path: Path, allowed: list[Path]) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return any(
        resolved == item
        or item in resolved.parents
        or (
            resolved.parent == item.parent
            and (
                resolved.name.startswith(f"{item.name}.")
                or resolved.name.lstrip(".").startswith(f"{item.name}.")
            )
        )
        for item in allowed
    )


def install_permission_guard(entrypoint: Path, permissions: set[str], allowed_paths: list[Path]) -> None:
    runtime_roots = []
    for item in sys.path:
        if item:
            try:
                runtime_roots.append(Path(item).resolve())
            except OSError:
                pass
    read_roots = runtime_roots + [entrypoint.parent.resolve()]
    explicit = [item.resolve() for item in allowed_paths]

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event == "open" and args:
            try:
                path = Path(args[0])
            except TypeError:
                return
            mode = args[1] if len(args) > 1 else "r"
            writing = isinstance(mode, str) and any(flag in mode for flag in "wax+")
            if isinstance(mode, int):
                writing = bool(mode & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if writing:
                if "filesystem.write" not in permissions and not _explicit_path(path, explicit):
                    raise PermissionError(f"Запись запрещена: {path}")
            elif "filesystem.read" not in permissions and not _inside(path, explicit + read_roots):
                raise PermissionError(f"Чтение запрещено: {path}")
        elif event in {"subprocess.Popen", "os.system", "os.posix_spawn"} and "process" not in permissions:
            raise PermissionError("Запуск процессов запрещён")
        elif event.startswith("socket.") and event not in {"socket.__new__"} and "network" not in permissions:
            raise PermissionError("Сетевой доступ запрещён")
        elif event in {"ctypes.dlopen", "ctypes.dlsym"} and "native" not in permissions:
            raise PermissionError("Загрузка нативного кода запрещена")
        elif event in {"os.remove", "os.rename", "os.replace", "os.rmdir"}:
            paths = [Path(value) for value in args[:2] if isinstance(value, (str, bytes, os.PathLike))]
            denied = [path for path in paths if not _explicit_path(path, explicit)]
            if "filesystem.write" not in permissions and denied:
                raise PermissionError(f"Изменение файлов запрещено: {denied[0]}")

    sys.addaudithook(audit)


def load_plugin(entrypoint: Path, permissions: set[str], allowed_paths: list[Path]) -> PluginAPI:
    install_permission_guard(entrypoint, permissions, allowed_paths)
    module_name = f"uzyro_isolated_plugin_{entrypoint.stem}_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, entrypoint)
    if spec is None or spec.loader is None:
        raise ImportError("Не удалось создать модуль плагина")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    register = getattr(module, "register", None)
    if not callable(register):
        raise ValueError("Плагин должен экспортировать register(api)")
    api = PluginAPI(permissions)
    register(api)
    return api


def execute(request: dict[str, Any]) -> dict[str, Any]:
    entrypoint = Path(request["entrypoint"]).resolve()
    permissions = {str(item) for item in request.get("permissions", [])}
    args = [str(item) for item in request.get("args", [])]
    allowed_paths = [Path(item) for item in request.get("allowed_paths", [])]
    api = load_plugin(entrypoint, permissions, allowed_paths)
    operation = str(request.get("operation", "inspect"))
    if operation == "inspect":
        return api.metadata()
    if operation == "filter":
        name, source, target, parameters = args
        callback = api.filters[name][0]
        pixels = np.load(source, allow_pickle=False)
        params = json.loads(Path(parameters).read_text(encoding="utf-8"))
        output = np.asarray(callback(pixels.copy(), params))
        np.save(target, output, allow_pickle=False)
        return {"ok": True}
    if operation == "action":
        name, source, target, parameters = args
        document = Document.open_project(source)
        params = json.loads(Path(parameters).read_text(encoding="utf-8"))
        api.actions[name][0](document, params)
        document.save_project(target)
        return {"ok": True}
    if operation == "import":
        name, source, target, parameters = args
        params = json.loads(Path(parameters).read_text(encoding="utf-8"))
        document = api.importers[name][0](source, params)
        if not isinstance(document, Document):
            raise TypeError("Импортёр должен вернуть Document")
        document.save_project(target)
        return {"ok": True}
    if operation == "export":
        name, source, target, parameters = args
        document = Document.open_project(source)
        params = json.loads(Path(parameters).read_text(encoding="utf-8"))
        api.exporters[name][0](document, target, params)
        return {"ok": True}
    raise ValueError(f"Неизвестная операция host: {operation}")


def main() -> None:
    try:
        request = json.loads(sys.stdin.read())
        result = execute(request)
        print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
