from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable

import numpy as np

from .core import Document


PLUGIN_API_VERSION = 2
MANIFEST_FORMAT = "PhotoRedactor plugin v1"
KNOWN_PERMISSIONS = {"pixels", "document", "filesystem.read", "filesystem.write", "network", "process", "native"}


@dataclass
class PluginInfo:
    id: str
    name: str
    version: str
    api_version: int
    entrypoint: Path
    requested_permissions: set[str] = field(default_factory=set)
    granted_permissions: set[str] = field(default_factory=set)
    description: str = ""
    legacy: bool = False

    @property
    def blocked_permissions(self) -> set[str]:
        return self.requested_permissions - self.granted_permissions


@dataclass
class PluginExtension:
    name: str
    plugin_id: str
    description: str = ""
    extensions: tuple[str, ...] = ()


class PluginAPI:
    """Stable SDK surface used only inside the isolated plugin host."""

    def __init__(self, permissions: set[str]) -> None:
        self.api_version = PLUGIN_API_VERSION
        self.permissions = frozenset(permissions)
        self.filters: dict[str, tuple[Callable, str]] = {}
        self.actions: dict[str, tuple[Callable, str]] = {}
        self.importers: dict[str, tuple[Callable, str, tuple[str, ...]]] = {}
        self.exporters: dict[str, tuple[Callable, str, tuple[str, ...]]] = {}

    def _require(self, permission: str) -> None:
        if permission not in self.permissions:
            raise PermissionError(f"Плагину не выдано разрешение {permission}")

    @staticmethod
    def _validate(name: str, callback: Callable) -> None:
        if not name or not callable(callback):
            raise ValueError("Расширению нужны непустое имя и функция")

    def register_filter(self, name: str, callback: Callable, description: str = "") -> None:
        self._require("pixels")
        self._validate(name, callback)
        self.filters[name] = (callback, description)

    def register_action_command(self, name: str, callback: Callable, description: str = "") -> None:
        self._require("document")
        self._validate(name, callback)
        self.actions[name] = (callback, description)

    def register_importer(self, name: str, extensions: list[str] | tuple[str, ...], callback: Callable, description: str = "") -> None:
        self._require("document")
        self._require("filesystem.read")
        self._validate(name, callback)
        self.importers[name] = (callback, description, _normalize_extensions(extensions))

    def register_exporter(self, name: str, extensions: list[str] | tuple[str, ...], callback: Callable, description: str = "") -> None:
        self._require("document")
        self._require("filesystem.write")
        self._validate(name, callback)
        self.exporters[name] = (callback, description, _normalize_extensions(extensions))

    def register_external_filter(self, name: str, executable: str | Path, description: str = "", timeout: int = 120) -> None:
        self._require("pixels")
        self._require("process")
        command = str(Path(executable).resolve())

        def callback(pixels: np.ndarray, params: dict[str, Any]) -> np.ndarray:
            from PIL import Image
            from .core import pil_to_rgba_array, rgba_array_to_pil

            with tempfile.TemporaryDirectory(prefix="photoredactor-external-") as temp:
                source, target = Path(temp) / "input.png", Path(temp) / "output.png"
                rgba_array_to_pil(pixels).save(source)
                completed = subprocess.run(
                    [command, str(source), str(target), json.dumps(params, ensure_ascii=False)],
                    timeout=max(1, int(timeout)),
                    capture_output=True,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                )
                if completed.returncode != 0 or not target.exists():
                    raise RuntimeError(completed.stderr.strip() or f"Внешний фильтр завершился с кодом {completed.returncode}")
                return pil_to_rgba_array(Image.open(target))

        self.filters[name] = (callback, description)

    def metadata(self) -> dict[str, Any]:
        return {
            "filters": [{"name": name, "description": value[1]} for name, value in self.filters.items()],
            "actions": [{"name": name, "description": value[1]} for name, value in self.actions.items()],
            "importers": [{"name": name, "description": value[1], "extensions": list(value[2])} for name, value in self.importers.items()],
            "exporters": [{"name": name, "description": value[1], "extensions": list(value[2])} for name, value in self.exporters.items()],
        }


class PluginRegistry:
    def __init__(self, directories: list[str | Path] | None = None, permission_file: str | Path | None = None) -> None:
        default = Path(os.environ.get("APPDATA", Path.home())) / "PhotoRedactor" / "plugins"
        self.directories = [Path(item) for item in (directories or [default, Path.cwd() / "plugins"])]
        self.permission_file = Path(permission_file) if permission_file else default.parent / "plugin_permissions.json"
        self.plugins: dict[str, PluginInfo] = {}
        self.filters: dict[str, PluginExtension] = {}
        self.action_commands: dict[str, Callable] = {}
        self.importers: dict[str, PluginExtension] = {}
        self.exporters: dict[str, PluginExtension] = {}
        self.errors: list[str] = []
        self._grants = self._load_grants()

    def discover(self) -> int:
        self.plugins.clear()
        self.filters.clear()
        self.action_commands.clear()
        self.importers.clear()
        self.exporters.clear()
        self.errors.clear()
        for info in self._candidates():
            self.plugins[info.id] = info
            if info.blocked_permissions:
                continue
            try:
                metadata = self._host(info, "inspect")
                self._register_metadata(info, metadata)
            except Exception as exc:
                self.errors.append(f"{info.name}: {exc}")
        return len(self.filters) + len(self.action_commands) + len(self.importers) + len(self.exporters)

    def _candidates(self) -> list[PluginInfo]:
        result: list[PluginInfo] = []
        for directory in self.directories:
            directory.mkdir(parents=True, exist_ok=True)
            for manifest_path in sorted(directory.glob("*/plugin.json")):
                try:
                    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if raw.get("format") != MANIFEST_FORMAT:
                        raise ValueError("неподдерживаемый формат манифеста")
                    plugin_id = str(raw["id"])
                    requested = {str(item) for item in raw.get("permissions", [])}
                    unknown = requested - KNOWN_PERMISSIONS
                    if unknown:
                        raise ValueError(f"неизвестные разрешения: {', '.join(sorted(unknown))}")
                    api_version = int(raw.get("api_version", 0))
                    if api_version != PLUGIN_API_VERSION:
                        raise ValueError(f"нужен API {api_version}, редактор предоставляет {PLUGIN_API_VERSION}")
                    entrypoint = (manifest_path.parent / str(raw.get("entrypoint", "plugin.py"))).resolve()
                    if not entrypoint.is_file() or entrypoint.parent != manifest_path.parent.resolve():
                        raise ValueError("entrypoint должен быть файлом внутри папки плагина")
                    grants = set(self._grants.get(plugin_id, [])) & requested
                    result.append(PluginInfo(plugin_id, str(raw.get("name", plugin_id)), str(raw.get("version", "0.0.0")), api_version, entrypoint, requested, grants, str(raw.get("description", ""))))
                except Exception as exc:
                    self.errors.append(f"{manifest_path}: {exc}")
            for path in sorted(directory.glob("*.py")):
                plugin_id = f"legacy.{path.stem}"
                result.append(PluginInfo(plugin_id, path.stem, "legacy", 1, path.resolve(), {"pixels"}, {"pixels"}, legacy=True))
        return result

    def _register_metadata(self, info: PluginInfo, metadata: dict[str, Any]) -> None:
        for raw in metadata.get("filters", []):
            extension = PluginExtension(str(raw["name"]), info.id, str(raw.get("description", "")))
            self.filters[extension.name] = extension
        for raw in metadata.get("actions", []):
            name = str(raw["name"])
            self.action_commands[name] = lambda document, params, value=name, plugin=info: self._run_document_action(plugin, value, document, params)
        for kind, target in (("importers", self.importers), ("exporters", self.exporters)):
            for raw in metadata.get(kind, []):
                extension = PluginExtension(str(raw["name"]), info.id, str(raw.get("description", "")), tuple(raw.get("extensions", [])))
                target[extension.name] = extension

    def set_permissions(self, plugin_id: str, permissions: set[str]) -> None:
        info = self.plugins[plugin_id]
        invalid = set(permissions) - info.requested_permissions
        if invalid:
            raise ValueError("Нельзя выдать не запрошенные плагином разрешения")
        self._grants[plugin_id] = sorted(permissions)
        self.permission_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.permission_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._grants, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.permission_file)

    def apply_filter(self, name: str, pixels: np.ndarray, params: dict[str, Any] | None = None) -> np.ndarray:
        extension = self.filters.get(name)
        if extension is None:
            raise KeyError(f"Фильтр-плагин не найден: {name}")
        info = self.plugins[extension.plugin_id]
        with tempfile.TemporaryDirectory(prefix="photoredactor-plugin-") as temp:
            source, target, parameters = Path(temp) / "input.npy", Path(temp) / "output.npy", Path(temp) / "params.json"
            np.save(source, pixels, allow_pickle=False)
            parameters.write_text(json.dumps(params or {}, ensure_ascii=False), encoding="utf-8")
            self._host(info, "filter", name, source, target, parameters, allowed_paths=[source, target, parameters])
            output = np.load(target, allow_pickle=False)
        if output.shape != pixels.shape or output.dtype != np.uint8:
            raise ValueError("Фильтр должен вернуть uint8 RGBA исходного размера")
        return np.ascontiguousarray(output)

    def _run_document_action(self, info: PluginInfo, name: str, document: Document, params: dict[str, Any]) -> None:
        with tempfile.TemporaryDirectory(prefix="photoredactor-plugin-") as temp:
            source, target, parameters = Path(temp) / "input.prdx", Path(temp) / "output.prdx", Path(temp) / "params.json"
            document.save_project(source)
            parameters.write_text(json.dumps(params, ensure_ascii=False), encoding="utf-8")
            self._host(info, "action", name, source, target, parameters, allowed_paths=[source, target, parameters])
            result = Document.open_project(target)
            document.restore_raw_state(result.raw_state())
            document.dirty = True

    def import_document(self, name: str, source: str | Path, params: dict[str, Any] | None = None) -> Document:
        extension = self.importers.get(name)
        if extension is None:
            raise KeyError(f"Импортёр не найден: {name}")
        info = self.plugins[extension.plugin_id]
        source_path = Path(source).resolve()
        with tempfile.TemporaryDirectory(prefix="photoredactor-plugin-") as temp:
            target, parameters = Path(temp) / "output.prdx", Path(temp) / "params.json"
            parameters.write_text(json.dumps(params or {}, ensure_ascii=False), encoding="utf-8")
            self._host(info, "import", name, source_path, target, parameters, allowed_paths=[source_path, target, parameters])
            return Document.open_project(target)

    def export_document(self, name: str, document: Document, target: str | Path, params: dict[str, Any] | None = None) -> Path:
        extension = self.exporters.get(name)
        if extension is None:
            raise KeyError(f"Экспортёр не найден: {name}")
        info = self.plugins[extension.plugin_id]
        target_path = Path(target).resolve()
        with tempfile.TemporaryDirectory(prefix="photoredactor-plugin-") as temp:
            source, parameters = Path(temp) / "input.prdx", Path(temp) / "params.json"
            document.save_project(source)
            parameters.write_text(json.dumps(params or {}, ensure_ascii=False), encoding="utf-8")
            self._host(info, "export", name, source, target_path, parameters, allowed_paths=[source, target_path, parameters])
        if not target_path.exists():
            raise RuntimeError("Экспортёр не создал выходной файл")
        return target_path

    def _host(self, info: PluginInfo, operation: str, *args: Any, allowed_paths: list[Path] | None = None) -> dict[str, Any]:
        command = [sys.executable]
        command += ["--plugin-host"] if getattr(sys, "frozen", False) else ["-m", "photoredactor.plugin_host"]
        request = {
            "entrypoint": str(info.entrypoint),
            "permissions": sorted(info.granted_permissions),
            "operation": operation,
            "args": [str(item) for item in args],
            "allowed_paths": [str(item.resolve()) for item in (allowed_paths or [])],
        }
        completed = subprocess.run(
            command,
            input=json.dumps(request, ensure_ascii=False),
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=180,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "Изолированный процесс плагина завершился с ошибкой")
        lines = [line for line in completed.stdout.splitlines() if line.startswith("PHOTO_REDACTOR_PLUGIN_RESULT=")]
        if not lines:
            raise RuntimeError("Плагин не вернул корректный ответ")
        return json.loads(lines[-1].split("=", 1)[1])

    def _load_grants(self) -> dict[str, list[str]]:
        try:
            raw = json.loads(self.permission_file.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except (OSError, ValueError):
            return {}


def _normalize_extensions(extensions: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted({item.lower() if str(item).startswith(".") else f".{str(item).lower()}" for item in extensions}))


__all__ = ["MANIFEST_FORMAT", "PLUGIN_API_VERSION", "PluginAPI", "PluginInfo", "PluginRegistry"]
