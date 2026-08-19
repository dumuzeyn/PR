from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np

from .core import Document
from .security.files import load_pillow_image, validate_array
from .security.limits import LIMITS
from .security.paths import canonical_path, ensure_within, validate_local_input
from .security.processes import run_checked, validate_executable
from .security.temporary import secure_temporary_directory
from .security.validation import load_bounded_json, loads_bounded_json, validate_identifier, validate_json_tree


PLUGIN_API_VERSION = 2
MANIFEST_FORMAT = "UZYRO plugin v1"
LEGACY_MANIFEST_FORMAT = f"{'Photo' + 'Redactor'} plugin v1"
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
    author: str = ""
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
        command = validate_executable(executable)

        def callback(pixels: np.ndarray, params: dict[str, Any]) -> np.ndarray:
            from PIL import Image
            from .core import pil_to_rgba_array, rgba_array_to_pil

            validate_json_tree(params)
            with secure_temporary_directory("external-") as temp:
                source, target = temp / "input.png", temp / "output.png"
                rgba_array_to_pil(pixels).save(source)
                completed = run_checked(
                    [command, source, target, json.dumps(params, ensure_ascii=False)],
                    timeout=max(1, min(600, int(timeout))), maximum_output=LIMITS.max_process_output_bytes,
                )
                if completed.returncode != 0 or not target.exists():
                    raise RuntimeError(completed.stderr.strip() or f"Внешний фильтр завершился с кодом {completed.returncode}")
                output = pil_to_rgba_array(load_pillow_image(target))
                if output.shape != pixels.shape:
                    raise ValueError("Внешний фильтр вернул изображение другого размера")
                return output

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
        default = Path(os.environ.get("APPDATA", Path.home())) / "UZYRO" / "plugins"
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
                    root = canonical_path(directory, must_exist=True)
                    manifest_path = ensure_within(manifest_path, root, must_exist=True)
                    if manifest_path.is_symlink() or manifest_path.parent.is_symlink():
                        raise ValueError("ссылки и junction в каталоге плагина запрещены")
                    raw = load_bounded_json(manifest_path, maximum=1024 * 1024)
                    if not isinstance(raw, dict):
                        raise ValueError("манифест должен быть JSON-объектом")
                    if raw.get("format") not in {MANIFEST_FORMAT, LEGACY_MANIFEST_FORMAT}:
                        raise ValueError("неподдерживаемый формат манифеста")
                    plugin_id = validate_identifier(raw["id"], "ID плагина")
                    requested = {str(item) for item in raw.get("permissions", [])}
                    unknown = requested - KNOWN_PERMISSIONS
                    if unknown:
                        raise ValueError(f"неизвестные разрешения: {', '.join(sorted(unknown))}")
                    api_version = int(raw.get("api_version", 0))
                    if api_version != PLUGIN_API_VERSION:
                        raise ValueError(f"нужен API {api_version}, редактор предоставляет {PLUGIN_API_VERSION}")
                    entrypoint = ensure_within(manifest_path.parent / str(raw.get("entrypoint", "plugin.py")), manifest_path.parent, must_exist=True)
                    if not entrypoint.is_file() or entrypoint.is_symlink() or entrypoint.suffix.lower() != ".py":
                        raise ValueError("entrypoint должен быть файлом внутри папки плагина")
                    grants = set(self._grants.get(plugin_id, [])) & requested
                    result.append(PluginInfo(
                        plugin_id, str(raw.get("name", plugin_id))[:128], str(raw.get("version", "0.0.0"))[:64],
                        api_version, entrypoint, requested, grants, str(raw.get("description", ""))[:4096],
                        str(raw.get("author", "Не указан"))[:128],
                    ))
                except Exception as exc:
                    self.errors.append(f"{manifest_path}: {exc}")
            for path in sorted(directory.glob("*.py")):
                plugin_id = f"legacy.{path.stem}"
                grants = set(self._grants.get(plugin_id, [])) & {"pixels"}
                result.append(PluginInfo(plugin_id, path.stem, "legacy", 1, path.resolve(), {"pixels"}, grants, legacy=True))
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
        validate_json_tree(params or {})
        validate_array(pixels, expected_channels={4})
        with secure_temporary_directory("plugin-") as temp:
            source, target, parameters = temp / "input.npy", temp / "output.npy", temp / "params.json"
            np.save(source, pixels, allow_pickle=False)
            parameters.write_text(json.dumps(params or {}, ensure_ascii=False), encoding="utf-8")
            self._host(info, "filter", name, source, target, parameters, allowed_paths=[source, target, parameters])
            output = validate_array(np.load(target, allow_pickle=False), expected_channels={4})
        if output.shape != pixels.shape or output.dtype != np.uint8:
            raise ValueError("Фильтр должен вернуть uint8 RGBA исходного размера")
        return np.ascontiguousarray(output)

    def _run_document_action(self, info: PluginInfo, name: str, document: Document, params: dict[str, Any]) -> None:
        validate_json_tree(params)
        with secure_temporary_directory("plugin-") as temp:
            source, target, parameters = temp / "input.prdx", temp / "output.prdx", temp / "params.json"
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
        source_path = validate_local_input(source)
        validate_json_tree(params or {})
        with secure_temporary_directory("plugin-") as temp:
            target, parameters = temp / "output.prdx", temp / "params.json"
            parameters.write_text(json.dumps(params or {}, ensure_ascii=False), encoding="utf-8")
            self._host(info, "import", name, source_path, target, parameters, allowed_paths=[source_path, target, parameters])
            return Document.open_project(target)

    def export_document(self, name: str, document: Document, target: str | Path, params: dict[str, Any] | None = None) -> Path:
        extension = self.exporters.get(name)
        if extension is None:
            raise KeyError(f"Экспортёр не найден: {name}")
        info = self.plugins[extension.plugin_id]
        target_path = canonical_path(target)
        validate_json_tree(params or {})
        with secure_temporary_directory("plugin-") as temp:
            source, parameters = temp / "input.prdx", temp / "params.json"
            document.save_project(source)
            parameters.write_text(json.dumps(params or {}, ensure_ascii=False), encoding="utf-8")
            self._host(info, "export", name, source, target_path, parameters, allowed_paths=[source, target_path, parameters])
        if not target_path.exists():
            raise RuntimeError("Экспортёр не создал выходной файл")
        return target_path

    def _host(self, info: PluginInfo, operation: str, *args: Any, allowed_paths: list[Path] | None = None) -> dict[str, Any]:
        command = [sys.executable]
        command += ["--plugin-host"] if getattr(sys, "frozen", False) else ["-m", "uzyro.plugin_host"]
        request = {
            "entrypoint": str(info.entrypoint),
            "permissions": sorted(info.granted_permissions),
            "operation": operation,
            "args": [str(item) for item in args],
            "allowed_paths": [str(item.resolve()) for item in (allowed_paths or [])],
        }
        request_payload = json.dumps(request, ensure_ascii=False)
        if len(request_payload.encode("utf-8")) > LIMITS.max_plugin_message_bytes:
            raise ValueError("Запрос к плагину превышает безопасный размер")
        host_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        if not getattr(sys, "frozen", False):
            package_root = str(Path(__file__).resolve().parent.parent)
            current_pythonpath = host_env.get("PYTHONPATH", "")
            host_env["PYTHONPATH"] = package_root if not current_pythonpath else os.pathsep.join((package_root, current_pythonpath))
        completed = run_checked(
            command, input_data=request_payload, timeout=180, maximum_output=LIMITS.max_plugin_message_bytes,
            env=host_env,
            cwd=info.entrypoint.parent, allow_python=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "Изолированный процесс плагина завершился с ошибкой")
        lines = [line for line in completed.stdout.splitlines() if line.startswith("UZYRO_PLUGIN_RESULT=")]
        if not lines:
            raise RuntimeError("Плагин не вернул корректный ответ")
        result = loads_bounded_json(lines[-1].split("=", 1)[1], maximum=LIMITS.max_plugin_message_bytes)
        if not isinstance(result, dict):
            raise RuntimeError("Плагин вернул ответ неверного типа")
        return result

    def _load_grants(self) -> dict[str, list[str]]:
        try:
            raw = load_bounded_json(self.permission_file, maximum=1024 * 1024)
            if not isinstance(raw, dict):
                return {}
            return {
                validate_identifier(key, "ID плагина"): [item for item in value if item in KNOWN_PERMISSIONS]
                for key, value in raw.items() if isinstance(value, list)
            }
        except (OSError, ValueError):
            return {}


def _normalize_extensions(extensions: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted({item.lower() if str(item).startswith(".") else f".{str(item).lower()}" for item in extensions}))


__all__ = ["MANIFEST_FORMAT", "PLUGIN_API_VERSION", "PluginAPI", "PluginInfo", "PluginRegistry"]
