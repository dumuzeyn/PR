from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable

import numpy as np
from PIL import Image

from .core import pil_to_rgba_array, rgba_array_to_pil


PLUGIN_API_VERSION = 1


@dataclass
class FilterPlugin:
    name: str
    callback: Callable[[np.ndarray, dict[str, Any]], np.ndarray]
    description: str = ""


class PluginAPI:
    def __init__(self, registry: "PluginRegistry") -> None:
        self.api_version = PLUGIN_API_VERSION
        self._registry = registry

    def register_filter(self, name: str, callback: Callable[[np.ndarray, dict[str, Any]], np.ndarray], description: str = "") -> None:
        self._registry.register_filter(name, callback, description)

    def register_action_command(self, name: str, callback: Callable) -> None:
        self._registry.action_commands[name] = callback

    def register_external_filter(self, name: str, executable: str | Path, description: str = "", timeout: int = 120) -> None:
        self._registry.register_external_filter(name, executable, description, timeout)


class PluginRegistry:
    def __init__(self, directories: list[str | Path] | None = None) -> None:
        default = Path(os.environ.get("APPDATA", Path.home())) / "PhotoRedactor" / "plugins"
        self.directories = [Path(item) for item in (directories or [default, Path.cwd() / "plugins"])]
        self.filters: dict[str, FilterPlugin] = {}
        self.action_commands: dict[str, Callable] = {}
        self.errors: list[str] = []

    def register_filter(self, name: str, callback: Callable[[np.ndarray, dict[str, Any]], np.ndarray], description: str = "") -> None:
        if not name or not callable(callback):
            raise ValueError("A plugin filter needs a name and callable")
        self.filters[name] = FilterPlugin(name, callback, description)

    def discover(self) -> int:
        self.filters.clear()
        self.action_commands.clear()
        self.errors.clear()
        for directory in self.directories:
            directory.mkdir(parents=True, exist_ok=True)
            for path in sorted(directory.glob("*.py")):
                try:
                    module_name = f"photoredactor_user_plugin_{path.stem}_{abs(hash(path))}"
                    spec = importlib.util.spec_from_file_location(module_name, path)
                    if spec is None or spec.loader is None:
                        raise ImportError("Could not load module")
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    register = getattr(module, "register", None)
                    if not callable(register):
                        raise ValueError("Plugin must export register(api)")
                    register(PluginAPI(self))
                except Exception as exc:
                    self.errors.append(f"{path.name}: {exc}")
        return len(self.filters) + len(self.action_commands)

    def apply_filter(self, name: str, pixels: np.ndarray, params: dict[str, Any] | None = None) -> np.ndarray:
        plugin = self.filters.get(name)
        if plugin is None:
            raise KeyError(f"Plugin filter not found: {name}")
        output = np.asarray(plugin.callback(pixels.copy(), dict(params or {})))
        if output.shape != pixels.shape or output.dtype != np.uint8:
            raise ValueError("Plugin filter must return uint8 RGBA pixels with the original shape")
        return np.ascontiguousarray(output)

    def register_external_filter(self, name: str, executable: str | Path, description: str = "", timeout: int = 120) -> None:
        command = str(executable)

        def callback(pixels: np.ndarray, params: dict[str, Any]) -> np.ndarray:
            with tempfile.TemporaryDirectory(prefix="photoredactor-plugin-") as temp:
                source = Path(temp) / "input.png"
                target = Path(temp) / "output.png"
                rgba_array_to_pil(pixels).save(source)
                completed = subprocess.run(
                    [command, str(source), str(target), json.dumps(params, ensure_ascii=False)],
                    check=False,
                    timeout=max(1, int(timeout)),
                    capture_output=True,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                )
                if completed.returncode != 0 or not target.exists():
                    raise RuntimeError(completed.stderr.strip() or f"External filter exited with {completed.returncode}")
                return pil_to_rgba_array(Image.open(target))

        self.register_filter(name, callback, description)
