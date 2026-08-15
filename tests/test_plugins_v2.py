from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from uzyro.core import Document
from uzyro.plugins import MANIFEST_FORMAT, PLUGIN_API_VERSION, PluginRegistry


def write_plugin(root: Path, plugin_id: str, permissions: list[str], source: str) -> Path:
    folder = root / plugin_id
    folder.mkdir()
    (folder / "plugin.py").write_text(source, encoding="utf-8")
    (folder / "plugin.json").write_text(
        json.dumps(
            {
                "format": MANIFEST_FORMAT,
                "id": plugin_id,
                "name": plugin_id,
                "version": "1.2.3",
                "api_version": PLUGIN_API_VERSION,
                "entrypoint": "plugin.py",
                "permissions": permissions,
            }
        ),
        encoding="utf-8",
    )
    return folder


def test_manifest_plugin_requires_explicit_permissions_and_runs_isolated(tmp_path: Path) -> None:
    write_plugin(
        tmp_path,
        "example.invert",
        ["pixels"],
        "def register(api):\n"
        "    def invert(pixels, params):\n"
        "        result = pixels.copy(); result[:, :, :3] = 255 - result[:, :, :3]; return result\n"
        "    api.register_filter('Изолированная инверсия', invert, 'Инвертирует RGB')\n",
    )
    registry = PluginRegistry([tmp_path], tmp_path / "permissions.json")
    assert registry.discover() == 0
    assert registry.plugins["example.invert"].blocked_permissions == {"pixels"}
    registry.set_permissions("example.invert", {"pixels"})
    assert registry.discover() == 1

    pixels = np.full((6, 8, 4), (20, 60, 100, 255), dtype=np.uint8)
    result = registry.apply_filter("Изолированная инверсия", pixels)
    np.testing.assert_array_equal(result[:, :, :3], 255 - pixels[:, :, :3])
    np.testing.assert_array_equal(result[:, :, 3], pixels[:, :, 3])


def test_permission_guard_blocks_unapproved_file_access(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("private", encoding="utf-8")
    write_plugin(
        tmp_path,
        "example.reader",
        ["pixels"],
        "from pathlib import Path\n"
        "def register(api):\n"
        "    def read_file(pixels, params):\n"
        "        Path(params['path']).read_text(); return pixels\n"
        "    api.register_filter('Чтение файла', read_file)\n",
    )
    registry = PluginRegistry([tmp_path], tmp_path / "permissions.json")
    registry.discover()
    registry.set_permissions("example.reader", {"pixels"})
    registry.discover()
    with pytest.raises(RuntimeError, match="Чтение запрещено"):
        registry.apply_filter("Чтение файла", np.zeros((2, 2, 4), dtype=np.uint8), {"path": str(secret)})


def test_document_actions_run_outside_editor_process(tmp_path: Path) -> None:
    write_plugin(
        tmp_path,
        "example.document",
        ["document"],
        "def register(api):\n"
        "    def opacity(document, params):\n"
        "        document.layer.opacity = float(params['value']); document.dirty = True\n"
        "    api.register_action_command('plugin.opacity', opacity, 'Меняет непрозрачность')\n",
    )
    registry = PluginRegistry([tmp_path], tmp_path / "permissions.json")
    registry.discover()
    registry.set_permissions("example.document", {"document"})
    assert registry.discover() == 1
    document = Document.new(8, 6)
    registry.action_commands["plugin.opacity"](document, {"value": 0.35})
    assert document.layer.opacity == 0.35


def test_plugin_crash_is_contained_by_host_process(tmp_path: Path) -> None:
    write_plugin(
        tmp_path,
        "example.crash",
        ["pixels"],
        "import os\n"
        "def register(api):\n"
        "    def crash(pixels, params): os._exit(9)\n"
        "    api.register_filter('Авария', crash)\n",
    )
    registry = PluginRegistry([tmp_path], tmp_path / "permissions.json")
    registry.discover()
    registry.set_permissions("example.crash", {"pixels"})
    registry.discover()
    with pytest.raises(RuntimeError):
        registry.apply_filter("Авария", np.zeros((2, 2, 4), dtype=np.uint8))
    assert "example.crash" in registry.plugins


def test_importer_and_exporter_extension_points_exchange_documents(tmp_path: Path) -> None:
    write_plugin(
        tmp_path,
        "example.formats",
        ["document", "filesystem.read", "filesystem.write"],
        "from pathlib import Path\n"
        "from uzyro.core import Document\n"
        "def register(api):\n"
        "    def load(source, params):\n"
        "        size = int(Path(source).read_text()); return Document.new(size, size, (10, 20, 30, 255))\n"
        "    def save(document, target, params):\n"
        "        Path(target).write_text(f'{document.width}x{document.height}')\n"
        "    api.register_importer('Квадрат', ['.square'], load)\n"
        "    api.register_exporter('Размер', ['.size'], save)\n",
    )
    source = tmp_path / "sample.square"
    source.write_text("7", encoding="utf-8")
    registry = PluginRegistry([tmp_path], tmp_path / "permissions.json")
    registry.discover()
    permissions = {"document", "filesystem.read", "filesystem.write"}
    registry.set_permissions("example.formats", permissions)
    assert registry.discover() == 2
    document = registry.import_document("Квадрат", source)
    assert (document.width, document.height) == (7, 7)
    target = tmp_path / "result.size"
    registry.export_document("Размер", document, target)
    assert target.read_text(encoding="utf-8") == "7x7"


def test_legacy_manifest_format_remains_discoverable(tmp_path: Path) -> None:
    folder = write_plugin(tmp_path, "example.legacy-brand", [], "def register(api):\n    pass\n")
    manifest_path = folder / "plugin.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format"] = f"{'Photo' + 'Redactor'} plugin v1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    registry = PluginRegistry([tmp_path], tmp_path / "permissions.json")
    registry.discover()
    assert "example.legacy-brand" in registry.plugins
    assert not registry.errors
