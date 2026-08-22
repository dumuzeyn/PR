from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import zipfile

import pytest

from uzyro.automation import ActionRunner, load_action
from uzyro.batch_queue import BatchQueue
from uzyro.color_management import validate_icc_bytes
from uzyro.document import Document
from uzyro.plugins import PluginRegistry
from uzyro.psd_compat import PSDCompatibilityError, load_psd
from uzyro.security.errors import ResourceLimitError, SecurityValidationError
from uzyro.security.files import safe_extract_zip, validate_zip_archive
from uzyro.security.network import validate_loopback_url
from uzyro.security.processes import run_checked
from uzyro.security.validation import loads_bounded_json


def test_zip_traversal_and_absolute_paths_are_rejected(tmp_path: Path) -> None:
    for name in ("../escape.txt", "/absolute.txt", "C:/drive.txt"):
        archive_path = tmp_path / f"{abs(hash(name))}.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr(name, b"blocked")
        with zipfile.ZipFile(archive_path) as archive:
            with pytest.raises(SecurityValidationError):
                safe_extract_zip(archive, tmp_path / "output")


def test_zip_symlink_member_is_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = 0o120777 << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(info, "target")
    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(SecurityValidationError):
            validate_zip_archive(archive)


def test_prdx_huge_declared_dimensions_are_rejected_before_allocation(tmp_path: Path) -> None:
    project = tmp_path / "huge.prdx"
    manifest = {
        "format": "UZYRO project", "format_version": 3,
        "width": 100_000, "height": 100_000, "dpi": 300, "bit_depth": 8,
        "layers": [{"name": "Layer", "kind": "raster", "pixels": "layer.png"}],
    }
    with zipfile.ZipFile(project, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("layer.png", b"not decoded")
    with pytest.raises(ResourceLimitError):
        Document.open_project(project)


def test_json_nan_infinity_and_excessive_actions_are_rejected() -> None:
    with pytest.raises(SecurityValidationError):
        loads_bounded_json('{"value": NaN}')
    with pytest.raises(SecurityValidationError):
        loads_bounded_json('{"value": Infinity}')
    with pytest.raises(ResourceLimitError):
        load_action({"format": "UZYRO action v3", "steps": [{"command": "invert"}] * 10_001})


def test_plugin_entrypoint_escape_and_default_denial(tmp_path: Path) -> None:
    root = tmp_path / "plugins"
    escaped = tmp_path / "escape.py"
    escaped.write_text("def register(api): pass", encoding="utf-8")
    malicious = root / "malicious"
    malicious.mkdir(parents=True)
    (malicious / "plugin.json").write_text(json.dumps({
        "format": "UZYRO plugin v1", "id": "malicious.plugin", "name": "Malicious",
        "version": "1", "api_version": 2, "entrypoint": "../../escape.py", "permissions": ["pixels"],
    }), encoding="utf-8")
    blocked = root / "blocked"
    blocked.mkdir()
    (blocked / "plugin.py").write_text("def register(api): pass", encoding="utf-8")
    (blocked / "plugin.json").write_text(json.dumps({
        "format": "UZYRO plugin v1", "id": "blocked.plugin", "name": "Blocked",
        "version": "1", "api_version": 2, "entrypoint": "plugin.py", "permissions": ["network", "process"],
    }), encoding="utf-8")

    registry = PluginRegistry([root], tmp_path / "permissions.json")
    registry.discover()

    assert "malicious.plugin" not in registry.plugins
    assert "blocked.plugin" in registry.plugins
    assert registry.plugins["blocked.plugin"].granted_permissions == set()
    assert registry.plugins["blocked.plugin"].blocked_permissions == {"network", "process"}


def test_external_process_timeout_is_enforced() -> None:
    with pytest.raises(TimeoutError):
        run_checked(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            timeout=0.1, allow_python=True,
        )


def test_unknown_action_and_batch_path_escape_are_rejected() -> None:
    action = load_action({"format": "UZYRO action v3", "steps": [{"command": "not_registered"}]})
    report = ActionRunner().run_with_report(Document.new(8, 8), action)
    assert report.executed == 0
    assert report.stopped
    assert report.errors and "Неизвестная команда" in report.errors[0].message
    for suffix in ("../png", "..\\png", ".png.exe", "/tmp"):
        with pytest.raises(ValueError):
            BatchQueue._validate_suffix(suffix)


def test_corrupted_psd_and_icc_are_rejected_at_import_boundary(tmp_path: Path) -> None:
    psd = tmp_path / "corrupt.psd"
    psd.write_bytes(b"8BPS\x00\x01" + b"\x00" * 20)
    with pytest.raises((PSDCompatibilityError, SecurityValidationError)):
        load_psd(psd, Document)
    invalid_icc = bytearray(128)
    invalid_icc[:4] = (128).to_bytes(4, "big")
    with pytest.raises(SecurityValidationError):
        validate_icc_bytes(bytes(invalid_icc))


def test_local_ai_rejects_non_loopback_server() -> None:
    assert validate_loopback_url("http://127.0.0.1:8123/api").startswith("http://127.0.0.1")
    with pytest.raises(SecurityValidationError):
        validate_loopback_url("http://0.0.0.0:8123/api")
    with pytest.raises(SecurityValidationError):
        validate_loopback_url("https://example.com/api")


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symbolic links unavailable")
def test_plugin_symlink_entrypoint_is_rejected_when_supported(tmp_path: Path) -> None:
    root = tmp_path / "plugins"
    plugin = root / "linked"
    plugin.mkdir(parents=True)
    real = tmp_path / "real.py"
    real.write_text("def register(api): pass", encoding="utf-8")
    link = plugin / "plugin.py"
    try:
        link.symlink_to(real)
    except OSError:
        pytest.skip("symbolic link creation is not permitted")
    (plugin / "plugin.json").write_text(json.dumps({
        "format": "UZYRO plugin v1", "id": "linked.plugin", "name": "Linked",
        "version": "1", "api_version": 2, "entrypoint": "plugin.py", "permissions": ["pixels"],
    }), encoding="utf-8")
    registry = PluginRegistry([root], tmp_path / "permissions.json")
    registry.discover()
    assert "linked.plugin" not in registry.plugins
