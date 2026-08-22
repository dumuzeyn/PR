from __future__ import annotations

import ipaddress
import socket

import pytest


@pytest.fixture(autouse=True)
def isolate_user_application_data(monkeypatch, tmp_path) -> None:
    """Never let application tests read or overwrite a user's real settings."""
    local_app_data = tmp_path / "localappdata"
    local_app_data.mkdir()
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch) -> None:
    """Allow loopback integration tests while rejecting accidental Internet access."""
    original_connect = socket.socket.connect

    def guarded_connect(sock: socket.socket, address) -> None:
        unix_family = getattr(socket, "AF_UNIX", None)
        if unix_family is not None and sock.family == unix_family:
            return original_connect(sock, address)
        host = address[0] if isinstance(address, tuple) else str(address)
        if host == "localhost":
            return original_connect(sock, address)
        try:
            if ipaddress.ip_address(host).is_loopback:
                return original_connect(sock, address)
        except ValueError:
            pass
        raise AssertionError(f"Tests must not access external network host: {host}")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture(autouse=True)
def cleanup_tk_roots(request):
    """Destroy a leaked Tk root after UI tests so collection order cannot matter."""
    yield
    if request.node.get_closest_marker("ui") is None:
        return
    import tkinter as tk

    owner = getattr(request.node, "cls", None)
    for attribute in ("app", "root"):
        shared_root = getattr(owner, attribute, None) if owner is not None else None
        if shared_root is not None:
            try:
                if shared_root.winfo_exists():
                    return
            except tk.TclError:
                pass
    root = getattr(tk, "_default_root", None)
    if root is not None:
        try:
            if root.winfo_exists():
                root.destroy()
        except tk.TclError:
            pass
