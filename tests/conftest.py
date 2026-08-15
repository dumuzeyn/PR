from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_user_application_data(monkeypatch, tmp_path) -> None:
    """Never let application tests read or overwrite a user's real settings."""
    local_app_data = tmp_path / "localappdata"
    local_app_data.mkdir()
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
