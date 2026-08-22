from __future__ import annotations

from pathlib import Path

import pytest


UI_ROOT = Path(__file__).parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """The directory is the explicit boundary for tests that create Tk UI."""
    for item in items:
        if Path(item.path).is_relative_to(UI_ROOT):
            item.add_marker(pytest.mark.ui)
