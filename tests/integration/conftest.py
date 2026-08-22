from __future__ import annotations

from pathlib import Path

import pytest


INTEGRATION_ROOT = Path(__file__).parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if Path(item.path).is_relative_to(INTEGRATION_ROOT):
            item.add_marker(pytest.mark.integration)
