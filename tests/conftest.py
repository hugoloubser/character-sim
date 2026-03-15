"""Shared pytest fixtures and configuration."""

from __future__ import annotations

import logging

from typing import TYPE_CHECKING

import pytest

from character_creator.utils.path import get_project_root

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="session")
def integration_log_dir() -> Path:
    """Return (and create) the integration-test log directory under local/logs/."""
    d = get_project_root() / "local" / "logs" / "tests"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(autouse=False)
def file_logger(integration_log_dir: Path, request: pytest.FixtureRequest) -> logging.Logger:
    """Set up per-test file logging under local/logs/tests/.

    Use by requesting this fixture in any test that needs file-persisted logs.
    The log file is named after the test node id (e.g. ``test_integration_e2e__TestE2E__test_full_run.log``).
    """
    safe_name = request.node.nodeid.replace("/", "_").replace("::", "__").replace(".py", "")
    log_file = integration_log_dir / f"{safe_name}.log"

    test_logger = logging.getLogger("character_creator")
    test_logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler(log_file, encoding="utf-8", mode="w")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(name)s | %(levelname)-7s | %(message)s")
    )
    test_logger.addHandler(handler)

    yield test_logger

    handler.close()
    test_logger.removeHandler(handler)
