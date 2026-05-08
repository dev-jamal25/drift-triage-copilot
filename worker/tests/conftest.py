"""Pytest configuration for worker tests.

E2E tests requiring Docker are skipped by default.
To run them, set RUN_DOCKER_TESTS=1 or use: pytest -m requires_docker
"""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip requires_docker tests unless RUN_DOCKER_TESTS=1."""
    if os.environ.get("RUN_DOCKER_TESTS") != "1":
        skip_marker = pytest.mark.skip(reason="Requires Docker; set RUN_DOCKER_TESTS=1 to run")
        for item in items:
            if "requires_docker" in item.keywords:
                item.add_marker(skip_marker)
