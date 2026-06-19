"""Test configuration and fixtures."""

import tempfile
from pathlib import Path

import pytest

from agent.config import AppConfig
from agent.store import BuildingStore


@pytest.fixture
def temp_db_path():
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def store(temp_db_path):
    store = BuildingStore(temp_db_path)
    yield store
    store.close()


@pytest.fixture
def config():
    return AppConfig()
