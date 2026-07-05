"""Pytest configuration and fixtures."""

import pytest

from graph_rag.config import Settings


@pytest.fixture
def settings():
    """Provide test settings."""
    return Settings(
        data_dir="/tmp/test_data",
        persist_dir="/tmp/test_chroma",
        collection_name="test_collection",
    )
