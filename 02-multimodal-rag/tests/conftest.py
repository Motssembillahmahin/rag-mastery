"""Test configuration and fixtures."""

import pytest
from pathlib import Path

from multimodal_rag.config import MultimodalRAGConfig


@pytest.fixture
def config():
    """Create a test configuration."""
    return MultimodalRAGConfig(
        data_dir="test_data",
        documents_dir="test_data/documents",
        images_dir="test_data/images",
        persist_dir="./test_chroma_db",
        chunk_size=500,
        chunk_overlap=100,
        top_k=3,
    )


@pytest.fixture
def sample_text_content():
    """Sample text content for testing."""
    return """
    This is a sample document for testing purposes.
    It contains multiple sentences to test text splitting.
    The multimodal RAG system should handle this correctly.
    """


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return {
        "source": "test.pdf",
        "type": "text",
        "page": 1,
    }
