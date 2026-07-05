"""Pytest configuration and fixtures."""

import pytest

from adaptive_rag.config import Settings


@pytest.fixture
def settings():
    """Provide test settings."""
    return Settings(
        ollama_base_url="http://localhost:11434",
        llm_model="llama3",
        embedding_model="all-minilm:latest",
        collection_name="test_adaptive_rag",
        persist_dir="./test_chroma_db",
        chunk_size=500,
        chunk_overlap=100,
        use_query_classification=True,
        confidence_threshold=0.7,
        data_dir="data",
    )
