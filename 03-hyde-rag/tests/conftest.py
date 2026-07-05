"""Pytest configuration and fixtures."""

import pytest
from hyde_rag.config import Settings


@pytest.fixture
def settings():
    """Provide test settings."""
    return Settings(
        ollama_base_url="http://localhost:11434",
        llm_model="llama3",
        embedding_model="all-minilm:latest",
        collection_name="test_hyde_rag",
        persist_directory="./test_chroma_db",
        chunk_size=500,
        chunk_overlap=100,
        hyde_temperature=0.7,
        num_hypothetical_docs=2,
        use_multi_hyde=True,
        top_k=3,
        score_threshold=0.5,
        data_dir="data",
    )
