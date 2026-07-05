"""Test configuration and fixtures."""

import pytest

from hybrid_rag.config import HybridRAGConfig


@pytest.fixture
def config():
    """Create a test configuration."""
    return HybridRAGConfig(
        data_dir="test_data",
        persist_dir="./test_chroma_db",
        chunk_size=500,
        chunk_overlap=100,
        top_k=3,
        initial_retrieval_k=5,
        semantic_weight=0.6,
        keyword_weight=0.4,
        use_rrf=True,
        rrf_k=60,
        bm25_k1=1.5,
        bm25_b=0.75,
    )


@pytest.fixture
def sample_text_content():
    """Sample text content for testing."""
    return """
    This is a sample document for testing purposes.
    It contains multiple sentences to test text splitting.
    The hybrid RAG system should handle this correctly.
    """


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return {
        "source": "test.pdf",
        "type": "text",
        "page": 1,
    }
