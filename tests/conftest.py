"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def data_dir():
    """Get the data directory path."""
    return Path("data")


@pytest.fixture
def sample_text():
    """Get sample text for testing."""
    return """
    This is a sample document for testing RAG implementations.
    It contains multiple sentences to test chunking and retrieval.
    The document covers various topics that can be used for query testing.
    """


@pytest.fixture
def sample_documents():
    """Get sample LangChain documents for testing."""
    from langchain_core.documents import Document

    return [
        Document(
            page_content="This is document 1 about Python programming.",
            metadata={"source": "doc1.txt", "page": 1}
        ),
        Document(
            page_content="This is document 2 about machine learning.",
            metadata={"source": "doc2.txt", "page": 1}
        ),
    ]
