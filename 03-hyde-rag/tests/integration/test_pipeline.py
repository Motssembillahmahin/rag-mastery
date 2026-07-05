"""Tests for HyDE pipeline."""

import pytest
from unittest.mock import Mock, patch

from hyde_rag.config import Settings
from hyde_rag.models.schemas import (
    DocumentChunk,
    HyDERetrievalResult,
    HypotheticalDocument,
    QueryResult,
)
from hyde_rag.pipelines.hyde import HyDEPipeline


@pytest.fixture
def mock_settings():
    """Provide mock settings."""
    return Settings(
        collection_name="test_pipeline",
        persist_directory="./test_pipeline_db",
    )


def test_query_result_schema():
    """Test QueryResult schema."""
    result = QueryResult(
        question="What is Python?",
        answer="Python is a programming language.",
        context=[
            DocumentChunk(content="Python is a language", source="test.txt", chunk_index=0)
        ],
        hypothetical_docs=[
            HypotheticalDocument(content="Python is a language", query="What is Python?")
        ],
    )

    assert result.question == "What is Python?"
    assert result.answer == "Python is a programming language."
    assert len(result.context) == 1
    assert len(result.hypothetical_docs) == 1


def test_hypothetical_document_schema():
    """Test HypotheticalDocument schema."""
    doc = HypotheticalDocument(
        content="Test content",
        confidence=0.8,
        query="Test query",
    )

    assert doc.content == "Test content"
    assert doc.confidence == 0.8
    assert doc.query == "Test query"


def test_hyde_retrieval_result_schema():
    """Test HyDERetrievalResult schema."""
    result = HyDERetrievalResult(
        hypothetical_docs=[
            HypotheticalDocument(content="Hyp doc", query="test")
        ],
        real_matches=[
            DocumentChunk(content="Real doc", source="test.txt", chunk_index=0)
        ],
        scores=[0.75],
    )

    assert len(result.hypothetical_docs) == 1
    assert len(result.real_matches) == 1
    assert result.scores == [0.75]
