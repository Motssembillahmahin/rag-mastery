"""Tests for Adaptive RAG pipeline schemas."""

from adaptive_rag.config import QueryType
from adaptive_rag.models.schemas import (
    ClassificationResult,
    DocumentChunk,
    QueryResult,
    RetrievalResult,
)


def test_classification_result_schema():
    """Test ClassificationResult schema."""
    result = ClassificationResult(
        query_type=QueryType.FACTUAL,
        confidence=0.9,
    )

    assert result.query_type == QueryType.FACTUAL
    assert result.confidence == 0.9


def test_document_chunk_schema():
    """Test DocumentChunk schema."""
    chunk = DocumentChunk(
        content="Test content",
        source="test.txt",
        chunk_index=0,
        metadata={"page": 1},
    )

    assert chunk.content == "Test content"
    assert chunk.source == "test.txt"
    assert chunk.chunk_index == 0
    assert chunk.metadata == {"page": 1}


def test_retrieval_result_schema():
    """Test RetrievalResult schema."""
    result = RetrievalResult(
        chunks=[
            DocumentChunk(content="Chunk 1", source="test.txt", chunk_index=0),
            DocumentChunk(content="Chunk 2", source="test.txt", chunk_index=1),
        ],
        scores=[0.8, 0.7],
        used_hyde=True,
    )

    assert len(result.chunks) == 2
    assert len(result.scores) == 2
    assert result.used_hyde is True


def test_query_result_schema():
    """Test QueryResult schema."""
    result = QueryResult(
        question="What is Python?",
        answer="Python is a programming language.",
        query_type=QueryType.SIMPLE,
        confidence=0.95,
        context=[DocumentChunk(content="Python is a language", source="test.txt", chunk_index=0)],
        used_hyde=False,
    )

    assert result.question == "What is Python?"
    assert result.answer == "Python is a programming language."
    assert result.query_type == QueryType.SIMPLE
    assert result.confidence == 0.95
    assert len(result.context) == 1
    assert result.used_hyde is False
