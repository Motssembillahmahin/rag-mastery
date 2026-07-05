"""Integration tests for the pipeline."""

import pytest

from corrective_rag.models.schemas import (
    AnswerQuality,
    GradedDocument,
    GradingResult,
    RetrievalResult,
)


def test_graded_document_creation():
    """Test GradedDocument model creation."""
    doc = GradedDocument(
        content="Test content",
        metadata={"source": "test"},
        score=0.85,
        relevant=True,
    )

    assert doc.content == "Test content"
    assert doc.score == 0.85
    assert doc.relevant is True


def test_grading_result_creation():
    """Test GradingResult model creation."""
    result = GradingResult(
        relevant=True,
        score=0.9,
        reason="Highly relevant",
    )

    assert result.relevant is True
    assert result.score == 0.9


def test_answer_quality_creation():
    """Test AnswerQuality model creation."""
    quality = AnswerQuality(
        satisfactory=True,
        reason="Answer is comprehensive",
        suggestions=["Add more details"],
    )

    assert quality.satisfactory is True
    assert len(quality.suggestions) == 1


def test_retrieval_result_creation():
    """Test RetrievalResult model creation."""
    docs = [
        GradedDocument(content="Doc 1", score=0.9, relevant=True),
        GradedDocument(content="Doc 2", score=0.7, relevant=True),
    ]

    result = RetrievalResult(
        query="test query",
        documents=docs,
        attempts=2,
        filtered_count=2,
    )

    assert result.query == "test query"
    assert len(result.documents) == 2
    assert result.attempts == 2
