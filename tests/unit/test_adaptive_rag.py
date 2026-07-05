"""Unit tests for Adaptive RAG."""

import pytest


class TestQueryType:
    """Test QueryType enum."""

    def test_query_types(self):
        """Test all query types exist."""
        from rag_mastery.config import QueryType

        assert QueryType.SIMPLE.value == "simple"
        assert QueryType.COMPLEX.value == "complex"
        assert QueryType.FACTUAL.value == "factual"
        assert QueryType.ANALYTICAL.value == "analytical"
        assert QueryType.CREATIVE.value == "creative"


class TestAdaptiveConfig:
    """Test Adaptive RAG configuration."""

    def test_default_config(self):
        """Test default configuration."""
        from rag_mastery.config import AdaptiveRAGConfig

        config = AdaptiveRAGConfig()
        assert config.use_query_classification is True
        assert config.confidence_threshold == 0.7
        assert len(config.retrieval_config) == 5
