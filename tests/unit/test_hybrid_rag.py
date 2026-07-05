"""Unit tests for Hybrid RAG."""

import pytest
import numpy as np


class TestHybridConfig:
    """Test cases for Hybrid RAG configuration."""

    def test_default_config(self):
        """Test default configuration."""
        from rag_mastery.config import HybridRAGConfig

        config = HybridRAGConfig()
        assert config.semantic_weight == 0.6
        assert config.keyword_weight == 0.4
        assert config.use_rrf is True
        assert config.rrf_k == 60

    def test_weights_sum_to_one(self):
        """Test that semantic and keyword weights sum to 1."""
        from rag_mastery.config import HybridRAGConfig

        config = HybridRAGConfig()
        total = config.semantic_weight + config.keyword_weight
        assert abs(total - 1.0) < 0.001


class TestRRF:
    """Test Reciprocal Rank Fusion."""

    def test_rrf_basic(self):
        """Test basic RRF calculation."""
        k = 60
        rank = 0
        score = 1 / (k + rank + 1)
        assert score == pytest.approx(1/61)
