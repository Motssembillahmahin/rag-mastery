"""Unit tests for Corrective RAG."""

import pytest
from unittest.mock import Mock, patch


class TestCorrectiveConfig:
    """Test cases for Corrective RAG configuration."""

    def test_default_config(self):
        """Test default configuration."""
        from rag_mastery.config import CorrectiveRAGConfig

        config = CorrectiveRAGConfig()
        assert config.relevance_threshold == 0.7
        assert config.max_correction_attempts == 3
        assert config.use_grading is True
        assert config.use_self_rag is True

    def test_custom_config(self):
        """Test custom configuration."""
        from rag_mastery.config import CorrectiveRAGConfig

        config = CorrectiveRAGConfig(
            relevance_threshold=0.5,
            max_correction_attempts=5,
            use_grading=False
        )
        assert config.relevance_threshold == 0.5
        assert config.max_correction_attempts == 5
        assert config.use_grading is False
