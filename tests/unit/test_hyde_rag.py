"""Unit tests for HyDE RAG."""

import pytest
from unittest.mock import Mock, patch


class TestHyDEConfig:
    """Test cases for HyDE configuration."""

    def test_default_config(self):
        """Test default HyDE configuration."""
        from rag_mastery.config import HyDEConfig

        config = HyDEConfig()
        assert config.hyde_temperature == 0.7
        assert config.num_hypothetical_docs == 3
        assert config.use_multi_hyde is True

    def test_custom_config(self):
        """Test custom HyDE configuration."""
        from rag_mastery.config import HyDEConfig

        config = HyDEConfig(
            hyde_temperature=0.5,
            num_hypothetical_docs=5,
            use_multi_hyde=False
        )
        assert config.hyde_temperature == 0.5
        assert config.num_hypothetical_docs == 5
        assert config.use_multi_hyde is False
