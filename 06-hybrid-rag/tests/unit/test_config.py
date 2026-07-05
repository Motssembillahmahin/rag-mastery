"""Unit tests for configuration."""

import pytest
from pathlib import Path

from hybrid_rag.config import HybridRAGConfig


class TestHybridRAGConfig:
    """Tests for HybridRAGConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HybridRAGConfig()

        assert config.ollama_base_url == "http://localhost:11434"
        assert config.llm_model == "llama3"
        assert config.embedding_model == "all-minilm:latest"
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.semantic_weight == 0.6
        assert config.keyword_weight == 0.75
        assert config.use_rrf is True
        assert config.rrf_k == 60
        assert config.top_k == 4
        assert config.initial_retrieval_k == 10
        assert config.bm25_k1 == 1.5
        assert config.bm25_b == 0.75

    def test_custom_config(self):
        """Test custom configuration values."""
        config = HybridRAGConfig(
            llm_model="llama2",
            chunk_size=500,
            top_k=10,
            semantic_weight=0.7,
            keyword_weight=0.3,
            use_rrf=False,
            rrf_k=30,
        )

        assert config.llm_model == "llama2"
        assert config.chunk_size == 500
        assert config.top_k == 10
        assert config.semantic_weight == 0.7
        assert config.keyword_weight == 0.3
        assert config.use_rrf is False
        assert config.rrf_k == 30

    def test_get_ollama_url(self):
        """Test Ollama URL generation."""
        config = HybridRAGConfig()
        assert config.get_ollama_url() == "http://localhost:11434"

    def test_get_supported_document_extensions(self):
        """Test supported document extensions."""
        config = HybridRAGConfig()
        extensions = config.get_supported_document_extensions()

        assert ".pdf" in extensions
        assert ".txt" in extensions

    def test_directories_created(self, tmp_path):
        """Test that directories are created on initialization."""
        test_dir = tmp_path / "test_rag"
        config = HybridRAGConfig(
            data_dir=str(test_dir / "data"),
        )

        assert Path(config.data_dir).exists()

    def test_weights_sum_to_one(self):
        """Test that semantic and keyword weights sum to 1.0."""
        config = HybridRAGConfig()
        total = config.semantic_weight + config.keyword_weight
        assert abs(total - 1.0) < 0.01
