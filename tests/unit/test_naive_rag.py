"""Unit tests for Naive RAG."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


class TestNaiveRAG:
    """Test cases for NaiveRAG class."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from rag_mastery.config import RAGConfig

        config = RAGConfig()
        assert config.llm_model == "llama3"
        assert config.embedding_model == "all-minilm:latest"
        assert config.chunk_size == 1000
        assert config.top_k == 4

    def test_config_urls(self):
        """Test URL generation methods."""
        from rag_mastery.config import RAGConfig

        config = RAGConfig()
        assert config.get_chroma_url() == "http://localhost:8000"
        assert config.get_ollama_url() == "http://localhost:11434"

    def test_config_custom_values(self):
        """Test custom configuration values."""
        from rag_mastery.config import RAGConfig

        config = RAGConfig(
            llm_model="mistral",
            chunk_size=500,
            top_k=8
        )
        assert config.llm_model == "mistral"
        assert config.chunk_size == 500
        assert config.top_k == 8
