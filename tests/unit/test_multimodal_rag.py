"""Unit tests for Multimodal RAG."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


class TestMultimodalRAG:
    """Test cases for MultimodalRAG class."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from config import MultimodalRAGConfig

        config = MultimodalRAGConfig()
        assert config.llm_model == "llama3"
        assert config.embedding_model == "all-minilm:latest"
        assert config.vision_model == "llava:latest"
        assert config.chunk_size == 1000
        assert config.top_k == 4
        assert config.ocr_enabled is True

    def test_config_urls(self):
        """Test URL generation methods."""
        from config import MultimodalRAGConfig

        config = MultimodalRAGConfig()
        assert config.get_ollama_url() == "http://localhost:11434"

    def test_config_custom_values(self):
        """Test custom configuration values."""
        from config import MultimodalRAGConfig

        config = MultimodalRAGConfig(
            llm_model="mistral",
            chunk_size=500,
            top_k=8,
            ocr_enabled=False,
        )
        assert config.llm_model == "mistral"
        assert config.chunk_size == 500
        assert config.top_k == 8
        assert config.ocr_enabled is False

    def test_config_collections(self):
        """Test collection name defaults."""
        from config import MultimodalRAGConfig

        config = MultimodalRAGConfig()
        assert config.text_collection == "multimodal_text"
        assert config.image_collection == "multimodal_images"

    def test_config_weights(self):
        """Test text and image weight defaults."""
        from config import MultimodalRAGConfig

        config = MultimodalRAGConfig()
        assert config.text_weight == 0.7
        assert config.image_weight == 0.3
