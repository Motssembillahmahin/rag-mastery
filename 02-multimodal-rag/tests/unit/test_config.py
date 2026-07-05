"""Unit tests for configuration."""

import pytest
from pathlib import Path

from multimodal_rag.config import MultimodalRAGConfig


class TestMultimodalRAGConfig:
    """Tests for MultimodalRAGConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MultimodalRAGConfig()

        assert config.ollama_base_url == "http://localhost:11434"
        assert config.llm_model == "llama3"
        assert config.embedding_model == "all-minilm:latest"
        assert config.vision_model == "llava:latest"
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.top_k == 4
        assert config.text_weight == 0.7
        assert config.image_weight == 0.3
        assert config.ocr_enabled is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MultimodalRAGConfig(
            llm_model="llama2",
            chunk_size=500,
            top_k=10,
            text_weight=0.8,
            image_weight=0.2,
        )

        assert config.llm_model == "llama2"
        assert config.chunk_size == 500
        assert config.top_k == 10
        assert config.text_weight == 0.8
        assert config.image_weight == 0.2

    def test_get_ollama_url(self):
        """Test Ollama URL generation."""
        config = MultimodalRAGConfig()
        assert config.get_ollama_url() == "http://localhost:11434"

    def test_get_supported_image_extensions(self):
        """Test supported image extensions."""
        config = MultimodalRAGConfig()
        extensions = config.get_supported_image_extensions()

        assert ".png" in extensions
        assert ".jpg" in extensions
        assert ".jpeg" in extensions
        assert ".gif" in extensions
        assert ".bmp" in extensions

    def test_get_supported_document_extensions(self):
        """Test supported document extensions."""
        config = MultimodalRAGConfig()
        extensions = config.get_supported_document_extensions()

        assert ".pdf" in extensions
        assert ".txt" in extensions
        assert ".md" in extensions
        assert ".docx" in extensions

    def test_directories_created(self, tmp_path):
        """Test that directories are created on initialization."""
        test_dir = tmp_path / "test_rag"
        config = MultimodalRAGConfig(
            data_dir=str(test_dir / "data"),
            documents_dir=str(test_dir / "data" / "documents"),
            images_dir=str(test_dir / "data" / "images"),
        )

        assert Path(config.data_dir).exists()
        assert Path(config.documents_dir).exists()
        assert Path(config.images_dir).exists()
