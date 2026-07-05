"""Tests for configuration."""

from corrective_rag.config import Settings


def test_default_settings():
    """Test default settings values."""
    settings = Settings()

    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.llm_model == "llama3"
    assert settings.embedding_model == "all-minilm:latest"
    assert settings.relevance_threshold == 0.7
    assert settings.max_correction_attempts == 3
    assert settings.use_grading is True
    assert settings.use_self_rag is True


def test_custom_settings():
    """Test custom settings values."""
    settings = Settings(
        llm_model="custom-model",
        relevance_threshold=0.8,
        max_correction_attempts=5,
    )

    assert settings.llm_model == "custom-model"
    assert settings.relevance_threshold == 0.8
    assert settings.max_correction_attempts == 5


def test_data_path_property():
    """Test data_path property."""
    settings = Settings(data_dir="my_data")
    assert str(settings.data_path) == "my_data"


def test_persist_path_property():
    """Test persist_path property."""
    settings = Settings(persist_dir="./my_chroma")
    assert str(settings.persist_path) == "./my_chroma"
