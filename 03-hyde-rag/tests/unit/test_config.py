"""Tests for configuration."""

from hyde_rag.config import Settings


def test_settings_defaults():
    """Test default settings values."""
    settings = Settings()

    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.llm_model == "llama3"
    assert settings.embedding_model == "all-minilm:latest"
    assert settings.collection_name == "hyde_rag"
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.hyde_temperature == 0.7
    assert settings.num_hypothetical_docs == 3
    assert settings.use_multi_hyde is True
    assert settings.top_k == 4
    assert settings.score_threshold == 0.5


def test_settings_custom():
    """Test custom settings values."""
    settings = Settings(
        llm_model="custom-model",
        chunk_size=500,
        hyde_temperature=0.5,
        num_hypothetical_docs=2,
    )

    assert settings.llm_model == "custom-model"
    assert settings.chunk_size == 500
    assert settings.hyde_temperature == 0.5
    assert settings.num_hypothetical_docs == 2
