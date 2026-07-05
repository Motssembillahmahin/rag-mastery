"""Tests for configuration."""

from adaptive_rag.config import QueryType, Settings


def test_settings_defaults():
    """Test default settings values."""
    settings = Settings()

    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.llm_model == "llama3"
    assert settings.embedding_model == "all-minilm:latest"
    assert settings.collection_name == "adaptive_rag"
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.use_query_classification is True
    assert settings.confidence_threshold == 0.7
    assert settings.data_dir == "data"


def test_settings_custom():
    """Test custom settings values."""
    settings = Settings(
        llm_model="custom-model",
        chunk_size=500,
        confidence_threshold=0.8,
    )

    assert settings.llm_model == "custom-model"
    assert settings.chunk_size == 500
    assert settings.confidence_threshold == 0.8


def test_query_type_enum():
    """Test QueryType enum values."""
    assert QueryType.SIMPLE.value == "simple"
    assert QueryType.COMPLEX.value == "complex"
    assert QueryType.FACTUAL.value == "factual"
    assert QueryType.ANALYTICAL.value == "analytical"
    assert QueryType.CREATIVE.value == "creative"


def test_retrieval_config_defaults():
    """Test default retrieval configuration per query type."""
    settings = Settings()

    assert settings.retrieval_config["simple"]["top_k"] == 2
    assert settings.retrieval_config["simple"]["use_hyde"] is False
    assert settings.retrieval_config["complex"]["top_k"] == 6
    assert settings.retrieval_config["complex"]["use_hyde"] is True
    assert settings.retrieval_config["factual"]["top_k"] == 4
    assert settings.retrieval_config["analytical"]["top_k"] == 8
    assert settings.retrieval_config["creative"]["top_k"] == 4
