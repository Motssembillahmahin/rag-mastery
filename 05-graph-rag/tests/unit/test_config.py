"""Tests for configuration."""

from graph_rag.config import Settings


def test_default_settings():
    """Test default settings values."""
    settings = Settings()

    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.llm_model == "llama3"
    assert settings.embedding_model == "all-minilm:latest"
    assert settings.neo4j_uri == "bolt://localhost:7687"
    assert settings.neo4j_user == "neo4j"
    assert settings.max_entities == 50
    assert settings.graph_traversal_depth == 2


def test_custom_settings():
    """Test custom settings values."""
    settings = Settings(
        llm_model="custom-model",
        neo4j_uri="bolt://remote:7687",
        max_entities=20,
    )

    assert settings.llm_model == "custom-model"
    assert settings.neo4j_uri == "bolt://remote:7687"
    assert settings.max_entities == 20


def test_data_path_property():
    """Test data_path property."""
    settings = Settings(data_dir="my_data")
    assert str(settings.data_path) == "my_data"


def test_persist_path_property():
    """Test persist_path property."""
    settings = Settings(persist_dir="./my_chroma")
    assert str(settings.persist_path) == "./my_chroma"
