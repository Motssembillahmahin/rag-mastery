"""Configuration for Corrective RAG."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # LLM settings
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3"
    embedding_model: str = "all-minilm:latest"

    # ChromaDB settings
    collection_name: str = "corrective_rag"
    persist_dir: str = "./chroma_db"

    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Corrective settings
    relevance_threshold: float = 0.7
    max_correction_attempts: int = 3
    use_grading: bool = True
    use_self_rag: bool = True

    # Retrieval settings
    top_k: int = 4
    initial_retrieval_k: int = 8

    # Paths
    data_dir: str = "data"

    @property
    def data_path(self) -> Path:
        """Get data directory path."""
        return Path(self.data_dir)

    @property
    def persist_path(self) -> Path:
        """Get persist directory path."""
        return Path(self.persist_dir)

    model_config = {"env_prefix": "CORRECTIVE_RAG_", "env_file": ".env"}


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()
