"""Configuration for Adaptive RAG."""

from enum import Enum
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings


class QueryType(Enum):
    """Types of queries for routing."""

    SIMPLE = "simple"
    COMPLEX = "complex"
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"


class Settings(BaseSettings):
    """Application settings using pydantic-settings."""

    # Ollama settings
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    llm_model: str = Field(default="llama3", description="LLM model name")
    embedding_model: str = Field(default="all-minilm:latest", description="Embedding model name")

    # ChromaDB settings
    collection_name: str = Field(default="adaptive_rag", description="ChromaDB collection name")
    persist_dir: str = Field(default="./chroma_db", description="ChromaDB persistence directory")

    # Chunking settings
    chunk_size: int = Field(default=1000, ge=100, description="Text chunk size")
    chunk_overlap: int = Field(default=200, ge=0, description="Text chunk overlap")

    # Adaptive routing settings
    use_query_classification: bool = Field(
        default=True, description="Enable LLM-based query classification"
    )
    confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )

    # Retrieval settings per query type
    retrieval_config: dict[str, dict[str, Any]] = Field(
        default_factory=lambda: {
            QueryType.SIMPLE: {"top_k": 2, "use_hyde": False},
            QueryType.COMPLEX: {"top_k": 6, "use_hyde": True},
            QueryType.FACTUAL: {"top_k": 4, "use_hyde": False},
            QueryType.ANALYTICAL: {"top_k": 8, "use_hyde": True},
            QueryType.CREATIVE: {"top_k": 4, "use_hyde": False},
        },
        description="Retrieval configuration per query type",
    )

    # Paths
    data_dir: str = Field(default="data", description="Directory containing documents")

    model_config = {"env_prefix": "ADAPTIVE_", "env_file": ".env", "env_file_encoding": "utf-8"}


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()
