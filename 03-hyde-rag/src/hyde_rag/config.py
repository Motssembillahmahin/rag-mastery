"""Configuration for HyDE RAG."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings using pydantic-settings."""

    # Ollama settings
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    llm_model: str = Field(default="llama3", description="LLM model name")
    embedding_model: str = Field(default="all-minilm:latest", description="Embedding model name")

    # ChromaDB settings
    collection_name: str = Field(default="hyde_rag", description="ChromaDB collection name")
    persist_dir: str = Field(default="./chroma_db", description="ChromaDB persistence directory")

    # Chunking settings
    chunk_size: int = Field(default=1000, ge=100, description="Text chunk size")
    chunk_overlap: int = Field(default=200, ge=0, description="Text chunk overlap")

    # HyDE settings
    hyde_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for HyDE generation")
    num_hypothetical_docs: int = Field(default=3, ge=1, le=10, description="Number of hypothetical documents to generate")
    use_multi_hyde: bool = Field(default=True, description="Use multiple hypothetical documents")

    # Retrieval settings
    top_k: int = Field(default=4, ge=1, description="Number of documents to retrieve")
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum similarity score")

    # Paths
    data_dir: str = Field(default="data", description="Directory containing documents")
    persist_directory: str = Field(default="./chroma_db", description="ChromaDB persistence directory")

    model_config = {"env_prefix": "HYDE_", "env_file": ".env", "env_file_encoding": "utf-8"}


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()
