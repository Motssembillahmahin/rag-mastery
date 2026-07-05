"""Configuration for Naive RAG."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RAGConfig:
    """RAG configuration settings."""

    # LLM settings
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3"
    embedding_model: str = "all-minilm:latest"

    # ChromaDB settings
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    collection_name: str = "naive_rag"

    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval settings
    top_k: int = 4
    score_threshold: Optional[float] = None

    # Paths
    data_dir: str = "data"
    persist_dir: str = "./chroma_db"

    def get_chroma_url(self) -> str:
        """Get ChromaDB connection URL."""
        return f"http://{self.chroma_host}:{self.chroma_port}"

    def get_ollama_url(self) -> str:
        """Get Ollama connection URL."""
        return self.ollama_base_url