"""Configuration for Hybrid RAG."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class HybridRAGConfig:
    """Hybrid RAG configuration settings."""

    # LLM settings
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3"
    embedding_model: str = "all-minilm:latest"

    # ChromaDB settings
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    collection_name: str = "hybrid_rag"

    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Hybrid search settings
    semantic_weight: float = 0.6
    keyword_weight: float = 0.4
    use_rrf: bool = True
    rrf_k: int = 60

    # Retrieval settings
    top_k: int = 4
    initial_retrieval_k: int = 10

    # BM25 settings
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Paths
    data_dir: str = "data"
    persist_dir: str = "./chroma_db"

    def __post_init__(self):
        """Ensure directories exist."""
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

    def get_ollama_url(self) -> str:
        """Get Ollama connection URL."""
        return self.ollama_base_url

    def get_supported_document_extensions(self) -> list[str]:
        """Get list of supported document file extensions."""
        return [".pdf", ".txt"]
