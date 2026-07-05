"""Configuration for Multimodal RAG."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MultimodalRAGConfig:
    """Multimodal RAG configuration settings."""

    # LLM settings
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3"
    embedding_model: str = "all-minilm:latest"
    vision_model: str = "llava:latest"

    # ChromaDB settings
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    text_collection: str = "multimodal_text"
    image_collection: str = "multimodal_images"

    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Image settings
    ocr_enabled: bool = True

    # Retrieval settings
    top_k: int = 4
    text_weight: float = 0.7
    image_weight: float = 0.3

    # Paths
    data_dir: str = "data"
    documents_dir: str = "data/documents"
    images_dir: str = "data/images"
    persist_dir: str = "./chroma_db"

    def __post_init__(self):
        """Ensure directories exist."""
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.documents_dir).mkdir(parents=True, exist_ok=True)
        Path(self.images_dir).mkdir(parents=True, exist_ok=True)

    def get_ollama_url(self) -> str:
        """Get Ollama connection URL."""
        return self.ollama_base_url

    def get_supported_image_extensions(self) -> list[str]:
        """Get list of supported image file extensions."""
        return [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"]

    def get_supported_document_extensions(self) -> list[str]:
        """Get list of supported document file extensions."""
        return [".pdf", ".txt", ".md", ".docx"]
