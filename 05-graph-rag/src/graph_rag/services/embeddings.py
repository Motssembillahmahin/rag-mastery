"""Embedding service."""

from langchain_ollama import OllamaEmbeddings

from graph_rag.config import Settings


class EmbeddingService:
    """Manages embedding generation."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._embeddings = None

    @property
    def embeddings(self) -> OllamaEmbeddings:
        """Get or create embeddings instance."""
        if self._embeddings is None:
            self._embeddings = OllamaEmbeddings(
                model=self.settings.embedding_model,
                base_url=self.settings.ollama_base_url,
            )
        return self._embeddings
