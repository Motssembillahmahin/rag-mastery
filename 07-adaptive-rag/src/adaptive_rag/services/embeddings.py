"""Embedding service."""

from langchain_ollama import OllamaEmbeddings

from adaptive_rag.config import Settings
from adaptive_rag.exceptions import EmbeddingError


class EmbeddingService:
    """Manages embedding generation."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._embeddings = None

    @property
    def embeddings(self) -> OllamaEmbeddings:
        """Get or create embeddings instance."""
        if self._embeddings is None:
            try:
                self._embeddings = OllamaEmbeddings(
                    model=self.settings.embedding_model,
                    base_url=self.settings.ollama_base_url,
                )
            except Exception as e:
                raise EmbeddingError(f"Failed to initialize embeddings: {e}") from e
        return self._embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a query text."""
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            raise EmbeddingError(f"Failed to embed query: {e}") from e

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents."""
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            raise EmbeddingError(f"Failed to embed documents: {e}") from e
