"""Vector store service for Hybrid RAG."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from hybrid_rag.config import HybridRAGConfig
from hybrid_rag.exceptions import VectorStoreError
from hybrid_rag.services.embeddings import EmbeddingService


class VectorStoreService:
    """Manages vector store for semantic search."""

    def __init__(self, config: HybridRAGConfig, embedding_service: EmbeddingService):
        """Initialize the vector store service.

        Args:
            config: Configuration settings.
            embedding_service: Embedding service instance.
        """
        self.config = config
        self.embedding_service = embedding_service
        self.store: Chroma | None = None

    def create_store(self, documents: list[Document]) -> Chroma:
        """Create or get the vector store.

        Args:
            documents: List of documents to index.

        Returns:
            Chroma vector store.

        Raises:
            VectorStoreError: If creation fails.
        """
        try:
            persist_dir = Path(self.config.persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)

            self.store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_service.embeddings,
                collection_name=self.config.collection_name,
                persist_directory=str(persist_dir),
            )
            return self.store
        except Exception as e:
            raise VectorStoreError(f"Failed to create vector store: {e}") from e

    def get_store(self) -> Chroma | None:
        """Get the vector store.

        Returns:
            Chroma vector store or None.
        """
        return self.store

    def search(self, query: str, top_k: int = 4) -> list[tuple[Document, float]]:
        """Search the vector store with similarity scores.

        Args:
            query: Search query.
            top_k: Number of results.

        Returns:
            List of (document, score) tuples.

        Raises:
            VectorStoreError: If search fails.
        """
        if self.store is None:
            return []

        try:
            return self.store.similarity_search_with_score(query, k=top_k)
        except Exception as e:
            raise VectorStoreError(f"Vector search failed: {e}") from e

    def get_collection_stats(self) -> dict[str, int]:
        """Get statistics about the vector store.

        Returns:
            Dictionary with collection count.
        """
        stats = {"vector_count": 0}

        if self.store:
            stats["vector_count"] = self.store._collection.count()

        return stats
