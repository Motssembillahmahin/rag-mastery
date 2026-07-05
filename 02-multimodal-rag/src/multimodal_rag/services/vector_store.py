"""Vector store service for Multimodal RAG."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from multimodal_rag.config import MultimodalRAGConfig
from multimodal_rag.exceptions import VectorStoreError
from multimodal_rag.services.embeddings import EmbeddingService


class VectorStoreService:
    """Manages vector stores for text and images."""

    def __init__(self, config: MultimodalRAGConfig, embedding_service: EmbeddingService):
        """Initialize the vector store service.

        Args:
            config: Configuration settings.
            embedding_service: Embedding service instance.
        """
        self.config = config
        self.embedding_service = embedding_service
        self.text_store: Chroma | None = None
        self.image_store: Chroma | None = None

    def create_text_store(self, documents: list[Document]) -> Chroma:
        """Create or get the text vector store.

        Args:
            documents: List of documents to index.

        Returns:
            Chroma vector store for text.

        Raises:
            VectorStoreError: If creation fails.
        """
        try:
            persist_dir = Path(self.config.persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)

            self.text_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_service.embeddings,
                collection_name=self.config.text_collection,
                persist_directory=str(persist_dir),
            )
            return self.text_store
        except Exception as e:
            raise VectorStoreError(f"Failed to create text store: {e}") from e

    def create_image_store(self, documents: list[Document]) -> Chroma | None:
        """Create or get the image vector store.

        Args:
            documents: List of image documents to index.

        Returns:
            Chroma vector store for images, or None if no documents.

        Raises:
            VectorStoreError: If creation fails.
        """
        if not documents:
            return None

        try:
            persist_dir = Path(self.config.persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)

            self.image_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_service.embeddings,
                collection_name=self.config.image_collection,
                persist_directory=str(persist_dir),
            )
            return self.image_store
        except Exception as e:
            raise VectorStoreError(f"Failed to create image store: {e}") from e

    def get_text_store(self) -> Chroma | None:
        """Get the text vector store.

        Returns:
            Chroma vector store or None.
        """
        return self.text_store

    def get_image_store(self) -> Chroma | None:
        """Get the image vector store.

        Returns:
            Chroma vector store or None.
        """
        return self.image_store

    def search_text(self, query: str, top_k: int = 4) -> list[Document]:
        """Search the text vector store.

        Args:
            query: Search query.
            top_k: Number of results.

        Returns:
            List of matching documents.

        Raises:
            VectorStoreError: If search fails.
        """
        if self.text_store is None:
            return []

        try:
            return self.text_store.similarity_search(query, k=top_k)
        except Exception as e:
            raise VectorStoreError(f"Text search failed: {e}") from e

    def search_images(self, query: str, top_k: int = 4) -> list[Document]:
        """Search the image vector store.

        Args:
            query: Search query.
            top_k: Number of results.

        Returns:
            List of matching documents.

        Raises:
            VectorStoreError: If search fails.
        """
        if self.image_store is None:
            return []

        try:
            return self.image_store.similarity_search(query, k=top_k)
        except Exception as e:
            raise VectorStoreError(f"Image search failed: {e}") from e

    def get_collection_stats(self) -> dict[str, int]:
        """Get statistics about the vector stores.

        Returns:
            Dictionary with collection counts.
        """
        stats = {"text_count": 0, "image_count": 0}

        if self.text_store:
            stats["text_count"] = self.text_store._collection.count()

        if self.image_store:
            stats["image_count"] = self.image_store._collection.count()

        return stats
