"""Vector store service."""

from langchain_chroma import Chroma
from langchain_core.documents import Document

from adaptive_rag.config import Settings
from adaptive_rag.exceptions import VectorStoreError
from adaptive_rag.services.embeddings import EmbeddingService


class VectorStore:
    """Manages ChromaDB vector store operations."""

    def __init__(self, settings: Settings, embedding_service: EmbeddingService):
        self.settings = settings
        self.embedding_service = embedding_service
        self._store = None

    @property
    def store(self) -> Chroma:
        """Get or create vector store instance."""
        if self._store is None:
            try:
                self._store = Chroma(
                    collection_name=self.settings.collection_name,
                    embedding_function=self.embedding_service.embeddings,
                    persist_directory=self.settings.persist_dir,
                )
            except Exception as e:
                raise VectorStoreError(f"Failed to initialize vector store: {e}") from e
        return self._store

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the vector store."""
        try:
            self.store.add_documents(documents)
        except Exception as e:
            raise VectorStoreError(f"Failed to add documents: {e}") from e

    def similarity_search(
        self, query: str, k: int = 4, score_threshold: float = 0.0
    ) -> list[tuple[Document, float]]:
        """Search for similar documents."""
        try:
            results = self.store.similarity_search_with_score(query, k=k)
            return [(doc, score) for doc, score in results if score >= score_threshold]
        except Exception as e:
            raise VectorStoreError(f"Failed to search vector store: {e}") from e

    def as_retriever(self, k: int = 4):
        """Get a LangChain retriever interface."""
        return self.store.as_retriever(search_kwargs={"k": k})
