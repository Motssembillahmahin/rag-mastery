"""Vector store service."""

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from graph_rag.config import Settings
from graph_rag.exceptions import VectorStoreError
from graph_rag.services.embeddings import EmbeddingService


class VectorStoreService:
    """Manages ChromaDB vector store operations."""

    def __init__(self, settings: Settings, embedding_service: EmbeddingService):
        self.settings = settings
        self.embedding_service = embedding_service
        self._vector_store = None

    @property
    def vector_store(self) -> Chroma:
        """Get or create vector store instance."""
        if self._vector_store is None:
            raise VectorStoreError("Vector store not initialized. Call create() first.")
        return self._vector_store

    def create(self, documents: list[Document]) -> Chroma:
        """Create vector store from documents."""
        if not documents:
            raise VectorStoreError("No documents provided for vector store creation.")

        self._vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_service.embeddings,
            collection_name=self.settings.collection_name,
            persist_directory=str(self.settings.persist_path),
        )
        return self._vector_store

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Search for similar documents."""
        return self.vector_store.similarity_search(query, k=k)

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to existing vector store."""
        return self.vector_store.add_documents(documents)
