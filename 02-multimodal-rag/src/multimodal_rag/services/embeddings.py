"""Embedding service for Multimodal RAG."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

from multimodal_rag.config import MultimodalRAGConfig
from multimodal_rag.exceptions import ConfigurationError


class EmbeddingService:
    """Manages embeddings and text splitting."""

    def __init__(self, config: MultimodalRAGConfig):
        """Initialize the embedding service.

        Args:
            config: Configuration settings.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        self.config = config
        self._embeddings: OllamaEmbeddings | None = None

    @property
    def embeddings(self) -> OllamaEmbeddings:
        """Get or create embeddings instance.

        Returns:
            OllamaEmbeddings instance.
        """
        if self._embeddings is None:
            self._embeddings = OllamaEmbeddings(
                model=self.config.embedding_model,
                base_url=self.config.ollama_base_url,
            )
        return self._embeddings

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks.

        Args:
            documents: List of documents to split.

        Returns:
            List of chunked documents.
        """
        if not documents:
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

        return text_splitter.split_documents(documents)

    def embed_documents(self, documents: list[Document]) -> list[list[float]]:
        """Generate embeddings for documents.

        Args:
            documents: List of documents to embed.

        Returns:
            List of embedding vectors.
        """
        texts = [doc.page_content for doc in documents]
        return self.embeddings.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a query.

        Args:
            query: Query string.

        Returns:
            Embedding vector.
        """
        return self.embeddings.embed_query(query)
