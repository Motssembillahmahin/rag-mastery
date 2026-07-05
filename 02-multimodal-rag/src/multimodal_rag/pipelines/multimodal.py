"""Multimodal RAG pipeline orchestrator."""

from langchain_core.documents import Document

from multimodal_rag.config import MultimodalRAGConfig
from multimodal_rag.exceptions import (
    DocumentLoadError,
    ImageProcessingError,
    MultimodalRAGError,
    VectorStoreError,
)
from multimodal_rag.models.schemas import (
    MultimodalQueryResult,
    ProcessedImage,
    QueryRequest,
    QueryResponse,
)
from multimodal_rag.services.document_loader import DocumentLoader
from multimodal_rag.services.embeddings import EmbeddingService
from multimodal_rag.services.generator import Generator
from multimodal_rag.services.image_processor import ImageProcessor
from multimodal_rag.services.retriever import MultimodalRetriever
from multimodal_rag.services.vector_store import VectorStoreService


class MultimodalRAGPipeline:
    """Orchestrates the multimodal RAG workflow."""

    def __init__(self, config: MultimodalRAGConfig | None = None):
        """Initialize the pipeline.

        Args:
            config: Configuration settings. Uses defaults if None.
        """
        self.config = config or MultimodalRAGConfig()
        self.document_loader = DocumentLoader(self.config)
        self.image_processor = ImageProcessor(self.config)
        self.embedding_service = EmbeddingService(self.config)
        self.vector_store = VectorStoreService(self.config, self.embedding_service)
        self.retriever = MultimodalRetriever(self.config, self.vector_store)
        self.generator = Generator(self.config)

    def ingest_documents(self, directory: str | None = None) -> dict[str, int]:
        """Ingest documents from a directory.

        Args:
            directory: Directory path. Uses config.documents_dir if None.

        Returns:
            Dictionary with counts of ingested documents and chunks.
        """
        try:
            documents = self.document_loader.load_documents_from_directory(directory)
            if not documents:
                return {"documents": 0, "chunks": 0}

            chunks = self.embedding_service.split_documents(documents)
            self.vector_store.create_text_store(chunks)

            return {"documents": len(documents), "chunks": len(chunks)}
        except (DocumentLoadError, VectorStoreError) as e:
            raise MultimodalRAGError(f"Document ingestion failed: {e}") from e

    def ingest_images(self, directory: str | None = None) -> dict[str, int]:
        """Ingest images from a directory.

        Args:
            directory: Directory path. Uses config.images_dir if None.

        Returns:
            Dictionary with counts of ingested images.
        """
        try:
            processed_images = self.image_processor.load_and_process_images(directory)
            if not processed_images:
                return {"images": 0}

            documents = self.image_processor.to_documents(processed_images)
            self.vector_store.create_image_store(documents)

            return {"images": len(processed_images)}
        except ImageProcessingError as e:
            raise MultimodalRAGError(f"Image ingestion failed: {e}") from e

    def ingest_all(self) -> dict[str, int]:
        """Ingest all documents and images.

        Returns:
            Dictionary with ingestion statistics.
        """
        stats = {}
        stats.update(self.ingest_documents())
        stats.update(self.ingest_images())
        return stats

    def query(self, request: QueryRequest) -> QueryResponse:
        """Execute a query against the RAG pipeline.

        Args:
            request: Query request.

        Returns:
            QueryResponse with answer and sources.
        """
        try:
            # Retrieve relevant documents
            results = self.retriever.retrieve(
                query=request.question,
                top_k=request.top_k,
                text_weight=request.text_weight,
                image_weight=request.image_weight,
            )

            if not results:
                return QueryResponse(
                    question=request.question,
                    answer="No relevant content found.",
                    sources=[],
                    num_text_chunks=0,
                    num_images=0,
                )

            # Generate answer
            answer = self.generator.generate(request.question, results)

            # Count sources by type
            num_text = sum(1 for r in results if r.source_type == "text")
            num_images = sum(1 for r in results if r.source_type == "image")

            return QueryResponse(
                question=request.question,
                answer=answer,
                sources=results,
                num_text_chunks=num_text,
                num_images=num_images,
            )
        except MultimodalRAGError:
            raise
        except Exception as e:
            raise MultimodalRAGError(f"Query failed: {e}") from e

    def get_stats(self) -> dict[str, int]:
        """Get pipeline statistics.

        Returns:
            Dictionary with vector store statistics.
        """
        return self.vector_store.get_collection_stats()
