"""Hybrid RAG pipeline orchestrator."""

from hybrid_rag.config import HybridRAGConfig
from hybrid_rag.exceptions import (
    DocumentLoadError,
    FusionError,
    HybridRAGError,
    KeywordSearchError,
    RetrievalError,
    VectorStoreError,
)
from hybrid_rag.models.schemas import (
    HybridQueryResult,
    QueryRequest,
    QueryResponse,
)
from hybrid_rag.services.document_loader import DocumentLoader
from hybrid_rag.services.embeddings import EmbeddingService
from hybrid_rag.services.fusion import FusionService
from hybrid_rag.services.generator import Generator
from hybrid_rag.services.keyword_search import KeywordSearchService
from hybrid_rag.services.retriever import HybridRetriever
from hybrid_rag.services.vector_store import VectorStoreService


class HybridPipeline:
    """Orchestrates the hybrid RAG workflow."""

    def __init__(self, config: HybridRAGConfig | None = None):
        """Initialize the pipeline.

        Args:
            config: Configuration settings. Uses defaults if None.
        """
        self.config = config or HybridRAGConfig()
        self.document_loader = DocumentLoader(self.config)
        self.embedding_service = EmbeddingService(self.config)
        self.vector_store = VectorStoreService(self.config, self.embedding_service)
        self.keyword_search = KeywordSearchService(self.config)
        self.fusion_service = FusionService(self.config)
        self.retriever = HybridRetriever(
            self.config, self.vector_store, self.keyword_search, self.fusion_service
        )
        self.generator = Generator(self.config)

    def ingest_documents(self, directory: str | None = None) -> dict[str, int]:
        """Ingest documents from a directory.

        Args:
            directory: Directory path. Uses config.data_dir if None.

        Returns:
            Dictionary with counts of ingested documents and chunks.
        """
        try:
            documents = self.document_loader.load_documents_from_directory(directory)
            if not documents:
                return {"documents": 0, "chunks": 0}

            chunks = self.embedding_service.split_documents(documents)

            # Dual indexing: vector store + BM25
            self.vector_store.create_store(chunks)
            self.keyword_search.build_index(chunks)

            return {"documents": len(documents), "chunks": len(chunks)}
        except (DocumentLoadError, VectorStoreError, KeywordSearchError) as e:
            raise HybridRAGError(f"Document ingestion failed: {e}") from e

    def query(self, request: QueryRequest) -> QueryResponse:
        """Execute a query against the RAG pipeline.

        Args:
            request: Query request.

        Returns:
            QueryResponse with answer and sources.
        """
        try:
            results = self.retriever.retrieve(
                query=request.question,
                top_k=request.top_k,
                use_rrf=request.use_rrf,
            )

            if not results:
                return QueryResponse(
                    question=request.question,
                    answer="No relevant content found.",
                    sources=[],
                    fusion_method="rrf" if self.config.use_rrf else "weighted",
                    num_results=0,
                )

            answer = self.generator.generate(request.question, results)

            fusion_method = "rrf"
            if request.use_rrf is not None:
                fusion_method = "rrf" if request.use_rrf else "weighted"
            elif self.config.use_rrf:
                fusion_method = "rrf"
            else:
                fusion_method = "weighted"

            return QueryResponse(
                question=request.question,
                answer=answer,
                sources=results,
                fusion_method=fusion_method,
                num_results=len(results),
            )
        except (RetrievalError, FusionError):
            raise
        except Exception as e:
            raise HybridRAGError(f"Query failed: {e}") from e

    def get_stats(self) -> dict[str, int]:
        """Get pipeline statistics.

        Returns:
            Dictionary with index statistics.
        """
        stats = self.vector_store.get_collection_stats()
        stats["bm25_count"] = self.keyword_search.get_index_size()
        return stats
