"""Hybrid retriever service for combining semantic and keyword search."""

from langchain_core.documents import Document

from hybrid_rag.config import HybridRAGConfig
from hybrid_rag.exceptions import RetrievalError
from hybrid_rag.models.schemas import HybridQueryResult
from hybrid_rag.services.fusion import FusionService
from hybrid_rag.services.keyword_search import KeywordSearchService
from hybrid_rag.services.vector_store import VectorStoreService


class HybridRetriever:
    """Retrieves documents combining semantic and keyword search."""

    def __init__(
        self,
        config: HybridRAGConfig,
        vector_store: VectorStoreService,
        keyword_search: KeywordSearchService,
        fusion_service: FusionService,
    ):
        """Initialize the hybrid retriever.

        Args:
            config: Configuration settings.
            vector_store: Vector store service instance.
            keyword_search: Keyword search service instance.
            fusion_service: Fusion service instance.
        """
        self.config = config
        self.vector_store = vector_store
        self.keyword_search = keyword_search
        self.fusion_service = fusion_service

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        use_rrf: bool | None = None,
    ) -> list[HybridQueryResult]:
        """Retrieve relevant documents using hybrid search.

        Args:
            query: Search query.
            top_k: Number of final results to return.
            use_rrf: Override RRF fusion setting.

        Returns:
            List of HybridQueryResult with scores.

        Raises:
            RetrievalError: If retrieval fails.
        """
        k = top_k or self.config.top_k
        initial_k = self.config.initial_retrieval_k

        try:
            # Semantic search
            semantic_results = self.vector_store.search(query, top_k=initial_k)

            # Keyword search
            keyword_results = self.keyword_search.search(query, top_k=initial_k)

            # Fuse results
            fused_docs = self.fusion_service.fuse(
                semantic_results, keyword_results, use_rrf=use_rrf
            )

            # Convert to HybridQueryResult
            results = []
            for i, doc in enumerate(fused_docs[:k]):
                # Determine source type based on presence in initial results
                source_type = "semantic"
                raw_score = 0.0

                for sem_doc, sem_score in semantic_results:
                    if sem_doc.page_content[:100] == doc.page_content[:100]:
                        source_type = "semantic"
                        raw_score = sem_score
                        break

                for kw_doc, kw_score in keyword_results:
                    if kw_doc.page_content[:100] == doc.page_content[:100]:
                        if source_type == "semantic":
                            source_type = "hybrid"
                        else:
                            source_type = "keyword"
                        raw_score = max(raw_score, kw_score)
                        break

                results.append(
                    HybridQueryResult(
                        content=doc.page_content,
                        source=doc.metadata.get("source", "Unknown"),
                        source_type=source_type,
                        raw_score=raw_score,
                        weighted_score=float(1.0 - (i / k)),
                        metadata=doc.metadata,
                    )
                )

            return results
        except Exception as e:
            raise RetrievalError(f"Hybrid retrieval failed: {e}") from e

    def get_context_text(self, results: list[HybridQueryResult]) -> str:
        """Generate context text from retrieval results.

        Args:
            results: List of retrieval results.

        Returns:
            Formatted context string.
        """
        context_parts = []
        for result in results:
            source_type = result.source_type.upper()
            source = result.source
            context_parts.append(
                f"[{source_type}] Source: {source}\n{result.content}"
            )
        return "\n\n".join(context_parts)
