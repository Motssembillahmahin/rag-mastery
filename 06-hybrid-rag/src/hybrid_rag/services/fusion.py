"""Fusion service for combining semantic and keyword search results."""

from langchain_core.documents import Document

from hybrid_rag.config import HybridRAGConfig
from hybrid_rag.exceptions import FusionError


class FusionService:
    """Combines results from semantic and keyword search."""

    def __init__(self, config: HybridRAGConfig):
        """Initialize the fusion service.

        Args:
            config: Configuration settings.
        """
        self.config = config

    def reciprocal_rank_fusion(
        self,
        semantic_results: list[tuple[Document, float]],
        keyword_results: list[tuple[Document, float]],
        k: int | None = None,
    ) -> list[Document]:
        """Combine results using Reciprocal Rank Fusion (RRF).

        RRF_score(d) = Sum 1/(k + rank_i(d))

        Args:
            semantic_results: Results from semantic search.
            keyword_results: Results from keyword search.
            k: RRF constant (default from config).

        Returns:
            Fused list of documents.

        Raises:
            FusionError: If fusion fails.
        """
        rrf_k = k if k is not None else self.config.rrf_k

        try:
            doc_map: dict[str, Document] = {}
            rrf_scores: dict[str, float] = {}

            for rank, (doc, _) in enumerate(semantic_results):
                doc_key = doc.page_content[:100]
                if doc_key not in doc_map:
                    doc_map[doc_key] = doc
                    rrf_scores[doc_key] = 0
                rrf_scores[doc_key] += 1 / (rrf_k + rank + 1)

            for rank, (doc, _) in enumerate(keyword_results):
                doc_key = doc.page_content[:100]
                if doc_key not in doc_map:
                    doc_map[doc_key] = doc
                    rrf_scores[doc_key] = 0
                rrf_scores[doc_key] += 1 / (rrf_k + rank + 1)

            sorted_keys = sorted(
                rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
            )

            return [doc_map[key] for key in sorted_keys]
        except Exception as e:
            raise FusionError(f"RRF fusion failed: {e}") from e

    def weighted_fusion(
        self,
        semantic_results: list[tuple[Document, float]],
        keyword_results: list[tuple[Document, float]],
    ) -> list[Document]:
        """Combine results using weighted scoring.

        Args:
            semantic_results: Results from semantic search.
            keyword_results: Results from keyword search.

        Returns:
            Fused list of documents.

        Raises:
            FusionError: If fusion fails.
        """
        try:
            doc_scores: dict[str, dict] = {}

            if semantic_results:
                max_semantic = max(score for _, score in semantic_results)
                min_semantic = min(score for _, score in semantic_results)
                range_semantic = (
                    max_semantic - min_semantic if max_semantic != min_semantic else 1
                )

                for doc, score in semantic_results:
                    normalized = 1 - (score - min_semantic) / range_semantic
                    doc_key = doc.page_content[:100]
                    doc_scores[doc_key] = {
                        "doc": doc,
                        "score": normalized * self.config.semantic_weight,
                    }

            if keyword_results:
                max_keyword = max(score for _, score in keyword_results)
                min_keyword = min(score for _, score in keyword_results)
                range_keyword = (
                    max_keyword - min_keyword if max_keyword != min_keyword else 1
                )

                for doc, score in keyword_results:
                    normalized = (score - min_keyword) / range_keyword
                    doc_key = doc.page_content[:100]
                    if doc_key in doc_scores:
                        doc_scores[doc_key]["score"] += (
                            normalized * self.config.keyword_weight
                        )
                    else:
                        doc_scores[doc_key] = {
                            "doc": doc,
                            "score": normalized * self.config.keyword_weight,
                        }

            sorted_docs = sorted(
                doc_scores.values(),
                key=lambda x: x["score"],
                reverse=True,
            )

            return [item["doc"] for item in sorted_docs]
        except Exception as e:
            raise FusionError(f"Weighted fusion failed: {e}") from e

    def fuse(
        self,
        semantic_results: list[tuple[Document, float]],
        keyword_results: list[tuple[Document, float]],
        use_rrf: bool | None = None,
    ) -> list[Document]:
        """Fuse results using configured strategy.

        Args:
            semantic_results: Results from semantic search.
            keyword_results: Results from keyword search.
            use_rrf: Override RRF setting from config.

        Returns:
            Fused list of documents.
        """
        if use_rrf is None:
            use_rrf = self.config.use_rrf

        if use_rrf:
            return self.reciprocal_rank_fusion(semantic_results, keyword_results)
        else:
            return self.weighted_fusion(semantic_results, keyword_results)
