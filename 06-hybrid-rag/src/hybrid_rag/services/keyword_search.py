"""Keyword search service for Hybrid RAG using BM25."""

import numpy as np
from rank_bm25 import BM25Okapi

from langchain_core.documents import Document

from hybrid_rag.config import HybridRAGConfig
from hybrid_rag.exceptions import KeywordSearchError


class KeywordSearchService:
    """Manages BM25 index for keyword search."""

    def __init__(self, config: HybridRAGConfig):
        """Initialize the keyword search service.

        Args:
            config: Configuration settings.
        """
        self.config = config
        self.bm25: BM25Okapi | None = None
        self.documents: list[Document] = []

    def build_index(self, documents: list[Document]) -> None:
        """Build BM25 index from documents.

        Args:
            documents: List of documents to index.

        Raises:
            KeywordSearchError: If index building fails.
        """
        if not documents:
            raise KeywordSearchError("Cannot build index from empty document list")

        try:
            self.documents = documents
            tokenized_docs = [
                doc.page_content.lower().split() for doc in documents
            ]

            self.bm25 = BM25Okapi(
                tokenized_docs,
                k1=self.config.bm25_k1,
                b=self.config.bm25_b,
            )
        except Exception as e:
            raise KeywordSearchError(f"Failed to build BM25 index: {e}") from e

    def search(self, query: str, top_k: int = 10) -> list[tuple[Document, float]]:
        """Search using BM25 keyword matching.

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            List of (document, score) tuples.

        Raises:
            KeywordSearchError: If search fails.
        """
        if self.bm25 is None:
            raise KeywordSearchError("BM25 index not built. Call build_index() first.")

        try:
            tokenized_query = query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)

            top_indices = np.argsort(scores)[::-1][:top_k]

            results = []
            for idx in top_indices:
                if idx < len(self.documents):
                    results.append((self.documents[idx], float(scores[idx])))

            return results
        except Exception as e:
            raise KeywordSearchError(f"BM25 search failed: {e}") from e

    def get_index_size(self) -> int:
        """Get the number of documents in the index.

        Returns:
            Number of indexed documents.
        """
        return len(self.documents)
