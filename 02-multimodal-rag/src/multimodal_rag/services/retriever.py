"""Retriever service for Multimodal RAG."""

from langchain_core.documents import Document

from multimodal_rag.config import MultimodalRAGConfig
from multimodal_rag.exceptions import RetrievalError
from multimodal_rag.models.schemas import MultimodalQueryResult
from multimodal_rag.services.vector_store import VectorStoreService


class MultimodalRetriever:
    """Retrieves documents from both text and image collections."""

    def __init__(self, config: MultimodalRAGConfig, vector_store: VectorStoreService):
        """Initialize the retriever.

        Args:
            config: Configuration settings.
            vector_store: Vector store service instance.
        """
        self.config = config
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        text_weight: float | None = None,
        image_weight: float | None = None,
    ) -> list[MultimodalQueryResult]:
        """Retrieve relevant documents from both collections.

        Args:
            query: Search query.
            top_k: Number of results per collection.
            text_weight: Weight for text results.
            image_weight: Weight for image results.

        Returns:
            List of MultimodalQueryResult with weighted scores.

        Raises:
            RetrievalError: If retrieval fails.
        """
        k = top_k or self.config.top_k
        t_weight = text_weight or self.config.text_weight
        i_weight = image_weight or self.config.image_weight

        results: list[MultimodalQueryResult] = []

        # Search text collection
        try:
            text_docs = self.vector_store.search_text(query, top_k=k)
            for i, doc in enumerate(text_docs):
                raw_score = 1.0 - (i / k)  # Simple ranking score
                weighted_score = raw_score * t_weight

                results.append(
                    MultimodalQueryResult(
                        content=doc.page_content,
                        source=doc.metadata.get("source", "Unknown"),
                        source_type="text",
                        raw_score=raw_score,
                        weighted_score=weighted_score,
                        metadata=doc.metadata,
                    )
                )
        except RetrievalError as e:
            print(f"Warning: Text search failed: {e}")

        # Search image collection
        try:
            image_docs = self.vector_store.search_images(query, top_k=k)
            for i, doc in enumerate(image_docs):
                raw_score = 1.0 - (i / k)  # Simple ranking score
                weighted_score = raw_score * i_weight

                results.append(
                    MultimodalQueryResult(
                        content=doc.page_content,
                        source=doc.metadata.get("source", "Unknown"),
                        source_type="image",
                        raw_score=raw_score,
                        weighted_score=weighted_score,
                        metadata=doc.metadata,
                    )
                )
        except RetrievalError as e:
            print(f"Warning: Image search failed: {e}")

        # Sort by weighted score
        results.sort(key=lambda x: x.weighted_score, reverse=True)

        return results[:k]

    def get_context_text(self, results: list[MultimodalQueryResult]) -> str:
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
