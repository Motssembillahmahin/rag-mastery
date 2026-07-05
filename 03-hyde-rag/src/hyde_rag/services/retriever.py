"""HyDE retriever service."""

from langchain_core.documents import Document

from hyde_rag.config import Settings
from hyde_rag.exceptions import RetrievalError
from hyde_rag.models.schemas import (
    DocumentChunk,
    HyDERetrievalResult,
    HypotheticalDocument,
)
from hyde_rag.services.hypothetical_generator import HypotheticalGenerator
from hyde_rag.services.vector_store import VectorStore


class HyDERetriever:
    """Retrieves documents using HyDE approach."""

    def __init__(
        self,
        settings: Settings,
        hypothetical_generator: HypotheticalGenerator,
        vector_store: VectorStore,
    ):
        self.settings = settings
        self.hypothetical_generator = hypothetical_generator
        self.vector_store = vector_store

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> HyDERetrievalResult:
        """Retrieve documents using HyDE."""
        k = top_k or self.settings.top_k

        try:
            # Generate hypothetical documents
            if self.settings.use_multi_hyde:
                hypothetical_docs = self.hypothetical_generator.generate_multiple(query)
            else:
                hypothetical_docs = [self.hypothetical_generator.generate(query)]

            # Retrieve using each hypothetical document
            all_results = []
            for hyp_doc in hypothetical_docs:
                results = self.vector_store.similarity_search(
                    hyp_doc.content, k=k, score_threshold=self.settings.score_threshold
                )
                all_results.extend(results)

            # Deduplicate results
            unique_results = self._deduplicate_results(all_results)

            # Sort by score and take top_k
            unique_results.sort(key=lambda x: x[1])
            top_results = unique_results[:k]

            # Convert to schema format
            real_matches = [
                DocumentChunk(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    chunk_index=doc.metadata.get("chunk_index", 0),
                    metadata=doc.metadata,
                )
                for doc, _ in top_results
            ]
            scores = [score for _, score in top_results]

            return HyDERetrievalResult(
                hypothetical_docs=hypothetical_docs,
                real_matches=real_matches,
                scores=scores,
            )
        except RetrievalError:
            raise
        except Exception as e:
            raise RetrievalError(f"HyDE retrieval failed: {e}") from e

    def _deduplicate_results(
        self, results: list[tuple[Document, float]]
    ) -> list[tuple[Document, float]]:
        """Deduplicate results by content."""
        seen_content = set()
        unique_results = []

        for doc, score in results:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                unique_results.append((doc, score))

        return unique_results
