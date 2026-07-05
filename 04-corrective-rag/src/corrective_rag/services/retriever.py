"""Corrective retrieval service."""

from langchain_core.documents import Document

from corrective_rag.config import Settings
from corrective_rag.models.schemas import GradedDocument, RetrievalResult
from corrective_rag.services.grader import DocumentGrader
from corrective_rag.services.vector_store import VectorStoreService


class CorrectiveRetriever:
    """Retrieves and filters documents with self-correction."""

    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStoreService,
        grader: DocumentGrader,
    ):
        self.settings = settings
        self.vector_store = vector_store
        self.grader = grader

    def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve documents with self-correction loop."""
        attempt = 0
        all_relevant_docs = []
        used_doc_contents = set()

        while attempt < self.settings.max_correction_attempts:
            attempt += 1

            k = self.settings.initial_retrieval_k if attempt == 1 else self.settings.top_k
            docs = self.vector_store.similarity_search(query, k=k)

            if self.settings.use_grading:
                graded_docs = self._grade_and_filter(query, docs)
            else:
                graded_docs = [
                    GradedDocument(
                        content=doc.page_content,
                        metadata=doc.metadata,
                        score=1.0,
                        relevant=True,
                    )
                    for doc in docs
                ]

            for doc in graded_docs:
                if doc.content not in used_doc_contents:
                    all_relevant_docs.append(doc)
                    used_doc_contents.add(doc.content)

            if len(all_relevant_docs) >= self.settings.top_k:
                break

            if attempt < self.settings.max_correction_attempts:
                pass  # Continue to next attempt

        final_docs = all_relevant_docs[:self.settings.top_k]

        return RetrievalResult(
            query=query,
            documents=final_docs,
            attempts=attempt,
            filtered_count=len(final_docs),
        )

    def _grade_and_filter(self, query: str, docs: list[Document]) -> list[GradedDocument]:
        """Grade documents and filter by relevance."""
        graded_docs = [
            GradedDocument(
                content=doc.page_content,
                metadata=doc.metadata,
                score=0.0,
                relevant=False,
            )
            for doc in docs
        ]

        graded_docs = self.grader.grade_documents(query, graded_docs)
        return self.grader.filter_relevant(graded_docs)
