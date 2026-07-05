"""Adaptive RAG pipeline."""

from pathlib import Path

from adaptive_rag.config import Settings, get_settings
from adaptive_rag.models.schemas import QueryResult
from adaptive_rag.services.document_loader import DocumentLoader
from adaptive_rag.services.embeddings import EmbeddingService
from adaptive_rag.services.generator import AdaptiveGenerator
from adaptive_rag.services.query_classifier import QueryClassifier
from adaptive_rag.services.retriever import AdaptiveRetriever
from adaptive_rag.services.vector_store import VectorStore


class AdaptivePipeline:
    """Orchestrates the Adaptive RAG workflow: classify -> route -> retrieve -> generate."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

        # Initialize services
        self.document_loader = DocumentLoader(self.settings)
        self.embedding_service = EmbeddingService(self.settings)
        self.vector_store = VectorStore(self.settings, self.embedding_service)
        self.classifier = QueryClassifier(self.settings)
        self.retriever = AdaptiveRetriever(self.settings, self.vector_store)
        self.generator = AdaptiveGenerator(self.settings)

    def ingest_documents(self, source: str) -> int:
        """Ingest documents from a file or directory."""
        path = Path(source)

        if path.is_file():
            documents = self.document_loader.load_file(source)
        elif path.is_dir():
            documents = self.document_loader.load_directory(source)
        else:
            raise ValueError(f"Source not found: {source}")

        if not documents:
            return 0

        chunks = self.document_loader.split_documents(documents)
        self.vector_store.add_documents(chunks)
        return len(chunks)

    def query(self, question: str) -> QueryResult:
        """Execute the full adaptive RAG pipeline."""
        # Step 1: Classify query
        classification = self.classifier.classify(question)

        # Step 2: Adaptive retrieval based on query type
        retrieval_result = self.retriever.retrieve(question, classification.query_type)

        # Step 3: Generate type-specific answer
        answer = self.generator.generate_answer(
            question, retrieval_result.chunks, classification.query_type
        )

        return QueryResult(
            question=question,
            answer=answer,
            query_type=classification.query_type,
            confidence=classification.confidence,
            context=retrieval_result.chunks,
            used_hyde=retrieval_result.used_hyde,
        )
