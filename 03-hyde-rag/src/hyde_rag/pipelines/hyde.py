"""HyDE RAG pipeline."""

from hyde_rag.config import Settings, get_settings
from hyde_rag.models.schemas import QueryResult
from hyde_rag.services.document_loader import DocumentLoader
from hyde_rag.services.embeddings import EmbeddingService
from hyde_rag.services.generator import Generator
from hyde_rag.services.hypothetical_generator import HypotheticalGenerator
from hyde_rag.services.retriever import HyDERetriever
from hyde_rag.services.vector_store import VectorStore


class HyDEPipeline:
    """Orchestrates the HyDE RAG workflow."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

        # Initialize services
        self.document_loader = DocumentLoader(self.settings)
        self.embedding_service = EmbeddingService(self.settings)
        self.vector_store = VectorStore(self.settings, self.embedding_service)
        self.hypothetical_generator = HypotheticalGenerator(self.settings)
        self.retriever = HyDERetriever(
            self.settings, self.hypothetical_generator, self.vector_store
        )
        self.generator = Generator(self.settings)

    def ingest_documents(self, source: str) -> int:
        """Ingest documents from a file or directory."""
        from pathlib import Path

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

    def query(self, question: str, top_k: int | None = None) -> QueryResult:
        """Execute a HyDE RAG query."""
        # Retrieve using HyDE
        retrieval_result = self.retriever.retrieve(question, top_k=top_k)

        # Generate answer
        answer = self.generator.generate_answer(question, retrieval_result.real_matches)

        return QueryResult(
            question=question,
            answer=answer,
            context=retrieval_result.real_matches,
            hypothetical_docs=retrieval_result.hypothetical_docs,
        )
