"""Corrective RAG pipeline."""

from corrective_rag.config import Settings
from corrective_rag.exceptions import PipelineError
from corrective_rag.models.schemas import GradedDocument, RetrievalResult
from corrective_rag.services.document_loader import DocumentLoader
from corrective_rag.services.embeddings import EmbeddingService
from corrective_rag.services.generator import Generator
from corrective_rag.services.grader import AnswerGrader, DocumentGrader
from corrective_rag.services.retriever import CorrectiveRetriever
from corrective_rag.services.vector_store import VectorStoreService


class CorrectivePipeline:
    """Orchestrates the corrective RAG workflow."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()

        # Initialize services
        self.document_loader = DocumentLoader(self.settings)
        self.embedding_service = EmbeddingService(self.settings)
        self.vector_store = VectorStoreService(self.settings, self.embedding_service)
        self.document_grader = DocumentGrader(self.settings)
        self.answer_grader = AnswerGrader(self.settings)
        self.generator = Generator(self.settings)
        self.retriever = CorrectiveRetriever(
            self.settings, self.vector_store, self.document_grader
        )

    def ingest_documents(self, directory: str | None = None) -> int:
        """Load and ingest documents into vector store."""
        data_dir = directory or str(self.settings.data_path)

        documents = self.document_loader.load_directory(data_dir)
        if not documents:
            raise PipelineError(f"No documents found in {data_dir}")

        chunks = self.document_loader.split_documents(documents)
        self.vector_store.create(chunks)

        return len(chunks)

    def query(self, question: str, use_grading: bool | None = None, use_self_rag: bool | None = None) -> str:
        """Execute the corrective RAG pipeline."""
        use_grading = use_grading if use_grading is not None else self.settings.use_grading
        use_self_rag = use_self_rag if use_self_rag is not None else self.settings.use_self_rag

        # Retrieval with self-correction
        retrieval_result = self.retriever.retrieve(question)

        if not retrieval_result.documents:
            return "No relevant documents found after multiple attempts."

        # Generate answer
        answer = self.generator.generate(question, retrieval_result.documents)

        # Self-RAG validation and regeneration
        if use_self_rag:
            quality = self.answer_grader.grade(question, answer, retrieval_result.documents)

            if not quality.satisfactory:
                answer = self.generator.regenerate(question, retrieval_result.documents)

        return answer

    def get_retrieval_info(self, question: str) -> RetrievalResult:
        """Get detailed retrieval information."""
        return self.retriever.retrieve(question)
