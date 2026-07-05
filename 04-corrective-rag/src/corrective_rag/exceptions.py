"""Custom exceptions for Corrective RAG."""


class CorrectiveRAGError(Exception):
    """Base exception for Corrective RAG."""


class DocumentLoadError(CorrectiveRAGError):
    """Error loading documents."""


class VectorStoreError(CorrectiveRAGError):
    """Error with vector store operations."""


class GradingError(CorrectiveRAGError):
    """Error during document or answer grading."""


class PipelineError(CorrectiveRAGError):
    """Error in the RAG pipeline."""
