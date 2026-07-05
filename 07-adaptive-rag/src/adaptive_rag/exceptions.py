"""Custom exceptions for Adaptive RAG."""


class AdaptiveRAGError(Exception):
    """Base exception for Adaptive RAG."""

    pass


class ClassificationError(AdaptiveRAGError):
    """Raised when query classification fails."""

    pass


class DocumentLoadError(AdaptiveRAGError):
    """Raised when document loading fails."""

    pass


class EmbeddingError(AdaptiveRAGError):
    """Raised when embedding generation fails."""

    pass


class VectorStoreError(AdaptiveRAGError):
    """Raised when vector store operations fail."""

    pass


class RetrievalError(AdaptiveRAGError):
    """Raised when retrieval operations fail."""

    pass


class GenerationError(AdaptiveRAGError):
    """Raised when answer generation fails."""

    pass


class ConfigurationError(AdaptiveRAGError):
    """Raised when configuration is invalid."""

    pass
