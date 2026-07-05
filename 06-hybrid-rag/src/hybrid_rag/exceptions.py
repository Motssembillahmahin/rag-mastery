"""Custom exceptions for Hybrid RAG."""


class HybridRAGError(Exception):
    """Base exception for Hybrid RAG."""

    pass


class DocumentLoadError(HybridRAGError):
    """Raised when document loading fails."""

    pass


class VectorStoreError(HybridRAGError):
    """Raised when vector store operations fail."""

    pass


class KeywordSearchError(HybridRAGError):
    """Raised when keyword search operations fail."""

    pass


class FusionError(HybridRAGError):
    """Raised when result fusion fails."""

    pass


class RetrievalError(HybridRAGError):
    """Raised when retrieval operations fail."""

    pass


class GenerationError(HybridRAGError):
    """Raised when answer generation fails."""

    pass


class ConfigurationError(HybridRAGError):
    """Raised when configuration is invalid."""

    pass
