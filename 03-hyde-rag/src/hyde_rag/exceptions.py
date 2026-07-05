"""Custom exceptions for HyDE RAG."""


class HyDEError(Exception):
    """Base exception for HyDE RAG."""

    pass


class DocumentLoadError(HyDEError):
    """Raised when document loading fails."""

    pass


class EmbeddingError(HyDEError):
    """Raised when embedding generation fails."""

    pass


class VectorStoreError(HyDEError):
    """Raised when vector store operations fail."""

    pass


class HypotheticalGenerationError(HyDEError):
    """Raised when hypothetical document generation fails."""

    pass


class RetrievalError(HyDEError):
    """Raised when retrieval operations fail."""

    pass


class ConfigurationError(HyDEError):
    """Raised when configuration is invalid."""

    pass
