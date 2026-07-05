"""Custom exceptions for Multimodal RAG."""


class MultimodalRAGError(Exception):
    """Base exception for Multimodal RAG."""

    pass


class DocumentLoadError(MultimodalRAGError):
    """Raised when document loading fails."""

    pass


class ImageProcessingError(MultimodalRAGError):
    """Raised when image processing fails."""

    pass


class OCRError(ImageProcessingError):
    """Raised when OCR extraction fails."""

    pass


class VisionModelError(ImageProcessingError):
    """Raised when vision model inference fails."""

    pass


class VectorStoreError(MultimodalRAGError):
    """Raised when vector store operations fail."""

    pass


class RetrievalError(MultimodalRAGError):
    """Raised when retrieval operations fail."""

    pass


class GenerationError(MultimodalRAGError):
    """Raised when answer generation fails."""

    pass


class ConfigurationError(MultimodalRAGError):
    """Raised when configuration is invalid."""

    pass
