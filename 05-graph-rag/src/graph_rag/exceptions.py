"""Custom exceptions for Graph RAG."""


class GraphRAGError(Exception):
    """Base exception for Graph RAG."""


class DocumentLoadError(GraphRAGError):
    """Error loading documents."""


class VectorStoreError(GraphRAGError):
    """Error with vector store operations."""


class EntityExtractionError(GraphRAGError):
    """Error during entity or relationship extraction."""


class KnowledgeGraphError(GraphRAGError):
    """Error with knowledge graph operations."""


class PipelineError(GraphRAGError):
    """Error in the RAG pipeline."""
