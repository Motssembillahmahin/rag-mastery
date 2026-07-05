"""Integration tests for the hybrid RAG pipeline."""

import pytest

from hybrid_rag.config import HybridRAGConfig
from hybrid_rag.pipelines.hybrid import HybridPipeline
from hybrid_rag.models.schemas import QueryRequest


@pytest.fixture
def pipeline():
    """Create a test pipeline."""
    config = HybridRAGConfig(
        data_dir="test_data",
        persist_dir="./test_chroma_db",
    )
    return HybridPipeline(config)


class TestHybridPipeline:
    """Tests for HybridPipeline."""

    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.config is not None
        assert pipeline.document_loader is not None
        assert pipeline.embedding_service is not None
        assert pipeline.vector_store is not None
        assert pipeline.keyword_search is not None
        assert pipeline.fusion_service is not None
        assert pipeline.retriever is not None
        assert pipeline.generator is not None

    def test_ingest_empty_directory(self, pipeline):
        """Test ingestion with empty directory."""
        stats = pipeline.ingest_documents()
        assert stats["documents"] == 0
        assert stats["chunks"] == 0

    def test_get_stats_empty(self, pipeline):
        """Test stats with empty stores."""
        stats = pipeline.get_stats()
        assert stats["vector_count"] == 0
        assert stats["bm25_count"] == 0

    def test_query_empty_pipeline(self, pipeline):
        """Test querying empty pipeline."""
        request = QueryRequest(question="Test question")
        response = pipeline.query(request)

        assert response.answer == "No relevant content found."
        assert len(response.sources) == 0
        assert response.num_results == 0
