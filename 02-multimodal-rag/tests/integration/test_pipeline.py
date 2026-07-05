"""Integration tests for the multimodal RAG pipeline."""

import pytest
from pathlib import Path

from multimodal_rag.config import MultimodalRAGConfig
from multimodal_rag.pipelines.multimodal import MultimodalRAGPipeline
from multimodal_rag.models.schemas import QueryRequest


@pytest.fixture
def pipeline():
    """Create a test pipeline."""
    config = MultimodalRAGConfig(
        data_dir="test_data",
        documents_dir="test_data/documents",
        images_dir="test_data/images",
        persist_dir="./test_chroma_db",
    )
    return MultimodalRAGPipeline(config)


class TestMultimodalRAGPipeline:
    """Tests for MultimodalRAGPipeline."""

    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.config is not None
        assert pipeline.document_loader is not None
        assert pipeline.image_processor is not None
        assert pipeline.embedding_service is not None
        assert pipeline.vector_store is not None
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
        assert stats["text_count"] == 0
        assert stats["image_count"] == 0

    def test_query_empty_pipeline(self, pipeline):
        """Test querying empty pipeline."""
        request = QueryRequest(question="Test question")
        response = pipeline.query(request)

        assert response.answer == "No relevant content found."
        assert len(response.sources) == 0
        assert response.num_text_chunks == 0
        assert response.num_images == 0
