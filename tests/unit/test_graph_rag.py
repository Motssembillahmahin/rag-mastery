"""Unit tests for Graph RAG."""

import pytest


class TestGraphConfig:
    """Test Graph RAG configuration."""

    def test_default_config(self):
        """Test default configuration."""
        from rag_mastery.config import GraphRAGConfig

        config = GraphRAGConfig()
        assert config.neo4j_uri == "bolt://localhost:7687"
        assert config.neo4j_user == "neo4j"
        assert config.max_entities == 50
        assert config.graph_traversal_depth == 2
