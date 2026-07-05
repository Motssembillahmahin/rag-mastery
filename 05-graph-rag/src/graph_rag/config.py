"""Configuration for Graph RAG."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # LLM settings
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3"
    embedding_model: str = "all-minilm:latest"

    # ChromaDB settings
    collection_name: str = "graph_rag"
    persist_dir: str = "./chroma_db"

    # Neo4j settings
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "rag-mastery-password"

    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Graph settings
    max_entities: int = 50
    max_relations: int = 100
    entity_extraction_model: str = "llama3"
    graph_traversal_depth: int = 2

    # Retrieval settings
    top_k: int = 4

    # Paths
    data_dir: str = "data"

    @property
    def data_path(self) -> Path:
        """Get data directory path."""
        return Path(self.data_dir)

    @property
    def persist_path(self) -> Path:
        """Get persist directory path."""
        return Path(self.persist_dir)

    model_config = {"env_prefix": "GRAPH_RAG_", "env_file": ".env"}


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()
