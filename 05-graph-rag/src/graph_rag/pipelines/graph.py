"""Graph RAG pipeline."""

from pathlib import Path

from graph_rag.config import Settings
from graph_rag.exceptions import PipelineError
from graph_rag.models.schemas import GraphQueryResult
from graph_rag.services.document_loader import DocumentLoader
from graph_rag.services.embeddings import EmbeddingService
from graph_rag.services.entity_extractor import EntityExtractor
from graph_rag.services.generator import Generator
from graph_rag.services.knowledge_graph import KnowledgeGraph
from graph_rag.services.retriever import GraphRetriever
from graph_rag.services.vector_store import VectorStoreService


class GraphPipeline:
    """Orchestrates the graph RAG workflow."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()

        # Initialize services
        self.document_loader = DocumentLoader(self.settings)
        self.embedding_service = EmbeddingService(self.settings)
        self.vector_store = VectorStoreService(self.settings, self.embedding_service)
        self.entity_extractor = EntityExtractor(self.settings)
        self.knowledge_graph = KnowledgeGraph(self.settings)
        self.generator = Generator(self.settings)
        self.retriever = GraphRetriever(
            self.settings, self.vector_store, self.knowledge_graph
        )

    def ingest_documents(self, directory: str | None = None) -> int:
        """Load, chunk, and ingest documents. Build knowledge graph."""
        data_dir = directory or str(self.settings.data_path)

        # Connect to Neo4j if available
        self.knowledge_graph.connect()

        # Load and split documents
        documents = self.document_loader.load_directory(data_dir)
        if not documents:
            raise PipelineError(f"No documents found in {data_dir}")

        chunks = self.document_loader.split_documents(documents)

        # Create vector store
        self.vector_store.create(chunks)

        # Extract entities and build knowledge graph
        all_entities = []
        all_relationships = []

        for doc in documents:
            entities = self.entity_extractor.extract_entities(doc.page_content)
            all_entities.extend(entities)

            relationships = self.entity_extractor.extract_relationships(
                doc.page_content, entities
            )
            all_relationships.extend(relationships)

        self.knowledge_graph.build_from_documents(all_entities, all_relationships)

        return len(chunks)

    def query(self, question: str) -> str:
        """Execute the graph RAG pipeline."""
        docs, graph_results = self.retriever.retrieve(question)

        if not docs:
            return "No relevant documents found."

        answer = self.generator.generate(question, docs, graph_results)

        return answer

    def get_retrieval_info(self, question: str) -> tuple[list, GraphQueryResult]:
        """Get detailed retrieval information."""
        return self.retriever.retrieve(question)
