"""Combined vector and graph retrieval service."""

from langchain_core.documents import Document

from graph_rag.config import Settings
from graph_rag.models.schemas import GraphQueryResult
from graph_rag.services.knowledge_graph import KnowledgeGraph
from graph_rag.services.vector_store import VectorStoreService


class GraphRetriever:
    """Retrieves context using both vector search and graph traversal."""

    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStoreService,
        knowledge_graph: KnowledgeGraph,
    ):
        self.settings = settings
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph

    def retrieve(self, query: str) -> tuple[list[Document], GraphQueryResult]:
        """Retrieve using vector search and graph query, returning combined results."""
        vector_docs = self.vector_store.similarity_search(query, k=self.settings.top_k)
        graph_results = self.knowledge_graph.query(query)

        enriched_docs = self._enrich_with_graph_context(vector_docs, graph_results)

        return enriched_docs, graph_results

    def _enrich_with_graph_context(
        self,
        docs: list[Document],
        graph_results: GraphQueryResult,
    ) -> list[Document]:
        """Enrich documents with related graph entity information."""
        entity_names = {e.name.lower() for e in graph_results.entities}

        for doc in docs:
            related_entities = []
            for entity in graph_results.entities:
                if entity.name.lower() in doc.page_content.lower():
                    related_entities.append(entity)

            doc.metadata["graph_entities"] = [
                {"name": e.name, "type": e.type} for e in related_entities
            ]

        return docs
