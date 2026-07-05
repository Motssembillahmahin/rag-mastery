"""Knowledge graph service with Neo4j and NetworkX hybrid storage."""

import networkx as nx

from graph_rag.config import Settings
from graph_rag.exceptions import KnowledgeGraphError
from graph_rag.models.schemas import Entity, GraphQueryResult, Relationship


class KnowledgeGraph:
    """Knowledge graph with Neo4j backend and NetworkX fallback."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.driver = None
        self.graph = nx.DiGraph()

    def connect(self) -> None:
        """Connect to Neo4j database."""
        try:
            from neo4j import GraphDatabase

            self.driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_user, self.settings.neo4j_password),
            )
            # Verify connection
            self.driver.verify_connectivity()
        except Exception:
            self.driver = None

    def build_from_documents(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> None:
        """Build knowledge graph from extracted entities and relationships."""
        for entity in entities:
            self.graph.add_node(entity.name, type=entity.type)

        for rel in relationships:
            if rel.source in self.graph and rel.target in self.graph:
                self.graph.add_edge(rel.source, rel.target, relationship=rel.relationship)

        if self.driver:
            self._store_in_neo4j(entities, relationships)

    def _store_in_neo4j(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> None:
        """Store graph in Neo4j."""
        try:
            with self.driver.session() as session:
                for entity in entities:
                    session.run(
                        "MERGE (n:Entity {name: $name}) SET n.type = $type",
                        name=entity.name,
                        type=entity.type,
                    )

                for rel in relationships:
                    session.run(
                        """MATCH (a:Entity {name: $source})
                           MATCH (b:Entity {name: $target})
                           MERGE (a)-[:RELATED_TO {relationship: $rel}]->(b)""",
                        source=rel.source,
                        target=rel.target,
                        rel=rel.relationship,
                    )
        except Exception as e:
            raise KnowledgeGraphError(f"Failed to store in Neo4j: {e}")

    def query(self, query: str) -> GraphQueryResult:
        """Query the knowledge graph for matching entities and their relationships."""
        matched_entities = []
        matched_relationships = []

        for node in self.graph.nodes():
            if query.lower() in node.lower():
                node_type = self.graph.nodes[node].get("type", "UNKNOWN")
                matched_entities.append(Entity(name=node, type=node_type))

        for entity in matched_entities:
            if entity.name in self.graph:
                for neighbor in list(self.graph.neighbors(entity.name))[:5]:
                    edge_data = self.graph[entity.name][neighbor]
                    matched_relationships.append(
                        Relationship(
                            source=entity.name,
                            target=neighbor,
                            relationship=edge_data.get("relationship", "related"),
                        )
                    )

        return GraphQueryResult(
            entities=matched_entities,
            relationships=matched_relationships,
        )

    def get_entity_context(self, entity_name: str, max_depth: int = 2) -> list[str]:
        """Get contextual information about an entity via graph traversal."""
        if entity_name not in self.graph:
            return []

        contexts = []
        visited = set()

        def _traverse(node: str, depth: int, path: list[str]) -> None:
            if depth > max_depth or node in visited:
                return
            visited.add(node)

            for neighbor in self.graph.neighbors(node):
                edge_data = self.graph[node][neighbor]
                rel = edge_data.get("relationship", "related")
                contexts.append(f"{node} --[{rel}]--> {neighbor}")
                _traverse(neighbor, depth + 1, path + [neighbor])

        _traverse(entity_name, 0, [entity_name])
        return contexts

    @property
    def entity_count(self) -> int:
        """Get number of entities in the graph."""
        return self.graph.number_of_nodes()

    @property
    def relationship_count(self) -> int:
        """Get number of relationships in the graph."""
        return self.graph.number_of_edges()
