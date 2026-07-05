"""Integration tests for the pipeline."""

from graph_rag.models.schemas import Entity, GraphQueryResult, RAGResponse, Relationship


def test_entity_creation():
    """Test Entity model creation."""
    entity = Entity(name="Python", type="CONCEPT")

    assert entity.name == "Python"
    assert entity.type == "CONCEPT"


def test_relationship_creation():
    """Test Relationship model creation."""
    rel = Relationship(source="Python", target="Programming", relationship="is_a")

    assert rel.source == "Python"
    assert rel.target == "Programming"
    assert rel.relationship == "is_a"


def test_graph_query_result_creation():
    """Test GraphQueryResult model creation."""
    entities = [Entity(name="A", type="PERSON"), Entity(name="B", type="ORG")]
    relationships = [Relationship(source="A", target="B", relationship="works_at")]

    result = GraphQueryResult(entities=entities, relationships=relationships)

    assert len(result.entities) == 2
    assert len(result.relationships) == 1


def test_rag_response_creation():
    """Test RAGResponse model creation."""
    response = RAGResponse(
        answer="Test answer",
        documents=["doc1", "doc2"],
        graph_entities=[Entity(name="X", type="CONCEPT")],
        graph_relationships=[],
    )

    assert response.answer == "Test answer"
    assert len(response.documents) == 2
    assert len(response.graph_entities) == 1
