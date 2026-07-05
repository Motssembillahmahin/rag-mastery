"""LLM-based entity and relationship extraction service."""

import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from graph_rag.config import Settings
from graph_rag.exceptions import EntityExtractionError
from graph_rag.models.schemas import Entity, Relationship


class EntityExtractor:
    """Extracts entities and relationships from text using LLM."""

    ENTITY_PROMPT = """Extract all entities (people, organizations, concepts, locations) from the text.
Return as a JSON list with format:
[{{"name": "Entity Name", "type": "PERSON|ORG|CONCEPT|LOCATION"}}]

Text: {text}

Entities:"""

    RELATIONSHIP_PROMPT = """Extract relationships between the following entities from the text.
Return as a JSON list with format:
[{{"source": "Entity1", "target": "Entity2", "relationship": "relates to"}}]

Entities: {entities}

Text: {text}

Relationships:"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = ChatOllama(
            model=settings.entity_extraction_model,
            base_url=settings.ollama_base_url,
        )

    def extract_entities(self, text: str) -> list[Entity]:
        """Extract entities from text."""
        prompt = ChatPromptTemplate.from_template(self.ENTITY_PROMPT)
        chain = prompt | self.llm

        try:
            response = chain.invoke({"text": text})
            raw_entities = self._parse_json_list(response.content)

            return [
                Entity(name=e.get("name", ""), type=e.get("type", "UNKNOWN"))
                for e in raw_entities
                if e.get("name")
            ]
        except Exception as e:
            raise EntityExtractionError(f"Failed to extract entities: {e}")

    def extract_relationships(self, text: str, entities: list[Entity]) -> list[Relationship]:
        """Extract relationships between entities."""
        if not entities:
            return []

        entity_names = [e.name for e in entities]
        entity_str = ", ".join(entity_names)

        prompt = ChatPromptTemplate.from_template(self.RELATIONSHIP_PROMPT)
        chain = prompt | self.llm

        try:
            response = chain.invoke({"entities": entity_str, "text": text})
            raw_rels = self._parse_json_list(response.content)

            return [
                Relationship(
                    source=r.get("source", ""),
                    target=r.get("target", ""),
                    relationship=r.get("relationship", "related"),
                )
                for r in raw_rels
                if r.get("source") and r.get("target")
            ]
        except Exception as e:
            raise EntityExtractionError(f"Failed to extract relationships: {e}")

    def _parse_json_list(self, text: str) -> list[dict]:
        """Parse JSON list from LLM response."""
        try:
            start = text.find("[")
            end = text.rfind("]") + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        return []
