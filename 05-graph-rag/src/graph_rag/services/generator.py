"""Answer generation service."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from graph_rag.config import Settings
from graph_rag.models.schemas import Entity, GraphQueryResult, Relationship


class Generator:
    """Generates answers using LLM with vector and graph context."""

    ANSWER_PROMPT = """Answer the question based on the following context and knowledge graph.

Vector Search Results:
{vector_context}

Knowledge Graph Information:
{graph_context}

Question: {question}

Answer:"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
        )

    def generate(
        self,
        query: str,
        context: list,
        graph_results: GraphQueryResult,
    ) -> str:
        """Generate answer from vector context and graph results."""
        prompt = ChatPromptTemplate.from_template(self.ANSWER_PROMPT)
        chain = prompt | self.llm

        vector_context = "\n\n".join([doc.page_content for doc in context])
        graph_info = self._format_graph_context(graph_results)

        response = chain.invoke({
            "vector_context": vector_context,
            "graph_context": graph_info,
            "question": query,
        })

        return response.content

    def _format_graph_context(self, graph_results: GraphQueryResult) -> str:
        """Format graph results into readable context string."""
        parts = []

        if graph_results.entities:
            parts.append("Entities:")
            for entity in graph_results.entities[:10]:
                parts.append(f"- {entity.name} ({entity.type})")

        if graph_results.relationships:
            parts.append("\nRelationships:")
            for rel in graph_results.relationships[:10]:
                parts.append(f"- {rel.source} -> {rel.target} ({rel.relationship})")

        return "\n".join(parts) if parts else "No graph information available."
