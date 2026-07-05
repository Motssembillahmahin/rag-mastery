"""Adaptive answer generator service with type-specific prompts."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from adaptive_rag.config import QueryType, Settings
from adaptive_rag.exceptions import GenerationError
from adaptive_rag.models.schemas import DocumentChunk

PROMPTS = {
    QueryType.SIMPLE: """Answer this simple question concisely:
Context: {context}
Question: {question}
Answer:""",
    QueryType.COMPLEX: """Provide a comprehensive, detailed answer to this complex question:
Context: {context}
Question: {question}
Answer:""",
    QueryType.FACTUAL: """Extract and list the specific facts that answer this question:
Context: {context}
Question: {question}
Facts:""",
    QueryType.ANALYTICAL: """Provide an analytical answer with reasoning and comparison:
Context: {context}
Question: {question}
Analysis:""",
    QueryType.CREATIVE: """Provide a creative, thoughtful response:
Context: {context}
Question: {question}
Response:""",
}


class AdaptiveGenerator:
    """Generates answers using type-specific prompts."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._llm = None

    @property
    def llm(self) -> ChatOllama:
        """Get or create LLM instance."""
        if self._llm is None:
            try:
                self._llm = ChatOllama(
                    model=self.settings.llm_model,
                    base_url=self.settings.ollama_base_url,
                )
            except Exception as e:
                raise GenerationError(f"Failed to initialize LLM: {e}") from e
        return self._llm

    def generate_answer(
        self, query: str, context: list[DocumentChunk], query_type: QueryType
    ) -> str:
        """Generate an answer tailored to the query type."""
        if not context:
            return "No relevant context found to answer the question."

        context_text = "\n\n".join([doc.content for doc in context])
        prompt_template = PROMPTS.get(query_type, PROMPTS[QueryType.SIMPLE])

        try:
            prompt = ChatPromptTemplate.from_template(prompt_template)
            chain = prompt | self.llm
            response = chain.invoke({"context": context_text, "question": query})
            return response.content
        except Exception as e:
            raise GenerationError(f"Failed to generate answer: {e}") from e
