"""Answer generator service."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from hyde_rag.config import Settings
from hyde_rag.exceptions import HyDEError
from hyde_rag.models.schemas import DocumentChunk

ANSWER_TEMPLATE = """Answer the question based on the following context.
The context was retrieved using HyDE (Hypothetical Document Embeddings).

Context:
{context}

Question: {question}

Answer:"""


class Generator:
    """Generates answers using retrieved context."""

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
                raise HyDEError(f"Failed to initialize LLM: {e}") from e
        return self._llm

    def generate_answer(self, query: str, context: list[DocumentChunk]) -> str:
        """Generate an answer from query and context."""
        if not context:
            return "No relevant context found to answer the question."

        context_text = "\n\n".join([doc.content for doc in context])

        try:
            prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)
            chain = prompt | self.llm
            response = chain.invoke({"context": context_text, "question": query})
            return response.content
        except Exception as e:
            raise HyDEError(f"Failed to generate answer: {e}") from e
