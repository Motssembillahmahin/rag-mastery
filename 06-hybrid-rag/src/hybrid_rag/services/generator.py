"""Generator service for Hybrid RAG."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from hybrid_rag.config import HybridRAGConfig
from hybrid_rag.exceptions import GenerationError
from hybrid_rag.models.schemas import HybridQueryResult


class Generator:
    """Generates answers using retrieved context."""

    def __init__(self, config: HybridRAGConfig):
        """Initialize the generator.

        Args:
            config: Configuration settings.
        """
        self.config = config
        self._llm: ChatOllama | None = None
        self._prompt: ChatPromptTemplate | None = None

    @property
    def llm(self) -> ChatOllama:
        """Get or create LLM instance.

        Returns:
            ChatOllama instance.
        """
        if self._llm is None:
            self._llm = ChatOllama(
                model=self.config.llm_model,
                base_url=self.config.ollama_base_url,
            )
        return self._llm

    @property
    def prompt(self) -> ChatPromptTemplate:
        """Get or create prompt template.

        Returns:
            ChatPromptTemplate instance.
        """
        if self._prompt is None:
            self._prompt = ChatPromptTemplate.from_template(
                """Answer the question based on the following context.
The context was retrieved using hybrid search (keyword + semantic).

Context:
{context}

Question: {question}

Answer:"""
            )
        return self._prompt

    def generate(
        self,
        question: str,
        context: list[HybridQueryResult],
    ) -> str:
        """Generate an answer for the given question.

        Args:
            question: User question.
            context: List of retrieval results.

        Returns:
            Generated answer string.

        Raises:
            GenerationError: If generation fails.
        """
        try:
            context_text = "\n\n".join([r.content for r in context])
            chain = self.prompt | self.llm
            response = chain.invoke({"context": context_text, "question": question})
            return response.content
        except Exception as e:
            raise GenerationError(f"Failed to generate answer: {e}") from e
