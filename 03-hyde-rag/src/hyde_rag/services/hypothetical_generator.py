"""Hypothetical document generator service."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from hyde_rag.config import Settings
from hyde_rag.exceptions import HypotheticalGenerationError
from hyde_rag.models.schemas import HypotheticalDocument

HYPOTHESIS_TEMPLATE = """Please write a passage that answers the following question.
The passage should be detailed and informative, as if it were from a textbook or reference document.

Question: {question}

Passage:"""


class HypotheticalGenerator:
    """Generates hypothetical documents for HyDE retrieval."""

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
                    temperature=self.settings.hyde_temperature,
                )
            except Exception as e:
                raise HypotheticalGenerationError(f"Failed to initialize LLM: {e}") from e
        return self._llm

    def generate(self, query: str) -> HypotheticalDocument:
        """Generate a single hypothetical document from a query."""
        try:
            prompt = ChatPromptTemplate.from_template(HYPOTHESIS_TEMPLATE)
            chain = prompt | self.llm
            response = chain.invoke({"question": query})

            return HypotheticalDocument(
                content=response.content,
                confidence=0.5,
                query=query,
            )
        except Exception as e:
            raise HypotheticalGenerationError(
                f"Failed to generate hypothetical document: {e}"
            ) from e

    def generate_multiple(self, query: str, num_docs: int | None = None) -> list[HypotheticalDocument]:
        """Generate multiple hypothetical documents."""
        count = num_docs or self.settings.num_hypothetical_docs
        hypothetical_docs = []

        for i in range(count):
            try:
                doc = self.generate(query)
                hypothetical_docs.append(doc)
            except Exception as e:
                # Log warning but continue with other generations
                print(f"Warning: Failed to generate doc {i+1}: {e}")
                continue

        if not hypothetical_docs:
            raise HypotheticalGenerationError("Failed to generate any hypothetical documents")

        return hypothetical_docs
