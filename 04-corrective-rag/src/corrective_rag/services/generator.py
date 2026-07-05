"""Answer generation service."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from corrective_rag.config import Settings
from corrective_rag.models.schemas import GradedDocument


class Generator:
    """Generates answers using LLM."""

    ANSWER_PROMPT = """Answer the question based on the following context.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""

    REGENERATION_PROMPT = """Based on the following context, provide a comprehensive answer.
Make sure to directly address the question using only the information provided.

Context:
{context}

Question: {question}

Detailed Answer:"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
        )

    def generate(self, query: str, context: list[GradedDocument]) -> str:
        """Generate answer from context."""
        prompt = ChatPromptTemplate.from_template(self.ANSWER_PROMPT)
        chain = prompt | self.llm

        context_text = "\n\n".join([doc.content for doc in context])
        response = chain.invoke({"context": context_text, "question": query})

        return response.content

    def regenerate(self, query: str, context: list[GradedDocument]) -> str:
        """Regenerate answer with emphasis on quality."""
        prompt = ChatPromptTemplate.from_template(self.REGENERATION_PROMPT)
        chain = prompt | self.llm

        context_text = "\n\n".join([doc.content for doc in context])
        response = chain.invoke({"context": context_text, "question": query})

        return response.content
