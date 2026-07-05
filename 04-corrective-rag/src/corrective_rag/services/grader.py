"""Grading services for documents and answers."""

import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from corrective_rag.config import Settings
from corrective_rag.exceptions import GradingError
from corrective_rag.models.schemas import AnswerQuality, GradedDocument, GradingResult


class DocumentGrader:
    """Grades document relevance to queries."""

    GRADING_PROMPT = """You are a relevance grader. Score how relevant this document is to the query.
Return ONLY a JSON object with:
- "score": float between 0 and 1 (1 = perfectly relevant)
- "relevant": boolean (true if score >= 0.7)

Query: {query}
Document: {document}

Score:"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
        )

    def grade(self, query: str, document: GradedDocument) -> GradingResult:
        """Grade a single document's relevance."""
        prompt = ChatPromptTemplate.from_template(self.GRADING_PROMPT)
        chain = prompt | self.llm

        try:
            response = chain.invoke({"query": query, "document": document.content})
            result = self._parse_response(response.content)

            return GradingResult(
                relevant=result.get("relevant", False),
                score=result.get("score", 0.0),
                reason=f"Graded with score {result.get('score', 0.0):.2f}",
            )
        except Exception as e:
            raise GradingError(f"Failed to grade document: {e}")

    def grade_documents(self, query: str, documents: list[GradedDocument]) -> list[GradedDocument]:
        """Grade multiple documents and return with scores."""
        graded_docs = []

        for doc in documents:
            grading_result = self.grade(query, doc)
            graded_doc = GradedDocument(
                content=doc.content,
                metadata=doc.metadata,
                score=grading_result.score,
                relevant=grading_result.relevant,
            )
            graded_docs.append(graded_doc)

        return graded_docs

    def filter_relevant(self, graded_docs: list[GradedDocument]) -> list[GradedDocument]:
        """Filter documents by relevance threshold."""
        return [
            doc for doc in graded_docs
            if doc.score >= self.settings.relevance_threshold
        ]

    def _parse_response(self, text: str) -> dict:
        """Parse JSON response from LLM."""
        try:
            start = text.find("{")
            end = text.find("}") + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        return {"relevant": False, "score": 0.0}


class AnswerGrader:
    """Validates answer quality."""

    GRADING_PROMPT = """Evaluate this answer to the question based on the context.
Return ONLY a JSON object with:
- "satisfactory": boolean
- "reason": string explaining why

Question: {question}
Answer: {answer}
Context: {context}

Evaluation:"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
        )

    def grade(self, query: str, answer: str, context: list[GradedDocument]) -> AnswerQuality:
        """Grade answer quality."""
        prompt = ChatPromptTemplate.from_template(self.GRADING_PROMPT)
        chain = prompt | self.llm

        context_text = "\n\n".join([doc.content[:500] for doc in context])

        try:
            response = chain.invoke({
                "question": query,
                "answer": answer,
                "context": context_text,
            })
            result = self._parse_response(response.content)

            return AnswerQuality(
                satisfactory=result.get("satisfactory", True),
                reason=result.get("reason", ""),
            )
        except Exception as e:
            raise GradingError(f"Failed to grade answer: {e}")

    def _parse_response(self, text: str) -> dict:
        """Parse JSON response from LLM."""
        try:
            start = text.find("{")
            end = text.find("}") + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        return {"satisfactory": True, "reason": "Parsing failed"}
