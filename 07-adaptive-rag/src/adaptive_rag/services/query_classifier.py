"""LLM-based query classification service."""

import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from adaptive_rag.config import QueryType, Settings
from adaptive_rag.exceptions import ClassificationError
from adaptive_rag.models.schemas import ClassificationResult

CLASSIFICATION_TEMPLATE = """Classify this query into one of these categories:
- SIMPLE: Simple factual question, definition, or basic info
- COMPLEX: Multi-part question requiring deep understanding
- FACTUAL: Specific fact-based question (who, what, when, where)
- ANALYTICAL: Question requiring analysis, comparison, or reasoning
- CREATIVE: Open-ended, creative, or hypothetical question

Return ONLY a JSON object with:
- "query_type": one of [SIMPLE, COMPLEX, FACTUAL, ANALYTICAL, CREATIVE]
- "confidence": float between 0 and 1

Query: {query}

Classification:"""


class QueryClassifier:
    """Classifies queries into different types for adaptive routing."""

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
                raise ClassificationError(f"Failed to initialize LLM: {e}") from e
        return self._llm

    def classify(self, query: str) -> ClassificationResult:
        """Classify a query and return type with confidence score."""
        try:
            prompt = ChatPromptTemplate.from_template(CLASSIFICATION_TEMPLATE)
            chain = prompt | self.llm
            response = chain.invoke({"query": query})

            text = response.content
            start = text.find("{")
            end = text.find("}") + 1

            if start != -1 and end > 0:
                result = json.loads(text[start:end])
                query_type = QueryType(result["query_type"].lower())
                confidence = float(result.get("confidence", 0.8))
                return ClassificationResult(query_type=query_type, confidence=confidence)

            return ClassificationResult(query_type=QueryType.SIMPLE, confidence=0.5)
        except (json.JSONDecodeError, KeyError, ValueError):
            return ClassificationResult(query_type=QueryType.SIMPLE, confidence=0.5)
        except ClassificationError:
            raise
        except Exception as e:
            raise ClassificationError(f"Classification failed: {e}") from e
