"""Adaptive retriever service with type-specific strategies."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from adaptive_rag.config import QueryType, Settings
from adaptive_rag.exceptions import RetrievalError
from adaptive_rag.models.schemas import DocumentChunk, RetrievalResult
from adaptive_rag.services.vector_store import VectorStore

HYDE_TEMPLATE = """Write a detailed passage that would answer this question.
Question: {query}

Passage:"""


class AdaptiveRetriever:
    """Retrieves documents using type-specific strategies."""

    def __init__(self, settings: Settings, vector_store: VectorStore):
        self.settings = settings
        self.vector_store = vector_store
        self._llm = None

    @property
    def llm(self) -> ChatOllama:
        """Get or create LLM instance for HyDE."""
        if self._llm is None:
            try:
                self._llm = ChatOllama(
                    model=self.settings.llm_model,
                    base_url=self.settings.ollama_base_url,
                )
            except Exception as e:
                raise RetrievalError(f"Failed to initialize LLM: {e}") from e
        return self._llm

    def generate_hypothetical_document(self, query: str) -> str:
        """Generate a hypothetical document for HyDE retrieval."""
        try:
            prompt = ChatPromptTemplate.from_template(HYDE_TEMPLATE)
            chain = prompt | self.llm
            response = chain.invoke({"query": query})
            return response.content
        except Exception as e:
            raise RetrievalError(f"Failed to generate hypothetical document: {e}") from e

    def retrieve(self, query: str, query_type: QueryType) -> RetrievalResult:
        """Retrieve documents based on query type configuration."""
        config = self.settings.retrieval_config.get(
            query_type.value, {"top_k": 4, "use_hyde": False}
        )

        top_k = config.get("top_k", 4)
        use_hyde = config.get("use_hyde", False)

        try:
            if use_hyde:
                search_query = self.generate_hypothetical_document(query)
            else:
                search_query = query

            results = self.vector_store.similarity_search(search_query, k=top_k)

            chunks = [
                DocumentChunk(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    chunk_index=doc.metadata.get("chunk_index", 0),
                    metadata=doc.metadata,
                )
                for doc, _ in results
            ]
            scores = [score for _, score in results]

            return RetrievalResult(chunks=chunks, scores=scores, used_hyde=use_hyde)
        except RetrievalError:
            raise
        except Exception as e:
            raise RetrievalError(f"Retrieval failed: {e}") from e
