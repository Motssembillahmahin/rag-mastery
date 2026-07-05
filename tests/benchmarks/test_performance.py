"""Performance benchmarks for RAG implementations."""

import pytest
import time


class TestPerformance:
    """Performance benchmark tests."""

    def test_chunking_speed(self):
        """Test document chunking performance."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        text = "This is a test document. " * 1000

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        start = time.time()
        chunks = splitter.split_text(text)
        duration = time.time() - start

        assert len(chunks) > 0
        assert duration < 1.0  # Should complete in under 1 second

    def test_bm25_indexing_speed(self):
        """Test BM25 indexing performance."""
        from rank_bm25 import BM25Okapi

        documents = [f"Document {i} with some test content" for i in range(1000)]
        tokenized = [doc.lower().split() for doc in documents]

        start = time.time()
        bm25 = BM25Okapi(tokenized)
        duration = time.time() - start

        assert bm25 is not None
        assert duration < 1.0  # Should complete in under 1 second
