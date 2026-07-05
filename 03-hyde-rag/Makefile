.PHONY: help setup run ingest query interactive test lint clean

help:
	@echo "HyDE RAG - Available Commands:"
	@echo "  make setup       - Install dependencies"
	@echo "  make run         - Run interactive mode"
	@echo "  make ingest      - Ingest documents (usage: make ingest SOURCE=data/)"
	@echo "  make query       - Query the system (usage: make query QUESTION='your question')"
	@echo "  make interactive - Start interactive mode"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Run linter"
	@echo "  make clean       - Clean cache files"

setup:
	uv sync

run: interactive

ingest:
	uv run python -m hyde_rag ingest $(SOURCE)

query:
	uv run python -m hyde_rag query "$(QUESTION)"

interactive:
	uv run python -m hyde_rag interactive

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache chroma_db test_chroma_db
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
