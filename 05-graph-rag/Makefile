.PHONY: help setup run test lint format clean typecheck

help:
	@echo "Graph RAG - Available Commands:"
	@echo "  make setup     - Install dependencies"
	@echo "  make run       - Run the application"
	@echo "  make test      - Run tests"
	@echo "  make lint      - Run linter"
	@echo "  make format    - Format code"
	@echo "  make typecheck - Run type checking"
	@echo "  make clean     - Clean cache files"

setup:
	uv sync

run:
	uv run graph-rag

test:
	uv run pytest

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

typecheck:
	uv run mypy src/

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache
	rm -rf src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
