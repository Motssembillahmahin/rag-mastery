.PHONY: help setup run test clean

help:
	@echo "Naive RAG - Available Commands:"
	@echo "  make setup   - Install dependencies"
	@echo "  make run     - Run the application"
	@echo "  make test    - Run tests"
	@echo "  make clean   - Clean cache files"

setup:
	uv sync

run:
	uv run python main.py

test:
	uv run pytest

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache