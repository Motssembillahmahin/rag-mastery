.PHONY: help setup run test lint typecheck clean infra-up infra-down

help:
	@echo "Hybrid RAG - Available Commands:"
	@echo "  make setup      - Install dependencies"
	@echo "  make run        - Run the application"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linter"
	@echo "  make typecheck  - Run type checker"
	@echo "  make clean      - Clean cache files"
	@echo "  make infra-up   - Start infrastructure services"
	@echo "  make infra-down - Stop infrastructure services"

setup:
	uv sync

run:
	uv run hybrid-rag

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/

typecheck:
	uv run mypy src/

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache
	rm -rf src/*.egg-info dist build
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

infra-up:
	docker compose -f ../../docker-compose.yaml up -d chromadb ollama

infra-down:
	docker compose -f ../../docker-compose.yaml down
