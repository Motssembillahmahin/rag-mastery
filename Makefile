.PHONY: help setup run test clean lint format infra-init infra-plan infra-up infra-down infra-destroy infra-status test-all

PROJECT ?= 01-naive-rag

help:
	@echo "RAG Mastery - Unified Command Center"
	@echo ""
	@echo "Infrastructure:"
	@echo "  make infra-init                    - Initialize Terraform"
	@echo "  make infra-plan                    - Show execution plan"
	@echo "  make infra-up PROJECT=<project>    - Start required services"
	@echo "  make infra-down                    - Stop all services"
	@echo "  make infra-destroy                 - Destroy all infrastructure"
	@echo "  make infra-status                  - Show running services"
	@echo ""
	@echo "Development:"
	@echo "  make setup PROJECT=<project>       - Install project dependencies"
	@echo "  make run PROJECT=<project>         - Run project"
	@echo "  make test PROJECT=<project>        - Test project"
	@echo "  make test-all                      - Test all projects"
	@echo ""
	@echo "Utilities:"
	@echo "  make lint                          - Lint all code"
	@echo "  make format                        - Format all code"
	@echo "  make clean                         - Clean cache files"
	@echo ""
	@echo "Examples:"
	@echo "  make infra-up PROJECT=05-graph-rag   # Starts ChromaDB + Ollama + Neo4j"
	@echo "  make infra-up PROJECT=01-naive-rag   # Starts ChromaDB + Ollama only"

infra-init:
	@cd terraform/environments/dev && terraform init

infra-plan:
	@cd terraform/environments/dev && terraform plan

infra-up:
	@echo "Checking services for $(PROJECT)..."
	@./scripts/infra/start.sh $(PROJECT)

infra-down:
	@./scripts/infra/stop.sh
	@echo "All services stopped"

infra-destroy:
	@./scripts/infra/destroy.sh
	@echo "Infrastructure destroyed"

infra-status:
	@./scripts/infra/status.sh

setup:
	@cd $(PROJECT) && uv sync
	@echo "Dependencies installed for $(PROJECT)"

run:
	@./scripts/project/run.sh $(PROJECT)

test:
	@cd $(PROJECT) && uv run pytest
	@echo "Tests passed for $(PROJECT)"

test-all:
	@for dir in 0*/; do \
		echo "Testing $$dir"; \
		cd $$dir && uv run pytest && cd ..; \
	done

lint:
	@for dir in 0*/; do \
		cd $$dir && uv run ruff check . && cd ..; \
	done

format:
	@for dir in 0*/; do \
		cd $$dir && uv run ruff format . && cd ..; \
	done

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".terraform" -exec rm -rf {} + 2>/dev/null || true
