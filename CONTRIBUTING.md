# Contributing to RAG Mastery

Thank you for your interest in contributing to RAG Mastery! This document provides guidelines and instructions for contributing.

## How to Contribute

### 1. Fork the Repository

```bash
git clone https://github.com/yourusername/rag-mastery.git
cd rag-mastery
```

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Make Your Changes

- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed

### 4. Run Tests

```bash
# Run all tests
make test-all

# Lint code
make lint

# Format code
make format
```

### 5. Submit a Pull Request

- Provide a clear description of changes
- Reference any related issues
- Ensure all tests pass

## Development Setup

### Prerequisites

- Python 3.11+
- uv (package manager)
- Docker
- Terraform
- Ollama

### Getting Started

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/yourusername/rag-mastery.git
cd rag-mastery

# Initialize infrastructure
make infra-init

# Start services
make infra-up PROJECT=01-naive-rag

# Install dependencies
make setup PROJECT=01-naive-rag
```

## Code Style

### Python

- Use type hints
- Follow PEP 8
- Use ruff for linting
- Maximum line length: 100 characters

### Documentation

- Use Markdown
- Include code examples
- Add Mermaid diagrams where appropriate

## Adding a New RAG Type

1. Create a new directory: `08-your-rag-type/`
2. Add the following files:
   - `pyproject.toml` - Project dependencies
   - `Makefile` - Build commands
   - `services.yaml` - Required services
   - `config.py` - Configuration
   - `main.py` - Implementation
   - `README.md` - Documentation
3. Update root `README.md` with the new project
4. Add tests in `tests/`

## Reporting Issues

- Use GitHub Issues
- Provide detailed reproduction steps
- Include error messages and logs

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
