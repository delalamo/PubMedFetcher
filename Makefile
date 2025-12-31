.PHONY: help install install-dev lint format test clean pre-commit

# Default target
help:
	@echo "Available commands:"
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo "  make lint         - Run linters (ruff)"
	@echo "  make format       - Format code (ruff format)"
	@echo "  make test         - Run tests with pytest"
	@echo "  make pre-commit   - Run pre-commit hooks on all files"
	@echo "  make clean        - Remove build artifacts and cache files"

# Install production dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
install-dev:
	pip install -r requirements.txt
	pip install pre-commit ruff mypy pytest
	pre-commit install

# Run linters
lint:
	ruff check .
	ruff format --check .

# Format code
format:
	ruff check --fix .
	ruff format .

# Run tests
test:
	pytest

# Run pre-commit on all files
pre-commit:
	pre-commit run --all-files

# Clean up build artifacts and caches
clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache
	rm -rf build dist *.egg-info
	rm -rf .coverage htmlcov coverage.xml
	rm -f *.jsonl
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
