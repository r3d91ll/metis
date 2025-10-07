# Contributing to Metis

Thank you for your interest in contributing to Metis! This document provides guidelines for contributing to the Metis semantic knowledge infrastructure.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Go Proxy Development](#go-proxy-development)
- [Documentation](#documentation)

---

## Getting Started

### Prerequisites

- **Python**: 3.11 or higher
- **Poetry**: For Python dependency management
- **Go**: 1.19+ (if working on database proxies)
- **ArangoDB**: 3.11+ (for database integration)
- **CUDA**: 11.8+ (optional, for GPU-accelerated embeddings)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/metis-ai/metis.git
cd metis

# Install dependencies with development tools
poetry install --with dev --all-extras

# Activate the virtual environment
poetry shell

# Run tests to verify setup
pytest
```

---

## Development Setup

### Python Environment

Metis uses Poetry for dependency management:

```bash
# Install all dependencies including optional ones
poetry install --all-extras

# Install only core dependencies
poetry install

# Add a new dependency
poetry add package-name

# Add a development dependency
poetry add --group dev package-name
```

### GPU Support

For GPU-accelerated embedding generation:

```bash
# Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# The embedders will automatically use GPU if available
```

### ArangoDB Setup

Metis requires ArangoDB for graph storage:

```bash
# Using Unix sockets (recommended for performance)
# Ensure ArangoDB is running and accessible at:
# /run/arangodb3/arangod.sock (or configured path)

# Or use TCP connection as fallback
# Default: localhost:8529
```

### Go Proxy Build (Optional)

If working on the database security proxies:

```bash
cd metis/database/proxies

# Build both proxies
make build

# Install to system (requires sudo)
sudo make install

# Run tests
make test
```

---

## Project Structure

```text
metis/
├── metis/                    # Main package
│   ├── embedders/           # Text embedding models
│   │   ├── base.py         # Base embedder interface
│   │   ├── jina_v4.py      # Jina v4 embedder (32k context)
│   │   ├── sentence.py     # Sentence Transformers
│   │   └── factory.py      # Embedder factory pattern
│   │
│   ├── extractors/          # Document extraction
│   │   ├── base.py         # Base extractor interface
│   │   ├── docling.py      # PDF extraction with Docling
│   │   ├── latex.py        # LaTeX document processing
│   │   ├── code.py         # Code file extraction
│   │   └── factory.py      # Extractor factory pattern
│   │
│   ├── database/            # ArangoDB integration
│   │   ├── client.py       # HTTP/2 optimized client
│   │   ├── memory.py       # High-level memory operations
│   │   ├── admin.py        # Admin operations
│   │   └── proxies/        # Go security layer (RO/RW)
│   │
│   ├── gnn/                 # Graph Neural Networks
│   │   ├── graphsage_model.py   # Multi-relational GraphSAGE
│   │   ├── trainer.py           # Training pipeline
│   │   ├── inference.py         # Fast retrieval
│   │   └── graph_builder.py     # ArangoDB to PyG conversion
│   │
│   ├── config/              # Configuration management
│   │   └── loader.py       # YAML config loading
│   │
│   ├── utils/               # Utilities
│   │   └── hashing.py      # Content hashing
│   │
│   └── chunking/            # Late chunking support
│       └── __init__.py
│
├── examples/                # Usage examples
│   ├── basic_setup.py
│   ├── paper_ingestion.py
│   └── custom_extractors.py
│
├── tests/                   # Test suite
│   └── ...
│
├── docs/                    # Documentation
│   └── theory/             # Theoretical foundations
│
└── metis/database/proxies/ # Go proxy source
    ├── ro_proxy.go         # Read-only proxy
    ├── rw_proxy.go         # Read-write proxy
    └── Makefile
```

### Module Overview

| Module | Purpose | Key Components |
|--------|---------|----------------|
| `embedders` | Generate text embeddings | JinaV4, SentenceTransformers, Factory |
| `extractors` | Extract text from documents | Docling (PDF), LaTeX, Code parsers |
| `database` | ArangoDB integration | HTTP/2 client, Unix socket proxies |
| `gnn` | Graph neural networks | GraphSAGE, Training, Inference |
| `config` | Configuration management | YAML loader, Config validation |

---

## Development Workflow

### Creating a New Feature

1. **Create a feature branch:**

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines below

3. **Add tests** for new functionality (see [Testing](#testing))

4. **Run the test suite:**

   ```bash
   pytest
   ```

5. **Format and lint your code:**

   ```bash
   ruff format metis tests
   ruff check metis tests
   ```

6. **Commit your changes:**

   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

### Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

**Examples:**

```text
feat: add support for Claude embeddings
fix: handle PDF extraction timeout errors
docs: update embedder configuration examples
refactor: simplify extractor factory pattern
```

---

## Code Style Guidelines

### Python Code Style

Metis uses **Ruff** for formatting and linting:

```bash
# Format code
ruff format metis tests

# Check for issues
ruff check metis tests

# Auto-fix issues where possible
ruff check --fix metis tests
```

### Key Style Principles

1. **Follow existing patterns** in the codebase
2. **Use type hints** for all function signatures:

   ```python
   def process_document(text: str, max_length: int = 1000) -> ProcessedDocument:
       ...
   ```

3. **Use dataclasses** for structured data:

   ```python
   from dataclasses import dataclass
   
   @dataclass
   class DocumentMetadata:
       title: str
       authors: List[str]
       word_count: int
   ```

4. **Factory pattern** for component creation:

   ```python
   # Good
   embedder = create_embedder("jinaai/jina-embeddings-v4")
   
   # Avoid direct instantiation in application code
   embedder = JinaV4Embedder(...)
   ```

5. **Error handling** with specific exceptions:

   ```python
   from metis.extractors.base import ExtractionError
   
   if not result.success:
       raise ExtractionError(f"Failed to extract {file_path}: {result.error}")
   ```

6. **Docstrings** for all public functions:

   ```python
   def extract_document(path: Path) -> ExtractionResult:
       """
       Extract text content from document.
       
       Args:
           path: Path to document file
           
       Returns:
           ExtractionResult with text and metadata
           
       Raises:
           ExtractionError: If extraction fails
       """
   ```

### Go Code Style

For database proxy development:

```bash
# Format code
cd metis/database/proxies
go fmt ./...

# Run linter
golangci-lint run
```

Follow standard Go conventions:

- Use `gofmt` formatting
- Keep functions focused and small
- Add comments for exported functions
- Handle all errors explicitly

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=metis --cov-report=html

# Run specific test file
pytest tests/test_embedders.py

# Run specific test
pytest tests/test_embedders.py::test_jina_v4_embedding
```

### Writing Tests

Place tests in the `tests/` directory matching the source structure:

```text
metis/embedders/jina_v4.py  →  tests/test_embedders.py
```

**Example test:**

```python
import pytest
from metis import create_embedder

def test_embedder_creation():
    """Test embedder factory creates correct instance."""
    embedder = create_embedder("jinaai/jina-embeddings-v4")
    assert embedder.embedding_dimension == 2048

def test_embedding_generation():
    """Test embedding generation produces correct dimensions."""
    embedder = create_embedder("jinaai/jina-embeddings-v4")
    embeddings = embedder.embed_texts(["test text"])
    assert embeddings.shape == (1, 2048)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_gpu_acceleration():
    """Test GPU acceleration works."""
    embedder = create_embedder("jinaai/jina-embeddings-v4", device="cuda")
    # Test GPU usage
```

### Testing Best Practices

- **Write focused tests**: One concept per test
- **Use descriptive names**: `test_jina_v4_handles_empty_input`
- **Test edge cases**: Empty inputs, large inputs, malformed data
- **Mock external services**: Don't hit real APIs in tests
- **Clean up resources**: Use fixtures for setup/teardown

---

## Submitting Changes

### Pull Request Process

1. **Ensure all tests pass:**

   ```bash
   pytest
   ruff check metis tests
   ```

2. **Update documentation** if you changed APIs or added features

3. **Create a Pull Request** with:
   - Clear title following commit conventions
   - Description of changes
   - Link to related issues
   - Screenshots (if UI changes)

4. **Address review feedback** promptly

5. **Squash commits** if requested before merging

### Pull Request Checklist

- [ ] Code follows style guidelines
- [ ] Tests added for new functionality
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Commit messages follow conventions
- [ ] No breaking changes (or clearly documented)

### Review Process

- PRs require at least one approval
- Automated checks must pass (tests, linting)
- Maintainers may request changes or clarifications

---

## Go Proxy Development

The database proxies provide security boundaries for ArangoDB access.

### Building Proxies

```bash
cd metis/database/proxies

# Build both proxies
make build

# Build individual proxy
make build-roproxy
make build-rwproxy

# Clean build artifacts
make clean
```

### Testing Proxies

```bash
# Run Go tests
make test

# Test proxy deployment (requires sudo)
sudo make install
systemctl status metis-roproxy.service
systemctl status metis-rwproxy.service
```

### Proxy Architecture

- **Read-Only Proxy**: Blocks mutations, allows queries
- **Read-Write Proxy**: Allows DB-scoped operations, blocks admin endpoints

See `metis/database/proxies/README.md` for detailed proxy documentation.

---

## Documentation

### Updating Documentation

- Keep `README.md` updated for user-facing changes
- Update docstrings for API changes
- Add examples to `examples/` for new features
- Update `CHANGELOG.md` with notable changes

### Documentation Style

- Use clear, concise language
- Provide code examples for complex features
- Include expected output where helpful
- Link to related documentation

---

## Getting Help

- **Issues**: Search [existing issues](https://github.com/metis-ai/metis/issues) or create a new one
- **Discussions**: Use GitHub Discussions for questions
- **Code Review**: Tag maintainers for review assistance

---

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you agree to uphold this code. Please report unacceptable behavior to the maintainers.

---

## License

By contributing to Metis, you agree that your contributions will be licensed under the Apache License 2.0.

---

Thank you for contributing to Metis! 🚀
