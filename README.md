# Metis - Semantic Knowledge Infrastructure

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Metis** is a Python library for building semantic graph databases with ArangoDB and embedding models. Named after the Titaness of wisdom and transformation, Metis helps you transform raw documents into structured, searchable knowledge graphs.

## Features

- **Multi-Format Document Extraction**: Extract text and structure from PDFs, LaTeX sources, and code files
- **State-of-the-Art Embeddings**: Built-in support for Jina v4 embeddings with 32k context window
- **High-Performance Database**: ArangoDB client with Unix socket support for low-latency operations
- **Late Chunking**: Intelligent document chunking that preserves semantic context
- **Extensible Architecture**: Factory patterns for easy integration of new models and extractors

## Installation

```bash
pip install metis
```

For PDF extraction support:
```bash
pip install metis[pdf]
```

For all optional features:
```bash
pip install metis[all]
```

## Quick Start

### Basic Usage

```python
from metis import create_embedder, create_extractor_for_file

# Create embedder
embedder = create_embedder("jinaai/jina-embeddings-v4", device="cuda:0")

# Extract document
extractor = create_extractor_for_file("paper.pdf")
result = extractor.extract("paper.pdf")

# Generate embeddings
embeddings = embedder.embed_texts([result.text])
```

### Database Integration

```python
from metis.database import ArangoMemoryClient, resolve_memory_config

# Configure database connection
config = resolve_memory_config(
    database="my_knowledge_base",
    socket_path="/var/run/arangodb/socket"
)

# Create client
with ArangoMemoryClient(config) as client:
    # Bulk import documents
    client.bulk_import("documents", [
        {"_key": "doc1", "text": "...", "embedding": ...},
        {"_key": "doc2", "text": "...", "embedding": ...}
    ])

    # Query documents
    results = client.execute_query(
        "FOR doc IN documents RETURN doc",
        batch_size=100
    )
```

## Components

### Embedders

Metis provides a unified interface for text embedding models:

- **JinaV4Embedder**: State-of-the-art embeddings with 32k context window
- **Late Chunking**: Intelligent chunking that preserves semantic boundaries
- **Multi-GPU Support**: Distribute embedding workloads across multiple GPUs

### Extractors

Extract structured content from various document formats:

- **DoclingExtractor**: High-quality PDF extraction with structure preservation
- **LaTeXExtractor**: Perfect extraction from LaTeX source with equation support
- **CodeExtractor**: Extract code files with symbol tables using Tree-sitter

### Database

High-performance ArangoDB integration:

- **Unix Socket Support**: Low-latency local connections
- **HTTP/2 Protocol**: Efficient multiplexed requests
- **Bulk Operations**: Optimized batch insertion and querying
- **Transaction Support**: ACID guarantees for complex operations

## Architecture

Metis follows a modular architecture with clear separation of concerns:

```
metis/
├── embedders/     # Text embedding models
├── extractors/    # Document extraction
├── database/      # ArangoDB client
├── config/        # Configuration management
└── utils/         # Shared utilities
```

## Configuration

Metis can be configured via YAML files:

```yaml
# embedder_config.yaml
model_name: jinaai/jina-embeddings-v4
device: cuda:0
batch_size: 48
chunk_size_tokens: 500
chunk_overlap_tokens: 200
```

Load configuration:

```python
from metis import EmbeddingConfig, create_embedder
import yaml

with open("embedder_config.yaml") as f:
    config_dict = yaml.safe_load(f)

config = EmbeddingConfig(**config_dict)
embedder = create_embedder(config=config)
```

## Performance

Metis is designed for high-throughput production workloads:

- **Embedding Speed**: 40+ papers/sec on 2x A6000 GPUs
- **Database Latency**: <1ms for document lookups via Unix sockets
- **Batch Processing**: Efficient pipeline for processing large corpora

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/metis-ai/metis.git
cd metis

# Install with development dependencies
pip install -e ".[dev,all]"
```

### Testing

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=metis --cov-report=html
```

### Linting

```bash
# Format code
ruff format metis tests

# Check code
ruff check metis tests
```

## Examples

See the [examples/](examples/) directory for complete examples:

- `basic_setup.py`: Simple document processing pipeline
- `paper_ingestion.py`: Academic paper ingestion workflow
- `custom_extractors.py`: Extending Metis with custom extractors

## Documentation

Full documentation is available at [docs.metis-ai.org](https://docs.metis-ai.org):

- [API Reference](https://docs.metis-ai.org/api/)
- [User Guide](https://docs.metis-ai.org/guides/)
- [Examples](https://docs.metis-ai.org/examples/)

## License

Metis is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use Metis in your research, please cite:

```bibtex
@software{metis2025,
  title = {Metis: Semantic Knowledge Infrastructure},
  author = {Metis Contributors},
  year = {2025},
  url = {https://github.com/metis-ai/metis}
}
```

## Acknowledgments

Metis builds upon excellent open-source projects:

- [Sentence Transformers](https://www.sbert.net/)
- [Jina Embeddings](https://jina.ai/)
- [ArangoDB](https://www.arangodb.com/)
- [Docling](https://github.com/DS4SD/docling)
- [Tree-sitter](https://tree-sitter.github.io/)

---

**Metis**: Transforming information into actionable knowledge.
