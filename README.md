# Metis - Semantic Knowledge Infrastructure

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Metis** is a Python library for building semantic graph databases with ArangoDB and embedding models. Named after the Titaness of wisdom and deep thought, Metis helps you transform raw documents into structured, searchable knowledge graphs.

## System Requirements

### Core Requirements

**Python**

- Python 3.12+ (3.11 minimum, 3.12+ recommended)
- pip 23.0 or later

**GPU & CUDA**

- NVIDIA GPU with CUDA Compute Capability 7.0+ (required)
- CUDA 11.8 or 12.1+
- cuDNN 8.9+
- 16GB+ VRAM recommended for Jina v4 embeddings
- Note: CPU-only inference is not currently supported for production workloads

**Database**

- ArangoDB 3.11+ (3.12+ recommended for vector index support)
- Unix socket access enabled
- 16GB+ RAM for database operations
- SSD storage recommended for optimal performance

**Go (for proxy compilation)**

- Go 1.23.0+ (for building Unix socket proxies)
- Make (optional, for build automation)

### Optional Requirements

**PDF Processing**

- Docling 2.54.0+ (optional, for high-quality PDF extraction)
- PyMuPDF (fallback PDF extraction)

**Code Analysis**

- Tree-sitter 0.25.0+
- Tree-sitter language grammars 1.10.0+

### Operating System

**Supported Platforms**

- Linux (Ubuntu 22.04+, Debian 12+, RHEL 9+)
- GPU drivers: NVIDIA 525.x+ recommended

**Not Currently Supported**

- Windows (WSL2 may work but is untested)
- macOS (no CUDA support)

### Hardware Recommendations

**Minimum (Development)**

- CPU: 4 cores
- RAM: 16GB
- GPU: NVIDIA GTX 3090 Ti (24GB VRAM)
- Storage: 50GB SSD

**Recommended (Production)**

- CPU: 16+ cores
- RAM: 64GB+
- GPU: NVIDIA A6000 (48GB VRAM) or better
- Storage: 500GB+ NVMe SSD
- Network: 10Gbps for distributed setups

**High-Performance (Large-Scale)**

- CPU: 32+ cores
- RAM: 128GB+
- GPU: 2x NVIDIA A100 (80GB VRAM each)
- Storage: 2TB+ NVMe SSD RAID
- Network: 25Gbps or higher

### Runtime Dependencies

**Core Libraries**

- PyTorch 2.0+ with CUDA support
- sentence-transformers 3.3.0+
- httpx 0.27.0+ (with HTTP/2 support)
- numpy 1.26.0+
- pyyaml 6.0+

**Database Client**

- h2 4.1.0+ (HTTP/2 protocol)
- python-arango alternative not supported (uses custom HTTP/2 client)

### Verification

Check your system meets requirements:

```bash
# Python version
python --version  # Should show 3.12+

# CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# GPU info
nvidia-smi

# ArangoDB version
arangod --version  # Should show 3.11+

# Go version (for building proxies)
go version  # Should show 1.23+
```

## Features

- **Multi-Format Document Extraction**: Extract text and structure from PDFs, LaTeX sources, and code files
- **State-of-the-Art Embeddings**: Built-in support for Jina v4 embeddings with 32k context window
- **High-Performance Database**: ArangoDB client with Unix socket support for low-latency operations
- **Graph Neural Networks**: Multi-relational GraphSAGE for inductive learning on heterogeneous code and knowledge graphs
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

For GNN support (GraphSAGE):

```bash
pip install metis[gnn]
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
from metis.database import ArangoClient, resolve_client_config

# Configure database connection
config = resolve_client_config(
    database="my_knowledge_base",
    socket_path="/var/run/arangodb/socket"
)

# Create client
with ArangoClient(config) as client:
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

High-performance ArangoDB integration with security-hardened Unix socket access:

- **Unix Socket Proxies**: Read-only and read-write security layers (Go-based)
- **HTTP/2 Protocol**: Efficient multiplexed requests with persistent connections
- **Bulk Operations**: Optimized batch insertion and querying with NDJSON
- **Security Enforcement**: Policy-based access control at the socket level
- **Admin Operations**: Idempotent database, index, and view management

### Graph Neural Networks

Multi-relational GraphSAGE implementation for learning on heterogeneous graphs:

- **MultiRelationalGraphSAGE**: Inductive node embedding model supporting multiple edge types
- **GraphSAGEInference**: Fast inference wrapper for retrieval (<50ms per batch)
- **GraphBuilder**: Convert ArangoDB graphs to PyTorch Geometric format
- **Edge Types**: `imports`, `contains`, `references`, `cites`, `authored_by`, `part_of`
- **Architecture**: 2048-dim (Jina v4) → 1024-dim → 512-dim for efficient retrieval
- **Training**: Contrastive learning with query-node similarity optimization
- **Inductive Learning**: Generate embeddings for new nodes without retraining

## Architecture

Metis follows a modular architecture with clear separation of concerns:

```
metis/
├── embedders/     # Text embedding models (Jina v4, Sentence Transformers)
├── extractors/    # Document extraction (PDF, LaTeX, Code)
├── database/      # ArangoDB client with Unix socket proxies
│   ├── client.py  # HTTP/2 client
│   ├── memory.py  # High-level wrapper
│   ├── admin.py   # Admin operations
│   └── proxies/   # Go security proxies (RO/RW)
├── gnn/           # Graph Neural Networks (GraphSAGE)
│   ├── graphsage_model.py  # Multi-relational GraphSAGE
│   ├── trainer.py          # Training pipeline
│   ├── inference.py        # Fast retrieval inference
│   └── graph_builder.py    # ArangoDB to PyG converter
├── config/        # Configuration management
├── chunking/      # Late chunking support
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
- **Database Latency**: <0.4ms p50 upstream via Unix sockets, ~0.2ms proxy overhead
- **Batch Processing**: Efficient NDJSON pipeline for processing large corpora
- **Security Overhead**: Minimal (~0.2ms) with HTTP/2 connection pooling

## Unix Socket Proxies

Metis includes Go-based security proxies for ArangoDB Unix socket access. These proxies provide fine-grained access control at the socket level:

### Read-Only Proxy

- Allows GET, HEAD, OPTIONS requests
- Permits AQL queries (POST to `/_api/cursor`) but blocks mutation keywords
- Socket permissions: 0660 (group access)

### Read-Write Proxy

- Allows all read-only operations
- Permits DB-scoped mutations (document CRUD, index management)
- Blocks admin endpoints (database, view, analyzer creation)
- Socket permissions: 0660 (dev), 0600 (production)

### Building Proxies

```bash
cd metis/database/proxies
make build          # Build both proxies
sudo make install   # Install to /usr/local/bin
```

For detailed proxy documentation, see [metis/database/proxies/README.md](metis/database/proxies/README.md).

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

### Graph Neural Network Usage

Build and train GraphSAGE models on code and knowledge graphs:

```python
from metis.gnn import MultiRelationalGraphSAGE, GraphSAGEInference
from metis.gnn.graph_builder import GraphBuilder
from metis.database import ArangoClient, resolve_client_config

# Build graph from ArangoDB
config = resolve_client_config(database="my_kb")
with ArangoClient(config) as client:
    builder = GraphBuilder(client)
-    data, node_id_map = builder.build_graph(
-        include_collections=["repo_docs"],
-        edge_types=["imports", "contains", "references"]
    data, node_id_map = builder.build_graph(
        include_collections=["repo_docs"],
        edge_types=["imports", "contains", "references", "cites"]
    )

# Create GraphSAGE model
model = MultiRelationalGraphSAGE(
    in_channels=2048,      # Jina v4 embedding dimension
    hidden_channels=1024,
    out_channels=512,
    num_layers=3,
    edge_types=["imports", "contains", "references", "cites"]
)

# Train model (see metis.gnn.trainer for full training pipeline)
from metis.gnn.trainer import GraphSAGETrainer, TrainingConfig

config = TrainingConfig(
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    device="cuda"
)
trainer = GraphSAGETrainer(config, edge_types=model.edge_types)

# Train the model (requires labeled query-node pairs for contrastive learning)
# trainer.fit(
#     data,                  # PyG Data object with graph
#     query_embeddings,      # Query embeddings [num_queries, 2048]
#     node_indices,          # Target node index for each query [num_queries]
#     labels,                # Binary labels: 1=relevant, 0=irrelevant [num_queries]
#     train_mask,            # Boolean mask for training samples
#     val_mask              # Boolean mask for validation samples
# )

# Fast inference for retrieval
inference = GraphSAGEInference.from_checkpoint(
    checkpoint_path="models/checkpoints/best.pt",
    node_embeddings=precomputed_embeddings,
    node_ids=list(node_id_map.values()),
    device="cuda"
)

# Find relevant nodes for a query
query_embedding = embedder.embed_texts(["your search query"])[0]
candidates = inference.find_relevant_nodes(
    query_embedding=query_embedding,
    top_k=50,
    min_score=0.5  # Cosine similarity threshold (0-1), tune based on use case
)

# Results: [(node_id, similarity_score), ...]
for node_id, score in candidates[:10]:
    print(f"{node_id}: {score:.3f}")
```
