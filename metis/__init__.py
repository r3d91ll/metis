"""
Metis - Semantic Knowledge Infrastructure

A Python library for building semantic graph databases with ArangoDB and embedding models.

Named after the Titaness of wisdom and transformation, Metis helps you transform
raw documents into structured, searchable knowledge graphs.

Main Components:
- embedders: Text embedding with Jina v4 and other models
- extractors: Document extraction (PDF, LaTeX, code)
- database: ArangoDB client with Unix socket support
- config: Configuration management

Example:
    >>> from metis import create_embedder, create_extractor_for_file
    >>>
    >>> # Create embedder
    >>> embedder = create_embedder("jinaai/jina-embeddings-v4", device="cuda:0")
    >>>
    >>> # Extract and embed a document
    >>> extractor = create_extractor_for_file("paper.pdf")
    >>> result = extractor.extract("paper.pdf")
    >>> embeddings = embedder.embed_texts([result.text])
"""

__version__ = "0.1.0"
__author__ = "Metis Contributors"
__license__ = "Apache-2.0"

# Import main components
from .embedders import (
    ChunkWithEmbedding,
    EmbedderBase,
    EmbedderFactory,
    EmbeddingConfig,
    JinaV4Embedder,
    create_embedder,
)
from .extractors import (
    CodeExtractor,
    DoclingExtractor,
    ExtractionResult,
    ExtractorBase,
    ExtractorConfig,
    ExtractorFactory,
    LaTeXExtractor,
    create_extractor_for_file,
)
from .database import (
    ArangoHttp2Client,
    ArangoHttp2Config,
    ArangoHttpError,
    ArangoMemoryClient,
    ArangoMemoryClientConfig,
    CollectionDefinition,
    MemoryServiceError,
    resolve_memory_config,
)

__all__ = [
    # Version
    "__version__",
    # Embedders
    "EmbedderBase",
    "EmbeddingConfig",
    "EmbedderFactory",
    "JinaV4Embedder",
    "ChunkWithEmbedding",
    "create_embedder",
    # Extractors
    "ExtractorBase",
    "ExtractorConfig",
    "ExtractionResult",
    "ExtractorFactory",
    "DoclingExtractor",
    "LaTeXExtractor",
    "CodeExtractor",
    "create_extractor_for_file",
    # Database
    "ArangoHttp2Client",
    "ArangoHttp2Config",
    "ArangoHttpError",
    "ArangoMemoryClient",
    "ArangoMemoryClientConfig",
    "CollectionDefinition",
    "MemoryServiceError",
    "resolve_memory_config",
]
