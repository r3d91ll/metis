"""
Embedders Module

Provides embedding models for transforming text into vector representations.
Supports multiple embedding backends with a consistent interface.
"""

from .base import EmbedderBase, EmbeddingConfig
from .factory import EmbedderFactory

# Auto-register available embedders
try:
    from .jina_v4 import ChunkWithEmbedding, JinaV4Embedder

    EmbedderFactory.register("jina", JinaV4Embedder)
except ImportError:
    JinaV4Embedder = None  # type: ignore[misc]
    ChunkWithEmbedding = None  # type: ignore[misc]

__all__ = [
    "EmbedderBase",
    "EmbeddingConfig",
    "EmbedderFactory",
    "JinaV4Embedder",
    "ChunkWithEmbedding",
]


# Convenience function
def create_embedder(model_name: str = "jinaai/jina-embeddings-v4", **kwargs):
    """
    Create an embedder instance.

    Args:
        model_name: Model name or path
        **kwargs: Additional configuration (device, batch_size, etc.)

    Returns:
        Embedder instance

    Example:
        >>> embedder = create_embedder("jinaai/jina-embeddings-v4", device="cuda:0")
        >>> embeddings = embedder.embed_texts(["Hello world"])
    """
    return EmbedderFactory.create(model_name, **kwargs)
