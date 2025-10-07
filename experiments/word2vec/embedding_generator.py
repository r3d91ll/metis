"""
Embedding Generator for Word2Vec CF Experiments

Generates Jina v4 embeddings for combined paper+code contexts with late chunking.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from metis.embedders.base import EmbeddingConfig
from metis.embedders.jina_v4 import JinaV4Embedder

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings for Word2Vec family papers using Jina v4.

    Features:
    - Jina v4 with 2048-dimensional embeddings
    - 32k token context window
    - Late chunking for semantic boundary preservation
    - Task-specific prompts (passage encoding)
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v4",
        device: str = "cuda",
        batch_size: int = 4
    ):
        """
        Initialize embedding generator.

        Args:
            model_name: Jina model identifier
            device: Compute device (cuda/cpu)
            batch_size: Batch size for encoding
        """
        logger.info(f"Initializing Jina v4 embedder on {device}")

        config = EmbeddingConfig(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            trust_remote_code=True,
            chunk_size_tokens=32000,  # Full context window
            chunk_overlap_tokens=0,    # No overlap for single-context embedding
        )

        self.embedder = JinaV4Embedder(config)
        logger.info(
            f"Embedder ready: {self.embedder._dim} dims, "
            f"{self.embedder.chunk_size_tokens} token window"
        )

    def generate_embedding(
        self,
        context: str,
        task: str = "retrieval",
        prompt_name: str = "passage"
    ) -> np.ndarray:
        """
        Generate embedding for a single context.

        Args:
            context: Combined paper+code context
            task: Embedding task (default: retrieval)
            prompt_name: Jina prompt name (passage/query)

        Returns:
            2048-dimensional embedding vector
        """
        logger.info(
            f"Generating embedding for {len(context):,} char context "
            f"(~{len(context)//4:,} tokens)"
        )

        embedding = self.embedder.embed_texts(
            texts=[context],
            task=task,
            prompt_name=prompt_name
        )

        logger.info(f"Generated embedding: shape {embedding.shape}")
        return embedding[0]

    def generate_batch_embeddings(
        self,
        contexts: List[str],
        task: str = "retrieval",
        prompt_name: str = "passage"
    ) -> np.ndarray:
        """
        Generate embeddings for multiple contexts.

        Args:
            contexts: List of combined contexts
            task: Embedding task
            prompt_name: Jina prompt name

        Returns:
            Array of embeddings (n_contexts x 2048)
        """
        logger.info(f"Generating batch embeddings for {len(contexts)} contexts")

        embeddings = self.embedder.embed_texts(
            texts=contexts,
            task=task,
            prompt_name=prompt_name
        )

        logger.info(f"Generated embeddings: shape {embeddings.shape}")
        return embeddings

    def generate_for_paper(
        self,
        arxiv_id: str,
        context: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Generate embedding with metadata for a paper.

        Args:
            arxiv_id: arXiv identifier
            context: Combined context
            metadata: Additional metadata

        Returns:
            Dictionary with embedding and metadata
        """
        start_time = datetime.now()

        embedding = self.generate_embedding(context)

        processing_time = (datetime.now() - start_time).total_seconds()

        result = {
            "arxiv_id": arxiv_id,
            "embedding": embedding.tolist(),  # Convert to list for JSON serialization
            "dimensions": len(embedding),
            "context_chars": len(context),
            "context_tokens_estimate": len(context) // 4,
            "processing_time_seconds": processing_time,
            "timestamp": datetime.now().isoformat(),
            "model": "jinaai/jina-embeddings-v4",
            "task": "retrieval",
            "prompt": "passage"
        }

        if metadata:
            result["metadata"] = metadata

        logger.info(
            f"Paper {arxiv_id}: embedding generated in {processing_time:.2f}s "
            f"({len(context):,} chars)"
        )

        return result

    def close(self):
        """Clean up resources."""
        # JinaV4Embedder doesn't require explicit cleanup
        pass


def estimate_embedding_time(context_chars: int, chars_per_second: int = 50000) -> float:
    """
    Estimate embedding generation time.

    Args:
        context_chars: Number of characters in context
        chars_per_second: Processing throughput (default: 50k chars/s)

    Returns:
        Estimated time in seconds
    """
    return context_chars / chars_per_second


def estimate_memory_usage(embedding_dim: int = 2048, dtype_bytes: int = 4) -> int:
    """
    Estimate memory usage for embedding.

    Args:
        embedding_dim: Embedding dimensions
        dtype_bytes: Bytes per dimension (4 for float32)

    Returns:
        Memory usage in bytes
    """
    return embedding_dim * dtype_bytes
