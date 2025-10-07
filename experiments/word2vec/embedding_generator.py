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
        Create an EmbeddingGenerator configured for Jina v4 models.
        
        Initializes and prepares an underlying JinaV4Embedder using the given model, device, and batch size. The embedder is configured to use the full token context window and no chunk overlap.
        
        Parameters:
            model_name (str): Jina model identifier to load.
            device (str): Compute device to run the model on (e.g., "cuda" or "cpu").
            batch_size (int): Number of texts to encode per batch.
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
        Generate an embedding vector for the given text context.
        
        Parameters:
            context (str): Text to embed (e.g., combined paper and code context).
            task (str): Embedding task to use (e.g., "retrieval").
            prompt_name (str): Named prompt to guide the embedder (e.g., "passage" or "query").
        
        Returns:
            np.ndarray: A NumPy array of shape (2048,) containing the embedding for the provided context.
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
        Create embeddings for multiple input texts.
        
        Encodes each string in `contexts` using the configured embedder with the given task and prompt.
        
        Parameters:
            contexts (List[str]): Texts to encode, one entry per item to embed.
            task (str): Embedding task to request from the model (e.g., "retrieval").
            prompt_name (str): Prompt identifier to use with the Jina embedder.
        
        Returns:
            np.ndarray: Array of embeddings with shape (n_contexts, embedding_dim) — `embedding_dim` is the model's output dimension (2048 by default).
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
        Generate an embedding for a paper and return a metadata-rich result dictionary.
        
        Parameters:
            arxiv_id (str): arXiv identifier for the paper.
            context (str): Combined text context to encode into an embedding.
            metadata (Optional[Dict]): Optional additional metadata to include in the returned result.
        
        Returns:
            dict: A result dictionary containing:
                - "arxiv_id" (str): the provided arXiv identifier.
                - "embedding" (List[float]): embedding vector serialized as a list.
                - "dimensions" (int): embedding dimensionality.
                - "context_chars" (int): number of characters in the provided context.
                - "context_tokens_estimate" (int): approximate token count (context length // 4).
                - "processing_time_seconds" (float): wall-clock time taken to generate the embedding.
                - "timestamp" (str): ISO-formatted timestamp when the result was created.
                - "model" (str): model identifier used to produce the embedding.
                - "task" (str): task name used for the embedding call.
                - "prompt" (str): prompt name used for the embedding call.
                - "metadata" (Dict, optional): the provided additional metadata when supplied.
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
        """
        Placeholder method to release external resources associated with the embedder.
        
        This embedder does not require explicit cleanup, so this method performs no action.
        """
        # JinaV4Embedder doesn't require explicit cleanup
        pass


def estimate_embedding_time(context_chars: int, chars_per_second: int = 50000) -> float:
    """
    Estimate time to generate an embedding for a given text length.
    
    Parameters:
        context_chars (int): Number of characters to embed.
        chars_per_second (int): Assumed throughput in characters per second (default 50000).
    
    Returns:
        float: Estimated time in seconds.
    """
    return context_chars / chars_per_second


def estimate_memory_usage(embedding_dim: int = 2048, dtype_bytes: int = 4) -> int:
    """
    Estimate the memory (in bytes) required to store a single embedding vector.
    
    Parameters:
        embedding_dim (int): Number of dimensions in the embedding.
        dtype_bytes (int): Number of bytes per element (e.g., 4 for float32).
    
    Returns:
        int: Memory usage in bytes for the embedding.
    """
    return embedding_dim * dtype_bytes