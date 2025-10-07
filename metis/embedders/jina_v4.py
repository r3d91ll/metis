#!/usr/bin/env python3
"""SentenceTransformers-backed Jina v4 embedder.

The upstream Jina embeddings model is now exposed via sentence-transformers:
https://huggingface.co/jinaai/jina-embeddings-v4

This implementation provides a unified interface with late chunking support
and prompt selection (`query` vs `passage`) matching Jina's recommended usage.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import EmbeddingConfig
from .sentence import (
    SentenceTransformersEmbedder,
    STChunkWithEmbedding,
)

ChunkWithEmbedding = STChunkWithEmbedding

logger = logging.getLogger(__name__)


class JinaV4Embedder(SentenceTransformersEmbedder):
    """SentenceTransformers wrapper for jinaai/jina-embeddings-v4."""

    DEFAULT_MODEL = "jinaai/jina-embeddings-v4"

    def __init__(self, config: Optional[EmbeddingConfig] = None) -> None:
        """
        Initialize the JinaV4Embedder with Jina v4-specific defaults and enforced settings.
        
        If `config` is omitted or lacks a model_name, the embedder uses the class DEFAULT_MODEL; if a different model_name is provided it will be overridden to DEFAULT_MODEL. The constructor enables trust_remote_code to allow Jina's custom modules, calls the parent initializer with the resulting config, and forces the embedding dimensionality to 2048. It also sets sensible defaults for chunk_size_tokens (1024) and chunk_overlap_tokens (256) when not specified in the config.
        
        Parameters:
            config (Optional[EmbeddingConfig]): Optional embedding configuration; if provided, its `model_name` may be replaced by the embedder's DEFAULT_MODEL and `trust_remote_code` will be set to True.
        """
        if config is None:
            config = EmbeddingConfig(model_name=self.DEFAULT_MODEL)
        else:
            if not config.model_name:
                config.model_name = self.DEFAULT_MODEL
            if config.model_name != self.DEFAULT_MODEL:
                logger.info("Overriding model name to %s", self.DEFAULT_MODEL)
                config.model_name = self.DEFAULT_MODEL

        # Always trust remote code for Jina's custom modules
        config.trust_remote_code = True

        super().__init__(config)

        # Force Jina v4 to use 2048 dimensions (full fidelity)
        # The model supports 512/768/2048, but we want max dimensional fidelity
        self._dim = 2048
        logger.info(f"Jina v4 embedder initialized with {self._dim} dimensions")

        # Jina v4 tolerates long contexts; allow wider chunks by default
        self.chunk_size_tokens = self.config.chunk_size_tokens or 1024
        self.chunk_overlap_tokens = self.config.chunk_overlap_tokens or 256

    def _infer_embedding_dim(self, default: int) -> int:
        """
        Always use 2048 as the embedding dimension for Jina v4 models.
        
        Parameters:
            default (int): Ignored. Present for signature compatibility with the base class.
        
        Returns:
            embedding_dim (int): The fixed embedding dimensionality (2048).
        """
        return 2048

    def _load_model(self, model_name: str, device: str) -> SentenceTransformer:
        """
        Load a SentenceTransformer instance configured for the Jina v4 embedder.
        
        Constructs and returns a SentenceTransformer using the given model name and device while:
        - forcing `trust_remote_code=True` to allow Jina's custom modules,
        - setting `model_kwargs={"default_task": "retrieval"}`,
        - propagating `local_files_only` from `self.config` if present,
        - using `self.config.cache_dir` as `cache_folder` when provided.
        
        Returns:
            A configured SentenceTransformer instance.
        """
        kwargs: Dict[str, Any] = {
            "device": device,
            "trust_remote_code": True,  # Required for Jina v4 custom code
            "local_files_only": getattr(self.config, "local_files_only", False),
            "model_kwargs": {"default_task": "retrieval"},
        }
        cache_dir = getattr(self.config, "cache_dir", None)
        if cache_dir:
            kwargs["cache_folder"] = cache_dir
        return SentenceTransformer(model_name, **kwargs)

    def embed_texts(
        self,
        texts: List[str],
        task: str = "retrieval",
        batch_size: Optional[int] = None,
        prompt_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts using the configured Jina v4 model.
        
        Parameters:
            texts (List[str]): Input texts to embed.
            task (str): Embedding task hint; determines default prompt type (e.g., contains "passage" selects passage prompt).
            batch_size (Optional[int]): Number of texts to process per batch; if omitted, uses the embedder's configured batch size.
            prompt_name (Optional[str]): Explicit prompt name to use; when omitted a prompt is chosen based on `task`.
        
        Returns:
            np.ndarray: Array of embedding vectors with shape (len(texts), embedding_dimension) and dtype float32. For an empty `texts` list returns an array with shape (0, embedding_dimension).
        """
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        jina_prompt = prompt_name or ("passage" if "passage" in task else "query")
        embeddings = self.model.encode(
            texts,
            prompt_name=jina_prompt,
            batch_size=batch_size or self.config.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            truncate_dim=None,  # Use full 2048 dimensions (don't truncate to 512/768)
            device=self.model.device,  # Ensure inputs are on same device as model
        )
        return embeddings.astype(np.float32, copy=False)

    def embed_single(
        self,
        text: str,
        task: str = "retrieval",
        prompt_name: Optional[str] = None,
    ) -> np.ndarray:
        emb = self.embed_texts([text], task=task, batch_size=1, prompt_name=prompt_name)
        return emb[0] if emb.size else np.zeros(self._dim, dtype=np.float32)

    def embed_code(self, code_snippets: List[str], batch_size: int = 4) -> np.ndarray:
        return self.embed_texts(
            code_snippets,
            task="code",
            prompt_name="passage",
            batch_size=batch_size,
        )

    def embed_images(self, images):  # type: ignore[override]
        raise NotImplementedError("Image embeddings are not supported via sentence-transformers")

    def embed_multimodal(self, pairs):  # type: ignore[override]
        raise NotImplementedError("Multimodal embeddings are not supported via sentence-transformers")

    @property
    def embedding_dimension(self) -> int:
        return self._dim

    @property
    def max_sequence_length(self) -> int:
        # Jina embeddings support very long contexts; ST wrapper will tokenize/chunk as needed.
        return 32768

    @property
    def supports_images(self) -> bool:
        return False

    @property
    def supports_multimodal(self) -> bool:
        return False