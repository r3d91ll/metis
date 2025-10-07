#!/usr/bin/env python3
"""Direct transformers-based Jina v4 embedder for multi-GPU support.

This bypasses sentence-transformers to properly support multi-GPU with CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .base import EmbedderBase, EmbeddingConfig

logger = logging.getLogger(__name__)


class JinaV4TransformersEmbedder(EmbedderBase):
    """Direct transformers implementation of Jina v4 for multi-GPU support."""

    DEFAULT_MODEL = "jinaai/jina-embeddings-v4"

    def __init__(self, config: Optional[EmbeddingConfig] = None) -> None:
        """
        Initialize the embedder, load the Jina v4 transformer model, and configure device and embedding dimension.
        
        Parameters:
            config (Optional[EmbeddingConfig]): Optional configuration object. If provided, the following fields are honored:
                - model_name: used to select the pretrained model; names ending with "-transformers" are normalized to "jinaai/jina-embeddings-v4".
                - device: overrides automatic device selection; if omitted, uses "cuda" when available, otherwise "cpu".
                - local_files_only: when True, restricts model loading to local files.
        
        Notes:
            - Loads the model via AutoModel.from_pretrained(..., trust_remote_code=True) and moves it to the selected device.
            - Sets self.model to the loaded model, self.device to the chosen device string, and self._dim to 2048.
        """
        super().__init__(config)

        # Normalize model name: allow hints like "...-transformers" but load the actual repo
        model_name = (self.config.model_name or self.DEFAULT_MODEL)
        if model_name.lower().endswith("-transformers"):
            model_name = self.DEFAULT_MODEL
            self.config.model_name = model_name

        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Verify which GPU is actually visible (respects CUDA_VISIBLE_DEVICES)
        if "cuda" in device and torch.cuda.is_available():
            try:
                visible_devices = torch.cuda.device_count()
                actual_device = torch.cuda.current_device()
                name = torch.cuda.get_device_name(actual_device)
                logger.info(f"CUDA visible devices: {visible_devices}, current: {actual_device} ({name})")
            except Exception:
                logger.info("CUDA device info not available")

        logger.info(f"Loading Jina v4 with transformers on device: {device}")

        # Load model with explicit device placement (trusts remote code which exposes encode_text)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            attn_implementation="sdpa",
            local_files_only=getattr(self.config, "local_files_only", False),
        )

        # Explicitly move to device AFTER loading
        self.model = self.model.to(device)
        self.model.eval()

        self.device = device
        self._dim = 2048  # Jina v4 full dimensions

        logger.info(f"Jina v4 loaded on {device}, embedding dim: {self._dim}")

    @property
    def embedding_dim(self) -> int:
        """
        Return the embedder's output vector dimensionality.
        
        Returns:
            dim (int): Number of elements in each embedding vector.
        """
        return self._dim

    @property
    def embedding_dimension(self) -> int:
        """
        Compatibility alias for the embedder's embedding dimension.
        
        Returns:
            embedding_dimension (int): The size of the output embeddings (number of dimensions).
        """
        return self._dim

    @property
    def max_sequence_length(self) -> int:
        """
        The maximum sequence length accepted by the embedder.
        
        Returns:
            int: Maximum number of tokens for input sequences; uses `config.max_seq_length` if set, otherwise 8192.
        """
        return getattr(self.config, "max_seq_length", 8192)

    def embed_texts(
        self,
        texts: List[str],
        task: str = "retrieval",
        batch_size: Optional[int] = None,
        prompt_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Compute embeddings for a list of texts using the loaded Jina v4 transformers model.
        
        The method resolves batch size from the argument, configuration, or defaults to 32, and forwards the task and prompt name to the model's encode_text API. If `texts` is empty, returns an empty float32 array with shape (0, embedding_dim).
        
        Parameters:
        	texts (List[str]): Input texts to embed.
        	task (str): Semantic task to guide encoding (defaults to "retrieval").
        	batch_size (Optional[int]): Number of texts processed per batch; if None, uses config.batch_size or 32.
        	prompt_name (Optional[str]): Prompt slot name to use with the model (defaults to "query").
        
        Returns:
        	np.ndarray: Float32 array of shape (len(texts), self._dim) containing the embeddings. 
        """
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        batch_size = batch_size or self.config.batch_size or 32

        # Jina v4 supports encode_text with task and prompt_name; use that API
        jina_prompt = prompt_name or "query"
        task_name = task or "retrieval"

        with torch.no_grad():
            arr = self.model.encode_text(
                texts,
                task=task_name,
                max_length=getattr(self.config, "max_seq_length", 8192),
                batch_size=batch_size,
                return_multivector=False,
                return_numpy=True,
                truncate_dim=None,
                prompt_name=jina_prompt,
            )

        # encode_text returns numpy array for list inputs
        if isinstance(arr, list):
            # Should not happen with return_numpy=True and list inputs, but guard anyway
            arr = np.vstack([np.asarray(x) for x in arr])
        return arr.astype(np.float32, copy=False)

    def embed_single(self, text: str, task: str = "retrieval") -> np.ndarray:
        """
        Produce an embedding vector for a single text.
        
        Parameters:
            task (str): The embedding task to perform (e.g., "retrieval"). Defaults to "retrieval".
        
        Returns:
            np.ndarray: 1-D float32 NumPy array of length 2048 containing the embedding for the input text.
        """
        return self.embed_texts([text], task=task)[0]