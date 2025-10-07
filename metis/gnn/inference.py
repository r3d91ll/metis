#!/usr/bin/env python3
"""
GraphSAGE Inference for Retrieval
==================================

Fast inference wrapper for GraphSAGE in retrieval pipeline.

Target: <50ms per batch for integration with PathRAG
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from .graphsage_model import MultiRelationalGraphSAGE

logger = logging.getLogger(__name__)


class GraphSAGEInference:
    """
    Fast inference wrapper for GraphSAGE retrieval.

    Pipeline:
    1. Query embedding (Jina v4)
    2. GraphSAGE finds relevant nodes → candidates (may be many)
    3. PathRAG prunes candidates to paths within budget
    """

    def __init__(
        self,
        model: MultiRelationalGraphSAGE,
        query_projection: Optional[nn.Linear] = None,
        node_embeddings: Optional[Tensor] = None,
        node_ids: Optional[List[str]] = None,
        device: str = "cpu",
    ):
        """
        Wrap a trained MultiRelationalGraphSAGE model with optional projection and precomputed embeddings for device-aware inference.
        
        Stores the model on the specified device and prepares optional components for fast retrieval-based inference.
        
        Parameters:
            model: A trained MultiRelationalGraphSAGE model; it will be moved to `device` and set to evaluation mode.
            query_projection: Optional linear layer that maps query embeddings into the node-embedding space (e.g., 2048→512); if provided it will be moved to `device` and set to evaluation mode.
            node_embeddings: Optional tensor of shape [num_nodes, out_channels] containing precomputed node embeddings; if provided it will be moved to `device` and used for similarity search.
            node_ids: Optional list of node identifiers corresponding positionally to `node_embeddings`; used to construct an index mapping from node_id to embedding index.
            device: Device identifier (e.g., "cpu" or "cuda") used for model and tensor placement.
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        # Query projection layer (2048→512)
        self.query_projection = query_projection.to(device) if query_projection is not None else None
        if self.query_projection is not None:
            self.query_projection.eval()

        # Precomputed embeddings for fast lookup
        self.node_embeddings = node_embeddings.to(device) if node_embeddings is not None else None
        self.node_ids = node_ids or []

        # Build index mapping node_id → embedding index
        self.node_id_to_idx = {nid: i for i, nid in enumerate(self.node_ids)}

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        node_embeddings: Optional[Tensor] = None,
        node_ids: Optional[List[str]] = None,
        device: str = "cpu",
    ) -> GraphSAGEInference:
        """
        Construct a GraphSAGEInference instance by loading a saved model (and optional query projection) from a checkpoint file.
        
        Security: only load checkpoints from trusted sources because checkpoint files execute code during deserialization.
        
        Parameters:
            checkpoint_path (Path): Filesystem path to the saved checkpoint containing model state and optional metadata.
            node_embeddings (Optional[Tensor]): Optional precomputed node embeddings tensor shaped [num_nodes, out_channels]; use None to omit.
            node_ids (Optional[List[str]]): Optional list mapping embedding indices to node identifier strings; use None to omit.
            device (str): Device identifier for model placement (e.g., "cpu" or "cuda").
        
        Returns:
            GraphSAGEInference: An inference wrapper with the loaded model, optional query projection, and the provided node embeddings and IDs.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

        model = MultiRelationalGraphSAGE(
            in_channels=checkpoint.get("in_channels", 2048),
            hidden_channels=checkpoint.get("hidden_channels", 1024),
            out_channels=checkpoint.get("out_channels", 512),
            num_layers=checkpoint.get("num_layers", 3),
            dropout=checkpoint.get("dropout", 0.3),
            edge_types=checkpoint.get("edge_types"),
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Load query projection if available
        query_projection = None
        if "query_projection_state_dict" in checkpoint:
            in_channels = checkpoint.get("in_channels", 2048)
            out_channels = checkpoint.get("out_channels", 512)
            query_projection = nn.Linear(in_channels, out_channels)
            query_projection.load_state_dict(checkpoint["query_projection_state_dict"])
            query_projection.eval()

        return cls(
            model=model,
            query_projection=query_projection,
            node_embeddings=node_embeddings,
            node_ids=node_ids,
            device=device,
        )

    def find_relevant_nodes(
        self,
        query_embedding: np.ndarray,
        top_k: int = 50,
        min_score: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve the most relevant node IDs and their similarity scores for a given query embedding.
        
        Parameters:
            query_embedding (np.ndarray): Query embedding vector to match against precomputed node embeddings.
            top_k (int): Maximum number of candidate nodes to return.
            min_score (float): Minimum cosine similarity required for a node to be considered.
        
        Returns:
            List[Tuple[str, float]]: List of (node_id, score) tuples sorted by descending score.
        
        Raises:
            ValueError: If precomputed node embeddings are not available or the query projection layer is missing.
        """
        if self.node_embeddings is None:
            raise ValueError("No precomputed node embeddings available")

        if self.query_projection is None:
            raise ValueError("No query projection layer available - checkpoint missing query_projection_state_dict")

        # Convert query to tensor
        query_tensor = torch.from_numpy(query_embedding).float().to(self.device)
        query_tensor = query_tensor.unsqueeze(0)  # [1, 2048]

        # Project query to match node embedding dimension (2048→512)
        with torch.no_grad():
            query_tensor = self.query_projection(query_tensor)  # [1, 512]

        # Normalize query
        query_tensor = torch.nn.functional.normalize(query_tensor, p=2, dim=1)

        # Compute cosine similarity with all nodes
        with torch.no_grad():
            similarities = torch.matmul(query_tensor, self.node_embeddings.t())  # [1, num_nodes]
            similarities = similarities.squeeze(0)  # [num_nodes]

        # Filter by minimum score
        mask = similarities >= min_score
        filtered_scores = similarities[mask]
        filtered_indices = torch.nonzero(mask, as_tuple=True)[0]

        if len(filtered_scores) == 0:
            logger.warning(f"No nodes found with score >= {min_score}")
            return []

        # Get top-k
        k = min(top_k, len(filtered_scores))
        top_scores, top_idx = torch.topk(filtered_scores, k=k)

        # Map back to node IDs
        results = []
        for score, idx in zip(top_scores.cpu().numpy(), top_idx.cpu().numpy()):
            node_idx = filtered_indices[idx].item()
            node_id = self.node_ids[node_idx]
            results.append((node_id, float(score)))

        return results

    def embed_new_node(
        self,
        node_features: np.ndarray,
        edge_index_dict: Dict[str, Tensor],
    ) -> np.ndarray:
        """
        Compute an inductive embedding for a newly added node.
        
        Parameters:
            node_features (np.ndarray): 1-D array of node features with length equal to the model's input channel size.
            edge_index_dict (Dict[str, Tensor]): Mapping of edge type names to edge index tensors that connect the new node to the existing graph.
        
        Returns:
            np.ndarray: Embedding vector for the new node with shape [out_channels].
        """
        node_tensor = torch.from_numpy(node_features).float().to(self.device)
        node_tensor = node_tensor.unsqueeze(0)  # [1, in_channels]

        with torch.no_grad():
            embedding = self.model.inductive_embed(node_tensor, edge_index_dict)

        return embedding.cpu().numpy().squeeze(0)

    def precompute_embeddings(
        self,
        x: Tensor,
        edge_index_dict: Dict[str, Tensor],
    ) -> Tensor:
        """
        Precompute embeddings for all nodes in the graph.
        
        Parameters:
            x (Tensor): Node feature matrix with shape [num_nodes, in_channels].
            edge_index_dict (Dict[str, Tensor]): Mapping of edge type to edge index tensors.
        
        Returns:
            Tensor: Node embeddings with shape [num_nodes, out_channels].
        """
        with torch.no_grad():
            embeddings = self.model(x, edge_index_dict)

        return embeddings