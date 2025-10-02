#!/usr/bin/env python3
"""
Multi-Relational GraphSAGE Model
=================================

GraphSAGE implementation for heterogeneous knowledge graphs with multiple edge types.

Architecture:
- Input: Jina v4 embeddings (2048-dim)
- 3x SAGEConv layers with different aggregators per edge type
- Output: 512-dim node embeddings for fast retrieval

Edge Types (Research Use Cases):
- imports: code → code (Python imports, package dependencies)
- contains: directory → file (filesystem structure, document hierarchy)
- references: code → code (function calls, cross-references)
- cites: paper → paper (academic citations)
- authored_by: document → author (authorship relationships)
- part_of: section → document (document structure)

Inductive Learning:
- Generates embeddings for NEW nodes without retraining
- Critical for dynamic knowledge graphs that grow over time
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from torch_geometric.nn import SAGEConv
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    SAGEConv = None


class MultiRelationalGraphSAGE(nn.Module):
    """
    Multi-relational GraphSAGE for heterogeneous code + memory graph.

    Args:
        in_channels: Input feature dimension (default: 2048 for Jina v4)
        hidden_channels: Hidden layer dimension (default: 1024)
        out_channels: Output embedding dimension (default: 512)
        num_layers: Number of GraphSAGE layers (default: 3)
        dropout: Dropout probability (default: 0.3)
        edge_types: List of edge type names for multi-relational learning
    """

    def __init__(
        self,
        in_channels: int = 2048,
        hidden_channels: int = 1024,
        out_channels: int = 512,
        num_layers: int = 3,
        dropout: float = 0.3,
        edge_types: Optional[List[str]] = None,
    ):
        super().__init__()

        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "PyTorch Geometric not installed. "
                "Install with: poetry install --extras gnn"
            )

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        # Default edge types for research knowledge graphs
        self.edge_types = edge_types or [
            "imports",
            "contains",
            "references",
            "cites",
            "authored_by",
            "part_of",
        ]

        # Create separate SAGEConv layers for each edge type
        self.convs = nn.ModuleDict()

        for edge_type in self.edge_types:
            layers = nn.ModuleList()

            # First layer: in_channels → hidden_channels
            layers.append(SAGEConv(in_channels, hidden_channels, aggr='mean'))

            # Middle layers: hidden_channels → hidden_channels
            for _ in range(num_layers - 2):
                layers.append(SAGEConv(hidden_channels, hidden_channels, aggr='mean'))

            # Final layer: hidden_channels → out_channels
            layers.append(SAGEConv(hidden_channels, out_channels, aggr='mean'))

            self.convs[edge_type] = layers

        # Attention weights for combining multiple edge types
        self.edge_type_attention = nn.Parameter(torch.ones(len(self.edge_types)))

        # Fallback projection when no edges are available
        self.fallback_projection = nn.Linear(in_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize model parameters."""
        for edge_type in self.edge_types:
            for conv in self.convs[edge_type]:
                conv.reset_parameters()

        nn.init.ones_(self.edge_type_attention)
        self.fallback_projection.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index_dict: Dict[str, Tensor],
    ) -> Tensor:
        """
        Forward pass through multi-relational GraphSAGE.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index_dict: Dict mapping edge_type → edge_index [2, num_edges]

        Returns:
            Node embeddings [num_nodes, out_channels]
        """

        # Aggregate embeddings from each edge type
        edge_embeddings = []
        edge_weights = []

        for i, edge_type in enumerate(self.edge_types):
            if edge_type not in edge_index_dict:
                continue

            edge_index = edge_index_dict[edge_type]
            h = x

            # Apply GraphSAGE layers for this edge type
            for layer_idx, conv in enumerate(self.convs[edge_type]):
                h = conv(h, edge_index)

                # Apply ReLU + dropout (except last layer)
                if layer_idx < self.num_layers - 1:
                    h = F.relu(h)
                    h = F.dropout(h, p=self.dropout, training=self.training)

            edge_embeddings.append(h)
            edge_weights.append(self.edge_type_attention[i])

        if not edge_embeddings:
            # No edges available - use fallback projection
            return F.normalize(self.fallback_projection(x), p=2, dim=1)

        # Weighted combination of edge type embeddings
        edge_weights = F.softmax(torch.stack(edge_weights), dim=0)
        combined = sum(w * emb for w, emb in zip(edge_weights, edge_embeddings))

        # L2 normalize for cosine similarity in retrieval
        combined = F.normalize(combined, p=2, dim=1)

        return combined

    def inductive_embed(
        self,
        x_new: Tensor,
        edge_index_dict: Dict[str, Tensor],
    ) -> Tensor:
        """
        Generate embeddings for NEW nodes inductively.

        This is the key capability for dynamic experiential memory:
        - New observations/reflections added continuously
        - No retraining needed

        Args:
            x_new: Features for new nodes [num_new_nodes, in_channels]
            edge_index_dict: Edge indices including connections to new nodes

        Returns:
            Embeddings for new nodes [num_new_nodes, out_channels]
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x_new, edge_index_dict)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.in_channels}, "
            f"hidden_channels={self.hidden_channels}, "
            f"out_channels={self.out_channels}, "
            f"num_layers={self.num_layers}, "
            f"edge_types={self.edge_types})"
        )
