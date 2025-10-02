"""
GraphSAGE GNN Module
====================

Graph Neural Network implementation for inductive node embeddings on heterogeneous graphs.

Architecture:
- Input: Jina v4 embeddings (2048-dim) for rich semantic space
- Multi-relational GraphSAGE for heterogeneous graphs
- Hidden: 1024-dim, Output: 512-dim for fast retrieval
- Inductive learning: works on unseen nodes without retraining
- Edge types: imports, contains, references, cites, authored_by, part_of

Features:
- Multi-relational aggregation (different weights per edge type)
- Contrastive learning for query-node similarity
- Fast inference (<50ms per batch)
- Works with PyTorch Geometric for graph operations
- Supports dynamic graph updates without retraining

Usage:
    from metis.gnn import MultiRelationalGraphSAGE, GraphSAGEInference

    # Create model
    model = MultiRelationalGraphSAGE(
        in_channels=2048,
        hidden_channels=1024,
        out_channels=512
    )

    # Inference
    inference = GraphSAGEInference(model, device='cuda')
    candidates = inference.retrieve_candidates(query_embedding, k=50)
"""

from .graphsage_model import MultiRelationalGraphSAGE  # noqa: F401
from .inference import GraphSAGEInference  # noqa: F401

__all__ = [
    "MultiRelationalGraphSAGE",
    "GraphSAGEInference",
]
