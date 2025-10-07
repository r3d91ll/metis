#!/usr/bin/env python3
"""
Graph Builder - ArangoDB to PyTorch Geometric
==============================================

Exports knowledge graphs from ArangoDB to PyG Data format for GraphSAGE training.

Collections:
- repo_docs: Code and text files with Jina v4 embeddings (2048-dim)
- papers: Academic papers and documents
- code_edges: imports, contains, references edges
- citations: cites, authored_by edges
- directories: Directory nodes for hierarchical structure

Output: PyG Data object ready for GraphSAGE training
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    Data = None

from metis.database import ArangoClient

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Build PyG Data object from ArangoDB graph.
    """

    def __init__(self, client: ArangoClient):
        """
        Create a GraphBuilder bound to the given ArangoDB client.
        
        Parameters:
            client (ArangoClient): Read-only ArangoDB client used to query node and edge collections.
        
        Raises:
            ImportError: If PyTorch Geometric is not installed.
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "PyTorch Geometric not installed. "
                "Install with: poetry install --extras gnn"
            )

        self.client = client

    def build_graph(
        self,
        include_collections: Optional[List[str]] = None,
        edge_types: Optional[List[str]] = None,
    ) -> Tuple[Data, Dict[int, str]]:
        """
        Assemble a PyTorch Geometric Data object from ArangoDB node and edge collections.
        
        Parameters:
            include_collections (Optional[List[str]]): Node collections to include; defaults to ["repo_docs"] when not provided.
            edge_types (Optional[List[str]]): Edge collection types to include; defaults to all available edge types when not provided.
        
        Returns:
            Tuple[Data, Dict[int, str]]: 
                - data: PyG Data object with fields `x` (node feature tensor), `node_metadata` (list of per-node metadata dicts), `edge_index_dict` (mapping edge type -> edge index tensor), and `edge_attr_dict` (mapping edge type -> edge attribute tensor).
                - node_id_map: Mapping from local node index to the corresponding ArangoDB document _id.
        """
        include_collections = include_collections or ["repo_docs"]

        logger.info(f"Building graph from collections: {include_collections}")

        # Step 1: Extract nodes
        nodes, node_id_map = self._extract_nodes(include_collections)

        # Step 2: Extract edges
        edge_index_dict, edge_attr_dict = self._extract_edges(node_id_map, edge_types)

        # Step 3: Create PyG Data object
        data = Data(
            x=nodes["features"],  # [num_nodes, 2048]
            node_metadata=nodes["metadata"],  # List of dicts
            edge_index_dict=edge_index_dict,  # Dict[edge_type, [2, num_edges]]
            edge_attr_dict=edge_attr_dict,  # Dict[edge_type, [num_edges, feat_dim]]
        )

        logger.info(
            f"Graph built: {data.x.shape[0]} nodes, "
            f"{sum(e.shape[1] for e in edge_index_dict.values())} edges"
        )

        return data, node_id_map

    def _extract_nodes(
        self,
        collections: List[str],
    ) -> Tuple[Dict, Dict[int, str]]:
        """
        Load node embeddings and metadata from the specified ArangoDB collections.
        
        Parameters:
            collections (List[str]): Names of ArangoDB document collections to scan for documents containing an embedding.
        
        Returns:
            nodes_dict (Dict): Dictionary with:
                - "features": torch.FloatTensor of shape [num_nodes, 2048] containing node embeddings.
                - "metadata": List[dict] where each dict contains keys `arango_id`, `key`, `path`, `arxiv_id`, and `collection`.
            node_id_map (Dict[int, str]): Mapping from local node index to the ArangoDB document `_id`.
        
        Raises:
            ValueError: If no documents with a valid 2048-dimensional embedding are found.
        """
        features_list = []
        metadata_list = []
        node_id_map = {}
        node_idx = 0

        for collection in collections:
            # Try different embedding field names (code repos vs arXiv papers)
            # First try combined_embedding (arXiv), then embedding (code repos)
            aql = f"""
            FOR doc IN {collection}
              LET emb = doc.combined_embedding != null ? doc.combined_embedding : doc.embedding
              FILTER emb != null
              RETURN {{
                _id: doc._id,
                _key: doc._key,
                path: doc.path,
                arxiv_id: doc.arxiv_id,
                embedding: emb,
                metadata: doc.metadata
              }}
            """

            docs = self.client.execute_query(aql, {})

            for doc in docs:
                # Extract embedding
                embedding = doc.get("embedding")
                if not embedding or len(embedding) != 2048:
                    logger.warning(f"Skipping {doc['_id']}: invalid embedding (expected 2048-dim, got {len(embedding) if embedding else 0})")
                    continue

                features_list.append(embedding)

                # Store metadata (handle both code repos and arXiv papers)
                metadata_list.append({
                    "arango_id": doc["_id"],
                    "key": doc["_key"],
                    "path": doc.get("path", ""),
                    "arxiv_id": doc.get("arxiv_id", ""),
                    "collection": collection,
                })

                # Map node index → ArangoDB ID
                node_id_map[node_idx] = doc["_id"]
                node_idx += 1

        if not features_list:
            raise ValueError("No nodes with embeddings found!")

        # Convert to tensor
        features_tensor = torch.tensor(features_list, dtype=torch.float32)

        logger.info(f"Extracted {len(features_list)} nodes from {collections}")

        return {
            "features": features_tensor,
            "metadata": metadata_list,
        }, node_id_map

    def _extract_edges(
        self,
        node_id_map: Dict[int, str],
        edge_types: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Build per-edge-type PyG edge index and edge attribute tensors from ArangoDB edge collections.
        
        Queries the specified edge collections (or a set of common defaults) and groups edges by their reported `type`. Edges whose endpoints are not present in `node_id_map` are skipped. Each returned edge attribute is the edge's `weight` (defaults to 1.0 when missing).
        
        Parameters:
            node_id_map (Dict[int, str]): Mapping from local node index to ArangoDB document _id.
            edge_types (Optional[List[str]]): Edge collection names to query; if None, common edge collection names are attempted.
        
        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
                - edge_index_dict: mapping edge_type -> tensor of shape [2, num_edges] containing source and target node indices.
                - edge_attr_dict: mapping edge_type -> tensor of shape [num_edges, 1] containing edge weights as float32.
        """
        # Reverse map: arango_id → node_idx
        id_to_idx = {aid: idx for idx, aid in node_id_map.items()}

        edge_index_dict = {}
        edge_attr_dict = {}

        # Default edge collection names (try common patterns)
        if edge_types is None:
            edge_types = [
                "code_edges",           # HADES code repositories
                "category_links",       # arXiv category co-occurrence
                "temporal_succession",  # arXiv temporal edges
                "citations",            # Citation networks
            ]

        # Query edges from each collection (use batching to avoid chunked encoding issues)
        all_edges = []
        batch_size = 1000  # Process edges in batches

        for edge_collection in edge_types:
            try:
                # First, count edges
                count_result = self.client.execute_query(
                    f"RETURN LENGTH({edge_collection})", {}
                )
                total_edges = count_result[0] if count_result else 0

                if total_edges == 0:
                    continue

                logger.warning(f"Found {total_edges} edges in {edge_collection}, loading in batches...")

                # Load edges in batches using LIMIT/SKIP
                for offset in range(0, total_edges, batch_size):
                    aql = f"""
                    FOR edge IN {edge_collection}
                      LIMIT {offset}, {batch_size}
                      RETURN {{
                        _from: edge._from,
                        _to: edge._to,
                        type: edge.type,
                        weight: edge.weight,
                        collection: "{edge_collection}"
                      }}
                    """
                    edges = self.client.execute_query(aql, {})
                    all_edges.extend(edges)

                logger.warning(f"Loaded {total_edges} edges from {edge_collection}")
            except Exception as e:
                logger.warning(f"Skipping collection {edge_collection}: {e}")

        logger.warning(f"Total edges loaded from all collections: {len(all_edges)}")

        # Group by edge type
        edges_by_type = {}
        skipped_count = 0

        for edge in all_edges:
            edge_type = edge.get("type", "unknown")

            from_id = edge["_from"]
            to_id = edge["_to"]

            # Skip if nodes not in graph
            if from_id not in id_to_idx or to_id not in id_to_idx:
                skipped_count += 1
                continue

            from_idx = id_to_idx[from_id]
            to_idx = id_to_idx[to_id]
            weight = edge.get("weight", 1.0)

            if edge_type not in edges_by_type:
                edges_by_type[edge_type] = []

            edges_by_type[edge_type].append((from_idx, to_idx, weight))

        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} edges (nodes not in graph)")
        if edges_by_type:
            logger.info(f"Grouped edges by type: {[(k, len(v)) for k, v in edges_by_type.items()]}")
        else:
            logger.warning(f"No edges matched after grouping! All edges loaded: {len(all_edges)}, All skipped: {skipped_count}")

        # Convert to PyG format
        for edge_type, edge_list in edges_by_type.items():
            if not edge_list:
                continue

            # Separate source, target, weights
            sources = [e[0] for e in edge_list]
            targets = [e[1] for e in edge_list]
            weights = [e[2] for e in edge_list]

            # Create edge_index [2, num_edges]
            edge_index = torch.tensor([sources, targets], dtype=torch.long)

            # Create edge_attr [num_edges, 1]
            edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)

            edge_index_dict[edge_type] = edge_index
            edge_attr_dict[edge_type] = edge_attr

            logger.info(f"Edge type '{edge_type}': {len(edge_list)} edges")

        if not edge_index_dict:
            logger.warning("No edges found in graph!")

        return edge_index_dict, edge_attr_dict

    def create_train_val_split(
        self,
        num_nodes: int,
        val_ratio: float = 0.2,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create an inductive train/validation split by selecting a subset of node indices for validation.
        
        Parameters:
        	num_nodes (int): Total number of nodes to split.
        	val_ratio (float): Fraction of nodes assigned to validation (between 0 and 1).
        	seed (int): Random seed for reproducible shuffling.
        
        Returns:
        	train_mask (np.ndarray): Boolean array of length `num_nodes` with `True` for training nodes.
        	val_mask (np.ndarray): Boolean array of length `num_nodes` with `True` for validation nodes.
        """
        np.random.seed(seed)

        indices = np.arange(num_nodes)
        np.random.shuffle(indices)

        val_size = int(num_nodes * val_ratio)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        train_mask = np.zeros(num_nodes, dtype=bool)
        val_mask = np.zeros(num_nodes, dtype=bool)

        train_mask[train_indices] = True
        val_mask[val_indices] = True

        logger.info(f"Split: {len(train_indices)} train, {len(val_indices)} val")

        return train_mask, val_mask