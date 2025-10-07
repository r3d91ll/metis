#!/usr/bin/env python3
"""
Graph Pipeline - Export ArangoDB to PyTorch Geometric
======================================================

Exports arXiv graph from ArangoDB to PyG Data format for GNN training.

Pipeline:
1. Connect to ArangoDB via Unix socket
2. Extract nodes (papers with embeddings) and edges (category_links, temporal_succession)
3. Build PyG Data object with multi-relational edge types
4. Create train/val split
5. Save to disk

Output: PyG Data object ready for GraphSAGE training

Usage:
    python graph_pipeline.py
    python graph_pipeline.py --output models/custom_graph.pt
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import yaml
import argparse

import torch
import numpy as np

# Add metis to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from metis.database import resolve_client_config
from metis.gnn import GraphBuilder

try:
    from metis.database.memory import ArangoMemoryClient
except ImportError:
    # Fallback if memory client not available
    from metis.database import ArangoClient as ArangoMemoryClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ArxivGraphPipeline:
    """
    Export arXiv corpus from ArangoDB to PyG Data format.

    Uses metis.gnn.GraphBuilder for extraction.
    """

    def __init__(self, config_path: Path):
        """Initialize pipeline with configuration."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {config_path}")

        # Database configuration
        socket_path = self.config['database']['socket_path']
        self.db_config = resolve_client_config(
            database=self.config['database']['name'],
            socket_path=socket_path,
            use_proxies=False
        )

        self.graph_config = self.config.get('graph', {})

    def export_graph(
        self,
        output_path: Optional[Path] = None,
        val_ratio: float = 0.2,
        seed: int = 42
    ) -> Path:
        """
        Export graph from ArangoDB to PyG Data.

        Args:
            output_path: Path to save graph (default from config)
            val_ratio: Fraction of nodes for validation split
            seed: Random seed for reproducibility

        Returns:
            Path to saved graph file
        """
        if output_path is None:
            output_path = Path(self.graph_config.get('output_path', 'models/arxiv_graph.pt'))

        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Exporting arXiv Graph to PyG Data")
        logger.info("=" * 60)

        with ArangoMemoryClient(self.db_config) as client:
            # Initialize GraphBuilder
            builder = GraphBuilder(client)

            # Build graph
            logger.info("Building graph from ArangoDB...")
            node_collection = self.config['database']['collections']['embeddings']

            data, node_id_map = builder.build_graph(
                include_collections=[node_collection],
                edge_types=None  # Use all available edge types
            )

            logger.info("Graph structure:")
            logger.info(f"  Nodes: {data.x.shape[0]:,}")
            logger.info(f"  Node features: {data.x.shape[1]}-dim")
            logger.info(f"  Edge types: {list(data.edge_index_dict.keys())}")
            for edge_type, edge_index in data.edge_index_dict.items():
                logger.info(f"    {edge_type}: {edge_index.shape[1]:,} edges")

            # Create train/val split
            logger.info(f"Creating train/val split (val_ratio={val_ratio})...")
            train_mask, val_mask = builder.create_train_val_split(
                num_nodes=data.x.shape[0],
                val_ratio=val_ratio,
                seed=seed
            )

            # Add masks to data object
            data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
            data.val_mask = torch.tensor(val_mask, dtype=torch.bool)

            # Add node_id_map for later retrieval
            data.node_id_map = node_id_map

            logger.info(f"  Train nodes: {train_mask.sum():,}")
            logger.info(f"  Val nodes: {val_mask.sum():,}")

            # Save to disk
            logger.info(f"Saving graph to {output_path}...")
            torch.save({
                'data': data,
                'node_id_map': node_id_map,
                'config': self.config,
            }, output_path)

            # Report file size
            file_size_gb = output_path.stat().st_size / (1024**3)
            logger.info(f"Graph saved: {file_size_gb:.2f} GB")

        logger.info("=" * 60)
        logger.info("Graph Export Complete")
        logger.info("=" * 60)

        return output_path


def main():
    parser = argparse.ArgumentParser(description='Export arXiv graph to PyG Data format')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path(__file__).parent / 'config' / 'arxiv_import.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output path for graph file (overrides config)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='Validation set ratio (default: 0.2)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    pipeline = ArxivGraphPipeline(args.config)
    output_path = pipeline.export_graph(
        output_path=args.output,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    logger.info(f"Graph saved to: {output_path}")


if __name__ == "__main__":
    main()
