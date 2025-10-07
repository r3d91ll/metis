#!/usr/bin/env python3
"""
GNN Training Pipeline - GraphSAGE for arXiv Corpus
===================================================

Trains multi-relational GraphSAGE on arXiv graph.

Training Strategy:
- Self-supervised learning from graph structure
- Positive pairs: Papers connected by edges (any edge type)
- Negative pairs: Random unconnected papers
- InfoNCE loss for contrastive learning

Model:
- Input: 2048-dim Jina embeddings (semantic WHAT)
- Graph: Multi-relational edges (structural WHERE)
- Output: 512-dim embeddings (semantic + structural)

Usage:
    python train_gnn.py
    python train_gnn.py --graph models/arxiv_graph.pt --epochs 100
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import argparse

import torch
import torch.nn.functional as F
import numpy as np

# Add metis to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from metis.gnn import GraphSAGETrainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SelfSupervisedDataGenerator:
    """
    Generate self-supervised training data from graph structure.

    Strategy:
    - Positive samples: Papers connected by edges (similar via structure)
    - Negative samples: Random unconnected papers (likely dissimilar)
    """

    def __init__(self, data, seed: int = 42):
        """
        Initialize the data generator, RNG, and an undirected adjacency index from a PyG Data object.
        
        Parameters:
            data: PyG `Data` object containing node features (`x`) and graph edge indices used to build the adjacency index.
            seed (int): Seed for the internal random number generator.
        """
        self.data = data
        self.num_nodes = data.x.shape[0]
        self.rng = np.random.RandomState(seed)

        # Build adjacency set for fast membership testing
        logger.info("Building adjacency index...")
        self.adjacency = self._build_adjacency()
        logger.info(f"Adjacency built: {len(self.adjacency)} nodes have neighbors")

    def _build_adjacency(self) -> Dict[int, set]:
        """
        Builds a neighbor adjacency mapping from the graph's edge indices across all edge types, treating edges as undirected.
        
        Returns:
            adj (Dict[int, set]): Mapping from node index to a set of neighboring node indices aggregated from all edge types.
        """
        adj = {}

        for edge_type, edge_index in self.data.edge_index_dict.items():
            sources = edge_index[0].numpy()
            targets = edge_index[1].numpy()

            for src, tgt in zip(sources, targets):
                if src not in adj:
                    adj[src] = set()
                if tgt not in adj:
                    adj[tgt] = set()

                adj[src].add(tgt)
                adj[tgt].add(src)  # Treat all edges as undirected for sampling

        return adj

    def generate_training_pairs(
        self,
        num_samples: int,
        pos_neg_ratio: float = 1.0
    ) -> Tuple[torch.Tensor, List[int], torch.Tensor]:
        """
        Generate a batch of node pairs labeled as positive (connected) or negative (unconnected) for self-supervised contrastive training.
        
        Parameters:
            num_samples (int): Total number of pairs to generate.
            pos_neg_ratio (float): Ratio of positive to negative samples (positive_count / negative_count).
        
        Returns:
            tuple: A 3-tuple (query_embeddings, node_indices, labels)
                - query_embeddings (torch.Tensor): Tensor of shape [N, F] with embeddings for the query nodes.
                - node_indices (List[int]): List of length N with target node indices for each pair.
                - labels (torch.Tensor): 1 for positive (connected) pairs, 0 for negative (unconnected) pairs.
        """
        num_pos = int(num_samples * pos_neg_ratio / (1 + pos_neg_ratio))
        num_neg = num_samples - num_pos

        query_indices = []
        node_indices = []
        labels = []

        # Generate positive samples (connected pairs)
        for _ in range(num_pos):
            # Sample a random node with neighbors
            attempts = 0
            while attempts < 100:
                query_idx = self.rng.randint(0, self.num_nodes)
                if query_idx in self.adjacency and len(self.adjacency[query_idx]) > 0:
                    # Sample a random neighbor
                    neighbor_idx = self.rng.choice(list(self.adjacency[query_idx]))
                    query_indices.append(query_idx)
                    node_indices.append(neighbor_idx)
                    labels.append(1)  # Positive
                    break
                attempts += 1

        # Generate negative samples (random unconnected pairs)
        for _ in range(num_neg):
            attempts = 0
            while attempts < 100:
                query_idx = self.rng.randint(0, self.num_nodes)
                node_idx = self.rng.randint(0, self.num_nodes)

                # Check that they're not connected (and not the same node)
                if query_idx != node_idx:
                    if query_idx not in self.adjacency or node_idx not in self.adjacency[query_idx]:
                        query_indices.append(query_idx)
                        node_indices.append(node_idx)
                        labels.append(0)  # Negative
                        break
                attempts += 1

        # Convert to tensors
        # Query embeddings are just the node embeddings at query indices
        query_embeddings = self.data.x[query_indices]
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        logger.info(f"Generated {len(labels)} training pairs ({num_pos} positive, {num_neg} negative)")

        return query_embeddings, node_indices, labels_tensor


def load_graph(graph_path: Path):
    """
    Load a saved PyG graph checkpoint from disk.
    
    Parameters:
        graph_path (Path): Path to a torch checkpoint file produced by the graph export; the checkpoint must contain keys `'data'` (a PyG Data object) and `'node_id_map'` (a mapping of original node identifiers to node indices).
    
    Returns:
        tuple: `(data, node_id_map)` where `data` is the loaded PyG `Data` object and `node_id_map` is a dictionary mapping original node IDs to graph node indices.
    """
    logger.info(f"Loading graph from {graph_path}...")
    checkpoint = torch.load(graph_path, weights_only=False)

    data = checkpoint['data']
    node_id_map = checkpoint['node_id_map']

    logger.info(f"Graph loaded:")
    logger.info(f"  Nodes: {data.x.shape[0]:,}")
    logger.info(f"  Edge types: {list(data.edge_index_dict.keys())}")

    return data, node_id_map


def train_gnn(config_path: Path, graph_path: Optional[Path] = None, epochs: Optional[int] = None):
    """
    Run GraphSAGE training on the arXiv graph using settings from a YAML configuration.
    
    Loads the graph and configuration, constructs a training configuration, generates self‑supervised training pairs, and runs the GraphSAGE training loop while logging progress and saving checkpoints.
    
    Parameters:
        config_path (Path): Path to the YAML configuration file that defines architecture, training, sampling, and checkpoint settings.
        graph_path (Optional[Path]): Optional path to a saved graph checkpoint; when provided, this overrides the graph path specified in the configuration.
        epochs (Optional[int]): Optional override for the number of training epochs defined in the configuration.
    """
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    gnn_config = config.get('gnn', {})

    # Determine graph path
    if graph_path is None:
        graph_path = Path(config['graph']['output_path'])

    # Load graph
    data, node_id_map = load_graph(graph_path)

    # Create training configuration
    arch_config = gnn_config.get('architecture', {})
    train_config_dict = gnn_config.get('training', {})
    sampling_config = gnn_config.get('sampling', {})
    checkpoint_config = gnn_config.get('checkpoints', {})

    training_config = TrainingConfig(
        in_channels=arch_config.get('in_channels', 2048),
        hidden_channels=arch_config.get('hidden_channels', 1024),
        out_channels=arch_config.get('out_channels', 512),
        num_layers=arch_config.get('num_layers', 3),
        dropout=arch_config.get('dropout', 0.3),
        epochs=epochs or train_config_dict.get('epochs', 100),
        batch_size=train_config_dict.get('batch_size', 512),
        learning_rate=train_config_dict.get('learning_rate', 0.001),
        weight_decay=train_config_dict.get('weight_decay', 1e-5),
        temperature=train_config_dict.get('temperature', 0.07),
        use_sampling=sampling_config.get('use_neighbor_sampling', True),
        num_neighbors=sampling_config.get('num_neighbors', [10, 10, 10]),
        patience=train_config_dict.get('patience', 15),
        checkpoint_dir=Path(checkpoint_config.get('dir', 'models/arxiv_checkpoints')),
        save_best_only=checkpoint_config.get('save_best_only', True),
    )

    logger.info("=" * 60)
    logger.info("Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Architecture: {training_config.in_channels} → {training_config.hidden_channels} → {training_config.out_channels}")
    logger.info(f"Layers: {training_config.num_layers}")
    logger.info(f"Dropout: {training_config.dropout}")
    logger.info(f"Epochs: {training_config.epochs}")
    logger.info(f"Batch size: {training_config.batch_size}")
    logger.info(f"Learning rate: {training_config.learning_rate}")
    logger.info(f"Neighbor sampling: {training_config.num_neighbors}")
    logger.info("=" * 60)

    # Generate self-supervised training data
    logger.info("Generating self-supervised training data...")
    data_generator = SelfSupervisedDataGenerator(data, seed=42)

    # Generate training samples (10x the number of nodes for sufficient diversity)
    num_samples = min(data.x.shape[0] * 10, 1_000_000)  # Cap at 1M samples
    query_embeddings, node_indices, labels = data_generator.generate_training_pairs(
        num_samples=num_samples,
        pos_neg_ratio=1.0  # Equal positive and negative samples
    )

    # Split into train/val based on node masks
    train_mask = data.train_mask.numpy()
    val_mask = data.val_mask.numpy()

    # Filter samples based on whether query node is in train/val set
    train_samples_mask = np.array([train_mask[i] for i in range(len(query_embeddings))
                                    if i < len(train_mask)], dtype=bool)
    val_samples_mask = np.array([val_mask[i] for i in range(len(query_embeddings))
                                  if i < len(val_mask)], dtype=bool)

    # For simplicity, just use the generated samples with the node masks
    # More sophisticated: re-generate samples specifically for train/val nodes
    train_sample_mask = train_mask
    val_sample_mask = val_mask

    # Initialize trainer
    edge_types = list(data.edge_index_dict.keys())
    logger.info(f"Training with edge types: {edge_types}")

    trainer = GraphSAGETrainer(
        config=training_config,
        edge_types=edge_types
    )

    # Train model
    logger.info("Starting training...")
    trainer.fit(
        data=data,
        query_embeddings=query_embeddings,
        node_indices=node_indices,
        labels=labels,
        train_mask=train_sample_mask,
        val_mask=val_sample_mask
    )

    logger.info("=" * 60)
    logger.info("Training Complete")
    logger.info("=" * 60)
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Checkpoints saved to: {training_config.checkpoint_dir}")


def main():
    """
    CLI entry point that parses command-line options and launches GraphSAGE training.
    
    Parses the following arguments from sys.argv and passes them to train_gnn:
    - --config: path to the YAML configuration file (defaults to config/arxiv_import.yaml next to this script)
    - --graph: optional path to a graph file that overrides the config
    - --epochs: optional integer to override the configured number of training epochs
    
    On uncaught exceptions, logs the error with a stack trace and exits the process with code 1.
    """
    parser = argparse.ArgumentParser(description='Train GraphSAGE on arXiv graph')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path(__file__).parent / 'config' / 'arxiv_import.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--graph',
        type=Path,
        default=None,
        help='Path to graph file (overrides config)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )

    args = parser.parse_args()

    try:
        train_gnn(
            config_path=args.config,
            graph_path=args.graph,
            epochs=args.epochs
        )
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()