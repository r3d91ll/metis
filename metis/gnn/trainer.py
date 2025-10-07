#!/usr/bin/env python3
"""
GraphSAGE Trainer with Ranking Loss
====================================

Training pipeline for multi-relational GraphSAGE with contrastive/triplet loss.

Loss Function:
- Contrastive loss for query-node retrieval task
- Positive pairs: (query, relevant_node) → cosine_sim close to 1
- Negative pairs: (query, irrelevant_node) → cosine_sim close to 0

Optimization:
- Adam optimizer with learning rate scheduling
- Early stopping on validation loss
- Model checkpointing (best + latest)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from torch_geometric.loader import NeighborLoader
    NEIGHBOR_SAMPLING_AVAILABLE = True
except ImportError:
    NEIGHBOR_SAMPLING_AVAILABLE = False
    NeighborLoader = None

from .graphsage_model import MultiRelationalGraphSAGE

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Model architecture
    in_channels: int = 2048
    hidden_channels: int = 1024
    out_channels: int = 512
    num_layers: int = 3
    dropout: float = 0.3

    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5

    # Loss function
    margin: float = 0.5  # Contrastive loss margin (unused with InfoNCE)
    temperature: float = 0.07  # Temperature for InfoNCE loss

    # Mini-batch sampling (for scalability)
    use_sampling: bool = False  # Enable neighbor sampling for large graphs
    num_neighbors: Optional[List[int]] = None  # Neighbors per layer [layer1, layer2, layer3]

    def __post_init__(self):
        """
        Set a default neighbor sampling schedule when sampling is enabled but no neighbor counts were provided.
        
        If `use_sampling` is True and `num_neighbors` is None, assigns `[10, 10, 10]` as the per-layer neighbor counts.
        """
        if self.use_sampling and self.num_neighbors is None:
            # Default: sample 10 neighbors per layer for 3-layer model
            self.num_neighbors = [10, 10, 10]

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4

    # Checkpointing
    checkpoint_dir: Path = Path("models/checkpoints")
    save_best_only: bool = True

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) loss for retrieval.

    Treats the problem as a classification task where each query should
    match its corresponding positive node among all nodes in the batch.

    Benefits over margin-based contrastive loss:
    - Better gradient signal from all negative samples in batch
    - Temperature parameter controls hardness of negatives
    - More stable training with larger batches
    """

    def __init__(self, temperature: float = 0.07):
        """
        Create an InfoNCE loss module that scales cosine similarities by a temperature.
        
        Parameters:
            temperature (float): Positive scalar that scales (divides) cosine similarities before softmax, controlling sharpness of the distribution (smaller values produce sharper distributions).
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query_embeddings: Tensor,
        node_embeddings: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """
        Compute the InfoNCE loss between query and node embeddings.
        
        Args:
            query_embeddings: Tensor of shape [batch_size, embed_dim] containing query vectors.
            node_embeddings: Tensor of shape [batch_size, embed_dim] containing node vectors.
            labels: 1D Tensor of length batch_size where `1` indicates the corresponding query/node pair is a positive match and `0` indicates a negative.
        
        Returns:
            Scalar tensor containing the InfoNCE loss.
        
        Notes:
            This function assumes positive pairs appear on the diagonal of the similarity matrix; if no positives are present it returns a zero-valued tensor that preserves gradient flow.
        """
        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        node_embeddings = F.normalize(node_embeddings, p=2, dim=1)

        # Compute similarity matrix [batch_size, batch_size]
        logits = torch.matmul(query_embeddings, node_embeddings.t()) / self.temperature

        # Create targets: positive pairs should be on diagonal
        # Find indices where labels == 1
        pos_indices = torch.where(labels == 1)[0]

        if len(pos_indices) == 0:
            # No positive samples - return zero loss with gradients
            # Use logits to maintain gradient flow
            return (logits * 0.0).sum()

        # For each positive sample, the target is its position in the batch
        # Use cross-entropy: positive should have highest similarity
        targets = torch.arange(len(logits), device=logits.device)

        # Only compute loss for samples that have positives
        loss = F.cross_entropy(logits[pos_indices], targets[pos_indices])

        return loss


class GraphSAGETrainer:
    """
    Trainer for multi-relational GraphSAGE.
    """

    def __init__(
        self,
        config: TrainingConfig,
        edge_types: Optional[List[str]] = None,
    ):
        """
        Initialize the trainer by constructing the multi-relational GraphSAGE model, a query projection layer, the InfoNCE loss, optimizer, learning-rate scheduler, and default training state; also ensure the checkpoint directory exists.
        
        Parameters:
            config (TrainingConfig): Training and model configuration (architecture, optimization, checkpointing, device, etc.).
            edge_types (Optional[List[str]]): Optional list of edge/relation type names for the multi-relational GraphSAGE model.
        """
        self.config = config
        self.device = torch.device(config.device)

        # Initialize model
        self.model = MultiRelationalGraphSAGE(
            in_channels=config.in_channels,
            hidden_channels=config.hidden_channels,
            out_channels=config.out_channels,
            num_layers=config.num_layers,
            dropout=config.dropout,
            edge_types=edge_types,
        ).to(self.device)

        # Query projection: map query embeddings (2048-dim) to match model output (512-dim)
        self.query_projection = nn.Linear(config.in_channels, config.out_channels).to(self.device)

        # Loss function
        self.criterion = InfoNCELoss(temperature=config.temperature)

        # Optimizer (include both model and query projection)
        self.optimizer = Adam(
            list(self.model.parameters()) + list(self.query_projection.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []

        # Create checkpoint directory
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(
        self,
        data,
        query_embeddings: Tensor,
        node_indices: List[int],
        labels: Tensor,
        train_mask: np.ndarray,
    ) -> float:
        """
        Run a single training epoch using full-graph forward passes and update model parameters.
        
        Parameters:
            data: PyG Data object containing node features and edge_index_dict for the full graph.
            query_embeddings (Tensor): [num_queries, 2048] tensor of query vectors.
            node_indices (List[int]): List mapping each query to its target node index in the graph.
            labels (Tensor): [num_queries] binary labels (1 = positive pair, 0 = negative).
            train_mask (np.ndarray): Boolean mask selecting which queries are used for training.
        
        Returns:
            float: Average training loss across processed batches.
        """
        self.model.train()

        # Filter training samples
        train_idx = np.where(train_mask)[0]
        num_train = len(train_idx)

        # Shuffle training data
        perm = np.random.permutation(num_train)
        train_idx = train_idx[perm]

        total_loss = 0.0
        num_batches = 0

        # Move graph to device once (full-graph training)
        # This is memory-efficient for small/medium graphs
        data_x = data.x.to(self.device)
        data_edge_index = {k: v.to(self.device) for k, v in data.edge_index_dict.items()}

        for start_idx in range(0, num_train, self.config.batch_size):
            end_idx = min(start_idx + self.config.batch_size, num_train)
            batch_idx = train_idx[start_idx:end_idx]

            # Get batch data
            batch_queries = query_embeddings[batch_idx].to(self.device)
            batch_nodes = [node_indices[i] for i in batch_idx]
            batch_labels = labels[batch_idx].to(self.device)

            # Project queries to match node embedding dimension (2048 → 512)
            batch_queries_proj = self.query_projection(batch_queries)

            # Forward pass: compute ALL node embeddings (full-graph)
            # For large graphs, use train_epoch_sampled() instead
            node_embs = self.model(data_x, data_edge_index)

            # Get embeddings for batch nodes
            batch_node_embs = node_embs[batch_nodes]

            # Compute loss
            loss = self.criterion(batch_queries_proj, batch_node_embs, batch_labels.float())

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def train_epoch_sampled(
        self,
        data,
        query_embeddings: Tensor,
        node_indices: List[int],
        labels: Tensor,
        train_mask: np.ndarray,
    ) -> float:
        """
        Perform one training epoch using mini-batch neighbor sampling.
        
        Parameters:
            data: PyG Data object containing the full graph.
            query_embeddings (Tensor): [num_queries, 2048] query embeddings.
            node_indices (List[int]): Target node index for each query.
            labels (Tensor): [num_queries] binary labels (1 = positive, 0 = negative).
            train_mask (np.ndarray): Boolean mask selecting training samples from queries.
        
        Returns:
            float: Average training loss over processed batches.
        
        Raises:
            ImportError: If torch_geometric.loader.NeighborLoader is not available.
        """
        if not NEIGHBOR_SAMPLING_AVAILABLE:
            raise ImportError(
                "NeighborLoader not available. Install torch-geometric or use train_epoch() instead."
            )

        self.model.train()

        # Filter training samples
        train_idx = np.where(train_mask)[0]

        if len(train_idx) == 0:
            return 0.0

        # Get unique node indices we need to train on
        train_nodes = torch.tensor([node_indices[i] for i in train_idx], dtype=torch.long)

        # Create NeighborLoader for sampling subgraphs
        loader = NeighborLoader(
            data,
            num_neighbors=self.config.num_neighbors,
            batch_size=self.config.batch_size,
            input_nodes=train_nodes,
            shuffle=True,
        )

        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            # batch contains sampled subgraph
            batch = batch.to(self.device)

            # Step 1: Identify which training queries have target nodes in this subgraph
            # batch.n_id contains original node IDs in the sampled subgraph
            batch_original_nodes = batch.n_id.cpu().numpy()

            # Find queries whose target nodes are in this sampled subgraph
            # node_indices[i] is the target node for query i
            batch_query_mask = np.isin(node_indices, batch_original_nodes)
            if not batch_query_mask.any():
                continue  # No relevant queries for this subgraph

            # Get the query indices that match this subgraph
            batch_query_idx = np.where(batch_query_mask)[0]
            batch_queries = query_embeddings[batch_query_idx].to(self.device)
            batch_labels = labels[batch_query_idx].to(self.device)

            # Project queries to match node embedding dimension
            batch_queries_proj = self.query_projection(batch_queries)

            # Step 2: Forward pass on sampled subgraph (memory efficient!)
            # Only computes embeddings for nodes in this batch + their neighbors
            node_embs = self.model(batch.x, batch.edge_index_dict)

            # Step 3: Map original node IDs to batch-local indices
            # The sampled subgraph has its own indexing (0, 1, 2, ...)
            # batch.n_id[i] is the original node ID for batch position i
            node_to_batch_idx = {nid.item(): i for i, nid in enumerate(batch.n_id)}

            # Get batch-local indices for the target nodes of our queries
            batch_node_batch_indices = [
                node_to_batch_idx[node_indices[i]]
                for i in batch_query_idx
                # Filter out queries whose target nodes aren't in this subgraph
                # (shouldn't happen given the mask above, but defensive)
                if node_indices[i] in node_to_batch_idx
            ]

            if not batch_node_batch_indices:
                continue  # Safety check: no valid node mappings

            batch_node_embs = node_embs[batch_node_batch_indices]

            # Compute loss
            loss = self.criterion(batch_queries_proj, batch_node_embs, batch_labels.float())

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    @torch.no_grad()
    def validate(
        self,
        data,
        query_embeddings: Tensor,
        node_indices: List[int],
        labels: Tensor,
        val_mask: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate the model on the validation set and compute loss and similarity-based metrics.
        
        Parameters:
            data: Graph data object containing node features (`data.x`) and `edge_index_dict`.
            query_embeddings (Tensor): All query embeddings; used to select validation queries.
            node_indices (List[int]): Mapping from query positions to target node indices in the graph.
            labels (Tensor): Binary labels for query–node pairs (1 for positive, 0 for negative).
            val_mask (np.ndarray): Boolean mask selecting which queries belong to the validation set.
        
        Returns:
            Tuple[float, Dict[str, float]]: `val_loss` (float) computed by the InfoNCE criterion, and `metrics` dictionary with keys
            `"accuracy"`, `"pos_sim_mean"`, and `"neg_sim_mean"` describing cosine-based evaluation on the validation pairs.
        """
        self.model.eval()

        val_idx = np.where(val_mask)[0]

        if len(val_idx) == 0:
            return 0.0, {}

        # Get validation data
        val_queries = query_embeddings[val_idx].to(self.device)
        val_nodes = [node_indices[i] for i in val_idx]
        val_labels = labels[val_idx].to(self.device)

        # Project queries (2048 → 512)
        val_queries_proj = self.query_projection(val_queries)

        # Forward pass (full-graph)
        # Move data once for efficiency
        data_x = data.x.to(self.device)
        data_edge_index = {k: v.to(self.device) for k, v in data.edge_index_dict.items()}
        node_embs = self.model(data_x, data_edge_index)

        val_node_embs = node_embs[val_nodes]

        # Compute loss
        loss = self.criterion(val_queries_proj, val_node_embs, val_labels.float())

        # Compute metrics
        metrics = self._compute_metrics(val_queries_proj, val_node_embs, val_labels)

        return loss.item(), metrics

    def _compute_metrics(
        self,
        query_embs: Tensor,
        node_embs: Tensor,
        labels: Tensor,
    ) -> Dict[str, float]:
        """
        Compute accuracy and mean cosine similarities between query and node embeddings.
        
        Returns:
            metrics (Dict[str, float]): Dictionary with keys:
                - "accuracy": Fraction of labels correctly predicted using a 0.5 cosine-similarity threshold.
                - "pos_sim_mean": Mean cosine similarity for samples with label `1`, `0.0` if none.
                - "neg_sim_mean": Mean cosine similarity for samples with label `0`, `0.0` if none.
        """
        # Normalize embeddings
        query_embs = F.normalize(query_embs, p=2, dim=1)
        node_embs = F.normalize(node_embs, p=2, dim=1)

        # Cosine similarities
        cosine_sim = torch.sum(query_embs * node_embs, dim=1)

        # Accuracy (threshold at 0.5)
        predictions = (cosine_sim > 0.5).long()
        accuracy = (predictions == labels).float().mean().item()

        # Separate positive and negative
        pos_mask = labels == 1
        neg_mask = labels == 0

        pos_sim = cosine_sim[pos_mask].mean().item() if pos_mask.any() else 0.0
        neg_sim = cosine_sim[neg_mask].mean().item() if neg_mask.any() else 0.0

        return {
            "accuracy": accuracy,
            "pos_sim_mean": pos_sim,
            "neg_sim_mean": neg_sim,
        }

    def fit(
        self,
        data,
        query_embeddings: Tensor,
        node_indices: List[int],
        labels: Tensor,
        train_mask: np.ndarray,
        val_mask: np.ndarray,
    ):
        """
        Run the full training loop for the GraphSAGE model, performing epoch-wise training, validation, learning-rate scheduling, checkpointing, and early stopping.
        
        This method updates the trainer's internal state (current_epoch, train_losses, val_losses, best_val_loss, patience_counter) and saves checkpoints according to the configuration.
        
        Parameters:
            data: PyTorch Geometric Data object containing the graph (node features, edge index/types, etc.).
            query_embeddings (Tensor): Per-query feature embeddings (one row per query).
            node_indices (List[int]): Target node index for each query in query_embeddings.
            labels (Tensor): Binary labels for each query–node pair (1 for positive, 0 for negative).
            train_mask (np.ndarray): Boolean or binary mask selecting training samples from the queries.
            val_mask (np.ndarray): Boolean or binary mask selecting validation samples from the queries.
        """
        logger.info("Starting GraphSAGE training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model: {self.model}")
        logger.info(f"Train samples: {train_mask.sum()}, Val samples: {val_mask.sum()}")

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch(
                data, query_embeddings, node_indices, labels, train_mask
            )
            self.train_losses.append(train_loss)

            # Validate
            val_loss, metrics = self.validate(
                data, query_embeddings, node_indices, labels, val_mask
            )
            self.val_losses.append(val_loss)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Acc: {metrics.get('accuracy', 0):.3f} | "
                f"Pos Sim: {metrics.get('pos_sim_mean', 0):.3f} | "
                f"Neg Sim: {metrics.get('neg_sim_mean', 0):.3f}"
            )

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if is_best or not self.config.save_best_only:
                self.save_checkpoint(is_best=is_best)

            # Early stopping
            if self.patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        logger.info("Training complete!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

    def save_checkpoint(self, is_best: bool = False):
        """
        Save the trainer state and model weights to the configured checkpoint directory.
        
        Saves a checkpoint containing training state (current epoch, best validation loss, training/validation loss histories), optimizer and scheduler states, model and query projection parameters, the trainer config and architecture fields, and model edge types. Writes the checkpoint to "latest.pt" and, if `is_best` is True, also writes a copy to "best.pt" and logs the saved best checkpoint.
        
        Parameters:
            is_best (bool): If True, also persist a copy as the best-known checkpoint ("best.pt").
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "query_projection_state_dict": self.query_projection.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "config": self.config.__dict__,
            "in_channels": self.config.in_channels,
            "hidden_channels": self.config.hidden_channels,
            "out_channels": self.config.out_channels,
            "num_layers": self.config.num_layers,
            "dropout": self.config.dropout,
            "edge_types": self.model.edge_types,
        }

        # Save latest
        latest_path = self.config.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = self.config.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint: {best_path}")

    @classmethod
    def load_checkpoint(cls, checkpoint_path: Path, config: Optional[TrainingConfig] = None):
        """
        Reconstructs a GraphSAGETrainer instance from a saved checkpoint file.
        
        If no config is provided, the trainer configuration is reconstructed from the checkpoint. The checkpointed model weights, optional query projection weights, optimizer state, scheduler state, current epoch, best validation loss, and training/validation loss histories are restored onto the new trainer instance.
        
        Parameters:
            checkpoint_path (Path): Path to the checkpoint file produced by save_checkpoint.
            config (Optional[TrainingConfig]): Optional TrainingConfig to override the saved config; if omitted, the config embedded in the checkpoint is used.
        
        Returns:
            trainer (GraphSAGETrainer): A trainer instance restored to the checkpointed state.
        
        Security:
            Only load checkpoints from trusted sources because loading arbitrary files can execute code or load untrusted data.
        """
        checkpoint = torch.load(checkpoint_path, weights_only=True)

        if config is None:
            config = TrainingConfig(**checkpoint["config"])

        trainer = cls(config, edge_types=checkpoint["edge_types"])
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        if "query_projection_state_dict" in checkpoint:
            trainer.query_projection.load_state_dict(checkpoint["query_projection_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        trainer.current_epoch = checkpoint["epoch"]
        trainer.best_val_loss = checkpoint["best_val_loss"]
        trainer.train_losses = checkpoint["train_losses"]
        trainer.val_losses = checkpoint["val_losses"]

        return trainer