"""
Metis Snapshot Manager

Provides snapshot/backup capabilities for ArangoDB and bulk storage files.
"""

from .snapshot_manager import SnapshotManager, SnapshotMetadata

__all__ = ["SnapshotManager", "SnapshotMetadata"]
