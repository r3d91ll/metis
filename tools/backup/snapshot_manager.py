"""
Snapshot Manager for ArangoDB and Bulk Storage

Provides reliable snapshot/restore functionality for safe experimentation.
"""

import json
import logging
import shutil
import subprocess
import tarfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import os

logger = logging.getLogger(__name__)


@dataclass
class SnapshotMetadata:
    """Metadata for a database/storage snapshot."""

    snapshot_name: str
    timestamp: str  # ISO format
    description: str
    git_commit: Optional[str]
    git_branch: Optional[str]
    permanent: bool
    collections: Dict[str, int]  # collection_name -> document_count
    bulk_store_size_mb: float
    database_size_mb: float
    compression: Optional[str]  # None, "gzip", "tar.gz"

    @classmethod
    def from_dict(cls, data: Dict) -> "SnapshotMetadata":
        """Create metadata from dictionary."""
        return cls(**data)

    def to_dict(self) -> Dict:
        """Convert metadata to dictionary."""
        return asdict(self)


class SnapshotError(Exception):
    """Base exception for snapshot operations."""

    pass


class SnapshotManager:
    """
    Manages snapshots of ArangoDB database and bulk storage files.

    Provides create, restore, list, and delete operations with automatic
    retention policy management.
    """

    def __init__(
        self,
        db_name: str = "arxiv_datastore",
        db_endpoint: str = "http://127.0.0.1:8529",
        db_username: str = "root",
        db_password: str | None = None,
        snapshot_root: Path = Path("/bulk-store/metis/snapshots"),
        bulk_store_root: Path = Path("/bulk-store/metis"),
    ):
        """
        Initialize snapshot manager.

        Args:
            db_name: ArangoDB database name
            db_endpoint: ArangoDB endpoint (uses TCP for backups)
            db_username: Database username
            db_password: Database password (defaults to ARANGO_PASSWORD env var)
            snapshot_root: Directory for storing snapshots
            bulk_store_root: Root directory of bulk storage to backup

        Note:
            Uses TCP connection (http://127.0.0.1:8529) instead of Unix socket for backups
            because arangodump is not yet supported by the Go security proxies (issue #5).
            This is acceptable for backups since they're not time-sensitive operations.
        """
        self.db_name = db_name
        self.db_endpoint = db_endpoint  # TCP for backups only
        self.db_username = db_username
        self.db_password = db_password or os.getenv("ARANGO_PASSWORD", "")
        self.snapshot_root = Path(snapshot_root)
        self.bulk_store_root = Path(bulk_store_root)

        # Ensure snapshot directory exists
        self.snapshot_root.mkdir(parents=True, exist_ok=True)

        logger.info(f"SnapshotManager initialized: {self.snapshot_root}")

    def create_snapshot(
        self,
        description: str,
        permanent: bool = False,
        include_bulk: bool = False,  # Default False - bulk storage is already the archive
    ) -> str:
        """
        Create new snapshot of database.

        Note: Bulk storage (/bulk-store/metis/experiments) is already the archive
        and doesn't need to be backed up. Use include_bulk=True only for special cases.

        Args:
            description: Human-readable description
            permanent: If True, won't be auto-deleted by retention policy
            include_bulk: If True, backup bulk storage files (rarely needed)

        Returns:
            Snapshot name (identifier for restoration)

        Raises:
            SnapshotError: If snapshot creation fails
        """
        logger.info(f"Creating snapshot: {description}")

        # Generate snapshot name
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        safe_desc = description.replace(" ", "_").replace("/", "_").lower()[:40]
        snapshot_name = f"metis_{timestamp_str}_{safe_desc}"

        # Create snapshot directory
        snapshot_path = self.snapshot_root / snapshot_name
        try:
            snapshot_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            raise SnapshotError(f"Snapshot {snapshot_name} already exists")

        try:
            # Get git information
            git_commit, git_branch = self._get_git_info()

            # Backup database
            logger.info("Backing up database...")
            db_path = snapshot_path / "database"
            collections = self._backup_database(db_path)
            db_size_mb = self._calculate_directory_size(db_path)

            # Backup bulk storage if requested
            bulk_size_mb = 0.0
            if include_bulk:
                logger.info("Backing up bulk storage...")
                bulk_path = snapshot_path / "bulk_store"
                self._backup_bulk_storage(bulk_path)
                bulk_size_mb = self._calculate_directory_size(bulk_path)

            # Create metadata
            metadata = SnapshotMetadata(
                snapshot_name=snapshot_name,
                timestamp=timestamp.isoformat(),
                description=description,
                git_commit=git_commit,
                git_branch=git_branch,
                permanent=permanent,
                collections=collections,
                bulk_store_size_mb=bulk_size_mb,
                database_size_mb=db_size_mb,
                compression=None,
            )

            # Save metadata
            metadata_path = snapshot_path / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata.to_dict(), f, indent=2)

            logger.info(
                f"✓ Snapshot created: {snapshot_name} "
                f"(DB: {db_size_mb:.1f}MB, Bulk: {bulk_size_mb:.1f}MB)"
            )

            # Apply retention policy (cleanup old snapshots)
            if not permanent:
                self.apply_retention_policy()

            return snapshot_name

        except Exception as e:
            # Cleanup partial snapshot on failure
            logger.error(f"Snapshot creation failed: {e}")
            if snapshot_path.exists():
                shutil.rmtree(snapshot_path)
            raise SnapshotError(f"Failed to create snapshot: {e}") from e

    def list_snapshots(self, verbose: bool = False) -> List[SnapshotMetadata]:
        """
        List all available snapshots.

        Args:
            verbose: If True, include detailed information

        Returns:
            List of snapshot metadata, sorted by timestamp (newest first)
        """
        snapshots = []

        for snapshot_dir in self.snapshot_root.iterdir():
            if not snapshot_dir.is_dir():
                continue

            metadata_path = snapshot_dir / "metadata.json"
            if not metadata_path.exists():
                logger.warning(f"No metadata found for {snapshot_dir.name}")
                continue

            try:
                with open(metadata_path) as f:
                    data = json.load(f)
                metadata = SnapshotMetadata.from_dict(data)
                snapshots.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to load metadata for {snapshot_dir.name}: {e}")

        # Sort by timestamp, newest first
        snapshots.sort(key=lambda s: s.timestamp, reverse=True)

        return snapshots

    def restore_snapshot(
        self,
        snapshot_name: str,
        confirm: bool = False,
        restore_bulk: bool = True,
    ) -> bool:
        """
        Restore database and bulk storage from snapshot.

        Args:
            snapshot_name: Snapshot identifier
            confirm: Must be True to actually restore (safety check)
            restore_bulk: If True, restore bulk storage files

        Returns:
            True if restoration successful

        Raises:
            SnapshotError: If restoration fails
        """
        if not confirm:
            raise SnapshotError(
                "restore_snapshot requires confirm=True for safety. "
                "This will overwrite current database and files!"
            )

        snapshot_path = self.snapshot_root / snapshot_name
        if not snapshot_path.exists():
            raise SnapshotError(f"Snapshot not found: {snapshot_name}")

        logger.info(f"Restoring snapshot: {snapshot_name}")

        # Load metadata
        metadata_path = snapshot_path / "metadata.json"
        if not metadata_path.exists():
            raise SnapshotError(f"Metadata not found for snapshot: {snapshot_name}")

        with open(metadata_path) as f:
            metadata = SnapshotMetadata.from_dict(json.load(f))

        # Check if compressed
        if metadata.compression:
            logger.info(f"Decompressing snapshot ({metadata.compression})...")
            snapshot_path = self._decompress_snapshot(snapshot_path, metadata)

        try:
            # Restore database
            db_path = snapshot_path / "database"
            if not db_path.exists():
                raise SnapshotError("Database backup not found in snapshot")

            logger.info("Restoring database...")
            self._restore_database(db_path)

            # Restore bulk storage if requested
            if restore_bulk:
                bulk_path = snapshot_path / "bulk_store"
                if bulk_path.exists():
                    logger.info("Restoring bulk storage...")
                    self._restore_bulk_storage(bulk_path)
                else:
                    logger.warning("No bulk storage backup found in snapshot")

            logger.info(f"✓ Snapshot restored: {snapshot_name}")
            logger.info(f"  Collections restored: {len(metadata.collections)}")
            logger.info(f"  Total documents: {sum(metadata.collections.values())}")

            return True

        except Exception as e:
            logger.error(f"Restoration failed: {e}")
            raise SnapshotError(f"Failed to restore snapshot: {e}") from e

    def delete_snapshot(
        self,
        snapshot_name: str,
        force: bool = False,
    ) -> bool:
        """
        Delete a snapshot.

        Args:
            snapshot_name: Snapshot identifier
            force: If True, delete even if marked permanent

        Returns:
            True if deletion successful

        Raises:
            SnapshotError: If deletion fails
        """
        snapshot_path = self.snapshot_root / snapshot_name
        if not snapshot_path.exists():
            raise SnapshotError(f"Snapshot not found: {snapshot_name}")

        # Load metadata to check if permanent
        metadata_path = snapshot_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = SnapshotMetadata.from_dict(json.load(f))

            if metadata.permanent and not force:
                raise SnapshotError(
                    f"Snapshot {snapshot_name} is marked permanent. "
                    f"Use force=True to delete anyway."
                )

        # Delete snapshot directory
        try:
            shutil.rmtree(snapshot_path)
            logger.info(f"✓ Snapshot deleted: {snapshot_name}")
            return True
        except Exception as e:
            raise SnapshotError(f"Failed to delete snapshot: {e}") from e

    def apply_retention_policy(self, dry_run: bool = False):
        """
        Apply retention policy to clean up old snapshots.

        Policy:
        - Keep last 10 snapshots
        - Keep weekly snapshots for 3 months
        - Never delete permanent snapshots
        - Compress snapshots older than 1 week

        Args:
            dry_run: If True, don't actually delete, just report
        """
        logger.info("Applying retention policy...")

        snapshots = self.list_snapshots()
        now = datetime.now()
        one_week_ago = now - timedelta(weeks=1)
        three_months_ago = now - timedelta(days=90)

        # Separate permanent and non-permanent snapshots
        permanent = [s for s in snapshots if s.permanent]
        non_permanent = [s for s in snapshots if not s.permanent]

        # Keep last 10 non-permanent snapshots
        to_keep = set()
        to_delete = []

        # Keep most recent 10
        to_keep.update(s.snapshot_name for s in non_permanent[:10])

        # Keep weekly snapshots within 3 months
        weekly_kept = {}
        for snapshot in non_permanent:
            snap_date = datetime.fromisoformat(snapshot.timestamp)
            if snap_date < three_months_ago:
                continue

            week_key = snap_date.strftime("%Y-W%W")
            if week_key not in weekly_kept:
                weekly_kept[week_key] = snapshot.snapshot_name
                to_keep.add(snapshot.snapshot_name)

        # Determine what to delete
        for snapshot in non_permanent:
            if snapshot.snapshot_name not in to_keep:
                snap_date = datetime.fromisoformat(snapshot.timestamp)
                if snap_date < three_months_ago:
                    to_delete.append(snapshot.snapshot_name)

        # Compress old snapshots
        to_compress = []
        for snapshot in snapshots:
            if snapshot.compression is None:
                snap_date = datetime.fromisoformat(snapshot.timestamp)
                if snap_date < one_week_ago:
                    to_compress.append(snapshot.snapshot_name)

        # Execute deletions
        if to_delete:
            logger.info(f"Deleting {len(to_delete)} old snapshots...")
            for name in to_delete:
                if not dry_run:
                    try:
                        self.delete_snapshot(name, force=False)
                    except Exception as e:
                        logger.warning(f"Failed to delete {name}: {e}")
                else:
                    logger.info(f"  Would delete: {name}")

        # Execute compressions
        if to_compress:
            logger.info(f"Compressing {len(to_compress)} old snapshots...")
            for name in to_compress:
                if not dry_run:
                    try:
                        self._compress_snapshot(name)
                    except Exception as e:
                        logger.warning(f"Failed to compress {name}: {e}")
                else:
                    logger.info(f"  Would compress: {name}")

        logger.info(
            f"✓ Retention policy applied: "
            f"{len(permanent)} permanent, "
            f"{len(to_keep)} kept, "
            f"{len(to_delete)} deleted, "
            f"{len(to_compress)} compressed"
        )

    # Private helper methods

    def _backup_database(self, output_dir: Path) -> Dict[str, int]:
        """
        Backup ArangoDB using arangodump.

        Returns:
            Dictionary of collection_name -> document_count
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "arangodump",
            "--server.database",
            self.db_name,
            "--server.endpoint",
            self.db_endpoint,
            "--server.username",
            self.db_username,
            "--output-directory",
            str(output_dir),
            "--overwrite",
            "true",
            "--include-system-collections",
            "false",
        ]

        if self.db_password:
            cmd.extend(["--server.password", self.db_password])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"arangodump failed with exit code {result.returncode}")
            logger.error(f"stdout: {result.stdout}")
            logger.error(f"stderr: {result.stderr}")
            raise SnapshotError(f"Database backup failed: {result.stderr}\nstdout: {result.stdout}")

        # Parse collection counts from output
        collections = {}
        for line in result.stdout.split("\n"):
            if "Processed" in line and "document(s)" in line:
                # Example: "Processed 5 document(s) in 0.001s for collection 'arxiv_markdown'"
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        count = int(parts[1])
                        coll_name = parts[-1].strip("'\"")
                        collections[coll_name] = count
                    except (ValueError, IndexError):
                        pass

        return collections

    def _restore_database(self, backup_dir: Path):
        """Restore ArangoDB using arangorestore."""
        cmd = [
            "arangorestore",
            "--server.database",
            self.db_name,
            "--server.endpoint",
            self.db_endpoint,
            "--server.username",
            self.db_username,
            "--input-directory",
            str(backup_dir),
            "--overwrite",
            "true",
            "--create-database",
            "true",
        ]

        if self.db_password:
            cmd.extend(["--server.password", self.db_password])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise SnapshotError(f"Database restore failed: {result.stderr}")

    def _backup_bulk_storage(self, output_dir: Path):
        """Backup bulk storage files using rsync."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Only backup experiments/ directory to avoid unnecessary files
        source = self.bulk_store_root / "experiments"
        if not source.exists():
            logger.warning(f"Bulk storage source not found: {source}")
            return

        cmd = [
            "rsync",
            "-av",
            "--exclude",
            "*.tmp",
            "--exclude",
            "__pycache__",
            "--exclude",
            ".git",
            str(source) + "/",
            str(output_dir / "experiments") + "/",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise SnapshotError(f"Bulk storage backup failed: {result.stderr}")

    def _restore_bulk_storage(self, backup_dir: Path):
        """Restore bulk storage files using rsync."""
        source = backup_dir / "experiments"
        target = self.bulk_store_root / "experiments"

        if not source.exists():
            logger.warning("No experiments directory in backup")
            return

        target.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "rsync",
            "-av",
            "--delete",  # Remove files not in backup
            str(source) + "/",
            str(target) + "/",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise SnapshotError(f"Bulk storage restore failed: {result.stderr}")

    def _compress_snapshot(self, snapshot_name: str | Path):
        """Compress a snapshot to save space."""
        # Handle both string name and Path object
        if isinstance(snapshot_name, Path):
            snapshot_path = snapshot_name
            snapshot_name = snapshot_path.name
        else:
            snapshot_path = self.snapshot_root / snapshot_name

        # Load metadata
        metadata_path = snapshot_path / "metadata.json"
        with open(metadata_path) as f:
            metadata = SnapshotMetadata.from_dict(json.load(f))

        if metadata.compression:
            logger.debug(f"Snapshot {snapshot_name} already compressed")
            return

        # Create tar.gz archive
        archive_path = snapshot_path.with_suffix(".tar.gz")
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(snapshot_path, arcname=snapshot_name)

        # Remove original directory (keep only compressed)
        shutil.rmtree(snapshot_path)

        # Extract metadata.json from archive and update it to mark as compressed
        with tarfile.open(archive_path, "r:gz") as tar:
            metadata_member = tar.getmember(f"{snapshot_name}/metadata.json")
            tar.extract(metadata_member, path=self.snapshot_root)

        # Update extracted metadata to mark as compressed
        extracted_metadata_path = self.snapshot_root / snapshot_name / "metadata.json"
        metadata.compression = "tar.gz"
        with open(extracted_metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        logger.info(f"✓ Compressed snapshot: {snapshot_name}")

    def _decompress_snapshot(
        self, snapshot_path: Path, metadata: SnapshotMetadata
    ) -> Path:
        """
        Decompress a snapshot for restoration.

        Returns:
            Path to decompressed snapshot directory
        """
        if metadata.compression == "tar.gz":
            archive_path = snapshot_path.with_suffix(".tar.gz")
            if not archive_path.exists():
                raise SnapshotError(f"Compressed archive not found: {archive_path}")

            # Extract to temporary location
            temp_dir = self.snapshot_root / f"{metadata.snapshot_name}_temp"
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=temp_dir)

            return temp_dir / metadata.snapshot_name

        return snapshot_path

    def _get_git_info(self) -> tuple[Optional[str], Optional[str]]:
        """Get current git commit and branch."""
        try:
            # Get commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )
            commit = result.stdout.strip() if result.returncode == 0 else None

            # Get branch name
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )
            branch = result.stdout.strip() if result.returncode == 0 else None

            return commit, branch
        except Exception:
            return None, None

    def _calculate_directory_size(self, directory: Path) -> float:
        """Calculate directory size in MB."""
        total = 0
        for entry in directory.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
        return total / (1024 * 1024)  # Convert to MB
