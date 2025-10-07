"""
Tests for Metis Snapshot Manager.

Run with: pytest tests/tools/test_snapshot_manager.py -v
"""

import json
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.backup.snapshot_manager import (
    SnapshotError,
    SnapshotManager,
    SnapshotMetadata,
)


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    snapshot_dir = Path(tempfile.mkdtemp(prefix="metis_snapshots_"))
    bulk_store_dir = Path(tempfile.mkdtemp(prefix="metis_bulk_"))

    # Create test bulk store structure
    (bulk_store_dir / "experiments").mkdir()
    (bulk_store_dir / "experiments" / "test.txt").write_text("test data")

    yield {
        "snapshot_root": snapshot_dir,
        "bulk_store_root": bulk_store_dir,
    }

    # Cleanup
    shutil.rmtree(snapshot_dir, ignore_errors=True)
    shutil.rmtree(bulk_store_dir, ignore_errors=True)


@pytest.fixture
def mock_manager(temp_dirs):
    """Create SnapshotManager with mocked database operations."""
    manager = SnapshotManager(
        db_name="test_db",
        db_endpoint="http://127.0.0.1:8529",
        db_username="test",
        db_password="test",
        snapshot_root=temp_dirs["snapshot_root"],
        bulk_store_root=temp_dirs["bulk_store_root"],
    )

    # Mock database operations with side effects to create directories
    def mock_backup_side_effect(output_dir: Path):
        """Mock backup that creates the database directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        # Create a fake dump file
        (output_dir / "test_collection.data.json").write_text('{"test": "data"}')
        return {
            "test_collection": 100,
            "another_collection": 50,
        }

    def mock_restore_side_effect(input_dir: Path):
        """Mock restore that verifies directory exists."""
        if not input_dir.exists():
            raise SnapshotError(f"Database backup not found: {input_dir}")
        return None

    with patch.object(
        manager, "_backup_database", side_effect=mock_backup_side_effect
    ), patch.object(manager, "_restore_database", side_effect=mock_restore_side_effect):
        yield manager


class TestSnapshotMetadata:
    """Test SnapshotMetadata dataclass."""

    def test_metadata_creation(self):
        """Test creating snapshot metadata."""
        metadata = SnapshotMetadata(
            snapshot_name="test_snapshot",
            timestamp="2025-10-07T14:30:00",
            description="Test snapshot",
            git_commit="abc123",
            git_branch="main",
            permanent=False,
            collections={"test": 100},
            bulk_store_size_mb=1.5,
            database_size_mb=2.0,
            compression=None,
        )

        assert metadata.snapshot_name == "test_snapshot"
        assert metadata.permanent is False
        assert metadata.collections == {"test": 100}

    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = SnapshotMetadata(
            snapshot_name="test_snapshot",
            timestamp="2025-10-07T14:30:00",
            description="Test",
            git_commit=None,
            git_branch=None,
            permanent=True,
            collections={"test": 50},
            bulk_store_size_mb=1.0,
            database_size_mb=2.0,
            compression="gzip",
        )

        data = metadata.to_dict()
        assert data["permanent"] is True
        assert data["compression"] == "gzip"


class TestSnapshotManager:
    """Test SnapshotManager core functionality."""

    def test_initialization(self, temp_dirs):
        """Test SnapshotManager initialization."""
        manager = SnapshotManager(
            snapshot_root=temp_dirs["snapshot_root"],
            bulk_store_root=temp_dirs["bulk_store_root"],
        )

        assert manager.snapshot_root == temp_dirs["snapshot_root"]
        assert manager.bulk_store_root == temp_dirs["bulk_store_root"]
        assert manager.snapshot_root.exists()

    def test_create_snapshot(self, mock_manager):
        """Test creating a snapshot."""
        snapshot_name = mock_manager.create_snapshot(
            description="test_snapshot",
            permanent=False,
            include_bulk=True,
        )

        # Verify snapshot name format
        assert snapshot_name.startswith("metis_")
        assert "test_snapshot" in snapshot_name

        # Verify snapshot directory exists
        snapshot_dir = mock_manager.snapshot_root / snapshot_name
        assert snapshot_dir.exists()

        # Verify metadata file
        metadata_file = snapshot_dir / "metadata.json"
        assert metadata_file.exists()

        metadata = json.loads(metadata_file.read_text())
        assert metadata["description"] == "test_snapshot"
        assert metadata["permanent"] is False
        assert "test_collection" in metadata["collections"]

        # Verify bulk storage was copied
        bulk_dir = snapshot_dir / "bulk_store" / "experiments"
        assert bulk_dir.exists()
        assert (bulk_dir / "test.txt").exists()

    def test_create_snapshot_no_bulk(self, mock_manager):
        """Test creating snapshot without bulk storage."""
        snapshot_name = mock_manager.create_snapshot(
            description="db_only",
            permanent=False,
            include_bulk=False,
        )

        snapshot_dir = mock_manager.snapshot_root / snapshot_name
        assert snapshot_dir.exists()
        assert (snapshot_dir / "database").exists()
        assert not (snapshot_dir / "bulk_store").exists()

    def test_create_permanent_snapshot(self, mock_manager):
        """Test creating permanent snapshot."""
        snapshot_name = mock_manager.create_snapshot(
            description="permanent_test",
            permanent=True,
            include_bulk=True,
        )

        snapshots = mock_manager.list_snapshots()
        snapshot = next(s for s in snapshots if s.snapshot_name == snapshot_name)
        assert snapshot.permanent is True

    def test_list_snapshots_empty(self, mock_manager):
        """Test listing snapshots when none exist."""
        snapshots = mock_manager.list_snapshots()
        assert len(snapshots) == 0

    def test_list_snapshots(self, mock_manager):
        """Test listing multiple snapshots."""
        # Create multiple snapshots
        names = []
        for i in range(3):
            name = mock_manager.create_snapshot(
                description=f"test_{i}",
                permanent=(i == 0),
                include_bulk=False,
            )
            names.append(name)

        # List and verify
        snapshots = mock_manager.list_snapshots()
        assert len(snapshots) == 3

        # Verify sorting (most recent first)
        snapshot_names = [s.snapshot_name for s in snapshots]
        assert snapshot_names == sorted(names, reverse=True)

        # Verify permanent flag
        assert snapshots[-1].permanent is True  # First created (i==0)

    def test_restore_snapshot_requires_confirmation(self, mock_manager):
        """Test that restore requires explicit confirmation."""
        snapshot_name = mock_manager.create_snapshot(
            description="test_restore",
            permanent=False,
            include_bulk=False,
        )

        # Should raise error without confirm=True
        with pytest.raises(SnapshotError, match="confirm=True"):
            mock_manager.restore_snapshot(
                snapshot_name=snapshot_name,
                confirm=False,
                restore_bulk=False,
            )

    def test_restore_snapshot(self, mock_manager):
        """Test restoring a snapshot."""
        # Create snapshot
        snapshot_name = mock_manager.create_snapshot(
            description="test_restore",
            permanent=False,
            include_bulk=True,
        )

        # Restore
        success = mock_manager.restore_snapshot(
            snapshot_name=snapshot_name,
            confirm=True,
            restore_bulk=True,
        )
        assert success is True

    def test_restore_nonexistent_snapshot(self, mock_manager):
        """Test restoring snapshot that doesn't exist."""
        with pytest.raises(SnapshotError, match="not found"):
            mock_manager.restore_snapshot(
                snapshot_name="nonexistent",
                confirm=True,
                restore_bulk=False,
            )

    def test_delete_snapshot(self, mock_manager):
        """Test deleting a snapshot."""
        snapshot_name = mock_manager.create_snapshot(
            description="test_delete",
            permanent=False,
            include_bulk=False,
        )

        # Verify exists
        snapshots = mock_manager.list_snapshots()
        assert len(snapshots) == 1

        # Delete
        mock_manager.delete_snapshot(snapshot_name)

        # Verify deleted
        snapshots = mock_manager.list_snapshots()
        assert len(snapshots) == 0

    def test_delete_permanent_snapshot_fails(self, mock_manager):
        """Test that deleting permanent snapshot fails."""
        snapshot_name = mock_manager.create_snapshot(
            description="permanent",
            permanent=True,
            include_bulk=False,
        )

        with pytest.raises(SnapshotError, match="marked permanent"):
            mock_manager.delete_snapshot(snapshot_name)

    def test_delete_nonexistent_snapshot(self, mock_manager):
        """Test deleting snapshot that doesn't exist."""
        with pytest.raises(SnapshotError, match="not found"):
            mock_manager.delete_snapshot("nonexistent")

    def test_calculate_directory_size(self, mock_manager, temp_dirs):
        """Test calculating directory size."""
        test_dir = temp_dirs["bulk_store_root"] / "test_size"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("x" * 1000)  # 1KB
        (test_dir / "file2.txt").write_text("x" * 2000)  # 2KB

        size_mb = mock_manager._calculate_directory_size(test_dir)
        assert size_mb > 0
        assert size_mb < 0.01  # Should be ~0.003 MB

    def test_get_git_info(self, mock_manager):
        """Test getting git information."""
        with patch("subprocess.run") as mock_run:
            # Mock two subprocess calls: one for commit, one for branch
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout="abc123def456\n"),  # commit
                MagicMock(returncode=0, stdout="main\n"),          # branch
            ]

            commit, branch = mock_manager._get_git_info()

            assert commit == "abc123def456"
            assert branch == "main"

    def test_get_git_info_not_git_repo(self, mock_manager):
        """Test getting git info when not in a git repo."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1

            commit, branch = mock_manager._get_git_info()

            assert commit is None
            assert branch is None


class TestRetentionPolicy:
    """Test snapshot retention policy."""

    def create_test_snapshot_metadata(
        self,
        manager: SnapshotManager,
        name: str,
        days_ago: int,
        permanent: bool = False,
    ):
        """Helper to create snapshot metadata with custom timestamp."""
        timestamp = datetime.now() - timedelta(days=days_ago)
        snapshot_dir = manager.snapshot_root / name
        snapshot_dir.mkdir()

        # Create database directory to make it look like a real snapshot
        (snapshot_dir / "database").mkdir()

        metadata = SnapshotMetadata(
            snapshot_name=name,
            timestamp=timestamp.isoformat(),
            description=f"Test snapshot {days_ago} days ago",
            git_commit=None,
            git_branch=None,
            permanent=permanent,
            collections={"test": 10},
            bulk_store_size_mb=1.0,
            database_size_mb=1.0,
            compression=None,
        )

        metadata_file = snapshot_dir / "metadata.json"
        metadata_file.write_text(json.dumps(metadata.to_dict(), indent=2))

    def test_retention_keeps_last_10(self, mock_manager):
        """Test that retention policy keeps last 10 snapshots."""
        # Create 15 snapshots all older than 3 months (so weekly policy doesn't apply)
        for i in range(15):
            self.create_test_snapshot_metadata(
                mock_manager,
                f"metis_snapshot_{i:02d}",
                days_ago=100 + i,  # All beyond 3 months
                permanent=False,
            )

        # Apply policy
        mock_manager.apply_retention_policy(dry_run=False)

        # Should keep only 10 (most recent 10 of the old snapshots)
        snapshots = mock_manager.list_snapshots()
        assert len(snapshots) == 10

    def test_retention_keeps_permanent(self, mock_manager):
        """Test that permanent snapshots are never deleted."""
        # Create old permanent snapshot
        self.create_test_snapshot_metadata(
            mock_manager,
            "metis_permanent_old",
            days_ago=365,
            permanent=True,
        )

        # Create 10 newer snapshots
        for i in range(10):
            self.create_test_snapshot_metadata(
                mock_manager,
                f"metis_snapshot_{i:02d}",
                days_ago=i,
                permanent=False,
            )

        # Apply policy
        mock_manager.apply_retention_policy(dry_run=False)

        # Permanent snapshot should still exist
        snapshots = mock_manager.list_snapshots()
        snapshot_names = [s.snapshot_name for s in snapshots]
        assert "metis_permanent_old" in snapshot_names

    def test_retention_keeps_weekly(self, mock_manager):
        """Test that weekly snapshots are kept for 3 months."""
        # Create snapshots spanning 100 days
        for i in range(0, 100, 7):  # Weekly
            self.create_test_snapshot_metadata(
                mock_manager,
                f"metis_weekly_{i:03d}",
                days_ago=i,
                permanent=False,
            )

        # Apply policy
        mock_manager.apply_retention_policy(dry_run=False)

        # Should keep snapshots within 90 days
        snapshots = mock_manager.list_snapshots()
        assert len(snapshots) > 0

        # Check that very old weekly snapshots are deleted
        snapshot_names = [s.snapshot_name for s in snapshots]
        assert "metis_weekly_098" not in snapshot_names  # 98 days ago

    def test_retention_dry_run(self, mock_manager):
        """Test retention policy dry run doesn't delete."""
        # Create 15 snapshots
        for i in range(15):
            self.create_test_snapshot_metadata(
                mock_manager,
                f"metis_snapshot_{i:02d}",
                days_ago=i,
                permanent=False,
            )

        initial_count = len(mock_manager.list_snapshots())

        # Dry run
        mock_manager.apply_retention_policy(dry_run=True)

        # Should not delete anything
        final_count = len(mock_manager.list_snapshots())
        assert final_count == initial_count


class TestCompression:
    """Test snapshot compression."""

    def test_compress_snapshot(self, mock_manager):
        """Test compressing a snapshot."""
        # Create snapshot
        snapshot_name = mock_manager.create_snapshot(
            description="test_compress",
            permanent=False,
            include_bulk=False,
        )

        snapshot_dir = mock_manager.snapshot_root / snapshot_name

        # Compress
        mock_manager._compress_snapshot(snapshot_dir)

        # Verify tar.gz exists
        tar_file = Path(str(snapshot_dir) + ".tar.gz")
        assert tar_file.exists()

        # Verify original directory exists but only has metadata.json
        # (compression extracts metadata back for listing purposes)
        assert snapshot_dir.exists()
        assert (snapshot_dir / "metadata.json").exists()
        assert not (snapshot_dir / "database").exists()  # Database files are compressed

    def test_decompress_snapshot(self, mock_manager):
        """Test decompressing a snapshot."""
        # Create and compress snapshot
        snapshot_name = mock_manager.create_snapshot(
            description="test_decompress",
            permanent=False,
            include_bulk=False,
        )

        snapshot_dir = mock_manager.snapshot_root / snapshot_name

        # Load metadata before compression
        metadata_path = snapshot_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata_dict = json.load(f)

        mock_manager._compress_snapshot(snapshot_dir)

        tar_file = Path(str(snapshot_dir) + ".tar.gz")
        assert tar_file.exists()

        # Update metadata with compression info
        metadata_dict["compression"] = "tar.gz"
        metadata = SnapshotMetadata.from_dict(metadata_dict)

        # Decompress
        decompressed_path, temp_root = mock_manager._decompress_snapshot(snapshot_dir, metadata)

        # Verify directory exists
        assert decompressed_path.exists()
        assert (decompressed_path / "metadata.json").exists()

        # Cleanup temp directory
        if temp_root and temp_root.exists():
            shutil.rmtree(temp_root)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_description(self, mock_manager):
        """Test creating snapshot with empty description."""
        # Should work but use empty string
        snapshot_name = mock_manager.create_snapshot(
            description="",
            permanent=False,
            include_bulk=False,
        )

        assert snapshot_name.startswith("metis_")

    def test_special_characters_in_description(self, mock_manager):
        """Test description with special characters."""
        # Spaces and hyphens should work
        snapshot_name = mock_manager.create_snapshot(
            description="test-snapshot with spaces",
            permanent=False,
            include_bulk=False,
        )

        assert "test-snapshot" in snapshot_name or "test_snapshot" in snapshot_name

    def test_restore_compressed_snapshot(self, mock_manager):
        """Test restoring a compressed snapshot."""
        # Create snapshot
        snapshot_name = mock_manager.create_snapshot(
            description="test_compressed_restore",
            permanent=False,
            include_bulk=False,
        )

        # Compress it
        snapshot_dir = mock_manager.snapshot_root / snapshot_name
        mock_manager._compress_snapshot(snapshot_dir)

        # Restore should handle decompression automatically
        success = mock_manager.restore_snapshot(
            snapshot_name=snapshot_name,
            confirm=True,
            restore_bulk=False,
        )
        assert success is True
