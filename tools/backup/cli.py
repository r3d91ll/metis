"""
Command-line interface for Metis Snapshot Manager.

Usage:
    metis snapshot create "description" [--permanent]
    metis snapshot list [--verbose]
    metis snapshot restore snapshot_name [--yes]
    metis snapshot delete snapshot_name [--force]
    metis snapshot cleanup [--dry-run]
"""

import argparse
import sys
from typing import Optional

from .snapshot_manager import SnapshotError, SnapshotManager


def create_snapshot_cmd(args: argparse.Namespace) -> int:
    """Handle 'create' command."""
    try:
        manager = SnapshotManager()
        print(f"Creating snapshot: {args.description}")

        snapshot_name = manager.create_snapshot(
            description=args.description,
            permanent=args.permanent,
            include_bulk=not args.no_bulk,
        )

        print(f"✓ Snapshot created: {snapshot_name}")

        # Show snapshot details
        snapshots = manager.list_snapshots()
        snapshot = next((s for s in snapshots if s.snapshot_name == snapshot_name), None)
        if snapshot:
            print("\nDetails:")
            print(f"  Collections: {sum(snapshot.collections.values())} documents")
            print(f"  Database: {snapshot.database_size_mb:.2f} MB")
            print(f"  Bulk storage: {snapshot.bulk_store_size_mb:.2f} MB")
            if snapshot.git_commit:
                print(f"  Git: {snapshot.git_branch}@{snapshot.git_commit[:8]}")

        return 0

    except SnapshotError as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}", file=sys.stderr)
        return 1


def list_snapshots_cmd(args: argparse.Namespace) -> int:
    """Handle 'list' command."""
    try:
        manager = SnapshotManager()
        snapshots = manager.list_snapshots()

        if not snapshots:
            print("No snapshots found.")
            return 0

        print(f"Found {len(snapshots)} snapshot(s):\n")

        for snapshot in snapshots:
            # Basic info
            marker = "📌" if snapshot.permanent else "📦"
            compressed = " [compressed]" if snapshot.compression else ""
            print(f"{marker} {snapshot.snapshot_name}{compressed}")
            print(f"   {snapshot.description}")
            print(f"   Created: {snapshot.timestamp}")

            if args.verbose:
                # Detailed info
                print("   Collections:")
                for coll, count in sorted(snapshot.collections.items()):
                    print(f"     - {coll}: {count} documents")
                print(f"   Database size: {snapshot.database_size_mb:.2f} MB")
                print(f"   Bulk storage: {snapshot.bulk_store_size_mb:.2f} MB")
                if snapshot.git_commit:
                    print(f"   Git: {snapshot.git_branch}@{snapshot.git_commit[:8]}")
            else:
                # Summary
                total_docs = sum(snapshot.collections.values())
                total_size = snapshot.database_size_mb + snapshot.bulk_store_size_mb
                print(f"   {total_docs} docs, {total_size:.2f} MB total")

            print()

        return 0

    except SnapshotError as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}", file=sys.stderr)
        return 1


def restore_snapshot_cmd(args: argparse.Namespace) -> int:
    """Handle 'restore' command."""
    try:
        manager = SnapshotManager()

        # Load snapshot metadata to show what will be restored
        snapshots = manager.list_snapshots()
        snapshot = next((s for s in snapshots if s.snapshot_name == args.snapshot_name), None)

        if not snapshot:
            print(f"✗ Snapshot not found: {args.snapshot_name}", file=sys.stderr)
            return 1

        # Show what will be restored
        print(f"Restore snapshot: {snapshot.snapshot_name}")
        print(f"  Description: {snapshot.description}")
        print(f"  Created: {snapshot.timestamp}")
        print(f"  Collections: {sum(snapshot.collections.values())} documents")
        print(f"  Database: {snapshot.database_size_mb:.2f} MB")
        print(f"  Bulk storage: {snapshot.bulk_store_size_mb:.2f} MB")

        # Confirmation
        if not args.yes:
            print("\n⚠️  WARNING: This will REPLACE current database and bulk storage!")
            response = input("Type 'yes' to confirm restoration: ")
            if response.lower() != "yes":
                print("Restoration cancelled.")
                return 0

        print("\nRestoring snapshot...")
        success = manager.restore_snapshot(
            snapshot_name=args.snapshot_name,
            confirm=True,
            restore_bulk=not args.no_bulk,
        )

        if success:
            print("✓ Snapshot restored successfully")
            return 0
        else:
            print("✗ Restoration failed", file=sys.stderr)
            return 1

    except SnapshotError as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}", file=sys.stderr)
        return 1


def delete_snapshot_cmd(args: argparse.Namespace) -> int:
    """Handle 'delete' command."""
    try:
        manager = SnapshotManager()

        # Load snapshot metadata
        snapshots = manager.list_snapshots()
        snapshot = next((s for s in snapshots if s.snapshot_name == args.snapshot_name), None)

        if not snapshot:
            print(f"✗ Snapshot not found: {args.snapshot_name}", file=sys.stderr)
            return 1

        # Check if permanent
        if snapshot.permanent and not args.force:
            print("✗ Cannot delete permanent snapshot without --force flag", file=sys.stderr)
            return 1

        # Confirmation
        if not args.yes:
            print(f"Delete snapshot: {snapshot.snapshot_name}")
            print(f"  Description: {snapshot.description}")
            if snapshot.permanent:
                print("  ⚠️  This is a PERMANENT snapshot!")
            response = input("Type 'yes' to confirm deletion: ")
            if response.lower() != "yes":
                print("Deletion cancelled.")
                return 0

        manager.delete_snapshot(args.snapshot_name)
        print(f"✓ Snapshot deleted: {args.snapshot_name}")
        return 0

    except SnapshotError as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}", file=sys.stderr)
        return 1


def cleanup_snapshots_cmd(args: argparse.Namespace) -> int:
    """Handle 'cleanup' command."""
    try:
        manager = SnapshotManager()

        print("Applying retention policy...")
        if args.dry_run:
            print("(DRY RUN - no changes will be made)")

        manager.apply_retention_policy(dry_run=args.dry_run)

        if args.dry_run:
            print("\n✓ Dry run complete. Use without --dry-run to apply changes.")
        else:
            print("\n✓ Cleanup complete.")

        return 0

    except SnapshotError as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}", file=sys.stderr)
        return 1


def main(argv: Optional[list] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Metis Snapshot Manager - Backup and restore ArangoDB and bulk storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    subparsers.required = True

    # Create command
    create_parser = subparsers.add_parser(
        "create",
        help="Create a new snapshot",
    )
    create_parser.add_argument(
        "description",
        help="Description of the snapshot",
    )
    create_parser.add_argument(
        "--permanent",
        action="store_true",
        help="Mark snapshot as permanent (never auto-deleted)",
    )
    create_parser.add_argument(
        "--no-bulk",
        action="store_true",
        help="Skip bulk storage backup (database only)",
    )
    create_parser.set_defaults(func=create_snapshot_cmd)

    # List command
    list_parser = subparsers.add_parser(
        "list",
        help="List all snapshots",
    )
    list_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed information",
    )
    list_parser.set_defaults(func=list_snapshots_cmd)

    # Restore command
    restore_parser = subparsers.add_parser(
        "restore",
        help="Restore from a snapshot",
    )
    restore_parser.add_argument(
        "snapshot_name",
        help="Name of snapshot to restore",
    )
    restore_parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )
    restore_parser.add_argument(
        "--no-bulk",
        action="store_true",
        help="Skip bulk storage restore (database only)",
    )
    restore_parser.set_defaults(func=restore_snapshot_cmd)

    # Delete command
    delete_parser = subparsers.add_parser(
        "delete",
        help="Delete a snapshot",
    )
    delete_parser.add_argument(
        "snapshot_name",
        help="Name of snapshot to delete",
    )
    delete_parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )
    delete_parser.add_argument(
        "--force",
        action="store_true",
        help="Force deletion of permanent snapshots",
    )
    delete_parser.set_defaults(func=delete_snapshot_cmd)

    # Cleanup command
    cleanup_parser = subparsers.add_parser(
        "cleanup",
        help="Apply retention policy to remove old snapshots",
    )
    cleanup_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    cleanup_parser.set_defaults(func=cleanup_snapshots_cmd)

    # Parse and execute
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
