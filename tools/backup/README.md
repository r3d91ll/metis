# Metis Snapshot Manager

Backup and restore system for ArangoDB databases and bulk storage files. Enables safe experimentation with rollback capability.

## Quick Start

```bash
# Take a snapshot before starting work
python -m tools.backup.cli create "Before CF validation Phase 4"

# List all snapshots
python -m tools.backup.cli list

# Restore if needed
python -m tools.backup.cli restore metis_20251007_143022_before_cf_validation_phase_4

# Cleanup old snapshots
python -m tools.backup.cli cleanup --dry-run
```

## Features

- **Full System Snapshots**: Backs up both ArangoDB collections and bulk storage files
- **Retention Policy**: Automatically manages snapshot lifecycle
  - Keeps last 10 snapshots
  - Keeps weekly snapshots for 3 months
  - Compresses snapshots older than 1 week
- **Permanent Snapshots**: Flag important snapshots to never be auto-deleted
- **Git Integration**: Tracks git commit and branch in snapshot metadata
- **Safety Features**: Requires explicit confirmation for destructive operations

## Installation

The snapshot manager is part of the Metis tools package. No additional installation needed.

## Usage

### Create a Snapshot

```bash
# Basic snapshot
python -m tools.backup.cli create "Before major refactor"

# Permanent snapshot (never auto-deleted)
python -m tools.backup.cli create "Production baseline" --permanent

# Database only (skip bulk storage)
python -m tools.backup.cli create "Quick checkpoint" --no-bulk
```

### List Snapshots

```bash
# List all snapshots
python -m tools.backup.cli list

# Detailed view with collection counts
python -m tools.backup.cli list --verbose
```

Example output:
```
Found 3 snapshot(s):

📌 metis_20251007_143022_production_baseline [compressed]
   Production baseline before v2.0 deployment
   Created: 2025-10-07T14:30:22
   5 collections, 125430 docs, 2.4 GB total

📦 metis_20251007_120045_before_major_refactor
   Before major refactor
   Created: 2025-10-07T12:00:45
   5 collections, 98234 docs, 1.8 GB total
```

### Restore a Snapshot

```bash
# Interactive restore (prompts for confirmation)
python -m tools.backup.cli restore metis_20251007_143022_production_baseline

# Skip confirmation prompt
python -m tools.backup.cli restore metis_20251007_143022_production_baseline --yes

# Restore database only (skip bulk storage)
python -m tools.backup.cli restore metis_20251007_143022_production_baseline --no-bulk
```

⚠️ **Warning**: Restoration will **REPLACE** your current database and bulk storage!

### Delete a Snapshot

```bash
# Interactive delete (prompts for confirmation)
python -m tools.backup.cli delete metis_20251007_120045_before_major_refactor

# Skip confirmation prompt
python -m tools.backup.cli delete metis_20251007_120045_before_major_refactor --yes

# Force delete permanent snapshot
python -m tools.backup.cli delete metis_20251007_143022_production_baseline --force --yes
```

### Cleanup Old Snapshots

Apply retention policy to remove old snapshots:

```bash
# Dry run (show what would be deleted)
python -m tools.backup.cli cleanup --dry-run

# Apply cleanup
python -m tools.backup.cli cleanup
```

## Common Workflows

### Workflow 1: Safe Experimentation

Use this workflow when running experiments that modify your database:

```bash
# 1. Take a snapshot before starting
python -m tools.backup.cli create "Before CF validation Phase 4 - clean baseline"

# Output:
# ✓ Snapshot created: metis_20251007_143022_before_cf_validation_phase_4_clean_baseline
#   Collections: 15000 documents
#   Database: 250.5 MB
#   Bulk storage: 1200.3 MB
#   Git: feature/case-study-data-collection@5a49f37

# 2. Run your experiments
cd experiments/word2vec
python cf_validation_phase4.py

# 3a. If experiments succeed, mark the result
python -m tools.backup.cli create "After Phase 4 - results validated" --permanent

# 3b. If experiments fail or data is corrupted, restore
python -m tools.backup.cli list  # Find the snapshot name
python -m tools.backup.cli restore metis_20251007_143022_before_cf_validation_phase_4_clean_baseline

# Confirmation prompt will show:
# Restore snapshot: metis_20251007_143022_before_cf_validation_phase_4_clean_baseline
#   Description: Before CF validation Phase 4 - clean baseline
#   Created: 2025-10-07T14:30:22
#   Collections: 15000 documents
#   Database: 250.5 MB
#   Bulk storage: 1200.3 MB
#
# ⚠️  WARNING: This will REPLACE current database and bulk storage!
# Type 'yes' to confirm restoration: yes
#
# Restoring snapshot...
# ✓ Snapshot restored successfully
```

### Workflow 2: Production Baselines

Create permanent snapshots at major milestones:

```bash
# After completing Word2Vec data collection
python -m tools.backup.cli create "Word2Vec dataset complete - 5 papers + code" --permanent

# Before starting new research direction
python -m tools.backup.cli create "Baseline before Transformer analysis" --permanent

# List permanent snapshots
python -m tools.backup.cli list | grep 📌
```

### Workflow 3: Daily Development

Quick snapshots for day-to-day work:

```bash
# Morning: Take snapshot before starting work
python -m tools.backup.cli create "Daily backup $(date +%Y%m%d)"

# End of day: Review what was created
python -m tools.backup.cli list | head -5

# Weekly: Cleanup old snapshots
python -m tools.backup.cli cleanup --dry-run
python -m tools.backup.cli cleanup
```

### Workflow 4: Database-Only Operations

When bulk storage hasn't changed, snapshot just the database:

```bash
# Database-only snapshot (faster)
python -m tools.backup.cli create "Before schema migration" --no-bulk

# Restore database only
python -m tools.backup.cli restore metis_20251007_120000_before_schema_migration --no-bulk
```

## What Gets Backed Up

**Database (via arangodump):**

- All collections in your ArangoDB database (default: `arxiv_datastore`)
- Document counts per collection
- Indexes and graph structures
- Collection metadata

**Bulk Storage (via rsync):**

- `/bulk-store/metis/experiments/` directory
- All paper sources, code repositories, embeddings
- Excludes: `.tmp`, `__pycache__`, `.git`

**Metadata (JSON file):**
```json
{
  "snapshot_name": "metis_20251007_143022_before_cf_validation_phase_4",
  "timestamp": "2025-10-07T14:30:22",
  "description": "Before CF validation Phase 4 - clean baseline",
  "git_commit": "5a49f37abc123...",
  "git_branch": "feature/case-study-data-collection",
  "permanent": false,
  "collections": {
    "arxiv_markdown": 5,
    "arxiv_code": 5,
    "cf_embeddings": 5
  },
  "bulk_store_size_mb": 1200.3,
  "database_size_mb": 250.5,
  "compression": null
}
```

## Retention Policy

The snapshot manager automatically applies a retention policy:

1. **Keep Last 10**: Always keeps the 10 most recent snapshots
2. **Weekly for 3 Months**: Keeps one snapshot per week for 3 months
3. **Permanent Protection**: Never deletes snapshots marked as permanent
4. **Automatic Compression**: Compresses snapshots older than 1 week using tar.gz

### How It Works

When creating a new snapshot or running `cleanup`, the system:
1. Identifies snapshots eligible for deletion
2. Excludes the last 10 snapshots
3. Excludes weekly snapshots from the last 3 months
4. Excludes permanent snapshots
5. Deletes remaining snapshots
6. Compresses snapshots older than 1 week

## Storage Layout

Snapshots are stored in `/bulk-store/metis/snapshots/`:

```
/bulk-store/metis/snapshots/
├── metis_20251007_143022_production_baseline/
│   ├── metadata.json              # Snapshot metadata
│   ├── database/                  # ArangoDB dump
│   │   ├── arxiv_markdown.data.json
│   │   ├── arxiv_code.data.json
│   │   └── cf_embeddings.data.json
│   └── bulk_store/                # Bulk storage files
│       └── experiments/
└── metis_20251007_120045_before_major_refactor.tar.gz  # Compressed
```

## Programmatic Usage

You can also use the snapshot manager programmatically:

```python
from tools.backup import SnapshotManager

# Initialize
manager = SnapshotManager()

# Create snapshot
snapshot_name = manager.create_snapshot(
    description="Before experiment",
    permanent=False,
    include_bulk=True,
)

# List snapshots
snapshots = manager.list_snapshots()
for snapshot in snapshots:
    print(f"{snapshot.snapshot_name}: {snapshot.description}")

# Restore snapshot
manager.restore_snapshot(
    snapshot_name=snapshot_name,
    confirm=True,  # Must be True for safety
    restore_bulk=True,
)

# Apply retention policy
manager.apply_retention_policy(dry_run=False)
```

## Best Practices

### When to Create Snapshots

- **Before major changes**: Refactoring, schema changes, bulk updates
- **Before experiments**: Testing new algorithms, parameter tuning
- **Periodic backups**: Daily/weekly production snapshots
- **Release milestones**: Mark as permanent for version baselines

### Snapshot Naming

The snapshot name is auto-generated: `metis_YYYYMMDD_HHMMSS_description`

- Use descriptive names: "before_schema_migration", "baseline_v2.0"
- Keep descriptions concise (will be part of filename)
- Avoid special characters (use underscores/hyphens)

### Managing Storage

- Compression reduces size by ~70% for text-heavy data
- Monitor `/bulk-store/metis/snapshots/` disk usage
- Run `cleanup` regularly to apply retention policy
- Use `--no-bulk` for quick database-only snapshots
- Mark important snapshots as `--permanent`

## Troubleshooting

### "Snapshot directory already exists"

A snapshot with the same timestamp already exists. Wait a second and try again, or manually remove the directory.

### "arangodump not found"

Install ArangoDB tools:
```bash
# Ubuntu/Debian
sudo apt-get install arangodb3-client
```

### "Permission denied" on socket

Ensure your user is in the `metis` group:
```bash
sudo usermod -aG metis $USER
# Log out and back in
```

### Restore fails with "database not empty"

By default, `arangorestore` creates collections. If collections exist, it may fail. The snapshot manager handles this automatically, but if you see errors, check ArangoDB logs.

### Out of disk space

- Run `cleanup --dry-run` to see what can be deleted
- Delete old snapshots manually if needed
- Consider storing snapshots on a separate volume

## Integration Testing

To test the snapshot manager:

```bash
# 1. Create test data in ArangoDB
python -c "from experiments.word2vec.storage import Storage; s = Storage(); print('Test data ready')"

# 2. Create snapshot
python -m tools.backup.cli create "integration_test"

# 3. Modify data (delete a collection, change documents, etc.)

# 4. List snapshots
python -m tools.backup.cli list --verbose

# 5. Restore snapshot
python -m tools.backup.cli restore metis_YYYYMMDD_HHMMSS_integration_test --yes

# 6. Verify data restored correctly

# 7. Cleanup
python -m tools.backup.cli delete metis_YYYYMMDD_HHMMSS_integration_test --yes
```

## Technical Details

### Database Backup

Uses `arangodump` with:
- Unix socket connection for performance
- System collections excluded
- Overwrite mode enabled
- Per-collection document counts captured

### Bulk Storage Backup

Uses `rsync` with:
- Archive mode (`-av`)
- Excludes: `.tmp`, `__pycache__`, `.git`
- Efficient incremental copying
- Preserves permissions and timestamps

### Compression

Uses `tar.gz` compression:
- Applied to snapshots older than 1 week
- Compresses database dumps and bulk storage together
- Automatic decompression during restoration
- Reduces storage by ~70% for text-heavy data

### Git Tracking

Captures git metadata at snapshot creation:
- Current commit hash (full SHA)
- Current branch name
- Stored in `metadata.json`
- Helps correlate snapshots with code versions

## See Also

- [ArangoDB Backup Documentation](https://docs.arangodb.com/stable/operations/backup-and-restore/)
- [rsync Manual](https://linux.die.net/man/1/rsync)
- Metis project documentation: `/home/todd/olympus/metis/README.md`
