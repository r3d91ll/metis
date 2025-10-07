# Snapshot Manager Quick Reference

**Location**: `tools/backup/`
**Full Documentation**: [tools/backup/README.md](backup/README.md)

## Essential Commands

```bash
# Create snapshot
python -m tools.backup.cli create "Description here"

# List snapshots
python -m tools.backup.cli list

# Restore snapshot
python -m tools.backup.cli restore <snapshot_name>

# Delete snapshot
python -m tools.backup.cli delete <snapshot_name>

# Cleanup old snapshots
python -m tools.backup.cli cleanup --dry-run
```

## Before CF Validation Phase 4

```bash
# 1. Take snapshot
python -m tools.backup.cli create "Before CF validation Phase 4"

# 2. Run your experiments
cd experiments/word2vec
python your_experiment.py

# 3. If something goes wrong, restore
python -m tools.backup.cli list
python -m tools.backup.cli restore metis_YYYYMMDD_HHMMSS_before_cf_validation_phase_4
```

## Key Points

- **Snapshots backup**: ArangoDB database + `/bulk-store/metis/` files
- **Location**: `/bulk-store/metis/snapshots/`
- **Restore is destructive**: Always confirms before replacing data
- **Retention**: Keeps last 10, weekly for 3 months, permanent snapshots forever
- **Compression**: Automatic after 1 week (tar.gz)

## Common Flags

- `--permanent` - Never auto-delete this snapshot
- `--no-bulk` - Database only (skip bulk storage)
- `--yes` - Skip confirmation prompts
- `--verbose` - Show detailed information
- `--dry-run` - Preview without making changes

## Examples

```bash
# Permanent snapshot at milestone
python -m tools.backup.cli create "Word2Vec dataset complete" --permanent

# Quick database-only snapshot
python -m tools.backup.cli create "Before schema change" --no-bulk

# Restore with auto-confirm (careful!)
python -m tools.backup.cli restore metis_20251007_143022_baseline --yes

# See what cleanup would delete
python -m tools.backup.cli cleanup --dry-run
```

## Help

```bash
python -m tools.backup.cli --help
python -m tools.backup.cli create --help
python -m tools.backup.cli restore --help
```
