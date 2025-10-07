# Metis Backup Strategy

This document describes the two-tier backup system for Metis data and databases.

## Overview

**Two backup systems with different purposes:**

1. **DR Backup (Disaster Recovery)** - Full system snapshots for catastrophic failure
2. **Operational Snapshots** - Single database backups for experiment rollback

## 1. DR Backup (Disaster Recovery)

**Purpose**: Complete ArangoDB instance backup for disaster recovery

**Scope**: Entire `dbpool/arangodb` dataset (all databases, all collections)

**Method**: ZFS snapshots (instant, copy-on-write)

**Schedule**: Nightly at 2 AM, 30-day retention

**Storage**: On `dbpool` (same pool, minimal space overhead due to COW)

### Manual DR Backup

```bash
# Create DR snapshot
sudo zfs snapshot dbpool/arangodb@dr_golden_image_$(date +%Y%m%d_%H%M%S)

# List all DR snapshots
zfs list -t snapshot -r dbpool/arangodb

# Check snapshot size
zfs list -t snapshot -r dbpool/arangodb -o name,used,referenced
```

### DR Restore Procedure

**⚠️ WARNING**: This will restore **ALL databases** to the snapshot state!

```bash
# 1. Stop ArangoDB
sudo systemctl stop arangodb3

# 2. Rollback to snapshot
sudo zfs rollback dbpool/arangodb@dr_golden_image_20251007_122624

# 3. Start ArangoDB
sudo systemctl start arangodb3

# 4. Verify
sudo systemctl status arangodb3
```

### Automated Nightly DR Backups

**Cron configuration** in `/etc/cron.d/metis-dr-backup`:

```cron
# Metis DR Backup - Nightly ZFS snapshot of ArangoDB
# Runs at 2 AM daily, keeps last 30 snapshots

0 2 * * * root /usr/local/bin/metis-dr-backup.sh
```

**Backup script** at `/usr/local/bin/metis-dr-backup.sh`:

```bash
#!/bin/bash
#
# Metis DR Backup Script
# Creates nightly ZFS snapshot of entire ArangoDB instance
# Retains last 30 snapshots (30 days)
#

set -euo pipefail

DATASET="dbpool/arangodb"
SNAPSHOT_PREFIX="dr_nightly"
RETENTION_DAYS=30

# Create timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SNAPSHOT_NAME="${DATASET}@${SNAPSHOT_PREFIX}_${TIMESTAMP}"

# Create snapshot
echo "$(date): Creating DR snapshot: ${SNAPSHOT_NAME}"
zfs snapshot "${SNAPSHOT_NAME}"

if [ $? -eq 0 ]; then
    echo "$(date): ✓ DR snapshot created successfully"
else
    echo "$(date): ✗ DR snapshot failed!" >&2
    exit 1
fi

# Clean up old snapshots (keep last 30)
echo "$(date): Cleaning up old DR snapshots..."
zfs list -t snapshot -r "${DATASET}" -o name -s creation | \
    grep "${SNAPSHOT_PREFIX}" | \
    head -n -${RETENTION_DAYS} | \
    while read snapshot; do
        echo "$(date): Deleting old snapshot: ${snapshot}"
        zfs destroy "${snapshot}"
    done

echo "$(date): DR backup complete"
```

### Setup Instructions

```bash
# 1. Create backup script
sudo tee /usr/local/bin/metis-dr-backup.sh > /dev/null << 'EOF'
[paste script above]
EOF

# 2. Make executable
sudo chmod +x /usr/local/bin/metis-dr-backup.sh

# 3. Test manually
sudo /usr/local/bin/metis-dr-backup.sh

# 4. Create cron job
sudo tee /etc/cron.d/metis-dr-backup > /dev/null << 'EOF'
# Metis DR Backup - Nightly ZFS snapshot of ArangoDB
0 2 * * * root /usr/local/bin/metis-dr-backup.sh >> /var/log/metis-dr-backup.log 2>&1
EOF

# 5. Verify cron job
sudo crontab -l | grep metis
```

## 2. Operational Snapshots

**Purpose**: On-demand backup of single database before experiments

**Scope**: Single database (e.g., `arxiv_datastore`) only

**Method**: `arangodump` for granular database export

**Schedule**: On-demand via CLI

**Storage**: `/bulk-store/metis/snapshots/`

### Create Operational Snapshot

```bash
# Before CF validation or experiments
cd ~/olympus/metis
python -m tools.backup.cli create "Before CF validation Phase 4" --database arxiv_datastore
```

### Restore Operational Snapshot

```bash
# List available snapshots
python -m tools.backup.cli list

# Restore specific database only (does NOT affect other databases)
python -m tools.backup.cli restore metis_20251007_143022_before_cf_validation_phase_4
```

### Key Differences from DR Backup

- **Granular**: Restores only one database, not entire ArangoDB instance
- **Selective**: Can restore while other databases keep running
- **Portable**: Dump files can be moved to other systems
- **Slower**: Takes time to dump/restore vs instant ZFS snapshots

**Full documentation**: [tools/backup/README.md](../tools/backup/README.md)

## Comparison Matrix

| Feature | DR Backup | Operational Snapshot |
|---------|-----------|---------------------|
| **Scope** | All databases | Single database |
| **Method** | ZFS snapshot | arangodump |
| **Speed** | Instant | Minutes |
| **Storage** | dbpool (COW) | bulk-store |
| **Schedule** | Nightly (automated) | On-demand (manual) |
| **Use Case** | Hardware failure, corruption | Experiment rollback |
| **Granularity** | Entire ArangoDB | One database |
| **Downtime** | Requires ArangoDB restart | No restart needed |

## When to Use Each

### Use DR Backup When:

- Hardware failure or disk corruption
- Complete system restore needed
- Rolling back all databases to a known state
- Scheduled maintenance/upgrades

### Use Operational Snapshot When:

- Starting an experiment that modifies data
- Testing schema changes on one database
- Need to rollback without affecting other databases
- Want portable backup for migration

## Monitoring

### Check DR Snapshot Status

```bash
# List all DR snapshots
zfs list -t snapshot -r dbpool/arangodb

# Check space used by snapshots
zfs list -t snapshot -r dbpool/arangodb -o name,used,referenced

# See how much has changed since snapshot
zfs diff dbpool/arangodb@dr_golden_image_20251007_122624
```

### Check Operational Snapshot Status

```bash
# List operational snapshots
python -m tools.backup.cli list --verbose

# Check snapshot storage usage
du -sh /bulk-store/metis/snapshots/*
```

## Log Files

- **DR backups**: `/var/log/metis-dr-backup.log`
- **Operational snapshots**: stdout/stderr from CLI

## Success Criteria

**DR Backup Test:**
- ✅ Create DR snapshot: `sudo zfs snapshot dbpool/arangodb@dr_golden_image_$(date +%Y%m%d)`
- ✅ Verify snapshot exists: `zfs list -t snapshot -r dbpool/arangodb`
- ✅ Test restore on non-production system (optional)

**Operational Snapshot Test:**
- ✅ Create golden-image: `python -m tools.backup.cli create "golden-image-phase3"`
- ✅ Verify backup: `python -m tools.backup.cli list`
- ✅ Test restore: Create, modify data, restore, verify rollback

## Related Documentation

- [Snapshot Manager Quick Reference](../tools/SNAPSHOT_QUICKSTART.md)
- [Snapshot Manager Full Documentation](../tools/backup/README.md)
- ZFS Snapshots: `man zfs-snapshot`
- [ArangoDB Backup](https://docs.arangodb.com/stable/operations/backup-and-restore/)
