# Metis Tools

Production infrastructure and data processing tools for the Metis semantic knowledge system.

## Directory Structure

### `backup/`

**Snapshot manager for ArangoDB and bulk storage backup/restore**

Comprehensive backup system enabling safe experimentation with rollback capability.

**Key Features:**
- Full system snapshots (database + bulk storage)
- Automatic retention policy (last 10, weekly for 3 months)
- Permanent snapshot flag for milestones
- CLI interface with safety confirmations
- Git commit/branch tracking

**Quick Start:**
```bash
# Take snapshot
python -m tools.backup.cli create "Before experiment"

# List snapshots
python -m tools.backup.cli list

# Restore if needed
python -m tools.backup.cli restore <snapshot_name>
```

**Documentation:**
- [Quick Reference](SNAPSHOT_QUICKSTART.md) - Essential commands
- [Full Documentation](backup/README.md) - Detailed workflows and examples

**Status:** ✅ Production ready - 26/26 tests passing

### `arxiv_import/`

**Production pipeline for importing arXiv papers into ArangoDB**

This is the infrastructure that successfully imported 2.8M arXiv papers with metadata and embeddings.

**Key Components:**
- `import_pipeline_multigpu.py` - Multi-GPU parallel import pipeline (what we used for 2.8M papers)
- `import_pipeline.py` - Single-GPU version
- `edge_builder.py` - Graph edge construction (category links, temporal succession)
- `graph_pipeline.py` - Complete graph processing pipeline
- `train_gnn.py` - GraphSAGE GNN training
- `arxiv_parser.py` - arXiv ID parsing utilities

**Usage:**
```bash
# Import with multi-GPU (used for 2.8M papers)
cd tools/arxiv_import
poetry run python import_pipeline_multigpu.py --multi-gpu 0,1 --workers 2

# Build graph edges
poetry run python edge_builder.py

# Train GraphSAGE
poetry run python train_gnn.py
```

**Status:** ✅ Production - Successfully imported 2.8M papers

## Why `tools/` Instead of `experiments/`?

This directory contains **production infrastructure** that:
- Has been tested and validated at scale
- Is reusable for future imports and processing
- Forms the foundation for CF validation experiments

The `experiments/` directory (when created) will contain:
- Specific CF validation experiments
- One-off analyses
- Research prototypes

## Related Directories

- `metis/` - Core library code
- `scripts/` - Operational scripts and utilities
- `old-donotuse/` - Archived code from previous approaches