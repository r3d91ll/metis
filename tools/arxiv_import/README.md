# arXiv Import Pipeline with GraphSAGE GNN

Complete pipeline for importing 2.8M arXiv papers, building a relational graph, and training a multi-relational GraphSAGE GNN.

## Philosophy: Vector WHAT, Graph WHERE

**Metis separates two orthogonal dimensions:**
- **Vector embeddings (WHAT):** Semantic content via Jina v4 (2048-dim)
- **Graph structure (WHERE):** Relational position via edges (category overlap, temporal succession)

**No semantic edges from embeddings** - keeps dimensions independent.

## Pipeline Stages

### Stage 1: Import Papers
Import arXiv papers with metadata and embeddings into ArangoDB.

**Collections created:**
- `arxiv_papers` - Metadata (authors, categories, dates)
- `arxiv_abstracts` - Full text (title, abstract)
- `arxiv_embeddings` - Jina v4 embeddings (2048-dim × 3 types)

**Command:**
```bash
poetry run python import_pipeline.py --limit 1000  # Test with 1000 papers
poetry run python import_pipeline.py              # Full 2.8M papers
```

**Time:** ~2-4 hours for 2.8M papers (with GPU embeddings)

---

### Stage 2: Build Graph Edges
Construct relational edges capturing structural position.

**Edge types:**
1. **Category links** - Papers sharing ≥2 categories (multi-disciplinary bridges)
2. **Temporal succession** - Papers in same field, 1-3 months apart (influence flow)

**Collections created:**
- `category_links` (~20M edges)
- `temporal_succession` (~30-50M edges)

**Command:**
```bash
poetry run python edge_builder.py
poetry run python edge_builder.py --dry-run  # Count without inserting
```

**Time:** ~1-2 hours for 2.8M papers

---

### Stage 3: Export Graph
Export ArangoDB graph to PyTorch Geometric format.

**Output:**
- `models/arxiv_graph.pt` (~30GB file)
- PyG Data object with multi-relational edges
- Train/val split (80/20)

**Command:**
```bash
poetry run python graph_pipeline.py
poetry run python graph_pipeline.py --output models/custom.pt
```

**Time:** ~10-20 minutes

---

### Stage 4: Train GraphSAGE
Train multi-relational GraphSAGE on the graph.

**Model:**
- Input: 2048-dim (Jina embeddings)
- Hidden: 1024-dim
- Output: 512-dim (compressed for retrieval)
- 3 GraphSAGE layers
- Multi-relational aggregation (different weights per edge type)

**Training:**
- Self-supervised (edge connectivity as supervision)
- InfoNCE loss (contrastive learning)
- NeighborLoader for memory efficiency
- 100 epochs with early stopping

**Output:**
- `models/arxiv_checkpoints/best.pt` - Best model checkpoint
- `models/arxiv_checkpoints/latest.pt` - Latest checkpoint

**Command:**
```bash
poetry run python train_gnn.py
poetry run python train_gnn.py --epochs 50  # Override config
```

**Time:** ~4-8 hours for 2.8M papers

---

## Full Pipeline (All Stages)

Run the complete workflow in one command:

```bash
# Test with 1000 papers
poetry run python import_pipeline.py --limit 1000 --full-pipeline

# Full 2.8M papers (remove --limit)
poetry run python import_pipeline.py --full-pipeline
```

**Total time:** ~6-12 hours for 2.8M papers

**Stages:**
1. Import papers → ArangoDB collections
2. Build edges → Edge collections
3. Export graph → PyG Data file
4. Train GNN → Model checkpoints

---

## Individual Stage Flags

Run specific stages independently:

```bash
# Just import
python import_pipeline.py --limit 1000

# Import + build edges
python import_pipeline.py --limit 1000 --build-edges

# Import + edges + export graph
python import_pipeline.py --limit 1000 --build-edges --export-graph

# Import + edges + export + train
python import_pipeline.py --limit 1000 --full-pipeline
```

---

## Configuration

Edit `config/arxiv_import.yaml` to customize:

**Database:**
```yaml
database:
  name: "arxiv_datastore"
  socket_path: "/run/metis/readwrite/arangod.sock"
```

**Embeddings:**
```yaml
embeddings:
  model: "jinaai/jina-embeddings-v4"
  device: "cuda"
  batch_size: 48
```

**Edges:**
```yaml
edges:
  category_links:
    min_shared_categories: 2

  temporal_succession:
    min_months: 1
    max_months: 3
    max_edges_per_paper: 50
```

**GNN:**
```yaml
gnn:
  architecture:
    hidden_channels: 1024
    out_channels: 512

  training:
    epochs: 100
    batch_size: 512
    learning_rate: 0.001
```

---

## Edge Design Rationale

### Category Links (Multi-disciplinary Bridges)

**Rule:** Papers sharing ≥2 categories

**Why:** Single-category papers are intra-field. Multi-category papers bridge disciplines.

**Example:**
- Paper A: `["cs.LG", "stat.ML"]`
- Paper B: `["stat.ML", "math.ST"]`
- **Shared:** `stat.ML` → Creates edge (weight=1)

**Target:** Theory-practice bridges, interdisciplinary work

---

### Temporal Succession (Influence Flow)

**Rule:** Same primary_category, 1-3 months apart, directional

**Why:** Papers published soon after each other in the same field likely build on each other.

**Example:**
- Paper A: `cs.AI`, `202301` (Jan 2023)
- Paper B: `cs.AI`, `202303` (Mar 2023)
- **Edge:** A → B (weight = exp(-0.3 × 2) = 0.55)

**Target:** Field evolution, paradigm development, influence chains

---

## Death of the Author

**No author-based edges.** Author name disambiguation is complex and error-prone. We focus on semantic and structural signals that don't require entity resolution.

**Later:** When adding PDFs, we'll build citation networks (explicit references). That's a separate graph with different semantics.

---

## Memory Requirements

### GPU (Training)
- **Minimum:** 24GB VRAM (A6000, RTX 4090)
- **Recommended:** 48GB VRAM (A6000 × 2, A100)
- Uses NeighborLoader for memory-efficient training

### RAM (Graph Export)
- **Minimum:** 32GB RAM
- **Recommended:** 64GB RAM
- Loads entire graph into memory during export

### Disk
- **Embeddings:** ~23GB (2.8M × 2048 × 4 bytes)
- **Graph file:** ~30GB (nodes + edges)
- **Checkpoints:** ~5GB per checkpoint
- **Total:** ~100GB for full pipeline

---

## Output Files

```plaintext
models/
├── arxiv_graph.pt              # PyG Data object (~30GB)
└── arxiv_checkpoints/
    ├── best.pt                 # Best model checkpoint
    └── latest.pt               # Latest checkpoint
```

---

## Validation

After training, validate the graph captures useful structure:

### Test 1: Bridge Paper Detection
Papers with high betweenness centrality should be multi-category papers.

### Test 2: Temporal Influence
Papers with high out-degree in temporal edges should be foundational work.

### Test 3: Retrieval Quality
Compare retrieval with GraphSAGE embeddings vs. raw Jina embeddings.

**Metric:** MRR@10, Recall@50 on category-based ground truth

---

## Troubleshooting

### "No edges created"
- Check that papers have `categories` and `year_month` fields
- Verify edge config in `arxiv_import.yaml`
- Run with `--dry-run` to see edge counts before inserting

### "Out of memory" during training
- Reduce `batch_size` in config
- Reduce `num_neighbors` (sample fewer neighbors per layer)
- Use CPU training (slower but no memory limit): set `device: "cpu"`

### "Graph file too large"
- This is expected for 2.8M papers (~30GB)
- Use NVMe SSD for fast loading
- Consider filtering to specific categories for smaller graphs

---

## Next Steps

After completing this pipeline:

1. **Retrieval evaluation:** Compare raw embeddings vs. GraphSAGE embeddings
2. **Bridge paper analysis:** Identify top theory-practice bridges via centrality
3. **Citation network:** Add PDF parsing and citation extraction (separate graph)
4. **HADES integration:** Use GraphSAGE embeddings for experiential memory retrieval

---

## Related Files

- [import_pipeline.py](import_pipeline.py) - Main pipeline orchestration
- [edge_builder.py](edge_builder.py) - Graph edge construction
- [graph_pipeline.py](graph_pipeline.py) - ArangoDB → PyG export
- [train_gnn.py](train_gnn.py) - GraphSAGE training
- [arxiv_parser.py](arxiv_parser.py) - arXiv ID parsing utilities
- [config/arxiv_import.yaml](config/arxiv_import.yaml) - Configuration

---

## References

- **GraphSAGE:** Hamilton et al. (2017) - Inductive Representation Learning on Large Graphs
- **Multi-relational GNNs:** Schlichtkrull et al. (2018) - Modeling Relational Data with Graph Convolutional Networks
- **InfoNCE Loss:** van den Oord et al. (2018) - Representation Learning with Contrastive Predictive Coding
- **Jina Embeddings:** Jina AI - jina-embeddings-v4 (2048-dim, 32k context)
