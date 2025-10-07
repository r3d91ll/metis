# Word2Vec Conveyance Framework Validation Experiment

This experiment implements the first CF validation test using the Word2Vec paper family to validate the hypothesis that embedding paper adoption follows super-linear context amplification (α ∈ [1.5, 2.0]).

## Status: Phase 1 Complete ✓

### Implemented Components

#### ✅ Phase 1: Foundation (Complete)
- [x] Base experiment framework (`experiments/base.py`)
- [x] ArXiv paper fetcher with Docling integration (`arxiv_fetcher.py`)
- [x] ArangoDB storage manager (`storage.py`)
- [x] Experiment configuration (`config.yaml`)
- [x] Phase 1 test script (`test_phase1.py`)

#### 🔄 Phase 2-5: Pending
- [ ] GitHub code fetcher
- [ ] Combined context builder
- [ ] Embedding generation pipeline
- [ ] Full experiment orchestrator
- [ ] Quality validation

## Quick Start

### Prerequisites

```bash
# Install dependencies (from metis root)
poetry install -E pdf

# Ensure ArangoDB is running
# Default: http://localhost:8529
```

### Run Phase 1 Test

```bash
# Test downloading and storing Word2Vec paper
poetry run python experiments/word2vec/test_phase1.py
```

This will:
1. Download Word2Vec paper (1301.3781) from arXiv
2. Convert PDF to markdown using Docling
3. Store in ArangoDB collection `arxiv_markdown`
4. Verify storage and retrieval

### Expected Output

```
======================================================================
Phase 1 Test: Word2Vec Paper Processing
======================================================================

Initializing components...
✓ ArxivPaperFetcher initialized
✓ CFExperimentStorage initialized

Ensuring database collections...
  • arxiv_markdown: 0 documents
  • arxiv_code: 0 documents
  • arxiv_embeddings: 0 documents

Fetching paper 1301.3781...
✓ Paper fetched successfully!
  Title: Efficient Estimation of Word Representations in Vector Space
  Authors: Tomas Mikolov, Kai Chen...
  Markdown: 45231 characters
  Word count: 7845 words

Storing paper in database...
✓ Paper stored with key: 1301_3781
✓ Verified: Paper retrievable from database

======================================================================
Phase 1 Test: SUCCESSFUL
======================================================================
```

## Architecture

### Data Flow (Phase 1)

```
arXiv API → ArxivPaperFetcher → DoclingExtractor → ArangoDB
   (PDF)         (download)         (markdown)      (storage)
```

### Components

#### `arxiv_fetcher.py`
Downloads papers from arXiv and converts to markdown.

**Key Features:**
- PDF caching to avoid re-downloads
- Exponential backoff retry logic
- Integration with Docling for PDF extraction
- Metadata extraction from arXiv API

#### `storage.py`
Manages ArangoDB storage for CF experiments.

**Key Features:**
- Automatic collection creation
- NDJSON bulk operations
- Document versioning (update on duplicate)
- Collection statistics

#### `base.py`
Abstract base class for CF experiments.

**Key Features:**
- Common experiment pipeline
- Progress tracking
- Result aggregation
- JSON export

## Configuration

Edit `config.yaml` to customize:

```yaml
database:
  name: "cf_experiments"
  tcp_host: "localhost"
  tcp_port: 8529

infrastructure:
  cache_dir: "data/experiments/word2vec/cache"

fetching:
  arxiv:
    max_retries: 3
```

## Database Schema

### Collection: `arxiv_markdown`

```json
{
  "_key": "1301_3781",
  "arxiv_id": "1301.3781",
  "title": "Efficient Estimation of Word Representations...",
  "authors": ["Tomas Mikolov", "Kai Chen", ...],
  "markdown_content": "# Title\n\n## Abstract...",
  "processing_metadata": {
    "tool": "docling",
    "word_count": 7845,
    "processing_time_seconds": 3.2
  },
  "experiment_tags": ["word2vec_family", "cf_validation"]
}
```

## Troubleshooting

### ArangoDB Connection Issues

If you see `401 not authorized`:

```bash
# Check ArangoDB is running
curl http://localhost:8529/_api/version

# Create database if needed
# Access ArangoDB web UI at http://localhost:8529
# Create database: cf_experiments
```

### Docling Not Available

```bash
# Install PDF support
poetry install -E pdf
```

### Module Import Errors

```bash
# Reinstall dependencies
poetry lock
poetry install -E pdf
```

## Next Steps (Phase 2+)

1. **GitHub Integration** - Fetch official repositories
2. **Context Builder** - Combine paper + code
3. **Embedding Pipeline** - Generate Jina v4 embeddings
4. **Orchestrator** - Process all 5 papers
5. **CF Analysis** - Calculate α values

## File Structure

```
experiments/word2vec/
├── README.md              # This file
├── config.yaml            # Experiment configuration
├── arxiv_fetcher.py       # Paper download and conversion
├── storage.py             # ArangoDB storage manager
├── test_phase1.py         # Phase 1 validation test
├── github_fetcher.py      # TODO: GitHub integration
├── context_builder.py     # TODO: Paper+code merger
├── experiment.py          # TODO: Main orchestrator
└── __init__.py
```

## References

- [Implementation Document](../../docs/experiments/Word2Vec_CF_Validation_Implementation.md)
- [Metis Documentation](../../README.md)
- [Conveyance Framework Papers](../../../ConveyanceTheory/papers/)

## License

Apache 2.0 (same as Metis project)