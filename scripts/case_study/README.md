# Case Study Data Collection: Transformers vs Capsules

This directory contains a complete data collection pipeline for analyzing the differential adoption of two influential AI papers from NIPS 2017:

- **Transformers**: "Attention Is All You Need" (arXiv:1706.03762)
- **Capsule Networks**: "Dynamic Routing Between Capsules" (arXiv:1710.09829)

This pipeline supports the Conveyance Framework research by collecting empirical evidence of how knowledge diffused through academic and practitioner communities.

## Overview

The pipeline collects five types of data:

1. **Citation timelines** - Monthly citation counts from Semantic Scholar
2. **GitHub implementations** - Repository tracking with classification
3. **Paper content** - Full text extraction with section identification
4. **Boundary objects** - Documentation, code examples, tutorials
5. **Embeddings** - Jina v4 semantic embeddings stored in ArangoDB

All data is processed into visualizations showing semantic maps and temporal evolution.

## Quick Start

### Prerequisites

```bash
# Install dependencies
cd /home/todd/olympus/metis
poetry install

# Optional: Set GitHub token for higher rate limits
export GITHUB_TOKEN=your_token_here
```

### Run Full Pipeline

```bash
cd scripts/case_study
./run_all.sh
```

This executes all 6 collection scripts in sequence and generates visualizations.

### Run Individual Scripts

```bash
# Step 1: Citation data
python 01_collect_citations.py

# Step 2: GitHub repositories
python 02_collect_github_repos.py

# Step 3: Paper extraction
python 03_extract_papers.py

# Step 4: Boundary objects
python 04_collect_boundary_objects.py

# Step 5: Generate embeddings
python 05_generate_embeddings.py

# Step 6: Create visualizations
python 06_create_visualizations.py
```

## Pipeline Details

### Script 01: Citation Collection

**Purpose**: Collect citation timeline data from Semantic Scholar

**Data Source**: Semantic Scholar API (free, no key required)

**Output**:
- `data/case_study/citations/transformers_citations.json`
- `data/case_study/citations/capsules_citations.json`

**Output Schema**:
```json
{
  "paper_id": "1706.03762",
  "title": "Attention Is All You Need",
  "authors": ["Vaswani", "Shazeer", ...],
  "publication_date": "2017-06-12",
  "total_citations": 35000,
  "monthly_citations": [
    {"year": 2017, "month": 6, "count": 5},
    {"year": 2017, "month": 7, "count": 12}
  ],
  "collected_at": "2025-10-04T12:00:00"
}
```

**Rate Limiting**: 1.5 requests/second (respects Semantic Scholar limits)

**Error Handling**: 3 retries with exponential backoff

### Script 02: GitHub Repository Tracking

**Purpose**: Find and classify GitHub implementations

**Data Source**: GitHub API

**Search Queries**:
- Transformers: "attention is all you need", "transformer attention mechanism", etc.
- Capsules: "capsule networks", "dynamic routing capsules", etc.

**Filters**:
- Created after paper publication date
- Minimum 10 stars
- Minimum 5 forks
- Primary language: Python

**Classification Types**:
- `official` - From paper authors
- `framework` - Integrated into major framework
- `tutorial` - Educational implementation
- `application` - Used for specific task
- `research` - Extension or variation

**Output**:
- `data/case_study/implementations/transformers_repos.json`
- `data/case_study/implementations/capsules_repos.json`

**Output Schema**:
```json
{
  "paper": "transformers",
  "arxiv_id": "1706.03762",
  "title": "Attention Is All You Need",
  "total_repositories": 50,
  "type_counts": {
    "official": 1,
    "framework": 5,
    "tutorial": 20
  },
  "repositories": [
    {
      "url": "https://github.com/tensorflow/tensor2tensor",
      "full_name": "tensorflow/tensor2tensor",
      "created_at": "2017-06-12T00:00:00Z",
      "stars": 15000,
      "forks": 3000,
      "language": "Python",
      "type": "official",
      "description": "...",
      "from_authors": true,
      "has_wiki": true,
      "open_issues": 100,
      "topics": ["tensorflow", "transformer"],
      "last_updated": "2025-10-01T00:00:00Z"
    }
  ],
  "collected_at": "2025-10-04T12:00:00"
}
```

**Authentication**: Uses `GITHUB_TOKEN` environment variable if available

**Rate Limiting**: Respects GitHub rate limits (5000/hour authenticated, 60/hour unauthenticated)

### Script 03: Paper Extraction

**Purpose**: Download and extract paper content from ArXiv

**Data Source**: ArXiv API + PDF downloads

**Processing**:
1. Download PDFs using `arxiv` library
2. Extract text using Metis PDFExtractor
3. Identify sections (abstract, introduction, methods, experiments, conclusion)
4. Store metadata and extracted text

**Output**:
- `data/case_study/papers/1706.03762.pdf`
- `data/case_study/papers/1710.09829.pdf`
- `data/case_study/papers/extracted/transformers_extracted.json`
- `data/case_study/papers/extracted/capsules_extracted.json`
- `data/case_study/papers/extracted/all_papers.json`

**Output Schema**:
```json
{
  "paper_name": "transformers",
  "arxiv_id": "1706.03762",
  "metadata": {
    "title": "Attention Is All You Need",
    "authors": ["Ashish Vaswani", ...],
    "abstract": "...",
    "published": "2017-06-12T00:00:00",
    "primary_category": "cs.CL",
    "categories": ["cs.CL", "cs.AI"],
    "pdf_url": "...",
    "arxiv_url": "..."
  },
  "sections": {
    "abstract": {"text": "..."},
    "introduction": {"text": "..."},
    "method": {"text": "..."}
  },
  "pdf_path": "data/case_study/papers/1706.03762.pdf"
}
```

**Dependencies**: Requires `metis[pdf]` extra (Docling)

### Script 04: Boundary Objects Collection

**Purpose**: Collect documentation, code examples, and tutorials

**Data Sources**:
- GitHub repositories (official and community)
- Wayback Machine (historical documentation)

**Collection Strategy**:

**For Transformers** (official code available):
- Clone `tensorflow/tensor2tensor` at earliest commit after paper release
- Extract README, documentation, example scripts
- Capture early tutorials and blog posts

**For Capsules** (no official code):
- Note absence of official implementation
- Collect first community implementations
- Document quality comparison

**Output**:
- `data/case_study/boundary_objects/transformers/boundary_objects.json`
- `data/case_study/boundary_objects/capsules/boundary_objects.json`

**Output Schema**:
```json
{
  "paper": "transformers",
  "arxiv_id": "1706.03762",
  "total_objects": 15,
  "boundary_objects": [
    {
      "type": "documentation",
      "source": "tensor2tensor",
      "file": "README.md",
      "content": "..."
    },
    {
      "type": "code_example",
      "source": "tensor2tensor",
      "file": "examples/transformer.py",
      "content": "..."
    }
  ],
  "collected_at": "2025-10-04T12:00:00"
}
```

**Note**: This script may have errors if repositories are no longer available or have been restructured. This is expected and logged.

### Script 05: Embeddings Generation

**Purpose**: Generate Jina v4 embeddings and store in ArangoDB

**Processing**:
1. Load extracted papers and boundary objects
2. Generate embeddings for all text content
3. Store in ArangoDB with metadata

**ArangoDB Schema**:

**Collection**: `case_study_papers`

**Paper Document**:
```json
{
  "_key": "1706_03762",
  "paper_name": "transformers",
  "title": "Attention Is All You Need",
  "authors": ["Vaswani", "Shazeer", ...],
  "arxiv_id": "1706.03762",
  "published_date": "2017-06-12",
  "abstract": "...",
  "primary_category": "cs.CL",
  "categories": ["cs.CL", "cs.AI"],
  "sections": {
    "abstract": {
      "text": "...",
      "embedding": [0.123, 0.456, ...]  // 2048-dim Jina v4
    },
    "introduction": {
      "text": "...",
      "embedding": [...]
    }
  }
}
```

**Boundary Object Document**:
```json
{
  "_key": "transformers_tensor2tensor_README_md",
  "paper_name": "transformers",
  "type": "documentation",
  "source": "tensor2tensor",
  "file": "README.md",
  "content": "...",
  "embedding": [0.123, 0.456, ...]  // 2048-dim Jina v4
}
```

**Database Connection**: Unix socket at `/tmp/arangodb.sock`

**Embedding Model**: Jina v4 (2048 dimensions)

### Script 06: Visualization Generation

**Purpose**: Create interactive and static visualizations

**Visualizations Created**:

1. **Paper Sections Semantic Map** (`semantic_map_papers.html`)
   - 2D UMAP projection of paper section embeddings
   - Color-coded by paper
   - Interactive hover showing section details

2. **Full Ecosystem Map** (`full_ecosystem.html`)
   - Papers + boundary objects combined
   - Circles = paper sections, Diamonds = boundary objects
   - Shows semantic relationships

3. **Citation Timeline** (`citation_timeline.html`)
   - Cumulative citations over time
   - Comparison between papers

4. **Repository Comparison** (`repository_comparison.html`)
   - Repository count by type
   - Total stars comparison

**Output Formats**:
- HTML (interactive with Plotly)
- PNG (static images)
- JSON (raw coordinates for further analysis)

**Output Location**: `data/case_study/visualizations/`

**Dimensionality Reduction**: UMAP (n_neighbors=15, min_dist=0.1)

## Configuration

Edit `config/case_study.yaml` to customize:

```yaml
papers:
  transformers:
    arxiv_id: "1706.03762"
    title: "Attention Is All You Need"
    published_date: "2017-06-12"
    official_code: "https://github.com/tensorflow/tensor2tensor"

collection:
  github:
    min_stars: 10
    min_forks: 5
    max_results_per_query: 100
    search_queries:
      transformers:
        - "attention is all you need"
        - "transformer attention mechanism"

  citations:
    start_year: 2017
    end_year: 2025

api:
  semantic_scholar:
    requests_per_second: 1.5

  github:
    rate_limit: 5000  # per hour
```

## Testing

Run tests:

```bash
cd /home/todd/olympus/metis
pytest tests/case_study/ -v
```

Tests cover:
- Utility functions (logging, rate limiting, retry logic)
- Data validation (citation data, repository data)
- File operations (incremental saving, backups)

## Output Structure

```
data/case_study/
├── citations/
│   ├── transformers_citations.json
│   └── capsules_citations.json
├── implementations/
│   ├── transformers_repos.json
│   └── capsules_repos.json
├── papers/
│   ├── 1706.03762.pdf
│   ├── 1710.09829.pdf
│   └── extracted/
│       ├── transformers_extracted.json
│       ├── capsules_extracted.json
│       └── all_papers.json
├── boundary_objects/
│   ├── transformers/
│   │   └── boundary_objects.json
│   └── capsules/
│       └── boundary_objects.json
└── visualizations/
    ├── semantic_map_papers.html
    ├── semantic_map_papers.png
    ├── full_ecosystem.html
    ├── full_ecosystem.png
    ├── citation_timeline.html
    ├── citation_timeline.png
    ├── repository_comparison.html
    ├── repository_comparison.png
    └── embedding_coordinates.json
```

## Troubleshooting

### GitHub Rate Limiting

**Problem**: "Rate limit exceeded" errors from GitHub API

**Solution**: Set `GITHUB_TOKEN` environment variable:
```bash
export GITHUB_TOKEN=ghp_your_token_here
```

This increases limit from 60/hour to 5000/hour.

### ArangoDB Connection Errors

**Problem**: "Connection refused" or socket errors

**Solution**:
1. Check ArangoDB is running
2. Verify socket path in configuration
3. Ensure socket permissions allow access

### PDF Extraction Errors

**Problem**: "PDFExtractor not available"

**Solution**: Install PDF extras:
```bash
poetry install -E pdf
```

### Memory Issues with Embeddings

**Problem**: Out of memory when generating embeddings

**Solution**:
1. Process in smaller batches (modify script)
2. Ensure sufficient GPU memory (16GB+ recommended)
3. Use CPU if GPU unavailable (slower but uses system RAM)

### Missing Boundary Objects

**Problem**: Few or no boundary objects collected

**Solution**: This is expected for some papers (e.g., Capsules had no official code). The script logs warnings but continues. Check logs for details.

## Dependencies

Core dependencies (from `pyproject.toml`):

```toml
arxiv = "^2.1.0"           # ArXiv API client
PyGithub = "^2.1.1"        # GitHub API client
requests = "^2.31.0"       # HTTP requests
pyyaml = "^6.0.1"          # Config loading
plotly = "^5.18.0"         # Visualizations
umap-learn = "^0.5.5"      # Dimensionality reduction
waybackpy = "^3.0.6"       # Wayback Machine API
kaleido = "0.2.1"          # PNG export for Plotly
```

Metis dependencies (already installed):

```toml
sentence-transformers = "^5.1.1"  # For Jina embeddings
docling = "^2.54.0"               # PDF extraction
httpx = "^0.27.0"                 # ArangoDB client
```

## Performance

Expected runtime (with good internet connection):

- Script 01 (Citations): ~30 seconds
- Script 02 (GitHub): ~5-10 minutes (rate limited)
- Script 03 (Papers): ~2 minutes
- Script 04 (Boundary Objects): ~10-15 minutes (repo cloning)
- Script 05 (Embeddings): ~3-5 minutes (GPU) or ~15-20 minutes (CPU)
- Script 06 (Visualizations): ~1-2 minutes

**Total**: ~25-35 minutes for full pipeline

## Data Quality Checks

After running the pipeline, verify:

1. **Citation data**: Both papers have monthly data from 2017-2025
2. **GitHub repos**: At least 50 repositories per paper
3. **Papers**: Both PDFs downloaded and extracted
4. **Boundary objects**: At least 10 objects for Transformers
5. **Embeddings**: All sections have 2048-dim embeddings
6. **Visualizations**: All HTML files open and render correctly

Check logs at `logs/case_study_collection.log` for warnings or errors.

## Next Steps

After data collection:

1. **Review visualizations** - Open HTML files in browser
2. **Validate data quality** - Check JSON files for completeness
3. **Run analysis** - Use Jupyter notebook for statistical analysis
4. **Calculate conveyance scores** - Apply Conveyance Framework metrics
5. **Write case study** - Use data for anthropological analysis

## Support

For issues:

1. Check logs: `logs/case_study_collection.log`
2. Review configuration: `config/case_study.yaml`
3. Run tests: `pytest tests/case_study/`
4. Check individual script outputs

## License

Apache 2.0 (same as Metis project)
