# Data Collection System for Transformers vs Capsules Case Study

## Project Context

I'm building an anthropological case study comparing the cultural adoption of two AI papers from NIPS 2017:

- **Transformers** ("Attention Is All You Need" - arXiv:1706.03762)
- **Capsule Networks** ("Dynamic Routing Between Capsules" - arXiv:1710.09829)

This is part of my Conveyance Framework research validating a mathematical model of knowledge transfer. I need to collect empirical data showing how these papers diffused through academic and practitioner communities.

## Existing Infrastructure

**Metis System** (already implemented):

- ArangoDB graph database (Unix socket connection at `/tmp/arangodb.sock`)
- Jina v4 embeddings for semantic analysis
- PDF/LaTeX extraction pipeline
- Python 3.11 environment
- 256GB RAM, 2x RTX A6000 GPUs available

**Location**: Project root at `/home/todd/projects/metis/`

## Your Task

Create a complete data collection pipeline on a new git branch `feature/case-study-data-collection` that gathers all empirical evidence for the Transformers vs Capsules comparison.

## Data Collection Requirements

### 1. Citation Timeline Data (Monthly, 2017-Present)

**Data Source**: Semantic Scholar API (free, no key needed)

- Endpoint: `https://api.semanticscholar.org/v1/paper/arXiv:{arxiv_id}`
- Get citations with timestamps for both papers
- Aggregate into monthly buckets

**Output Format**:

```json
{
  "paper_id": "1706.03762",
  "title": "Attention Is All You Need",
  "monthly_citations": [
    {"year": 2017, "month": 6, "count": 5},
    {"year": 2017, "month": 7, "count": 12},
    ...
  ],
  "total_citations": 35000
}
```

**Store in**: `data/case_study/citations/transformers_citations.json` and `capsules_citations.json`

### 2. GitHub Implementation Tracking

**Data Source**: GitHub API

- Search for repositories implementing each paper
- Track creation date, stars, forks, language
- Classify by implementation type

**Search Queries**:

```python
transformer_queries = [
    "attention is all you need",
    "transformer attention mechanism",
    "vaswani transformer",
    "self-attention implementation"
]

capsule_queries = [
    "capsule networks",
    "dynamic routing capsules",
    "hinton capsules",
    "capsnet"
]
```

**Filters**:

- Created after paper publication date
- Minimum 10 stars (community validation)
- Has actual code (not just forks)
- Language: Python, PyTorch, TensorFlow

**Classification Types**:

- `official`: From paper authors
- `framework`: Integrated into major framework
- `tutorial`: Educational implementation
- `application`: Used for specific task
- `research`: Extension or variation

**Output Format**:

```json
{
  "paper": "transformers",
  "repositories": [
    {
      "url": "https://github.com/tensorflow/tensor2tensor",
      "created_at": "2017-06-12T00:00:00Z",
      "stars": 15000,
      "forks": 3000,
      "language": "Python",
      "type": "official",
      "description": "...",
      "from_authors": true
    }
  ]
}
```

**Store in**: `data/case_study/implementations/transformers_repos.json` and `capsules_repos.json`

### 3. Paper Content Extraction and Embeddings

**Data Source**: ArXiv API + PDFs

- Download both papers
- Extract text by section (abstract, intro, methods, results, conclusion)
- Generate Jina v4 embeddings for each section
- Store in ArangoDB

**ArXiv IDs**:

- Transformers: `1706.03762`
- Capsules: `1710.09829`

**Process**:

1. Download PDFs using `arxiv` library
2. Extract sections using existing Metis PDF extraction
3. Generate embeddings using Jina v4
4. Store in ArangoDB collections

**ArangoDB Schema**:

```javascript
// Collection: case_study_papers
{
  "_key": "1706.03762",
  "title": "Attention Is All You Need",
  "authors": ["Vaswani", "Shazeer", ...],
  "arxiv_id": "1706.03762",
  "published_date": "2017-06-12",
  "venue": "NIPS 2017",
  "sections": {
    "abstract": {
      "text": "...",
      "embedding": [...]  // 1024-dim Jina v4
    },
    "introduction": {...},
    "methods": {...}
  }
}
```

### 4. Boundary Objects Collection (C_ext Evidence)

**For Transformers** (released same day as paper):

- Official TensorFlow implementation (tensor2tensor)
- Documentation from original release
- Early tutorials and blog posts
- Model checkpoints and configs

**For Capsules** (no official code):

- Note absence of official implementation
- First community implementations (with dates)
- Documentation quality comparison

**Collection Tasks**:

**4a. Official Code Repositories**:

- Clone `tensorflow/tensor2tensor` at earliest commit after paper (June 2017)
- Extract README, documentation, example scripts
- Generate embeddings for all documentation

**4b. Documentation Snapshots**:

- Use Wayback Machine API for original docs
- Capture blog posts announcing releases
- Tutorial content from 2017-2018

**4c. Embed All Boundary Objects**:

```python
# For each boundary object:
boundary_objects = [
    {"type": "code", "source": "tensor2tensor", "file": "README.md"},
    {"type": "docs", "source": "tensor2tensor", "file": "transformer.md"},
    {"type": "tutorial", "source": "blog", "url": "..."},
]

# Generate embeddings
for obj in boundary_objects:
    obj["embedding"] = jina_embed(obj["content"])
```

**Store in**: `data/case_study/boundary_objects/transformers/` and `capsules/`

### 5. Semantic Maps Generation

**Create visualizations**:

**5a. Paper-only semantic map**:

- 2D UMAP projection of paper embeddings
- Include ~50 related ML papers from 2017 for context
- Color-code by paper type

**5b. Paper + Boundary Objects map**:

- Show papers + documentation + code + tutorials
- Draw edges showing citation/dependency links
- Size nodes by community impact (stars, citations)

**5c. Full ecosystem map**:

- Papers + boundary objects + implementations
- Show temporal evolution (animate or color by date)
- Highlight "bridge" documents connecting paper to implementations

**Visualization Requirements**:

- Use Plotly for interactive HTML outputs
- Generate static PNGs for papers/presentations
- Save raw coordinate data for further analysis

**Store in**: `data/case_study/visualizations/`

## Technical Specifications

### File Structure

```text
metis/
├── data/
│   └── case_study/
│       ├── citations/
│       │   ├── transformers_citations.json
│       │   └── capsules_citations.json
│       ├── implementations/
│       │   ├── transformers_repos.json
│       │   └── capsules_repos.json
│       ├── papers/
│       │   ├── 1706.03762.pdf
│       │   ├── 1710.09829.pdf
│       │   └── extracted/
│       ├── boundary_objects/
│       │   ├── transformers/
│       │   └── capsules/
│       └── visualizations/
│           ├── semantic_map_papers.html
│           ├── semantic_map_with_docs.html
│           └── full_ecosystem.html
├── scripts/
│   └── case_study/
│       ├── 01_collect_citations.py
│       ├── 02_collect_github_repos.py
│       ├── 03_extract_papers.py
│       ├── 04_collect_boundary_objects.py
│       ├── 05_generate_embeddings.py
│       ├── 06_create_visualizations.py
│       └── run_all.sh
└── notebooks/
    └── case_study_analysis.ipynb
```

### Code Requirements

**Use existing Metis components**:

```python
# Import from existing Metis codebase
from metis.database import ArangoDBClient
from metis.embeddings import JinaEmbedder
from metis.extraction import PDFExtractor

# Connect to ArangoDB
db = ArangoDBClient(unix_socket="/tmp/arangodb.sock")

# Use Jina for embeddings
embedder = JinaEmbedder(model="jina-embeddings-v4")

# Extract PDFs
extractor = PDFExtractor()
```

**Error Handling**:

- Retry logic for API calls (3 retries with exponential backoff)
- Rate limiting (respect API limits: Semantic Scholar 100/min, GitHub 5000/hour)
- Save progress incrementally (don't lose work if script fails)
- Log all errors to `logs/case_study_collection.log`

**Configuration**:

```python
# config/case_study.yaml
papers:
  transformers:
    arxiv_id: "1706.03762"
    title: "Attention Is All You Need"
    published_date: "2017-06-12"
    official_code: "https://github.com/tensorflow/tensor2tensor"
  
  capsules:
    arxiv_id: "1710.09829"
    title: "Dynamic Routing Between Capsules"
    published_date: "2017-10-26"
    official_code: null

api:
  semantic_scholar:
    base_url: "https://api.semanticscholar.org/v1"
    rate_limit: 100  # per minute
  
  github:
    rate_limit: 5000  # per hour
    # Will use GITHUB_TOKEN env var if available

collection:
  github:
    min_stars: 10
    min_forks: 5
    max_results_per_query: 100
  
  citations:
    start_year: 2017
    end_year: 2025
```

### Dependencies to Add

```toml
# Add to pyproject.toml
[tool.poetry.dependencies]
arxiv = "^2.1.0"
PyGithub = "^2.1.1"
requests = "^2.31.0"
pyyaml = "^6.0.1"
plotly = "^5.18.0"
umap-learn = "^0.5.5"
waybackpy = "^3.0.6"  # For Wayback Machine API
```

### Testing Requirements

**Unit Tests** (`tests/case_study/`):

- Test API response parsing
- Test embedding generation
- Test data validation
- Mock API calls (don't hit real APIs in tests)

**Integration Tests**:

- Test end-to-end pipeline with sample data
- Verify ArangoDB storage
- Check visualization generation

**Data Validation**:

```python
# Validate collected data
def validate_citation_data(data):
    assert "monthly_citations" in data
    assert all("year" in m and "month" in m and "count" in m 
               for m in data["monthly_citations"])
    assert data["total_citations"] == sum(m["count"] for m in data["monthly_citations"])

def validate_repo_data(data):
    assert "repositories" in data
    for repo in data["repositories"]:
        assert "url" in repo
        assert "created_at" in repo
        assert "type" in repo
        assert repo["type"] in ["official", "framework", "tutorial", "application", "research"]
```

## Execution Plan

**Phase 1: Setup** (Day 1)

- Create branch `feature/case-study-data-collection`
- Set up directory structure
- Add dependencies
- Create configuration file
- Write base utilities (API clients, retry logic, logging)

**Phase 2: Data Collection** (Days 2-4)

- Script 01: Citation data (Semantic Scholar)
- Script 02: GitHub repositories
- Script 03: Paper extraction
- Script 04: Boundary objects collection
- Test each script independently

**Phase 3: Processing** (Days 5-6)

- Script 05: Generate all embeddings
- Store everything in ArangoDB
- Validate data quality
- Run data validation tests

**Phase 4: Visualization** (Day 7)

- Script 06: Create semantic maps
- Generate interactive visualizations
- Export static versions
- Create summary notebook

**Phase 5: Documentation** (Day 8)

- Write README for case_study scripts
- Document data schemas
- Create usage examples
- Write collection report

## Success Criteria

**Data Completeness**:

- [ ] Citation data for both papers (monthly from 2017-2025)
- [ ] At least 50 GitHub repositories per paper
- [ ] Full text extraction for both papers
- [ ] At least 10 boundary objects for Transformers
- [ ] All data stored in ArangoDB
- [ ] All visualizations generated

**Code Quality**:

- [ ] All scripts run without errors
- [ ] Tests pass with >80% coverage
- [ ] Code follows Metis project conventions
- [ ] Comprehensive error handling
- [ ] Clear logging and progress indicators

**Documentation**:

- [ ] README explains how to run each script
- [ ] Data schemas documented
- [ ] API usage documented
- [ ] Example outputs provided

## Deliverables

**Pull Request** should include:

1. All collection scripts (working and tested)
2. Configuration files
3. Collected data (or script to collect it)
4. ArangoDB schemas and migrations
5. Visualizations (HTML and PNG)
6. Jupyter notebook with initial analysis
7. Documentation (README, data dictionary)
8. Test suite

**Final Output**: Complete dataset enabling me to:

- Calculate conveyance scores (W, R, C_ext, P_ij, T)
- Show differential adoption patterns
- Validate framework predictions
- Create compelling visualizations for papers

## Notes

**Don't worry about**:

- Rubric scoring (I'll do manual assessment)
- Statistical analysis (handled separately)
- Writing interpretation (anthropological analysis is my job)

**Do focus on**:

- Robust data collection
- Clean, reproducible code
- Comprehensive documentation
- Quality visualizations

**Environment**:

- Use existing Metis virtual environment
- GPU available but not required for this task
- Can run long-running scripts overnight
- Rate limits are important - implement proper throttling

**Questions/Clarifications**:
If you need any clarification about:

- Metis infrastructure details
- ArangoDB schema decisions
- Specific API endpoints
- Visualization preferences

Please ask before implementing. I want this to integrate seamlessly with the existing Metis system.

---

**Start by**:

1. Creating the branch
2. Setting up the directory structure
3. Writing a project plan showing what you'll implement
4. Then proceed with implementation

I'll be working on dataset sorting while you handle the implementation. Let's make this a clean, reproducible data collection pipeline!
