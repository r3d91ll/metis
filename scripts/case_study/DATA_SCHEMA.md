# Case Study Data Schema Reference

This document provides detailed schema definitions for all data collected by the pipeline.

## Citation Data

**File**: `data/case_study/citations/{paper_name}_citations.json`

```typescript
interface CitationData {
  paper_id: string;              // ArXiv ID (e.g., "1706.03762")
  title: string;                 // Paper title
  authors: string[];             // List of author names
  publication_date: string;      // ISO 8601 date (e.g., "2017-06-12")
  total_citations: number;       // Total citation count
  monthly_citations: MonthlyCount[];
  collected_at: string;          // ISO 8601 timestamp
}

interface MonthlyCount {
  year: number;                  // Year (e.g., 2017)
  month: number;                 // Month (1-12)
  count: number;                 // Citation count for this month
}
```

**Example**:
```json
{
  "paper_id": "1706.03762",
  "title": "Attention Is All You Need",
  "authors": ["Ashish Vaswani", "Noam Shazeer"],
  "publication_date": "2017-06-12",
  "total_citations": 35000,
  "monthly_citations": [
    {"year": 2017, "month": 6, "count": 5},
    {"year": 2017, "month": 7, "count": 12}
  ],
  "collected_at": "2025-10-04T12:00:00"
}
```

## GitHub Repository Data

**File**: `data/case_study/implementations/{paper_name}_repos.json`

```typescript
interface RepositoryData {
  paper: string;                 // Paper identifier (e.g., "transformers")
  arxiv_id: string;              // ArXiv ID
  title: string;                 // Paper title
  total_repositories: number;    // Total count
  type_counts: {[key: string]: number};  // Count by repository type
  repositories: Repository[];
  collected_at: string;          // ISO 8601 timestamp
}

interface Repository {
  url: string;                   // GitHub URL
  full_name: string;             // owner/repo
  created_at: string;            // ISO 8601 timestamp
  stars: number;                 // Star count
  forks: number;                 // Fork count
  language: string;              // Primary language
  type: RepositoryType;          // Classification
  description: string;           // Repository description
  from_authors: boolean;         // True if from paper authors
  has_wiki: boolean;            // True if wiki exists
  open_issues: number;          // Open issue count
  topics: string[];             // GitHub topics
  last_updated: string;         // ISO 8601 timestamp
}

type RepositoryType =
  | "official"      // From paper authors
  | "framework"     // Integrated into major framework
  | "tutorial"      // Educational implementation
  | "application"   // Used for specific task
  | "research";     // Extension or variation
```

**Example**:
```json
{
  "paper": "transformers",
  "arxiv_id": "1706.03762",
  "title": "Attention Is All You Need",
  "total_repositories": 50,
  "type_counts": {
    "official": 1,
    "framework": 5,
    "tutorial": 20,
    "application": 15,
    "research": 9
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
      "description": "Library of deep learning models",
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

## Paper Extraction Data

**File**: `data/case_study/papers/extracted/{paper_name}_extracted.json`

```typescript
interface PaperExtraction {
  paper_name: string;            // Paper identifier
  arxiv_id: string;              // ArXiv ID
  metadata: PaperMetadata;
  sections: {[key: string]: Section};
  pdf_path: string;              // Relative path to PDF
}

interface PaperMetadata {
  title: string;
  authors: string[];
  abstract: string;
  published: string;             // ISO 8601 timestamp
  updated: string;               // ISO 8601 timestamp
  primary_category: string;      // Primary ArXiv category
  categories: string[];          // All ArXiv categories
  pdf_url: string;               // ArXiv PDF URL
  arxiv_url: string;             // ArXiv entry URL
}

interface Section {
  text: string;                  // Section text content
}
```

**Example**:
```json
{
  "paper_name": "transformers",
  "arxiv_id": "1706.03762",
  "metadata": {
    "title": "Attention Is All You Need",
    "authors": ["Ashish Vaswani", "Noam Shazeer"],
    "abstract": "The dominant sequence transduction models...",
    "published": "2017-06-12T00:00:00",
    "updated": "2017-08-02T00:00:00",
    "primary_category": "cs.CL",
    "categories": ["cs.CL", "cs.LG", "cs.AI"],
    "pdf_url": "https://arxiv.org/pdf/1706.03762",
    "arxiv_url": "https://arxiv.org/abs/1706.03762"
  },
  "sections": {
    "abstract": {
      "text": "The dominant sequence transduction models..."
    },
    "introduction": {
      "text": "Recurrent neural networks..."
    }
  },
  "pdf_path": "data/case_study/papers/1706.03762.pdf"
}
```

## Boundary Objects Data

**File**: `data/case_study/boundary_objects/{paper_name}/boundary_objects.json`

```typescript
interface BoundaryObjectData {
  paper: string;                 // Paper identifier
  arxiv_id: string;              // ArXiv ID
  total_objects: number;         // Total count
  boundary_objects: BoundaryObject[];
  collected_at: string;          // ISO 8601 timestamp
}

interface BoundaryObject {
  type: ObjectType;              // Object classification
  source: string;                // Source repository/site
  file: string;                  // Relative file path
  content: string;               // Full text content
}

type ObjectType =
  | "documentation"              // Official documentation
  | "code_example"               // Code example/tutorial
  | "community_documentation";   // Community-created docs
```

**Example**:
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
      "content": "# Tensor2Tensor\n\nTensor2Tensor..."
    },
    {
      "type": "code_example",
      "source": "tensor2tensor",
      "file": "examples/transformer.py",
      "content": "# Transformer example\nimport tensorflow..."
    }
  ],
  "collected_at": "2025-10-04T12:00:00"
}
```

## ArangoDB Schema

### Collection: `case_study_papers`

**Paper Document**:

```typescript
interface PaperDocument {
  _key: string;                  // ArXiv ID with underscores (e.g., "1706_03762")
  _id?: string;                  // Auto-generated by ArangoDB
  _rev?: string;                 // Auto-generated by ArangoDB
  paper_name: string;            // Paper identifier
  title: string;                 // Paper title
  authors: string[];             // Author names
  arxiv_id: string;              // Original ArXiv ID
  published_date: string;        // ISO 8601 date
  abstract: string;              // Paper abstract
  primary_category: string;      // Primary ArXiv category
  categories: string[];          // All categories
  sections: {[key: string]: EmbeddedSection};
}

interface EmbeddedSection {
  text: string;                  // Section text
  embedding: number[];           // 2048-dim Jina v4 embedding
}
```

**Boundary Object Document**:

```typescript
interface BoundaryObjectDocument {
  _key: string;                  // Composite key: {paper}_{source}_{file}
  _id?: string;                  // Auto-generated
  _rev?: string;                 // Auto-generated
  paper_name: string;            // Paper identifier
  type: string;                  // Object type
  source: string;                // Source repository
  file: string;                  // File path
  content: string;               // Full text
  embedding: number[];           // 2048-dim Jina v4 embedding
}
```

**Indexes**:
- Hash index on `arxiv_id` (unique)

## Visualization Data

### Embedding Coordinates

**File**: `data/case_study/visualizations/embedding_coordinates.json`

```typescript
interface EmbeddingCoordinates {
  coordinates: number[][];       // Nx2 UMAP coordinates
  labels: string[];              // Point labels
  metadata: PointMetadata[];     // Point metadata
}

interface PointMetadata {
  type: "paper_section" | "boundary_object";
  paper: string;                 // Paper identifier

  // For paper_section type:
  section?: string;              // Section name
  title?: string;                // Paper title

  // For boundary_object type:
  object_type?: string;          // Object type
  source?: string;               // Source repository
}
```

**Example**:
```json
{
  "coordinates": [
    [1.23, -0.45],
    [2.34, 1.56]
  ],
  "labels": [
    "transformers: abstract",
    "transformers: documentation"
  ],
  "metadata": [
    {
      "type": "paper_section",
      "paper": "transformers",
      "section": "abstract",
      "title": "Attention Is All You Need"
    },
    {
      "type": "boundary_object",
      "paper": "transformers",
      "object_type": "documentation",
      "source": "tensor2tensor"
    }
  ]
}
```

## Data Validation

### Citation Data Validation

```python
def validate_citation_data(data: dict) -> bool:
    """Validate citation data structure."""
    assert "paper_id" in data
    assert "title" in data
    assert "monthly_citations" in data

    for month_data in data["monthly_citations"]:
        assert "year" in month_data
        assert "month" in month_data
        assert "count" in month_data

    if "total_citations" in data:
        expected = sum(m["count"] for m in data["monthly_citations"])
        assert data["total_citations"] == expected

    return True
```

### Repository Data Validation

```python
def validate_repo_data(data: dict) -> bool:
    """Validate repository data structure."""
    assert "paper" in data
    assert "repositories" in data

    valid_types = ["official", "framework", "tutorial", "application", "research"]

    for repo in data["repositories"]:
        assert "url" in repo
        assert "created_at" in repo
        assert "type" in repo
        assert repo["type"] in valid_types
        assert "stars" in repo
        assert "language" in repo

    return True
```

## Field Definitions

### Common Fields

- **ISO 8601 Timestamp**: `YYYY-MM-DDTHH:MM:SS` or `YYYY-MM-DDTHH:MM:SSZ`
- **ArXiv ID**: Format `YYMM.NNNNN` (e.g., `1706.03762`)
- **Paper Identifier**: Lowercase slug (e.g., `transformers`, `capsules`)

### Repository Types

- **official**: Repository from paper authors or official release
- **framework**: Integration into major ML framework (TensorFlow, PyTorch, etc.)
- **tutorial**: Educational implementation with documentation
- **application**: Production use for specific task
- **research**: Academic extension or variation

### Object Types

- **documentation**: Official documentation (README, docs, guides)
- **code_example**: Working code examples or tutorials
- **community_documentation**: Community-created documentation

### ArXiv Categories

Common categories:
- `cs.CL` - Computation and Language
- `cs.LG` - Machine Learning
- `cs.AI` - Artificial Intelligence
- `cs.CV` - Computer Vision
- `stat.ML` - Machine Learning (Statistics)

## Data Sizes

Typical sizes for full dataset:

- **Citation JSON**: ~10-50 KB per paper
- **Repository JSON**: ~100-500 KB per paper
- **Paper Extraction JSON**: ~50-200 KB per paper
- **Boundary Objects JSON**: ~500 KB - 5 MB per paper
- **Embedding Coordinates JSON**: ~1-5 MB
- **PDFs**: ~1-2 MB per paper
- **ArangoDB**: ~100-500 MB (includes embeddings)

## Notes

1. All timestamps are in UTC
2. All text is UTF-8 encoded
3. Embeddings are 2048-dimensional float arrays
4. File paths are relative to project root
5. JSON files use 2-space indentation
