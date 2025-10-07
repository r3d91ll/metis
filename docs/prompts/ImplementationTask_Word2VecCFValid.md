# Implementation Task: Word2Vec CF Validation Experiment

## Your Mission

You are implementing the first Conveyance Framework (CF) validation experiment using the Word2Vec paper family. Your task is to assess existing infrastructure ("lab equipment"), identify what needs to be built, and propose an implementation plan.

---

## Experiment Objective

**Research Question**: Does the adoption of embedding papers follow CF's predicted super-linear context amplification (α ∈ [1.5, 2.0])?

**Test Corpus**: Word2Vec family papers with associated code

- Word2Vec (arXiv: 1301.3781)
- Doc2Vec (arXiv: 1405.4053)
- GloVe (arXiv: 1504.06654)
- FastText (arXiv: 1607.04606)
- Node2Vec (arXiv: 1607.00653)

**Target Output**: Combined paper+code embeddings in same vector space for CF measurement

---

## Required Capabilities

### 1. Paper Processing Pipeline

- Download arXiv PDFs on-demand (5-10 papers at a time)
- Convert PDF → clean markdown
- Store markdown in ArangoDB collection `arxiv_markdown`

### 2. Code Discovery & Integration

- Find associated GitHub repositories for each paper
- Download/clone code repositories
- Store code in ArangoDB collection `arxiv_code`

### 3. Combined Embedding Generation

- Combine paper markdown + code into single 32k context window
- Structure: `"This is the [TITLE] paper. CONTENT: [markdown]. CODE: [code]"`
- Generate Jina v4 embeddings with late chunking
- Store embeddings in ArangoDB collection `arxiv_embeddings`

### 4. Experiment Reproducibility

- Hard-code the Word2Vec family paper list
- Make this a reusable template for future experiments (physics papers, etc.)
- Use factory pattern for swappable components

---

## Your Task Breakdown

### STEP 1: Workspace Inventory

**Search the workspace and identify:**

1. **Existing "Lab Equipment"** (what we already have):
   - Docling integration for PDF → markdown
   - jina_embedder_v4 module (tested and working)
   - ArangoDB connection infrastructure
   - Existing Metis utilities (logging, config, etc.)

2. **Component Status Assessment**:
   - What works out-of-the-box?
   - What needs configuration/wiring?
   - What's completely missing?

3. **Integration Points**:
   - How do existing components connect?
   - What interfaces are available?
   - What patterns are already established?

### STEP 2: Gap Analysis

**Identify what needs to be built:**

- [ ] arXiv PDF download functionality
- [ ] Docling → ArangoDB integration pipeline
- [ ] GitHub repository discovery logic
- [ ] Code repository download/storage
- [ ] Combined paper+code context window builder
- [ ] Jina v4 → ArangoDB embedding storage
- [ ] Experiment orchestration class
- [ ] Quality validation tooling
- [ ] Configuration management
- [ ] Error handling and retry logic

### STEP 3: Implementation Plan

**Propose a plan that includes:**

1. **Architecture Diagram**: How components fit together
2. **Component List**: What needs building vs. what's reused
3. **Factory Interfaces**: What needs to be swappable
4. **Data Flow**: How data moves through the pipeline
5. **Implementation Order**: What to build first, dependencies
6. **Testing Strategy**: How to validate each component
7. **Experiment Execution**: How to run the Word2Vec experiment

---

## Design Constraints

### Reusability Requirements

- **Experiment Template**: Word2Vec is first, but physics papers, bio papers, etc. coming
- **Factory Pattern**: PDF converter, storage backend, quality validators should be swappable
- **Hard-coded Specifics**: Paper lists, hypotheses, expected α ranges per experiment

### Infrastructure Patterns

- **Follow existing Metis patterns**: Use established logging, config, error handling
- **Use existing components**: Don't rebuild what we have (docling, jina_embedder_v4)
- **ArangoDB schema**: Follow established collection patterns

### Scale & Performance

- **Processing scope**: 5-10 papers at a time (not bulk processing)
- **Interactive mode**: Real-time progress feedback
- **Quality over speed**: Conversion quality matters more than throughput
- **Memory footprint**: <4GB for typical operations

---

## Expected Deliverables from Your Analysis

### 1. Inventory Report

```markdown
## Existing Infrastructure
- **Docling**: [status, location, capabilities]
- **Jina Embedder v4**: [status, location, interface]
- **ArangoDB**: [connection pattern, existing collections]
- **Utilities**: [logging, config, other useful modules]

## Integration Assessment
- What works together already?
- What needs adapters/glue code?
- What patterns should we follow?
```

### 2. Gap Analysis

```markdown
## Missing Components
1. Component name
   - Purpose
   - Estimated complexity
   - Dependencies
   
## Required Integrations
1. Integration point
   - What needs connecting
   - Proposed approach
```

### 3. Implementation Plan

```markdown
## Architecture
[Diagram or description of how components fit]

## Component Specification
For each new component:
- Purpose
- Interface (inputs/outputs)
- Dependencies
- Factory pattern applicability

## Build Order
1. First component (why)
2. Second component (why)
...

## Testing Strategy
- Unit tests needed
- Integration tests
- Acceptance criteria

## Experiment Execution Plan
Step-by-step: How to run Word2Vec CF validation
```

---

## Technical Specifications

### ArangoDB Schema

#### Collection: `arxiv_markdown`

```python
{
    "_key": "1301.3781",
    "arxiv_id": "1301.3781",
    "title": "Efficient Estimation of Word Representations...",
    "authors": ["Tomas Mikolov", "Kai Chen", ...],
    "markdown_content": "# Title\n\n## Abstract...",
    "processing_metadata": {
        "tool": "docling",
        "timestamp": "2025-01-06T...",
        "word_count": 3847,
        "section_count": 6
    },
    "experiment_tags": ["word2vec_family", "cf_validation"]
}
```

#### Collection: `arxiv_code`

```python
{
    "_key": "1301.3781_code",
    "arxiv_id": "1301.3781",
    "github_url": "https://github.com/...",
    "code_files": {
        "word2vec.c": "...",
        "word2vec.py": "...",
        ...
    },
    "processing_metadata": {
        "timestamp": "2025-01-06T...",
        "total_files": 12,
        "total_lines": 2847
    }
}
```

#### Collection: `arxiv_embeddings`

```python
{
    "_key": "1301.3781_emb_0",
    "arxiv_id": "1301.3781",
    "chunk_index": 0,
    "combined_content": "Paper: [markdown]... Code: [code]...",
    "embedding": [0.123, -0.456, ...],  # 2048-dim Jina v4
    "embedding_metadata": {
        "model": "jina-embeddings-v4",
        "context_window": 32768,
        "late_chunking": true
    }
}
```

### Factory Interfaces

```python
# PDF Converter Factory
class PDFConverterFactory:
    @staticmethod
    def create(tool: str = "docling") -> PDFConverter:
        """Returns configured PDF converter"""
        pass

# Code Fetcher Factory  
class CodeFetcherFactory:
    @staticmethod
    def create(source: str = "github") -> CodeFetcher:
        """Returns configured code repository fetcher"""
        pass

# Embedder Factory
class EmbedderFactory:
    @staticmethod
    def create(model: str = "jina_v4") -> Embedder:
        """Returns configured embedder"""
        pass
```

### Experiment Template

```python
class CFPaperExperiment:
    """Base template for CF validation experiments"""
    
    PAPERS: List[str] = []  # Override in subclass
    HYPOTHESIS: str = ""     # Override in subclass
    
    def __init__(self, 
                 converter: PDFConverter,
                 code_fetcher: CodeFetcher,
                 embedder: Embedder,
                 storage: StorageBackend):
        pass
    
    def run_experiment(self) -> ExperimentResults:
        """Execute the complete experiment pipeline"""
        pass

class Word2VecCFExperiment(CFPaperExperiment):
    """Reproducible Word2Vec CF validation"""
    
    PAPERS = [
        "1301.3781",  # Word2Vec
        "1405.4053",  # Doc2Vec
        "1504.06654", # GloVe
        "1607.04606", # FastText
        "1607.00653"  # Node2Vec
    ]
    
    HYPOTHESIS = "Embedding paper adoption follows α ∈ [1.5, 2.0]"
```

---

## Success Criteria

Your implementation plan should enable:

- [ ] Download and process 5 Word2Vec papers
- [ ] Find and store associated GitHub code
- [ ] Generate combined paper+code embeddings
- [ ] Store everything in ArangoDB with proper schema
- [ ] Provide clean factory interfaces for future experiments
- [ ] Enable reproducible experiment execution
- [ ] Complete pipeline runs in <30 minutes for 5 papers

---

## Workflow Expectations

1. **Inventory Phase**: Search workspace, document existing infrastructure
2. **Analysis Phase**: Identify gaps, propose architecture
3. **Planning Phase**: Define build order, testing strategy
4. **Presentation**: Show me your plan before implementation

---

## Questions to Answer in Your Plan

1. How will you integrate docling with the PDF download pipeline?
2. How does jina_embedder_v4 interface work? Can we pass combined paper+code?
3. What's the existing ArangoDB connection pattern? How do we add new collections?
4. What's missing for GitHub code discovery and download?
5. How do we chunk the combined 32k context window effectively?
6. What's the cleanest way to make this reusable for physics papers later?
7. Where should configuration live (paper lists, GitHub tokens, etc.)?
8. How do we handle errors (missing code, download failures, etc.)?

---

## Begin Your Analysis

**Start by exploring the workspace and answering:**

1. What do we have?
2. What do we need?
3. How should we build it?

Present your findings and proposed plan before writing any implementation code.
