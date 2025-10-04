#!/bin/bash
# Orchestration script for case study data collection pipeline

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check for GITHUB_TOKEN
if [ -z "$GITHUB_TOKEN" ]; then
    log_warning "GITHUB_TOKEN not set. GitHub API will have low rate limits."
    log_info "Set token with: export GITHUB_TOKEN=your_token_here"
fi

# Create necessary directories
log_info "Creating output directories..."
mkdir -p ../../data/case_study/{citations,implementations,papers/extracted,boundary_objects/{transformers,capsules},visualizations}
mkdir -p ../../logs

log_info "=========================================="
log_info "Case Study Data Collection Pipeline"
log_info "=========================================="
echo ""

# Step 1: Collect citations
log_info "Step 1/6: Collecting citation data from Semantic Scholar..."
if python 01_collect_citations.py; then
    log_success "Citation collection complete"
else
    log_error "Citation collection failed"
    exit 1
fi
echo ""

# Step 2: Collect GitHub repositories
log_info "Step 2/6: Collecting GitHub repository data..."
if python 02_collect_github_repos.py; then
    log_success "GitHub repository collection complete"
else
    log_error "GitHub repository collection failed"
    exit 1
fi
echo ""

# Step 3: Extract papers
log_info "Step 3/6: Downloading and extracting papers from ArXiv..."
if python 03_extract_papers.py; then
    log_success "Paper extraction complete"
else
    log_error "Paper extraction failed"
    exit 1
fi
echo ""

# Step 4: Collect boundary objects
log_info "Step 4/6: Collecting boundary objects (documentation, code)..."
if python 04_collect_boundary_objects.py; then
    log_success "Boundary object collection complete"
else
    log_warning "Boundary object collection had errors (may be expected)"
fi
echo ""

# Step 5: Generate embeddings
log_info "Step 5/6: Generating embeddings and storing in ArangoDB..."
if python 05_generate_embeddings.py; then
    log_success "Embedding generation complete"
else
    log_error "Embedding generation failed"
    exit 1
fi
echo ""

# Step 6: Create visualizations
log_info "Step 6/6: Creating visualizations..."
if python 06_create_visualizations.py; then
    log_success "Visualization generation complete"
else
    log_error "Visualization generation failed"
    exit 1
fi
echo ""

log_success "=========================================="
log_success "Pipeline complete!"
log_success "=========================================="
echo ""
log_info "Output locations:"
log_info "  - Citations: ../../data/case_study/citations/"
log_info "  - GitHub repos: ../../data/case_study/implementations/"
log_info "  - Papers: ../../data/case_study/papers/"
log_info "  - Boundary objects: ../../data/case_study/boundary_objects/"
log_info "  - Visualizations: ../../data/case_study/visualizations/"
log_info "  - Logs: ../../logs/case_study_collection.log"
echo ""
log_info "Next steps:"
log_info "  1. Review collected data for quality"
log_info "  2. Open visualizations in browser"
log_info "  3. Run analysis in Jupyter notebook"
