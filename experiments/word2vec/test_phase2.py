"""
Phase 2 Test: Paper + Code Fetching

Tests complete pipeline for fetching papers and code repositories.
"""

import logging
import sys
import time
from pathlib import Path

import yaml

# Add metis to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.word2vec.arxiv_fetcher import ArxivPaperFetcher
from experiments.word2vec.github_fetcher import GitHubCodeFetcher
from experiments.word2vec.storage import CFExperimentStorage

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_phase2():
    """Test Phase 2: Paper + Code Fetching."""
    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("=" * 80)
    logger.info("Phase 2 Test: Paper + Code Fetching")
    logger.info("=" * 80)

    # Initialize components
    paper_fetcher = ArxivPaperFetcher(
        cache_dir=config["infrastructure"]["cache_dir"],
        max_retries=config["fetching"]["arxiv"]["max_retries"],
    )
    code_fetcher = GitHubCodeFetcher(config)
    storage = CFExperimentStorage(config=config)

    try:
        storage.ensure_collections()

        # Test with Word2Vec paper
        arxiv_id = "1301.3781"
        paper_title = "Efficient Estimation of Word Representations in Vector Space"
        github_url = "https://github.com/tmikolov/word2vec"

        logger.info(f"\nProcessing: {paper_title}")
        logger.info(f"arXiv ID: {arxiv_id}")
        logger.info("-" * 80)

        # Step 1: Fetch paper
        logger.info("\nStep 1: Fetch paper from arXiv")
        start_time = time.time()

        paper_doc = paper_fetcher.fetch_paper(arxiv_id)
        paper_time = time.time() - start_time

        if paper_doc:
            logger.info(f"✓ Paper fetched in {paper_time:.2f}s")
            logger.info(f"  Title: {paper_doc.title[:60]}...")
            logger.info(
                f"  Markdown length: {len(paper_doc.markdown_content)} chars"
            )
            logger.info(
                f"  Word count: {len(paper_doc.markdown_content.split())} words"
            )
            if paper_doc.latex_source:
                logger.info(
                    f"  LaTeX source: {len(paper_doc.latex_source)} chars, "
                    f"{len(paper_doc.latex_source.split())} words"
                )
            else:
                logger.info("  LaTeX source: Not available")

            # Convert PaperDocument to dict for storage
            paper_dict = {
                "title": paper_doc.title,
                "authors": paper_doc.authors,
                "abstract": paper_doc.abstract,
                "markdown_content": paper_doc.markdown_content,
                "latex_source": paper_doc.latex_source,
                "processing_time": paper_doc.processing_time,
            }

            # Store paper
            paper_key = storage.store_paper_markdown(arxiv_id, paper_dict)
            logger.info(f"  Stored with key: {paper_key}")
        else:
            logger.error("✗ Failed to fetch paper")
            return False

        # Step 2: Fetch code
        logger.info("\nStep 2: Fetch code from GitHub")
        start_time = time.time()

        code_doc = code_fetcher.fetch_code(github_url, is_official=True)
        code_time = time.time() - start_time

        if code_doc:
            logger.info(f"✓ Code fetched in {code_time:.2f}s")
            logger.info(f"  Repository: {code_doc.github_url}")
            logger.info(f"  Files: {len(code_doc.code_files)}")
            logger.info(f"  Total lines: {code_doc.total_lines}")
            logger.info(f"  Language: {code_doc.metadata.get('language', 'Unknown')}")

            # Store code
            code_doc_dict = {
                "github_url": code_doc.github_url,
                "is_official": code_doc.is_official,
                "code_files": code_doc.code_files,
                "metadata": code_doc.metadata,
                "total_lines": code_doc.total_lines,
            }
            code_key = storage.store_code(arxiv_id, code_doc_dict)
            logger.info(f"  Stored with key: {code_key}")
        else:
            logger.error("✗ Failed to fetch code")
            return False

        # Step 3: Verify retrieval
        logger.info("\nStep 3: Verify data in ArangoDB")
        logger.info("-" * 80)

        # Verify paper
        retrieved_paper = storage.get_paper(arxiv_id)
        if retrieved_paper:
            logger.info("✓ Paper retrieved from database")
            logger.info(f"  Title: {retrieved_paper['title'][:60]}...")
            logger.info(
                f"  Authors: {', '.join(retrieved_paper.get('authors', [])[:2])}..."
            )
        else:
            logger.error("✗ Failed to retrieve paper")
            return False

        # Verify code
        retrieved_code = storage.get_code(arxiv_id)
        if retrieved_code:
            logger.info("✓ Code retrieved from database")
            logger.info(f"  URL: {retrieved_code['github_url']}")
            logger.info(f"  Files: {len(retrieved_code.get('code_files', {}))}")
        else:
            logger.error("✗ Failed to retrieve code")
            return False

        # Step 4: Show combined context preview
        logger.info("\nStep 4: Combined context preview")
        logger.info("-" * 80)

        paper_content = retrieved_paper.get("markdown_content", "")
        code_content = "\n\n".join(
            f"// File: {path}\n{content}"
            for path, content in retrieved_code.get("code_files", {}).items()
        )

        paper_chars = len(paper_content)
        code_chars = len(code_content)
        total_chars = paper_chars + code_chars

        logger.info(f"Paper content: {paper_chars:,} chars")
        logger.info(f"Code content: {code_chars:,} chars")
        logger.info(f"Total content: {total_chars:,} chars")

        # Jina v4 supports 32k tokens ≈ 128k chars (4 chars/token average)
        max_chars = 128_000
        if total_chars > max_chars:
            logger.warning(
                f"⚠ Content exceeds {max_chars:,} chars - will require chunking"
            )
        else:
            logger.info(f"✓ Content fits within context window ({max_chars:,} chars)")

        # Step 5: Statistics
        logger.info("\nStep 5: Collection statistics")
        logger.info("-" * 80)

        stats = storage.get_collection_stats()
        for collection_name, count in stats.items():
            logger.info(f"  {collection_name}: {count} documents")

        logger.info("\n" + "=" * 80)
        logger.info("Phase 2 Test Complete!")
        logger.info("=" * 80)
        logger.info(f"Total time: {paper_time + code_time:.2f}s")
        logger.info(f"  Paper: {paper_time:.2f}s")
        logger.info(f"  Code: {code_time:.2f}s")

        return True

    finally:
        storage.close()
        paper_fetcher.close()
        code_fetcher.close()


if __name__ == "__main__":
    success = test_phase2()
    sys.exit(0 if success else 1)
