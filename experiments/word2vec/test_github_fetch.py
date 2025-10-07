"""
Test GitHub Code Fetcher

Tests fetching Word2Vec code from GitHub with filtering.
"""

import logging
import os
import sys
from pathlib import Path

import yaml

# Add metis to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.word2vec.github_fetcher import GitHubCodeFetcher
from experiments.word2vec.storage import CFExperimentStorage

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_github_fetch():
    """
    Run an end-to-end test of GitHub code fetching, filtering, storage, and optional repository search for the Word2Vec project.
    
    Performs the following checks:
    - Fetches code from a known Word2Vec GitHub repository and collects metadata.
    - Verifies that non-code files (README, docs, tests, examples, licenses, etc.) are excluded.
    - Stores the fetched code into ArangoDB and verifies retrieval.
    - Optionally searches for the official repository when a GitHub token is available.
    
    Returns:
        bool: `True` if all tests pass, `False` otherwise.
    """
    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("=" * 80)
    logger.info("GitHub Code Fetcher Test - Word2Vec")
    logger.info("=" * 80)

    # Initialize GitHub fetcher
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        logger.warning("No GITHUB_TOKEN found - API will be rate limited")

    fetcher = GitHubCodeFetcher(config, github_token=github_token)

    # Test 1: Fetch from known repository URL
    logger.info("\nTest 1: Fetch from known Word2Vec repository")
    logger.info("-" * 80)

    word2vec_url = "https://github.com/tmikolov/word2vec"
    code_doc = fetcher.fetch_code(word2vec_url, is_official=True)

    if code_doc:
        logger.info(f"✓ Successfully fetched code from {word2vec_url}")
        logger.info(f"  Files: {len(code_doc.code_files)}")
        logger.info(f"  Total lines: {code_doc.total_lines}")
        logger.info(f"  Language: {code_doc.metadata.get('language', 'Unknown')}")
        logger.info(f"  Stars: {code_doc.metadata.get('stars', 0)}")

        # Show file list
        logger.info("\n  Code files:")
        for file_path in sorted(code_doc.code_files.keys()):
            lines = len(code_doc.code_files[file_path].splitlines())
            size = len(code_doc.code_files[file_path])
            logger.info(f"    - {file_path} ({lines} lines, {size} bytes)")
    else:
        logger.error("✗ Failed to fetch code")
        return False

    # Test 2: Verify filtering (should not include README, docs, etc.)
    logger.info("\nTest 2: Verify code-only filtering")
    logger.info("-" * 80)

    excluded_patterns = [".md", "readme", "license", "doc", "test", "example"]
    violations = []
    for file_path in code_doc.code_files.keys():
        for pattern in excluded_patterns:
            if pattern in file_path.lower():
                violations.append((file_path, pattern))

    if violations:
        logger.error("✗ Found files that should be excluded:")
        for file_path, pattern in violations:
            logger.error(f"  - {file_path} (matches '{pattern}')")
        return False
    else:
        logger.info("✓ All files pass filtering rules (code-only)")

    # Test 3: Store in ArangoDB
    logger.info("\nTest 3: Store code in ArangoDB")
    logger.info("-" * 80)

    try:
        storage = CFExperimentStorage(config=config)
        storage.ensure_collections()

        arxiv_id = "1301.3781"  # Word2Vec paper
        code_doc_dict = {
            "github_url": code_doc.github_url,
            "is_official": code_doc.is_official,
            "code_files": code_doc.code_files,
            "metadata": code_doc.metadata,
            "total_lines": code_doc.total_lines,
        }

        doc_key = storage.store_code(arxiv_id, code_doc_dict)
        logger.info(f"✓ Stored code with key: {doc_key}")

        # Verify retrieval
        retrieved = storage.get_code(arxiv_id)
        if retrieved:
            logger.info(
                f"✓ Successfully retrieved code "
                f"({len(retrieved.get('code_files', {}))} files)"
            )
        else:
            logger.error("✗ Failed to retrieve stored code")
            return False

        storage.close()

    except Exception as e:
        logger.error(f"✗ Storage failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test 4: Repository search (optional, may be rate limited)
    logger.info("\nTest 4: Repository search (optional)")
    logger.info("-" * 80)

    if github_token:
        try:
            repo_url = fetcher.find_official_repo(
                paper_title="Efficient Estimation of Word Representations in Vector Space",
                authors=["Tomas Mikolov", "Kai Chen", "Greg Corrado", "Jeffrey Dean"],
                arxiv_id="1301.3781",
            )
            if repo_url:
                logger.info(f"✓ Found repository: {repo_url}")
            else:
                logger.warning("⚠ Could not find repository via search")
        except Exception as e:
            logger.warning(f"⚠ Search failed (possibly rate limited): {e}")
    else:
        logger.info("  Skipped (no GITHUB_TOKEN)")

    logger.info("\n" + "=" * 80)
    logger.info("All tests passed!")
    logger.info("=" * 80)

    return True


if __name__ == "__main__":
    success = test_github_fetch()
    sys.exit(0 if success else 1)