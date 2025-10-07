#!/usr/bin/env python3
"""
Phase 1 Test: Process Word2Vec Paper

Tests the basic infrastructure by downloading and storing the Word2Vec paper.
"""

import logging
import sys
from pathlib import Path

import yaml

# Add metis to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.word2vec.arxiv_fetcher import ArxivPaperFetcher
from experiments.word2vec.storage import CFExperimentStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_word2vec_paper():
    """
    Orchestrates end-to-end fetching and storage of the Word2Vec arXiv paper (ID 1301.3781) to validate Phase 1 processing.
    
    Loads configuration, prepares the cache directory, initializes the ArxivPaperFetcher and CFExperimentStorage, ensures required database collections exist, fetches the paper from arXiv, stores the paper markdown and metadata in the database, verifies retrieval, logs progress and statistics, and closes storage.
    
    Returns:
        True if all steps completed successfully and the stored paper was verified, False otherwise.
    """

    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("=" * 70)
    logger.info("Phase 1 Test: Word2Vec Paper Processing")
    logger.info("=" * 70)

    # Create cache directory
    cache_dir = Path(config["infrastructure"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Initialize components
    logger.info("\nInitializing components...")

    fetcher = ArxivPaperFetcher(cache_dir=cache_dir)
    logger.info("✓ ArxivPaperFetcher initialized")

    storage = CFExperimentStorage(config=config)
    logger.info("✓ CFExperimentStorage initialized")

    # Ensure collections exist
    logger.info("\nEnsuring database collections...")
    try:
        storage.ensure_collections()
        stats = storage.get_collection_stats()
        for collection, count in stats.items():
            logger.info(f"  • {collection}: {count} documents")
    except Exception as e:
        logger.error(f"Failed to ensure collections: {e}")
        logger.info("\nNote: Make sure ArangoDB is running and accessible.")
        logger.info("If using TCP, ensure the database 'cf_experiments' exists.")
        return False

    # Test paper fetching
    arxiv_id = "1301.3781"
    logger.info(f"\nFetching paper {arxiv_id}...")

    try:
        paper_doc = fetcher.fetch_paper(arxiv_id)

        logger.info("✓ Paper fetched successfully!")
        logger.info(f"  Title: {paper_doc.title}")
        logger.info(f"  Authors: {', '.join(paper_doc.authors[:2])}...")
        logger.info(f"  Abstract: {paper_doc.abstract[:100]}...")
        logger.info(f"  Markdown: {len(paper_doc.markdown_content)} characters")
        logger.info(f"  Word count: {len(paper_doc.markdown_content.split())} words")

    except Exception as e:
        logger.error(f"Failed to fetch paper: {e}")
        return False

    # Test storage
    logger.info("\nStoring paper in database...")

    try:
        # Prepare paper data for storage
        paper_data = {
            "title": paper_doc.title,
            "authors": paper_doc.authors,
            "abstract": paper_doc.abstract,
            "markdown_content": paper_doc.markdown_content,
            "processing_time": paper_doc.processing_time
        }

        doc_key = storage.store_paper_markdown(arxiv_id, paper_data)
        logger.info(f"✓ Paper stored with key: {doc_key}")

        # Verify storage
        stored_doc = storage.get_paper(arxiv_id)
        if stored_doc:
            logger.info("✓ Verified: Paper retrievable from database")
            logger.info(f"  Stored word count: {stored_doc['processing_metadata']['word_count']}")
        else:
            logger.error("Failed to retrieve stored paper")
            return False

    except Exception as e:
        logger.error(f"Failed to store paper: {e}")
        return False

    # Final statistics
    logger.info("\n" + "=" * 70)
    logger.info("Phase 1 Test: SUCCESSFUL")
    logger.info("=" * 70)

    stats = storage.get_collection_stats()
    logger.info("\nCollection statistics:")
    for collection, count in stats.items():
        logger.info(f"  • {collection}: {count} documents")

    logger.info("\n✨ Word2Vec paper successfully processed and stored!")
    logger.info(f"   Processing time: {paper_doc.processing_time:.2f} seconds")

    # Clean up
    storage.close()
    return True


if __name__ == "__main__":
    success = test_word2vec_paper()
    sys.exit(0 if success else 1)