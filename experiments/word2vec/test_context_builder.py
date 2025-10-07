"""
Test Combined Context Builder

Tests building unified contexts from paper + code for embedding.
"""

import logging
import sys
from pathlib import Path

import yaml

# Add metis to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.word2vec.context_builder import CombinedContextBuilder
from experiments.word2vec.storage import CFExperimentStorage

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_context_builder():
    """
    Builds combined contexts for the configured Word2Vec papers and logs diagnostic summaries.
    
    Loads configuration from a nearby config.yaml, retrieves paper and code documents from CFExperimentStorage, and uses CombinedContextBuilder (with LaTeX included) to construct unified contexts. For each paper it logs title, size and token estimates, component breakdown, metadata, and a content preview, then collects per-paper results and emits an overall summary. Storage is closed in all cases.
    
    Returns:
    	bool: `True` if at least one context was built and none were truncated, `False` otherwise.
    """
    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("=" * 80)
    logger.info("Combined Context Builder Test")
    logger.info("=" * 80)

    # Initialize components
    storage = CFExperimentStorage(config=config)
    builder = CombinedContextBuilder(max_tokens=32000)

    try:
        # Test with all 5 papers
        papers = config.get("papers", [])
        results = []

        for i, paper_config in enumerate(papers, 1):
            arxiv_id = paper_config["arxiv_id"]
            logger.info(f"\n[{i}/{len(papers)}] Building context for {arxiv_id}")
            logger.info("-" * 80)

            # Retrieve from database
            paper_doc = storage.get_paper(arxiv_id)
            code_doc = storage.get_code(arxiv_id)

            if not paper_doc:
                logger.error(f"✗ Paper {arxiv_id} not found in database")
                continue

            # Build context
            context = builder.build_context(
                paper_doc=paper_doc,
                code_doc=code_doc,
                include_latex=True
            )

            # Report results
            logger.info(f"✓ Context built for: {context.title[:60]}...")
            logger.info(f"  Total: {context.total_chars:,} chars "
                       f"(~{context.total_tokens_estimate:,} tokens)")
            logger.info(f"  Fits in context window: "
                       f"{'Yes' if not context.truncated else 'No (truncated)'}")

            logger.info("  Components:")
            for comp_name, comp_size in context.components.items():
                pct = (comp_size / context.total_chars * 100) if context.total_chars > 0 else 0
                logger.info(f"    - {comp_name}: {comp_size:,} chars ({pct:.1f}%)")

            logger.info("  Metadata:")
            for key, value in context.metadata.items():
                logger.info(f"    - {key}: {value}")

            # Show preview
            logger.info("\n  Content Preview (first 500 chars):")
            logger.info("  " + "-" * 76)
            preview = context.content[:500].replace("\n", "\n  ")
            logger.info(f"  {preview}")
            if len(context.content) > 500:
                logger.info("  [...]")

            results.append({
                "arxiv_id": arxiv_id,
                "title": context.title,
                "chars": context.total_chars,
                "tokens_estimate": context.total_tokens_estimate,
                "truncated": context.truncated,
                "has_code": context.metadata["has_code"],
                "has_latex": context.metadata["has_latex"]
            })

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)

        total_contexts = len(results)
        with_code = sum(1 for r in results if r["has_code"])
        with_latex = sum(1 for r in results if r["has_latex"])
        truncated = sum(1 for r in results if r["truncated"])

        logger.info(f"\nContexts built: {total_contexts}")
        logger.info(f"With code: {with_code}/{total_contexts}")
        logger.info(f"With LaTeX: {with_latex}/{total_contexts}")
        logger.info(f"Truncated: {truncated}/{total_contexts}")

        logger.info("\nSize Distribution:")
        logger.info("-" * 80)
        for r in results:
            status = "✓" if not r["truncated"] else "⚠"
            code_marker = " [+code]" if r["has_code"] else ""
            logger.info(
                f"{status} {r['arxiv_id']}: {r['chars']:>7,} chars "
                f"(~{r['tokens_estimate']:>5,} tokens){code_marker}"
            )

        # Check if all fit
        if truncated == 0:
            logger.info("\n✅ All contexts fit within 32k token window!")
        else:
            logger.info(f"\n⚠ {truncated} contexts were truncated")

        logger.info("\n" + "=" * 80)

        return total_contexts > 0 and truncated == 0

    finally:
        storage.close()


if __name__ == "__main__":
    success = test_context_builder()
    sys.exit(0 if success else 1)