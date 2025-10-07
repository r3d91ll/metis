"""
Generate Embeddings for Word2Vec Family

Generates Jina v4 embeddings for all papers and stores them in ArangoDB.
"""

import logging
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.word2vec.context_builder import CombinedContextBuilder
from experiments.word2vec.embedding_generator import EmbeddingGenerator
from experiments.word2vec.storage import CFExperimentStorage

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_all_embeddings(config_path: Path, device: str = "cuda"):
    """
    Generate Jina v4 embeddings for all papers listed in a YAML configuration and store them in the ArangoDB experiment collection.
    
    Loads the config at config_path, iterates the configured papers, builds a combined context (including LaTeX), produces a Jina v4 embedding with metadata for each paper, ensures the target "cf_embeddings" collection exists, and inserts or replaces the embedding document in the database. Processing results are aggregated into a final summary that is logged.
    
    Parameters:
        config_path (Path): Path to the YAML configuration file that contains the `papers` list and storage settings.
        device (str): Compute device to use for embedding generation (e.g., "cuda" or "cpu").
    
    Returns:
        bool: `true` if embeddings were successfully generated and stored for every configured paper, `false` otherwise.
    """
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("=" * 80)
    logger.info("Word2Vec Family Embedding Generation")
    logger.info("=" * 80)

    # Initialize components
    storage = CFExperimentStorage(config=config)
    context_builder = CombinedContextBuilder(max_tokens=32000)
    embedding_generator = EmbeddingGenerator(device=device, batch_size=1)

    try:
        papers = config.get("papers", [])
        results = []

        logger.info(f"\nProcessing {len(papers)} papers")
        logger.info(f"Device: {device}")
        logger.info("Model: jinaai/jina-embeddings-v4 (2048 dims)")
        logger.info("-" * 80)

        for i, paper_config in enumerate(papers, 1):
            arxiv_id = paper_config["arxiv_id"]

            logger.info(f"\n[{i}/{len(papers)}] Processing {arxiv_id}")
            logger.info("-" * 80)

            start_time = time.time()

            # Step 1: Retrieve from database
            logger.info("Step 1: Retrieving paper and code from database...")
            paper_doc = storage.get_paper(arxiv_id)
            code_doc = storage.get_code(arxiv_id)

            if not paper_doc:
                logger.error(f"✗ Paper {arxiv_id} not found")
                continue

            logger.info(f"  ✓ Paper: {paper_doc['title'][:50]}...")
            if code_doc:
                logger.info(f"  ✓ Code: {code_doc['github_url']}")
            else:
                logger.info("  ℹ No code repository")

            # Step 2: Build combined context
            logger.info("\nStep 2: Building combined context...")
            context = context_builder.build_context(
                paper_doc=paper_doc,
                code_doc=code_doc,
                include_latex=True
            )

            logger.info(
                f"  ✓ Context: {context.total_chars:,} chars "
                f"(~{context.total_tokens_estimate:,} tokens)"
            )
            if context.truncated:
                logger.warning("  ⚠ Content was truncated to fit window")

            # Step 3: Generate embedding
            logger.info("\nStep 3: Generating Jina v4 embedding...")
            embedding_result = embedding_generator.generate_for_paper(
                arxiv_id=arxiv_id,
                context=context.content,
                metadata={
                    "title": context.title,
                    "components": context.components,
                    "truncated": context.truncated,
                    "has_code": context.metadata["has_code"],
                    "has_latex": context.metadata["has_latex"],
                    "code_language": context.metadata.get("code_language")
                }
            )

            logger.info(
                f"  ✓ Embedding: {embedding_result['dimensions']} dims, "
                f"{embedding_result['processing_time_seconds']:.2f}s"
            )

            # Step 4: Store embedding
            logger.info("\nStep 4: Storing embedding in database...")
            embedding_doc = {
                "_key": f"word2vec_cf_{arxiv_id.replace('.', '_')}",
                "arxiv_id": arxiv_id,
                "experiment": "word2vec_cf_validation",
                "embedding": embedding_result["embedding"],
                "dimensions": embedding_result["dimensions"],
                "model": embedding_result["model"],
                "context_metadata": embedding_result["metadata"],
                "processing_metadata": {
                    "context_chars": embedding_result["context_chars"],
                    "context_tokens_estimate": embedding_result["context_tokens_estimate"],
                    "processing_time_seconds": embedding_result["processing_time_seconds"],
                    "timestamp": embedding_result["timestamp"]
                }
            }

            # Create CF embeddings collection if needed
            try:
                storage.client.request(
                    "GET",
                    f"/_db/{storage.db_name}/_api/collection/cf_embeddings/properties"
                )
            except Exception:
                # Collection doesn't exist, create it
                storage.client.request(
                    "POST",
                    f"/_db/{storage.db_name}/_api/collection",
                    json={"name": "cf_embeddings", "type": 2}  # type 2 = document collection
                )
                logger.info("  Created cf_embeddings collection")

            try:
                storage.client.insert_documents(
                    "cf_embeddings",
                    [embedding_doc],
                    on_duplicate="replace"
                )
                logger.info(f"  ✓ Stored: {embedding_doc['_key']}")
            except Exception as e:
                logger.error(f"  ✗ Storage failed: {e}")
                continue

            total_time = time.time() - start_time
            logger.info(f"\n✓ Completed in {total_time:.2f}s")

            results.append({
                "arxiv_id": arxiv_id,
                "title": context.title,
                "embedding_dims": embedding_result["dimensions"],
                "context_chars": context.total_chars,
                "truncated": context.truncated,
                "processing_time": total_time
            })

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("EMBEDDING GENERATION SUMMARY")
        logger.info("=" * 80)

        logger.info(f"\nPapers processed: {len(results)}/{len(papers)}")

        total_chars = sum(r["context_chars"] for r in results)
        total_time = sum(r["processing_time"] for r in results)
        avg_time = total_time / len(results) if results else 0

        logger.info(f"Total context: {total_chars:,} chars")
        logger.info(f"Total time: {total_time:.1f}s")
        logger.info(f"Average time per paper: {avg_time:.1f}s")
        logger.info(
            f"Throughput: {total_chars/total_time:,.0f} chars/s" if total_time > 0 else "N/A"
        )

        logger.info("\nDetails:")
        logger.info("-" * 80)
        for r in results:
            status = "✓" if not r["truncated"] else "⚠"
            logger.info(
                f"{status} {r['arxiv_id']}: {r['context_chars']:>7,} chars, "
                f"{r['embedding_dims']} dims, {r['processing_time']:.1f}s"
            )

        # Database stats
        logger.info("\nDatabase Statistics:")
        logger.info("-" * 80)
        stats = storage.get_collection_stats()
        for collection_name, count in stats.items():
            logger.info(f"  {collection_name}: {count:,} documents")

        logger.info("\n" + "=" * 80)

        return len(results) == len(papers)

    finally:
        storage.close()
        embedding_generator.close()


def main():
    """
    Run the CLI that parses device selection and invokes embedding generation using the adjacent config.yaml.
    
    Parses a --device argument (choices: "cuda", "cpu"), constructs the config.yaml path next to the script, and calls generate_all_embeddings with those settings.
    
    Returns:
        exit_code (int): 0 on success, 1 on failure.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Jina v4 embeddings for Word2Vec family papers"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device (default: cuda)"
    )
    args = parser.parse_args()

    config_path = Path(__file__).parent / "config.yaml"
    success = generate_all_embeddings(config_path, device=args.device)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())