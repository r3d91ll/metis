#!/usr/bin/env python3
"""
Academic Paper Ingestion Example

This example demonstrates how to use Metis to ingest academic papers,
extract their content, generate embeddings, and store them in ArangoDB.
"""

import os
from pathlib import Path

from metis import create_embedder
from metis.database import ArangoClient, CollectionDefinition, resolve_client_config
from metis.extractors import DoclingExtractor, LaTeXExtractor
from metis.utils import sanitize_key, sha1_hex


def main():
    """Run paper ingestion example."""
    print("Metis Paper Ingestion Example")
    print("=" * 50)

    # Configure database connection
    print("\n1. Configuring database connection...")
    config = resolve_client_config(
        database="papers_db",
        use_proxies=False,  # Direct connection for this example
    )
    print(f"   Database: {config.database}")
    print(f"   Read socket: {config.read_socket}")
    print(f"   Write socket: {config.write_socket}")

    # Create embedder
    print("\n2. Creating embedder...")
    embedder = create_embedder(
        "jinaai/jina-embeddings-v4",
        device="cuda" if Path("/dev/nvidia0").exists() else "cpu",
        batch_size=16,
    )
    print(f"   Model: {embedder.config.model_name}")
    print(f"   Embedding dimension: {embedder.embedding_dimension}")

    # Setup extractors
    print("\n3. Setting up extractors...")
    pdf_extractor = DoclingExtractor()
    latex_extractor = LaTeXExtractor()
    print("   PDF and LaTeX extractors ready")

    # Connect to database
    print("\n4. Connecting to database...")
    with ArangoClient(config) as client:
        # Create collections
        print("   Creating collections...")
        collections = [
            CollectionDefinition(
                name="papers",
                type="document",
                indexes=[
                    {"type": "persistent", "fields": ["arxiv_id"]},
                    {"type": "persistent", "fields": ["title"]},
                ],
            ),
            CollectionDefinition(
                name="paper_chunks",
                type="document",
                indexes=[
                    {"type": "persistent", "fields": ["paper_key"]},
                    {"type": "persistent", "fields": ["chunk_index"]},
                ],
            ),
        ]

        client.create_collections(collections)
        print("   Collections created")

        # Example: Process a paper directory
        paper_dir = Path("papers")  # Hypothetical directory
        if paper_dir.exists():
            process_papers(client, embedder, pdf_extractor, latex_extractor, paper_dir)
        else:
            print(f"\n   Note: {paper_dir} not found. Skipping paper processing.")
            print("   To use this example, create a 'papers' directory with PDF or LaTeX files.")

    print("\n" + "=" * 50)
    print("Example completed!")


def process_papers(client, embedder, pdf_extractor, latex_extractor, paper_dir):
    """Process all papers in directory."""
    print(f"\n5. Processing papers from {paper_dir}...")

    papers = list(paper_dir.glob("*.pdf")) + list(paper_dir.glob("*.tex.gz"))

    for paper_path in papers:
        print(f"\n   Processing: {paper_path.name}")

        # Extract based on format
        if paper_path.suffix == ".pdf":
            result = pdf_extractor.extract(paper_path)
            content = result.get("full_text", "")
        elif paper_path.name.endswith(".tex.gz"):
            result = latex_extractor.extract(paper_path)
            content = result.get("full_text", "")
        else:
            print(f"   Skipping unsupported format: {paper_path.suffix}")
            continue

        if not content:
            print("   No content extracted, skipping")
            continue

        # Generate embedding
        embedding = embedder.embed_single(content)

        # Create document key
        content_hash = sha1_hex(content.encode("utf-8"))
        paper_key = sanitize_key(f"paper_{paper_path.stem}_{content_hash[:8]}")

        # Store in database
        paper_doc = {
            "_key": paper_key,
            "title": result.get("metadata", {}).get("title", paper_path.stem),
            "file_path": str(paper_path),
            "content": content,
            "embedding": embedding.tolist(),
            "metadata": result.get("metadata", {}),
        }

        client.bulk_import("papers", [paper_doc], on_duplicate="update")

        print(f"   Stored as {paper_key}")
        print(f"   Content length: {len(content)} characters")


if __name__ == "__main__":
    main()
