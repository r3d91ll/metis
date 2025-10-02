#!/usr/bin/env python3
"""
Basic Metis Setup Example

This example demonstrates the basic usage of Metis for document extraction
and embedding generation.
"""

from pathlib import Path

from metis import create_embedder, create_extractor_for_file


def main():
    """Run basic setup example."""
    print("Metis Basic Setup Example")
    print("=" * 50)

    # Create embedder with Jina v4
    print("\n1. Creating embedder...")
    embedder = create_embedder(
        "jinaai/jina-embeddings-v4",
        device="cuda" if Path("/dev/nvidia0").exists() else "cpu",
        batch_size=32,
    )

    print(f"   Model: {embedder.config.model_name}")
    print(f"   Device: {embedder.config.device}")
    print(f"   Embedding dimension: {embedder.embedding_dimension}")

    # Extract a sample document
    print("\n2. Extracting document...")

    # For this example, we'll create a simple text file
    sample_file = Path("sample.txt")
    sample_file.write_text(
        """
        Metis: Semantic Knowledge Infrastructure

        Metis is a Python library for building semantic graph databases.
        It provides high-quality document extraction, state-of-the-art embeddings,
        and efficient database operations.

        Key features:
        - Multi-format document extraction (PDF, LaTeX, code)
        - Jina v4 embeddings with 32k context window
        - ArangoDB integration with Unix socket support
        - Late chunking for better semantic preservation
        """
    )

    extractor = create_extractor_for_file(sample_file)
    result = extractor.extract(sample_file)

    print(f"   Extracted {len(result.text)} characters")
    print(f"   Text preview: {result.text[:100]}...")

    # Generate embeddings
    print("\n3. Generating embeddings...")
    embeddings = embedder.embed_texts([result.text])

    print(f"   Generated {embeddings.shape[0]} embeddings")
    print(f"   Embedding shape: {embeddings.shape}")
    print(f"   First 5 dimensions: {embeddings[0][:5]}")

    # Clean up
    sample_file.unlink()

    print("\n" + "=" * 50)
    print("Example completed successfully!")


if __name__ == "__main__":
    main()
