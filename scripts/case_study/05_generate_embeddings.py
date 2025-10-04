#!/usr/bin/env python3
"""Generate embeddings for papers and boundary objects, store in ArangoDB."""

import json
from datetime import datetime
from pathlib import Path

from utils import load_config, setup_logging


def load_extracted_papers(papers_dir: Path, logger) -> dict:
    """Load extracted paper data.

    Args:
        papers_dir: Directory containing extracted papers
        logger: Logger instance

    Returns:
        Dictionary of paper data
    """
    extracted_dir = papers_dir / "extracted"
    all_papers_file = extracted_dir / "all_papers.json"

    if not all_papers_file.exists():
        logger.error(f"Papers not found at {all_papers_file}")
        raise FileNotFoundError("Run 03_extract_papers.py first")

    with open(all_papers_file) as f:
        return json.load(f)


def load_boundary_objects(bo_dir: Path, paper_name: str, logger) -> list[dict]:
    """Load boundary objects for a paper.

    Args:
        bo_dir: Boundary objects directory
        paper_name: Name of the paper
        logger: Logger instance

    Returns:
        List of boundary objects
    """
    bo_file = bo_dir / paper_name / "boundary_objects.json"

    if not bo_file.exists():
        logger.warning(f"No boundary objects found for {paper_name}")
        return []

    with open(bo_file) as f:
        data = json.load(f)
        return data.get("boundary_objects", [])


def generate_embeddings_batch(texts: list[str], embedder, logger) -> list[list[float]]:
    """Generate embeddings for a batch of texts.

    Args:
        texts: List of text strings
        embedder: Embedding model
        logger: Logger instance

    Returns:
        List of embedding vectors
    """
    logger.info(f"Generating embeddings for {len(texts)} texts")

    try:
        # Use metis embedder
        embeddings = embedder.embed(texts)
        logger.info(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
        return embeddings

    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise


def store_paper_in_db(
    paper_name: str, paper_data: dict, embeddings: dict, db, collection_name: str, logger
) -> str:
    """Store paper document in ArangoDB.

    Args:
        paper_name: Name of the paper
        paper_data: Paper data
        embeddings: Section embeddings
        db: Database client
        collection_name: Collection name
        logger: Logger instance

    Returns:
        Document key
    """
    arxiv_id = paper_data["arxiv_id"]
    metadata = paper_data["metadata"]

    # Create document
    doc = {
        "_key": arxiv_id.replace(".", "_"),
        "paper_name": paper_name,
        "title": metadata["title"],
        "authors": metadata["authors"],
        "arxiv_id": arxiv_id,
        "published_date": metadata["published"],
        "abstract": metadata["abstract"],
        "primary_category": metadata["primary_category"],
        "categories": metadata["categories"],
        "sections": {},
    }

    # Add sections with embeddings
    for section_name, section_data in paper_data["sections"].items():
        doc["sections"][section_name] = {
            "text": section_data["text"],
            "embedding": embeddings.get(section_name, []),
        }

    # Insert or update document
    try:
        collection = db.collection(collection_name)
        result = collection.insert(doc, overwrite=True)
        logger.info(f"Stored paper {arxiv_id} in database")
        return result["_key"]

    except Exception as e:
        logger.error(f"Error storing paper in database: {e}")
        raise


def store_boundary_object_in_db(
    paper_name: str,
    boundary_object: dict,
    embedding: list[float],
    db,
    collection_name: str,
    logger,
) -> str:
    """Store boundary object in ArangoDB.

    Args:
        paper_name: Name of the paper
        boundary_object: Boundary object data
        embedding: Embedding vector
        db: Database client
        collection_name: Collection name
        logger: Logger instance

    Returns:
        Document key
    """
    # Create unique key
    obj_type = boundary_object["type"]
    source = boundary_object["source"]
    file_name = boundary_object["file"]
    doc_key = f"{paper_name}_{source}_{file_name}".replace("/", "_").replace(".", "_")

    # Create document
    doc = {
        "_key": doc_key,
        "paper_name": paper_name,
        "type": obj_type,
        "source": source,
        "file": file_name,
        "content": boundary_object["content"],
        "embedding": embedding,
    }

    # Insert or update document
    try:
        collection = db.collection(collection_name)
        result = collection.insert(doc, overwrite=True)
        return result["_key"]

    except Exception as e:
        logger.error(f"Error storing boundary object in database: {e}")
        raise


def setup_database(db, config: dict, logger):
    """Set up database collections.

    Args:
        db: Database client
        config: Configuration
        logger: Logger instance
    """
    collection_name = config["database"]["collection_name"]

    try:
        # Create collection if it doesn't exist
        if not db.has_collection(collection_name):
            db.create_collection(collection_name)
            logger.info(f"Created collection: {collection_name}")

            # Create index on arxiv_id
            collection = db.collection(collection_name)
            collection.add_hash_index(fields=["arxiv_id"], unique=True)
            logger.info("Created index on arxiv_id")

        else:
            logger.info(f"Collection {collection_name} already exists")

    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        raise


def main():
    """Main entry point."""
    # Setup
    logger = setup_logging()
    config = load_config()

    # Initialize embedder
    try:
        from metis.embedders import JinaV4Embedder

        logger.info("Initializing Jina v4 embedder")
        embedder = JinaV4Embedder()

    except ImportError:
        logger.error("JinaV4Embedder not available. Check metis installation.")
        raise

    # Initialize database
    try:
        from metis.database import ArangoDBClient

        logger.info("Connecting to ArangoDB")
        db = ArangoDBClient(socket_path="/tmp/arangodb.sock")

        # Setup collections
        setup_database(db, config, logger)

    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise

    # Load data
    papers_dir = Path("data/case_study/papers")
    bo_dir = Path("data/case_study/boundary_objects")

    papers_data = load_extracted_papers(papers_dir, logger)

    # Process each paper
    collection_name = config["database"]["collection_name"]

    for paper_name, paper_data in papers_data.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {paper_name}")
        logger.info(f"{'='*60}")

        # Generate embeddings for paper sections
        sections = paper_data["sections"]
        section_texts = []
        section_names = []

        for section_name, section_data in sections.items():
            section_texts.append(section_data["text"])
            section_names.append(section_name)

        logger.info(f"Generating embeddings for {len(section_texts)} sections")
        section_embeddings = generate_embeddings_batch(section_texts, embedder, logger)

        # Map embeddings to sections
        embeddings_dict = {
            name: emb.tolist() for name, emb in zip(section_names, section_embeddings)
        }

        # Store paper in database
        paper_key = store_paper_in_db(
            paper_name, paper_data, embeddings_dict, db, collection_name, logger
        )

        # Load and process boundary objects
        boundary_objects = load_boundary_objects(bo_dir, paper_name, logger)

        if boundary_objects:
            logger.info(f"Processing {len(boundary_objects)} boundary objects")

            # Generate embeddings for boundary objects
            bo_texts = [obj["content"] for obj in boundary_objects]
            bo_embeddings = generate_embeddings_batch(bo_texts, embedder, logger)

            # Store boundary objects
            for obj, embedding in zip(boundary_objects, bo_embeddings):
                store_boundary_object_in_db(
                    paper_name, obj, embedding.tolist(), db, collection_name, logger
                )

            logger.info(f"Stored {len(boundary_objects)} boundary objects")

    logger.info("\n" + "=" * 60)
    logger.info("Embedding generation and storage complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
