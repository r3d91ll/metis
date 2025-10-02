"""
arXiv Import Pipeline

Streams papers from arxiv-kaggle-latest.json, generates embeddings, and imports to ArangoDB.

Usage:
    # Test with small sample
    python import_pipeline.py --limit 100

    # Full import
    python import_pipeline.py

    # Resume from specific point
    python import_pipeline.py --skip 100000
"""

import json
import logging
import sys
from pathlib import Path
from typing import Iterator, Dict, Any, Optional
from datetime import datetime
import yaml
import argparse

# Add metis to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from metis import create_embedder
from metis.database import ArangoClient, resolve_client_config, CollectionDefinition
from arxiv_parser import ArxivIdParser

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ArxivImportPipeline:
    """
    Import arXiv papers with embeddings and metadata.

    Pipeline stages:
    1. Stream papers from JSON (NDJSON format)
    2. Parse arXiv IDs → internal IDs
    3. Generate embeddings (batch processing)
    4. Insert into ArangoDB (bulk operations)
    """

    def __init__(self, config_path: Path):
        """Initialize pipeline with configuration."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {config_path}")

        # Initialize embedder
        emb_config = self.config['embeddings']
        logger.info(f"Initializing embedder: {emb_config['model']}")
        self.embedder = create_embedder(
            emb_config['model'],
            device=emb_config['device'],
            batch_size=emb_config['batch_size']
        )

        # Database configuration
        use_tcp = self.config['database'].get('use_tcp', False)

        if use_tcp:
            # Use TCP for all operations (fallback when socket has permission issues)
            logger.info("Using TCP endpoint (socket permissions issue)")
            self.db_config = resolve_client_config(
                database=self.config['database']['name'],
                socket_path=None,  # Use TCP
                use_proxies=False
            )
        else:
            # Use socket for data operations (preferred for performance)
            logger.info("Using Unix socket for data operations")
            self.db_config = resolve_client_config(
                database=self.config['database']['name'],
                socket_path=self.config['database']['socket_path'],
                use_proxies=False
            )

        self.data_file = Path(self.config['data']['source_file'])
        self.batch_size = self.config['import']['batch_size']
        self.log_interval = self.config['import']['log_interval']

        logger.info(f"Data source: {self.data_file}")
        logger.info(f"Target database: {self.config['database']['name']}")
        logger.info(f"Socket path: {self.config['database']['socket_path']}")
        logger.info(f"TCP endpoint: http://localhost:8529 (for admin operations)")

    def read_papers(self, skip: int = 0, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """
        Stream papers from NDJSON file.

        Args:
            skip: Number of papers to skip (for resuming)
            limit: Maximum number of papers to process
        """
        logger.info(f"Reading papers from {self.data_file}")
        if skip > 0:
            logger.info(f"Skipping first {skip:,} papers")
        if limit:
            logger.info(f"Limiting to {limit:,} papers")

        with open(self.data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                # Skip lines if requested
                if i < skip:
                    continue

                # Check limit
                if limit and (i - skip) >= limit:
                    break

                try:
                    paper = json.loads(line)
                    yield paper
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse line {i}: {e}")
                    continue

    def process_paper(self, paper: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Transform raw paper JSON into database document.

        Returns:
            Document ready for insertion, or None if processing fails
        """
        try:
            # Parse arXiv ID
            parsed = ArxivIdParser.parse(paper['id'])

            # Parse submission date
            submission_date = None
            if paper.get('versions') and len(paper['versions']) > 0:
                date_str = paper['versions'][0]['created']
                submission_date = ArxivIdParser.extract_date_from_version(date_str)

            # Split categories (space-separated in source data)
            categories_str = paper.get('categories', '')
            categories = categories_str.split() if categories_str else []
            primary_category = categories[0] if categories else None

            # Parse authors (handle both string and list formats)
            authors_field = paper.get('authors', '')
            if isinstance(authors_field, str):
                authors = [a.strip() for a in authors_field.split(',') if a.strip()]
            else:
                authors = authors_field

            # Prepare document
            doc = {
                "_key": parsed.internal_id,
                "arxiv_id": paper['id'],

                # Content
                "title": paper.get('title', '').strip(),
                "abstract": paper.get('abstract', '').strip(),

                # Authors
                "authors": authors,
                "authors_parsed": paper.get('authors_parsed', []),

                # Categories
                "categories": categories,
                "primary_category": primary_category,

                # Temporal data
                "submission_date": submission_date.isoformat() if submission_date else None,
                "update_date": paper.get('update_date'),
                "year": parsed.year,
                "month": parsed.month,
                "year_month": f"{parsed.year:04d}-{parsed.month:02d}",

                # Publication metadata
                "journal_ref": paper.get('journal-ref'),
                "doi": paper.get('doi'),
                "comments": paper.get('comments'),
                "license": paper.get('license'),
                "report_no": paper.get('report-no'),
                "submitter": paper.get('submitter'),

                # Version history
                "versions": paper.get('versions', []),
                "version_count": len(paper.get('versions', [])),

                # Embeddings (will be filled in next step)
                "title_embedding": None,
                "abstract_embedding": None,
                "combined_embedding": None,

                # Metadata
                "created_at": datetime.utcnow().isoformat(),
                "import_batch": None  # Will be set during import
            }

            return doc

        except Exception as e:
            logger.error(f"Failed to process paper {paper.get('id', 'unknown')}: {e}")
            return None

    def generate_embeddings(self, documents: list[Dict[str, Any]]) -> None:
        """
        Generate embeddings for batch of documents (in-place modification).

        Uses Jina v4 with 32k context window and 2048 dimensions.
        """
        # Extract texts (handle None/empty values)
        titles = [doc.get('title', '') or '' for doc in documents]
        abstracts = [doc.get('abstract', '') or '' for doc in documents]
        combined = [
            f"{doc.get('title', '') or ''}\n\n{doc.get('abstract', '') or ''}"
            for doc in documents
        ]

        # Generate embeddings (batched internally by embedder)
        logger.debug(f"Generating embeddings for {len(documents)} documents...")
        title_embeds = self.embedder.embed_texts(titles)
        abstract_embeds = self.embedder.embed_texts(abstracts)
        combined_embeds = self.embedder.embed_texts(combined)

        # Attach to documents
        for i, doc in enumerate(documents):
            doc['title_embedding'] = title_embeds[i].tolist()
            doc['abstract_embedding'] = abstract_embeds[i].tolist()
            doc['combined_embedding'] = combined_embeds[i].tolist()

    def import_batch(
        self,
        documents: list[Dict[str, Any]],
        client: ArangoClient,
        batch_id: str
    ) -> Dict[str, int]:
        """
        Import batch of documents with embeddings.

        Returns:
            Statistics dictionary with counts
        """
        # Set batch ID
        for doc in documents:
            doc['import_batch'] = batch_id

        # Insert documents
        collection = self.config['database']['collections']['papers']
        result = client.bulk_import(collection, documents)

        stats = {
            'created': result.get('created', 0),
            'errors': result.get('errors', 0),
            'empty': result.get('empty', 0)
        }

        logger.info(
            f"Batch {batch_id}: created={stats['created']}, "
            f"errors={stats['errors']}, empty={stats['empty']}"
        )

        return stats

    def create_database_if_needed(self) -> None:
        """Create database via TCP if it doesn't exist (admin operation)."""
        import httpx

        db_name = self.config['database']['name']
        logger.info(f"Checking if database '{db_name}' exists...")

        # Use TCP endpoint for admin operations
        tcp_url = "http://localhost:8529"

        try:
            # Check if database exists
            with httpx.Client() as client:
                response = client.get(
                    f"{tcp_url}/_api/database/user",
                    auth=("root", "")  # Try with empty password
                )

                if response.status_code == 200:
                    databases = response.json().get('result', [])
                    if db_name in databases:
                        logger.info(f"Database '{db_name}' already exists")
                        return

                    # Create database
                    logger.info(f"Creating database '{db_name}' via TCP...")
                    create_response = client.post(
                        f"{tcp_url}/_api/database",
                        json={"name": db_name},
                        auth=("root", "")
                    )

                    if create_response.status_code in [200, 201]:
                        logger.info(f"Successfully created database '{db_name}'")
                    elif create_response.status_code == 409:
                        logger.info(f"Database '{db_name}' already exists")
                    else:
                        logger.warning(
                            f"Database creation returned status {create_response.status_code}: "
                            f"{create_response.text}"
                        )
                else:
                    logger.warning(
                        f"Could not check databases (status {response.status_code}). "
                        f"Database '{db_name}' may need to be created manually via web UI."
                    )
        except Exception as e:
            logger.warning(
                f"Could not create database via TCP: {e}. "
                f"Please create database '{db_name}' manually via web UI at http://localhost:8529"
            )

    def setup_database(self, client: ArangoClient) -> None:
        """Create collections and indexes (uses socket client)."""
        logger.info("Setting up database schema...")

        papers_collection = self.config['database']['collections']['papers']

        # Define collection with indexes
        indexes = [
            {
                'type': 'persistent',
                'fields': ['arxiv_id'],
                'unique': True,
                'name': 'idx_arxiv_id'
            },
            {
                'type': 'persistent',
                'fields': ['categories[*]'],
                'name': 'idx_categories'
            },
            {
                'type': 'persistent',
                'fields': ['year', 'month'],
                'name': 'idx_year_month'
            },
            {
                'type': 'persistent',
                'fields': ['year_month'],
                'name': 'idx_year_month_str'
            },
            {
                'type': 'persistent',
                'fields': ['primary_category'],
                'name': 'idx_primary_category'
            },
            {
                'type': 'persistent',
                'fields': ['submission_date'],
                'name': 'idx_submission_date'
            },
        ]

        # Create collection with indexes using CollectionDefinition
        collection_def = CollectionDefinition(
            name=papers_collection,
            type="document",
            indexes=indexes
        )

        try:
            client.create_collections([collection_def])
            logger.info(f"Created collection: {papers_collection} with {len(indexes)} indexes")
        except Exception as e:
            # Collection may already exist
            logger.info(f"Collection setup: {e}")

        logger.info("Database setup complete")

    def run(
        self,
        skip: int = 0,
        limit: Optional[int] = None,
        setup_db: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete import pipeline.

        Args:
            skip: Number of papers to skip (for resuming)
            limit: Maximum number of papers to import
            setup_db: Whether to create collections/indexes

        Returns:
            Statistics dictionary
        """
        stats = {
            'total_processed': 0,
            'total_created': 0,
            'total_errors': 0,
            'batches': 0
        }

        # Create database via TCP if needed (admin operation)
        if setup_db:
            self.create_database_if_needed()

        with ArangoClient(self.db_config) as client:
            # Setup collections and indexes via socket (data operations)
            if setup_db:
                self.setup_database(client)

            batch = []
            batch_num = 0

            for i, paper in enumerate(self.read_papers(skip=skip, limit=limit)):
                # Process paper
                doc = self.process_paper(paper)
                if doc is None:
                    stats['total_errors'] += 1
                    continue

                batch.append(doc)

                # Process batch when full
                if len(batch) >= self.batch_size:
                    try:
                        # Generate embeddings
                        self.generate_embeddings(batch)

                        # Import to database
                        batch_id = f"batch_{batch_num:06d}"
                        batch_stats = self.import_batch(batch, client, batch_id)

                        stats['total_created'] += batch_stats['created']
                        stats['total_errors'] += batch_stats['errors']
                        stats['batches'] += 1

                    except Exception as e:
                        logger.error(f"Batch processing failed: {e}", exc_info=True)
                        stats['total_errors'] += len(batch)

                    finally:
                        stats['total_processed'] += len(batch)
                        batch = []
                        batch_num += 1

                    # Log progress
                    if stats['total_processed'] % self.log_interval == 0:
                        logger.info(
                            f"Progress: {stats['total_processed']:,} papers processed, "
                            f"{stats['total_created']:,} created, "
                            f"{stats['total_errors']:,} errors"
                        )

            # Process remaining papers
            if batch:
                try:
                    self.generate_embeddings(batch)
                    batch_id = f"batch_{batch_num:06d}"
                    batch_stats = self.import_batch(batch, client, batch_id)

                    stats['total_created'] += batch_stats['created']
                    stats['total_errors'] += batch_stats['errors']
                    stats['batches'] += 1

                except Exception as e:
                    logger.error(f"Final batch processing failed: {e}", exc_info=True)
                    stats['total_errors'] += len(batch)

                finally:
                    stats['total_processed'] += len(batch)

        logger.info("=" * 60)
        logger.info("Import Complete")
        logger.info("=" * 60)
        logger.info(f"Total processed: {stats['total_processed']:,}")
        logger.info(f"Total created:   {stats['total_created']:,}")
        logger.info(f"Total errors:    {stats['total_errors']:,}")
        logger.info(f"Batches:         {stats['batches']:,}")
        logger.info("=" * 60)

        return stats


def main():
    parser = argparse.ArgumentParser(description='Import arXiv papers to ArangoDB')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path(__file__).parent / 'config' / 'arxiv_import.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of papers to import (for testing)'
    )
    parser.add_argument(
        '--skip',
        type=int,
        default=0,
        help='Number of papers to skip (for resuming)'
    )
    parser.add_argument(
        '--no-setup',
        action='store_true',
        help='Skip database setup (assume schema exists)'
    )

    args = parser.parse_args()

    # Run pipeline
    pipeline = ArxivImportPipeline(args.config)
    stats = pipeline.run(
        skip=args.skip,
        limit=args.limit,
        setup_db=not args.no_setup
    )

    # Exit code based on success
    if stats['total_errors'] > stats['total_created']:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
