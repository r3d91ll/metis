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
from datetime import datetime, timezone
import time
import yaml
import argparse

# Add metis to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from metis import create_embedder
from metis.database import ArangoClient, ArangoClientConfig, resolve_client_config, CollectionDefinition
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

        # Initialize embedder with late chunking support
        emb_config = self.config['embeddings']
        logger.info(f"Initializing embedder: {emb_config['model']}")
        logger.info(f"  Device: {emb_config['device']}, Batch: {emb_config['batch_size']}, FP16: {emb_config.get('use_fp16', True)}")
        logger.info(f"  Late chunking: {emb_config.get('chunk_size_tokens', 500)} tokens with {emb_config.get('chunk_overlap_tokens', 200)} overlap")
        self.embedder = create_embedder(
            emb_config['model'],
            device=emb_config['device'],
            batch_size=emb_config['batch_size'],
            use_fp16=emb_config.get('use_fp16', True),
            max_seq_length=emb_config.get('max_length', 32768),
            chunk_size_tokens=emb_config.get('chunk_size_tokens', 500),
            chunk_overlap_tokens=emb_config.get('chunk_overlap_tokens', 200)
        )

        # Database configuration
        use_tcp = self.config['database'].get('use_tcp', False)

        if use_tcp:
            # Use TCP for all operations (fallback when socket has permission issues)
            tcp_host = self.config['database'].get('tcp_host', 'localhost')
            tcp_port = self.config['database'].get('tcp_port', 8529)
            logger.info(f"Using TCP endpoint: http://{tcp_host}:{tcp_port}")
            # Force TCP by setting both read and write sockets to None
            self.db_config = ArangoClientConfig(
                database=self.config['database']['name'],
                username="",
                password="",
                base_url=f"http://{tcp_host}:{tcp_port}",
                read_socket=None,  # TCP mode
                write_socket=None,  # TCP mode
                connect_timeout=5.0,
                read_timeout=30.0,
                write_timeout=30.0
            )
        else:
            # Use socket for data operations (preferred for performance)
            socket_path = self.config['database']['socket_path']
            logger.info(f"Using Unix socket: {socket_path}")
            self.db_config = resolve_client_config(
                database=self.config['database']['name'],
                socket_path=socket_path,
                use_proxies=False
            )

        self.data_file = Path(self.config['data']['source_file'])
        self.batch_size = self.config['import']['batch_size']
        self.log_interval = self.config['import']['log_interval']

        logger.info(f"Data source: {self.data_file}")
        logger.info(f"Target database: {self.config['database']['name']}")
        logger.info(f"Socket path: {self.config['database']['socket_path']}")
        logger.info("TCP endpoint: http://localhost:8529 (for admin operations)")

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
                    logger.exception(f"Failed to parse line {i}")
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

            # Prepare three separate documents for three collections

            # 1. Metadata document (papers collection - lightweight, heavily indexed)
            metadata_doc = {
                "_key": parsed.internal_id,
                "arxiv_id": paper['id'],
                "authors": authors,
                "authors_parsed": paper.get('authors_parsed', []),
                "categories": categories,
                "primary_category": primary_category,
                "submission_date": submission_date.isoformat() if submission_date else None,
                "update_date": paper.get('update_date'),
                "year": parsed.year,
                "month": parsed.month,
                "year_month": f"{parsed.year:04d}-{parsed.month:02d}",
                "journal_ref": paper.get('journal-ref'),
                "doi": paper.get('doi'),
                "comments": paper.get('comments'),
                "license": paper.get('license'),
                "report_no": paper.get('report-no'),
                "submitter": paper.get('submitter'),
                "versions": paper.get('versions', []),
                "version_count": len(paper.get('versions', [])),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            # 2. Abstract document (abstracts collection - full text)
            abstract_doc = {
                "_key": parsed.internal_id,
                "arxiv_id": paper['id'],
                "title": paper.get('title', '').strip(),
                "abstract": paper.get('abstract', '').strip(),
            }

            # 3. Embedding document (embeddings collection - vectors only)
            embedding_doc = {
                "_key": parsed.internal_id,
                "arxiv_id": paper['id'],
                "title_embedding": None,  # Will be filled in next step
                "abstract_embedding": None,
                "combined_embedding": None,
            }

            return {
                "metadata": metadata_doc,
                "abstract": abstract_doc,
                "embedding": embedding_doc
            }

        except Exception as e:
            logger.exception(f"Failed to process paper {paper.get('id', 'unknown')}")
            return None

    def generate_embeddings(self, documents: list[Dict[str, Any]]) -> None:
        """
        Generate embeddings for batch of documents (in-place modification).

        Uses Jina v4 with 32k context window and 2048 dimensions.
        Updates the 'embedding' sub-document for each paper.
        """
        # Extract texts from abstract sub-documents
        titles = [doc['abstract'].get('title', '') or '' for doc in documents]
        abstracts = [doc['abstract'].get('abstract', '') or '' for doc in documents]
        combined = [
            f"{doc['abstract'].get('title', '') or ''}\n\n{doc['abstract'].get('abstract', '') or ''}"
            for doc in documents
        ]

        # Generate embeddings (batched internally by embedder)
        logger.debug(f"Generating embeddings for {len(documents)} documents...")
        title_embeds = self.embedder.embed_texts(titles)
        abstract_embeds = self.embedder.embed_texts(abstracts)
        combined_embeds = self.embedder.embed_texts(combined)

        # Attach to embedding sub-documents
        for i, doc in enumerate(documents):
            doc['embedding']['title_embedding'] = title_embeds[i].tolist()
            doc['embedding']['abstract_embedding'] = abstract_embeds[i].tolist()
            doc['embedding']['combined_embedding'] = combined_embeds[i].tolist()

    def import_batch(
        self,
        documents: list[Dict[str, Any]],
        client: ArangoClient,
        batch_id: str
    ) -> Dict[str, int]:
        """
        Import batch of documents to three separate collections.

        Returns:
            Statistics dictionary with counts
        """
        collections = self.config['database']['collections']

        # Add batch ID to metadata documents
        for doc in documents:
            doc['metadata']['import_batch'] = batch_id

        # Extract documents for each collection
        metadata_docs = [doc['metadata'] for doc in documents]
        abstract_docs = [doc['abstract'] for doc in documents]
        embedding_docs = [doc['embedding'] for doc in documents]

        # Insert into three collections
        metadata_count = client.bulk_import(collections['papers'], metadata_docs)
        abstract_count = client.bulk_import(collections['abstracts'], abstract_docs)
        embedding_count = client.bulk_import(collections['embeddings'], embedding_docs)

        stats = {
            'created': metadata_count,
            'errors': 0,
            'empty': 0
        }

        logger.info(f"Batch {batch_id}: inserted {metadata_count} metadata, {abstract_count} abstracts, {embedding_count} embeddings")

        return stats

    def create_database_if_needed(self) -> None:
        """Create database via Unix socket if it doesn't exist (admin operation)."""
        import httpx
        import os

        db_name = self.config['database']['name']
        logger.info(f"Checking if database '{db_name}' exists...")

        # Use Unix socket for admin operations (metis proxy now supports database creation)
        socket_path = self.config['database']['socket_path']

        # Get credentials from environment
        username = os.environ.get('ARANGO_USERNAME', 'root')
        password = os.environ.get('ARANGO_PASSWORD', '')

        try:
            # Check if database exists
            transport = httpx.HTTPTransport(uds=socket_path, retries=0)
            with httpx.Client(transport=transport, base_url="http://arangodb") as client:
                response = client.get("/_api/database/user", auth=(username, password))

                if response.status_code == 200:
                    databases = response.json().get('result', [])
                    if db_name in databases:
                        logger.info(f"Database '{db_name}' already exists")
                        return

                    # Create database
                    logger.info(f"Creating database '{db_name}' via Unix socket...")
                    create_response = client.post(
                        "/_api/database",
                        json={"name": db_name},
                        auth=(username, password)
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
                        f"Database '{db_name}' may need to be created manually."
                    )
        except Exception as e:
            logger.warning(
                f"Could not create database via Unix socket: {e}. "
                f"Please create database '{db_name}' manually"
            )

    def setup_database(self, client: ArangoClient) -> None:
        """Create collections and indexes (uses socket client)."""
        logger.info("Setting up database schema with separate collections...")

        collections_config = self.config['database']['collections']

        # 1. Papers collection - Metadata only (lightweight, heavily indexed)
        papers_indexes = [
            {'type': 'persistent', 'fields': ['arxiv_id'], 'unique': True, 'name': 'idx_arxiv_id'},
            {'type': 'persistent', 'fields': ['categories[*]'], 'name': 'idx_categories'},
            {'type': 'persistent', 'fields': ['year', 'month'], 'name': 'idx_year_month'},
            {'type': 'persistent', 'fields': ['year_month'], 'name': 'idx_year_month_str'},
            {'type': 'persistent', 'fields': ['primary_category'], 'name': 'idx_primary_category'},
            {'type': 'persistent', 'fields': ['submission_date'], 'name': 'idx_submission_date'},
        ]

        # 2. Abstracts collection - Full text content (minimal indexes)
        abstracts_indexes = [
            {'type': 'persistent', 'fields': ['arxiv_id'], 'unique': True, 'name': 'idx_arxiv_id'},
        ]

        # 3. Embeddings collection - Vector embeddings (minimal indexes, large documents)
        embeddings_indexes = [
            {'type': 'persistent', 'fields': ['arxiv_id'], 'unique': True, 'name': 'idx_arxiv_id'},
        ]

        collection_defs = [
            CollectionDefinition(name=collections_config['papers'], type="document", indexes=papers_indexes),
            CollectionDefinition(name=collections_config['abstracts'], type="document", indexes=abstracts_indexes),
            CollectionDefinition(name=collections_config['embeddings'], type="document", indexes=embeddings_indexes),
        ]

        try:
            client.create_collections(collection_defs)
            logger.info("Created 3 collections:")
            logger.info(f"  - {collections_config['papers']}: metadata ({len(papers_indexes)} indexes)")
            logger.info(f"  - {collections_config['abstracts']}: full text ({len(abstracts_indexes)} indexes)")
            logger.info(f"  - {collections_config['embeddings']}: vectors ({len(embeddings_indexes)} indexes)")
        except Exception as e:
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
    parser = argparse.ArgumentParser(description='Import arXiv papers and build graph with GNN')
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
    parser.add_argument(
        '--build-edges',
        action='store_true',
        help='Build graph edges after import'
    )
    parser.add_argument(
        '--export-graph',
        action='store_true',
        help='Export graph to PyG format after building edges'
    )
    parser.add_argument(
        '--train-gnn',
        action='store_true',
        help='Train GraphSAGE model after exporting graph'
    )
    parser.add_argument(
        '--full-pipeline',
        action='store_true',
        help='Run complete pipeline: import + edges + graph + training'
    )

    args = parser.parse_args()

    # Full pipeline enables all stages
    if args.full_pipeline:
        args.build_edges = True
        args.export_graph = True
        args.train_gnn = True

    # Stage 1: Import papers
    print("=" * 60)
    print("STAGE 1: Importing Papers")
    print("=" * 60)

    stage1_start = time.time()
    pipeline = ArxivImportPipeline(args.config)
    stats = pipeline.run(
        skip=args.skip,
        limit=args.limit,
        setup_db=not args.no_setup
    )
    stage1_time = time.time() - stage1_start

    print(f"\nImport complete: {stats['total_created']:,} papers created, {stats['total_errors']:,} errors")
    print(f"Time: {stage1_time:.1f}s ({stats['total_created']/stage1_time:.1f} papers/sec)" if stage1_time > 0 and stats['total_created'] > 0 else f"Time: {stage1_time:.1f}s")
    print(f"Batches processed: {stats['batches']:,}")

    if stats['total_errors'] > stats['total_created']:
        print("ERROR: Import failed with too many errors")
        sys.exit(1)

    # Stage 2: Build edges
    if args.build_edges:
        print("\n" + "=" * 60)
        print("STAGE 2: Building Graph Edges")
        print("=" * 60)

        stage2_start = time.time()
        from edge_builder import EdgeBuilder
        edge_builder = EdgeBuilder(args.config)
        edge_stats = edge_builder.run()
        stage2_time = time.time() - stage2_start

        print(f"\nEdge building complete: {edge_stats['total']:,} total edges")
        print(f"  - Category links: {edge_stats['category_links']:,}")
        print(f"  - Temporal succession: {edge_stats['temporal_succession']:,}")
        if stage2_time > 0 and edge_stats['total'] > 0:
            print(f"Time: {stage2_time:.1f}s ({edge_stats['total']/stage2_time:.1f} edges/sec)")
        else:
            print(f"Time: {stage2_time:.1f}s")

        if edge_stats['total'] == 0:
            print("ERROR: No edges were created!")
            sys.exit(1)

    # Stage 3: Export graph
    if args.export_graph:
        print("\n" + "=" * 60)
        print("STAGE 3: Exporting Graph to PyG")
        print("=" * 60)

        stage3_start = time.time()
        from graph_pipeline import ArxivGraphPipeline
        graph_pipeline = ArxivGraphPipeline(args.config)
        graph_path = graph_pipeline.export_graph()
        stage3_time = time.time() - stage3_start

        file_size_mb = graph_path.stat().st_size / (1024**2)
        print(f"\nGraph exported to: {graph_path}")
        print(f"File size: {file_size_mb:.1f} MB")
        print(f"Time: {stage3_time:.1f}s")

    # Stage 4: Train GNN
    if args.train_gnn:
        print("\n" + "=" * 60)
        print("STAGE 4: Training GraphSAGE")
        print("=" * 60)

        stage4_start = time.time()
        from train_gnn import train_gnn
        train_gnn(args.config)
        stage4_time = time.time() - stage4_start

        print(f"\nGNN training time: {stage4_time:.1f}s ({stage4_time/60:.1f} minutes)")

    # Final summary
    total_time = time.time() - stage1_start
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nPapers imported: {stats['total_created']:,}")
    if args.build_edges:
        print(f"Edges created: {edge_stats['total']:,}")
        print(f"  - Category links: {edge_stats['category_links']:,}")
        print(f"  - Temporal succession: {edge_stats['temporal_succession']:,}")
    if args.export_graph:
        print(f"Graph file: {graph_path}")
        print(f"Graph size: {file_size_mb:.1f} MB")
    if args.train_gnn:
        checkpoint_dir = Path("models/arxiv_checkpoints")
        best_checkpoint = checkpoint_dir / "best.pt"
        if best_checkpoint.exists():
            checkpoint_size_mb = best_checkpoint.stat().st_size / (1024**2)
            print(f"Model checkpoint: {best_checkpoint}")
            print(f"Model size: {checkpoint_size_mb:.1f} MB")

    print(f"\n{'Stage':<25} {'Time':<15} {'Throughput':<20}")
    print("-" * 60)
    if stats['total_created'] > 0 and stage1_time > 0:
        print(f"{'1. Import Papers':<25} {stage1_time:>8.1f}s      {stats['total_created']/stage1_time:>8.1f} papers/sec")
    if args.build_edges and edge_stats['total'] > 0 and stage2_time > 0:
        print(f"{'2. Build Edges':<25} {stage2_time:>8.1f}s      {edge_stats['total']/stage2_time:>8.1f} edges/sec")
    if args.export_graph:
        print(f"{'3. Export Graph':<25} {stage3_time:>8.1f}s")
    if args.train_gnn:
        print(f"{'4. Train GNN':<25} {stage4_time:>8.1f}s      {stage4_time/60:>8.1f} minutes")
    print("-" * 60)
    print(f"{'TOTAL':<25} {total_time:>8.1f}s      {total_time/60:>8.1f} minutes")
    print("=" * 60)

    sys.exit(0)


if __name__ == "__main__":
    main()
