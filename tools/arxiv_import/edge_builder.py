#!/usr/bin/env python3
"""
Edge Builder - Construct Relational Graph Structure
====================================================

Builds edge collections capturing relational position in arXiv corpus.

Philosophy: Vector WHAT, Graph WHERE
- Embeddings capture WHAT papers are about (semantic content)
- Graph edges capture WHERE papers sit relationally (structural position)

Edge Types:
1. Category Co-occurrence - Papers sharing ≥2 categories (multi-disciplinary bridges)
2. Temporal Succession - Papers in same field published 1-3 months apart (influence flow)

No semantic edges from embeddings - keeps dimensions orthogonal.

Usage:
    python edge_builder.py
    python edge_builder.py --dry-run  # Count edges without inserting
"""

import logging
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Any
from math import exp
import yaml
import argparse

# Add metis to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from metis.database import ArangoClient, resolve_client_config, CollectionDefinition

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EdgeBuilder:
    """
    Build graph edge collections from imported arXiv papers.

    Constructs two edge types:
    - category_links: Multi-category co-occurrence (theory-practice bridges)
    - temporal_succession: Temporal ordering within fields (influence flow)
    """

    def __init__(self, config_path: Path):
        """
        Create an EdgeBuilder configured from the YAML file at config_path.
        
        Loads the YAML configuration, resolves the database client configuration (including Unix socket and database name), and stores collection names and edge-related settings on the instance for later use.
        
        Parameters:
            config_path (Path): Path to the YAML configuration file used to initialize the builder.
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {config_path}")

        # Database configuration
        socket_path = self.config['database']['socket_path']
        logger.info(f"Using Unix socket: {socket_path}")

        self.db_config = resolve_client_config(
            database=self.config['database']['name'],
            socket_path=socket_path,
            use_proxies=False
        )

        self.collections = self.config['database']['collections']
        self.edge_config = self.config['edges']

    def load_paper_metadata(self, client: ArangoClient) -> List[Dict[str, Any]]:
        """
        Load paper metadata needed for edge construction.

        Returns:
            List of papers with _key, categories, primary_category, year_month
        """
        logger.info("Loading paper metadata from database...")

        aql = f"""
        FOR paper IN {self.collections['papers']}
          RETURN {{
            _key: paper._key,
            categories: paper.categories,
            primary_category: paper.primary_category,
            year_month: paper.year_month
          }}
        """

        papers = client.execute_query(aql, {})
        logger.info(f"Loaded {len(papers):,} papers")

        return papers

    def build_category_links(
        self,
        papers: List[Dict[str, Any]],
        min_shared: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Constructs undirected category co-occurrence edges between papers.
        
        Builds one edge for each unordered pair of papers that share at least `min_shared` categories. Each edge document references the embeddings collection for both papers and includes the overlap weight and the list of shared categories.
        
        Parameters:
            papers (List[Dict[str, Any]]): Paper metadata objects containing at minimum `_key` and `categories`.
            min_shared (int): Minimum number of shared categories required to create an edge.
        
        Returns:
            List[Dict[str, Any]]: Edge documents with fields `_from`, `_to`, `type` (set to `'category_overlap'`), `weight` (number of shared categories), and `shared_categories` (list of shared category identifiers).
        """
        logger.info(f"Building category links (min_shared={min_shared})...")

        # Build inverted index: category -> set of paper keys
        category_index: Dict[str, Set[str]] = defaultdict(set)

        for paper in papers:
            categories = paper.get('categories', [])
            if not categories:
                continue

            for cat in categories:
                category_index[cat].add(paper['_key'])

        logger.info(f"Found {len(category_index)} unique categories")

        # Build paper lookup index (O(1) access)
        papers_by_key = {p['_key']: p for p in papers}

        # Find papers sharing categories
        edges = []
        edge_set = set()  # Track (key1, key2) pairs to avoid duplicates

        for i, paper in enumerate(papers):
            if i % 10000 == 0 and i > 0:
                logger.info(f"Processed {i:,} papers, found {len(edges):,} edges")

            categories = paper.get('categories', [])
            if len(categories) < min_shared:
                continue  # Can't share ≥min_shared if paper has fewer categories

            # Find candidate papers (those sharing at least 1 category)
            candidates: Set[str] = set()
            for cat in categories:
                candidates.update(category_index[cat])

            # Remove self
            candidates.discard(paper['_key'])

            # Check each candidate for sufficient category overlap
            paper_cats = set(categories)
            for candidate_key in candidates:
                # Avoid duplicate edges (undirected)
                edge_pair = tuple(sorted([paper['_key'], candidate_key]))
                if edge_pair in edge_set:
                    continue

                # Find candidate's categories using index lookup
                candidate = papers_by_key.get(candidate_key)
                if not candidate:
                    continue

                candidate_cats = set(candidate.get('categories', []))
                shared = paper_cats & candidate_cats

                if len(shared) >= min_shared:
                    edges.append({
                        '_from': f"{self.collections['embeddings']}/{paper['_key']}",
                        '_to': f"{self.collections['embeddings']}/{candidate_key}",
                        'type': 'category_overlap',
                        'weight': len(shared),
                        'shared_categories': list(shared)
                    })
                    edge_set.add(edge_pair)

        logger.info(f"Created {len(edges):,} category links")
        return edges

    def build_temporal_succession(
        self,
        papers: List[Dict[str, Any]],
        min_months: int = 1,
        max_months: int = 3,
        max_edges_per_paper: int = 50,
        decay_alpha: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Builds directed temporal succession edges between papers within the same primary category.
        
        Parameters:
            papers (List[Dict[str, Any]]): Paper metadata containing at minimum '_key', 'primary_category', and 'year_month'.
            min_months (int): Minimum month difference required to create an edge.
            max_months (int): Maximum month difference allowed to create an edge.
            max_edges_per_paper (int): Maximum number of outgoing temporal edges to create per source paper.
            decay_alpha (float): Exponential decay rate used to compute edge weight from month difference.
        
        Returns:
            List[Dict[str, Any]]: Edge documents ready for insertion, each containing '_from', '_to', 'type', 'weight', and 'months_diff'.
        """
        logger.info(f"Building temporal succession edges (window={min_months}-{max_months} months)...")

        # Group papers by primary_category
        category_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for paper in papers:
            primary_cat = paper.get('primary_category')
            year_month = paper.get('year_month')

            if not primary_cat or not year_month:
                continue

            category_groups[primary_cat].append(paper)

        logger.info(f"Grouped papers into {len(category_groups)} primary categories")

        # Sort each category by year_month
        for cat, papers_in_cat in category_groups.items():
            papers_in_cat.sort(key=lambda p: p['year_month'])

        # Build edges within each category
        edges = []

        for cat_idx, (cat, papers_in_cat) in enumerate(category_groups.items()):
            if cat_idx % 10 == 0:
                logger.info(f"Processing category {cat_idx+1}/{len(category_groups)}: {cat} ({len(papers_in_cat)} papers)")

            for i, p1 in enumerate(papers_in_cat):
                edges_created = 0

                # Look at papers published after p1
                for p2 in papers_in_cat[i+1:]:
                    months_diff = self._month_difference(p1['year_month'], p2['year_month'])

                    if months_diff < min_months:
                        continue  # Too soon
                    elif months_diff > max_months:
                        break  # Papers are sorted, no need to continue

                    # Create edge with exponential decay weight
                    weight = exp(-decay_alpha * months_diff)

                    edges.append({
                        '_from': f"{self.collections['embeddings']}/{p1['_key']}",
                        '_to': f"{self.collections['embeddings']}/{p2['_key']}",
                        'type': 'temporal_succession',
                        'weight': weight,
                        'months_diff': months_diff
                    })

                    edges_created += 1
                    if edges_created >= max_edges_per_paper:
                        break  # Cap reached

        logger.info(f"Created {len(edges):,} temporal succession edges")
        return edges

    @staticmethod
    def _month_difference(ym1: str, ym2: str) -> int:
        """
        Calculate month difference between YYYYMM strings.

        Args:
            ym1: Earlier date (YYYYMM or YYYY-MM format)
            ym2: Later date (YYYYMM or YYYY-MM format)

        Returns:
            Number of months difference (positive if ym2 > ym1)

        Raises:
            ValueError: If inputs are not in valid YYYYMM or YYYY-MM format
        """
        # Normalize format: remove hyphen if present (YYYY-MM → YYYYMM)
        ym1_normalized = ym1.replace('-', '')
        ym2_normalized = ym2.replace('-', '')

        # Validate format: must be exactly 6 digits
        if len(ym1_normalized) != 6 or not ym1_normalized.isdigit():
            raise ValueError(f"Invalid date format for ym1: {ym1} (expected YYYYMM or YYYY-MM)")
        if len(ym2_normalized) != 6 or not ym2_normalized.isdigit():
            raise ValueError(f"Invalid date format for ym2: {ym2} (expected YYYYMM or YYYY-MM)")

        # Extract and validate year/month
        y1, m1 = int(ym1_normalized[:4]), int(ym1_normalized[4:])
        y2, m2 = int(ym2_normalized[:4]), int(ym2_normalized[4:])

        # Validate month range (01-12)
        if not (1 <= m1 <= 12):
            raise ValueError(f"Invalid month in ym1: {m1} (must be 01-12)")
        if not (1 <= m2 <= 12):
            raise ValueError(f"Invalid month in ym2: {m2} (must be 01-12)")

        return (y2 - y1) * 12 + (m2 - m1)

    def create_edge_collections(self, client: ArangoClient) -> None:
        """
        Ensure the edge collections for category_links and temporal_succession exist and have a persistent index on the `type` field.
        
        Creates two edge collections (names taken from the instance's collection mapping) with a minimal persistent index on `type`, logs creation outcomes, and catches/logs any exceptions encountered during creation.
        """
        logger.info("Creating edge collections...")

        # Define edge collections with minimal indexes
        collection_defs = [
            CollectionDefinition(
                name=self.collections['category_links'],
                type="edge",
                indexes=[
                    {'type': 'persistent', 'fields': ['type'], 'name': 'idx_type'},
                ]
            ),
            CollectionDefinition(
                name=self.collections['temporal_succession'],
                type="edge",
                indexes=[
                    {'type': 'persistent', 'fields': ['type'], 'name': 'idx_type'},
                ]
            ),
        ]

        try:
            client.create_collections(collection_defs)
            logger.info(f"Created {len(collection_defs)} edge collections:")
            for coll_def in collection_defs:
                logger.info(f"  - {coll_def.name} (edge collection)")
        except Exception as e:
            logger.info(f"Edge collection setup: {e}")

    def insert_edges(
        self,
        client: ArangoClient,
        collection: str,
        edges: List[Dict[str, Any]],
        batch_size: int = 10000
    ) -> int:
        """
        Bulk-insert edge documents into the specified ArangoDB edge collection in batches.
        
        Parameters:
            collection (str): Name of the target edge collection.
            edges (List[Dict[str, Any]]): Sequence of edge documents prepared for insertion (each must contain `_from` and `_to`).
            batch_size (int): Number of documents to import per batch; defaults to 10000.
        
        Returns:
            int: Total number of edges successfully inserted.
        """
        if not edges:
            logger.warning(f"No edges to insert into {collection}")
            return 0

        logger.info(f"Inserting {len(edges):,} edges into {collection}...")

        total_inserted = 0

        for i in range(0, len(edges), batch_size):
            batch = edges[i:i+batch_size]
            count = client.bulk_import(collection, batch)
            total_inserted += count

            if (i + batch_size) % 50000 == 0:
                logger.info(f"  Inserted {total_inserted:,}/{len(edges):,} edges")

        logger.info(f"Successfully inserted {total_inserted:,} edges into {collection}")
        return total_inserted

    def run(self, dry_run: bool = False) -> Dict[str, int]:
        """
        Execute the full edge construction pipeline for the configured corpus and optionally persist generated edges.
        
        Parameters:
            dry_run (bool): If True, build and count edges but do not create collections or insert edges into the database.
        
        Returns:
            stats (Dict[str, int]): Mapping with keys 'category_links', 'temporal_succession', and 'total' containing the number of edges produced for each type and the aggregate total.
        """
        stats = {
            'category_links': 0,
            'temporal_succession': 0,
            'total': 0
        }

        with ArangoClient(self.db_config) as client:
            # Load paper metadata
            papers = self.load_paper_metadata(client)

            if not papers:
                logger.error("No papers found in database!")
                return stats

            # Build category links
            if self.edge_config.get('category_links', {}).get('enabled', True):
                min_shared = self.edge_config['category_links'].get('min_shared_categories', 2)
                category_edges = self.build_category_links(papers, min_shared=min_shared)
                stats['category_links'] = len(category_edges)

                if not dry_run:
                    self.create_edge_collections(client)
                    self.insert_edges(
                        client,
                        self.collections['category_links'],
                        category_edges
                    )

            # Build temporal succession edges
            if self.edge_config.get('temporal_succession', {}).get('enabled', True):
                temp_config = self.edge_config['temporal_succession']
                temporal_edges = self.build_temporal_succession(
                    papers,
                    min_months=temp_config.get('min_months', 1),
                    max_months=temp_config.get('max_months', 3),
                    max_edges_per_paper=temp_config.get('max_edges_per_paper', 50),
                    decay_alpha=temp_config.get('decay_alpha', 0.3)
                )
                stats['temporal_succession'] = len(temporal_edges)

                if not dry_run:
                    self.create_edge_collections(client)
                    self.insert_edges(
                        client,
                        self.collections['temporal_succession'],
                        temporal_edges
                    )

            stats['total'] = stats['category_links'] + stats['temporal_succession']

        logger.info("=" * 60)
        logger.info("Edge Building Complete")
        logger.info("=" * 60)
        logger.info(f"Category links:        {stats['category_links']:,}")
        logger.info(f"Temporal succession:   {stats['temporal_succession']:,}")
        logger.info(f"Total edges:           {stats['total']:,}")
        logger.info("=" * 60)

        return stats


def main():
    """
    Parse command-line arguments, run the EdgeBuilder pipeline, and exit with a status reflecting whether any edges were created.
    
    This function accepts command-line options for the configuration file path and a dry-run flag, instantiates an EdgeBuilder with the provided config, executes the run process, and terminates the program with exit code 0 when edges were created or 1 when no edges were produced.
    """
    parser = argparse.ArgumentParser(description='Build graph edges for arXiv corpus')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path(__file__).parent / 'config' / 'arxiv_import.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Count edges without inserting to database'
    )

    args = parser.parse_args()

    builder = EdgeBuilder(args.config)
    stats = builder.run(dry_run=args.dry_run)

    # Exit code based on success
    if stats['total'] == 0:
        logger.error("No edges created!")
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()