#!/usr/bin/env python3
"""
arXiv Import Pipeline - Multi-GPU Version
==========================================

Parallel GPU workers for embedding generation on 2.8M arXiv papers.

Key differences from single-GPU version:
- Uses multiprocessing with spawn context
- Each worker sets CUDA_VISIBLE_DEVICES before loading models
- Queue-based work distribution
- Separate storage thread for database writes

Usage:
    # Default: single GPU using sentence-transformers backend (GPU 0)
    python import_pipeline_multigpu.py --limit 1000

    # Multi-GPU: choose GPUs explicitly (e.g., 0 and 1)
    python import_pipeline_multigpu.py --multi-gpu 0,1 --workers 2 --limit 1000
"""

import json
import logging
import sys
import os
import multiprocessing as mp
import threading
import signal
from pathlib import Path
from typing import Iterator, Dict, Any, Optional
from datetime import datetime, timezone
from queue import Empty
import time
import yaml
import argparse

# Add project root to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from arxiv_parser import ArxivIdParser

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_chunk(raw_lines, created_at_timestamp):
    """Parse a chunk of JSON lines (runs in separate process)."""
    from arxiv_parser import ArxivIdParser
    import json

    docs = []
    errors = 0

    for line in raw_lines:
        try:
            paper = json.loads(line)
            parsed = ArxivIdParser.parse(paper['id'])

            categories_str = paper.get('categories', '')
            categories = categories_str.split() if categories_str else []
            primary_category = categories[0] if categories else None

            authors_field = paper.get('authors', '')
            if isinstance(authors_field, str):
                authors = [a.strip() for a in authors_field.split(',') if a.strip()]
            else:
                authors = authors_field

            metadata_doc = {
                "_key": parsed.internal_id,
                "arxiv_id": paper['id'],
                "authors": authors,
                "categories": categories,
                "primary_category": primary_category,
                "year": parsed.year,
                "month": parsed.month,
                "year_month": f"{parsed.year:04d}{parsed.month:02d}",
                "created_at": created_at_timestamp,
            }

            abstract_doc = {
                "_key": parsed.internal_id,
                "arxiv_id": paper['id'],
                "title": paper.get('title', '').strip(),
                "abstract": paper.get('abstract', '').strip(),
            }

            # Pre-compute combined text for GPU (avoid string concat on GPU)
            title = paper.get('title', '').strip()
            abstract = paper.get('abstract', '').strip()
            combined_text = f"{title}\n\n{abstract}"

            embedding_doc = {
                "_key": parsed.internal_id,
                "arxiv_id": paper['id'],
                "title": title,
                "abstract": abstract,
                "combined_text": combined_text,  # Pre-computed for GPU
            }

            docs.append({
                "metadata": metadata_doc,
                "abstract": abstract_doc,
                "embedding": embedding_doc
            })
        except Exception:
            errors += 1

    return docs, errors


def postprocess_worker(
    worker_id: int,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    stop_event: Any,
):
    """
    CPU worker process - normalizes embeddings and converts to lists.
    Runs in parallel with storage to prevent bottleneck.
    """
    import numpy as np
    batches_processed = 0

    try:
        while not stop_event.is_set():
            try:
                batch = input_queue.get(timeout=1.0)

                if batch is None:  # Poison pill
                    break

                # Process embeddings on CPU: normalize + convert to lists
                for doc in batch:
                    emb_doc = doc['embedding']

                    # Normalize embeddings and convert to lists
                    for key in ['title_embedding', 'abstract_embedding', 'combined_embedding']:
                        if emb_doc.get(key) is not None:
                            arr = emb_doc[key]
                            if isinstance(arr, np.ndarray):
                                # L2 normalization
                                norm = np.linalg.norm(arr)
                                if norm > 0:
                                    arr = arr / norm
                                # Convert to list for JSON storage
                                emb_doc[key] = arr.tolist()

                    # Remove temporary fields used during processing
                    emb_doc.pop('combined_text', None)
                    emb_doc.pop('title', None)
                    emb_doc.pop('abstract', None)

                # Send to storage queue
                output_queue.put(batch)
                batches_processed += 1

            except Empty:
                continue
            except Exception:
                logger.exception(f"Postprocess worker {worker_id} error")

    except Exception:
        logger.exception(f"Postprocess worker {worker_id} failed")
    finally:
        print(f"Postprocess worker {worker_id} finished - {batches_processed} batches")


def embedding_worker(
    worker_id: int,
    gpu_id: int,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    stop_event: Any,
    config: dict,
    multi_gpu: bool
):
    """
    GPU worker process - generates embeddings.

    Sets CUDA_VISIBLE_DEVICES before any CUDA imports.
    """
    # CRITICAL: Set GPU BEFORE any imports that may pull in torch/transformers
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    print(f"Worker {worker_id} starting on GPU {gpu_id}, PID {os.getpid()}")

    batches_processed = 0

    try:
        # Import torch and set device AFTER CUDA_VISIBLE_DEVICES
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # 0 refers to the isolated GPU after CUDA_VISIBLE_DEVICES
            try:
                print(
                    f"Worker {worker_id} CUDA visible: {torch.cuda.device_count()}, current: {torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})"
                )
            except Exception:
                pass

        # Enable fast matmul on Ampere
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        # Initialize embedder (imports torch/cuda here)
        emb_config = config['embeddings']

        # Import create_embedder only after CUDA env is set to avoid early torch init
        from metis import create_embedder

        # Backend selection
        if multi_gpu:
            model_name = "jinaai/jina-embeddings-v4-transformers"  # transformers backend for multi-GPU
        else:
            model_name = "jinaai/jina-embeddings-v4"  # sentence-transformers backend for single-GPU

        embedder = create_embedder(
            model_name,
            device='cuda:0',  # Always 0 since CUDA_VISIBLE_DEVICES isolated the GPU
            batch_size=emb_config['batch_size'],
            use_fp16=emb_config.get('use_fp16', True),
            max_seq_length=emb_config.get('max_length', 32768),
            chunk_size_tokens=emb_config.get('chunk_size_tokens', 500),
            chunk_overlap_tokens=emb_config.get('chunk_overlap_tokens', 200)
        )

        print(f"Worker {worker_id} ready (GPU {gpu_id}, batch_size={emb_config['batch_size']})")

        # Process batches from queue
        while not stop_event.is_set():
            try:
                batch = input_queue.get(timeout=1.0)

                if batch is None:  # Poison pill
                    break

                # Generate embeddings
                combined_only = bool(emb_config.get('combined_only', False))

                if combined_only:
                    # Use pre-computed combined text from CPU parsing
                    combined = [doc['embedding'].get('combined_text', '') for doc in batch]
                    combined_embeds = embedder.embed_texts(combined, prompt_name='passage')

                    # Attach embeddings (keep as numpy, convert to list in storage worker)
                    for i, doc in enumerate(batch):
                        doc['embedding']['title_embedding'] = None
                        doc['embedding']['abstract_embedding'] = None
                        doc['embedding']['combined_embedding'] = combined_embeds[i]  # Keep as numpy array
                else:
                    titles = [doc['embedding'].get('title', '') for doc in batch]
                    abstracts = [doc['embedding'].get('abstract', '') for doc in batch]
                    combined = [doc['embedding'].get('combined_text', '') for doc in batch]

                    # Use appropriate prompts for Jina v4
                    title_embeds = embedder.embed_texts(titles, prompt_name='query')
                    abstract_embeds = embedder.embed_texts(abstracts, prompt_name='passage')
                    combined_embeds = embedder.embed_texts(combined, prompt_name='passage')

                    # Attach embeddings (keep as numpy, normalization done in storage worker)
                    for i, doc in enumerate(batch):
                        doc['embedding']['title_embedding'] = title_embeds[i]  # Keep as numpy
                        doc['embedding']['abstract_embedding'] = abstract_embeds[i]  # Keep as numpy
                        doc['embedding']['combined_embedding'] = combined_embeds[i]  # Keep as numpy

                # Send to output queue
                output_queue.put(batch)
                batches_processed += 1

                if batches_processed % 10 == 0:
                    print(f"Worker {worker_id}: {batches_processed} batches processed")

            except Empty:
                continue
            except Exception:
                logger.exception(f"Worker {worker_id} error")

    except Exception:
        logger.exception(f"Worker {worker_id} initialization failed")
    finally:
        print(f"Worker {worker_id} finished - {batches_processed} batches")


def storage_worker(
    output_queue: mp.Queue,
    stop_event: Any,
    db_config: Any,
    collections_config: dict,
    stats_dict: dict,
    chunk_sizes: dict | None = None,
):
    """
    Storage thread - writes batches to database.
    """
    print(f"Storage worker starting, PID {os.getpid()}")

    batches_stored = 0

    try:
        # Import DB client lazily to avoid pulling the metis package (and embedders) too early
        from metis.database import ArangoClient

        with ArangoClient(db_config) as client:
            while not stop_event.is_set() or not output_queue.empty():
                try:
                    batch = output_queue.get(timeout=1.0)

                    if batch is None:  # Poison pill
                        break

                    # Extract documents for each collection (already processed by postprocess workers)
                    metadata_docs = [doc['metadata'] for doc in batch]
                    abstract_docs = [doc['abstract'] for doc in batch]
                    embedding_docs = [doc['embedding'] for doc in batch]

                    # Resolve per-collection chunk sizes (NDJSON rows per import)
                    cs = chunk_sizes or {}
                    papers_chunk = int(cs.get('papers', 4000))
                    abstracts_chunk = int(cs.get('abstracts', 4000))
                    embeddings_chunk = int(cs.get('embeddings', 750))

                    # Sequential imports to maintain atomicity
                    # (All 3 must succeed or fail together for data consistency)
                    metadata_count = client.bulk_import(
                        collections_config['papers'], metadata_docs, chunk_size=papers_chunk
                    )
                    abstract_count = client.bulk_import(
                        collections_config['abstracts'], abstract_docs, chunk_size=abstracts_chunk
                    )
                    embedding_count = client.bulk_import(
                        collections_config['embeddings'], embedding_docs, chunk_size=embeddings_chunk
                    )

                    batches_stored += 1
                    stats_dict['total_created'] += metadata_count

                    # Log counts for verification
                    logger.debug(
                        f"Imported batch: {metadata_count} metadata, {abstract_count} abstracts, "
                        f"{embedding_count} embeddings"
                    )

                    if batches_stored % 10 == 0:
                        print(f"Storage: {batches_stored} batches stored ({stats_dict['total_created']} papers)")

                except Empty:
                    if stop_event.is_set() and output_queue.empty():
                        break
                    continue
                except Exception:
                    logger.exception(f"Storage error")

    except Exception:
        logger.exception(f"Storage worker failed")
    finally:
        print(f"Storage worker finished - {batches_stored} batches stored")


class MultiGPUArxivImporter:
    """arXiv importer with optional multi-GPU parallelism.

    Default: single GPU using sentence-transformers backend.
    With --multi-gpu: one worker per selected GPU using transformers backend.
    """

    def __init__(self, config_path: Path, num_workers: int = 1, gpu_ids: Optional[list[int]] = None):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.gpu_ids = list(gpu_ids) if gpu_ids else None
        self.multi_gpu = bool(self.gpu_ids)
        self.num_workers = num_workers
        self.data_file = Path(self.config['data']['source_file'])
        self.batch_size = self.config['import']['batch_size']

        # Database config
        # Prefer Metis RO/RW UDS proxies for best performance.
        from metis.database import resolve_client_config
        db_name = self.config['database']['name']
        use_tcp = bool(self.config['database'].get('use_tcp', False))

        if use_tcp:
            # Explicit TCP mode (fallback/dev)
            self.db_config = resolve_client_config(
                database=db_name,
                use_proxies=False,
            )
        else:
            # Default: use Metis RO/RW Unix sockets
            self.db_config = resolve_client_config(
                database=db_name,
                use_proxies=True,
            )

        # Helpful visibility on transport in logs
        try:
            print(
                f"Arango via UDS -> RO: {self.db_config.read_socket}, RW: {self.db_config.write_socket}, base_url: {self.db_config.base_url}"
            )
        except Exception:
            pass

        # Multiprocessing setup (spawn context for CUDA)
        ctx = mp.get_context('spawn')
        self.input_queue = ctx.Queue(maxsize=50)  # Main → GPU workers
        self.postprocess_queue = ctx.Queue(maxsize=50)  # GPU → Postprocess workers
        self.storage_queue = ctx.Queue(maxsize=50)  # Postprocess → Storage
        self.stop_event = ctx.Event()

        # Shared stats (use Manager for cross-process sharing)
        manager = ctx.Manager()
        self.stats_dict = manager.dict()
        self.stats_dict['total_created'] = 0
        self.stats_dict['total_errors'] = 0

        self.gpu_workers = []
        self.postprocess_workers = []
        self.storage_thread = None
        self.num_postprocess_workers = 8  # 8 CPU workers for post-processing

        # Cache timestamp for all papers (avoid 2.8M datetime calls)
        self.created_at_timestamp = datetime.now(timezone.utc).isoformat()

    def create_database_if_needed(self):
        """Create database via Unix socket if it doesn't exist."""
        import httpx
        import os

        db_name = self.config['database']['name']
        print(f"Checking if database '{db_name}' exists...")

        # Prefer configured RW socket, else env or Metis default
        socket_path = (
            self.config['database'].get('socket_path')
            or os.environ.get('ARANGO_RW_SOCKET')
            or '/run/metis/readwrite/arangod.sock'
        )
        username = os.environ.get('ARANGO_USERNAME', 'root')
        password = os.environ.get('ARANGO_PASSWORD', '')

        try:
            transport = httpx.HTTPTransport(uds=socket_path, retries=0)
            with httpx.Client(transport=transport, base_url="http://arangodb") as client:
                response = client.get("/_api/database/user", auth=(username, password))

                if response.status_code == 200:
                    databases = response.json().get('result', [])
                    if db_name in databases:
                        print(f"Database '{db_name}' already exists")
                        return

                    print(f"Creating database '{db_name}'...")
                    create_response = client.post(
                        "/_api/database",
                        json={"name": db_name},
                        auth=(username, password)
                    )

                    if create_response.status_code in [200, 201]:
                        print(f"Successfully created database '{db_name}'")
                    elif create_response.status_code == 409:
                        print(f"Database '{db_name}' already exists")
                    else:
                        print(f"Database creation returned status {create_response.status_code}: {create_response.text}")
                else:
                    print(f"Could not check databases (status {response.status_code}). Database '{db_name}' may need manual creation")
        except Exception as e:
            print(f"Could not create database: {e}. Please create database '{db_name}' manually")

    def setup_database(self):
        """Create collections and indexes."""
        print("Setting up database schema...")

        collections_config = self.config['database']['collections']

        papers_indexes = [
            {'type': 'persistent', 'fields': ['arxiv_id'], 'unique': True, 'name': 'idx_arxiv_id'},
            {'type': 'persistent', 'fields': ['categories[*]'], 'name': 'idx_categories'},
            {'type': 'persistent', 'fields': ['year', 'month'], 'name': 'idx_year_month'},
            {'type': 'persistent', 'fields': ['primary_category'], 'name': 'idx_primary_category'},
        ]

        abstracts_indexes = [
            {'type': 'persistent', 'fields': ['arxiv_id'], 'unique': True, 'name': 'idx_arxiv_id'},
        ]

        embeddings_indexes = [
            {'type': 'persistent', 'fields': ['arxiv_id'], 'unique': True, 'name': 'idx_arxiv_id'},
        ]

        # Import DB client + types lazily
        from metis.database import ArangoClient, CollectionDefinition

        collection_defs = [
            CollectionDefinition(name=collections_config['papers'], type="document", indexes=papers_indexes),
            CollectionDefinition(name=collections_config['abstracts'], type="document", indexes=abstracts_indexes),
            CollectionDefinition(name=collections_config['embeddings'], type="document", indexes=embeddings_indexes),
        ]

        with ArangoClient(self.db_config) as client:
            client.create_collections(collection_defs)

        print("Database setup complete")

    def start_workers(self):
        """Start GPU workers and storage thread."""
        # Detect GPUs using nvidia-smi (avoid importing torch in parent process)
        import subprocess
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True
            )
            available_gpu_indices = [int(x) for x in result.stdout.strip().split('\n') if x.strip()]
            num_gpus = len(available_gpu_indices)
            print(f"Detected {num_gpus} GPU(s)")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("No GPUs available or nvidia-smi not found")

        ctx = mp.get_context('spawn')

        if self.multi_gpu:
            # Validate selected GPUs
            invalid = [] if self.gpu_ids is None else [gid for gid in self.gpu_ids if gid not in available_gpu_indices]
            if invalid:
                raise ValueError(f"Invalid GPU id(s) in --multi-gpu: {invalid}. Available GPUs: {available_gpu_indices}")

            assigned_gpu_ids = list(self.gpu_ids or [])
            # Enforce one worker per selected GPU
            if self.num_workers != len(assigned_gpu_ids):
                raise ValueError(
                    f"--multi-gpu requires --workers == number of selected GPUs ({len(assigned_gpu_ids)}), got {self.num_workers}"
                )
            print(f"Using GPU IDs: {assigned_gpu_ids}")
        else:
            # Force single-worker single-GPU (GPU 0)
            if self.num_workers != 1:
                print(f"Single-GPU mode: overriding workers={self.num_workers} -> 1")
            self.num_workers = 1
            assigned_gpu_ids = [0]

        # Start GPU embedding workers
        for i, gpu_id in enumerate(assigned_gpu_ids):
            p = ctx.Process(
                target=embedding_worker,
                args=(i, gpu_id, self.input_queue, self.postprocess_queue, self.stop_event, self.config, self.multi_gpu)
            )
            p.start()
            self.gpu_workers.append(p)
            print(f"Started GPU worker {i} on GPU {gpu_id}")

        # Start CPU postprocess workers (8 workers for parallel normalization/conversion)
        for i in range(self.num_postprocess_workers):
            p = ctx.Process(
                target=postprocess_worker,
                args=(i, self.postprocess_queue, self.storage_queue, self.stop_event)
            )
            p.start()
            self.postprocess_workers.append(p)
        print(f"Started {self.num_postprocess_workers} postprocess workers")

        # Start storage thread
        self.storage_thread = threading.Thread(
            target=storage_worker,
            args=(
                self.storage_queue,
                self.stop_event,
                self.db_config,
                self.config['database']['collections'],
                self.stats_dict,
                self.config['database'].get('chunk_sizes', {})
            )
        )
        self.storage_thread.start()
        print("Started storage worker")

    def read_and_process_papers(self, limit: Optional[int] = None, skip: int = 0):
        """Read papers and push to input queue."""
        print(f"Reading papers from {self.data_file}")
        if skip > 0:
            print(f"Skipping first {skip:,} papers")
        if limit:
            print(f"Limiting to {limit:,} papers")

        # PHASE 1a: Load raw JSON lines into RAM
        print("Phase 1a: Loading raw JSON into RAM...")
        start_time = time.time()
        raw_papers = []

        with open(self.data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < skip:
                    continue
                if limit and (i - skip) >= limit:
                    break
                raw_papers.append(line)

        elapsed = time.time() - start_time
        print(f"Phase 1a complete: Loaded {len(raw_papers):,} raw papers in {elapsed:.1f}s")

        # PHASE 1b: Parse with multiprocessing (36 CPU workers)
        print("Phase 1b: Parsing papers with 36 CPU workers...")
        start_time = time.time()

        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Split into chunks for workers (~78k papers per worker)
        chunk_size = len(raw_papers) // 36 + 1
        chunks = [raw_papers[i:i + chunk_size] for i in range(0, len(raw_papers), chunk_size)]

        all_docs = []
        parse_errors = 0

        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=36) as executor:
            # Submit all chunks
            future_to_chunk = {executor.submit(parse_chunk, chunk, self.created_at_timestamp): i
                             for i, chunk in enumerate(chunks)}

            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk_docs, chunk_errors = future.result()
                all_docs.extend(chunk_docs)
                parse_errors += chunk_errors

                # Progress reporting
                elapsed = time.time() - start_time
                rate = len(all_docs) / elapsed if elapsed > 0 else 0
                print(f"Parsed {len(all_docs):,} papers ({rate:.1f} papers/sec, {len(future_to_chunk) - len([f for f in future_to_chunk if f.done()])} chunks remaining)")

        elapsed = time.time() - start_time
        rate = len(all_docs) / elapsed if elapsed > 0 else 0
        print(f"Phase 1b complete: Parsed {len(all_docs):,} papers in {elapsed:.1f}s ({rate:.1f} papers/sec)")
        if parse_errors > 0:
            print(f"Parse errors: {parse_errors:,}")

        # PHASE 2: Queue batches for GPU workers (from RAM, fast)
        print(f"Phase 2: Queueing {len(all_docs):,} papers for GPU workers...")
        start_time = time.time()

        for i in range(0, len(all_docs), self.batch_size):
            batch = all_docs[i:i + self.batch_size]
            self.input_queue.put(batch)

            if (i + len(batch)) % 10000 == 0:
                print(f"Queued {i + len(batch):,} papers")

        elapsed = time.time() - start_time
        print(f"Phase 2 complete: Queued {len(all_docs):,} papers in {elapsed:.1f}s")

        # Send poison pills
        for _ in range(self.num_workers):
            self.input_queue.put(None)

        print(f"Total papers ready for processing: {len(all_docs):,}")

    def process_paper(self, paper: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Transform paper JSON into document structure."""
        try:
            parsed = ArxivIdParser.parse(paper['id'])

            categories_str = paper.get('categories', '')
            categories = categories_str.split() if categories_str else []
            primary_category = categories[0] if categories else None

            authors_field = paper.get('authors', '')
            if isinstance(authors_field, str):
                authors = [a.strip() for a in authors_field.split(',') if a.strip()]
            else:
                authors = authors_field

            metadata_doc = {
                "_key": parsed.internal_id,
                "arxiv_id": paper['id'],
                "authors": authors,
                "categories": categories,
                "primary_category": primary_category,
                "year": parsed.year,
                "month": parsed.month,
                "year_month": f"{parsed.year:04d}{parsed.month:02d}",
                "created_at": self.created_at_timestamp,  # Use cached timestamp
            }

            abstract_doc = {
                "_key": parsed.internal_id,
                "arxiv_id": paper['id'],
                "title": paper.get('title', '').strip(),
                "abstract": paper.get('abstract', '').strip(),
            }

            embedding_doc = {
                "_key": parsed.internal_id,
                "arxiv_id": paper['id'],
                "title_embedding": None,
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

    def shutdown_workers(self):
        """Clean shutdown of workers and storage."""
        print("Shutting down workers...")

        # Wait for GPU workers to finish
        print("Waiting for GPU workers to complete...")
        for i, p in enumerate(self.gpu_workers):
            p.join(timeout=120)
            if p.is_alive():
                print(f"Warning: GPU worker {i} did not exit cleanly, terminating...")
                p.terminate()
                p.join(timeout=10)

        # Send poison pills to postprocess workers
        print("Sending poison pills to postprocess workers...")
        for _ in range(self.num_postprocess_workers):
            self.postprocess_queue.put(None)

        # Wait for postprocess workers to finish
        print("Waiting for postprocess workers to complete...")
        for i, p in enumerate(self.postprocess_workers):
            p.join(timeout=60)
            if p.is_alive():
                print(f"Warning: Postprocess worker {i} did not exit cleanly, terminating...")
                p.terminate()
                p.join(timeout=10)

        # Signal storage thread to stop after draining storage queue
        print("Waiting for storage thread to drain storage queue...")
        self.stop_event.set()

        # Give storage thread time to drain remaining batches
        if self.storage_thread:
            self.storage_thread.join(timeout=120)
            if self.storage_thread.is_alive():
                print("Warning: Storage thread did not exit cleanly")

        # Report final queue states
        print(f"Final queue states: input={self.input_queue.qsize()}, postprocess={self.postprocess_queue.qsize()}, storage={self.storage_queue.qsize()}")
        print("All workers stopped")

    def run(self, limit: Optional[int] = None, skip: int = 0, setup_db: bool = True):
        """Run the multi-GPU import pipeline."""
        start_time = time.time()

        # Register signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}, initiating graceful shutdown...")
            self.stop_event.set()
            # Send poison pills to workers
            for _ in range(self.num_workers):
                try:
                    self.input_queue.put(None, timeout=1.0)
                except:
                    pass

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            if setup_db:
                self.create_database_if_needed()
                self.setup_database()

            self.start_workers()

            self.read_and_process_papers(limit=limit, skip=skip)

            self.shutdown_workers()

            duration = time.time() - start_time
            total_created = self.stats_dict['total_created']
            throughput = total_created / duration if duration > 0 else 0

            return {
                'total_created': total_created,
                'total_errors': self.stats_dict['total_errors'],
                'duration': duration,
                'throughput': throughput
            }

        except KeyboardInterrupt:
            print("\nInterrupted by user, shutting down...")
            self.stop_event.set()
            self.shutdown_workers()
            raise
        except Exception as e:
            print(f"\nError during import: {e}")
            self.stop_event.set()
            self.shutdown_workers()
            raise


def main():
    parser = argparse.ArgumentParser(description='arXiv import pipeline (single-GPU by default, optional multi-GPU)')
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
        help='Maximum number of papers to import'
    )
    parser.add_argument(
        '--skip',
        type=int,
        default=0,
        help='Number of papers to skip'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Workers: single-GPU uses 1; multi-GPU requires workers == number of selected GPUs'
    )
    parser.add_argument(
        '--multi-gpu',
        type=str,
        default=None,
        metavar='GPU_LIST',
        help='Enable multi-GPU and select GPUs by IDs, e.g., "0,1,3"'
    )
    parser.add_argument(
        '--no-setup',
        action='store_true',
        help='Skip database setup'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("arXiv Import Pipeline")
    print("=" * 60)
    print(f"Workers: {args.workers}")
    print(f"Limit: {args.limit if args.limit else 'all'}")
    print(f"Multi-GPU: {args.multi_gpu if args.multi_gpu else False}")
    print("=" * 60)

    # Parse GPU list if provided
    gpu_ids = None
    if args.multi_gpu:
        try:
            gpu_ids = [int(x) for x in args.multi_gpu.split(',') if x.strip() != '']
        except ValueError:
            raise SystemExit("--multi-gpu must be a comma-separated list of integers, e.g., 0,1")

    importer = MultiGPUArxivImporter(args.config, num_workers=args.workers, gpu_ids=gpu_ids)
    stats = importer.run(
        limit=args.limit,
        skip=args.skip,
        setup_db=not args.no_setup
    )

    print("\n" + "=" * 60)
    print("IMPORT COMPLETE")
    print("=" * 60)
    print(f"Papers imported: {stats['total_created']:,}")
    print(f"Time: {stats['duration']:.1f}s")
    print(f"Throughput: {stats['throughput']:.1f} papers/sec")
    print("=" * 60)

    sys.exit(0)


if __name__ == "__main__":
    # CRITICAL: Set spawn method before any multiprocessing
    mp.set_start_method('spawn', force=True)
    main()
