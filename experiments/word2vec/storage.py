"""
CF Experiment Storage Manager

Manages ArangoDB storage for Conveyance Framework experiments.
"""

import logging
import os

# Add metis to path
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from metis.database import ArangoHttp2Client, ArangoHttp2Config

logger = logging.getLogger(__name__)


class CFExperimentStorage:
    """Manages ArangoDB storage for CF experiments."""

    def __init__(self,
                 client: Optional[ArangoHttp2Client] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize with configured ArangoDB client.

        Args:
            client: Existing ArangoDB client or None to create from config
            config: Database configuration dictionary
        """
        if client:
            self.client = client
        else:
            # Create client from config
            db_config = config.get("database", {}) if config else {}
            self.client = self._create_client(db_config)

        self.db_name = self.client.config.database
        self.collections = {
            "papers": "arxiv_markdown",
            "code": "arxiv_code",
            "embeddings": "arxiv_embeddings"
        }

    def _create_client(self, db_config: Dict[str, Any]) -> ArangoHttp2Client:
        """
        Create ArangoDB client from configuration.

        Args:
            db_config: Database configuration dictionary

        Returns:
            Configured ArangoDB client
        """
        # Use Unix sockets for optimal performance (0.4ms p50 latency)
        # Default to Metis read-write socket
        socket_path = db_config.get("socket_path", "/run/metis/readwrite/arangod.sock")

        # Get credentials from environment (standard for ArangoDB)
        username = os.environ.get("ARANGO_USERNAME", "root")
        password = os.environ.get("ARANGO_PASSWORD", "")

        config = ArangoHttp2Config(
            database=db_config.get("name", "cf_experiments"),
            socket_path=socket_path,
            base_url="http://localhost",  # Required but unused with socket
            username=username,
            password=password,
            connect_timeout=5.0,
            read_timeout=30.0,
            write_timeout=30.0
        )

        logger.info(f"Created ArangoDB client for database: {config.database}")
        return ArangoHttp2Client(config)

    def ensure_collections(self) -> None:
        """
        Create collections if they don't exist.

        Creates:
        - arxiv_markdown: Paper content
        - arxiv_code: Repository code
        - arxiv_embeddings: Vector embeddings
        """
        for collection_type, collection_name in self.collections.items():
            try:
                # Check if collection exists by trying to get its properties
                self.client.request(
                    "GET",
                    f"/_db/{self.db_name}/_api/collection/{collection_name}/properties"
                )
                logger.info(f"Collection {collection_name} already exists")

            except Exception as e:
                # Collection doesn't exist, create it
                if "404" in str(e) or "not found" in str(e).lower():
                    logger.info(f"Creating collection {collection_name}...")
                    self.client.request(
                        "POST",
                        f"/_db/{self.db_name}/_api/collection",
                        json={"name": collection_name}
                    )
                    logger.info(f"Created collection {collection_name}")
                else:
                    logger.error(f"Error checking collection {collection_name}: {e}")
                    raise

    def store_paper_markdown(self,
                           arxiv_id: str,
                           paper_doc: Dict[str, Any]) -> str:
        """
        Store paper markdown with metadata.

        Args:
            arxiv_id: arXiv identifier
            paper_doc: Paper document data

        Returns:
            Document key in database
        """
        # Prepare document
        doc_key = arxiv_id.replace(".", "_").replace("/", "_")

        document = {
            "_key": doc_key,
            "arxiv_id": arxiv_id,
            "title": paper_doc.get("title", ""),
            "authors": paper_doc.get("authors", []),
            "abstract": paper_doc.get("abstract", ""),
            "markdown_content": paper_doc.get("markdown_content", ""),
            "latex_source": paper_doc.get("latex_source"),
            "processing_metadata": {
                "tool": "docling",
                "version": "2.54.0",
                "timestamp": datetime.now().isoformat() + "Z",
                "word_count": len(paper_doc.get("markdown_content", "").split()),
                "processing_time_seconds": paper_doc.get("processing_time", 0),
                "has_latex_source": paper_doc.get("latex_source") is not None,
                "latex_chars": len(paper_doc.get("latex_source", "")),
                "latex_words": len(paper_doc.get("latex_source", "").split())
            },
            "experiment_tags": ["word2vec_family", "cf_validation"],
            "quality_metrics": {
                "conversion_success": bool(paper_doc.get("markdown_content")),
                "has_equations": "$$" in paper_doc.get("markdown_content", ""),
                "has_tables": "table" in paper_doc.get("markdown_content", "").lower(),
                "completeness_score": min(1.0, len(paper_doc.get("markdown_content", "")) / 10000)
            }
        }

        # Store document
        try:
            self.client.insert_documents(
                self.collections["papers"],
                [document],
                on_duplicate="update"
            )
            collection = self.collections["papers"]
            logger.info(f"Stored paper {arxiv_id} in {collection} (key: {doc_key})")
            return doc_key

        except Exception as e:
            logger.error(f"Failed to store paper {arxiv_id}: {e}")
            raise

    def store_code(self,
                  arxiv_id: str,
                  code_doc: Dict[str, Any]) -> str:
        """
        Store repository code with metadata.

        Args:
            arxiv_id: arXiv identifier
            code_doc: Code document data

        Returns:
            Document key in database
        """
        doc_key = f"{arxiv_id.replace('.', '_').replace('/', '_')}_code"

        document = {
            "_key": doc_key,
            "arxiv_id": arxiv_id,
            "github_url": code_doc.get("github_url"),
            "is_official": code_doc.get("is_official", False),
            "code_files": code_doc.get("code_files", {}),
            "repository_metadata": code_doc.get("metadata", {}),
            "processing_metadata": {
                "timestamp": datetime.now().isoformat() + "Z",
                "total_files": len(code_doc.get("code_files", {})),
                "total_lines": code_doc.get("total_lines", 0),
                "file_selection_strategy": "main_implementation"
            }
        }

        try:
            self.client.insert_documents(
                self.collections["code"],
                [document],
                on_duplicate="update"
            )
            collection = self.collections["code"]
            logger.info(f"Stored code for {arxiv_id} in {collection} (key: {doc_key})")
            return doc_key

        except Exception as e:
            logger.error(f"Failed to store code for {arxiv_id}: {e}")
            raise

    def store_embeddings(self,
                        arxiv_id: str,
                        embeddings: List[Dict[str, Any]]) -> int:
        """
        Store embedding vectors with chunk metadata.

        Args:
            arxiv_id: arXiv identifier
            embeddings: List of embedding documents

        Returns:
            Number of embeddings stored
        """
        documents = []

        for i, emb_data in enumerate(embeddings):
            doc_key = f"{arxiv_id.replace('.', '_').replace('/', '_')}_emb_{i}"

            document = {
                "_key": doc_key,
                "arxiv_id": arxiv_id,
                "chunk_index": i,
                "chunk_text_preview": emb_data.get("text_preview", "")[:200],
                "embedding": emb_data.get("embedding", []),
                "embedding_metadata": {
                    "model": "jinaai/jina-embeddings-v4",
                    "dimension": len(emb_data.get("embedding", [])),
                    "context_type": "combined_paper_code",
                    "context_tokens": emb_data.get("context_tokens", 0),
                    "chunk_tokens": emb_data.get("chunk_tokens", 0),
                    "chunk_overlap": emb_data.get("chunk_overlap", 200),
                    "timestamp": datetime.now().isoformat() + "Z"
                },
                "cf_metadata": {
                    "has_code": emb_data.get("has_code", False),
                    "paper_ratio": emb_data.get("paper_ratio", 1.0),
                    "code_ratio": emb_data.get("code_ratio", 0.0),
                    "experiment_id": "word2vec_cf_validation_v1"
                }
            }
            documents.append(document)

        if documents:
            try:
                self.client.insert_documents(
                    self.collections["embeddings"],
                    documents,
                    on_duplicate="update"
                )
                collection = self.collections["embeddings"]
                logger.info(f"Stored {len(documents)} embeddings for {arxiv_id} in {collection}")
                return len(documents)

            except Exception as e:
                logger.error(f"Failed to store embeddings for {arxiv_id}: {e}")
                raise

        return 0

    def get_paper(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve paper document by arXiv ID.

        Args:
            arxiv_id: arXiv identifier

        Returns:
            Paper document or None if not found
        """
        doc_key = arxiv_id.replace(".", "_").replace("/", "_")

        try:
            return self.client.get_document(self.collections["papers"], doc_key)
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                return None
            raise

    def get_code(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve code document by arXiv ID.

        Args:
            arxiv_id: arXiv identifier

        Returns:
            Code document or None if not found
        """
        doc_key = f"{arxiv_id.replace('.', '_').replace('/', '_')}_code"

        try:
            return self.client.get_document(self.collections["code"], doc_key)
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                return None
            raise

    def get_collection_stats(self) -> Dict[str, int]:
        """
        Get document counts for all collections.

        Returns:
            Dictionary mapping collection names to document counts
        """
        stats = {}

        for collection_type, collection_name in self.collections.items():
            try:
                result = self.client.request(
                    "GET",
                    f"/_db/{self.db_name}/_api/collection/{collection_name}/count"
                )
                stats[collection_name] = result.get("count", 0)
            except Exception as e:
                logger.error(f"Failed to get stats for {collection_name}: {e}")
                stats[collection_name] = -1

        return stats

    def archive_source_files(
        self,
        arxiv_id: str,
        source_files: List[Path],
        archive_config: Dict[str, str]
    ) -> Optional[Path]:
        """
        Archive source files after successful database storage.

        Moves downloaded files (PDFs, LaTeX tarballs) from cache to archive
        location to keep cache clean while preserving original sources.

        Args:
            arxiv_id: arXiv identifier
            source_files: List of file paths to archive
            archive_config: Configuration with archive_dir and archive_structure

        Returns:
            Path to archive directory or None if archival failed
        """
        import shutil

        try:
            # Get archive configuration
            archive_dir = Path(archive_config.get("archive_dir", "data/archive"))
            archive_structure = archive_config.get("archive_structure", "flat")

            # Determine archive location based on structure
            if archive_structure == "by_date":
                date_str = datetime.now().strftime("%Y-%m-%d")
                target_dir = archive_dir / date_str / arxiv_id.replace(".", "_")
            elif archive_structure == "by_family":
                family = archive_config.get("family", "default")
                target_dir = archive_dir / family / arxiv_id.replace(".", "_")
            else:  # flat
                target_dir = archive_dir / arxiv_id.replace(".", "_")

            # Create archive directory
            target_dir.mkdir(parents=True, exist_ok=True)

            # Move files to archive
            archived_files = []
            for source_file in source_files:
                if not source_file.exists():
                    logger.warning(f"Source file not found: {source_file}")
                    continue

                target_file = target_dir / source_file.name
                shutil.move(str(source_file), str(target_file))
                archived_files.append(target_file)
                logger.debug(f"Archived {source_file.name} → {target_file}")

            if archived_files:
                logger.info(
                    f"Archived {len(archived_files)} files for {arxiv_id} → {target_dir}"
                )
                return target_dir
            else:
                logger.warning(f"No files archived for {arxiv_id}")
                return None

        except Exception as e:
            logger.error(f"Failed to archive files for {arxiv_id}: {e}")
            return None

    def close(self):
        """Close database connection."""
        if self.client:
            self.client.close()
