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
                 Configure the CFExperimentStorage with an ArangoDB client and predefined collection names.
                 
                 Parameters:
                     client (Optional[ArangoHttp2Client]): An existing ArangoDB client to use; if omitted, a client will be created from `config`.
                     config (Optional[Dict[str, Any]]): Configuration dictionary used to create an ArangoDB client when `client` is not provided.
                 
                 Notes:
                     Sets `self.client`, `self.db_name`, and `self.collections` mapping for "papers", "code", and "embeddings".
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
        Create an ArangoDB HTTP/2 client configured from the provided database settings.
        
        Parameters:
            db_config (Dict[str, Any]): Configuration mapping. Recognized keys:
                - "name": database name (defaults to "cf_experiments")
                - "socket_path": Unix socket path to arangod (defaults to "/run/metis/readwrite/arangod.sock")
        
        Notes:
            This function reads ARANGO_USERNAME and ARANGO_PASSWORD from the environment (defaults: "root" and "") to populate client credentials.
        
        Returns:
            ArangoHttp2Client: A client configured with the resolved database name, socket path, base URL, credentials, and timeouts.
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
        Ensure the required ArangoDB collections for CF experiments exist in the configured database.
        
        If a collection from the instance's collection mapping is missing, it will be created. The managed collections include:
        - arxiv_markdown (paper content)
        - arxiv_code (repository code)
        - arxiv_embeddings (vector embeddings)
        
        Side effects:
        - Creates missing collections in the configured database.
        - Logs creation and existence checks.
        - May raise an exception if an unexpected error occurs while checking or creating a collection.
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
                           Store a paper's metadata, markdown content, and derived quality and processing metadata into the papers collection.
                           
                           Parameters:
                               arxiv_id (str): arXiv identifier used to derive the document key (dots and slashes replaced with underscores).
                               paper_doc (Dict[str, Any]): Paper data expected to contain keys such as `title`, `authors`, `abstract`, `markdown_content`,
                                   `latex_source`, and `processing_time`. Missing keys are treated as empty or default values.
                           
                           Returns:
                               str: The document key stored in the database (derived from `arxiv_id` by replacing '.' and '/' with '_'). If a document with the same key already exists, it will be updated.
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
                        Store embedding vectors and associated metadata for an arXiv paper as separate documents.
                        
                        Parameters:
                            arxiv_id (str): arXiv identifier used to derive document keys.
                            embeddings (List[Dict[str, Any]]): List of embedding entries where each entry may include:
                                - "text_preview" (str): text snippet for the chunk (used for preview, truncated to 200 chars),
                                - "embedding" (List[float]): numeric embedding vector,
                                - "context_tokens" (int): number of context tokens,
                                - "chunk_tokens" (int): number of tokens in the chunk,
                                - "chunk_overlap" (int): token overlap with previous chunk,
                                - "has_code" (bool): whether the chunk contains code,
                                - "paper_ratio" (float): proportion of paper content in the chunk,
                                - "code_ratio" (float): proportion of code content in the chunk.
                        
                        Returns:
                            int: Number of embedding documents inserted or updated.
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
        Retrieve the stored paper document for the given arXiv identifier.
        
        Parameters:
            arxiv_id (str): arXiv identifier used to derive the document key.
        
        Returns:
            dict | None: The paper document if found, otherwise `None`.
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
        Retrieve the stored code document for the given arXiv ID.
        
        Returns:
            dict: The code document if found, `None` otherwise.
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
        Retrieve document counts for all configured collections.
        
        Returns:
            dict: Mapping of ArangoDB collection name to its document count. If retrieving a collection's count fails, that collection's value will be -1.
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
        Archive source files for an arXiv submission into a structured archive directory.
        
        Moves the provided source files into an archive location defined by archive_config; supports "flat", "by_date", and "by_family" archive structures.
        
        Parameters:
            arxiv_id (str): arXiv identifier used to name the archive subdirectory.
            source_files (List[Path]): Paths to files to move into the archive.
            archive_config (Dict[str, str]): Archive configuration. Recognized keys:
                - archive_dir: base directory for archives (default "data/archive").
                - archive_structure: one of "flat", "by_date", or "by_family" (default "flat").
                - family: used when archive_structure == "by_family" to group archives.
        
        Returns:
            Optional[Path]: Path to the archive directory containing moved files, or `None` if no files were archived or an error occurred.
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