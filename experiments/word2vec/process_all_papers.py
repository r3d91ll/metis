"""
Process All Word2Vec Family Papers

Batch processes all 5 papers in the Word2Vec family:
- Downloads papers (PDF + LaTeX) from arXiv
- Fetches code repositories from GitHub
- Stores in ArangoDB
- Archives source files to /bulk-store/metis
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Add metis to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.word2vec.arxiv_fetcher import ArxivPaperFetcher
from experiments.word2vec.github_fetcher import GitHubCodeFetcher
from experiments.word2vec.storage import CFExperimentStorage

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Word2VecFamilyProcessor:
    """Batch processor for Word2Vec family papers."""

    def __init__(self, config_path: Path):
        """
        Create a Word2VecFamilyProcessor configured from a YAML file.
        
        Reads configuration from the given path, initializes the Arxiv paper fetcher,
        GitHub code fetcher, and experiment storage, and ensures required storage
        collections exist.
        
        Parameters:
            config_path (Path): Path to the YAML configuration file used to initialize
                fetchers and storage.
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.paper_fetcher = ArxivPaperFetcher(
            cache_dir=self.config["infrastructure"]["cache_dir"],
            max_retries=self.config["fetching"]["arxiv"]["max_retries"],
        )
        self.code_fetcher = GitHubCodeFetcher(self.config)
        self.storage = CFExperimentStorage(config=self.config)
        self.storage.ensure_collections()

    def process_paper(
        self,
        arxiv_id: str,
        title: str,
        authors: List[str],
        expected_repo: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Process a single paper through fetch, storage, and archival steps.
        
        Performs three ordered steps: fetch the paper (PDF/LaTeX/Markdown), attempt to locate and fetch associated GitHub code (optionally using a provided repo), and archive available source files. Records per-step status, timings, and any errors in the returned results dictionary.
        
        Parameters:
            expected_repo (Optional[str]): If provided, treat this GitHub URL as the official repository and try it first; otherwise the code fetcher will attempt to discover an official repo.
        
        Returns:
            dict: A results dictionary with at least the keys:
                - `arxiv_id` (str): The arXiv identifier processed.
                - `title` (str): Paper title.
                - `paper_fetched` (bool): `true` if the paper was fetched and stored, `false` otherwise.
                - `code_fetched` (bool): `true` if code was fetched and stored, `false` otherwise.
                - `archived` (bool): `true` if source files were archived, `false` otherwise.
                - `errors` (List[str]): List of error messages encountered during processing.
                - `paper_time` (float): Time in seconds spent fetching the paper (present if `paper_fetched`).
                - `code_time` (float): Time in seconds spent fetching code (present if `code_fetched`).
                - `code_files` (int): Number of code files fetched (present if `code_fetched`).
                - `code_lines` (int): Total lines of code fetched (present if `code_fetched`).
                - `archive_path` (str): Path to the archived files (present if `archived`).
                - `total_time` (float): Total time in seconds for the whole processing run.
        """
        logger.info("=" * 80)
        logger.info(f"Processing: {title}")
        logger.info(f"arXiv ID: {arxiv_id}")
        logger.info("=" * 80)

        results = {
            "arxiv_id": arxiv_id,
            "title": title,
            "paper_fetched": False,
            "code_fetched": False,
            "archived": False,
            "errors": []
        }

        start_time = time.time()

        # Step 1: Fetch paper (PDF + LaTeX + Markdown)
        logger.info("\nStep 1: Fetching paper from arXiv...")
        try:
            paper_doc = self.paper_fetcher.fetch_paper(arxiv_id, fetch_latex=True)
            results["paper_fetched"] = True
            results["paper_time"] = time.time() - start_time

            logger.info("✓ Paper fetched")
            logger.info(f"  Markdown: {len(paper_doc.markdown_content)} chars")
            if paper_doc.latex_source:
                logger.info(f"  LaTeX: {len(paper_doc.latex_source)} chars")
            else:
                logger.info("  LaTeX: Not available")

            # Store paper
            paper_dict = {
                "title": paper_doc.title,
                "authors": paper_doc.authors,
                "abstract": paper_doc.abstract,
                "markdown_content": paper_doc.markdown_content,
                "latex_source": paper_doc.latex_source,
                "processing_time": paper_doc.processing_time,
            }
            paper_key = self.storage.store_paper_markdown(arxiv_id, paper_dict)
            logger.info(f"  Stored: {paper_key}")

        except Exception as e:
            logger.error(f"✗ Failed to fetch paper: {e}")
            results["errors"].append(f"Paper fetch: {e}")
            return results

        # Step 2: Fetch code from GitHub
        logger.info("\nStep 2: Fetching code from GitHub...")
        code_doc = None
        code_start = time.time()

        try:
            # Try expected repo first
            if expected_repo:
                logger.info(f"  Trying known repo: {expected_repo}")
                code_doc = self.code_fetcher.fetch_code(expected_repo, is_official=True)
            else:
                # Search for repository
                logger.info("  Searching for repository...")
                repo_url = self.code_fetcher.find_official_repo(title, authors, arxiv_id)
                if repo_url:
                    logger.info(f"  Found: {repo_url}")
                    code_doc = self.code_fetcher.fetch_code(repo_url, is_official=True)

            if code_doc:
                results["code_fetched"] = True
                results["code_time"] = time.time() - code_start
                results["code_files"] = len(code_doc.code_files)
                results["code_lines"] = code_doc.total_lines

                logger.info("✓ Code fetched")
                logger.info(f"  Files: {len(code_doc.code_files)}")
                logger.info(f"  Lines: {code_doc.total_lines}")
                logger.info(f"  Language: {code_doc.metadata.get('language', 'Unknown')}")

                # Store code
                code_doc_dict = {
                    "github_url": code_doc.github_url,
                    "is_official": code_doc.is_official,
                    "code_files": code_doc.code_files,
                    "metadata": code_doc.metadata,
                    "total_lines": code_doc.total_lines,
                }
                code_key = self.storage.store_code(arxiv_id, code_doc_dict)
                logger.info(f"  Stored: {code_key}")
            else:
                logger.warning("⚠ No code repository found")
                results["errors"].append("No code repository found")

        except Exception as e:
            logger.error(f"✗ Failed to fetch code: {e}")
            results["errors"].append(f"Code fetch: {e}")

        # Step 3: Archive source files
        logger.info("\nStep 3: Archiving source files...")
        try:
            source_files = []
            if paper_doc.pdf_path.exists():
                source_files.append(paper_doc.pdf_path)

            latex_path = paper_doc.pdf_path.parent / f"{arxiv_id.replace('.', '_')}_source.tar.gz"
            if latex_path.exists():
                source_files.append(latex_path)

            if source_files:
                archive_dir = self.storage.archive_source_files(
                    arxiv_id,
                    source_files,
                    self.config.get("infrastructure", {})
                )
                if archive_dir:
                    results["archived"] = True
                    results["archive_path"] = str(archive_dir)
                    logger.info(f"✓ Archived {len(source_files)} files to {archive_dir}")
                else:
                    logger.warning("⚠ Failed to archive files")
                    results["errors"].append("Archive failed")
            else:
                logger.info("  No files to archive")

        except Exception as e:
            logger.error(f"✗ Failed to archive: {e}")
            results["errors"].append(f"Archive: {e}")

        results["total_time"] = time.time() - start_time
        logger.info(f"\n✓ Completed in {results['total_time']:.2f}s")

        return results

    def process_all(self) -> List[Dict[str, any]]:
        """
        Process every paper listed in the processor's configuration.
        
        Iterates over the configured papers, calls process_paper for each entry, and pauses briefly between papers to respect rate limits.
        
        Returns:
            results (List[Dict[str, any]]): A list of per-paper result dictionaries containing processing metadata (e.g., fetch and archive statuses, timings, and errors).
        """
        papers = self.config.get("papers", [])
        results = []

        logger.info("\n" + "=" * 80)
        logger.info(f"Processing {len(papers)} papers in Word2Vec family")
        logger.info("=" * 80 + "\n")

        for i, paper_config in enumerate(papers, 1):
            logger.info(f"\n[{i}/{len(papers)}] Starting paper: {paper_config['arxiv_id']}")

            result = self.process_paper(
                arxiv_id=paper_config["arxiv_id"],
                title=paper_config["title"],
                authors=paper_config.get("authors", []),
                expected_repo=paper_config.get("expected_repo")
            )
            results.append(result)

            # Brief pause between papers to respect rate limits
            if i < len(papers):
                logger.info("\nPausing 5s before next paper...")
                time.sleep(5)

        return results

    def print_summary(self, results: List[Dict[str, any]]):
        """Print summary of processing results."""
        logger.info("\n" + "=" * 80)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 80)

        total_papers = len(results)
        papers_fetched = sum(1 for r in results if r["paper_fetched"])
        code_fetched = sum(1 for r in results if r["code_fetched"])
        archived = sum(1 for r in results if r["archived"])

        logger.info(f"\nPapers processed: {papers_fetched}/{total_papers}")
        logger.info(f"Code repositories: {code_fetched}/{total_papers}")
        logger.info(f"Files archived: {archived}/{total_papers}")

        logger.info("\nDetails:")
        logger.info("-" * 80)
        for r in results:
            status = "✓" if r["paper_fetched"] else "✗"
            code_status = "✓" if r["code_fetched"] else "✗"
            logger.info(f"{status} {r['arxiv_id']}: {r['title'][:50]}...")
            total_time = r.get('total_time', 0)
            logger.info(f"   Paper: {status}  Code: {code_status}  Time: {total_time:.1f}s")
            if r.get("code_files"):
                logger.info(f"   Code files: {r['code_files']}, Lines: {r['code_lines']}")
            if r.get("errors"):
                for error in r["errors"]:
                    logger.info(f"   ⚠ {error}")

        # Collection stats
        logger.info("\nDatabase Statistics:")
        logger.info("-" * 80)
        stats = self.storage.get_collection_stats()
        for collection_name, count in stats.items():
            logger.info(f"  {collection_name}: {count:,} documents")

        logger.info("\n" + "=" * 80)

    def close(self):
        """
        Close and release external resources used by the processor.
        
        Specifically closes the storage backend, the arXiv paper fetcher, and the GitHub code fetcher.
        """
        self.storage.close()
        self.paper_fetcher.close()
        self.code_fetcher.close()


def main():
    """
    Execute batch processing for the configured Word2Vec papers.
    
    Initializes a Word2VecFamilyProcessor using the module-local config.yaml, runs processing for all configured papers, prints a processing summary, and ensures processor resources are closed.
    
    Returns:
        exit_code (int): 0 if every processed paper has `paper_fetched` set to `True`, 1 otherwise.
    """
    config_path = Path(__file__).parent / "config.yaml"
    processor = Word2VecFamilyProcessor(config_path)

    try:
        results = processor.process_all()
        processor.print_summary(results)

        # Return exit code based on success
        all_success = all(r["paper_fetched"] for r in results)
        return 0 if all_success else 1

    finally:
        processor.close()


if __name__ == "__main__":
    sys.exit(main())