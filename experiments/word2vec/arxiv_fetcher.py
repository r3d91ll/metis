"""
ArXiv Paper Fetcher

Downloads papers from arXiv and converts them to markdown using Docling.
"""

import logging

# Add metis to path if needed
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import arxiv

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from metis.extractors import DoclingExtractor

logger = logging.getLogger(__name__)


@dataclass
class PaperDocument:
    """Represents a downloaded and processed paper."""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    markdown_content: str
    latex_source: Optional[str]
    pdf_path: Path
    metadata: Dict[str, Any]
    processing_time: float


class ArxivPaperFetcher:
    """Downloads arXiv papers and converts to markdown."""

    def __init__(self,
                 extractor: Optional[DoclingExtractor] = None,
                 cache_dir: Optional[Path] = None,
                 max_retries: int = 3):
        """
                 Create an ArxivPaperFetcher configured with a Docling extractor, cache directory, and retry policy.
                 
                 If `extractor` is omitted, attempts to instantiate a DoclingExtractor and falls back to a minimal PyMuPDF-based extractor if Docling is unavailable or its API differs. Ensures `cache_dir` is a Path (defaults to /tmp/arxiv_cache) and creates the directory if it does not exist. Initializes the arXiv client and stores the maximum number of download retries.
                 
                 Parameters:
                     extractor (Optional[DoclingExtractor]): Preconfigured Docling extractor to use; if None, a default extractor will be created or a fallback configured when Docling cannot be initialized.
                     cache_dir (Optional[Path | str]): Path to the directory used for caching downloaded PDFs and sources; if None, defaults to /tmp/arxiv_cache. String paths will be converted to Path.
                     max_retries (int): Maximum number of attempts to retry network downloads before failing.
                 """
        # Initialize extractor - will use PyMuPDF fallback if Docling has API issues
        if extractor is None:
            try:
                self.extractor = DoclingExtractor()
            except (TypeError, ImportError) as e:
                # Docling API changed or not available, use fallback
                logger.warning(f"Docling initialization failed ({e}), using PyMuPDF fallback")
                # Create a minimal extractor that will use fallback
                self.extractor = DoclingExtractor.__new__(DoclingExtractor)
                self.extractor.converter = None
                self.extractor.use_fallback = True
        else:
            self.extractor = extractor

        # Convert cache_dir to Path if it's a string
        if cache_dir is None:
            self.cache_dir = Path("/tmp/arxiv_cache")
        elif isinstance(cache_dir, str):
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = cache_dir

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.client = arxiv.Client()

    def fetch_paper(self, arxiv_id: str, fetch_latex: bool = True) -> PaperDocument:
        """
        Fetches an arXiv paper, converts its PDF to markdown, and returns a PaperDocument.
        
        Caches downloaded PDF and source tarball in the fetcher's cache_dir and, when requested,
        attempts to download and concatenate the paper's LaTeX source files.
        
        Parameters:
            fetch_latex (bool): If True, attempt to download and extract the paper's LaTeX source; if False, skip LaTeX retrieval.
        
        Returns:
            PaperDocument: Contains arXiv metadata, extracted markdown content, optional concatenated LaTeX source, local PDF path, and processing metadata.
        """
        start_time = datetime.now()

        # Check cache first
        pdf_path = self.cache_dir / f"{arxiv_id.replace('.', '_')}.pdf"
        latex_path = self.cache_dir / f"{arxiv_id.replace('.', '_')}_source.tar.gz"

        if not pdf_path.exists():
            logger.info(f"Downloading paper {arxiv_id} from arXiv...")
            paper_metadata = self._download_pdf(arxiv_id, pdf_path)
        else:
            logger.info(f"Using cached PDF for {arxiv_id}")
            paper_metadata = self._get_metadata(arxiv_id)

        # Download LaTeX source if requested
        latex_content = None
        if fetch_latex:
            if not latex_path.exists():
                logger.info(f"Downloading LaTeX source for {arxiv_id}...")
                latex_content = self._download_latex_source(arxiv_id, latex_path)
            else:
                logger.info(f"Using cached LaTeX source for {arxiv_id}")
                latex_content = self._extract_latex_from_cache(latex_path)

        # Convert PDF to markdown
        logger.info(f"Converting PDF to markdown for {arxiv_id}...")
        markdown_result = self._extract_markdown(pdf_path)

        processing_time = (datetime.now() - start_time).total_seconds()

        return PaperDocument(
            arxiv_id=arxiv_id,
            title=paper_metadata["title"],
            authors=paper_metadata["authors"],
            abstract=paper_metadata["abstract"],
            markdown_content=markdown_result["markdown"],
            latex_source=latex_content,
            pdf_path=pdf_path,
            metadata={
                **paper_metadata,
                "extraction_metadata": markdown_result.get("metadata", {}),
                "word_count": len(markdown_result["markdown"].split()),
                "has_latex_source": latex_content is not None,
                "processing_seconds": processing_time
            },
            processing_time=processing_time
        )

    def _download_pdf(self, arxiv_id: str, pdf_path: Path) -> Dict[str, Any]:
        """
        Download the PDF for the given arXiv identifier, save it to pdf_path, and return extracted paper metadata.
        
        Parameters:
            arxiv_id (str): The arXiv identifier of the paper to download.
            pdf_path (Path): Filesystem path where the PDF will be written.
        
        Returns:
            dict: Metadata for the paper containing keys:
                - "title" (str)
                - "authors" (List[str])
                - "abstract" (str)
                - "published" (str, ISO 8601)
                - "updated" (Optional[str], ISO 8601 or None)
                - "categories" (List[str])
                - "primary_category" (str)
                - "pdf_url" (str)
                - "entry_id" (str)
        
        Raises:
            RuntimeError: If the PDF cannot be downloaded, the downloaded file is missing or empty,
                          or all retry attempts (controlled by the instance's max_retries) fail.
        """
        for attempt in range(self.max_retries):
            try:
                # Search for the paper
                search = arxiv.Search(id_list=[arxiv_id])
                paper = next(self.client.results(search))

                # Extract metadata
                metadata = {
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "abstract": paper.summary,
                    "published": paper.published.isoformat(),
                    "updated": paper.updated.isoformat() if paper.updated else None,
                    "categories": paper.categories,
                    "primary_category": paper.primary_category,
                    "pdf_url": paper.pdf_url,
                    "entry_id": paper.entry_id
                }

                # Download PDF
                paper.download_pdf(filename=str(pdf_path))

                # Verify file exists and has content
                if not pdf_path.exists():
                    raise RuntimeError("PDF download failed - file not created")

                if pdf_path.stat().st_size == 0:
                    pdf_path.unlink()  # Remove empty file
                    raise RuntimeError("PDF download failed - empty file")

                logger.info(f"Downloaded {pdf_path.stat().st_size / 1024:.1f} KB PDF")
                return metadata

            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise RuntimeError(
                        f"Failed to download {arxiv_id} after {self.max_retries} attempts: {e}"
                    )

    def _get_metadata(self, arxiv_id: str) -> Dict[str, Any]:
        """
        Retrieve metadata for an arXiv paper without downloading its PDF.
        
        Returns:
            A dictionary with the paper metadata. Expected keys:
            - `title` (str)
            - `authors` (List[str])
            - `abstract` (str)
            - `published` (ISO 8601 str or None)
            - `updated` (ISO 8601 str or None)
            - `categories` (List[str])
            - `primary_category` (str or None)
            - `pdf_url` (str or None)
            - `entry_id` (str or None)
        
            On failure, returns a fallback dictionary containing a generic `title`, empty `authors`
            and `abstract`, `published` set to None, and an empty `categories` list.
        """
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(self.client.results(search))

            return {
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "abstract": paper.summary,
                "published": paper.published.isoformat(),
                "updated": paper.updated.isoformat() if paper.updated else None,
                "categories": paper.categories,
                "primary_category": paper.primary_category,
                "pdf_url": paper.pdf_url,
                "entry_id": paper.entry_id
            }
        except Exception as e:
            logger.error(f"Failed to get metadata for {arxiv_id}: {e}")
            return {
                "title": f"Paper {arxiv_id}",
                "authors": [],
                "abstract": "",
                "published": None,
                "categories": []
            }

    def _extract_markdown(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract markdown content and associated metadata from a PDF using the configured extractor.
        
        Parameters:
            pdf_path (Path): Path to the PDF file to process.
        
        Returns:
            result (Dict[str, Any]): Dictionary with keys:
                - "markdown" (str): Extracted markdown/text content.
                - "metadata" (Dict[str, Any]): Metadata returned by the extractor.
        
        Raises:
            RuntimeError: If extraction fails or the extracted content is shorter than 100 characters.
        """
        try:
            result = self.extractor.extract(pdf_path)

            # Handle different result formats from DoclingExtractor
            if isinstance(result, dict):
                markdown = (
                    result.get("markdown") or result.get("full_text") or result.get("text", "")
                )
                metadata = result.get("metadata", {})
            else:
                # Handle ExtractionResult dataclass
                markdown = getattr(result, "text", "")
                metadata = getattr(result, "metadata", {})

            if not markdown or len(markdown) < 100:
                raise RuntimeError(
                    f"Extraction produced insufficient content ({len(markdown)} chars)"
                )

            logger.info(f"Extracted {len(markdown.split())} words from PDF")

            return {
                "markdown": markdown,
                "metadata": metadata
            }

        except Exception as e:
            raise RuntimeError(f"Failed to extract markdown from {pdf_path}: {e}")

    def _download_latex_source(self, arxiv_id: str, latex_path: Path) -> Optional[str]:
        """
        Download the arXiv LaTeX source tarball for an identifier and return the concatenated contents of its `.tex` files.
        
        Parameters:
            arxiv_id (str): The arXiv identifier of the paper.
            latex_path (Path): File path (including filename) where the downloaded source tarball will be saved in the cache.
        
        Returns:
            str or None: Concatenated contents of all `.tex` files extracted from the tarball, or `None` if the source is unavailable or extraction fails.
        """
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            result = next(self.client.results(search))

            # Download source to cache
            result.download_source(dirpath=str(latex_path.parent), filename=latex_path.name)

            # Extract and concatenate LaTeX files
            return self._extract_latex_from_cache(latex_path)

        except Exception as e:
            logger.warning(f"Could not download LaTeX source for {arxiv_id}: {e}")
            return None

    def _extract_latex_from_cache(self, latex_path: Path) -> Optional[str]:
        """
        Extract and concatenate all `.tex` files from a cached LaTeX tarball.
        
        Parameters:
            latex_path (Path): Path to the gzipped LaTeX source tarball.
        
        Returns:
            Optional[str]: Combined LaTeX content as a single string if one or more `.tex` files are found and readable; `None` if extraction fails, no `.tex` files are present, or files cannot be read.
        """
        import tarfile
        import tempfile

        try:
            # Extract tarball to temp directory
            with tempfile.TemporaryDirectory() as tmpdir:
                with tarfile.open(latex_path, 'r:gz') as tar:
                    tar.extractall(tmpdir)

                # Find and read all .tex files
                tmpdir_path = Path(tmpdir)
                tex_files = list(tmpdir_path.rglob("*.tex"))

                if not tex_files:
                    logger.warning(f"No .tex files found in {latex_path}")
                    return None

                # Concatenate all .tex files
                latex_content = []
                for tex_file in sorted(tex_files):
                    try:
                        with open(tex_file, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            latex_content.append(f"% File: {tex_file.name}\n{content}")
                    except Exception as e:
                        logger.warning(f"Could not read {tex_file}: {e}")

                if not latex_content:
                    return None

                combined = "\n\n".join(latex_content)
                logger.info(
                    f"Extracted {len(tex_files)} LaTeX files "
                    f"({len(combined)} chars, {len(combined.split())} words)"
                )
                return combined

        except Exception as e:
            logger.warning(f"Could not extract LaTeX from {latex_path}: {e}")
            return None

    def close(self):
        """Close the fetcher (cleanup if needed)."""
        # No cleanup needed for ArxivPaperFetcher
        pass


# Convenience function
def fetch_arxiv_paper(arxiv_id: str, cache_dir: Optional[Path] = None) -> PaperDocument:
    """
    Convenience function to fetch a single paper.

    Args:
        arxiv_id: arXiv identifier
        cache_dir: Optional cache directory

    Returns:
        PaperDocument with processed content
    """
    fetcher = ArxivPaperFetcher(cache_dir=cache_dir)
    return fetcher.fetch_paper(arxiv_id)