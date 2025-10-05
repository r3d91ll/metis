#!/usr/bin/env python3
"""Download and extract paper content from ArXiv."""

import json
import re
from pathlib import Path

import arxiv

from utils import load_config, retry_with_backoff, setup_logging


def download_paper(arxiv_id: str, output_dir: Path, logger) -> Path:
    """Download paper PDF from ArXiv.

    Args:
        arxiv_id: ArXiv ID
        output_dir: Directory to save PDF
        logger: Logger instance

    Returns:
        Path to downloaded PDF
    """
    logger.info(f"Downloading paper {arxiv_id} from ArXiv")

    @retry_with_backoff(logger=logger)
    def _download():
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(client.results(search))

        # Download PDF
        pdf_path = output_dir / f"{arxiv_id}.pdf"
        paper.download_pdf(filename=str(pdf_path))
        return pdf_path

    return _download()


def extract_paper_sections(pdf_path: Path, logger) -> dict[str, str]:
    """Extract text sections from PDF.

    Args:
        pdf_path: Path to PDF file
        logger: Logger instance

    Returns:
        Dictionary mapping section names to text content
    """
    try:
        from metis.extractors.pdf import PDFExtractor

        logger.info(f"Extracting sections from {pdf_path.name}")

        extractor = PDFExtractor()
        document = extractor.extract(str(pdf_path))

        sections = {}

        # Extract full text
        full_text = ""
        for page in document.pages:
            full_text += page.text + "\n\n"

        # Identify sections using common patterns
        section_patterns = {
            "abstract": r"(?i)abstract\s*\n(.*?)(?=\n\d+\s+introduction|\nintroduction)",
            "introduction": r"(?i)(?:\d+\s+)?introduction\s*\n(.*?)(?=\n\d+\s+|\n[A-Z])",
            "related_work": r"(?i)(?:\d+\s+)?related\s+work\s*\n(.*?)(?=\n\d+\s+|\n[A-Z])",
            "method": r"(?i)(?:\d+\s+)?(?:method|approach|model)\s*\n(.*?)(?=\n\d+\s+|\n[A-Z])",
            "experiments": r"(?i)(?:\d+\s+)?(?:experiment|result)s?\s*\n(.*?)(?=\n\d+\s+|\n[A-Z])",
            "conclusion": r"(?i)(?:\d+\s+)?conclusion\s*\n(.*?)(?=\nreferences|\nacknowledgment)",
        }

        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, full_text, re.DOTALL | re.MULTILINE)
            if match:
                section_text = match.group(1).strip()
                # Clean up extra whitespace
                section_text = re.sub(r"\s+", " ", section_text)
                sections[section_name] = section_text
                logger.info(f"  Extracted {section_name}: {len(section_text)} chars")

        # If section extraction failed, use full text as fallback
        if not sections:
            logger.warning("Could not identify sections, using full text")
            sections["full_text"] = full_text

        return sections

    except ImportError:
        logger.exception("PDFExtractor not available. Install with: poetry install -E pdf")
        raise
    except Exception as e:
        logger.exception("Error extracting sections")
        raise


def extract_metadata_from_arxiv(arxiv_id: str, logger) -> dict:
    """Get paper metadata from ArXiv API.

    Args:
        arxiv_id: ArXiv ID
        logger: Logger instance

    Returns:
        Paper metadata
    """
    logger.info(f"Fetching metadata for {arxiv_id}")

    @retry_with_backoff(logger=logger)
    def _fetch():
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(client.results(search))

        return {
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "abstract": paper.summary,
            "published": paper.published.isoformat(),
            "updated": paper.updated.isoformat(),
            "primary_category": paper.primary_category,
            "categories": paper.categories,
            "pdf_url": paper.pdf_url,
            "arxiv_url": paper.entry_id,
        }

    return _fetch()


def process_paper(paper_name: str, paper_config: dict, papers_dir: Path, logger) -> dict:
    """Download and extract a single paper.

    Args:
        paper_name: Name of the paper
        paper_config: Paper configuration
        papers_dir: Directory for paper files
        logger: Logger instance

    Returns:
        Extracted paper data
    """
    arxiv_id = paper_config["arxiv_id"]

    logger.info(f"\nProcessing {paper_config['title']}")

    # Download PDF
    pdf_path = download_paper(arxiv_id, papers_dir, logger)
    logger.info(f"Downloaded to {pdf_path}")

    # Get metadata
    metadata = extract_metadata_from_arxiv(arxiv_id, logger)

    # Extract sections
    sections = extract_paper_sections(pdf_path, logger)

    # Combine into output structure
    result = {
        "paper_name": paper_name,
        "arxiv_id": arxiv_id,
        "metadata": metadata,
        "sections": {name: {"text": text} for name, text in sections.items()},
        "pdf_path": str(pdf_path),
    }

    return result


def main():
    """Main entry point."""
    # Setup
    logger = setup_logging()
    config = load_config()

    # Output directories
    papers_dir = Path("data/case_study/papers")
    papers_dir.mkdir(parents=True, exist_ok=True)

    extracted_dir = papers_dir / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    # Process each paper
    all_papers = {}

    for paper_name, paper_config in config["papers"].items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {paper_name}")
        logger.info(f"{'='*60}")

        try:
            paper_data = process_paper(paper_name, paper_config, papers_dir, logger)
            all_papers[paper_name] = paper_data

            # Save individual paper extraction
            output_file = extracted_dir / f"{paper_name}_extracted.json"
            with open(output_file, "w") as f:
                json.dump(paper_data, f, indent=2)

            logger.info(f"Saved extraction to {output_file}")
            logger.info(f"Extracted {len(paper_data['sections'])} sections")

        except Exception as e:
            logger.error(f"Failed to process {paper_name}: {e}")
            raise

    # Save combined output
    output_file = extracted_dir / "all_papers.json"
    with open(output_file, "w") as f:
        json.dump(all_papers, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("Paper extraction complete!")
    logger.info(f"Processed {len(all_papers)} papers")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
