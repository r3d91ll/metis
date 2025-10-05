#!/usr/bin/env python3
"""Collect citation timeline data from Semantic Scholar API."""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests

from utils import RateLimiter, load_config, retry_with_backoff, save_json_incremental, setup_logging


def fetch_paper_details(arxiv_id: str, api_config: dict, logger) -> dict:
    """Fetch paper details from Semantic Scholar.

    Args:
        arxiv_id: ArXiv ID of the paper
        api_config: API configuration
        logger: Logger instance

    Returns:
        Paper details including citations
    """
    base_url = api_config["base_url"]
    url = f"{base_url}/paper/arXiv:{arxiv_id}"

    # Request paper with citations
    params = {"fields": "title,authors,citationCount,citations,citations.year,publicationDate"}

    @retry_with_backoff(logger=logger)
    def _fetch():
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    return _fetch()


def aggregate_citations_by_month(citations: list[dict], logger) -> list[dict]:
    """Aggregate citations into monthly buckets.

    Args:
        citations: List of citation objects
        logger: Logger instance

    Returns:
        List of monthly citation counts
    """
    monthly_counts = defaultdict(int)

    for citation in citations:
        if "publicationDate" in citation and citation["publicationDate"]:
            try:
                # Parse date (format: YYYY-MM-DD)
                date_str = citation["publicationDate"]
                date = datetime.fromisoformat(date_str)
                key = (date.year, date.month)
                monthly_counts[key] += 1
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not parse date: {citation.get('publicationDate')}: {e}")
                continue

    # Convert to sorted list
    result = []
    for (year, month), count in sorted(monthly_counts.items()):
        result.append({"year": year, "month": month, "count": count})

    return result


def collect_citations_for_paper(
    paper_config: dict, api_config: dict, output_dir: Path, logger
) -> dict:
    """Collect citation data for a single paper.

    Args:
        paper_config: Paper configuration
        api_config: API configuration
        output_dir: Output directory
        logger: Logger instance

    Returns:
        Citation data
    """
    arxiv_id = paper_config["arxiv_id"]
    title = paper_config["title"]

    logger.info(f"Fetching citation data for {title} ({arxiv_id})")

    # Fetch paper details
    paper_data = fetch_paper_details(arxiv_id, api_config, logger)

    # Get citations
    citations = paper_data.get("citations", [])
    logger.info(f"Found {len(citations)} citations")

    # Aggregate by month
    monthly_citations = aggregate_citations_by_month(citations, logger)

    # Create output structure
    result = {
        "paper_id": arxiv_id,
        "title": title,
        "authors": [author.get("name", "") for author in paper_data.get("authors", [])],
        "publication_date": paper_data.get("publicationDate"),
        "total_citations": len(citations),
        "monthly_citations": monthly_citations,
        "collected_at": datetime.now().isoformat(),
    }

    return result


def main():
    """Main entry point."""
    # Setup
    logger = setup_logging()
    config = load_config()

    # Create rate limiter
    rate_limiter = RateLimiter(config["api"]["semantic_scholar"]["requests_per_second"])

    # Output directory
    output_dir = Path("data/case_study/citations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect data for each paper
    for paper_name, paper_config in config["papers"].items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {paper_name}")
        logger.info(f"{'='*60}")

        rate_limiter.wait()

        try:
            citation_data = collect_citations_for_paper(
                paper_config, config["api"]["semantic_scholar"], output_dir, logger
            )

            # Save to file
            output_file = output_dir / f"{paper_name}_citations.json"
            save_json_incremental(citation_data, output_file, logger)

            logger.info(f"Successfully collected {citation_data['total_citations']} citations")
            monthly_citations = citation_data.get('monthly_citations', [])
            if monthly_citations and len(monthly_citations) > 0:
                logger.info(
                    f"Date range: {monthly_citations[0]['year']}-"
                    f"{monthly_citations[0]['month']} to "
                    f"{monthly_citations[-1]['year']}-"
                    f"{monthly_citations[-1]['month']}"
                )
            else:
                logger.info("Date range: no valid citation dates available")

        except Exception as e:
            logger.error(f"Failed to collect citations for {paper_name}: {e}")
            raise

    logger.info("\n" + "=" * 60)
    logger.info("Citation collection complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
