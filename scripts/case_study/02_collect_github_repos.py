#!/usr/bin/env python3
"""Collect GitHub repository data for paper implementations."""

import os
from datetime import datetime
from pathlib import Path

from github import Github, RateLimitExceededException

from utils import load_config, save_json_incremental, setup_logging


def classify_repository(repo, paper_config: dict, logger) -> str:
    """Classify repository type.

    Args:
        repo: GitHub repository object
        paper_config: Paper configuration
        logger: Logger instance

    Returns:
        Repository type classification
    """
    # Check if from official code
    official_code = paper_config.get("official_code")
    if official_code and repo.html_url == official_code:
        return "official"

    # Check if from paper authors (basic heuristic)
    if paper_config.get("authors"):
        repo_owner = repo.owner.login.lower()
        for author in paper_config["authors"]:
            if author.lower() in repo_owner:
                return "official"

    # Check for framework integration
    frameworks = ["tensorflow", "pytorch", "keras", "huggingface", "jax", "mxnet"]
    if any(fw in repo.full_name.lower() for fw in frameworks):
        return "framework"

    # Check description for classification hints
    desc = (repo.description or "").lower()
    if any(word in desc for word in ["tutorial", "implementation", "example", "guide"]):
        if any(word in desc for word in ["simple", "basic", "minimal", "clean"]):
            return "tutorial"

    # Check for research extensions
    if any(word in desc for word in ["research", "extension", "improved", "novel"]):
        return "research"

    # Check for applications
    if any(
        word in desc for word in ["application", "project", "applied", "system", "production"]
    ):
        return "application"

    # Default to tutorial if has good documentation
    if repo.has_wiki or (repo.description and len(repo.description) > 100):
        return "tutorial"

    return "application"


def search_repositories_for_paper(
    paper_name: str, paper_config: dict, search_queries: list[str], collection_config: dict, gh, logger
) -> list[dict]:
    """Search GitHub for repositories related to a paper.

    Args:
        paper_name: Name of the paper
        paper_config: Paper configuration
        search_queries: List of search queries
        collection_config: Collection configuration
        gh: GitHub API client
        logger: Logger instance

    Returns:
        List of repository data dictionaries
    """
    repositories = []
    seen_urls = set()

    min_stars = collection_config["min_stars"]
    min_forks = collection_config["min_forks"]
    max_results = collection_config["max_results_per_query"]

    publication_date = datetime.fromisoformat(paper_config["published_date"])

    logger.info(f"Searching for {paper_name} implementations")
    logger.info(f"Filters: min_stars={min_stars}, min_forks={min_forks}")

    for query in search_queries:
        logger.info(f"  Query: '{query}'")

        # Build search query with filters
        search_string = (
            f"{query} language:python stars:>={min_stars} "
            f"created:>={paper_config['published_date']}"
        )

        try:
            results = gh.search_repositories(query=search_string, sort="stars", order="desc")

            count = 0
            for repo in results:
                if count >= max_results:
                    break

                # Skip if already seen
                if repo.html_url in seen_urls:
                    continue

                # Skip forks with low activity
                if repo.fork and repo.stargazers_count < min_stars * 2:
                    continue

                # Check minimum forks
                if repo.forks_count < min_forks:
                    continue

                # Skip if created before paper publication
                if repo.created_at < publication_date:
                    logger.debug(f"    Skipping {repo.full_name} (created before paper)")
                    continue

                # Classify repository
                repo_type = classify_repository(repo, paper_config, logger)

                # Check if from authors
                from_authors = repo_type == "official"

                repo_data = {
                    "url": repo.html_url,
                    "full_name": repo.full_name,
                    "created_at": repo.created_at.isoformat(),
                    "stars": repo.stargazers_count,
                    "forks": repo.forks_count,
                    "language": repo.language or "Unknown",
                    "type": repo_type,
                    "description": repo.description or "",
                    "from_authors": from_authors,
                    "has_wiki": repo.has_wiki,
                    "open_issues": repo.open_issues_count,
                    "topics": repo.get_topics(),
                    "last_updated": repo.updated_at.isoformat(),
                }

                repositories.append(repo_data)
                seen_urls.add(repo.html_url)
                count += 1

                logger.info(
                    f"    Found: {repo.full_name} ({repo.stargazers_count} stars, "
                    f"type: {repo_type})"
                )

        except RateLimitExceededException:
            logger.warning("Rate limit exceeded. Waiting...")
            rate_limit = gh.get_rate_limit()
            reset_time = rate_limit.search.reset
            wait_time = (reset_time - datetime.now()).total_seconds() + 10
            logger.info(f"Waiting {wait_time:.0f} seconds for rate limit reset...")
            import time

            time.sleep(wait_time)

        except Exception as e:
            logger.error(f"Error searching for query '{query}': {e}")
            continue

    logger.info(f"Found {len(repositories)} unique repositories for {paper_name}")
    return repositories


def main():
    """Main entry point."""
    # Setup
    logger = setup_logging()
    config = load_config()

    # Initialize GitHub client
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        gh = Github(github_token)
        logger.info("Using authenticated GitHub API")
    else:
        gh = Github()
        logger.warning("No GITHUB_TOKEN found. Using unauthenticated API (low rate limit)")

    # Check rate limit
    rate_limit = gh.get_rate_limit()
    logger.info(f"GitHub API rate limit: {rate_limit.search.remaining}/{rate_limit.search.limit}")

    # Output directory
    output_dir = Path("data/case_study/implementations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect data for each paper
    for paper_name, paper_config in config["papers"].items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {paper_name}")
        logger.info(f"{'='*60}")

        try:
            # Get search queries for this paper
            search_queries = config["collection"]["github"]["search_queries"][paper_name]

            # Search repositories
            repositories = search_repositories_for_paper(
                paper_name,
                paper_config,
                search_queries,
                config["collection"]["github"],
                gh,
                logger,
            )

            # Create output structure
            result = {
                "paper": paper_name,
                "arxiv_id": paper_config["arxiv_id"],
                "title": paper_config["title"],
                "repositories": repositories,
                "total_repositories": len(repositories),
                "collected_at": datetime.now().isoformat(),
                "type_counts": {},
            }

            # Count by type
            for repo in repositories:
                repo_type = repo["type"]
                result["type_counts"][repo_type] = result["type_counts"].get(repo_type, 0) + 1

            # Save to file
            output_file = output_dir / f"{paper_name}_repos.json"
            save_json_incremental(result, output_file, logger)

            logger.info(f"Successfully collected {len(repositories)} repositories")
            logger.info(f"Type distribution: {result['type_counts']}")

        except Exception as e:
            logger.error(f"Failed to collect repositories for {paper_name}: {e}")
            raise

    logger.info("\n" + "=" * 60)
    logger.info("Repository collection complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
