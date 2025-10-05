#!/usr/bin/env python3
"""Collect boundary objects (documentation, code, tutorials) for papers."""

import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import requests
from waybackpy import WaybackMachineCDXServerAPI

from utils import RateLimiter, load_config, retry_with_backoff, setup_logging


def clone_repository_at_commit(
    repo_url: str, target_date: str, output_dir: Path, logger
) -> Path | None:
    """Clone a repository at the earliest commit after a target date.

    Args:
        repo_url: GitHub repository URL
        target_date: Target date (YYYY-MM-DD)
        output_dir: Directory to clone into
        logger: Logger instance

    Returns:
        Path to cloned repository, or None if failed
    """
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    clone_path = output_dir / repo_name

    logger.info(f"Cloning {repo_url}")

    try:
        # Clone repository
        subprocess.run(
            ["git", "clone", repo_url, str(clone_path)], check=True, capture_output=True, text=True
        )

        # Find earliest commit after target date
        result = subprocess.run(
            [
                "git",
                "-C",
                str(clone_path),
                "log",
                "--reverse",
                "--after",
                target_date,
                "--format=%H",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        commits = result.stdout.strip().split("\n")
        if commits and commits[0]:
            earliest_commit = commits[0]
            logger.info(f"Checking out earliest commit after {target_date}: {earliest_commit[:8]}")

            subprocess.run(
                ["git", "-C", str(clone_path), "checkout", earliest_commit],
                check=True,
                capture_output=True,
                text=True,
            )
        else:
            logger.warning(f"No commits found after {target_date}, using HEAD")

        return clone_path

    except subprocess.CalledProcessError as e:
        logger.exception("Failed to clone repository")
        return None


def extract_documentation_from_repo(repo_path: Path, logger) -> list[dict]:
    """Extract documentation files from a repository.

    Args:
        repo_path: Path to repository
        logger: Logger instance

    Returns:
        List of documentation objects
    """
    docs = []

    # Common documentation file patterns
    doc_patterns = [
        "README.md",
        "README.rst",
        "README.txt",
        "README",
        "DOCUMENTATION.md",
        "docs/**/*.md",
        "docs/**/*.rst",
        "documentation/**/*.md",
    ]

    logger.info(f"Extracting documentation from {repo_path.name}")

    for pattern in doc_patterns:
        if "**" in pattern:
            # Recursive search
            parts = pattern.split("**")
            base_dir = repo_path / parts[0].strip("/")
            if base_dir.exists():
                for doc_file in base_dir.rglob(parts[1].strip("/")):
                    if doc_file.is_file():
                        docs.append(
                            {
                                "type": "documentation",
                                "source": repo_path.name,
                                "file": str(doc_file.relative_to(repo_path)),
                                "path": str(doc_file),
                            }
                        )
        else:
            # Direct file
            doc_file = repo_path / pattern
            if doc_file.exists():
                docs.append(
                    {
                        "type": "documentation",
                        "source": repo_path.name,
                        "file": pattern,
                        "path": str(doc_file),
                    }
                )

    logger.info(f"Found {len(docs)} documentation files")
    return docs


def extract_code_examples(repo_path: Path, logger) -> list[dict]:
    """Extract code examples from a repository.

    Args:
        repo_path: Path to repository
        logger: Logger instance

    Returns:
        List of code example objects
    """
    examples = []

    # Look for example directories
    example_dirs = ["examples", "sample", "samples", "tutorials", "demos"]

    logger.info(f"Extracting code examples from {repo_path.name}")

    for dir_name in example_dirs:
        example_dir = repo_path / dir_name
        if example_dir.exists() and example_dir.is_dir():
            for code_file in example_dir.rglob("*.py"):
                examples.append(
                    {
                        "type": "code_example",
                        "source": repo_path.name,
                        "file": str(code_file.relative_to(repo_path)),
                        "path": str(code_file),
                    }
                )

    logger.info(f"Found {len(examples)} code examples")
    return examples


def fetch_wayback_snapshot(url: str, target_date: str, logger) -> dict | None:
    """Fetch a snapshot from Wayback Machine.

    Args:
        url: URL to fetch
        target_date: Target date (YYYY-MM-DD)
        logger: Logger instance

    Returns:
        Snapshot data or None
    """
    try:
        logger.info(f"Searching Wayback Machine for {url} near {target_date}")

        # Search for snapshots
        cdx = WaybackMachineCDXServerAPI(url)
        snapshots = cdx.snapshots()

        # Find closest snapshot after target date
        target_dt = datetime.fromisoformat(target_date)
        closest_snapshot = None
        min_diff = None

        for snapshot in snapshots:
            snapshot_dt = snapshot.datetime_timestamp
            if snapshot_dt >= target_dt:
                diff = (snapshot_dt - target_dt).total_seconds()
                if min_diff is None or diff < min_diff:
                    min_diff = diff
                    closest_snapshot = snapshot

        if closest_snapshot:
            logger.info(f"Found snapshot from {closest_snapshot.datetime_timestamp}")

            # Fetch content
            @retry_with_backoff(logger=logger)
            def _fetch():
                response = requests.get(closest_snapshot.archive_url, timeout=30)
                response.raise_for_status()
                return response.text

            content = _fetch()

            return {
                "url": url,
                "archive_url": closest_snapshot.archive_url,
                "timestamp": closest_snapshot.datetime_timestamp.isoformat(),
                "content": content,
            }
        else:
            logger.warning(f"No snapshot found for {url} after {target_date}")
            return None

    except Exception as e:
        logger.exception("Error fetching Wayback snapshot")
        return None


def collect_boundary_objects_for_paper(
    paper_name: str, paper_config: dict, config: dict, output_dir: Path, logger
) -> list[dict]:
    """Collect boundary objects for a paper.

    Args:
        paper_name: Name of the paper
        paper_config: Paper configuration
        config: Full configuration
        output_dir: Output directory
        logger: Logger instance

    Returns:
        List of boundary objects
    """
    boundary_objects = []

    # Create temp directory for cloning
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Get boundary object config
        bo_config = config["collection"]["boundary_objects"].get(paper_name, {})

        # Clone official repositories
        if "official_repos" in bo_config:
            for repo_name in bo_config["official_repos"]:
                repo_url = f"https://github.com/{repo_name}.git"
                target_date = bo_config.get("early_commit_date", paper_config["published_date"])

                clone_path = clone_repository_at_commit(repo_url, target_date, temp_path, logger)

                if clone_path:
                    # Extract documentation
                    docs = extract_documentation_from_repo(clone_path, logger)
                    for doc in docs:
                        # Read content
                        with open(doc["path"]) as f:
                            doc["content"] = f.read()
                        del doc["path"]  # Remove absolute path
                        boundary_objects.append(doc)

                    # Extract code examples
                    examples = extract_code_examples(clone_path, logger)
                    for example in examples:
                        # Read content
                        with open(example["path"]) as f:
                            example["content"] = f.read()
                        del example["path"]  # Remove absolute path
                        boundary_objects.append(example)

        # Clone community repositories (for papers without official code)
        if "community_repos" in bo_config:
            for repo_name in bo_config["community_repos"]:
                repo_url = f"https://github.com/{repo_name}.git"
                target_date = paper_config["published_date"]

                clone_path = clone_repository_at_commit(repo_url, target_date, temp_path, logger)

                if clone_path:
                    # Extract README only for community repos
                    readme_patterns = ["README.md", "README.rst", "README.txt", "README"]
                    for pattern in readme_patterns:
                        readme_path = clone_path / pattern
                        if readme_path.exists():
                            with open(readme_path) as f:
                                boundary_objects.append(
                                    {
                                        "type": "community_documentation",
                                        "source": repo_name,
                                        "file": pattern,
                                        "content": f.read(),
                                    }
                                )
                            break

    logger.info(f"Collected {len(boundary_objects)} boundary objects for {paper_name}")
    return boundary_objects


def main():
    """Main entry point."""
    # Setup
    logger = setup_logging()
    config = load_config()

    # Output directory
    output_dir = Path("data/case_study/boundary_objects")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each paper
    for paper_name, paper_config in config["papers"].items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {paper_name}")
        logger.info(f"{'='*60}")

        try:
            paper_output_dir = output_dir / paper_name
            paper_output_dir.mkdir(parents=True, exist_ok=True)

            # Collect boundary objects
            boundary_objects = collect_boundary_objects_for_paper(
                paper_name, paper_config, config, paper_output_dir, logger
            )

            # Save results
            result = {
                "paper": paper_name,
                "arxiv_id": paper_config["arxiv_id"],
                "boundary_objects": boundary_objects,
                "total_objects": len(boundary_objects),
                "collected_at": datetime.now().isoformat(),
            }

            output_file = paper_output_dir / "boundary_objects.json"
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)

            logger.info(f"Saved {len(boundary_objects)} objects to {output_file}")

        except Exception as e:
            logger.exception(f"Failed to collect boundary objects for {paper_name}")
            raise

    logger.info("\n" + "=" * 60)
    logger.info("Boundary object collection complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
