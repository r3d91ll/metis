"""Utility functions for case study data collection."""

import logging
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import requests
import yaml

T = TypeVar("T")


def setup_logging(log_file: str = "logs/case_study_collection.log") -> logging.Logger:
    """Set up logging configuration."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("case_study")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def load_config(config_path: str = "config/case_study.yaml") -> dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def retry_with_backoff(
    func: Callable[..., T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    logger: logging.Logger | None = None,
) -> Callable[..., T]:
    """Decorator for retrying functions with exponential backoff."""

    def wrapper(*args: Any, **kwargs: Any) -> T:
        delay = initial_delay
        last_exception = None

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if logger:
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                time.sleep(delay)
                delay *= backoff_factor

        if logger:
            logger.error(f"All {max_retries} attempts failed. Last error: {last_exception}")
        raise last_exception  # type: ignore

    return wrapper


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, requests_per_second: float):
        """Initialize rate limiter.

        Args:
            requests_per_second: Maximum number of requests per second
        """
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0

    def wait(self) -> None:
        """Wait if necessary to respect rate limit."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_interval:
            time.sleep(self.min_interval - time_since_last)

        self.last_request_time = time.time()


def validate_citation_data(data: dict[str, Any]) -> bool:
    """Validate citation data structure.

    Args:
        data: Citation data to validate

    Returns:
        True if valid, raises AssertionError otherwise
    """
    assert "paper_id" in data, "Missing paper_id"
    assert "title" in data, "Missing title"
    assert "monthly_citations" in data, "Missing monthly_citations"

    for month_data in data["monthly_citations"]:
        assert "year" in month_data, f"Missing year in {month_data}"
        assert "month" in month_data, f"Missing month in {month_data}"
        assert "count" in month_data, f"Missing count in {month_data}"

    if "total_citations" in data:
        expected_total = sum(m["count"] for m in data["monthly_citations"])
        assert (
            data["total_citations"] == expected_total
        ), f"Total mismatch: {data['total_citations']} != {expected_total}"

    return True


def validate_repo_data(data: dict[str, Any]) -> bool:
    """Validate repository data structure.

    Args:
        data: Repository data to validate

    Returns:
        True if valid, raises AssertionError otherwise
    """
    assert "paper" in data, "Missing paper identifier"
    assert "repositories" in data, "Missing repositories list"

    valid_types = ["official", "framework", "tutorial", "application", "research"]

    for repo in data["repositories"]:
        assert "url" in repo, f"Missing url in {repo}"
        assert "created_at" in repo, f"Missing created_at in {repo}"
        assert "type" in repo, f"Missing type in {repo}"
        assert repo["type"] in valid_types, f"Invalid type: {repo['type']}"
        assert "stars" in repo, f"Missing stars in {repo}"
        assert "language" in repo, f"Missing language in {repo}"

    return True


def save_json_incremental(data: Any, filepath: Path, logger: logging.Logger | None = None) -> None:
    """Save data to JSON file with backup.

    Args:
        data: Data to save
        filepath: Path to save to
        logger: Optional logger
    """
    import json

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Create backup if file exists
    if filepath.exists():
        backup_path = filepath.with_suffix(filepath.suffix + ".bak")
        filepath.rename(backup_path)
        if logger:
            logger.info(f"Created backup: {backup_path}")

    # Save new data
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    if logger:
        logger.info(f"Saved data to {filepath}")
