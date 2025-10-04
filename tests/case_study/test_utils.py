"""Tests for case study utilities."""

import json
import tempfile
import time
from pathlib import Path

import pytest

# Add scripts directory to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "case_study"))

from utils import (
    RateLimiter,
    load_config,
    retry_with_backoff,
    save_json_incremental,
    setup_logging,
    validate_citation_data,
    validate_repo_data,
)


def test_setup_logging(tmp_path):
    """Test logging setup."""
    log_file = tmp_path / "test.log"
    logger = setup_logging(str(log_file))

    assert logger is not None
    assert log_file.exists()

    logger.info("Test message")

    # Check log file contains message
    with open(log_file) as f:
        content = f.read()
        assert "Test message" in content


def test_load_config():
    """Test configuration loading."""
    config = load_config()

    assert "papers" in config
    assert "transformers" in config["papers"]
    assert "capsules" in config["papers"]
    assert "api" in config
    assert "collection" in config


def test_rate_limiter():
    """Test rate limiter."""
    limiter = RateLimiter(requests_per_second=10)

    start_time = time.time()

    for _ in range(3):
        limiter.wait()

    elapsed = time.time() - start_time

    # Should take at least 0.2 seconds for 3 requests at 10 req/s
    assert elapsed >= 0.2


def test_retry_with_backoff():
    """Test retry logic."""
    call_count = [0]

    @retry_with_backoff(max_retries=3, initial_delay=0.01)
    def failing_func():
        call_count[0] += 1
        if call_count[0] < 3:
            raise ValueError("Test error")
        return "success"

    result = failing_func()
    assert result == "success"
    assert call_count[0] == 3


def test_validate_citation_data():
    """Test citation data validation."""
    valid_data = {
        "paper_id": "1706.03762",
        "title": "Test Paper",
        "monthly_citations": [
            {"year": 2017, "month": 6, "count": 5},
            {"year": 2017, "month": 7, "count": 10},
        ],
        "total_citations": 15,
    }

    assert validate_citation_data(valid_data)

    # Test invalid data
    invalid_data = {"paper_id": "123"}

    with pytest.raises(AssertionError):
        validate_citation_data(invalid_data)


def test_validate_repo_data():
    """Test repository data validation."""
    valid_data = {
        "paper": "transformers",
        "repositories": [
            {
                "url": "https://github.com/test/repo",
                "created_at": "2017-06-12T00:00:00Z",
                "stars": 100,
                "forks": 20,
                "language": "Python",
                "type": "official",
            }
        ],
    }

    assert validate_repo_data(valid_data)

    # Test invalid type
    invalid_data = {
        "paper": "transformers",
        "repositories": [
            {
                "url": "https://github.com/test/repo",
                "created_at": "2017-06-12T00:00:00Z",
                "stars": 100,
                "forks": 20,
                "language": "Python",
                "type": "invalid_type",
            }
        ],
    }

    with pytest.raises(AssertionError):
        validate_repo_data(invalid_data)


def test_save_json_incremental(tmp_path):
    """Test incremental JSON saving with backup."""
    output_file = tmp_path / "test.json"

    # First save
    data1 = {"test": "data1"}
    save_json_incremental(data1, output_file)

    assert output_file.exists()
    with open(output_file) as f:
        assert json.load(f) == data1

    # Second save (should create backup)
    data2 = {"test": "data2"}
    save_json_incremental(data2, output_file)

    backup_file = output_file.with_suffix(".json.bak")
    assert backup_file.exists()

    with open(backup_file) as f:
        assert json.load(f) == data1

    with open(output_file) as f:
        assert json.load(f) == data2
