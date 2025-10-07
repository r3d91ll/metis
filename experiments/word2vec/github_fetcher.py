"""
GitHub Code Fetcher for CF Experiments

Fetches code repositories from GitHub with configurable filtering.
Only includes pure implementation code - no docs, tests, or examples.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from github import Github, GithubException
from github.Repository import Repository

logger = logging.getLogger(__name__)


class CodeDocument:
    """Represents a code repository document."""

    def __init__(
        self,
        github_url: str,
        is_official: bool = False,
        code_files: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.github_url = github_url
        self.is_official = is_official
        self.code_files = code_files or {}
        self.metadata = metadata or {}

    @property
    def total_lines(self) -> int:
        """Calculate total lines of code."""
        return sum(len(content.splitlines()) for content in self.code_files.values())


class GitHubCodeFetcher:
    """
    Fetches code from GitHub repositories with intelligent filtering.

    Uses multiple search strategies to find official repositories:
    1. arXiv ID in README
    2. Author name matching
    3. Title similarity matching
    """

    def __init__(self, config: Dict[str, Any], github_token: Optional[str] = None):
        """
        Initialize GitHub code fetcher.

        Args:
            config: Configuration dictionary
            github_token: GitHub personal access token (or from GITHUB_TOKEN env)
        """
        self.config = config
        code_config = config.get("fetching", {}).get("code", {})

        # File filtering configuration
        self.include_extensions = code_config.get("include_extensions", [".py"])
        self.exclude_patterns = code_config.get("exclude_patterns", [])
        self.max_files = code_config.get("max_files", 50)
        self.max_file_size = code_config.get("max_file_size", 1048576)  # 1MB

        # GitHub API setup
        token = github_token or os.environ.get("GITHUB_TOKEN")
        if token:
            self.github = Github(token)
            logger.info("Initialized GitHub API with authentication")
        else:
            self.github = Github()
            logger.warning("Initialized GitHub API without authentication (rate limited)")

        # Search configuration
        search_config = config.get("fetching", {}).get("github", {})
        self.search_strategies = search_config.get(
            "search_strategies",
            ["arxiv_id_in_readme", "author_match", "title_similarity"],
        )
        self.max_results_per_search = search_config.get("max_results_per_search", 10)

    def should_include_file(self, file_path: str) -> bool:
        """
        Check if file should be included based on filtering rules.

        Args:
            file_path: File path to check

        Returns:
            True if file should be included
        """
        file_path_lower = file_path.lower()

        # Check extension (must match)
        if not any(file_path.endswith(ext) for ext in self.include_extensions):
            return False

        # Check exclude patterns (must not match any)
        for pattern in self.exclude_patterns:
            if pattern.lower() in file_path_lower:
                logger.debug(f"Excluding {file_path} (matches pattern: {pattern})")
                return False

        return True

    def find_official_repo(
        self, paper_title: str, authors: List[str], arxiv_id: str
    ) -> Optional[str]:
        """
        Find official repository for a paper using multiple strategies.

        Args:
            paper_title: Paper title
            authors: List of author names
            arxiv_id: arXiv identifier (e.g., "1301.3781")

        Returns:
            Repository URL if found, None otherwise
        """
        logger.info(f"Searching for official repository for {arxiv_id}")

        # Try each search strategy
        for strategy in self.search_strategies:
            if strategy == "arxiv_id_in_readme":
                repo_url = self._search_by_arxiv_id(arxiv_id, paper_title)
            elif strategy == "author_match":
                repo_url = self._search_by_author(authors, paper_title)
            elif strategy == "title_similarity":
                repo_url = self._search_by_title(paper_title)
            else:
                logger.warning(f"Unknown search strategy: {strategy}")
                continue

            if repo_url:
                logger.info(f"Found repository via {strategy}: {repo_url}")
                return repo_url

        logger.warning(f"Could not find official repository for {arxiv_id}")
        return None

    def _search_by_arxiv_id(self, arxiv_id: str, paper_title: str) -> Optional[str]:
        """Search for repositories mentioning the arXiv ID."""
        try:
            # Search for arXiv ID in README
            query = f"{arxiv_id} in:readme"
            results = self.github.search_repositories(
                query, sort="stars", order="desc"
            )

            for i, repo in enumerate(results):
                if i >= self.max_results_per_search:
                    break

                # Check if this looks like the official repo
                if self._is_likely_official(repo, paper_title):
                    return repo.html_url

        except GithubException as e:
            logger.warning(f"GitHub search failed for arXiv ID {arxiv_id}: {e}")

        return None

    def _search_by_author(self, authors: List[str], paper_title: str) -> Optional[str]:
        """Search for repositories by author name."""
        if not authors:
            return None

        try:
            # Try first author
            author_query = authors[0].split()[-1]  # Last name
            query = f"{author_query} {paper_title.split()[0]} in:readme"
            results = self.github.search_repositories(
                query, sort="stars", order="desc"
            )

            for i, repo in enumerate(results):
                if i >= self.max_results_per_search:
                    break

                if self._is_likely_official(repo, paper_title):
                    return repo.html_url

        except GithubException as e:
            logger.warning(f"GitHub search failed for author {authors[0]}: {e}")

        return None

    def _search_by_title(self, paper_title: str) -> Optional[str]:
        """Search for repositories by paper title."""
        try:
            # Extract key terms from title (first 3 words)
            key_terms = " ".join(paper_title.split()[:3])
            query = f"{key_terms} in:readme"
            results = self.github.search_repositories(
                query, sort="stars", order="desc"
            )

            for i, repo in enumerate(results):
                if i >= self.max_results_per_search:
                    break

                if self._is_likely_official(repo, paper_title):
                    return repo.html_url

        except GithubException as e:
            logger.warning(f"GitHub search failed for title {paper_title}: {e}")

        return None

    def _is_likely_official(self, repo: Repository, paper_title: str) -> bool:
        """
        Heuristic to determine if repository is likely official.

        Args:
            repo: GitHub repository object
            paper_title: Paper title

        Returns:
            True if repository looks official
        """
        # Check if paper title terms appear in repo name or description
        title_terms = set(paper_title.lower().split())
        repo_terms = set(repo.name.lower().split("-"))
        if repo.description:
            repo_terms.update(repo.description.lower().split())

        # At least 1 key term should match
        if not title_terms.intersection(repo_terms):
            return False

        # Repository should have reasonable stars (indicates quality)
        if repo.stargazers_count < 10:
            return False

        return True

    def fetch_code(
        self, repo_url: str, is_official: bool = True
    ) -> Optional[CodeDocument]:
        """
        Fetch code files from a GitHub repository.

        Args:
            repo_url: GitHub repository URL
            is_official: Whether this is the official repository

        Returns:
            CodeDocument with filtered code files, or None on error
        """
        logger.info(f"Fetching code from {repo_url}")

        try:
            # Parse repository owner/name from URL
            parts = repo_url.rstrip("/").split("/")
            owner, repo_name = parts[-2], parts[-1]

            # Get repository
            repo = self.github.get_repo(f"{owner}/{repo_name}")

            # Collect metadata
            metadata = {
                "stars": repo.stargazers_count,
                "forks": repo.forks_count,
                "description": repo.description or "",
                "language": repo.language or "Unknown",
                "topics": repo.get_topics(),
            }

            # Get all files from default branch
            code_files = {}
            try:
                contents = repo.get_contents("")
                code_files = self._extract_code_files(repo, contents)
            except GithubException as e:
                logger.error(f"Failed to fetch repository contents: {e}")
                return None

            if not code_files:
                logger.warning(f"No code files found in {repo_url}")
                return None

            logger.info(
                f"Fetched {len(code_files)} code files "
                f"({sum(len(c) for c in code_files.values())} bytes)"
            )

            return CodeDocument(
                github_url=repo_url,
                is_official=is_official,
                code_files=code_files,
                metadata=metadata,
            )

        except GithubException as e:
            logger.error(f"Failed to fetch code from {repo_url}: {e}")
            return None

    def _extract_code_files(
        self, repo: Repository, contents: List, prefix: str = ""
    ) -> Dict[str, str]:
        """
        Recursively extract code files from repository.

        Args:
            repo: GitHub repository object
            contents: List of content files
            prefix: Path prefix for nested files

        Returns:
            Dictionary mapping file paths to contents
        """
        code_files = {}

        for content_file in contents:
            # Build full path
            file_path = (
                f"{prefix}/{content_file.name}" if prefix else content_file.name
            )

            # Handle directories recursively
            if content_file.type == "dir":
                try:
                    nested_contents = repo.get_contents(content_file.path)
                    nested_files = self._extract_code_files(
                        repo, nested_contents, file_path
                    )
                    code_files.update(nested_files)
                except GithubException as e:
                    logger.warning(f"Failed to fetch directory {file_path}: {e}")
                continue

            # Check file filtering rules
            if not self.should_include_file(file_path):
                continue

            # Check file size
            if content_file.size > self.max_file_size:
                logger.debug(
                    f"Skipping {file_path} (size {content_file.size} > "
                    f"{self.max_file_size})"
                )
                continue

            # Check file limit
            if len(code_files) >= self.max_files:
                logger.warning(f"Reached max files limit ({self.max_files})")
                break

            # Download file content
            try:
                content = content_file.decoded_content.decode("utf-8")
                code_files[file_path] = content
                logger.debug(
                    f"Added {file_path} ({len(content)} bytes, "
                    f"{len(content.splitlines())} lines)"
                )
            except Exception as e:
                logger.warning(f"Failed to decode {file_path}: {e}")
                continue

        return code_files

    def close(self):
        """Close GitHub API connection."""
        # GitHub API client doesn't require explicit closing
        pass
