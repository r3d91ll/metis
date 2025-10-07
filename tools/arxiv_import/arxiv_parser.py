"""
arXiv ID Parser and Date Utilities

Handles both old (archive/YYMMNNN) and new (YYMM.NNNNN) arXiv ID formats,
converting them to a canonical internal format for consistent sorting and retrieval.

Internal ID Format:
- Old format: ARCHIVE_YYYYMM_SSSSS (includes archive prefix for uniqueness)
- New format: YYYYMM_SSSSS (no archive prefix needed)

Examples:
    hep-ph/9901001    → hep_ph_199901_00001
    astro-ph/0703001  → astro_ph_200703_00001
    math.GT/0309136   → math_GT_200309_00136
    0704.0001         → 200704_00001
    2301.12345        → 202301_12345

Note: Archive prefix is REQUIRED for old format because different archives
      have independent sequence numbering (e.g., hep-ph/9411001 and
      astro-ph/9411001 are DIFFERENT papers).
"""

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Optional
from email.utils import parsedate_to_datetime


@dataclass
class ParsedArxivId:
    """Parsed arXiv ID with normalized internal ID."""

    original_id: str
    internal_id: str  # Old: ARCHIVE_YYYYMM_SSSSS, New: YYYYMM_SSSSS
    year: int
    month: int
    sequence: int
    archive: Optional[str]  # Only for old format (e.g., "hep-ph")
    is_old_format: bool


class ArxivIdParser:
    """
    Parse both old and new arXiv ID formats into canonical internal IDs.

    Old Format (pre-April 2007): archive/YYMMNNN
    - Examples: hep-ph/9901001, astro-ph/0703001, math.GT/0309136
    - Archive: subject classification
    - YY: 2-digit year (91-99 = 1990s, 00-06 = 2000s)
    - MM: 2-digit month
    - NNN: 3-4 digit sequence number

    New Format (April 2007 onwards): YYMM.NNNNN
    - Examples: 0704.0001, 2301.12345
    - YYMM: year and month
    - NNNNN: 4-5 digit sequence number (zero-padded)
    """

    # Old format: archive/YYMMNNN (before April 2007)
    OLD_PATTERN = re.compile(
        r'^([a-z\-\.]+)/(\d{2})(\d{2})(\d{3,4})$',
        re.IGNORECASE
    )

    # New format: YYMM.NNNNN (April 2007 onwards)
    NEW_PATTERN = re.compile(r'^(\d{2})(\d{2})\.(\d{4,5})$')

    @staticmethod
    def parse(arxiv_id: str) -> ParsedArxivId:
        """
        Parse arXiv ID and return normalized internal ID.

        Args:
            arxiv_id: Original arXiv ID in either old or new format

        Returns:
            ParsedArxivId with internal_id in YYYYMM_SSSSS format

        Raises:
            ValueError: If ID format is invalid

        Examples:
            >>> ArxivIdParser.parse("hep-ph/9901001").internal_id
            'hep_ph_199901_00001'
            >>> ArxivIdParser.parse("astro-ph/0703001").internal_id
            'astro_ph_200703_00001'
            >>> ArxivIdParser.parse("math.GT/0309136").internal_id
            'math_GT_200309_00136'
            >>> ArxivIdParser.parse("0704.0001").internal_id
            '200704_00001'
            >>> ArxivIdParser.parse("2301.12345").internal_id
            '202301_12345'
        """
        # Try old format first
        match = ArxivIdParser.OLD_PATTERN.match(arxiv_id)
        if match:
            archive, yy, mm, seq = match.groups()

            # Determine century
            # 91-99 = 1990s, 00-06 = 2000s (format changed in April 2007)
            year = int(yy)
            if year >= 91:
                year += 1900
            else:
                year += 2000

            month = int(mm)
            sequence = int(seq)

            # Validate month
            if not (1 <= month <= 12):
                raise ValueError(f"Invalid month in arXiv ID: {arxiv_id}")

            # CRITICAL: Include archive prefix for old format to ensure uniqueness
            # Different archives have independent sequence numbering, so:
            #   hep-ph/9411001 ≠ astro-ph/9411001 (different papers!)
            # Sanitize archive name: lowercase, replace dots/dashes with underscores for valid keys
            archive_clean = archive.lower().replace('.', '_').replace('-', '_')
            internal_id = f"{archive_clean}_{year:04d}{month:02d}_{sequence:05d}"

            return ParsedArxivId(
                original_id=arxiv_id,
                internal_id=internal_id,
                year=year,
                month=month,
                sequence=sequence,
                archive=archive,
                is_old_format=True
            )

        # Try new format
        match = ArxivIdParser.NEW_PATTERN.match(arxiv_id)
        if match:
            yy, mm, seq = match.groups()

            # New format is always 2000s
            year = 2000 + int(yy)
            month = int(mm)
            sequence = int(seq)

            # Validate month
            if not (1 <= month <= 12):
                raise ValueError(f"Invalid month in arXiv ID: {arxiv_id}")

            internal_id = f"{year:04d}{month:02d}_{sequence:05d}"

            return ParsedArxivId(
                original_id=arxiv_id,
                internal_id=internal_id,
                year=year,
                month=month,
                sequence=sequence,
                archive=None,
                is_old_format=False
            )

        raise ValueError(f"Invalid arXiv ID format: {arxiv_id}")

    @staticmethod
    def extract_date_from_version(version_str: str) -> datetime:
        """
        Parse version creation date from arXiv metadata.

        Args:
            version_str: Date string in RFC 2822 format

        Returns:
            Parsed datetime object

        Examples:
            >>> ArxivIdParser.extract_date_from_version("Mon, 2 Apr 2007 19:18:42 GMT")
            datetime.datetime(2007, 4, 2, 19, 18, 42, tzinfo=...)
        """
        # arXiv uses RFC 2822 format: "Mon, 2 Apr 2007 19:18:42 GMT"
        return parsedate_to_datetime(version_str)

    @staticmethod
    def validate_temporal_ordering(parsed_ids: list[ParsedArxivId]) -> bool:
        """
        Validate that internal IDs maintain temporal ordering.

        Args:
            parsed_ids: List of parsed arXiv IDs

        Returns:
            True if internal IDs sort chronologically

        Note:
            This is critical for ensuring our internal ID scheme works correctly.
        """
        if len(parsed_ids) < 2:
            return True

        # Sort by internal ID
        sorted_ids = sorted(parsed_ids, key=lambda x: x.internal_id)

        # Check that year-month increases monotonically
        for i in range(len(sorted_ids) - 1):
            current = sorted_ids[i]
            next_id = sorted_ids[i + 1]

            # Year-month should never decrease
            current_ym = current.year * 100 + current.month
            next_ym = next_id.year * 100 + next_id.month

            if current_ym > next_ym:
                return False

        return True


def normalize_arxiv_id(arxiv_id: str) -> str:
    """
    Convenience function to get internal ID from arXiv ID.

    Args:
        arxiv_id: Original arXiv ID

    Returns:
        Internal ID (format depends on old vs new)

    Examples:
        >>> normalize_arxiv_id("hep-ph/9901001")
        'hep_ph_199901_00001'
        >>> normalize_arxiv_id("0704.0001")
        '200704_00001'
    """
    return ArxivIdParser.parse(arxiv_id).internal_id
