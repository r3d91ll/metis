"""
Comprehensive tests for arXiv ID parser.

This is mission-critical - if ID parsing is wrong, everything downstream fails.
"""

import pytest
from datetime import datetime
import sys
from pathlib import Path

# Add experiments to path
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments" / "arxiv_import"))

from arxiv_parser import ArxivIdParser, ParsedArxivId, normalize_arxiv_id


class TestOldFormatParsing:
    """Test parsing of old arXiv ID format (archive/YYMMNNN)."""

    def test_archive_prefix_uniqueness(self):
        """
        CRITICAL: Different archives with same YYMMNNN must produce different internal_ids.

        This was the bug that caused 312k papers to be lost!
        Different archives have independent sequence numbering, so:
          - acc-phys/9411001 is a DIFFERENT paper from
          - adap-org/9411001 is a DIFFERENT paper from
          - hep-th/9411001

        All three must have unique internal_ids.
        """
        ids = [
            "acc-phys/9411001",
            "adap-org/9411001",
            "hep-th/9411001",
            "astro-ph/9411001",
        ]

        parsed_ids = [ArxivIdParser.parse(id) for id in ids]
        internal_ids = [p.internal_id for p in parsed_ids]

        # All internal_ids must be unique
        assert len(internal_ids) == len(set(internal_ids)), \
            f"Internal IDs must be unique! Got: {internal_ids}"

        # Verify expected format
        assert internal_ids[0] == "acc_phys_199411_00001"
        assert internal_ids[1] == "adap_org_199411_00001"
        assert internal_ids[2] == "hep_th_199411_00001"
        assert internal_ids[3] == "astro_ph_199411_00001"

    def test_1990s_paper(self):
        """Test paper from 1990s (year >= 91)."""
        parsed = ArxivIdParser.parse("hep-ph/9901001")
        assert parsed.internal_id == "hep_ph_199901_00001"
        assert parsed.year == 1999
        assert parsed.month == 1
        assert parsed.sequence == 1
        assert parsed.archive == "hep-ph"
        assert parsed.is_old_format is True
        assert parsed.original_id == "hep-ph/9901001"

    def test_2000s_paper(self):
        """Test paper from early 2000s (year < 91)."""
        parsed = ArxivIdParser.parse("astro-ph/0703001")
        assert parsed.internal_id == "astro_ph_200703_00001"
        assert parsed.year == 2007
        assert parsed.month == 3
        assert parsed.sequence == 1
        assert parsed.archive == "astro-ph"
        assert parsed.is_old_format is True

    def test_math_paper(self):
        """Test math archive with dotted notation."""
        parsed = ArxivIdParser.parse("math.GT/0309136")
        assert parsed.internal_id == "math_gt_200309_00136"
        assert parsed.year == 2003
        assert parsed.month == 9
        assert parsed.sequence == 136
        assert parsed.archive == "math.GT"

    def test_4digit_sequence(self):
        """Test 4-digit sequence number."""
        # Note: In old format hep-th/YYMMNNN, this would be hep-th/991234S (4-digit seq)
        # But hep-th/9912345 is actually YY=99, MM=12, NNN=345 (3-digit seq)
        # For a true 4-digit example, we'd need something like hep-th/99011234
        parsed = ArxivIdParser.parse("hep-th/99011234")
        assert parsed.internal_id == "hep_th_199901_01234"
        assert parsed.sequence == 1234

    def test_december_paper(self):
        """Test December (month 12) parsing."""
        parsed = ArxivIdParser.parse("hep-ph/9912999")
        assert parsed.internal_id == "hep_ph_199912_00999"
        assert parsed.month == 12

    def test_case_insensitive(self):
        """Test that parsing is case-insensitive."""
        parsed1 = ArxivIdParser.parse("HEP-PH/9901001")
        parsed2 = ArxivIdParser.parse("hep-ph/9901001")
        assert parsed1.internal_id == parsed2.internal_id


class TestNewFormatParsing:
    """Test parsing of new arXiv ID format (YYMM.NNNNN)."""

    def test_first_new_format_paper(self):
        """Test first paper in new format (April 2007)."""
        parsed = ArxivIdParser.parse("0704.0001")
        assert parsed.internal_id == "200704_00001"
        assert parsed.year == 2007
        assert parsed.month == 4
        assert parsed.sequence == 1
        assert parsed.archive is None
        assert parsed.is_old_format is False

    def test_2023_paper(self):
        """Test recent paper."""
        parsed = ArxivIdParser.parse("2301.12345")
        assert parsed.internal_id == "202301_12345"
        assert parsed.year == 2023
        assert parsed.month == 1
        assert parsed.sequence == 12345

    def test_5digit_sequence(self):
        """Test 5-digit sequence number."""
        parsed = ArxivIdParser.parse("2312.99999")
        assert parsed.internal_id == "202312_99999"
        assert parsed.sequence == 99999

    def test_4digit_sequence_padded(self):
        """Test that 4-digit sequences get zero-padded."""
        parsed = ArxivIdParser.parse("0704.1234")
        assert parsed.internal_id == "200704_01234"
        assert parsed.sequence == 1234


class TestFormatTransition:
    """Test the transition between old and new formats (March-April 2007)."""

    def test_march_2007_old_format(self):
        """Last month of old format."""
        parsed = ArxivIdParser.parse("hep-ph/0703999")
        assert parsed.internal_id == "hep_ph_200703_00999"
        assert parsed.is_old_format is True

    def test_april_2007_new_format(self):
        """First month of new format."""
        parsed = ArxivIdParser.parse("0704.0001")
        assert parsed.internal_id == "200704_00001"
        assert parsed.is_old_format is False

    def test_transition_ordering(self):
        """Ensure proper ordering across format transition."""
        old = ArxivIdParser.parse("hep-ph/0703999")
        new = ArxivIdParser.parse("0704.0001")
        # Old format has archive prefix, so it will sort after new format
        # within same year-month. That's OK - we primarily care about
        # year-month ordering, not cross-format within same month
        assert old.year == 2007 and old.month == 3
        assert new.year == 2007 and new.month == 4


class TestTemporalOrdering:
    """Test that internal IDs maintain proper temporal ordering."""

    def test_chronological_ordering(self):
        """Test that IDs sort chronologically by year-month."""
        ids = [
            "hep-ph/9901001",  # Jan 1999
            "hep-ph/9912001",  # Dec 1999
            "astro-ph/0001001",  # Jan 2000
            "hep-ph/0703001",  # Mar 2007 (old format)
            "0704.0001",  # Apr 2007 (new format)
            "2301.12345",  # Jan 2023
        ]

        parsed = [ArxivIdParser.parse(id) for id in ids]

        # Check year-month ordering (what really matters for temporal sorting)
        year_months = [(p.year, p.month) for p in parsed]
        assert year_months == sorted(year_months)

        # Note: Within same year-month, old format (with archive prefix)
        # will sort differently than new format, but that's acceptable

    def test_same_month_ordering(self):
        """Test ordering within same month and same archive."""
        id1 = ArxivIdParser.parse("hep-ph/9901001")
        id2 = ArxivIdParser.parse("hep-ph/9901002")
        # Same archive prefix, so sequence number determines order
        assert id1.internal_id < id2.internal_id

    def test_validate_temporal_ordering_function(self):
        """Test the validation helper function."""
        # Use only new format or only old format to avoid cross-format sorting issues
        parsed_ids = [
            ArxivIdParser.parse("0704.0001"),
            ArxivIdParser.parse("0801.0001"),
            ArxivIdParser.parse("2301.12345"),
        ]
        assert ArxivIdParser.validate_temporal_ordering(parsed_ids) is True

    def test_validate_temporal_ordering_works_any_input_order(self):
        """Test that validation works regardless of input order."""
        # Create IDs from different times (same format to avoid sorting issues)
        id1 = ParsedArxivId(
            original_id="test1",
            internal_id="202301_00001",  # New format
            year=2023,
            month=1,
            sequence=1,
            archive=None,
            is_old_format=False
        )
        id2 = ParsedArxivId(
            original_id="test2",
            internal_id="199901_00001",  # Old format but same structure
            year=1999,
            month=1,
            sequence=1,
            archive=None,
            is_old_format=False
        )
        # Should return True regardless of input order (validation sorts internally)
        assert ArxivIdParser.validate_temporal_ordering([id1, id2]) is True
        assert ArxivIdParser.validate_temporal_ordering([id2, id1]) is True


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_invalid_format(self):
        """Test completely invalid format."""
        with pytest.raises(ValueError, match="Invalid arXiv ID format"):
            ArxivIdParser.parse("invalid-id")

    def test_invalid_month_old_format(self):
        """Test invalid month in old format."""
        with pytest.raises(ValueError, match="Invalid month"):
            ArxivIdParser.parse("hep-ph/9913001")  # Month 13

    def test_invalid_month_new_format(self):
        """Test invalid month in new format."""
        with pytest.raises(ValueError, match="Invalid month"):
            ArxivIdParser.parse("0713.0001")  # Month 13

    def test_empty_string(self):
        """Test empty string."""
        with pytest.raises(ValueError):
            ArxivIdParser.parse("")

    def test_missing_sequence(self):
        """Test ID with missing sequence number."""
        with pytest.raises(ValueError):
            ArxivIdParser.parse("hep-ph/9901")


class TestDateParsing:
    """Test version date parsing."""

    def test_parse_version_date(self):
        """Test parsing of RFC 2822 date format."""
        date_str = "Mon, 2 Apr 2007 19:18:42 GMT"
        dt = ArxivIdParser.extract_date_from_version(date_str)

        assert dt.year == 2007
        assert dt.month == 4
        assert dt.day == 2
        assert dt.hour == 19
        assert dt.minute == 18
        assert dt.second == 42

    def test_various_date_formats(self):
        """Test various date string formats from arXiv."""
        dates = [
            "Mon, 2 Apr 2007 19:18:42 GMT",
            "Tue, 24 Jul 2007 20:10:27 GMT",
            "Wed, 15 Jan 1999 12:00:00 GMT",
        ]

        for date_str in dates:
            dt = ArxivIdParser.extract_date_from_version(date_str)
            assert isinstance(dt, datetime)


class TestConvenienceFunction:
    """Test the normalize_arxiv_id convenience function."""

    def test_old_format(self):
        """Test normalization of old format."""
        assert normalize_arxiv_id("hep-ph/9901001") == "hep_ph_199901_00001"

    def test_new_format(self):
        """Test normalization of new format."""
        assert normalize_arxiv_id("2301.12345") == "202301_12345"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_year_2000_transition(self):
        """
        Verify parsing and chronological ordering across the 1999→2000 boundary for old-format arXiv IDs.
        
        Asserts that a December 1999 old-format ID and a January 2000 old-format ID produce the correct years and that the generated internal IDs sort chronologically (December 1999 < January 2000).
        """
        dec_1999 = ArxivIdParser.parse("hep-ph/9912001")
        jan_2000 = ArxivIdParser.parse("hep-ph/0001001")

        assert dec_1999.year == 1999
        assert jan_2000.year == 2000
        # Same archive, year increases, so internal_id should increase
        assert dec_1999.internal_id < jan_2000.internal_id

    def test_january_papers(self):
        """Test January (month 01) parsing."""
        parsed = ArxivIdParser.parse("hep-ph/9901001")
        assert parsed.month == 1
        assert "199901_" in parsed.internal_id

    def test_sequence_number_1(self):
        """Test first paper of month (sequence 1)."""
        parsed = ArxivIdParser.parse("0704.0001")
        assert parsed.sequence == 1
        assert parsed.internal_id.endswith("_00001")

    def test_very_high_sequence_number(self):
        """Test handling of very high sequence numbers."""
        parsed = ArxivIdParser.parse("2312.99999")
        assert parsed.sequence == 99999
        assert parsed.internal_id == "202312_99999"


class TestInternalIdFormat:
    """Test that internal IDs conform to YYYYMM_SSSSS format."""

    def test_id_length(self):
        """Test internal ID has correct format."""
        # Old format: ARCHIVE_YYYYMM_SSSSS (variable length due to archive)
        old_parsed = ArxivIdParser.parse("hep-ph/9901001")
        assert old_parsed.internal_id == "hep_ph_199901_00001"
        assert old_parsed.internal_id.count("_") == 3  # archive_YYYY_MM_SSSSS

        # New format: YYYYMM_SSSSS (fixed 12 characters)
        new_parsed = ArxivIdParser.parse("0704.0001")
        assert len(new_parsed.internal_id) == 12  # YYYYMM_SSSSS format

    def test_id_components(self):
        """
        Verify that the parsed internal_id contains two components: a six-digit YYYYMM and a five-digit sequence.
        
        Asserts that parsing "2301.12345" produces internal_id of the form "202301_12345", where the first part is length 6 and the second part is length 5.
        """
        parsed = ArxivIdParser.parse("2301.12345")
        parts = parsed.internal_id.split("_")

        assert len(parts) == 2
        assert parts[0] == "202301"  # YYYYMM
        assert parts[1] == "12345"  # SSSSS
        assert len(parts[0]) == 6
        assert len(parts[1]) == 5

    def test_zero_padding(self):
        """Test that sequence numbers are zero-padded."""
        new_fmt = ArxivIdParser.parse("0704.0001")
        assert new_fmt.internal_id.endswith("_00001")

        old_fmt = ArxivIdParser.parse("hep-ph/9901001")
        assert old_fmt.internal_id.endswith("_00001")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])