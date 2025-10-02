"""
Extractors Module

Provides document extraction capabilities for various file formats.
Supports PDF, LaTeX, and code files with structured content extraction.
"""

from .base import ExtractionResult, ExtractorBase, ExtractorConfig
from .factory import ExtractorFactory

# Auto-register available extractors
try:
    from .docling import DoclingExtractor

    ExtractorFactory.register("docling", DoclingExtractor)
except ImportError:
    pass

try:
    from .latex import LaTeXExtractor

    ExtractorFactory.register("latex", LaTeXExtractor)
except ImportError:
    pass

try:
    from .code import CodeExtractor

    ExtractorFactory.register("code", CodeExtractor)
except ImportError:
    pass

try:
    from .treesitter import TreeSitterExtractor

    ExtractorFactory.register("treesitter", TreeSitterExtractor)
except ImportError:
    pass

try:
    from .robust import RobustExtractor

    ExtractorFactory.register("robust", RobustExtractor)
except ImportError:
    pass

# Make all classes available at module level
try:
    from .docling import DoclingExtractor
except ImportError:
    DoclingExtractor = None  # type: ignore[misc]

try:
    from .latex import LaTeXExtractor
except ImportError:
    LaTeXExtractor = None  # type: ignore[misc]

try:
    from .code import CodeExtractor
except ImportError:
    CodeExtractor = None  # type: ignore[misc]

try:
    from .treesitter import TreeSitterExtractor
except ImportError:
    TreeSitterExtractor = None  # type: ignore[misc]

try:
    from .robust import RobustExtractor
except ImportError:
    RobustExtractor = None  # type: ignore[misc]

__all__ = [
    "ExtractorBase",
    "ExtractorConfig",
    "ExtractionResult",
    "ExtractorFactory",
    "DoclingExtractor",
    "LaTeXExtractor",
    "CodeExtractor",
    "TreeSitterExtractor",
    "RobustExtractor",
]


# Convenience function
def create_extractor_for_file(file_path, **kwargs):
    """
    Create an extractor for a given file based on extension.

    Args:
        file_path: Path to the file
        **kwargs: Additional configuration

    Returns:
        Extractor instance

    Example:
        >>> extractor = create_extractor_for_file("paper.pdf")
        >>> result = extractor.extract("paper.pdf")
    """
    return ExtractorFactory.create_for_file(file_path, **kwargs)
