#!/usr/bin/env python3
"""
Custom Extractors Example

This example demonstrates how to extend Metis with custom document extractors.
"""

from pathlib import Path
from typing import List, Union

from metis.extractors import ExtractionResult, ExtractorBase, ExtractorConfig, ExtractorFactory


class MarkdownExtractor(ExtractorBase):
    """
    Custom extractor for Markdown files.

    This demonstrates how to create a custom extractor that follows
    the Metis extractor interface.
    """

    def __init__(self, config=None):
        """Initialize Markdown extractor."""
        super().__init__(config or ExtractorConfig())

    def extract(self, file_path: Union[str, Path], **kwargs) -> ExtractionResult:
        """
        Extract content from a Markdown file.

        Args:
            file_path: Path to the Markdown file

        Returns:
            ExtractionResult with extracted content and metadata
        """
        file_path = Path(file_path)

        if not self.validate_file(file_path):
            return ExtractionResult(
                text="", metadata={"error": "File validation failed"}, error="Invalid file"
            )

        # Read markdown content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract headings
        headings = []
        for line in content.split("\n"):
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                heading_text = line.lstrip("#").strip()
                headings.append({"level": level, "text": heading_text})

        # Extract code blocks
        code_blocks = []
        in_code_block = False
        current_block = []
        current_language = None

        for line in content.split("\n"):
            if line.startswith("```"):
                if in_code_block:
                    # End of code block
                    code_blocks.append(
                        {"language": current_language, "code": "\n".join(current_block)}
                    )
                    current_block = []
                    current_language = None
                    in_code_block = False
                else:
                    # Start of code block
                    current_language = line[3:].strip() or "text"
                    in_code_block = True
            elif in_code_block:
                current_block.append(line)

        # Build metadata
        metadata = {
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "num_headings": len(headings),
            "num_code_blocks": len(code_blocks),
            "extractor": "markdown",
        }

        return ExtractionResult(
            text=content, metadata=metadata, code_blocks=code_blocks, chunks=headings
        )

    def extract_batch(
        self, file_paths: List[Union[str, Path]], **kwargs
    ) -> List[ExtractionResult]:
        """Extract from multiple Markdown files."""
        return [self.extract(path, **kwargs) for path in file_paths]

    @property
    def supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return [".md", ".markdown"]


def main():
    """Run custom extractor example."""
    print("Metis Custom Extractors Example")
    print("=" * 50)

    # Register custom extractor
    print("\n1. Registering custom Markdown extractor...")
    ExtractorFactory.register("markdown", MarkdownExtractor)
    print("   Registered!")

    # Update format mapping
    print("\n2. Updating format mapping...")
    ExtractorFactory._format_mapping[".md"] = "markdown"
    ExtractorFactory._format_mapping[".markdown"] = "markdown"

    # Create a sample Markdown file
    print("\n3. Creating sample Markdown file...")
    sample_md = Path("sample.md")
    sample_md.write_text(
        """# Metis Custom Extractors

## Overview

Metis makes it easy to create custom extractors for any document format.

## Example

Here's a simple code example:

```python
from metis.extractors import ExtractorBase

class MyExtractor(ExtractorBase):
    def extract(self, file_path):
        # Your extraction logic here
        pass
```

## Features

- Clean interface
- Easy to extend
- Works with existing Metis infrastructure
"""
    )

    # Use factory to create extractor
    print("\n4. Creating extractor from factory...")
    extractor = ExtractorFactory.create_for_file(sample_md)
    print(f"   Created: {extractor.__class__.__name__}")

    # Extract content
    print("\n5. Extracting content...")
    result = extractor.extract(sample_md)

    print(f"   Extracted {len(result.text)} characters")
    print(f"   Found {result.metadata['num_headings']} headings")
    print(f"   Found {result.metadata['num_code_blocks']} code blocks")

    # Show extracted headings
    print("\n6. Extracted headings:")
    for chunk in result.chunks:
        indent = "  " * (chunk["level"] - 1)
        print(f"   {indent}- {chunk['text']}")

    # Show code blocks
    print("\n7. Extracted code blocks:")
    for i, block in enumerate(result.code_blocks, 1):
        print(f"   Block {i} (language: {block['language']})")
        print(f"   Lines: {len(block['code'].split(chr(10)))}")

    # Clean up
    sample_md.unlink()

    print("\n" + "=" * 50)
    print("Example completed successfully!")


if __name__ == "__main__":
    main()
