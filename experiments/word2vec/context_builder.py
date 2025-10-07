"""
Combined Context Builder

Builds unified contexts from paper (markdown + LaTeX) and code for embedding generation.
Intelligently merges content with section markers and handles Jina v4's 32k token context window.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CombinedContext:
    """Represents a combined paper+code context for embedding."""

    arxiv_id: str
    title: str
    content: str
    components: Dict[str, int]  # Component name -> char count
    total_chars: int
    total_tokens_estimate: int  # Rough estimate (chars / 4)
    truncated: bool
    metadata: Dict[str, any]


class CombinedContextBuilder:
    """
    Builds unified contexts from paper and code for embedding generation.

    Strategy:
    1. Combine markdown + LaTeX + code with section markers
    2. Prioritize content based on semantic value
    3. Fit within Jina v4's 32k token window (~128k chars)
    """

    def __init__(self, max_tokens: int = 32000):
        """
        Initialize context builder.

        Args:
            max_tokens: Maximum tokens for embedding model (default: 32000 for Jina v4)
        """
        self.max_tokens = max_tokens
        # Conservative estimate: 4 chars per token average
        self.max_chars = max_tokens * 4

    def build_context(
        self,
        paper_doc: Dict[str, any],
        code_doc: Optional[Dict[str, any]] = None,
        include_latex: bool = True
    ) -> CombinedContext:
        """
        Build combined context from paper and code documents.

        Args:
            paper_doc: Paper document from ArangoDB
            code_doc: Code document from ArangoDB (optional)
            include_latex: Whether to include LaTeX source (default: True)

        Returns:
            CombinedContext with merged content
        """
        arxiv_id = paper_doc["arxiv_id"]
        title = paper_doc["title"]

        logger.info(f"Building combined context for {arxiv_id}")

        # Collect content components
        components = []

        # 1. Paper metadata (always include)
        metadata_section = self._build_metadata_section(paper_doc)
        components.append(("metadata", metadata_section))

        # 2. Paper markdown (PDF extraction)
        markdown_content = paper_doc.get("markdown_content", "")
        if markdown_content:
            markdown_section = self._build_markdown_section(markdown_content)
            components.append(("markdown", markdown_section))

        # 3. LaTeX source (optional, for formulas/structure)
        if include_latex and paper_doc.get("latex_source"):
            latex_content = paper_doc["latex_source"]
            latex_section = self._build_latex_section(latex_content)
            components.append(("latex", latex_section))

        # 4. Code (optional)
        if code_doc and code_doc.get("code_files"):
            code_section = self._build_code_section(code_doc)
            components.append(("code", code_section))

        # Combine components and check size
        combined, component_sizes, truncated = self._merge_components(components)

        total_chars = len(combined)
        total_tokens_estimate = total_chars // 4

        logger.info(f"Context built: {total_chars:,} chars (~{total_tokens_estimate:,} tokens)")
        logger.info(f"  Components: {', '.join(f'{k}={v:,}' for k, v in component_sizes.items())}")
        if truncated:
            logger.warning(f"  ⚠ Content truncated to fit {self.max_chars:,} char limit")

        return CombinedContext(
            arxiv_id=arxiv_id,
            title=title,
            content=combined,
            components=component_sizes,
            total_chars=total_chars,
            total_tokens_estimate=total_tokens_estimate,
            truncated=truncated,
            metadata={
                "has_code": code_doc is not None,
                "has_latex": include_latex and bool(paper_doc.get("latex_source")),
                "code_language": code_doc.get("repository_metadata", {}).get("language")
                if code_doc else None
            }
        )

    def _build_metadata_section(self, paper_doc: Dict[str, any]) -> str:
        """Build metadata section."""
        title = paper_doc["title"]
        authors = paper_doc.get("authors", [])
        abstract = paper_doc.get("abstract", "")
        arxiv_id = paper_doc["arxiv_id"]

        section = f"""# {title}

**arXiv ID**: {arxiv_id}
**Authors**: {', '.join(authors)}

## Abstract

{abstract}

---
"""
        return section

    def _build_markdown_section(self, markdown_content: str) -> str:
        """Build markdown section from PDF extraction."""
        section = f"""
## Paper Content (PDF Extraction)

{markdown_content}

---
"""
        return section

    def _build_latex_section(self, latex_content: str) -> str:
        """Build LaTeX source section."""
        section = f"""
## LaTeX Source (Mathematical Formulas & Structure)

```latex
{latex_content}
```

---
"""
        return section

    def _build_code_section(self, code_doc: Dict[str, any]) -> str:
        """Build code section from repository."""
        github_url = code_doc.get("github_url", "Unknown")
        code_files = code_doc.get("code_files", {})
        language = code_doc.get("repository_metadata", {}).get("language", "Unknown")

        # Build header
        section_parts = [
            "\n## Code Repository\n",
            f"**Repository**: {github_url}\n",
            f"**Language**: {language}\n",
            f"**Files**: {len(code_files)}\n\n"
        ]

        # Add each code file
        for file_path, file_content in sorted(code_files.items()):
            # Detect language for syntax highlighting
            extension = file_path.split(".")[-1] if "." in file_path else ""
            lang_map = {
                "py": "python", "c": "c", "cpp": "cpp", "cc": "cpp",
                "h": "c", "hpp": "cpp", "java": "java", "js": "javascript",
                "ts": "typescript", "go": "go", "rs": "rust"
            }
            syntax_lang = lang_map.get(extension, language.lower() if language != "Unknown" else "")

            section_parts.append(f"### {file_path}\n\n")
            section_parts.append(f"```{syntax_lang}\n{file_content}\n```\n\n")

        section_parts.append("---\n")
        return "".join(section_parts)

    def _merge_components(
        self,
        components: List[tuple[str, str]]
    ) -> tuple[str, Dict[str, int], bool]:
        """
        Merge components and handle truncation if needed.

        Args:
            components: List of (name, content) tuples

        Returns:
            (merged_content, component_sizes, truncated)
        """
        component_sizes = {}
        truncated = False

        # Calculate sizes
        for name, content in components:
            component_sizes[name] = len(content)

        total_size = sum(component_sizes.values())

        if total_size <= self.max_chars:
            # Everything fits
            merged = "".join(content for _, content in components)
            return merged, component_sizes, False

        # Need to truncate - prioritize metadata > markdown > code > latex
        logger.warning(
            f"Content exceeds limit ({total_size:,} > {self.max_chars:,} chars), "
            f"prioritizing components"
        )

        # Build with priority
        merged_parts = []
        remaining_chars = self.max_chars

        # Priority order
        priority_order = ["metadata", "markdown", "code", "latex"]
        component_dict = dict(components)

        for component_name in priority_order:
            if component_name not in component_dict:
                continue

            content = component_dict[component_name]
            content_size = len(content)

            if content_size <= remaining_chars:
                # Full component fits
                merged_parts.append(content)
                remaining_chars -= content_size
            else:
                # Partial component
                if remaining_chars > 1000:  # Only include if meaningful space left
                    truncation_marker = "\n\n[... Content truncated due to size limit ...]\n"
                    available = remaining_chars - len(truncation_marker)
                    merged_parts.append(content[:available] + truncation_marker)
                    component_sizes[component_name] = available
                else:
                    component_sizes[component_name] = 0

                truncated = True
                break

        merged = "".join(merged_parts)
        return merged, component_sizes, truncated
