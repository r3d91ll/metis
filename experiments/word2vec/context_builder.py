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
        Create a CombinedContextBuilder configured with a token budget.
        
        Parameters:
            max_tokens (int): Maximum number of tokens allowed for the combined context; used to compute a conservative character budget (max_chars = max_tokens * 4).
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
        Builds a unified text context from a paper document and optional code suitable for embedding generation.
        
        Assembles sections in priority order (metadata, markdown extraction, optional LaTeX, optional code), merges them into a single text blob trimmed to the builder's character limit, and records per-component sizes and whether truncation occurred.
        
        Parameters:
            paper_doc (dict): Paper record containing at least `arxiv_id` and `title`. May include `markdown_content` and `latex_source` for body text.
            code_doc (Optional[dict]): Optional code repository record; expected keys include `code_files` (mapping of path->content) and optional `repository_metadata` (may include `language`).
            include_latex (bool): If True, include `latex_source` from `paper_doc` when present.
        
        Returns:
            CombinedContext: Dataclass containing `arxiv_id`, `title`, merged `content`, per-component `components` character counts, `total_chars`, `total_tokens_estimate` (chars//4), `truncated` flag, and a `metadata` dict describing presence of code/LaTeX and code language.
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
        """
        Constructs a formatted metadata section for a paper.
        
        Parameters:
            paper_doc (Dict[str, any]): Document dictionary containing paper metadata. Expected keys:
                - "title" (str): Paper title (required).
                - "arxiv_id" (str): arXiv identifier (required).
                - "authors" (List[str], optional): List of author names; defaults to empty list.
                - "abstract" (str, optional): Abstract text; defaults to empty string.
        
        Returns:
            str: A markdown-formatted metadata section including the title as a top-level header,
                 arXiv ID, a comma-separated authors line, an "Abstract" subsection, and a trailing
                 delimiter line.
        """
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
        """
        Create a Markdown section titled "Paper Content (PDF Extraction)" containing the provided extracted paper text.
        
        Parameters:
            markdown_content (str): Extracted paper text in Markdown format (from PDF extraction).
        
        Returns:
            section (str): Formatted Markdown string with the section header, the provided content, and a trailing horizontal rule delimiter.
        """
        section = f"""
## Paper Content (PDF Extraction)

{markdown_content}

---
"""
        return section

    def _build_latex_section(self, latex_content: str) -> str:
        """
        Builds a titled LaTeX section containing the provided LaTeX source inside a fenced code block labeled `latex`, followed by a section delimiter.
        
        Parameters:
            latex_content (str): Raw LaTeX source to include.
        
        Returns:
            section (str): Formatted Markdown section with the LaTeX code block and trailing delimiter.
        """
        section = f"""
## LaTeX Source (Mathematical Formulas & Structure)

```latex
{latex_content}
```

---
"""
        return section

    def _build_code_section(self, code_doc: Dict[str, any]) -> str:
        """
        Assembles a formatted "Code Repository" markdown section from repository metadata and files.
        
        Parameters:
            code_doc (Dict[str, any]): Repository information with optional keys:
                - "github_url": repository URL (string).
                - "code_files": mapping of file path to file content (dict).
                - "repository_metadata": dict that may contain "language" for the repo.
        
        Returns:
            str: A markdown string containing a repository header (URL, language, file count)
            and a sequential listing of each file as a fenced code block. Code fence language
            hints are chosen from the file extension when available, otherwise from the
            repository language, and the section is terminated with a separator.
        """
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
        Merge named text components into a single string constrained to the builder's max_chars, applying prioritization and truncation when necessary.
        
        Parameters:
            components (List[tuple[str, str]]): Ordered list of (name, content) tuples representing sections to include.
        
        Returns:
            tuple[str, Dict[str, int], bool]:
                - merged_content: The assembled text containing fully included components and, if space required, a partially included component followed by a truncation marker.
                - component_sizes: Mapping from component name to the number of characters from that component that were included in merged_content (0 if excluded). For a partially included component, this value equals the number of original content characters included (the truncation marker is not counted).
                - truncated: `True` if any component was partially or fully omitted due to the size limit, `False` if all components fit within max_chars.
        
        Notes:
            When content exceeds the size limit, components are included according to the priority order: "metadata", "markdown", "code", "latex". If a component is only partially included, a truncation marker is appended to the merged content. Components with very small remaining space may be excluded entirely (counted as 0).
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