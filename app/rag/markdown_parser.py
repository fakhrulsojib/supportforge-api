"""Markdown-aware document parser for structure-preserving chunking.

Parses markdown documents into logical sections based on heading
hierarchy, keeping atomic blocks (tables, code fences) intact.
Each section carries its heading path as metadata, enabling
structure-aware retrieval.

Only used for ``.md`` files — other file types bypass this parser
and go directly to the RecursiveChunker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MarkdownSection:
    """A logical section of a markdown document.

    Attributes:
        content: The text content of this section (including heading line).
        heading_path: Hierarchical heading path, e.g.
            ``"Shipping Policy > Delivery Issues > Damaged Package"``.
        heading_level: Depth of the section heading (1 for ``#``, 2 for ``##``, etc.).
    """

    content: str
    heading_path: str = ""
    heading_level: int = 0
    metadata: dict[str, object] = field(default_factory=dict)


# Regex to match markdown headings: captures level (# count) and title text
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")

# Regex to detect the start/end of a fenced code block
_CODE_FENCE_RE = re.compile(r"^(`{3,}|~{3,})")


def parse_markdown_sections(text: str) -> list[MarkdownSection]:
    """Parse a markdown document into sections based on headings.

    Rules:
        1. Split at heading boundaries (``#``, ``##``, ``###``, etc.)
        2. Each section includes its heading text prepended to the content
        3. Fenced code blocks are kept intact (never split mid-block)
        4. Tables are kept intact (consecutive ``|``-prefixed lines)
        5. Content before the first heading becomes a section with no heading path

    Args:
        text: Raw markdown text.

    Returns:
        List of MarkdownSection objects, each with content and heading_path.
        If the document has no headings, returns a single section with all content.
    """
    if not text or not text.strip():
        return []

    lines = text.split("\n")

    # Track current heading hierarchy: level -> heading text
    heading_stack: dict[int, str] = {}
    sections: list[MarkdownSection] = []

    current_lines: list[str] = []
    current_heading_path = ""
    current_heading_level = 0

    in_code_fence = False
    code_fence_marker = ""

    for line in lines:
        # Track code fence state (don't interpret headings inside code blocks)
        fence_match = _CODE_FENCE_RE.match(line)
        if fence_match:
            if not in_code_fence:
                in_code_fence = True
                code_fence_marker = fence_match.group(1)[0]  # ` or ~
            elif line.strip().startswith(code_fence_marker):
                in_code_fence = False
                code_fence_marker = ""
            current_lines.append(line)
            continue

        if in_code_fence:
            current_lines.append(line)
            continue

        # Check for heading
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            # Flush the current section
            section_text = "\n".join(current_lines).strip()
            if section_text:
                sections.append(MarkdownSection(
                    content=section_text,
                    heading_path=current_heading_path,
                    heading_level=current_heading_level,
                ))

            # Update heading hierarchy
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()

            # Clear any deeper headings from the stack
            heading_stack = {k: v for k, v in heading_stack.items() if k < level}
            heading_stack[level] = title

            # Build the full heading path
            current_heading_path = " > ".join(
                heading_stack[k] for k in sorted(heading_stack)
            )
            current_heading_level = level

            # Start new section with the heading line included
            current_lines = [line]
        else:
            current_lines.append(line)

    # Flush the last section
    section_text = "\n".join(current_lines).strip()
    if section_text:
        sections.append(MarkdownSection(
            content=section_text,
            heading_path=current_heading_path,
            heading_level=current_heading_level,
        ))

    logger.debug(
        "markdown_parsed",
        section_count=len(sections),
        headings=[s.heading_path for s in sections if s.heading_path],
    )

    return sections
