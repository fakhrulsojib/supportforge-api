"""Tests for the markdown structure-aware parser."""

from __future__ import annotations

from app.rag.markdown_parser import parse_markdown_sections


class TestParseMarkdownSections:
    """Test suite for parse_markdown_sections."""

    def test_empty_string_returns_empty(self) -> None:
        assert parse_markdown_sections("") == []

    def test_whitespace_only_returns_empty(self) -> None:
        assert parse_markdown_sections("   \n\n   ") == []

    def test_no_headings_returns_single_section(self) -> None:
        text = "Just some plain text without any headings.\nAnother line."
        sections = parse_markdown_sections(text)
        assert len(sections) == 1
        assert sections[0].heading_path == ""
        assert "Just some plain text" in sections[0].content

    def test_single_h1_heading(self) -> None:
        text = "# My Title\n\nSome content here."
        sections = parse_markdown_sections(text)
        assert len(sections) == 1
        assert sections[0].heading_path == "My Title"
        assert "# My Title" in sections[0].content
        assert "Some content here." in sections[0].content

    def test_multiple_h2_sections(self) -> None:
        text = (
            "# Main Title\n\nIntro text.\n\n"
            "## Section One\n\nFirst section content.\n\n"
            "## Section Two\n\nSecond section content."
        )
        sections = parse_markdown_sections(text)
        assert len(sections) == 3

        assert sections[0].heading_path == "Main Title"
        assert "Intro text." in sections[0].content

        assert sections[1].heading_path == "Main Title > Section One"
        assert "First section content." in sections[1].content

        assert sections[2].heading_path == "Main Title > Section Two"
        assert "Second section content." in sections[2].content

    def test_nested_heading_hierarchy(self) -> None:
        text = (
            "# Top\n\n"
            "## Middle\n\n"
            "### Deep\n\nDeep content.\n\n"
            "## Back to Middle\n\nMiddle content."
        )
        sections = parse_markdown_sections(text)

        # Find the deep section
        deep = [s for s in sections if "Deep" in s.heading_path and "content" in s.content]
        assert len(deep) == 1
        assert deep[0].heading_path == "Top > Middle > Deep"

        # "Back to Middle" should NOT include "Deep" in its path
        back = [s for s in sections if "Back to Middle" in s.heading_path]
        assert len(back) == 1
        assert back[0].heading_path == "Top > Back to Middle"
        assert "Deep" not in back[0].heading_path

    def test_content_before_first_heading(self) -> None:
        text = "Preamble text here.\n\n# First Heading\n\nHeading content."
        sections = parse_markdown_sections(text)
        assert len(sections) == 2
        assert sections[0].heading_path == ""
        assert "Preamble" in sections[0].content
        assert sections[1].heading_path == "First Heading"

    def test_code_fence_not_split(self) -> None:
        """Code blocks should not be interpreted as headings."""
        text = (
            "# Config Section\n\n"
            "```python\n"
            "# This is a comment, not a heading\n"
            "x = 42\n"
            "```\n\n"
            "After code."
        )
        sections = parse_markdown_sections(text)
        # Should be a single section — the # inside code is NOT a heading
        assert len(sections) == 1
        assert sections[0].heading_path == "Config Section"
        assert "# This is a comment" in sections[0].content
        assert "x = 42" in sections[0].content

    def test_tilde_code_fence(self) -> None:
        """Tilde-style code fences should also be preserved."""
        text = (
            "# Section\n\n"
            "~~~\n"
            "# not a heading\n"
            "~~~\n"
        )
        sections = parse_markdown_sections(text)
        assert len(sections) == 1
        assert "# not a heading" in sections[0].content

    def test_table_kept_intact(self) -> None:
        """Tables should stay within their section."""
        text = (
            "## Shipping Methods\n\n"
            "| Method | Cost |\n"
            "|--------|------|\n"
            "| Standard | $5.99 |\n"
            "| Express | $12.99 |\n\n"
            "## Returns\n\nReturn info."
        )
        sections = parse_markdown_sections(text)
        shipping = [s for s in sections if "Shipping" in s.heading_path]
        assert len(shipping) == 1
        # All table rows should be in the same section
        assert "Standard" in shipping[0].content
        assert "Express" in shipping[0].content
        assert "$5.99" in shipping[0].content

    def test_heading_level_stored(self) -> None:
        text = "# H1\n\n## H2\n\n### H3\n\nContent."
        sections = parse_markdown_sections(text)
        levels = [s.heading_level for s in sections]
        assert 1 in levels
        assert 2 in levels
        assert 3 in levels

    def test_real_world_novamart_structure(self) -> None:
        """Simulate a realistic NovaMart knowledge base document."""
        text = (
            "# Shipping & Delivery Policy\n\n"
            "**Last Updated:** January 2025\n\n"
            "---\n\n"
            "## Shipping Methods\n\n"
            "### Standard Shipping\n"
            "- **Cost:** Free on orders over $50\n"
            "- **Delivery Time:** 5–7 business days\n\n"
            "### Express Shipping\n"
            "- **Cost:** $12.99\n"
            "- **Delivery Time:** 2–3 business days\n\n"
            "## Order Processing\n\n"
            "Orders are processed within 1–2 business days.\n\n"
            "### Cut-off Times\n"
            "| Method | Cut-off |\n"
            "|--------|--------|\n"
            "| Standard | 11:59 PM |\n"
            "| Express | 2:00 PM |\n"
        )
        sections = parse_markdown_sections(text)

        # Verify key heading paths
        paths = [s.heading_path for s in sections]
        assert "Shipping & Delivery Policy" in paths
        assert "Shipping & Delivery Policy > Shipping Methods > Standard Shipping" in paths
        assert "Shipping & Delivery Policy > Shipping Methods > Express Shipping" in paths
        assert "Shipping & Delivery Policy > Order Processing" in paths
        assert "Shipping & Delivery Policy > Order Processing > Cut-off Times" in paths

        # Verify the Cut-off Times section contains the table
        cutoff = [s for s in sections if "Cut-off Times" in s.heading_path]
        assert len(cutoff) == 1
        assert "Standard" in cutoff[0].content
        assert "11:59 PM" in cutoff[0].content

    def test_horizontal_rules_do_not_create_sections(self) -> None:
        """--- lines should not create new sections."""
        text = "# Title\n\nContent.\n\n---\n\nMore content."
        sections = parse_markdown_sections(text)
        assert len(sections) == 1
        assert "---" in sections[0].content

    def test_heading_in_section_content_preserved(self) -> None:
        """The heading line itself should be included in the section content."""
        text = "## My Section\n\nBody text."
        sections = parse_markdown_sections(text)
        assert len(sections) == 1
        assert "## My Section" in sections[0].content
        assert "Body text." in sections[0].content


class TestMarkdownParserEdgeCases:
    """Edge cases and boundary conditions for the markdown parser."""

    def test_heading_only_no_body(self) -> None:
        """A heading with no body content should still produce a section."""
        text = "# Just A Heading"
        sections = parse_markdown_sections(text)
        assert len(sections) == 1
        assert sections[0].heading_path == "Just A Heading"
        assert "# Just A Heading" in sections[0].content

    def test_consecutive_headings_no_content_between(self) -> None:
        """Back-to-back headings with no body text between them."""
        text = "# Title\n## Section A\n## Section B\n\nContent in B."
        sections = parse_markdown_sections(text)
        paths = [s.heading_path for s in sections]
        assert "Title" in paths
        assert "Title > Section A" in paths
        assert "Title > Section B" in paths
        # Section B should have the content
        b_section = [s for s in sections if s.heading_path == "Title > Section B"]
        assert len(b_section) == 1
        assert "Content in B." in b_section[0].content

    def test_deep_headings_h4_h5_h6(self) -> None:
        """Parser should handle headings up to H6."""
        text = (
            "# L1\n\n"
            "## L2\n\n"
            "### L3\n\n"
            "#### L4\n\n"
            "##### L5\n\n"
            "###### L6\n\nDeepest content."
        )
        sections = parse_markdown_sections(text)
        deepest = [s for s in sections if "L6" in s.heading_path]
        assert len(deepest) == 1
        assert deepest[0].heading_path == "L1 > L2 > L3 > L4 > L5 > L6"
        assert deepest[0].heading_level == 6

    def test_fake_heading_no_space_after_hash(self) -> None:
        """#NoSpace should NOT be treated as a heading (no space after #)."""
        text = "#NoSpace\n\nSome text."
        sections = parse_markdown_sections(text)
        assert len(sections) == 1
        assert sections[0].heading_path == ""
        assert "#NoSpace" in sections[0].content

    def test_unclosed_code_fence(self) -> None:
        """An unclosed code fence should not break parsing — treat rest as code."""
        text = (
            "# Section\n\n"
            "```\n"
            "# inside code\n"
            "some code\n"
            "# also inside code\n"
        )
        sections = parse_markdown_sections(text)
        # Everything after the opening ``` is inside the fence
        assert len(sections) == 1
        assert "# inside code" in sections[0].content
        assert "# also inside code" in sections[0].content

    def test_nested_code_fences_different_markers(self) -> None:
        """Backtick fence inside a tilde fence should not confuse parser."""
        text = (
            "# Doc\n\n"
            "~~~\n"
            "```\n"
            "# not a heading\n"
            "```\n"
            "~~~\n\n"
            "## After Code\n\nPost-code content."
        )
        sections = parse_markdown_sections(text)
        # Should have two sections: Doc and After Code
        paths = [s.heading_path for s in sections]
        assert "Doc" in paths
        assert "Doc > After Code" in paths

    def test_empty_section_between_headings(self) -> None:
        """A heading followed immediately by another heading produces a minimal section."""
        text = "## First\n## Second\n\nContent."
        sections = parse_markdown_sections(text)
        # First has only its heading line, Second has content
        assert any(s.heading_path == "First" for s in sections)
        second = [s for s in sections if s.heading_path == "Second"]
        assert len(second) == 1
        assert "Content." in second[0].content

    def test_heading_with_trailing_whitespace(self) -> None:
        """Headings with trailing spaces should still be parsed correctly."""
        text = "# Title With Spaces   \n\nBody."
        sections = parse_markdown_sections(text)
        assert len(sections) == 1
        assert sections[0].heading_path == "Title With Spaces"

    def test_multiple_tables_in_one_section(self) -> None:
        """Multiple tables in the same section should both stay intact."""
        text = (
            "## Pricing\n\n"
            "| Plan | Price |\n|------|-------|\n| Free | $0 |\n| Pro | $10 |\n\n"
            "Some text between tables.\n\n"
            "| Feature | Free | Pro |\n|---------|------|-----|\n| Storage | 1GB | 100GB |\n"
        )
        sections = parse_markdown_sections(text)
        assert len(sections) == 1
        assert "$0" in sections[0].content
        assert "100GB" in sections[0].content

    def test_bullet_list_only_section(self) -> None:
        """A section containing only a bullet list."""
        text = "## Items\n\n- Item A\n- Item B\n- Item C"
        sections = parse_markdown_sections(text)
        assert len(sections) == 1
        assert "- Item A" in sections[0].content
        assert "- Item C" in sections[0].content

    def test_heading_level_skip_h1_to_h3(self) -> None:
        """Skipping heading levels (H1 → H3, no H2) should still build correct path."""
        text = "# Top\n\n### Skipped to H3\n\nContent."
        sections = parse_markdown_sections(text)
        h3 = [s for s in sections if "Skipped" in s.heading_path]
        assert len(h3) == 1
        # Path should be Top > Skipped to H3 (no phantom H2)
        assert h3[0].heading_path == "Top > Skipped to H3"

    def test_faq_style_qa_blocks(self) -> None:
        """FAQ-style Q/A blocks should stay within their section."""
        text = (
            "## FAQ\n\n"
            "**Q: Can I return an item?**\n"
            "A: Yes, within 30 days.\n\n"
            "**Q: What about gifts?**\n"
            "A: Gift returns get store credit.\n"
        )
        sections = parse_markdown_sections(text)
        assert len(sections) == 1
        assert "Can I return" in sections[0].content
        assert "store credit" in sections[0].content

    def test_seven_hashes_not_a_heading(self) -> None:
        """####### (7 hashes) is NOT a valid heading in markdown."""
        text = "####### Not a heading\n\nSome text."
        sections = parse_markdown_sections(text)
        assert len(sections) == 1
        assert sections[0].heading_path == ""

    def test_heading_path_resets_correctly_on_same_level(self) -> None:
        """Two H2 sections under H1: second H2 should not inherit first H2's children."""
        text = (
            "# Root\n\n"
            "## Branch A\n\n"
            "### Leaf A1\n\nA1 content.\n\n"
            "## Branch B\n\nB content."
        )
        sections = parse_markdown_sections(text)
        branch_b = [s for s in sections if s.heading_path == "Root > Branch B"]
        assert len(branch_b) == 1
        # Branch B should NOT have "Leaf A1" in its path
        assert "Leaf" not in branch_b[0].heading_path

