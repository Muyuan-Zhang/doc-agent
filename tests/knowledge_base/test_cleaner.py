"""Unit tests for TextCleaner."""
from app.knowledge_base.cleaner import TextCleaner
from app.knowledge_base.parser import Section


class TestTextCleanerClean:
    def test_unicode_nfkc_normalization(self):
        # Fullwidth chars → ASCII
        assert TextCleaner().clean("ｈｅｌｌｏ") == "hello"

    def test_strips_control_characters(self):
        result = TextCleaner().clean("hello\x00world\x01")
        assert "\x00" not in result
        assert "\x01" not in result

    def test_preserves_newlines(self):
        result = TextCleaner().clean("line1\nline2")
        assert "\n" in result

    def test_collapses_tabs_to_spaces(self):
        result = TextCleaner().clean("col1\tcol2")
        assert result == "col1 col2"

    def test_collapses_multiple_spaces(self):
        result = TextCleaner().clean("too   many    spaces")
        assert "  " not in result

    def test_collapses_three_or_more_blank_lines(self):
        result = TextCleaner().clean("a\n\n\n\n\nb")
        assert "\n\n\n" not in result

    def test_strips_leading_and_trailing_whitespace(self):
        result = TextCleaner().clean("  hello  ")
        assert result == "hello"

    def test_empty_string_returns_empty(self):
        assert TextCleaner().clean("") == ""

    def test_only_whitespace_returns_empty(self):
        assert TextCleaner().clean("   \n  \t  ") == ""

    def test_clean_section_updates_content(self):
        section = Section(section_id="s0000", heading=None, content="  hello  ")
        cleaned = TextCleaner().clean_section(section)
        assert cleaned.content == "hello"

    def test_clean_section_updates_heading(self):
        section = Section(section_id="s0000", heading="  Title  ", content="body")
        cleaned = TextCleaner().clean_section(section)
        assert cleaned.heading == "Title"

    def test_clean_section_none_heading_stays_none(self):
        section = Section(section_id="s0000", heading=None, content="body")
        cleaned = TextCleaner().clean_section(section)
        assert cleaned.heading is None

    def test_clean_section_preserves_section_id(self):
        section = Section(section_id="p0003", heading=None, content="text")
        cleaned = TextCleaner().clean_section(section)
        assert cleaned.section_id == "p0003"

    def test_clean_section_returns_frozen_section(self):
        section = Section(section_id="s0000", heading=None, content="text")
        cleaned = TextCleaner().clean_section(section)
        import pytest
        with pytest.raises(Exception):
            cleaned.content = "mutated"  # type: ignore[misc]
