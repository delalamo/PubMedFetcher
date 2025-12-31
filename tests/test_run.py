"""Tests for the run module."""

from datetime import datetime

import pytest


class TestFormatDate:
    """Tests for the format_date function."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Import format_date for each test."""
        from run import format_date

        self.format_date = format_date

    def test_format_date_default_separator(self):
        """Test date formatting with default separator."""
        date = datetime(2024, 3, 15)
        result = self.format_date(date)
        assert result == "2024/03/15"

    def test_format_date_dash_separator(self):
        """Test date formatting with dash separator."""
        date = datetime(2024, 3, 15)
        result = self.format_date(date, sep="-")
        assert result == "2024-03-15"

    def test_format_date_single_digit_month_day(self):
        """Test that single-digit months and days are zero-padded."""
        date = datetime(2024, 1, 5)
        result = self.format_date(date)
        assert result == "2024/01/05"

    def test_format_date_invalid_separator_length(self):
        """Test that multi-character separator raises assertion error."""
        date = datetime(2024, 3, 15)
        with pytest.raises(AssertionError):
            self.format_date(date, sep="--")


class TestOpenAISummaryPrompt:
    """Tests for the openai_summary_prompt function."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Import openai_summary_prompt for each test."""
        from run import openai_summary_prompt

        self.openai_summary_prompt = openai_summary_prompt

    def test_prompt_is_string(self):
        """Test that the prompt is a non-empty string."""
        prompt = self.openai_summary_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_prompt_contains_key_instructions(self):
        """Test that the prompt contains key instruction elements."""
        prompt = self.openai_summary_prompt()
        assert "summarize" in prompt.lower()
        assert "abstract" in prompt.lower()
        assert "sentence" in prompt.lower()
