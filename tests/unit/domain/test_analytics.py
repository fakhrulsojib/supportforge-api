"""Unit tests for analytics domain models."""

from __future__ import annotations

import pytest

from app.domain.models.analytics import DailyStatEntry, IntentEntry, SatisfactionSummary


class TestDailyStatEntry:
    """Tests for DailyStatEntry domain model."""

    def test_create_with_required_fields(self) -> None:
        """Create with only required date field."""
        entry = DailyStatEntry(date="2026-05-13")
        assert entry.date == "2026-05-13"
        assert entry.total_conversations == 0
        assert entry.total_messages == 0

    def test_create_with_all_fields(self) -> None:
        """Create with all fields populated."""
        entry = DailyStatEntry(
            date="2026-05-13",
            total_conversations=42,
            total_messages=128,
        )
        assert entry.date == "2026-05-13"
        assert entry.total_conversations == 42
        assert entry.total_messages == 128

    def test_negative_counts_rejected(self) -> None:
        """Negative counts should be rejected by ge=0 constraint."""
        with pytest.raises(Exception):  # noqa: B017
            DailyStatEntry(date="2026-05-13", total_conversations=-1)

    def test_zero_counts_valid(self) -> None:
        """Zero counts are valid (empty day)."""
        entry = DailyStatEntry(date="2026-05-13", total_conversations=0, total_messages=0)
        assert entry.total_conversations == 0
        assert entry.total_messages == 0


class TestIntentEntry:
    """Tests for IntentEntry domain model."""

    def test_create_with_required_fields(self) -> None:
        """Create with name and default count."""
        entry = IntentEntry(name="shipping_policy.pdf")
        assert entry.name == "shipping_policy.pdf"
        assert entry.count == 0

    def test_create_with_all_fields(self) -> None:
        """Create with name and count."""
        entry = IntentEntry(name="returns_guide.md", count=25)
        assert entry.name == "returns_guide.md"
        assert entry.count == 25

    def test_empty_name_rejected(self) -> None:
        """Empty name should be rejected by min_length=1."""
        with pytest.raises(Exception):  # noqa: B017
            IntentEntry(name="")

    def test_negative_count_rejected(self) -> None:
        """Negative count should be rejected."""
        with pytest.raises(Exception):  # noqa: B017
            IntentEntry(name="test", count=-1)


class TestSatisfactionSummary:
    """Tests for SatisfactionSummary domain model."""

    def test_create_defaults(self) -> None:
        """All fields default to zero."""
        summary = SatisfactionSummary()
        assert summary.positive == 0
        assert summary.negative == 0
        assert summary.total == 0
        assert summary.rate == 0.0

    def test_create_with_all_fields(self) -> None:
        """Create with computed rate."""
        summary = SatisfactionSummary(
            positive=80,
            negative=20,
            total=100,
            rate=0.8,
        )
        assert summary.positive == 80
        assert summary.negative == 20
        assert summary.total == 100
        assert summary.rate == 0.8

    def test_rate_boundary_zero(self) -> None:
        """Rate of 0.0 is valid (all negative)."""
        summary = SatisfactionSummary(positive=0, negative=10, total=10, rate=0.0)
        assert summary.rate == 0.0

    def test_rate_boundary_one(self) -> None:
        """Rate of 1.0 is valid (all positive)."""
        summary = SatisfactionSummary(positive=10, negative=0, total=10, rate=1.0)
        assert summary.rate == 1.0

    def test_rate_exceeding_one_rejected(self) -> None:
        """Rate > 1.0 should be rejected by le=1.0."""
        with pytest.raises(Exception):  # noqa: B017
            SatisfactionSummary(rate=1.5)

    def test_negative_rate_rejected(self) -> None:
        """Negative rate should be rejected by ge=0.0."""
        with pytest.raises(Exception):  # noqa: B017
            SatisfactionSummary(rate=-0.1)

    def test_negative_counts_rejected(self) -> None:
        """Negative counts should be rejected."""
        with pytest.raises(Exception):  # noqa: B017
            SatisfactionSummary(positive=-1)
