"""Unit tests for analytics API schemas."""

from __future__ import annotations

from app.api.schemas.analytics import (
    DailyStatEntrySchema,
    DailyStatsResponse,
    IntentEntrySchema,
    SatisfactionResponse,
    TopIntentsResponse,
)


class TestDailyStatsResponse:
    """Tests for DailyStatsResponse schema."""

    def test_empty_stats(self) -> None:
        """Empty stats list is valid."""
        resp = DailyStatsResponse(stats=[])
        assert resp.stats == []

    def test_with_entries(self) -> None:
        """Stats with entries serialize correctly."""
        entries = [
            DailyStatEntrySchema(date="2026-05-12", total_conversations=5, total_messages=10),
            DailyStatEntrySchema(date="2026-05-13", total_conversations=3, total_messages=7),
        ]
        resp = DailyStatsResponse(stats=entries)
        assert len(resp.stats) == 2
        assert resp.stats[0].date == "2026-05-12"
        assert resp.stats[0].total_conversations == 5
        assert resp.stats[1].total_messages == 7

    def test_serialization(self) -> None:
        """Response serializes to JSON-compatible dict."""
        resp = DailyStatsResponse(
            stats=[DailyStatEntrySchema(date="2026-05-13", total_conversations=1, total_messages=2)],
        )
        data = resp.model_dump()
        assert data["stats"][0]["date"] == "2026-05-13"
        assert data["stats"][0]["total_conversations"] == 1


class TestTopIntentsResponse:
    """Tests for TopIntentsResponse schema."""

    def test_empty_intents(self) -> None:
        """Empty intents list is valid."""
        resp = TopIntentsResponse(intents=[])
        assert resp.intents == []

    def test_with_entries(self) -> None:
        """Intents with entries serialize correctly."""
        intents = [
            IntentEntrySchema(name="shipping_policy.pdf", count=42),
            IntentEntrySchema(name="returns_guide.md", count=15),
        ]
        resp = TopIntentsResponse(intents=intents)
        assert len(resp.intents) == 2
        assert resp.intents[0].name == "shipping_policy.pdf"
        assert resp.intents[0].count == 42

    def test_serialization(self) -> None:
        """Response serializes to JSON-compatible dict."""
        resp = TopIntentsResponse(
            intents=[IntentEntrySchema(name="faq.pdf", count=10)],
        )
        data = resp.model_dump()
        assert data["intents"][0]["name"] == "faq.pdf"


class TestSatisfactionResponse:
    """Tests for SatisfactionResponse schema."""

    def test_zero_values(self) -> None:
        """All-zero response is valid (no feedback)."""
        resp = SatisfactionResponse(positive=0, negative=0, total=0, rate=0.0)
        assert resp.total == 0
        assert resp.rate == 0.0

    def test_with_data(self) -> None:
        """Response with real data."""
        resp = SatisfactionResponse(positive=80, negative=20, total=100, rate=0.8)
        assert resp.positive == 80
        assert resp.negative == 20
        assert resp.total == 100
        assert resp.rate == 0.8

    def test_serialization(self) -> None:
        """Response serializes to JSON-compatible dict."""
        resp = SatisfactionResponse(positive=5, negative=1, total=6, rate=0.8333)
        data = resp.model_dump()
        assert data["positive"] == 5
        assert data["rate"] == 0.8333
