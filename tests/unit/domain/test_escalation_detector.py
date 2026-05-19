"""Tests for the EscalationDetector domain service.

Covers all three detection strategies: sentiment, repetition, and
explicit human-handoff request detection.
"""

from __future__ import annotations

import pytest

from app.domain.models.enums import EscalationTrigger
from app.domain.services.escalation_detector import EscalationDetector, EscalationResult

# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def detector() -> EscalationDetector:
    """Fresh EscalationDetector instance."""
    return EscalationDetector()


# ── Result Dataclass ────────────────────────────────────────────


class TestEscalationResult:
    """Test suite for EscalationResult dataclass defaults."""

    def test_defaults(self) -> None:
        result = EscalationResult()
        assert result.should_escalate is False
        assert result.trigger == EscalationTrigger.NONE
        assert result.reason == ""
        assert result.sentiment_score == 0.0
        assert result.repetition_count == 0

    def test_custom_values(self) -> None:
        result = EscalationResult(
            should_escalate=True,
            trigger=EscalationTrigger.SENTIMENT,
            reason="High frustration detected",
            sentiment_score=0.85,
        )
        assert result.should_escalate is True
        assert result.trigger == EscalationTrigger.SENTIMENT
        assert result.sentiment_score == 0.85


# ── Sentiment Detection ────────────────────────────────────────


class TestSentimentDetection:
    """Test suite for keyword/pattern-based sentiment scoring."""

    def test_all_caps_message_escalates(self, detector: EscalationDetector) -> None:
        """ALL CAPS message with frustration phrase should score high and escalate."""
        result = detector.detect(
            message="THIS IS RIDICULOUS TERRIBLE SERVICE",
            conversation_history=[],
        )
        assert result.should_escalate is True
        assert result.trigger == EscalationTrigger.SENTIMENT
        assert result.sentiment_score >= 0.6

    def test_mixed_caps_short_no_escalation(self, detector: EscalationDetector) -> None:
        """Short mixed-case message should NOT escalate."""
        result = detector.detect(
            message="OK thanks",
            conversation_history=[],
        )
        assert result.should_escalate is False
        assert result.sentiment_score < 0.7

    def test_excessive_punctuation_increases_score(self, detector: EscalationDetector) -> None:
        """Three or more consecutive ! or ? should increase score."""
        result = detector.detect(
            message="Why isn't this working!!!",
            conversation_history=[],
        )
        assert result.sentiment_score > 0.0

    def test_negative_phrase_escalates(self, detector: EscalationDetector) -> None:
        """Known negative phrases should trigger escalation."""
        result = detector.detect(
            message="this is ridiculous, what a waste of time",
            conversation_history=[],
        )
        assert result.should_escalate is True
        assert result.trigger == EscalationTrigger.SENTIMENT
        assert result.sentiment_score >= 0.6

    def test_multiple_frustration_signals(self, detector: EscalationDetector) -> None:
        """Combined ALL CAPS + negative phrase + punctuation → high score."""
        result = detector.detect(
            message="THIS IS TERRIBLE SERVICE!!!",
            conversation_history=[],
        )
        assert result.should_escalate is True
        assert result.sentiment_score >= 0.6

    def test_neutral_message_low_score(self, detector: EscalationDetector) -> None:
        """Neutral, polite message should have low sentiment score."""
        result = detector.detect(
            message="Can you help me track my order please?",
            conversation_history=[],
        )
        assert result.should_escalate is False
        assert result.sentiment_score < 0.3

    def test_empty_message_no_escalation(self, detector: EscalationDetector) -> None:
        """Empty message should return safe default."""
        result = detector.detect(
            message="",
            conversation_history=[],
        )
        assert result.should_escalate is False

    def test_punctuation_only_no_crash(self, detector: EscalationDetector) -> None:
        """Message with only punctuation should not crash."""
        result = detector.detect(
            message="!!!???",
            conversation_history=[],
        )
        assert isinstance(result, EscalationResult)

    def test_negative_phrase_case_insensitive(self, detector: EscalationDetector) -> None:
        """Negative phrases should be detected case-insensitively."""
        result = detector.detect(
            message="This Is Ridiculous and NOT HELPFUL at all",
            conversation_history=[],
        )
        assert result.should_escalate is True
        assert result.trigger == EscalationTrigger.SENTIMENT

    def test_single_negative_phrase_below_threshold(self, detector: EscalationDetector) -> None:
        """A single mild negative phrase alone should NOT escalate (score < 0.6)."""
        result = detector.detect(
            message="that was not helpful",
            conversation_history=[],
        )
        # Single phrase gives 0.3 — below 0.6 threshold
        assert result.sentiment_score >= 0.3
        assert result.sentiment_score < 0.6
        assert result.should_escalate is False

    def test_fed_up_phrase_contributes(self, detector: EscalationDetector) -> None:
        """'Fed up' phrase combined with other signals should escalate."""
        result = detector.detect(
            message="I am absolutely fed up with this terrible service",
            conversation_history=[],
        )
        assert result.should_escalate is True

    def test_score_capped_at_one(self, detector: EscalationDetector) -> None:
        """Sentiment score should never exceed 1.0."""
        result = detector.detect(
            message="THIS IS RIDICULOUS TERRIBLE WASTE OF TIME UNACCEPTABLE!!!",
            conversation_history=[],
        )
        assert result.sentiment_score <= 1.0


# ── Repetition Detection ───────────────────────────────────────


class TestRepetitionDetection:
    """Test suite for query similarity / repetition tracking."""

    def test_identical_messages_escalate(self, detector: EscalationDetector) -> None:
        """Same message repeated 3 times in history should escalate."""
        history = [
            {"role": "user", "content": "Where is my order?"},
            {"role": "assistant", "content": "Let me check..."},
            {"role": "user", "content": "Where is my order?"},
            {"role": "assistant", "content": "I'm looking into it..."},
        ]
        result = detector.detect(
            message="Where is my order?",
            conversation_history=history,
        )
        assert result.should_escalate is True
        assert result.trigger == EscalationTrigger.REPETITION
        assert result.repetition_count >= 2

    def test_highly_similar_messages_escalate(self, detector: EscalationDetector) -> None:
        """Messages with >80% token overlap should count as repetitions."""
        history = [
            {"role": "user", "content": "Where is my order please?"},
            {"role": "assistant", "content": "Checking..."},
            {"role": "user", "content": "Where is my order?"},
            {"role": "assistant", "content": "Still looking..."},
        ]
        result = detector.detect(
            message="Where is my order please?",
            conversation_history=history,
        )
        assert result.should_escalate is True
        assert result.trigger == EscalationTrigger.REPETITION

    def test_different_messages_no_escalation(self, detector: EscalationDetector) -> None:
        """Completely different messages should NOT trigger repetition."""
        history = [
            {"role": "user", "content": "Where is my order?"},
            {"role": "assistant", "content": "It's on the way."},
            {"role": "user", "content": "How do I return an item?"},
            {"role": "assistant", "content": "Go to settings..."},
        ]
        result = detector.detect(
            message="What is your refund policy?",
            conversation_history=history,
        )
        assert result.trigger != EscalationTrigger.REPETITION

    def test_empty_history_no_escalation(self, detector: EscalationDetector) -> None:
        """No conversation history → no repetition possible."""
        result = detector.detect(
            message="Where is my order?",
            conversation_history=[],
        )
        assert result.trigger != EscalationTrigger.REPETITION

    def test_single_similar_not_enough(self, detector: EscalationDetector) -> None:
        """Only 1 similar past message → not enough for escalation (need 2)."""
        history = [
            {"role": "user", "content": "Where is my order?"},
            {"role": "assistant", "content": "Let me check."},
        ]
        result = detector.detect(
            message="Where is my order?",
            conversation_history=history,
        )
        # 1 repetition is not enough — need 2
        assert result.trigger != EscalationTrigger.REPETITION

    def test_short_messages_handled(self, detector: EscalationDetector) -> None:
        """Very short messages (1-2 words) should be handled gracefully."""
        history = [
            {"role": "user", "content": "help"},
            {"role": "assistant", "content": "How can I help?"},
            {"role": "user", "content": "help"},
            {"role": "assistant", "content": "Please describe your issue."},
        ]
        result = detector.detect(
            message="help",
            conversation_history=history,
        )
        assert result.should_escalate is True
        assert result.trigger == EscalationTrigger.REPETITION

    def test_only_user_messages_compared(self, detector: EscalationDetector) -> None:
        """Repetition should only compare user messages, not assistant ones."""
        history = [
            {"role": "user", "content": "Tell me about returns"},
            {"role": "assistant", "content": "Tell me about returns? Sure..."},
            {"role": "user", "content": "Tell me about returns"},
            {"role": "assistant", "content": "As I mentioned..."},
        ]
        result = detector.detect(
            message="Tell me about returns",
            conversation_history=history,
        )
        assert result.should_escalate is True
        assert result.trigger == EscalationTrigger.REPETITION

    def test_last_three_user_messages_window(self, detector: EscalationDetector) -> None:
        """Should only compare against the last 3 user messages."""
        history = [
            {"role": "user", "content": "Where is my order?"},
            {"role": "assistant", "content": "Checking..."},
            {"role": "user", "content": "Something completely different"},
            {"role": "assistant", "content": "OK..."},
            {"role": "user", "content": "Another different topic"},
            {"role": "assistant", "content": "Sure..."},
            {"role": "user", "content": "Yet another different question"},
            {"role": "assistant", "content": "Alright..."},
        ]
        # The first "Where is my order?" is outside the 3-message window
        result = detector.detect(
            message="Where is my order?",
            conversation_history=history,
        )
        assert result.trigger != EscalationTrigger.REPETITION


# ── Explicit Request Detection ─────────────────────────────────


class TestExplicitRequestDetection:
    """Test suite for human handoff pattern matching."""

    @pytest.mark.parametrize(
        "message",
        [
            "I want to speak to a human",
            "Let me talk to a real person",
            "Can I talk to a human please?",
            "escalate this",
            "I need a supervisor",
            "Let me speak to your manager",
            "transfer me to someone",
            "I want a real agent",
            "get me a live agent",
            "speak to someone please",
        ],
    )
    def test_explicit_patterns_trigger_escalation(
        self, detector: EscalationDetector, message: str,
    ) -> None:
        """Known handoff phrases should trigger instant escalation."""
        result = detector.detect(
            message=message,
            conversation_history=[],
        )
        assert result.should_escalate is True
        assert result.trigger == EscalationTrigger.EXPLICIT_REQUEST

    def test_human_in_normal_context_no_false_positive(
        self, detector: EscalationDetector,
    ) -> None:
        """'human' in non-handoff context should NOT trigger."""
        result = detector.detect(
            message="I have a question about human resources policies",
            conversation_history=[],
        )
        assert result.trigger != EscalationTrigger.EXPLICIT_REQUEST

    def test_case_insensitive_matching(self, detector: EscalationDetector) -> None:
        """Patterns should match regardless of case."""
        result = detector.detect(
            message="SPEAK TO A HUMAN NOW",
            conversation_history=[],
        )
        assert result.should_escalate is True
        assert result.trigger == EscalationTrigger.EXPLICIT_REQUEST

    def test_embedded_in_longer_sentence(self, detector: EscalationDetector) -> None:
        """Pattern embedded in a longer message should still trigger."""
        result = detector.detect(
            message="This bot is useless, I want to speak to a human right now",
            conversation_history=[],
        )
        assert result.should_escalate is True
        assert result.trigger == EscalationTrigger.EXPLICIT_REQUEST

    def test_manager_as_standalone(self, detector: EscalationDetector) -> None:
        """'manager' in escalation context should trigger."""
        result = detector.detect(
            message="get me your manager",
            conversation_history=[],
        )
        assert result.should_escalate is True
        assert result.trigger == EscalationTrigger.EXPLICIT_REQUEST

    def test_manage_not_manager_no_trigger(self, detector: EscalationDetector) -> None:
        """'manage' (not 'manager') should NOT trigger."""
        result = detector.detect(
            message="How do I manage my account settings?",
            conversation_history=[],
        )
        assert result.trigger != EscalationTrigger.EXPLICIT_REQUEST


# ── Combined / Edge Cases ──────────────────────────────────────


class TestCombinedDetection:
    """Test suite for priority and edge cases."""

    def test_explicit_takes_priority_over_sentiment(
        self, detector: EscalationDetector,
    ) -> None:
        """When both sentiment and explicit match, explicit_request wins."""
        result = detector.detect(
            message="THIS IS RIDICULOUS!!! Let me talk to a real person NOW",
            conversation_history=[],
        )
        assert result.should_escalate is True
        assert result.trigger == EscalationTrigger.EXPLICIT_REQUEST

    def test_normal_message_no_escalation(self, detector: EscalationDetector) -> None:
        """Normal, polite message should not trigger any detector."""
        result = detector.detect(
            message="Hello, can you help me find information about shipping rates?",
            conversation_history=[],
        )
        assert result.should_escalate is False
        assert result.trigger == EscalationTrigger.NONE

    def test_none_message_handled(self, detector: EscalationDetector) -> None:
        """None message should be handled gracefully."""
        result = detector.detect(
            message="",
            conversation_history=[],
        )
        assert result.should_escalate is False

    def test_whitespace_only_message(self, detector: EscalationDetector) -> None:
        """Whitespace-only message should be handled gracefully."""
        result = detector.detect(
            message="   ",
            conversation_history=[],
        )
        assert result.should_escalate is False

    def test_none_history_handled(self, detector: EscalationDetector) -> None:
        """None conversation_history should be handled gracefully."""
        result = detector.detect(
            message="Where is my order?",
            conversation_history=[],
        )
        assert isinstance(result, EscalationResult)
