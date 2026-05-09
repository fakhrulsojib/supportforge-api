"""Escalation detection service — smart human-handoff triggers.

Detects frustrated customers, repeated questions, and explicit
human-handoff requests to trigger escalation before or alongside
the RAG pipeline.

**Pure domain service** — ZERO framework imports.  All detection
is keyword/regex-based and deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.domain.models.enums import EscalationTrigger


# ── Result Types ────────────────────────────────────────────────


@dataclass
class EscalationResult:
    """Outcome of escalation detection.

    Attributes:
        should_escalate: Whether escalation was triggered.
        trigger: Which detection method fired.
        reason: Human-readable reason for escalation.
        sentiment_score: Frustration score (0.0–1.0).
        repetition_count: How many repeated messages were found.
    """

    should_escalate: bool = False
    trigger: EscalationTrigger = EscalationTrigger.NONE
    reason: str = ""
    sentiment_score: float = 0.0
    repetition_count: int = 0


# ── Constants ───────────────────────────────────────────────────

# Minimum alpha characters required for ALL CAPS detection
_CAPS_MIN_ALPHA = 5

# Fraction of alpha chars that must be uppercase for CAPS signal
_CAPS_THRESHOLD = 0.6

# Caps contribution to sentiment score
_CAPS_SCORE = 0.5

# Excessive punctuation contribution
_PUNCT_SCORE = 0.3

# Per-phrase contribution (capped at 1.0 total)
_PHRASE_SCORE = 0.3

# Sentiment threshold to trigger escalation
_SENTIMENT_THRESHOLD = 0.6

# Jaccard similarity threshold for repetition
_SIMILARITY_THRESHOLD = 0.8

# Number of similar past messages required before escalation
# (1 similar past message + current message = repeated question)
_REPETITION_ESCALATION_COUNT = 2

# Number of recent user messages to compare against
_REPETITION_WINDOW = 3


# ── Negative Phrases (sentiment detection) ──────────────────────

_NEGATIVE_PHRASES: list[re.Pattern[str]] = [
    re.compile(r"\bthis is ridiculous\b", re.IGNORECASE),
    re.compile(r"\bterrible service\b", re.IGNORECASE),
    re.compile(r"\bnot helpful\b", re.IGNORECASE),
    re.compile(r"\bwaste of time\b", re.IGNORECASE),
    re.compile(r"\babsolutely useless\b", re.IGNORECASE),
    re.compile(r"\bincompetent\b", re.IGNORECASE),
    re.compile(r"\bunacceptable\b", re.IGNORECASE),
    re.compile(r"\bfed up\b", re.IGNORECASE),
    re.compile(r"\bsick and tired\b", re.IGNORECASE),
    re.compile(r"\bhorrific\b", re.IGNORECASE),
    re.compile(r"\bdisgusting\b", re.IGNORECASE),
    re.compile(r"\bawful service\b", re.IGNORECASE),
    re.compile(r"\bworst experience\b", re.IGNORECASE),
    re.compile(r"\bcomplete joke\b", re.IGNORECASE),
    re.compile(r"\bpathetic\b", re.IGNORECASE),
]

# Excessive punctuation: 3+ consecutive ! or ?
_RE_EXCESSIVE_PUNCT = re.compile(r"[!?]{3,}")


# ── Explicit Handoff Patterns ───────────────────────────────────

_EXPLICIT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bspeak\s+to\s+a\s+human\b", re.IGNORECASE),
    re.compile(r"\btalk\s+to\s+a\s+human\b", re.IGNORECASE),
    re.compile(r"\btalk\s+to\s+a\s+(?:real\s+)?person\b", re.IGNORECASE),
    re.compile(r"\bspeak\s+to\s+(?:a\s+)?(?:real\s+)?person\b", re.IGNORECASE),
    re.compile(r"\bspeak\s+to\s+someone\b", re.IGNORECASE),
    re.compile(r"\btalk\s+to\s+someone\b", re.IGNORECASE),
    re.compile(r"\breal\s+agent\b", re.IGNORECASE),
    re.compile(r"\blive\s+agent\b", re.IGNORECASE),
    re.compile(r"\bhuman\s+agent\b", re.IGNORECASE),
    re.compile(r"\bescalate\s+this\b", re.IGNORECASE),
    re.compile(r"\btransfer\s+me\b", re.IGNORECASE),
    re.compile(r"\bspeak\s+to\s+(?:a\s+|your\s+)?manager\b", re.IGNORECASE),
    re.compile(r"\bget\s+me\s+(?:a\s+|your\s+)?manager\b", re.IGNORECASE),
    re.compile(r"\bsupervisor\b", re.IGNORECASE),
]


# ── Detector ────────────────────────────────────────────────────


class EscalationDetector:
    """Smart escalation detector with three strategies.

    Checks user messages for frustration signals, repeated questions,
    and explicit human-handoff requests.

    Priority order (first match wins):
        1. Explicit request (instant escalation)
        2. Repetition detection (conversation-aware)
        3. Sentiment analysis (keyword scoring)

    Usage::

        detector = EscalationDetector()
        result = detector.detect(
            message="THIS IS RIDICULOUS!!!",
            conversation_history=[...],
        )
        if result.should_escalate:
            # trigger escalation flow
    """

    def detect(
        self,
        message: str,
        conversation_history: list[dict[str, str]],
    ) -> EscalationResult:
        """Run all detection strategies on the user message.

        Args:
            message: The user's current message text.
            conversation_history: List of ``{"role": ..., "content": ...}``
                dicts representing the conversation so far.

        Returns:
            An :class:`EscalationResult` with detection outcome.
        """
        if not message or not message.strip():
            return EscalationResult()

        # Priority 1: Explicit handoff request (instant)
        explicit = self._check_explicit(message)
        if explicit.should_escalate:
            return explicit

        # Priority 2: Repetition detection (needs history)
        repetition = self._check_repetition(message, conversation_history)
        if repetition.should_escalate:
            return repetition

        # Priority 3: Sentiment analysis (scoring)
        sentiment = self._check_sentiment(message)
        if sentiment.should_escalate:
            return sentiment

        # No escalation — return sentiment score for visibility
        return EscalationResult(sentiment_score=sentiment.sentiment_score)

    # ── Private: Explicit Request ───────────────────────────────

    @staticmethod
    def _check_explicit(message: str) -> EscalationResult:
        """Check for explicit human-handoff patterns."""
        for pattern in _EXPLICIT_PATTERNS:
            match = pattern.search(message)
            if match:
                return EscalationResult(
                    should_escalate=True,
                    trigger=EscalationTrigger.EXPLICIT_REQUEST,
                    reason=f"User explicitly requested human agent: '{match.group()}'",
                )
        return EscalationResult()

    # ── Private: Repetition ─────────────────────────────────────

    @staticmethod
    def _check_repetition(
        message: str,
        history: list[dict[str, str]],
    ) -> EscalationResult:
        """Check for repeated similar messages within conversation.

        Compares the current message against the last N user messages
        using Jaccard token similarity. Escalates after seeing 2+
        messages with >80% similarity.
        """
        if not history:
            return EscalationResult()

        # Extract last N user messages from history
        user_messages = [
            h["content"]
            for h in history
            if h.get("role") == "user" and h.get("content")
        ]
        recent = user_messages[-_REPETITION_WINDOW:]

        if not recent:
            return EscalationResult()

        current_tokens = _tokenize(message)
        if not current_tokens:
            return EscalationResult()

        similar_count = 0
        for past_msg in recent:
            past_tokens = _tokenize(past_msg)
            if not past_tokens:
                continue
            similarity = _jaccard_similarity(current_tokens, past_tokens)
            if similarity >= _SIMILARITY_THRESHOLD:
                similar_count += 1

        if similar_count >= _REPETITION_ESCALATION_COUNT:
            return EscalationResult(
                should_escalate=True,
                trigger=EscalationTrigger.REPETITION,
                reason=f"User repeated similar question {similar_count + 1} times",
                repetition_count=similar_count,
            )

        return EscalationResult(repetition_count=similar_count)

    # ── Private: Sentiment ──────────────────────────────────────

    @staticmethod
    def _check_sentiment(message: str) -> EscalationResult:
        """Score frustration signals in the message.

        Signals:
            - ALL CAPS (>60% uppercase alpha, min 5 chars) → +0.4
            - Excessive punctuation (!!!  ???) → +0.3
            - Negative phrases → +0.3 per match

        Score capped at 1.0. Escalates at ≥0.7.
        """
        score = 0.0

        # ── ALL CAPS check ───────────────────────────────────────
        alpha_chars = [c for c in message if c.isalpha()]
        if len(alpha_chars) >= _CAPS_MIN_ALPHA:
            upper_count = sum(1 for c in alpha_chars if c.isupper())
            if upper_count / len(alpha_chars) >= _CAPS_THRESHOLD:
                score += _CAPS_SCORE

        # ── Excessive punctuation ────────────────────────────────
        if _RE_EXCESSIVE_PUNCT.search(message):
            score += _PUNCT_SCORE

        # ── Negative phrases ─────────────────────────────────────
        for pattern in _NEGATIVE_PHRASES:
            if pattern.search(message):
                score += _PHRASE_SCORE

        # Cap at 1.0
        score = min(score, 1.0)

        if score >= _SENTIMENT_THRESHOLD:
            return EscalationResult(
                should_escalate=True,
                trigger=EscalationTrigger.SENTIMENT,
                reason=f"High frustration detected (score: {score:.2f})",
                sentiment_score=score,
            )

        return EscalationResult(sentiment_score=score)


# ── Utility Functions ───────────────────────────────────────────


def _tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase word tokens for comparison.

    Strips punctuation from each token so that ``'order?'`` and
    ``'order'`` are treated as the same token.
    """
    tokens: set[str] = set()
    for word in text.split():
        cleaned = word.lower().strip(".,!?;:'\"()-")
        if cleaned:
            tokens.add(cleaned)
    return tokens


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two token sets.

    Returns:
        Float between 0.0 (no overlap) and 1.0 (identical).
    """
    if not set_a and not set_b:
        return 1.0
    intersection = set_a & set_b
    union = set_a | set_b
    if not union:
        return 0.0
    return len(intersection) / len(union)
