"""Content moderation service — input/output filtering.

Lightweight moderation layer that detects jailbreak attempts and
tenant-configurable banned terms in user input (before RAG pipeline)
and in LLM output (after streaming completes).

**Pure domain service** — ZERO framework imports.  No external API
dependencies; all detection is regex-based and deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ── Result Types ────────────────────────────────────────────────


@dataclass
class ModerationResult:
    """Outcome of a content moderation check.

    For input checks:
        - ``blocked``: Whether the input was rejected.
        - ``reason``: Machine-readable reason (e.g. ``jailbreak_detected``).
        - ``canned_response``: Pre-built response to return instead of LLM output.

    For output checks:
        - ``flagged``: Whether the output was flagged for review.
        - ``reason``: Machine-readable reason (e.g. ``blocklist_match``).
    """

    blocked: bool = False
    flagged: bool = False
    reason: str = ""
    matched_term: str = ""
    canned_response: str = ""


# ── Constants ───────────────────────────────────────────────────

_CANNED_RESPONSE = "I'm here to help with customer support questions. Could you please rephrase your question?"

# ── Compiled Regex Patterns (jailbreak detection) ───────────────
#
# All patterns use re.IGNORECASE for case-insensitive matching.
# Word boundaries (\b) prevent false positives on partial words
# (e.g., "reacting as" should NOT trigger "act as").

_JAILBREAK_PATTERNS: list[re.Pattern[str]] = [
    # Instruction override attempts
    re.compile(
        r"ignore\s+(?:previous|your|all)\s+instructions",
        re.IGNORECASE,
    ),
    # Persona injection
    re.compile(r"pretend\s+(?:you\s+are|to\s+be)\b", re.IGNORECASE),
    re.compile(r"\bact\s+as\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE),
    # Known jailbreak modes
    re.compile(r"\bDAN\s+mode\b", re.IGNORECASE),
    re.compile(r"\bdeveloper\s+mode\b", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    # Prompt extraction
    re.compile(r"\bsystem\s+prompt\b", re.IGNORECASE),
    re.compile(r"reveal\s+your\s+prompt", re.IGNORECASE),
    re.compile(r"show\s+your\s+instructions", re.IGNORECASE),
    # Rule circumvention
    re.compile(r"forget\s+your\s+rules", re.IGNORECASE),
    re.compile(r"disregard\s+your\b", re.IGNORECASE),
    re.compile(r"override\s+your\b", re.IGNORECASE),
]


# ── Moderator ───────────────────────────────────────────────────


class ContentModerator:
    """Input/output content moderator.

    Checks user messages for jailbreak attempts and banned terms
    before they reach the RAG pipeline.  Also checks LLM output
    for banned terms after streaming completes.

    Usage::

        moderator = ContentModerator()

        # Before RAG pipeline
        input_result = moderator.check_input(user_message, tenant_blocklist)
        if input_result.blocked:
            return input_result.canned_response

        # After streaming
        output_result = moderator.check_output(llm_answer, tenant_blocklist)
        if output_result.flagged:
            log_warning(...)
    """

    def check_input(
        self,
        message: str,
        blocklist: list[str],
    ) -> ModerationResult:
        """Check user input for jailbreak attempts and banned terms.

        Jailbreak patterns are checked first, then the blocklist.
        If either matches, the input is blocked and a canned response
        is returned.

        Args:
            message: The user's raw input message.
            blocklist: Tenant-specific list of banned terms.

        Returns:
            A :class:`ModerationResult` indicating whether the input
            is blocked, with reason and canned response if applicable.
        """
        if not message or not message.strip():
            return ModerationResult()

        # ── Jailbreak pattern check ──────────────────────────────
        for pattern in _JAILBREAK_PATTERNS:
            match = pattern.search(message)
            if match:
                return ModerationResult(
                    blocked=True,
                    reason="jailbreak_detected",
                    matched_term=match.group(),
                    canned_response=_CANNED_RESPONSE,
                )

        # ── Blocklist check ──────────────────────────────────────
        lowered_message = message.lower()
        for term in blocklist:
            clean_term = term.strip().lower()
            if not clean_term:
                continue
            if clean_term in lowered_message:
                return ModerationResult(
                    blocked=True,
                    reason="blocklist_match",
                    matched_term=term.strip(),
                    canned_response=_CANNED_RESPONSE,
                )

        return ModerationResult()

    def check_output(
        self,
        answer: str,
        blocklist: list[str],
    ) -> ModerationResult:
        """Check LLM output for banned terms.

        Only checks the blocklist — jailbreak patterns are not
        relevant for output since the LLM doesn't produce them.

        Args:
            answer: The full LLM response text.
            blocklist: Tenant-specific list of banned terms.

        Returns:
            A :class:`ModerationResult` indicating whether the output
            is flagged for review.
        """
        if not answer or not answer.strip():
            return ModerationResult()

        lowered_answer = answer.lower()
        for term in blocklist:
            clean_term = term.strip().lower()
            if not clean_term:
                continue
            if clean_term in lowered_answer:
                return ModerationResult(
                    flagged=True,
                    reason="blocklist_match",
                    matched_term=term.strip(),
                )

        return ModerationResult()
