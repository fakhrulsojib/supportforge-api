"""Output validation service — anti-hallucination guard.

Post-generation validation that detects fabricated contact info,
prices, percentages, and forbidden patterns in LLM responses.

**Pure domain service** — ZERO framework imports.  Cross-references
detected items against the retrieved context to distinguish real
data from hallucinations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.domain.models.enums import ValidationStatus


# ── Result Types ────────────────────────────────────────────────


@dataclass
class ValidationViolation:
    """A single validation rule violation.

    Attributes:
        rule: Machine-readable rule identifier (e.g. ``fabricated_phone``).
        snippet: The matched text that triggered the violation.
    """

    rule: str
    snippet: str


@dataclass
class ValidationResult:
    """Outcome of output validation.

    Attributes:
        status: Whether the response passed or was flagged.
        violations: List of individual rule violations found.
        disclaimer: Disclaimer text to append if flagged, empty if passed.
    """

    status: ValidationStatus
    violations: list[ValidationViolation] = field(default_factory=list)
    disclaimer: str = ""


# ── Constants ───────────────────────────────────────────────────

_DISCLAIMER = (
    "\u26a0\ufe0f Note: Some details in this response could not be "
    "verified against our documentation. Please confirm with our team."
)

# ── Compiled Regex Patterns ─────────────────────────────────────

# Phone: 10-digit numbers with optional separators (dot or dash)
_RE_PHONE = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")

# Email addresses
_RE_EMAIL = re.compile(r"[\w.-]+@[\w.-]+\.\w+")

# URLs (http or https) — negative lookbehind strips trailing sentence punctuation
_RE_URL = re.compile(r"https?://[\w.-]+(?:/[\w/?&=#%-]*)*(?<![.,;:!?)])")

# Prices: dollar sign + digits, optional cents
_RE_PRICE = re.compile(r"\$\d+(?:\.\d{2})?")

# Percentages: digits + %
_RE_PERCENTAGE = re.compile(r"\d+(?:\.\d+)?%")

# Forbidden: LaTeX commands
_RE_LATEX = re.compile(r"\\(?:boxed|text|frac)\{")

# Forbidden: third-person customer references
_RE_THIRD_PERSON = re.compile(
    r"\b(?:the\s+customer|the\s+user|the\s+client)\b",
    re.IGNORECASE,
)


# ── Validator ───────────────────────────────────────────────────


class OutputValidator:
    """Post-generation output validator.

    Checks LLM responses for fabricated contact information, prices,
    and forbidden patterns.  Items found in the retrieved context are
    considered legitimate and not flagged.

    Usage::

        validator = OutputValidator()
        result = validator.validate(answer_text, context_texts)
        if result.status == ValidationStatus.FLAGGED:
            # append result.disclaimer, log violations
    """

    def validate(
        self,
        answer: str,
        context_texts: list[str],
    ) -> ValidationResult:
        """Validate an LLM response against the retrieved context.

        Args:
            answer: The full LLM response text to validate.
            context_texts: List of retrieved document chunk strings
                used as ground truth for cross-referencing.

        Returns:
            A :class:`ValidationResult` with status, violations, and
            an optional disclaimer string.
        """
        if not answer or not answer.strip():
            return ValidationResult(status=ValidationStatus.PASSED)

        merged_context = "\n".join(context_texts)
        violations: list[ValidationViolation] = []

        # ── Cross-referenced checks (fabrication) ────────────────
        self._check_fabricated(
            answer, merged_context, _RE_PHONE, "fabricated_phone", violations,
        )
        self._check_fabricated(
            answer, merged_context, _RE_EMAIL, "fabricated_email", violations,
        )
        self._check_fabricated(
            answer, merged_context, _RE_URL, "fabricated_url", violations,
        )
        self._check_fabricated(
            answer, merged_context, _RE_PRICE, "fabricated_price", violations,
        )
        self._check_fabricated(
            answer,
            merged_context,
            _RE_PERCENTAGE,
            "fabricated_percentage",
            violations,
        )

        # ── Forbidden patterns (always flagged) ──────────────────
        self._check_forbidden(
            answer, _RE_LATEX, "forbidden_latex", violations,
        )
        self._check_forbidden(
            answer, _RE_THIRD_PERSON, "forbidden_third_person", violations,
        )

        # ── Build result ─────────────────────────────────────────
        if violations:
            return ValidationResult(
                status=ValidationStatus.FLAGGED,
                violations=violations,
                disclaimer=_DISCLAIMER,
            )
        return ValidationResult(status=ValidationStatus.PASSED)

    # ── Private helpers ──────────────────────────────────────────

    @staticmethod
    def _check_fabricated(
        answer: str,
        context: str,
        pattern: re.Pattern[str],
        rule_name: str,
        violations: list[ValidationViolation],
    ) -> None:
        """Flag matches in *answer* that are NOT present in *context*."""
        for match in pattern.finditer(answer):
            matched_text = match.group()
            if matched_text not in context:
                violations.append(
                    ValidationViolation(rule=rule_name, snippet=matched_text)
                )

    @staticmethod
    def _check_forbidden(
        answer: str,
        pattern: re.Pattern[str],
        rule_name: str,
        violations: list[ValidationViolation],
    ) -> None:
        """Flag matches in *answer* unconditionally (context-independent)."""
        for match in pattern.finditer(answer):
            violations.append(
                ValidationViolation(rule=rule_name, snippet=match.group())
            )
