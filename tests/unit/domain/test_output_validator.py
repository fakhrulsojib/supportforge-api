"""Unit tests for the OutputValidator domain service.

Covers:
    - Clean responses pass validation
    - Fabricated phone numbers detected (not in context)
    - Phone numbers from context pass
    - Fabricated emails detected
    - Emails from context pass
    - Fabricated URLs detected
    - URLs from context pass
    - Fabricated prices detected
    - Prices from context pass
    - Fabricated percentages detected
    - Percentages from context pass
    - Forbidden LaTeX patterns always flagged
    - Forbidden third-person references always flagged
    - Multiple violations in one answer
    - Empty answer passes
    - Empty context — all extracted numbers flagged
    - Disclaimer present on flag, absent on pass
    - Boundary: partial phone not matched
    - Boundary: order numbers not matched as phone
    - Context match is case-insensitive for URLs
    - Real-world mixed response
"""

from __future__ import annotations

import pytest

from app.domain.models.enums import ValidationStatus
from app.domain.services.output_validator import (
    OutputValidator,
    ValidationResult,
    ValidationViolation,
)

# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def validator() -> OutputValidator:
    """Create a fresh OutputValidator instance."""
    return OutputValidator()


@pytest.fixture
def empty_context() -> list[str]:
    """Empty context — nothing to cross-reference against."""
    return []


@pytest.fixture
def sample_context() -> list[str]:
    """Context containing specific contact info and prices."""
    return [
        "For assistance, call 800-555-1234 or email support@acme.com",
        "Visit https://acme.com/help for more information.",
        "The premium plan costs $49.99 per month with a 15% discount.",
    ]


# ── Clean Responses ─────────────────────────────────────────────


class TestCleanResponses:
    """Tests for responses that should pass validation."""

    def test_clean_response_passes(
        self, validator: OutputValidator, empty_context: list[str]
    ) -> None:
        """Simple clean answer with no fabricated info should pass."""
        result = validator.validate(
            "Your order will arrive in 3-5 business days.",
            empty_context,
        )
        assert result.status == ValidationStatus.PASSED
        assert len(result.violations) == 0
        assert result.disclaimer == ""

    def test_empty_answer_passes(
        self, validator: OutputValidator, empty_context: list[str]
    ) -> None:
        """Empty answer should pass validation."""
        result = validator.validate("", empty_context)
        assert result.status == ValidationStatus.PASSED
        assert len(result.violations) == 0

    def test_no_disclaimer_on_pass(
        self, validator: OutputValidator, sample_context: list[str]
    ) -> None:
        """Responses that pass should have empty disclaimer."""
        result = validator.validate(
            "I can help you with your order tracking.",
            sample_context,
        )
        assert result.disclaimer == ""


# ── Fabricated Phone Numbers ────────────────────────────────────


class TestFabricatedPhones:
    """Tests for phone number fabrication detection."""

    def test_fabricated_phone_detected(
        self, validator: OutputValidator, empty_context: list[str]
    ) -> None:
        """Phone number not in context should be flagged."""
        result = validator.validate(
            "You can call us at 555-123-4567 for help.",
            empty_context,
        )
        assert result.status == ValidationStatus.FLAGGED
        assert any(v.rule == "fabricated_phone" for v in result.violations)

    def test_fabricated_phone_with_dots(
        self, validator: OutputValidator, empty_context: list[str]
    ) -> None:
        """Phone with dot separators should also be detected."""
        result = validator.validate(
            "Contact us at 555.123.4567.",
            empty_context,
        )
        assert result.status == ValidationStatus.FLAGGED
        assert any(v.rule == "fabricated_phone" for v in result.violations)

    def test_fabricated_phone_no_separators(
        self, validator: OutputValidator, empty_context: list[str]
    ) -> None:
        """Phone without separators should also be detected."""
        result = validator.validate(
            "Our number is 5551234567.",
            empty_context,
        )
        assert result.status == ValidationStatus.FLAGGED
        assert any(v.rule == "fabricated_phone" for v in result.violations)

    def test_phone_from_context_passes(
        self, validator: OutputValidator, sample_context: list[str]
    ) -> None:
        """Phone number that exists in context should NOT be flagged."""
        result = validator.validate(
            "You can reach us at 800-555-1234.",
            sample_context,
        )
        # Should not have fabricated_phone violation for this number
        phone_violations = [
            v for v in result.violations if v.rule == "fabricated_phone"
        ]
        assert len(phone_violations) == 0


# ── Fabricated Emails ───────────────────────────────────────────


class TestFabricatedEmails:
    """Tests for email address fabrication detection."""

    def test_fabricated_email_detected(
        self, validator: OutputValidator, empty_context: list[str]
    ) -> None:
        """Email not in context should be flagged."""
        result = validator.validate(
            "Send an email to help@fakesupport.com.",
            empty_context,
        )
        assert result.status == ValidationStatus.FLAGGED
        assert any(v.rule == "fabricated_email" for v in result.violations)

    def test_email_from_context_passes(
        self, validator: OutputValidator, sample_context: list[str]
    ) -> None:
        """Email that exists in context should NOT be flagged."""
        result = validator.validate(
            "You can email us at support@acme.com.",
            sample_context,
        )
        email_violations = [
            v for v in result.violations if v.rule == "fabricated_email"
        ]
        assert len(email_violations) == 0


# ── Fabricated URLs ─────────────────────────────────────────────


class TestFabricatedURLs:
    """Tests for URL fabrication detection."""

    def test_fabricated_url_detected(
        self, validator: OutputValidator, empty_context: list[str]
    ) -> None:
        """URL not in context should be flagged."""
        result = validator.validate(
            "Visit https://made-up-support.com/help for more info.",
            empty_context,
        )
        assert result.status == ValidationStatus.FLAGGED
        assert any(v.rule == "fabricated_url" for v in result.violations)

    def test_url_from_context_passes(
        self, validator: OutputValidator, sample_context: list[str]
    ) -> None:
        """URL that exists in context should NOT be flagged."""
        result = validator.validate(
            "Please visit https://acme.com/help for more details.",
            sample_context,
        )
        url_violations = [
            v for v in result.violations if v.rule == "fabricated_url"
        ]
        assert len(url_violations) == 0

    def test_http_url_detected(
        self, validator: OutputValidator, empty_context: list[str]
    ) -> None:
        """HTTP (non-HTTPS) URL should also be detected."""
        result = validator.validate(
            "Go to http://example.com/support.",
            empty_context,
        )
        assert result.status == ValidationStatus.FLAGGED
        assert any(v.rule == "fabricated_url" for v in result.violations)


# ── Fabricated Prices ───────────────────────────────────────────


class TestFabricatedPrices:
    """Tests for price fabrication detection."""

    def test_fabricated_price_detected(
        self, validator: OutputValidator, empty_context: list[str]
    ) -> None:
        """Price not in context should be flagged."""
        result = validator.validate(
            "The basic plan is only $29.99 per month.",
            empty_context,
        )
        assert result.status == ValidationStatus.FLAGGED
        assert any(v.rule == "fabricated_price" for v in result.violations)

    def test_price_from_context_passes(
        self, validator: OutputValidator, sample_context: list[str]
    ) -> None:
        """Price that exists in context should NOT be flagged."""
        result = validator.validate(
            "The premium plan costs $49.99.",
            sample_context,
        )
        price_violations = [
            v for v in result.violations if v.rule == "fabricated_price"
        ]
        assert len(price_violations) == 0

    def test_whole_dollar_price_detected(
        self, validator: OutputValidator, empty_context: list[str]
    ) -> None:
        """Whole dollar amount without cents should be detected."""
        result = validator.validate(
            "That will be $100.",
            empty_context,
        )
        assert result.status == ValidationStatus.FLAGGED
        assert any(v.rule == "fabricated_price" for v in result.violations)


# ── Fabricated Percentages ──────────────────────────────────────


class TestFabricatedPercentages:
    """Tests for percentage fabrication detection."""

    def test_fabricated_percentage_detected(
        self, validator: OutputValidator, empty_context: list[str]
    ) -> None:
        """Percentage not in context should be flagged."""
        result = validator.validate(
            "We have a 99% satisfaction rate.",
            empty_context,
        )
        assert result.status == ValidationStatus.FLAGGED
        assert any(v.rule == "fabricated_percentage" for v in result.violations)

    def test_percentage_from_context_passes(
        self, validator: OutputValidator, sample_context: list[str]
    ) -> None:
        """Percentage that exists in context should NOT be flagged."""
        result = validator.validate(
            "You get a 15% discount on the premium plan.",
            sample_context,
        )
        pct_violations = [
            v for v in result.violations if v.rule == "fabricated_percentage"
        ]
        assert len(pct_violations) == 0

    def test_decimal_percentage_detected(
        self, validator: OutputValidator, empty_context: list[str]
    ) -> None:
        """Decimal percentage not in context should be flagged."""
        result = validator.validate(
            "Our uptime is 99.9%.",
            empty_context,
        )
        assert result.status == ValidationStatus.FLAGGED
        assert any(v.rule == "fabricated_percentage" for v in result.violations)


# ── Forbidden Patterns ──────────────────────────────────────────


class TestForbiddenPatterns:
    """Tests for forbidden patterns that are always flagged."""

    def test_latex_boxed_detected(
        self, validator: OutputValidator, sample_context: list[str]
    ) -> None:
        """LaTeX \\boxed{} should always be flagged."""
        result = validator.validate(
            "The answer is \\boxed{42}.",
            sample_context,
        )
        assert result.status == ValidationStatus.FLAGGED
        assert any(v.rule == "forbidden_latex" for v in result.violations)

    def test_latex_text_detected(
        self, validator: OutputValidator, sample_context: list[str]
    ) -> None:
        """LaTeX \\text{} should always be flagged."""
        result = validator.validate(
            "Use \\text{bold} for emphasis.",
            sample_context,
        )
        assert result.status == ValidationStatus.FLAGGED
        assert any(v.rule == "forbidden_latex" for v in result.violations)

    def test_latex_frac_detected(
        self, validator: OutputValidator, sample_context: list[str]
    ) -> None:
        """LaTeX \\frac{} should always be flagged."""
        result = validator.validate(
            "The ratio is \\frac{1}{2}.",
            sample_context,
        )
        assert result.status == ValidationStatus.FLAGGED
        assert any(v.rule == "forbidden_latex" for v in result.violations)

    def test_third_person_the_customer(
        self, validator: OutputValidator, sample_context: list[str]
    ) -> None:
        """'the customer' reference should always be flagged."""
        result = validator.validate(
            "The customer should reset their password.",
            sample_context,
        )
        assert result.status == ValidationStatus.FLAGGED
        assert any(
            v.rule == "forbidden_third_person" for v in result.violations
        )

    def test_third_person_the_user(
        self, validator: OutputValidator, sample_context: list[str]
    ) -> None:
        """'the user' reference should always be flagged."""
        result = validator.validate(
            "The user needs to update their email.",
            sample_context,
        )
        assert result.status == ValidationStatus.FLAGGED
        assert any(
            v.rule == "forbidden_third_person" for v in result.violations
        )

    def test_third_person_the_client(
        self, validator: OutputValidator, sample_context: list[str]
    ) -> None:
        """'the client' reference should always be flagged."""
        result = validator.validate(
            "The client has not completed registration.",
            sample_context,
        )
        assert result.status == ValidationStatus.FLAGGED
        assert any(
            v.rule == "forbidden_third_person" for v in result.violations
        )

    def test_third_person_case_insensitive(
        self, validator: OutputValidator, sample_context: list[str]
    ) -> None:
        """Third-person detection should be case-insensitive."""
        result = validator.validate(
            "THE CUSTOMER should contact support.",
            sample_context,
        )
        assert result.status == ValidationStatus.FLAGGED
        assert any(
            v.rule == "forbidden_third_person" for v in result.violations
        )

    def test_forbidden_patterns_flagged_regardless_of_context(
        self, validator: OutputValidator,
    ) -> None:
        """Forbidden patterns should be flagged even when in context."""
        context = ["The customer should contact \\boxed{support}."]
        result = validator.validate(
            "The customer should contact \\boxed{support}.",
            context,
        )
        assert result.status == ValidationStatus.FLAGGED
        assert any(v.rule == "forbidden_latex" for v in result.violations)
        assert any(v.rule == "forbidden_third_person" for v in result.violations)


# ── Multiple Violations ─────────────────────────────────────────


class TestMultipleViolations:
    """Tests for multiple simultaneous violations."""

    def test_multiple_violations_all_listed(
        self, validator: OutputValidator, empty_context: list[str]
    ) -> None:
        """Multiple violations in one answer should all be reported."""
        result = validator.validate(
            "Call 555-123-4567 or email fake@test.com. "
            "The answer is \\boxed{42}.",
            empty_context,
        )
        assert result.status == ValidationStatus.FLAGGED
        rules = {v.rule for v in result.violations}
        assert "fabricated_phone" in rules
        assert "fabricated_email" in rules
        assert "forbidden_latex" in rules

    def test_mixed_context_and_fabricated(
        self, validator: OutputValidator, sample_context: list[str]
    ) -> None:
        """Only non-context items should be violations."""
        result = validator.validate(
            "Call 800-555-1234 (our real number) or 999-888-7777 (not real).",
            sample_context,
        )
        phone_violations = [
            v for v in result.violations if v.rule == "fabricated_phone"
        ]
        # 800-555-1234 is in context — should not be a violation
        # 999-888-7777 is NOT in context — should be a violation
        assert len(phone_violations) == 1
        assert "999-888-7777" in phone_violations[0].snippet


# ── Edge Cases / Boundaries ─────────────────────────────────────


class TestEdgeCases:
    """Tests for boundary conditions and edge cases."""

    def test_order_number_not_matched_as_phone(
        self, validator: OutputValidator, empty_context: list[str]
    ) -> None:
        """Short numbers like order IDs (5-6 digits) should not match as phone."""
        result = validator.validate(
            "Your order #12345 has been shipped.",
            empty_context,
        )
        phone_violations = [
            v for v in result.violations if v.rule == "fabricated_phone"
        ]
        assert len(phone_violations) == 0

    def test_year_not_matched_as_price(
        self, validator: OutputValidator, empty_context: list[str]
    ) -> None:
        """Years should not trigger price detection."""
        result = validator.validate(
            "This policy was updated in 2024.",
            empty_context,
        )
        price_violations = [
            v for v in result.violations if v.rule == "fabricated_price"
        ]
        assert len(price_violations) == 0

    def test_disclaimer_appended_on_flag(
        self, validator: OutputValidator, empty_context: list[str]
    ) -> None:
        """Flagged results should include the standard disclaimer."""
        result = validator.validate(
            "Call 555-123-4567 for help.",
            empty_context,
        )
        assert result.disclaimer != ""
        assert "could not be verified" in result.disclaimer

    def test_context_match_ignores_surrounding_text(
        self, validator: OutputValidator,
    ) -> None:
        """Phone number match against context should find number within text."""
        context = ["Please call our support line at 800-555-1234 for help."]
        result = validator.validate(
            "Our number is 800-555-1234.",
            context,
        )
        phone_violations = [
            v for v in result.violations if v.rule == "fabricated_phone"
        ]
        assert len(phone_violations) == 0

    def test_validation_result_types(
        self, validator: OutputValidator, empty_context: list[str]
    ) -> None:
        """Verify ValidationResult and ValidationViolation types."""
        result = validator.validate(
            "Call 555-123-4567.",
            empty_context,
        )
        assert isinstance(result, ValidationResult)
        assert all(isinstance(v, ValidationViolation) for v in result.violations)

    def test_whitespace_only_answer_passes(
        self, validator: OutputValidator, empty_context: list[str]
    ) -> None:
        """Whitespace-only answer should pass."""
        result = validator.validate("   \n\t  ", empty_context)
        assert result.status == ValidationStatus.PASSED

    def test_url_trailing_period_not_captured(
        self, validator: OutputValidator,
    ) -> None:
        """URL at end of sentence should not capture trailing period.

        Context contains the clean URL; the answer ends it with a period.
        Without trailing-punctuation stripping, this would false-positive.
        """
        context = ["Visit https://acme.com/help for details."]
        result = validator.validate(
            "Go to https://acme.com/help.",
            context,
        )
        url_violations = [
            v for v in result.violations if v.rule == "fabricated_url"
        ]
        assert len(url_violations) == 0

    def test_url_in_parentheses_not_captured(
        self, validator: OutputValidator,
    ) -> None:
        """URL wrapped in parentheses should not capture closing paren."""
        context = ["See https://acme.com/faq for answers."]
        result = validator.validate(
            "Check our FAQ (https://acme.com/faq) for details.",
            context,
        )
        url_violations = [
            v for v in result.violations if v.rule == "fabricated_url"
        ]
        assert len(url_violations) == 0
