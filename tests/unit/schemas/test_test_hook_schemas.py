"""Schema validation tests for TestHookRequest / TestHookResponse.

Tests edge cases for Pydantic field constraints:
  - event_type min/max length
  - url min/max length
  - headers defaults to empty dict
  - TestHookResponse fields
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.api.schemas.tenant import TestHookRequest, TestHookResponse


class TestTestHookRequest:
    """Edge-case validation for TestHookRequest."""

    def test_valid_request(self) -> None:
        req = TestHookRequest(
            event_type="on_escalation",
            url="https://hooks.example.com/escalation",
        )
        assert req.event_type == "on_escalation"
        assert req.url == "https://hooks.example.com/escalation"
        assert req.headers == {}  # default

    def test_with_custom_headers(self) -> None:
        req = TestHookRequest(
            event_type="on_escalation",
            url="https://hooks.example.com",
            headers={"X-Secret": "abc123"},
        )
        assert req.headers == {"X-Secret": "abc123"}

    def test_empty_event_type_rejected(self) -> None:
        with pytest.raises(ValidationError, match="event_type"):
            TestHookRequest(event_type="", url="https://example.com")

    def test_empty_url_rejected(self) -> None:
        with pytest.raises(ValidationError, match="url"):
            TestHookRequest(event_type="on_escalation", url="")

    def test_event_type_max_length(self) -> None:
        with pytest.raises(ValidationError, match="event_type"):
            TestHookRequest(
                event_type="x" * 51,
                url="https://example.com",
            )

    def test_url_max_length(self) -> None:
        with pytest.raises(ValidationError, match="url"):
            TestHookRequest(
                event_type="on_escalation",
                url="https://example.com/" + "x" * 2048,
            )

    def test_missing_event_type_rejected(self) -> None:
        with pytest.raises(ValidationError, match="event_type"):
            TestHookRequest(url="https://example.com")  # type: ignore[call-arg]

    def test_missing_url_rejected(self) -> None:
        with pytest.raises(ValidationError, match="url"):
            TestHookRequest(event_type="on_escalation")  # type: ignore[call-arg]


class TestTestHookResponse:
    """Edge-case validation for TestHookResponse."""

    def test_success_response(self) -> None:
        resp = TestHookResponse(success=True, status_code=200)
        assert resp.success is True
        assert resp.status_code == 200
        assert resp.error is None

    def test_failure_response(self) -> None:
        resp = TestHookResponse(
            success=False,
            status_code=500,
            error="Internal Server Error",
        )
        assert resp.success is False
        assert resp.error == "Internal Server Error"

    def test_timeout_response(self) -> None:
        resp = TestHookResponse(success=False, error="Request timed out (10s)")
        assert resp.status_code is None
        assert resp.error == "Request timed out (10s)"

    def test_missing_success_rejected(self) -> None:
        with pytest.raises(ValidationError, match="success"):
            TestHookResponse()  # type: ignore[call-arg]
