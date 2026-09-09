"""Provider failures must be diagnosed, not dumped.

Every case here is an operational failure a user can actually hit, and each
maps to a remedy they can apply without reading the source.
"""

from __future__ import annotations

import pytest

from sectorlens.agent.errors import classify

CREDIT_ERROR = (
    "BadRequestError: Error code: 400 - {'type': 'error', 'error': {'type': "
    "'invalid_request_error', 'message': 'Your credit balance is too low to "
    "access the Anthropic API. Please go to Plans & Billing to upgrade or "
    "purchase credits.'}}"
)


@pytest.mark.parametrize("message,expected", [
    (CREDIT_ERROR, "insufficient_credit"),
    ("401 authentication_error: invalid x-api-key", "invalid_api_key"),
    ("403 permission_error: not allowed", "permission_denied"),
    ("404 not_found_error: model not found", "model_unavailable"),
    ("429 rate_limit_error: slow down", "rate_limited"),
    ("529 overloaded_error", "provider_overloaded"),
    ("connection reset by peer", "network_error"),
    ("something nobody anticipated", "provider_error"),
])
def test_failures_are_classified(message, expected):
    assert classify(RuntimeError(message)).kind == expected


def test_every_diagnosis_carries_an_actionable_remedy():
    for message in (CREDIT_ERROR, "401 authentication_error", "429 rate_limit",
                    "nothing recognisable"):
        failure = classify(RuntimeError(message))
        assert failure.title.endswith(".")
        assert len(failure.remedy) > 40, "a remedy has to say what to do"


def test_transient_failures_are_marked_retryable():
    assert classify(RuntimeError("429 rate_limit_error")).retryable is True
    assert classify(RuntimeError("529 overloaded_error")).retryable is True
    assert classify(RuntimeError("401 authentication_error")).retryable is False


def test_credit_exhaustion_still_allows_a_degraded_answer():
    """The data layer is untouched by a billing failure, so the deterministic
    provider remains a legitimate fallback."""
    assert classify(RuntimeError(CREDIT_ERROR)).degradable is True
