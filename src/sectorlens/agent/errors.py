"""Turning provider failures into something a person can act on.

An unhandled provider error reaches the UI as a raw exception repr -- a wall of
nested dicts in which the one useful sentence is buried. Worse, the failures
that actually happen in practice are all operational (no credit, expired key,
rate limit, a model the account cannot reach) and every one of them has a
specific remedy the user could apply in under a minute.

So each is classified once, here, and both interfaces render the same
diagnosis: what happened, and what to do about it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderFailure:
    kind: str
    title: str
    remedy: str
    retryable: bool
    #: True when answering with the deterministic provider is a reasonable
    #: consolation -- the request was well formed and the data layer is fine,
    #: only the model was unreachable.
    degradable: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "error": self.kind,
            "message": self.title,
            "remedy": self.remedy,
            "retryable": self.retryable,
        }


_UNKNOWN = ProviderFailure(
    kind="provider_error",
    title="The language model provider returned an error.",
    remedy="Check the server logs for the full response. The database and MCP "
           "layers are unaffected, so the deterministic provider still works: "
           "set SECTORLENS_LLM_PROVIDER=deterministic.",
    retryable=False,
)

# Ordered: the first matching signature wins, so put the specific ones first.
_SIGNATURES: tuple[tuple[tuple[str, ...], ProviderFailure], ...] = (
    (("credit balance", "billing", "insufficient_quota"),
     ProviderFailure(
         kind="insufficient_credit",
         title="The Anthropic account has no remaining credit.",
         remedy="Add credit at console.anthropic.com under Plans & Billing. "
                "Nothing is wrong with the key or the code -- the request never "
                "reached the model.",
         retryable=True)),

    (("authentication_error", "invalid x-api-key", "invalid_api_key",
      "401"),
     ProviderFailure(
         kind="invalid_api_key",
         title="The API key was rejected.",
         remedy="Check ANTHROPIC_API_KEY in .env: it must be the full key, "
                "unquoted, with no spaces around the '='. Restart the server "
                "after editing, since .env is read at startup.",
         retryable=False)),

    (("permission_error", "403"),
     ProviderFailure(
         kind="permission_denied",
         title="The key is valid but not permitted to make this request.",
         remedy="The account may lack access to this model or endpoint. Try a "
                "different model via SECTORLENS_MODEL.",
         retryable=False)),

    (("not_found_error", "model_not_found", "404"),
     ProviderFailure(
         kind="model_unavailable",
         title="The configured model is not available to this account.",
         remedy="Set SECTORLENS_MODEL in .env to a model the account can reach, "
                "for example claude-sonnet-5, and restart the server.",
         retryable=False)),

    (("rate_limit", "429"),
     ProviderFailure(
         kind="rate_limited",
         title="The provider is rate limiting this account.",
         remedy="Wait a few seconds and ask again. If it persists, the "
                "account's requests-per-minute limit is being hit by something "
                "else.",
         retryable=True)),

    (("overloaded", "529"),
     ProviderFailure(
         kind="provider_overloaded",
         title="The provider is temporarily overloaded.",
         remedy="This is transient and not caused by anything local. Retry in "
                "a moment.",
         retryable=True)),

    (("timeout", "timed out", "connection", "temporary failure in name "
      "resolution"),
     ProviderFailure(
         kind="network_error",
         title="Could not reach the provider.",
         remedy="Check network access and any proxy settings, then retry.",
         retryable=True)),
)


def classify(exc: BaseException) -> ProviderFailure:
    """Map a provider exception onto an actionable diagnosis.

    Matching is on the rendered message rather than exception classes: the
    same underlying condition arrives as several SDK types depending on where
    it is raised, and the wire message is the stable part.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    for needles, failure in _SIGNATURES:
        if any(needle in text for needle in needles):
            return failure
    return _UNKNOWN
