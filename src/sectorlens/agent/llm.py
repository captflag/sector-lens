"""LLM providers.

Two implementations sit behind one interface:

* `AnthropicProvider` -- runs a manual agentic loop against the Messages API.
  A manual loop rather than the SDK's tool runner because the tools are
  discovered over MCP at runtime, and because every round trip is recorded for
  the response's audit trail.

* `DeterministicProvider` -- answers with no LLM at all. It still goes through
  the same MCP tools and the same persona weighting, and composes the result
  from the persona's own answer structure. It exists so the system is runnable
  and reviewable without credentials, and so tests can assert on retrieval
  behaviour without paying for or depending on model output.

Both return the same `AgentAnswer`, and the response always names which one ran.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from ..config import Persona, Sector
from ..formatting import fmt_value, pct_rank
from ..settings import Settings
from .budget import RequestBudget
from .mcp_client import McpToolbox
from .prompts import build_system_prompt, build_user_prompt
from .schemas import AgentAnswer, Evidence, answer_json_schema


class LLMProvider(Protocol):
    name: str
    model: str | None

    async def answer(self, *, query: str, persona: Persona, sector: Sector,
                     toolbox: McpToolbox, max_rounds: int) -> AgentAnswer:
        ...


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicProvider:
    """Manual agentic loop over the Messages API."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        self.name = "anthropic"
        self.model = settings.model
        if client is None:
            from anthropic import AsyncAnthropic  # lazy: optional at runtime

            client = AsyncAnthropic(api_key=settings.anthropic_api_key or None)
        self._client = client
        # Server-side fallbacks route around a safety refusal instead of
        # returning nothing. Both this and the structured-output format are
        # dropped automatically if the account or model rejects them, so an
        # optional feature can never cost the whole request -- see `_create`.
        self._use_fallbacks = True
        self._use_output_format = True

    async def _create(self, **kwargs: Any) -> Any:
        """Send one request, degrading past optional features rather than failing.

        Two capabilities are opportunistic: the server-side refusal fallback
        beta, and the structured `output_config.format`. Each is retried once
        without the feature if the API rejects it specifically, and the
        downgrade is remembered so later rounds do not repeat the round trip.
        """
        for _ in range(3):
            try:
                if self._use_fallbacks:
                    return await self._client.beta.messages.create(
                        betas=["server-side-fallback-2026-07-01"],
                        fallbacks="default",
                        **kwargs,
                    )
                return await self._client.messages.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                if self._use_fallbacks and _rejects_feature(exc, _FALLBACK_MARKERS):
                    self._use_fallbacks = False
                    continue
                if (self._use_output_format
                        and "output_config" in kwargs
                        and _rejects_feature(exc, _OUTPUT_FORMAT_MARKERS)):
                    # Fall back to asking for JSON in the prompt; `_parse_answer`
                    # already tolerates a response that is not clean JSON.
                    self._use_output_format = False
                    kwargs = dict(kwargs)
                    config = dict(kwargs["output_config"])
                    config.pop("format", None)
                    if config:
                        kwargs["output_config"] = config
                    else:
                        kwargs.pop("output_config")
                    continue
                raise
        raise RuntimeError("exhausted retries downgrading optional API features")

    async def answer(self, *, query: str, persona: Persona, sector: Sector,
                     toolbox: McpToolbox, max_rounds: int) -> AgentAnswer:
        tools = await toolbox.discover()
        system = build_system_prompt(persona, sector, toolbox.tool_names)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": build_user_prompt(query, persona, sector)}
        ]

        request: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": system,
            "tools": tools,
            "output_config": {
                "effort": self.settings.effort,
                "format": {"type": "json_schema", "schema": answer_json_schema()},
            },
        }

        response = None
        for _ in range(max_rounds):
            response = await self._create(messages=messages, **request)

            if response.stop_reason == "refusal":
                return _refusal_answer(response, persona)

            if response.stop_reason == "pause_turn":
                # A server-side tool ran long; resend to let it continue.
                messages.append({"role": "assistant", "content": response.content})
                continue

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                break

            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in tool_uses:
                # Tool inputs are parsed JSON from the SDK; never string-matched.
                payload, ok = await toolbox.call(block.name, dict(block.input or {}))
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": payload,
                    **({"is_error": True} if not ok else {}),
                })
            messages.append({"role": "user", "content": results})
        else:
            # Budget exhausted with tools still pending. Tools are forced off
            # for this final call: leaving them available lets the model spend
            # the turn on another tool call and return a response with no text
            # block at all.
            _append_final_nudge(messages)
            response = await self._create(
                messages=messages, tool_choice={"type": "none"}, **request)

        return _parse_answer(response, persona)


_FINAL_NUDGE = (
    "You have used the full tool budget and no more tool calls are available. "
    "Answer now from what you have already retrieved, and record the gap in "
    "caveats."
)

#: Error text that means "this account/model does not accept that feature",
#: as opposed to a transport or auth failure that must not be swallowed.
_REJECTION_SIGNALS = ("unexpected keyword", "unsupported", "not supported",
                      "invalid_request", "400", "unrecognized", "unknown field")
_FALLBACK_MARKERS = ("beta", "fallback")
_OUTPUT_FORMAT_MARKERS = ("output_config", "output_format", "json_schema",
                          "schema")


def _rejects_feature(exc: Exception, markers: tuple[str, ...]) -> bool:
    """True when the error names this feature AND reads as a rejection.

    Both halves matter. Matching a feature name alone would treat an unrelated
    network error whose message happens to contain "schema" as a reason to
    silently downgrade; requiring a rejection signal keeps auth failures,
    timeouts and rate limits propagating to the caller.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    return (any(m in text for m in markers)
            and any(s in text for s in _REJECTION_SIGNALS))


def _append_final_nudge(messages: list[dict[str, Any]]) -> None:
    """Ask for the answer without breaking user/assistant alternation.

    The Messages API rejects two consecutive user turns, so where the
    conversation already ends with one -- the usual case, a tool-result turn --
    the instruction is appended to that turn's content instead of being sent as
    a new message. After a `pause_turn` the last turn is the assistant's, and a
    fresh user message is the correct shape.
    """
    if not messages:
        messages.append({"role": "user", "content": _FINAL_NUDGE})
        return

    last = messages[-1]
    if last.get("role") != "user":
        messages.append({"role": "user", "content": _FINAL_NUDGE})
        return

    content = last.get("content")
    nudge = {"type": "text", "text": _FINAL_NUDGE}
    if isinstance(content, list):
        messages[-1] = {"role": "user", "content": [*content, nudge]}
    else:
        messages[-1] = {
            "role": "user",
            "content": [{"type": "text", "text": str(content)}, nudge],
        }


def _refusal_answer(response: Any, persona: Persona) -> AgentAnswer:
    details = getattr(response, "stop_details", None)
    reason = getattr(details, "explanation", None) or "the request was declined"
    return AgentAnswer(
        answer=f"I can't answer this one: {reason}.",
        confidence="low",
        caveats=["The model declined this request; no analysis was produced."],
    )


def _parse_answer(response: Any, persona: Persona) -> AgentAnswer:
    """Read the structured JSON the model was constrained to produce."""
    text = next((b.text for b in response.content if b.type == "text"), "")
    if not text.strip():
        return AgentAnswer(
            answer="The model returned no text. Retry the request.",
            confidence="low",
            caveats=["Empty model response."])
    try:
        return AgentAnswer.model_validate_json(text)
    except Exception:  # noqa: BLE001
        # output_config guarantees valid JSON, but never let a parse failure
        # lose an answer the user could still read.
        return AgentAnswer(
            answer=text,
            confidence="low",
            caveats=["Response did not match the structured schema; the raw "
                     "model text is shown and the structured fields are empty."])


# ---------------------------------------------------------------------------
# Deterministic
# ---------------------------------------------------------------------------

#: Rough detector for company mentions in a question: quoted strings, all-caps
#: ticker-like tokens, and capitalised multi-word names. It is a heuristic and
#: is described as one -- its only job is to decide what to look up, and
#: `find_company` makes the actual in/out-of-database call.
_MENTION_RE = re.compile(
    r'"([^"]{2,40})"'
    r"|\b([A-Z]{2,5})\b"
    r"|\b((?:[A-Z][a-z]+)(?:\s+(?:[A-Z][a-z]+|&|of|and)){0,3})\b"
)

#: Sector words that this database does not cover. A question naming one is a
#: scope mismatch: the sector is a request parameter, so asking about biotech
#: while configured for tech would otherwise be answered with tech companies and
#: no indication that the question was not the one answered. The LLM provider
#: handles this from its system prompt; the deterministic provider needs the
#: list. Not exhaustive by design -- it names the sectors a user is most
#: likely to reach for, and `list_sectors` remains the authority on coverage.
_UNCOVERED_SECTOR_TERMS = (
    "biotech", "biotechnology", "pharma", "pharmaceutical", "healthcare",
    "health care", "energy", "oil", "gas", "utilities", "utility",
    "real estate", "reit", "financials", "banking", "banks", "insurance",
    "materials", "mining", "telecom", "telecommunications", "media",
    "agriculture", "hospitality", "restaurants",
)

#: Question shapes that should be answered from stored signals rather than
#: from the sector screen. The LLM provider decides this for itself by reading
#: the tool descriptions; the deterministic provider needs the rule spelled out.
_SIGNAL_QUERY_TERMS = (
    # Matched as substrings, so "employ" also covers employs / employed /
    # employee / employees / employment. An earlier version listed only the
    # noun forms and missed "how many people does X employ" -- the eval caught
    # it, which is the whole reason the eval phrases things differently from
    # the code.
    "headcount", "head count", "employ", "hiring", "hire", "recruit",
    "workforce", "staff", "payroll", "fte", "layoff", "attrition",
    "how many people",
)

_STOPWORDS = {
    "I", "The", "A", "An", "Which", "What", "Who", "Where", "When", "Why",
    "How", "Is", "Are", "Do", "Does", "If", "Walk", "Give", "Tell", "Show",
    "Should", "Would", "Could", "This", "That", "These", "Those", "PE", "MF",
    "EBITDA", "TTM", "USD", "CEO", "CFO", "IPO", "ROI",
}


def _uncovered_sector_note(query: str, sector: Sector,
                           toolbox: McpToolbox) -> str | None:
    """Flag a question aimed at a sector this database does not hold.

    Sector is a request parameter, so "how is biotech shaping up?" asked with
    `sector=tech` would otherwise be answered with technology companies and no
    sign that a different question was asked. Returning a note rather than
    refusing keeps the answer useful while making the mismatch explicit.
    """
    text = query.lower()
    covered = {sector.id.lower(), sector.label.lower()}
    hits = [
        term for term in _UNCOVERED_SECTOR_TERMS
        if term in text and not any(term in c for c in covered)
    ]
    if not hits:
        return None
    return (
        f"Note on scope: this question mentions {hits[0]}, which is not one of "
        f"the sectors loaded in this database. The answer below covers "
        f"{sector.label} only. Call list_sectors for the full coverage list."
    )


def _candidate_mentions(query: str) -> list[str]:
    out: list[str] = []
    for match in _MENTION_RE.finditer(query):
        token = next((g for g in match.groups() if g), "").strip(" .,?!")
        if not token or token in _STOPWORDS or len(token) < 2:
            continue
        if token not in out:
            out.append(token)
    return out[:6]


class DeterministicProvider:
    """Composes an answer from MCP tool output with no model in the loop."""

    def __init__(self, settings: Settings, reason: str | None = None) -> None:
        self.settings = settings
        self.name = "deterministic"
        self.model = None
        #: Why this provider is answering, when it is standing in for a model.
        self.reason = reason

    def _provider_note(self) -> str:
        """Say plainly that no model reasoned over this, and why."""
        base = ("Answer composed by the deterministic provider; no language "
                "model reasoned over this evidence.")
        if self.reason == "daily_budget_exhausted":
            return (base + " The daily model-request budget for this deployment "
                    "is spent; it resets at 00:00 UTC. Retrieval, ranking and "
                    "evidence are unaffected.")
        return base

    async def _answer_signals(self, query: str, persona: Persona,
                              sector: Sector, toolbox: McpToolbox,
                              tickers: list[str],
                              out_of_scope: list[str]) -> AgentAnswer:
        """Answer a workforce question strictly from stored signals."""
        lines: list[str] = []
        evidence: list[Evidence] = []
        caveats: list[str] = []
        held = 0

        unique_tickers = list(dict.fromkeys(tickers))[:4]
        for ticker in unique_tickers:
            payload = await toolbox.call_json(
                "get_company_signals", {"ticker": ticker, "signal_type": "headcount"})
            signals = payload.get("signals") or []
            name = payload.get("name", ticker)
            if not signals:
                lines.append(
                    f"{name} ({ticker}): no headcount or hiring signal is held "
                    f"in this database.")
                if payload.get("guidance"):
                    caveats.append(payload["guidance"])
                continue
            held += 1
            for sig in signals[:3]:
                lines.append(
                    f"{name} ({ticker}): {sig.get('value_text') or sig.get('value_num')} "
                    f"(as of {sig.get('as_of') or 'unknown date'}, "
                    f"source: {sig.get('source_name') or 'unrecorded'}).")
                evidence.append(Evidence(
                    company=ticker, metric="headcount",
                    value=sig.get("value_num"), unit="count",
                    context=sig.get("value_text"), as_of=sig.get("as_of"),
                    source=sig.get("source_name")))

        if held == 0:
            lines.append("")
            lines.append(
                "Nothing is being estimated here. The loaded ingest adapter "
                "carries market and financial data but no workforce data, so "
                "the honest answer is that this signal is absent. Running the "
                "EDGAR adapter populates headcount for filers that tag "
                "dei:EntityNumberOfEmployees.")

        caveats.append(self._provider_note())
        if out_of_scope:
            caveats.append(
                "These were named in the question but are not in the database: "
                + ", ".join(out_of_scope))

        return AgentAnswer(
            answer="\n".join(lines),
            companies_referenced=unique_tickers,
            evidence=evidence, caveats=caveats[:6],
            out_of_scope=out_of_scope,
            confidence="high" if held else "low")

    async def answer(self, *, query: str, persona: Persona, sector: Sector,
                     toolbox: McpToolbox, max_rounds: int) -> AgentAnswer:
        out_of_scope: list[str] = []
        resolved: list[str] = []
        for mention in _candidate_mentions(query):
            found = await toolbox.call_json("find_company", {"query": mention})
            if found.get("in_database") is False:
                out_of_scope.append(mention)
            else:
                resolved.extend(m["ticker"] for m in found.get("matches", [])[:2])

        # A question about workforce is a signals question, not a screening
        # question. Answer it from what is stored -- including when nothing is.
        if any(term in query.lower() for term in _SIGNAL_QUERY_TERMS) and resolved:
            return await self._answer_signals(
                query, persona, sector, toolbox, resolved, out_of_scope)

        scope_note = _uncovered_sector_note(query, sector, toolbox)

        screen = await toolbox.call_json(
            "screen_sector",
            {"sector": sector.id, "persona": persona.id, "limit": 5})
        benchmarks = await toolbox.call_json(
            "get_sector_benchmarks", {"sector": sector.id})
        coverage = await toolbox.call_json(
            "describe_data_coverage", {"sector": sector.id})

        results = screen.get("results", [])
        if not results:
            return AgentAnswer(
                answer=(f"No companies are loaded for {sector.label}, so there "
                        f"is nothing to analyse. Rebuild the database for this "
                        f"sector before asking again."),
                confidence="low",
                caveats=["Empty sector universe."],
                out_of_scope=out_of_scope)

        medians = {b["metric_code"]: b for b in benchmarks.get("benchmarks", [])}
        evidence: list[Evidence] = []
        lines: list[str] = []

        if scope_note:
            lines.append(scope_note)
            lines.append("")

        weight_summary = ", ".join(
            f"{metric} {spec['weight']:.0%} ({spec['direction']})"
            for metric, spec in
            screen.get("ranking_basis", {}).get("weights", {}).items())

        lines.append(
            f"Reading {sector.label} as a {persona.label}, across the "
            f"{screen.get('universe_size', 0)} companies loaded for this "
            f"sector. The ranking below comes from this persona's own "
            f"weighting -- {weight_summary} -- not a generic screen.")
        lines.append("")
        lines.append(persona.answer_sections[0])

        for record in results:
            drivers = record.get("contributions", [])[:3]
            driver_text = "; ".join(
                f"{d['metric']} {fmt_value(d['value'], d.get('unit'))} "
                f"({pct_rank(d['percentile_in_sector'])})"
                for d in drivers)
            lines.append(
                f"  {record['rank']}. {record['name']} ({record['ticker']}) "
                f"- score {record['score']:.2f}. Driven by: {driver_text}.")
            for d in drivers:
                evidence.append(Evidence(
                    company=record["ticker"], metric=d["metric"],
                    value=d["value"], unit=d.get("unit"),
                    context=(f"{pct_rank(d['percentile_in_sector'])} "
                             f"in {sector.id}; weight {d['weight']} "
                             f"({d['direction']} is better)"),
                    is_derived=bool(d.get("is_derived")),
                    as_of=d.get("period_end")))

        lines.append("")
        lines.append("Sector anchors")
        for code in persona.priority_metrics[:4]:
            bench = medians.get(code)
            if bench:
                lines.append(
                    f"  - median {code}: "
                    f"{fmt_value(bench['median'], bench.get('unit'))} "
                    f"(n={bench['n']}, IQR "
                    f"{fmt_value(bench.get('p25'), bench.get('unit'))} to "
                    f"{fmt_value(bench.get('p75'), bench.get('unit'))})")
                evidence.append(Evidence(
                    metric=f"{code}_sector_median", value=bench["median"],
                    unit=bench.get("unit"),
                    context=f"median across {bench['n']} {sector.id} companies",
                    is_derived=True))

        lines.append("")
        lines.append(
            "This is a deterministic screen, not a written analyst view: it "
            "reports the ranking and the numbers behind it without "
            "interpretation. Set SECTORLENS_LLM_PROVIDER=anthropic with an API "
            "key for reasoned commentary in the persona's voice.")

        caveats = [f["detail"] for f in coverage.get("data_quality_findings", [])
                   if f["severity"] in ("warn", "error")][:5]
        if scope_note:
            caveats.insert(0, scope_note)
        caveats.append(self._provider_note())
        if out_of_scope:
            caveats.append(
                "These were named in the question but are not in the database: "
                + ", ".join(out_of_scope))

        return AgentAnswer(
            answer="\n".join(lines),
            companies_referenced=[r["ticker"] for r in results],
            evidence=evidence[:20],
            caveats=caveats,
            out_of_scope=out_of_scope,
            confidence="low")


#: Process-wide, so every entry point shares one allowance.
_BUDGET: RequestBudget | None = None


def get_budget(settings: Settings) -> RequestBudget:
    global _BUDGET
    if _BUDGET is None:
        _BUDGET = RequestBudget(settings.daily_llm_budget)
    return _BUDGET


def build_provider(settings: Settings) -> LLMProvider:
    """Pick a provider for one request.

    Falls back to the deterministic provider when no key is configured OR when
    the day's model budget is spent. Both are degradations rather than
    failures: the retrieval, the persona weighting and the evidence are
    unaffected, and the response says which provider answered.
    """
    provider = settings.effective_provider()
    if provider != "anthropic":
        return DeterministicProvider(settings)
    if not get_budget(settings).try_spend():
        return DeterministicProvider(settings, reason="daily_budget_exhausted")
    return AnthropicProvider(settings)
