"""Scoring an answer against what its persona actually promised.

The existing tests prove the three personas *retrieve* different companies.
That is necessary and not sufficient: an answer can pull the right rows and
still fail to reason like the role, or quietly invent a number the tools never
returned. These checks put a measurable number on both, so a prompt change can
be compared against a previous run instead of eyeballed.

Three dimensions, deliberately weighted differently:

* grounding   -- did it stay inside the data? A failure here is disqualifying,
                 so any critical failure zeroes the case. Fluent invention is
                 worse than an unhelpful answer.
* discipline  -- did it retrieve, cite and qualify at all?
* persona     -- did it engage with the concepts the role is defined by?
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..agent.schemas import AgentResponse

SEVERITY_WEIGHT = {"critical": 4, "major": 2, "minor": 1}

#: A headcount-sized number sitting next to a workforce word. Requires three or
#: more digits (or a thousands separator) so honest prose like "0 companies
#: carry a headcount signal" is not mistaken for a fabricated figure.
_HEADCOUNT_CLAIM = re.compile(
    # a headcount-sized number followed by a workforce noun...
    r"(\d[\d,]{2,}\s*(?:\+|k|thousand|million)?\s*"
    r"(?:employees|staff|workers|people|headcount|ftes?)"
    # ...or a workforce word followed closely by one.
    r"|(?:employs|employed|employees|staff|headcount|workforce|workers)"
    r"\D{0,25}\d[\d,]{2,})",
    re.IGNORECASE)


@dataclass
class CheckResult:
    id: str
    dimension: str
    severity: str
    passed: bool
    detail: str

    @property
    def weight(self) -> int:
        return SEVERITY_WEIGHT[self.severity]


@dataclass
class CaseResult:
    case_id: str
    persona: str
    sector: str
    query: str
    response: AgentResponse
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def failed(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    @property
    def has_critical_failure(self) -> bool:
        return any(c.severity == "critical" and not c.passed for c in self.checks)

    @property
    def score(self) -> float:
        """0.0-1.0. Any critical failure zeroes the case.

        The mean of the DIMENSION scores rather than of the individual checks.
        Grounding and discipline carry many more checks than persona, so a flat
        per-check mean lets a strong grounding score paper over prose that
        never reasons like the role -- which is the thing this eval exists to
        measure.
        """
        if not self.checks or self.has_critical_failure:
            return 0.0
        scores = [s for s in (self.dimension_score(d)
                              for d in ("grounding", "discipline", "persona"))
                  if s is not None]
        return sum(scores) / len(scores) if scores else 0.0

    def dimension_score(self, dimension: str) -> float | None:
        subset = [c for c in self.checks if c.dimension == dimension]
        if not subset:
            return None
        return sum(c.weight for c in subset if c.passed) / sum(c.weight for c in subset)


# ---------------------------------------------------------------------------
# Database-backed facts the checks need
# ---------------------------------------------------------------------------

class Universe:
    """What actually exists, so 'invented' can be decided rather than guessed."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.tickers = {
            r[0].upper() for r in conn.execute("SELECT ticker FROM companies")
        }
        self.names = {
            r[0].lower() for r in conn.execute("SELECT name FROM companies")
        }
        self.headcount_tickers = {
            r[0].upper() for r in conn.execute(
                """SELECT c.ticker FROM company_signals s
                   JOIN companies c ON c.id = s.company_id
                   WHERE s.signal_type = 'headcount'""")
        }

    def is_known(self, token: str) -> bool:
        return token.upper() in self.tickers or token.lower() in self.names


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _check(id: str, dimension: str, severity: str, passed: bool,
           detail: str) -> CheckResult:
    return CheckResult(id=id, dimension=dimension, severity=severity,
                       passed=passed, detail=detail)


def grounding_checks(response: AgentResponse, expects: dict[str, Any],
                     universe: Universe) -> list[CheckResult]:
    checks: list[CheckResult] = []

    invented = [t for t in response.companies_referenced
                if not universe.is_known(t)]
    checks.append(_check(
        "no_invented_companies", "grounding", "critical", not invented,
        "all referenced tickers exist in the database" if not invented
        else f"referenced companies absent from the database: {invented}"))

    bad_evidence = [e.company for e in response.evidence
                    if e.company and not universe.is_known(e.company)]
    checks.append(_check(
        "evidence_companies_exist", "grounding", "critical", not bad_evidence,
        "evidence is attributed to real companies" if not bad_evidence
        else f"evidence cites unknown companies: {bad_evidence}"))

    for expected in expects.get("out_of_scope_contains", []):
        declared = [o.lower() for o in response.out_of_scope]
        hit = any(expected.lower() in o for o in declared)
        checks.append(_check(
            f"declares_out_of_scope[{expected}]", "grounding", "critical", hit,
            f"{expected!r} declared out of scope" if hit
            else f"{expected!r} was asked about but never declared out of scope"))

    for forbidden in expects.get("not_referenced", []):
        referenced = [t.lower() for t in response.companies_referenced]
        clean = not any(forbidden.lower() in t for t in referenced)
        checks.append(_check(
            f"did_not_analyse[{forbidden}]", "grounding", "critical", clean,
            f"{forbidden!r} was not analysed" if clean
            else f"{forbidden!r} is absent from the database yet was analysed"))

    if expects.get("no_invented_headcount"):
        cited_without_signal = [
            t for t in response.companies_referenced
            if t.upper() not in universe.headcount_tickers]
        match = _HEADCOUNT_CLAIM.search(response.answer)
        # Only a violation when a number appears for a company we hold none for.
        violated = bool(match) and bool(cited_without_signal)
        checks.append(_check(
            "no_invented_headcount", "grounding", "critical", not violated,
            "no headcount figure asserted without a stored signal" if not violated
            else f"asserted {match.group(0)!r} for a company with no stored "
                 f"headcount signal"))

    if expects.get("admits_missing_signal"):
        text = response.answer.lower()
        admitted = any(p in text for p in (
            "no headcount", "no workforce", "not held", "no signal",
            "does not hold", "don't hold", "do not hold", "no data"))
        checks.append(_check(
            "admits_missing_signal", "grounding", "critical", admitted,
            "states plainly that the signal is not held" if admitted
            else "never states that the requested signal is unavailable"))

    if expects.get("mentions_scope_limit"):
        text = response.answer.lower()
        # General English for "that is outside what I hold". Deliberately not
        # matched against this repository's own wording, so the check still
        # means something if the phrasing changes or a model writes its own.
        mentioned = any(p in text for p in (
            "not one of", "not in", "no data", "don't cover", "do not cover",
            "not covered", "outside", "out of scope", "not loaded",
            "only cover", "covers only", "not available", "not a sector",
            "cannot answer", "can't answer"))
        checks.append(_check(
            "mentions_scope_limit", "grounding", "major", mentioned,
            "acknowledges what it covers" if mentioned
            else "answered an out-of-scope sector without flagging scope"))

    return checks


def discipline_checks(response: AgentResponse,
                      expects: dict[str, Any]) -> list[CheckResult]:
    checks: list[CheckResult] = []

    used_tools = bool(response.tool_calls)
    checks.append(_check(
        "retrieved_something", "discipline", "critical", used_tools,
        f"made {len(response.tool_calls)} MCP call(s)" if used_tools
        else "answered without calling a single tool"))

    all_ok = all(c.ok for c in response.tool_calls)
    checks.append(_check(
        "tool_calls_succeeded", "discipline", "minor", all_ok,
        "all tool calls succeeded" if all_ok
        else f"failed calls: {[c.tool for c in response.tool_calls if not c.ok]}"))

    min_companies = expects.get("min_companies_referenced", 0)
    if min_companies:
        got = len(response.companies_referenced)
        checks.append(_check(
            "companies_referenced", "discipline", "major", got >= min_companies,
            f"referenced {got} companies (>= {min_companies})"))

    min_evidence = expects.get("min_evidence", 0)
    if min_evidence:
        got = len(response.evidence)
        checks.append(_check(
            "evidence_supplied", "discipline", "major", got >= min_evidence,
            f"supplied {got} evidence items (>= {min_evidence})"))

    wanted_metrics = expects.get("cites_metrics", [])
    if wanted_metrics:
        cited = {e.metric for e in response.evidence}
        missing = [m for m in wanted_metrics
                   if not any(m in c for c in cited)]
        checks.append(_check(
            "cites_expected_metrics", "discipline", "major", not missing,
            "cited the metrics this question turns on" if not missing
            else f"never cited: {missing}"))

    checks.append(_check(
        "qualified_the_answer", "discipline", "minor", bool(response.caveats),
        f"carried {len(response.caveats)} caveat(s)" if response.caveats
        else "returned no caveats despite known data limitations"))

    # Confidence is a claim like any other; "high" on thin evidence is a
    # calibration failure even when every individual number is real.
    overconfident = (response.confidence == "high" and len(response.evidence) < 2)
    checks.append(_check(
        "confidence_calibrated", "discipline", "minor", not overconfident,
        f"confidence {response.confidence!r} is consistent with the evidence"
        if not overconfident else
        "claimed high confidence on fewer than two pieces of evidence"))

    return checks


def _body_text(answer: str, answer_sections: Sequence[str]) -> str:
    """The answer minus its own section headings.

    Without this the rubric grades its own scaffolding: the PE persona's
    section heading is literally "Deal shape and entry multiple", so any answer
    that prints the headings scores a match on `entry_multiple` while saying
    nothing about one. Removing the headings makes the check measure the body.
    """
    headings = {s.strip().lower() for s in answer_sections}
    kept = [
        line for line in answer.splitlines()
        if line.strip().lower() not in headings
    ]
    return "\n".join(kept).lower()


def persona_checks(response: AgentResponse, rubric: Sequence[dict[str, Any]],
                   answer_sections: Sequence[str] = ()) -> list[CheckResult]:
    """Did the prose actually engage with the concepts the role is defined by?

    Matched as "any term from each group", so a persona may express an idea in
    its own words instead of hitting one exact phrase.
    """
    text = _body_text(response.answer, answer_sections)
    checks: list[CheckResult] = []
    for group in rubric:
        hits = [term for term in group["any_of"] if term.lower() in text]
        checks.append(_check(
            f"persona_concept[{group['name']}]", "persona", "major", bool(hits),
            f"engaged with {group['name']} via {hits[:2]}" if hits
            else f"never engaged with {group['name']}; expected one of "
                 f"{group['any_of'][:4]}"))
    return checks


def divergence_check(results: Iterable[CaseResult]) -> CheckResult:
    """Cross-persona divergence -- the property the whole design rests on.

    Compares the sets of companies each persona chose to talk about. Identical
    sets mean the personas collapsed into one voice over the same evidence,
    which is the failure this project exists to avoid.
    """
    results = list(results)
    sets = {r.persona: {t.upper() for t in r.response.companies_referenced}
            for r in results}
    populated = {p: s for p, s in sets.items() if s}
    if len(populated) < 2:
        return _check("cross_persona_divergence", "persona", "major", False,
                      "fewer than two personas returned any companies")

    pairs = []
    personas = sorted(populated)
    for i, a in enumerate(personas):
        for b in personas[i + 1:]:
            union = populated[a] | populated[b]
            overlap = len(populated[a] & populated[b]) / len(union) if union else 1.0
            pairs.append((f"{a}/{b}", overlap))

    worst_pair, worst = max(pairs, key=lambda p: p[1])
    passed = worst < 0.75
    detail = ("persona rankings diverge (highest pairwise overlap "
              f"{worst:.0%} on {worst_pair})") if passed else (
        f"personas returned near-identical companies: {worst:.0%} overlap on "
        f"{worst_pair} -- the persona is not changing retrieval")
    return _check("cross_persona_divergence", "persona", "critical", passed, detail)
