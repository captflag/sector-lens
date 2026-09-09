"""Tests for the evaluation harness itself.

An eval that scores a bad answer well is worse than no eval, so the checks are
driven with deliberately bad responses to confirm they actually fail.
"""

from __future__ import annotations

import pytest

from sectorlens.agent.schemas import AgentResponse, Evidence, ToolCallRecord
from sectorlens.evals.rubric import (CaseResult, Universe, _body_text,
                                  discipline_checks, divergence_check,
                                  grounding_checks, persona_checks)


@pytest.fixture
def universe(fixture_conn):
    return Universe(fixture_conn)


def make_response(**overrides) -> AgentResponse:
    base = dict(
        answer="An answer.", companies_referenced=["BIGC"],
        evidence=[Evidence(company="BIGC", metric="ebitda_margin", value=0.36)],
        caveats=["a caveat"], out_of_scope=[], confidence="medium",
        persona="pe_analyst", persona_label="PE Analyst",
        sector="tech", sector_label="Technology", provider="test",
        tool_calls=[ToolCallRecord(tool="screen_sector")])
    base.update(overrides)
    return AgentResponse(**base)


def find(checks, check_id):
    return next(c for c in checks if c.id == check_id)


# ---------------------------------------------------------------------------
# Grounding -- the checks that must never be lenient
# ---------------------------------------------------------------------------

def test_invented_company_is_a_critical_failure(universe):
    response = make_response(companies_referenced=["BIGC", "ENRON"])
    check = find(grounding_checks(response, {}, universe), "no_invented_companies")
    assert not check.passed
    assert check.severity == "critical"
    assert "ENRON" in check.detail


def test_real_companies_pass_grounding(universe):
    checks = grounding_checks(make_response(), {}, universe)
    assert all(c.passed for c in checks)


def test_evidence_attributed_to_an_unknown_company_fails(universe):
    response = make_response(
        evidence=[Evidence(company="NOPE", metric="ebitda_margin", value=1.0)])
    assert not find(grounding_checks(response, {}, universe),
                    "evidence_companies_exist").passed


def test_undeclared_out_of_scope_company_fails(universe):
    response = make_response()
    check = find(
        grounding_checks(response, {"out_of_scope_contains": ["Ferrari"]}, universe),
        "declares_out_of_scope[Ferrari]")
    assert not check.passed


def test_declared_out_of_scope_company_passes(universe):
    response = make_response(out_of_scope=["Ferrari"])
    assert find(
        grounding_checks(response, {"out_of_scope_contains": ["Ferrari"]}, universe),
        "declares_out_of_scope[Ferrari]").passed


@pytest.mark.parametrize("answer", [
    "UPS employs roughly 490,000 people worldwide.",
    "Headcount is about 500000 staff.",
    "employees: 145,000 at last count",
])
def test_fabricated_headcount_is_caught(universe, answer):
    """Fabricated headcount is the failure mode this whole design exists to stop."""
    response = make_response(answer=answer, companies_referenced=["MIDC"])
    check = find(
        grounding_checks(response, {"no_invented_headcount": True}, universe),
        "no_invented_headcount")
    assert not check.passed, f"failed to catch fabrication in {answer!r}"


@pytest.mark.parametrize("answer", [
    "No headcount signal is held for this company.",
    "0 companies in the database carry a headcount signal.",
    "This database holds no workforce data at all.",
])
def test_honest_absence_is_not_mistaken_for_fabrication(universe, answer):
    response = make_response(answer=answer, companies_referenced=["MIDC"])
    assert find(
        grounding_checks(response, {"no_invented_headcount": True}, universe),
        "no_invented_headcount").passed, f"false positive on {answer!r}"


def test_headcount_figure_is_allowed_when_the_signal_is_stored(universe):
    """FRGT carries a real 310,000 headcount signal in the fixture."""
    response = make_response(
        answer="Freight Co reports 310,000 employees.",
        companies_referenced=["FRGT"])
    assert find(
        grounding_checks(response, {"no_invented_headcount": True}, universe),
        "no_invented_headcount").passed


def test_missing_signal_must_be_admitted(universe):
    silent = make_response(answer="Freight looks well positioned.")
    assert not find(
        grounding_checks(silent, {"admits_missing_signal": True}, universe),
        "admits_missing_signal").passed

    honest = make_response(answer="I hold no headcount signal for this company.")
    assert find(
        grounding_checks(honest, {"admits_missing_signal": True}, universe),
        "admits_missing_signal").passed


# ---------------------------------------------------------------------------
# Discipline
# ---------------------------------------------------------------------------

def test_answering_without_any_tool_call_is_critical():
    check = find(discipline_checks(make_response(tool_calls=[]), {}),
                 "retrieved_something")
    assert not check.passed and check.severity == "critical"


def test_high_confidence_on_thin_evidence_is_flagged():
    thin = make_response(confidence="high", evidence=[])
    assert not find(discipline_checks(thin, {}), "confidence_calibrated").passed
    thick = make_response(confidence="high", evidence=[
        Evidence(metric="a", value=1), Evidence(metric="b", value=2)])
    assert find(discipline_checks(thick, {}), "confidence_calibrated").passed


def test_expected_metrics_must_actually_be_cited():
    response = make_response()
    expects = {"cites_metrics": ["ebitda_margin", "net_margin"]}
    check = find(discipline_checks(response, expects), "cites_expected_metrics")
    assert not check.passed
    assert "net_margin" in check.detail


# ---------------------------------------------------------------------------
# Persona -- including the anti-gaming fix
# ---------------------------------------------------------------------------

RUBRIC = [{"name": "entry_multiple", "any_of": ["entry multiple", "ev/ebitda"]}]


def test_section_headings_do_not_count_as_engaging_with_a_concept():
    """The PE persona's own heading is 'Deal shape and entry multiple'. Printing
    the heading must not score a match on the concept it names."""
    headings = ["Deal shape and entry multiple", "Exit path"]
    heading_only = make_response(
        answer="Deal shape and entry multiple\n  1. BIGC scored well.")
    assert not persona_checks(heading_only, RUBRIC, headings)[0].passed

    reasoned = make_response(
        answer="Deal shape and entry multiple\n  BIGC screens at an entry "
               "multiple of 5.1x, below the sector median.")
    assert persona_checks(reasoned, RUBRIC, headings)[0].passed


def test_body_text_strips_only_exact_headings():
    body = _body_text("Exit path\nThe exit path is a trade sale.", ["Exit path"])
    assert "exit path is a trade sale" in body
    assert body.count("exit path") == 1


# ---------------------------------------------------------------------------
# Cross-persona divergence
# ---------------------------------------------------------------------------

def _case(persona: str, tickers: list[str]) -> CaseResult:
    return CaseResult(case_id="c", persona=persona, sector="tech", query="q",
                      response=make_response(companies_referenced=tickers))


def test_identical_persona_rankings_are_a_critical_failure():
    """If the personas converge on the same names, the design has failed."""
    check = divergence_check([
        _case("mutual_fund_analyst", ["AAA", "BBB", "CCC"]),
        _case("pe_analyst", ["AAA", "BBB", "CCC"]),
    ])
    assert not check.passed
    assert check.severity == "critical"


def test_divergent_persona_rankings_pass():
    assert divergence_check([
        _case("mutual_fund_analyst", ["AAA", "BBB"]),
        _case("pe_analyst", ["CCC", "DDD"]),
    ]).passed


def test_partial_overlap_is_tolerated():
    """Personas may agree on a name or two without having collapsed."""
    assert divergence_check([
        _case("mutual_fund_analyst", ["AAA", "BBB", "CCC", "DDD"]),
        _case("pe_analyst", ["AAA", "EEE", "FFF", "GGG"]),
    ]).passed


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def test_a_critical_failure_zeroes_the_case(universe):
    result = CaseResult(case_id="c", persona="pe_analyst", sector="tech",
                        query="q",
                        response=make_response(companies_referenced=["ENRON"]))
    result.checks.extend(grounding_checks(result.response, {}, universe))
    result.checks.extend(discipline_checks(result.response, {}))
    assert result.has_critical_failure
    assert result.score == 0.0


def test_score_is_the_mean_of_dimensions_not_of_checks(universe):
    """Grounding carries more checks than persona; a flat per-check mean would
    let a strong grounding score hide prose that never reasons like the role."""
    response = make_response(answer="Nothing role-specific here.")
    result = CaseResult(case_id="c", persona="pe_analyst", sector="tech",
                        query="q", response=response)
    result.checks.extend(grounding_checks(response, {}, universe))
    result.checks.extend(discipline_checks(response, {}))
    result.checks.extend(persona_checks(response, RUBRIC, []))

    assert result.dimension_score("grounding") == 1.0
    assert result.dimension_score("persona") == 0.0
    # Mean of dimensions (1.0, 1.0, 0.0) = 0.67, not the ~0.9 a check-count
    # weighted mean would give.
    assert result.score == pytest.approx(2 / 3, abs=0.01)
