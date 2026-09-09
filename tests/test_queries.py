"""Retrieval behaviour, including the persona weighting and the honesty paths."""

from __future__ import annotations

import sqlite3

import pytest

from sectorlens.db.queries import (compare_companies, describe_data_coverage,
                                find_company, get_company_profile,
                                get_company_signals, get_sector_benchmarks,
                                list_companies, list_sectors, screen_sector)


def test_sectors_report_loaded_counts(fixture_conn):
    by_id = {s["id"]: s for s in list_sectors(fixture_conn)["sectors"]}
    assert by_id["tech"]["companies_loaded"] == 4
    assert by_id["logistics"]["companies_loaded"] == 3
    # Configured but unpopulated sectors must still be listed, at zero.
    assert by_id["retail"]["companies_loaded"] == 0


def test_find_company_exact_and_partial(fixture_conn):
    assert find_company(fixture_conn, "BIGC")["match_type"] == "exact"
    assert find_company(fixture_conn, "big")["match_type"] == "partial"


def test_find_company_reports_absence_without_guessing(fixture_conn):
    result = find_company(fixture_conn, "Ferrari")
    assert result["in_database"] is False
    assert result["matches"] == []
    # The guidance is what stops the model reaching for prior knowledge.
    assert "do not describe this company from prior knowledge" in \
        result["guidance"].lower()


def test_company_signals_present(fixture_conn):
    result = get_company_signals(fixture_conn, "FRGT", "headcount")
    assert result["count"] == 1
    assert result["signals"][0]["value_num"] == 310_000


def test_company_signals_absent_returns_explicit_guidance(fixture_conn):
    """Absence must be unambiguous, or the model will fill the gap itself."""
    result = get_company_signals(fixture_conn, "HAUL", "headcount")
    assert result["count"] == 0
    assert result["signals"] == []
    assert "holds no headcount signal" in result["guidance"]
    assert "do not estimate" in result["guidance"].lower()


def test_unknown_sector_is_rejected_not_guessed(fixture_conn):
    result = list_companies(fixture_conn, "biotech")
    assert result["error"] == "unknown_sector"
    assert "tech" in result["valid_sectors"]


# ---------------------------------------------------------------------------
# The central claim: persona changes retrieval, not just tone
# ---------------------------------------------------------------------------

def test_personas_rank_the_same_sector_differently(fixture_conn):
    mf = screen_sector(fixture_conn, "tech", "mutual_fund_analyst", limit=4)
    pe = screen_sector(fixture_conn, "tech", "pe_analyst", limit=4)
    eq = screen_sector(fixture_conn, "tech", "equity_analyst", limit=4)

    mf_order = [r["ticker"] for r in mf["results"]]
    pe_order = [r["ticker"] for r in pe["results"]]
    eq_order = [r["ticker"] for r in eq["results"]]

    assert mf_order != pe_order, "MF and PE must not produce the same ranking"
    assert mf_order != eq_order or eq_order != pe_order


def test_mutual_fund_prefers_the_large_compounder_and_pe_prefers_the_small_cheap_one(
        fixture_conn):
    """The inversion is the design, so it is asserted directly.

    BIGC is large, expensive and high margin. SMLC is small, cheap and low
    margin -- a poor core holding and a plausible take-private.
    """
    mf = screen_sector(fixture_conn, "tech", "mutual_fund_analyst", limit=4)
    pe = screen_sector(fixture_conn, "tech", "pe_analyst", limit=4)

    mf_rank = {r["ticker"]: r["rank"] for r in mf["results"]}
    pe_rank = {r["ticker"]: r["rank"] for r in pe["results"]}

    assert mf_rank["BIGC"] < mf_rank["SMLC"], \
        "the mutual fund lens should favour the large compounder"
    assert pe_rank["SMLC"] < pe_rank["BIGC"], \
        "the PE lens should favour the small, cheap operator"


def test_screen_explains_itself(fixture_conn):
    result = screen_sector(fixture_conn, "tech", "pe_analyst", limit=2)
    top = result["results"][0]
    assert top["contributions"], "each result must show what drove its score"
    for c in top["contributions"]:
        assert set(c) >= {"metric", "value", "percentile_in_sector", "weight",
                          "direction", "contribution"}
        assert 0.0 <= c["percentile_in_sector"] <= 1.0
    assert "weights" in result["ranking_basis"]
    assert result["ranking_basis"]["rationale"]


def test_screen_scores_are_bounded(fixture_conn):
    for persona in ("mutual_fund_analyst", "equity_analyst", "pe_analyst"):
        result = screen_sector(fixture_conn, "tech", persona, limit=10)
        for record in result["results"]:
            assert 0.0 <= record["score"] <= 1.0


def test_screen_rejects_unknown_persona(fixture_conn):
    assert screen_sector(fixture_conn, "tech", "day_trader")["error"] == \
        "unknown_persona"


# ---------------------------------------------------------------------------
# Benchmarks, comparison, coverage
# ---------------------------------------------------------------------------

def test_sector_benchmarks_are_computed(fixture_conn):
    result = get_sector_benchmarks(fixture_conn, "tech", ["ebitda_margin"])
    assert len(result["benchmarks"]) == 1
    bench = result["benchmarks"][0]
    assert bench["n"] == 4
    assert bench["p25"] <= bench["median"] <= bench["p75"]


def test_relative_metric_inverts_sign_as_documented(fixture_conn):
    """A company below the sector median must show a POSITIVE gap, because the
    PE persona scores that direction as headroom."""
    rows = {
        r["ticker"]: r["value"]
        for r in (dict(x) for x in fixture_conn.execute(
            """SELECT ticker, value FROM v_company_facts
               WHERE metric_code='ebitda_margin_gap_to_sector' AND sector='tech'"""))
    }
    # BIGC margin 36% is above the tech median; SMLC at 8% is below.
    assert rows["BIGC"] < 0
    assert rows["SMLC"] > 0


def test_compare_companies_reports_missing_tickers(fixture_conn):
    result = compare_companies(fixture_conn, ["BIGC", "NOPE"], ["ebitda_margin"])
    assert "BIGC" in result["companies"]
    assert result["not_in_database"] == ["NOPE"]


def test_coverage_reports_sources_and_findings(fixture_conn):
    result = describe_data_coverage(fixture_conn)
    assert result["coverage_by_sector"]
    assert result["sources"]
    assert "data_quality_findings" in result


def test_profile_marks_derived_values(fixture_conn):
    profile = get_company_profile(fixture_conn, "BIGC")
    assert profile["company"]["ticker"] == "BIGC"
    assert profile["metrics"]
    derived = [m for m in profile["metrics"] if m["is_derived"]]
    assert derived, "sector-relative metrics should be flagged as derived"
    assert all(m["derivation"] for m in derived)


def test_profile_of_unknown_ticker_falls_through_to_absence(fixture_conn):
    assert get_company_profile(fixture_conn, "NOPE")["in_database"] is False
