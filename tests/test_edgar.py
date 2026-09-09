"""The EDGAR adapter's parsing and multi-year loading.

`data.sec.gov` is not reachable from CI, so the network call is not exercised
here. Everything downstream of it is: these tests feed the loader the exact
shape `companyfacts` returns and assert on what lands in the database.
"""

from __future__ import annotations

import pytest

from sectorlens.db.store import (connect, init_schema, start_run, upsert_company,
                                 upsert_source)
from sectorlens.ingest.edgar import (EdgarAdapter, _annual_series, _by_period,
                                     _resolve)


def annual(concept_rows):
    """Wrap {end: value} into the units/rows shape companyfacts uses."""
    return {"units": {"USD": [
        {"end": end, "val": val, "form": "10-K", "fp": "FY",
         "filed": f"{int(end[:4]) + 1}-02-15"}
        for end, val in concept_rows.items()
    ]}}


FOUR_YEARS = {
    "us-gaap": {
        # revenue compounding, margin expanding: 8% -> 15%
        "Revenues": annual({
            "2021-12-31": 1000.0, "2022-12-31": 1200.0,
            "2023-12-31": 1500.0, "2024-12-31": 1750.0}),
        "OperatingIncomeLoss": annual({
            "2021-12-31": 50.0, "2022-12-31": 90.0,
            "2023-12-31": 150.0, "2024-12-31": 200.0}),
        "DepreciationDepletionAndAmortization": annual({
            "2021-12-31": 30.0, "2022-12-31": 36.0,
            "2023-12-31": 45.0, "2024-12-31": 62.5}),
        "NetIncomeLoss": annual({
            "2021-12-31": 20.0, "2022-12-31": 45.0,
            "2023-12-31": 90.0, "2024-12-31": 130.0}),
        "LongTermDebtNoncurrent": annual({
            "2023-12-31": 400.0, "2024-12-31": 380.0}),
        "CashAndCashEquivalentsAtCarryingValue": annual({
            "2023-12-31": 100.0, "2024-12-31": 150.0}),
        "NetCashProvidedByUsedInOperatingActivities": annual({
            "2024-12-31": 240.0}),
        "PaymentsToAcquirePropertyPlantAndEquipment": annual({
            "2024-12-31": 90.0}),
    }
}


@pytest.fixture
def loaded(tmp_path):
    """Run the loader over the synthetic filer and hand back a connection."""
    conn = connect(tmp_path / "edgar.db")
    init_schema(conn)
    run_id = start_run(conn, "edgar-test")
    source_id = upsert_source(conn, key="edgar-test", name="test",
                              adapter="edgar-test")
    company_id = upsert_company(conn, ticker="TEST", name="Test Co",
                                sector="tech", source_id=source_id)
    EdgarAdapter(years=4)._load_company(
        conn, FOUR_YEARS, company_id=company_id, source_id=source_id,
        run_id=run_id)
    conn.commit()
    yield conn
    conn.close()


def values(conn, metric):
    return {r[0]: r[1] for r in conn.execute(
        "SELECT period_end, value FROM company_metrics WHERE metric_code = ?",
        (metric,))}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_only_annual_filings_are_taken():
    facts = {"us-gaap": {"Revenues": {"units": {"USD": [
        {"end": "2024-12-31", "val": 100, "form": "10-K", "fp": "FY", "filed": "2025-02-01"},
        {"end": "2024-09-30", "val": 70, "form": "10-Q", "fp": "Q3", "filed": "2024-10-01"},
    ]}}}}
    assert list(_by_period(facts, ("Revenues",))) == ["2024-12-31"]


def test_a_restatement_supersedes_the_original():
    facts = {"us-gaap": {"Revenues": {"units": {"USD": [
        {"end": "2023-12-31", "val": 100, "form": "10-K", "fp": "FY", "filed": "2024-02-01"},
        {"end": "2023-12-31", "val": 105, "form": "10-K", "fp": "FY", "filed": "2025-02-01"},
    ]}}}}
    assert _by_period(facts, ("Revenues",))["2023-12-31"] == 105


def test_concept_aliases_fall_through_in_order():
    facts = {"us-gaap": {"SalesRevenueNet": annual({"2024-12-31": 42.0})}}
    assert _resolve(facts, ("Revenues", "SalesRevenueNet"))[0]["val"] == 42.0


# ---------------------------------------------------------------------------
# Multi-year loading
# ---------------------------------------------------------------------------

def test_every_year_in_the_window_is_stored(loaded):
    revenue = values(loaded, "revenue_ttm")
    assert sorted(revenue) == ["2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31"]
    assert revenue["2024-12-31"] == 1750.0


def test_the_window_is_respected(tmp_path):
    conn = connect(tmp_path / "two.db")
    init_schema(conn)
    run_id = start_run(conn, "t")
    source_id = upsert_source(conn, key="t", name="t", adapter="t")
    cid = upsert_company(conn, ticker="T2", name="T2", sector="tech",
                         source_id=source_id)
    EdgarAdapter(years=2)._load_company(conn, FOUR_YEARS, company_id=cid,
                                        source_id=source_id, run_id=run_id)
    conn.commit()
    assert sorted(values(conn, "revenue_ttm")) == ["2023-12-31", "2024-12-31"]
    conn.close()


def test_ebitda_is_composed_per_period_and_flagged_derived(loaded):
    ebitda = values(loaded, "ebitda")
    assert ebitda["2024-12-31"] == pytest.approx(262.5)   # 200 + 62.5
    assert ebitda["2021-12-31"] == pytest.approx(80.0)    # 50 + 30
    row = loaded.execute(
        """SELECT is_derived, derivation FROM company_metrics
           WHERE metric_code='ebitda' AND period_end='2024-12-31'""").fetchone()
    assert row[0] == 1
    assert "OperatingIncomeLoss" in row[1]


def test_a_period_missing_one_input_composes_nothing_for_it(loaded):
    """Free cash flow needs both cash-flow lines; only 2024 has both."""
    assert sorted(values(loaded, "free_cash_flow")) == ["2024-12-31"]
    assert values(loaded, "free_cash_flow")["2024-12-31"] == pytest.approx(150.0)


def test_margins_are_computed_for_each_year(loaded):
    margins = values(loaded, "ebitda_margin")
    assert margins["2021-12-31"] == pytest.approx(0.08)     # 80 / 1000
    assert margins["2024-12-31"] == pytest.approx(0.15)     # 262.5 / 1750


# ---------------------------------------------------------------------------
# Trajectory — the reason multi-year exists
# ---------------------------------------------------------------------------

def test_margin_trend_captures_the_direction(loaded):
    """A level cannot separate 'cheap and improving' from 'cheap and rotting'."""
    trend = values(loaded, "ebitda_margin_trend")["2024-12-31"]
    assert trend == pytest.approx(0.15 - 0.08, abs=1e-6)
    assert trend > 0


def test_revenue_cagr_is_annualised_over_the_window(loaded):
    cagr = values(loaded, "revenue_cagr_3y")["2024-12-31"]
    assert cagr == pytest.approx((1750 / 1000) ** (1 / 3) - 1, abs=1e-9)


def test_year_on_year_growth_uses_the_two_most_recent_years(loaded):
    yoy = values(loaded, "revenue_growth_yoy")["2024-12-31"]
    assert yoy == pytest.approx((1750 - 1500) / 1500)


def test_trajectory_metrics_record_their_arithmetic(loaded):
    for metric in ("revenue_cagr_3y", "ebitda_margin_trend", "revenue_growth_yoy"):
        row = loaded.execute(
            "SELECT is_derived, derivation FROM company_metrics WHERE metric_code=?",
            (metric,)).fetchone()
        assert row[0] == 1, metric
        assert row[1], f"{metric} stored no derivation"


def test_a_single_year_of_history_yields_no_trajectory(tmp_path):
    """One year is a level, not a trend, and must not be presented as one."""
    conn = connect(tmp_path / "one.db")
    init_schema(conn)
    run_id = start_run(conn, "t")
    source_id = upsert_source(conn, key="t", name="t", adapter="t")
    cid = upsert_company(conn, ticker="T1", name="T1", sector="tech",
                         source_id=source_id)
    one_year = {"us-gaap": {"Revenues": annual({"2024-12-31": 500.0})}}
    EdgarAdapter(years=4)._load_company(conn, one_year, company_id=cid,
                                        source_id=source_id, run_id=run_id)
    conn.commit()
    assert values(conn, "revenue_cagr_3y") == {}
    assert values(conn, "ebitda_margin_trend") == {}
    assert values(conn, "revenue_growth_yoy") == {}
    conn.close()


def test_the_latest_view_still_returns_one_row_per_metric(loaded):
    """Multi-year storage must not break the snapshot view everything reads."""
    rows = loaded.execute(
        """SELECT metric_code, COUNT(*) FROM v_company_latest_metrics
           GROUP BY metric_code HAVING COUNT(*) > 1""").fetchall()
    assert rows == []
    latest = loaded.execute(
        """SELECT value FROM v_company_latest_metrics
           WHERE metric_code = 'revenue_ttm'""").fetchone()[0]
    assert latest == 1750.0, "the view should surface the most recent year"
