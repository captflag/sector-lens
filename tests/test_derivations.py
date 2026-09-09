"""Derived metrics must be arithmetically right and refuse to guess."""

from __future__ import annotations

import pytest

from sectorlens.ingest.base import (derive_from_fundamentals, derive_from_snapshot,
                                 growth_rate)


def _codes(derived):
    return {d.code: d.value for d in derived}


def test_snapshot_derivations_are_correct():
    out = _codes(derive_from_snapshot({
        "market_cap": 1000.0, "price": 10.0, "price_to_sales": 4.0,
        "ebitda": 50.0, "eps": 2.0, "pe_ratio": 5.0,
        "week52_low": 8.0, "week52_high": 12.0,
    }))
    assert out["revenue_ttm"] == pytest.approx(250.0)
    assert out["shares_outstanding"] == pytest.approx(100.0)
    assert out["net_income_ttm"] == pytest.approx(200.0)
    assert out["ebitda_margin"] == pytest.approx(0.2)
    assert out["earnings_yield"] == pytest.approx(0.2)
    assert out["ev_to_ebitda_proxy"] == pytest.approx(20.0)
    assert out["price_vs_52w_range"] == pytest.approx(0.5)


def test_negative_ebitda_yields_no_ev_multiple():
    """A loss-making business has no meaningful EV/EBITDA -- it must be absent,
    not a large or negative number that reads as a valuation."""
    out = _codes(derive_from_snapshot({
        "market_cap": 1000.0, "price": 10.0, "price_to_sales": 4.0,
        "ebitda": -50.0,
    }))
    assert "ev_to_ebitda_proxy" not in out
    assert out["ebitda_margin"] == pytest.approx(-0.2)


def test_missing_inputs_produce_no_output():
    assert _codes(derive_from_snapshot({"market_cap": 1000.0})) == {}
    assert _codes(derive_from_snapshot({})) == {}


def test_zero_denominators_are_skipped():
    out = _codes(derive_from_snapshot({
        "market_cap": 1000.0, "price": 0.0, "price_to_sales": 0.0,
        "week52_low": 5.0, "week52_high": 5.0,
    }))
    assert out == {}


def test_fundamental_derivations():
    out = _codes(derive_from_fundamentals({
        "revenue_ttm": 1000.0, "ebitda": 200.0, "net_income_ttm": 100.0,
        "gross_profit": 400.0, "total_debt": 500.0,
        "cash_and_equivalents": 100.0, "free_cash_flow": 150.0,
    }))
    assert out["gross_margin"] == pytest.approx(0.4)
    assert out["net_debt"] == pytest.approx(400.0)
    assert out["net_debt_to_ebitda"] == pytest.approx(2.0)
    assert out["fcf_conversion"] == pytest.approx(0.75)


def test_growth_rate_guards_against_a_bad_base():
    assert growth_rate(110, 100) == pytest.approx(0.1)
    assert growth_rate(100, 0) is None
    assert growth_rate(100, -50) is None
    assert growth_rate(None, 100) is None
