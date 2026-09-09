"""Shared ingestion machinery.

Adapters are deliberately thin and interchangeable. Each one is responsible for
fetching from exactly one upstream, normalising into the registry's metric
codes, and registering a `sources` row that says where every value came from.
Derivations are centralised here so that a ratio means the same thing no matter
which adapter produced its inputs.
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class IngestResult:
    adapter: str
    companies: int = 0
    metric_values: int = 0
    signals: int = 0
    skipped: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class Adapter(ABC):
    """One upstream data source."""

    #: stable slug, also used as the `sources.key`
    key: str = "adapter"
    #: human label used in provenance output
    name: str = "Adapter"
    #: True when the adapter needs outbound network access beyond GitHub
    needs_network: bool = True

    @abstractmethod
    def run(self, conn: sqlite3.Connection, run_id: int,
            sectors: Iterable[str] | None = None) -> IngestResult:
        ...


# ---------------------------------------------------------------------------
# Derivations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Derived:
    code: str
    value: float
    derivation: str


def _pos(x: float | None) -> float | None:
    """Return x only when it is a usable positive number."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def derive_from_snapshot(m: dict[str, float | None]) -> list[Derived]:
    """Derive ratios from a market snapshot.

    Every output records the arithmetic that produced it, because the agent is
    required to distinguish a reported figure from an inferred one. Anything
    whose denominator is missing, zero or negative is skipped rather than
    stored as a misleading number -- a negative EBITDA makes EV/EBITDA
    meaningless, not large.
    """
    out: list[Derived] = []

    market_cap = _pos(m.get("market_cap"))
    price = _pos(m.get("price"))
    p_s = _pos(m.get("price_to_sales"))
    ebitda = m.get("ebitda")
    eps = m.get("eps")
    pe = m.get("pe_ratio")
    lo, hi = m.get("week52_low"), m.get("week52_high")

    revenue = None
    if market_cap and p_s:
        revenue = market_cap / p_s
        out.append(Derived("revenue_ttm", revenue,
                           "market_cap / price_to_sales"))

    shares = None
    if market_cap and price:
        shares = market_cap / price
        out.append(Derived("shares_outstanding", shares, "market_cap / price"))

    net_income = None
    if shares is not None and eps is not None:
        net_income = float(eps) * shares
        out.append(Derived("net_income_ttm", net_income,
                           "eps * (market_cap / price)"))

    if revenue and ebitda is not None:
        out.append(Derived("ebitda_margin", float(ebitda) / revenue,
                           "ebitda / (market_cap / price_to_sales)"))

    if revenue and net_income is not None:
        out.append(Derived("net_margin", net_income / revenue,
                           "net_income_ttm / revenue_ttm"))

    if pe is not None and float(pe) != 0:
        out.append(Derived("earnings_yield", 1.0 / float(pe), "1 / pe_ratio"))

    if market_cap and ebitda is not None and float(ebitda) > 0:
        out.append(Derived("ev_to_ebitda_proxy", market_cap / float(ebitda),
                           "market_cap / ebitda (EXCLUDES net debt)"))

    if price and lo is not None and hi is not None and float(hi) > float(lo):
        pos = (price - float(lo)) / (float(hi) - float(lo))
        out.append(Derived("price_vs_52w_range", min(max(pos, 0.0), 1.0),
                           "(price - week52_low) / (week52_high - week52_low)"))

    return out


def derive_from_fundamentals(m: dict[str, float | None]) -> list[Derived]:
    """Derive ratios from filing-level fundamentals (EDGAR adapter)."""
    out: list[Derived] = []
    revenue = _pos(m.get("revenue_ttm"))
    ebitda = m.get("ebitda")
    net_income = m.get("net_income_ttm")
    gross_profit = m.get("gross_profit")
    debt = m.get("total_debt")
    cash = m.get("cash_and_equivalents")
    fcf = m.get("free_cash_flow")

    if revenue and gross_profit is not None:
        out.append(Derived("gross_margin", float(gross_profit) / revenue,
                           "gross_profit / revenue_ttm"))
    if revenue and ebitda is not None:
        out.append(Derived("ebitda_margin", float(ebitda) / revenue,
                           "ebitda / revenue_ttm"))
    if revenue and net_income is not None:
        out.append(Derived("net_margin", float(net_income) / revenue,
                           "net_income_ttm / revenue_ttm"))
    if debt is not None and cash is not None:
        net_debt = float(debt) - float(cash)
        out.append(Derived("net_debt", net_debt,
                           "total_debt - cash_and_equivalents"))
        if ebitda is not None and float(ebitda) > 0:
            out.append(Derived("net_debt_to_ebitda", net_debt / float(ebitda),
                               "(total_debt - cash) / ebitda"))
    if fcf is not None and ebitda is not None and float(ebitda) > 0:
        out.append(Derived("fcf_conversion", float(fcf) / float(ebitda),
                           "free_cash_flow / ebitda"))
    return out


def growth_rate(current: float | None, prior: float | None) -> float | None:
    """Year-on-year growth, guarded against a zero or negative base."""
    c, p = _pos(current), _pos(prior)
    if c is None or p is None:
        return None
    return (c - p) / p
