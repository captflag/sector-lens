"""Adapter: SEC EDGAR XBRL company facts.

This is the authoritative fundamentals source. Company reference data (ticker,
name, GICS classification, CIK) still comes from the index constituents file,
because EDGAR has no sector taxonomy; every financial value comes from the
company's own filings.

Network note: `data.sec.gov` is unreachable from some locked-down environments
(including the one this repository was authored in), which is exactly why the
GitHub adapter exists. Run this one wherever you have normal outbound access:

    python -m sectorlens.ingest.build_db --adapter edgar --limit-per-sector 15

SEC fair-access rules require a real contact string in the User-Agent and cap
clients at 10 requests/second. Both are enforced below; do not raise the rate.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any, Iterable

import httpx

from ..config import classify_sector
from ..db.store import (record_finding, upsert_company, upsert_source,
                        write_metric, write_signal)
from ..settings import get_settings
from .base import Adapter, IngestResult, derive_from_fundamentals, growth_rate
from .public_datasets import CONSTITUENTS_URL, _fetch_csv

#: How the composed values were built, recorded with each stored fact so a
#: reader can tell them from a line the company actually reported.
_DERIVATIONS = {
    "ebitda": "OperatingIncomeLoss + DepreciationDepletionAndAmortization",
    "total_debt": "long-term debt + current portion / short-term borrowings",
    "free_cash_flow": "operating cash flow - capital expenditure",
}

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Companies tag the same economic quantity under different US-GAAP concepts
# depending on their filing history, so each metric resolves through an ordered
# list of aliases and takes the first that yields an annual value.
CONCEPT_ALIASES: dict[str, tuple[str, ...]] = {
    "revenue_ttm": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ),
    "net_income_ttm": ("NetIncomeLoss", "ProfitLoss"),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "cash_and_equivalents": (
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ),
}

# Components combined arithmetically rather than read from a single concept.
DEPRECIATION_ALIASES = (
    "DepreciationDepletionAndAmortization",
    "DepreciationAmortizationAndAccretionNet",
    "DepreciationAndAmortization",
)
DEBT_LONG_ALIASES = ("LongTermDebtNoncurrent", "LongTermDebt")
DEBT_SHORT_ALIASES = ("LongTermDebtCurrent", "ShortTermBorrowings",
                      "DebtCurrent")
OCF_ALIASES = ("NetCashProvidedByUsedInOperatingActivities",
               "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")
CAPEX_ALIASES = ("PaymentsToAcquirePropertyPlantAndEquipment",
                 "PaymentsToAcquireProductiveAssets")


def _annual_series(facts: dict[str, Any], taxonomy: str,
                   concept: str) -> list[dict[str, Any]]:
    """Annual (10-K, FY) observations for one concept, newest first."""
    node = facts.get(taxonomy, {}).get(concept)
    if not node:
        return []
    out: list[dict[str, Any]] = []
    for unit_rows in node.get("units", {}).values():
        for row in unit_rows:
            if row.get("form") == "10-K" and row.get("fp") == "FY" and row.get("end"):
                out.append(row)
    # Later `filed` wins for the same period: restatements supersede originals.
    out.sort(key=lambda r: (r.get("end", ""), r.get("filed", "")), reverse=True)
    deduped: dict[str, dict[str, Any]] = {}
    for row in out:
        deduped.setdefault(row["end"], row)
    return sorted(deduped.values(), key=lambda r: r["end"], reverse=True)


def _resolve(facts: dict[str, Any], aliases: Iterable[str],
             taxonomy: str = "us-gaap") -> list[dict[str, Any]]:
    for concept in aliases:
        series = _annual_series(facts, taxonomy, concept)
        if series:
            return series
    return []


def _by_period(facts: dict[str, Any], aliases: Iterable[str],
               taxonomy: str = "us-gaap") -> dict[str, float]:
    """Annual values for one concept, keyed by period end date.

    Loading several years rather than only the latest is what turns "this
    company has a 12% margin" into "this company's margin has gone from 8% to
    12%". A level cannot distinguish a business that is cheap because it is
    improving from one that is cheap because it is deteriorating.
    """
    out: dict[str, float] = {}
    for row in _resolve(facts, aliases, taxonomy):
        end = row.get("end")
        if not end:
            continue
        try:
            out[end] = float(row["val"])
        except (TypeError, ValueError, KeyError):
            continue
    return out


def _value_at(series: list[dict[str, Any]], index: int = 0) -> float | None:
    if len(series) > index:
        try:
            return float(series[index]["val"])
        except (TypeError, ValueError, KeyError):
            return None
    return None


class EdgarAdapter(Adapter):
    key = "sec_edgar_companyfacts"
    name = "SEC EDGAR XBRL company facts"
    needs_network = True

    #: SEC fair-access ceiling is 10 req/s; stay under it.
    requests_per_second = 8.0

    def __init__(self, limit_per_sector: int | None = 15,
                 timeout: float = 30.0, years: int = 4) -> None:
        self.limit_per_sector = limit_per_sector
        self.timeout = timeout
        #: Fiscal years to load per company. Four gives a three-year CAGR and a
        #: margin trend with a year to spare when a filer's history is short.
        self.years = max(1, int(years))

    def run(self, conn: sqlite3.Connection, run_id: int,
            sectors: Iterable[str] | None = None) -> IngestResult:
        settings = get_settings()
        wanted = set(sectors) if sectors else None
        result = IngestResult(adapter=self.key)

        headers = {
            "User-Agent": settings.sec_user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
        if "example.com" in settings.sec_user_agent:
            result.notes.append(
                "SECTORLENS_SEC_USER_AGENT is still the placeholder. SEC requires "
                "a real contact address and will throttle or block requests "
                "without one."
            )

        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            constituents = _fetch_csv(client, CONSTITUENTS_URL)

        ref_source = upsert_source(
            conn, key="github_sp500_constituents",
            name="S&P 500 constituents (GICS classification)",
            url=CONSTITUENTS_URL, publisher="datasets / Wikipedia",
            license="ODC-PDDL-1.0", adapter=self.key,
            notes="Company universe and sector mapping only; no financials.")
        fact_source = upsert_source(
            conn, key=self.key, name=self.name,
            url="https://data.sec.gov/api/xbrl/companyfacts/",
            publisher="U.S. Securities and Exchange Commission",
            license="Public domain (US Government work)", adapter=self.key,
            notes="Annual (10-K, FY) XBRL facts. Restatements supersede "
                  "originals by filing date.")

        selected: dict[str, list[dict[str, str]]] = {}
        for row in constituents:
            gics_sector = (row.get("GICS Sector") or "").strip()
            gics_sub = (row.get("GICS Sub-Industry") or "").strip()
            sector = classify_sector(gics_sector, gics_sub)
            if sector is None or (wanted and sector not in wanted):
                continue
            if not (row.get("CIK") or "").strip():
                continue
            selected.setdefault(sector, []).append(row)

        companies = metric_values = signals = 0
        interval = 1.0 / self.requests_per_second

        with httpx.Client(timeout=self.timeout, headers=headers,
                          follow_redirects=True) as client:
            for sector, rows in selected.items():
                # Largest, most-established names first: they have the deepest
                # and most consistently tagged XBRL history.
                rows.sort(key=lambda r: (r.get("Date added") or "9999"))
                if self.limit_per_sector:
                    rows = rows[: self.limit_per_sector]

                for row in rows:
                    ticker = row["Symbol"].strip()
                    cik = row["CIK"].strip().zfill(10)
                    company_id = upsert_company(
                        conn, ticker=ticker,
                        name=(row.get("Security") or ticker).strip(),
                        sector=sector, source_id=ref_source,
                        gics_sector=(row.get("GICS Sector") or "").strip(),
                        gics_sub_industry=(row.get("GICS Sub-Industry") or "").strip(),
                        hq_location=(row.get("Headquarters Location") or "").strip() or None,
                        cik=cik, founded=(row.get("Founded") or "").strip() or None,
                        index_added_date=(row.get("Date added") or "").strip() or None,
                    )
                    companies += 1

                    time.sleep(interval)
                    try:
                        resp = client.get(COMPANYFACTS_URL.format(cik=cik))
                        resp.raise_for_status()
                        facts = resp.json().get("facts", {})
                    except Exception as exc:  # noqa: BLE001 - one bad filer must not abort the run
                        result.skipped.append(f"{ticker}: {type(exc).__name__}: {exc}")
                        record_finding(
                            conn, scope="company", ref=ticker,
                            check_name="edgar_fetch_failed", severity="warn",
                            detail=f"companyfacts fetch failed for CIK {cik}: {exc}",
                            run_id=run_id)
                        continue

                    metric_values += self._load_company(
                        conn, facts, company_id=company_id,
                        source_id=fact_source, run_id=run_id)
                    signals += self._load_signals(
                        conn, facts, company_id=company_id,
                        source_id=fact_source, run_id=run_id)
                conn.commit()

        conn.commit()
        result.companies = companies
        result.metric_values = metric_values
        result.signals = signals
        return result

    def _load_company(self, conn: sqlite3.Connection, facts: dict[str, Any], *,
                      company_id: int, source_id: int, run_id: int) -> int:
        """Load every fiscal year in the window, not just the latest.

        Each period is written with its own `period_end`, so the facts table
        carries a real time series and `v_company_latest_metrics` still surfaces
        the most recent value for anything that wants a snapshot.
        """
        written = 0
        periods: dict[str, dict[str, float]] = {}

        def put(end: str | None, code: str, value: float | None) -> None:
            if end and value is not None:
                periods.setdefault(end, {})[code] = value

        for code, aliases in CONCEPT_ALIASES.items():
            for end, value in _by_period(facts, aliases).items():
                put(end, code, value)

        # EBITDA is not an XBRL concept; build it per period from operating
        # income plus D&A, and only where both sides describe the same year.
        operating = _by_period(facts, ("OperatingIncomeLoss",))
        depreciation = _by_period(facts, DEPRECIATION_ALIASES)
        for end in operating.keys() & depreciation.keys():
            put(end, "ebitda", operating[end] + depreciation[end])

        long_debt = _by_period(facts, DEBT_LONG_ALIASES)
        short_debt = _by_period(facts, DEBT_SHORT_ALIASES)
        for end in long_debt.keys() | short_debt.keys():
            put(end, "total_debt", long_debt.get(end, 0.0) + short_debt.get(end, 0.0))

        operating_cash = _by_period(facts, OCF_ALIASES)
        capex = _by_period(facts, CAPEX_ALIASES)
        for end in operating_cash.keys() & capex.keys():
            put(end, "free_cash_flow", operating_cash[end] - capex[end])

        # Newest first, then trimmed to the requested window.
        ordered = sorted(periods, reverse=True)[: self.years]
        if not ordered:
            return 0

        for end in ordered:
            values = periods[end]
            for code, value in values.items():
                derived = code in ("ebitda", "total_debt", "free_cash_flow")
                if write_metric(
                        conn, company_id=company_id, metric_code=code,
                        value=value, source_id=source_id, run_id=run_id,
                        period_end=end, fiscal_period="FY",
                        is_derived=derived,
                        derivation=_DERIVATIONS.get(code) if derived else None):
                    written += 1

            for d in derive_from_fundamentals(values):
                if write_metric(conn, company_id=company_id, metric_code=d.code,
                                value=d.value, source_id=source_id, run_id=run_id,
                                period_end=end, fiscal_period="FY",
                                is_derived=True, derivation=d.derivation):
                    written += 1

        written += self._load_trajectory(
            conn, periods, ordered, company_id=company_id,
            source_id=source_id, run_id=run_id)
        return written

    def _load_trajectory(self, conn: sqlite3.Connection,
                         periods: dict[str, dict[str, float]],
                         ordered: list[str], *, company_id: int,
                         source_id: int, run_id: int) -> int:
        """Metrics that only exist because several years were loaded."""
        if len(ordered) < 2:
            return 0

        written = 0
        latest, earliest = ordered[0], ordered[-1]
        span_years = len(ordered) - 1

        latest_rev = periods[latest].get("revenue_ttm")
        prior_rev = periods[ordered[1]].get("revenue_ttm")
        earliest_rev = periods[earliest].get("revenue_ttm")

        growth = growth_rate(latest_rev, prior_rev)
        if growth is not None:
            written += write_metric(
                conn, company_id=company_id, metric_code="revenue_growth_yoy",
                value=growth, source_id=source_id, run_id=run_id,
                period_end=latest, fiscal_period="FY", is_derived=True,
                derivation=f"({latest} revenue - {ordered[1]} revenue) "
                           f"/ {ordered[1]} revenue")

        if latest_rev and earliest_rev and earliest_rev > 0 and span_years >= 2:
            cagr = (latest_rev / earliest_rev) ** (1 / span_years) - 1
            written += write_metric(
                conn, company_id=company_id, metric_code="revenue_cagr_3y",
                value=cagr, source_id=source_id, run_id=run_id,
                period_end=latest, fiscal_period="FY", is_derived=True,
                derivation=f"({latest} revenue / {earliest} revenue) ^ "
                           f"(1/{span_years}) - 1")

        def margin(end: str) -> float | None:
            values = periods[end]
            revenue, ebitda = values.get("revenue_ttm"), values.get("ebitda")
            if revenue and revenue > 0 and ebitda is not None:
                return ebitda / revenue
            return None

        latest_margin, earliest_margin = margin(latest), margin(earliest)
        if latest_margin is not None and earliest_margin is not None:
            written += write_metric(
                conn, company_id=company_id, metric_code="ebitda_margin_trend",
                value=latest_margin - earliest_margin,
                source_id=source_id, run_id=run_id, period_end=latest,
                fiscal_period="FY", is_derived=True,
                derivation=f"{latest} EBITDA margin ({latest_margin:.4f}) - "
                           f"{earliest} EBITDA margin ({earliest_margin:.4f})")
        return written

    def _load_signals(self, conn: sqlite3.Connection, facts: dict[str, Any], *,
                      company_id: int, source_id: int, run_id: int) -> int:
        """Headcount, where the filer tags it on the cover page.

        `dei:EntityNumberOfEmployees` is not universally tagged, so absence
        here is normal and must stay visible: the agent answers "no headcount
        signal held" rather than inventing one.
        """
        series = _resolve(facts, ("EntityNumberOfEmployees",), taxonomy="dei")
        if not series:
            return 0
        row = series[0]
        return int(write_signal(
            conn, company_id=company_id, signal_type="headcount",
            value_num=float(row["val"]), as_of=row.get("end"),
            value_text=(f"{int(row['val']):,} employees as disclosed on the "
                        f"{row.get('fy', '')} Form 10-K cover page"),
            source_id=source_id, run_id=run_id))
