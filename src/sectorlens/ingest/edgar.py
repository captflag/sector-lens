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
                 timeout: float = 30.0) -> None:
        self.limit_per_sector = limit_per_sector
        self.timeout = timeout

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
        written = 0
        reported: dict[str, float | None] = {}
        period_end: str | None = None

        for code, aliases in CONCEPT_ALIASES.items():
            series = _resolve(facts, aliases)
            value = _value_at(series)
            reported[code] = value
            if series and period_end is None:
                period_end = series[0].get("end")
            if write_metric(conn, company_id=company_id, metric_code=code,
                            value=value, source_id=source_id, run_id=run_id,
                            period_end=series[0].get("end") if series else None,
                            fiscal_period="FY"):
                written += 1

        # EBITDA is not an XBRL concept; build it from operating income + D&A
        # and record the arithmetic so it is never mistaken for a reported line.
        op_series = _resolve(facts, ("OperatingIncomeLoss",))
        da_series = _resolve(facts, DEPRECIATION_ALIASES)
        op, da = _value_at(op_series), _value_at(da_series)
        if op is not None and da is not None:
            reported["ebitda"] = op + da
            if write_metric(conn, company_id=company_id, metric_code="ebitda",
                            value=op + da, source_id=source_id, run_id=run_id,
                            period_end=op_series[0].get("end"),
                            fiscal_period="FY", is_derived=True,
                            derivation="OperatingIncomeLoss + "
                                       "DepreciationDepletionAndAmortization"):
                written += 1

        long_debt = _value_at(_resolve(facts, DEBT_LONG_ALIASES)) or 0.0
        short_debt = _value_at(_resolve(facts, DEBT_SHORT_ALIASES)) or 0.0
        if long_debt or short_debt:
            reported["total_debt"] = long_debt + short_debt
            if write_metric(conn, company_id=company_id,
                            metric_code="total_debt", value=long_debt + short_debt,
                            source_id=source_id, run_id=run_id,
                            fiscal_period="FY", is_derived=True,
                            derivation="long-term debt + current portion / "
                                       "short-term borrowings"):
                written += 1

        ocf = _value_at(_resolve(facts, OCF_ALIASES))
        capex = _value_at(_resolve(facts, CAPEX_ALIASES))
        if ocf is not None and capex is not None:
            reported["free_cash_flow"] = ocf - capex
            if write_metric(conn, company_id=company_id,
                            metric_code="free_cash_flow", value=ocf - capex,
                            source_id=source_id, run_id=run_id,
                            fiscal_period="FY", is_derived=True,
                            derivation="operating cash flow - capital expenditure"):
                written += 1

        rev_series = _resolve(facts, CONCEPT_ALIASES["revenue_ttm"])
        growth = growth_rate(_value_at(rev_series, 0), _value_at(rev_series, 1))
        if growth is not None:
            if write_metric(conn, company_id=company_id,
                            metric_code="revenue_growth_yoy", value=growth,
                            source_id=source_id, run_id=run_id,
                            period_end=rev_series[0].get("end"),
                            fiscal_period="FY", is_derived=True,
                            derivation=(f"({rev_series[0]['end']} revenue - "
                                        f"{rev_series[1]['end']} revenue) / "
                                        f"{rev_series[1]['end']} revenue")):
                written += 1

        for d in derive_from_fundamentals(reported):
            if write_metric(conn, company_id=company_id, metric_code=d.code,
                            value=d.value, source_id=source_id, run_id=run_id,
                            period_end=period_end, fiscal_period="FY",
                            is_derived=True, derivation=d.derivation):
                written += 1
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
