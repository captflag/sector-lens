"""Adapter: public S&P 500 datasets published on GitHub (ODC-PDDL-1.0).

Two files:
  * s-and-p-500-companies/data/constituents.csv
      ticker, name, GICS sector + sub-industry, HQ, CIK, founded, date added.
      Sourced from Wikipedia, refreshed daily. This drives sector membership.
  * s-and-p-500-companies-financials/data/constituents-financials.csv
      a point-in-time market snapshot (price, P/E, dividend yield, EPS,
      52-week range, market cap, EBITDA, P/S, P/B) sourced via Yahoo Finance.

Why this adapter exists alongside the EDGAR one: it needs nothing but GitHub,
so `make db` works in a locked-down environment and in CI, and the committed
sample database is reproducible by anyone. It is NOT the authoritative source
-- the snapshot is undated upstream and its ratios are internally inconsistent
in places. Both problems are detected at build time and recorded as
`data_quality_findings` rather than hidden. Run the EDGAR adapter for current,
filing-grade fundamentals.
"""

from __future__ import annotations

import csv
import io
import sqlite3
from typing import Iterable

import httpx

from ..config import classify_sector
from ..db.store import (record_finding, upsert_company, upsert_source,
                        write_metric)
from .base import Adapter, IngestResult, derive_from_snapshot

CONSTITUENTS_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
    "main/data/constituents.csv"
)
FINANCIALS_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies-financials/"
    "main/data/constituents-financials.csv"
)

# CSV column -> registry metric code. Anything not listed is ignored, so an
# upstream column rename shows up as missing coverage rather than bad data.
FINANCIAL_COLUMNS = {
    "Price": "price",
    "Price/Earnings": "pe_ratio",
    "Dividend Yield": "dividend_yield",
    "Earnings/Share": "eps",
    "52 Week Low": "week52_low",
    "52 Week High": "week52_high",
    "Market Cap": "market_cap",
    "EBITDA": "ebitda",
    "Price/Sales": "price_to_sales",
    "Price/Book": "price_to_book",
}


def _to_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    s = raw.strip().replace(",", "").replace("$", "")
    if not s or s.upper() in {"N/A", "NA", "NULL", "-"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _fetch_csv(client: httpx.Client, url: str) -> list[dict[str, str]]:
    resp = client.get(url)
    resp.raise_for_status()
    return list(csv.DictReader(io.StringIO(resp.text)))


class PublicDatasetsAdapter(Adapter):
    key = "github_public_datasets"
    name = "S&P 500 public datasets (datasets.io mirrors on GitHub)"
    needs_network = True  # GitHub only

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def run(self, conn: sqlite3.Connection, run_id: int,
            sectors: Iterable[str] | None = None) -> IngestResult:
        wanted = set(sectors) if sectors else None
        result = IngestResult(adapter=self.key)

        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            constituents = _fetch_csv(client, CONSTITUENTS_URL)
            financials = _fetch_csv(client, FINANCIALS_URL)

        ref_source = upsert_source(
            conn, key="github_sp500_constituents",
            name="S&P 500 constituents (GICS classification)",
            url=CONSTITUENTS_URL, publisher="datasets / Wikipedia",
            license="ODC-PDDL-1.0", adapter=self.key,
            notes="Ticker, name, GICS sector and sub-industry, HQ, CIK, "
                  "founding year, index add date. Refreshed daily upstream.",
        )
        fin_source = upsert_source(
            conn, key="github_sp500_financials",
            name="S&P 500 market snapshot (via Yahoo Finance)",
            url=FINANCIALS_URL, publisher="datasets / Yahoo Finance",
            license="ODC-PDDL-1.0", adapter=self.key,
            notes="Undated point-in-time market snapshot. Vintage is inferred "
                  "at ingest from index membership; see data_quality_findings.",
        )

        fin_by_ticker = {r["Symbol"].strip(): r for r in financials if r.get("Symbol")}

        # The snapshot carries no publication date. The latest index-add date
        # among the companies it covers IS a sound lower bound: a company
        # cannot appear in a constituent snapshot taken before it joined.
        #
        # The mirror-image inference -- treating the earliest add-date among
        # MISSING members as an upper bound -- is NOT sound, and is deliberately
        # not implemented: ticker renames (MMC -> MRSH, BK -> BNY) make
        # long-standing members look absent, which would date the snapshot to
        # the 1980s. Only the floor is reported.
        add_dates = [
            (c.get("Date added") or "").strip()
            for c in constituents
            if c.get("Symbol", "").strip() in fin_by_ticker
        ]
        vintage_floor = max((d for d in add_dates if d), default=None)

        companies = metric_values = 0

        for row in constituents:
            ticker = (row.get("Symbol") or "").strip()
            if not ticker:
                continue
            gics_sector = (row.get("GICS Sector") or "").strip()
            gics_sub = (row.get("GICS Sub-Industry") or "").strip()
            sector = classify_sector(gics_sector, gics_sub)
            if sector is None or (wanted and sector not in wanted):
                continue

            company_id = upsert_company(
                conn, ticker=ticker, name=(row.get("Security") or ticker).strip(),
                sector=sector, source_id=ref_source,
                gics_sector=gics_sector, gics_sub_industry=gics_sub,
                hq_location=(row.get("Headquarters Location") or "").strip() or None,
                cik=(row.get("CIK") or "").strip() or None,
                founded=(row.get("Founded") or "").strip() or None,
                index_added_date=(row.get("Date added") or "").strip() or None,
            )
            companies += 1

            fin = fin_by_ticker.get(ticker)
            if not fin:
                result.skipped.append(f"{ticker}: no market snapshot row")
                record_finding(
                    conn, scope="company", ref=ticker,
                    check_name="missing_market_snapshot", severity="info",
                    detail=(f"{ticker} is in the current index but absent "
                            f"from the market snapshot, so it carries no "
                            f"financial metrics. Two causes are common and "
                            f"indistinguishable from here: the company joined "
                            f"the index after the snapshot was taken, or its "
                            f"ticker was renamed between the two files."),
                    run_id=run_id,
                )
                continue

            reported: dict[str, float | None] = {}
            for column, code in FINANCIAL_COLUMNS.items():
                value = _to_float(fin.get(column))
                reported[code] = value
                if write_metric(conn, company_id=company_id, metric_code=code,
                                value=value, source_id=fin_source, run_id=run_id,
                                fiscal_period="snapshot"):
                    metric_values += 1

            for d in derive_from_snapshot(reported):
                if write_metric(conn, company_id=company_id, metric_code=d.code,
                                value=d.value, source_id=fin_source,
                                run_id=run_id, fiscal_period="snapshot",
                                is_derived=True, derivation=d.derivation):
                    metric_values += 1

        conn.commit()

        if vintage_floor:
            record_finding(
                conn, scope="dataset", ref=self.key,
                check_name="undated_snapshot", severity="warn",
                detail=(
                    "The market snapshot is published without a date. The "
                    "latest index-add date among covered companies is "
                    f"{vintage_floor}, which is a firm lower bound on when the "
                    "snapshot was taken -- a company cannot appear in a "
                    "constituent list compiled before it joined the index. No "
                    "upper bound is claimed: ticker renames make the "
                    "mirror-image inference unsound. Treat every price, market "
                    "cap and multiple as point-in-time as of an unknown date "
                    "at or after that floor, and run the EDGAR adapter when "
                    "you need dated, filing-grade fundamentals."
                ),
                run_id=run_id,
            )
            result.notes.append(f"snapshot vintage floor: {vintage_floor}")

        conn.commit()
        result.companies = companies
        result.metric_values = metric_values
        return result
