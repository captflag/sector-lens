"""Test fixtures.

Tests build their own small database rather than leaning on the committed one,
so they assert on known values and stay green when the upstream data refreshes.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sectorlens.db.store import (compute_relative_metrics, compute_sector_benchmarks,
                              connect, init_schema, start_run, upsert_company,
                              upsert_source, write_metric, write_signal)
from sectorlens.settings import Settings

# ticker, name, sector, market_cap, ebitda, revenue, pe, margin inputs.
# Chosen so the personas must disagree: BIGC is a large, expensive, high-margin
# compounder (mutual fund catnip) and SMLC is a small, cheap, low-margin
# operator (the buyout candidate).
FIXTURE_COMPANIES = [
    # ticker,  name,        sector,  mkt_cap, ebitda,  revenue, pe,   dividend
    ("BIGC",  "Big Co",    "tech",  800e9,   90e9,    250e9,   35.0, 0.008),
    ("MIDC",  "Mid Co",    "tech",  120e9,   14e9,    60e9,    24.0, 0.012),
    ("SMLC", "Small Co", "tech",  9e9,     1.6e9,   20e9,    11.0, 0.030),
    ("THNC", "Thin Co",   "tech",  15e9,    0.9e9,   30e9,    45.0, 0.000),
    ("FRGT", "Freight Co", "logistics", 40e9, 6e9,  50e9,    18.0, 0.020),
    ("HAUL", "Hauler Co", "logistics", 12e9, 2e9,    25e9,    14.0, 0.025),
    ("AIRL", "Airlift Co", "logistics", 6e9, 1.2e9, 22e9,    9.0,  0.000),
]


@pytest.fixture
def fixture_db(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    conn = connect(path)
    init_schema(conn)
    run_id = start_run(conn, "fixture")
    source_id = upsert_source(conn, key="fixture", name="Test fixture",
                              adapter="fixture", license="n/a")

    for ticker, name, sector, cap, ebitda, revenue, pe, dividend in FIXTURE_COMPANIES:
        cid = upsert_company(conn, ticker=ticker, name=name, sector=sector,
                             source_id=source_id, gics_sector="Test")
        values = {
            "market_cap": cap,
            "ebitda": ebitda,
            "revenue_ttm": revenue,
            "pe_ratio": pe,
            "dividend_yield": dividend,
            "ebitda_margin": ebitda / revenue,
            "net_margin": (ebitda / revenue) * 0.5,
            "earnings_yield": 1 / pe,
            "ev_to_ebitda_proxy": cap / ebitda,
            "price_to_book": pe / 6.0,
            "price_vs_52w_range": 0.5,
            "eps": 5.0,
        }
        for code, value in values.items():
            write_metric(conn, company_id=cid, metric_code=code, value=value,
                         source_id=source_id, run_id=run_id)

    # One company carries a workforce signal; the rest deliberately do not, so
    # the "we hold nothing" path is exercised too.
    freight = conn.execute(
        "SELECT id FROM companies WHERE ticker='FRGT'").fetchone()[0]
    write_signal(conn, company_id=freight, signal_type="headcount",
                 value_num=310_000, as_of="2024-12-31",
                 value_text="310,000 employees per the FY2024 10-K cover page",
                 source_id=source_id, run_id=run_id)

    conn.commit()
    compute_sector_benchmarks(conn, run_id)
    compute_relative_metrics(conn, source_id, run_id)
    conn.close()
    return path


@pytest.fixture
def fixture_conn(fixture_db: Path) -> sqlite3.Connection:
    conn = connect(fixture_db, read_only=True)
    yield conn
    conn.close()


@pytest.fixture
def fixture_settings(fixture_db: Path) -> Settings:
    return Settings(db_path=fixture_db, llm_provider="deterministic")
