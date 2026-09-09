"""SQLite access layer.

Two connection modes matter here:

* writers  -- used only by the ingest pipeline.
* readers  -- opened with SQLite's `mode=ro` URI. The MCP server uses this, so
  a bug (or a prompt injection that reaches a tool argument) cannot mutate the
  database. Combined with parameterised queries everywhere, the tool surface is
  read-only by construction rather than by convention.
"""

from __future__ import annotations

import sqlite3
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .metrics import METRICS

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: Path | str, *, read_only: bool = False) -> sqlite3.Connection:
    path = Path(path)
    if read_only:
        if not path.exists():
            raise FileNotFoundError(
                f"database not found at {path}. Build it first: "
                f"python -m sectorlens.ingest.build_db"
            )
        # Build the URI with `as_uri()` rather than f-string interpolation.
        # Interpolating a raw path leaves Windows drive letters and backslashes
        # (`file:C:\Users\...`) and unescaped spaces in a field that is parsed
        # as a URI. `as_uri()` percent-encodes and normalises both.
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_PATH.read_text())
    seed_metrics(conn)
    conn.commit()


def seed_metrics(conn: sqlite3.Connection) -> None:
    conn.executemany(
        """INSERT INTO metrics (code, label, unit, description, higher_is_better)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(code) DO UPDATE SET
             label=excluded.label, unit=excluded.unit,
             description=excluded.description,
             higher_is_better=excluded.higher_is_better""",
        [
            (m.code, m.label, m.unit, m.description,
             None if m.higher_is_better is None else int(m.higher_is_better))
            for m in METRICS
        ],
    )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def upsert_source(
    conn: sqlite3.Connection,
    *,
    key: str,
    name: str,
    adapter: str,
    url: str | None = None,
    publisher: str | None = None,
    license: str | None = None,
    notes: str | None = None,
) -> int:
    conn.execute(
        """INSERT INTO sources (key, name, url, publisher, license,
                                retrieved_at, adapter, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(key) DO UPDATE SET
             name=excluded.name, url=excluded.url, publisher=excluded.publisher,
             license=excluded.license, retrieved_at=excluded.retrieved_at,
             adapter=excluded.adapter, notes=excluded.notes""",
        (key, name, url, publisher, license, utcnow(), adapter, notes),
    )
    return int(conn.execute("SELECT id FROM sources WHERE key = ?", (key,)).fetchone()[0])


def start_run(conn: sqlite3.Connection, adapter: str) -> int:
    cur = conn.execute(
        "INSERT INTO ingest_runs (adapter, started_at) VALUES (?, ?)",
        (adapter, utcnow()),
    )
    conn.commit()
    return int(cur.lastrowid)


def finish_run(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    status: str = "ok",
    companies: int = 0,
    metric_values: int = 0,
    signals: int = 0,
    error: str | None = None,
) -> None:
    conn.execute(
        """UPDATE ingest_runs
           SET finished_at=?, status=?, companies=?, metric_values=?,
               signals=?, error=?
           WHERE id=?""",
        (utcnow(), status, companies, metric_values, signals, error, run_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------

def upsert_company(conn: sqlite3.Connection, *, ticker: str, name: str,
                   sector: str, source_id: int, **extra: Any) -> int:
    conn.execute(
        """INSERT INTO companies (ticker, name, sector, gics_sector,
                                  gics_sub_industry, hq_location, cik, founded,
                                  index_added_date, source_id, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(ticker) DO UPDATE SET
             name=excluded.name, sector=excluded.sector,
             gics_sector=excluded.gics_sector,
             gics_sub_industry=excluded.gics_sub_industry,
             hq_location=excluded.hq_location, cik=excluded.cik,
             founded=excluded.founded,
             index_added_date=excluded.index_added_date,
             source_id=excluded.source_id, updated_at=excluded.updated_at""",
        (
            ticker, name, sector, extra.get("gics_sector"),
            extra.get("gics_sub_industry"), extra.get("hq_location"),
            extra.get("cik"), extra.get("founded"),
            extra.get("index_added_date"), source_id, utcnow(),
        ),
    )
    return int(conn.execute("SELECT id FROM companies WHERE ticker = ?",
                            (ticker,)).fetchone()[0])


def write_metric(
    conn: sqlite3.Connection,
    *,
    company_id: int,
    metric_code: str,
    value: float | None,
    source_id: int,
    run_id: int,
    period_end: str | None = None,
    fiscal_period: str = "snapshot",
    is_derived: bool = False,
    derivation: str | None = None,
) -> bool:
    """Write one fact. Returns False for values we refuse to store.

    Non-finite and None values are dropped rather than stored as NULL/NaN: a
    missing metric must be genuinely absent so that "we have no data on this"
    stays distinguishable from "the value is zero".
    """
    if value is None:
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if v != v or v in (float("inf"), float("-inf")):
        return False
    conn.execute(
        """INSERT INTO company_metrics (company_id, metric_code, value,
               period_end, fiscal_period, is_derived, derivation, source_id, run_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(company_id, metric_code, period_end, fiscal_period)
           DO UPDATE SET value=excluded.value, is_derived=excluded.is_derived,
                         derivation=excluded.derivation,
                         source_id=excluded.source_id, run_id=excluded.run_id""",
        (company_id, metric_code, v, period_end, fiscal_period,
         int(is_derived), derivation, source_id, run_id),
    )
    return True


def write_signal(
    conn: sqlite3.Connection,
    *,
    company_id: int,
    signal_type: str,
    source_id: int,
    run_id: int,
    value_num: float | None = None,
    value_text: str | None = None,
    as_of: str | None = None,
) -> bool:
    if value_num is None and not value_text:
        return False
    conn.execute(
        """INSERT INTO company_signals (company_id, signal_type, value_num,
               value_text, as_of, source_id, run_id)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(company_id, signal_type, as_of)
           DO UPDATE SET value_num=excluded.value_num,
                         value_text=excluded.value_text,
                         source_id=excluded.source_id, run_id=excluded.run_id""",
        (company_id, signal_type, value_num, value_text, as_of, source_id, run_id),
    )
    return True


# ---------------------------------------------------------------------------
# Derived aggregates
# ---------------------------------------------------------------------------

def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def compute_sector_benchmarks(conn: sqlite3.Connection, run_id: int) -> int:
    """Recompute per-sector distribution stats from the stored facts.

    The mutual fund persona is defined as benchmark-relative, so the sector
    median has to be a stored, citable number rather than something the model
    estimates from a list it was shown.
    """
    conn.execute("DELETE FROM sector_benchmarks")
    rows = conn.execute(
        """SELECT c.sector AS sector, v.metric_code AS metric_code, v.value AS value
           FROM v_company_latest_metrics v
           JOIN companies c ON c.id = v.company_id"""
    ).fetchall()

    buckets: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        buckets.setdefault((r["sector"], r["metric_code"]), []).append(float(r["value"]))

    written = 0
    now = utcnow()
    for (sector, metric_code), values in buckets.items():
        if len(values) < 3:  # a "median" over one or two names is noise
            continue
        conn.execute(
            """INSERT INTO sector_benchmarks
                 (sector, metric_code, median, p25, p75, mean, n, computed_at, run_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(sector, metric_code) DO UPDATE SET
                 median=excluded.median, p25=excluded.p25, p75=excluded.p75,
                 mean=excluded.mean, n=excluded.n,
                 computed_at=excluded.computed_at, run_id=excluded.run_id""",
            (sector, metric_code, statistics.median(values),
             _quantile(values, 0.25), _quantile(values, 0.75),
             statistics.fmean(values), len(values), now, run_id),
        )
        written += 1
    conn.commit()
    return written


def compute_relative_metrics(conn: sqlite3.Connection, source_id: int,
                             run_id: int) -> int:
    """Derive the sector-relative metrics the PE and MF personas key off.

    Must run after `compute_sector_benchmarks`.
    """
    written = 0
    bench = {
        (r["sector"], r["metric_code"]): r["median"]
        for r in conn.execute(
            "SELECT sector, metric_code, median FROM sector_benchmarks"
        )
    }
    rows = conn.execute(
        """SELECT c.id AS company_id, c.sector AS sector,
                  v.metric_code AS metric_code, v.value AS value
           FROM v_company_latest_metrics v
           JOIN companies c ON c.id = v.company_id
           WHERE v.metric_code IN ('ebitda_margin', 'pe_ratio')"""
    ).fetchall()

    for r in rows:
        median = bench.get((r["sector"], r["metric_code"]))
        if median is None:
            continue
        if r["metric_code"] == "ebitda_margin":
            written += write_metric(
                conn, company_id=r["company_id"],
                metric_code="ebitda_margin_gap_to_sector",
                value=median - float(r["value"]),
                source_id=source_id, run_id=run_id, is_derived=True,
                derivation=(f"sector median ebitda_margin ({median:.4f}) "
                            f"- company ebitda_margin ({float(r['value']):.4f})"),
            )
        elif r["metric_code"] == "pe_ratio" and median:
            written += write_metric(
                conn, company_id=r["company_id"],
                metric_code="pe_vs_sector_median",
                value=float(r["value"]) / median,
                source_id=source_id, run_id=run_id, is_derived=True,
                derivation=(f"company pe_ratio ({float(r['value']):.2f}) "
                            f"/ sector median pe_ratio ({median:.2f})"),
            )
    conn.commit()
    return written


def counts(conn: sqlite3.Connection) -> dict[str, Any]:
    def one(sql: str) -> int:
        return int(conn.execute(sql).fetchone()[0])

    return {
        "companies": one("SELECT COUNT(*) FROM companies"),
        "metric_values": one("SELECT COUNT(*) FROM company_metrics"),
        "signals": one("SELECT COUNT(*) FROM company_signals"),
        "benchmarks": one("SELECT COUNT(*) FROM sector_benchmarks"),
        "sources": one("SELECT COUNT(*) FROM sources"),
        "by_sector": {
            r["sector"]: r["n"]
            for r in conn.execute(
                "SELECT sector, COUNT(*) n FROM companies GROUP BY sector ORDER BY sector"
            )
        },
    }


def record_finding(conn: sqlite3.Connection, *, scope: str, check_name: str,
                   severity: str, detail: str, run_id: int,
                   ref: str | None = None) -> None:
    conn.execute(
        """INSERT INTO data_quality_findings
             (scope, ref, check_name, severity, detail, run_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (scope, ref, check_name, severity, detail, run_id, utcnow()),
    )
