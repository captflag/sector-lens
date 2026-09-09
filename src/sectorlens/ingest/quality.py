"""Build-time data quality checks.

Data-quality caveats belong in the database, not the README. Writing them as rows
rather than README prose means three things: the agent can cite them when it
answers, the API returns them as structured caveats, and they are recomputed
every build instead of drifting away from the data they describe.
"""

from __future__ import annotations

import sqlite3

from ..config import load_personas, load_sectors
from ..db.store import record_finding

# Bounds chosen to catch arithmetic that cannot be true, not to second-guess
# unusual-but-real businesses. A margin above 100% of revenue is impossible;
# a 60% EBITDA margin is merely unusual, and is left alone.
PLAUSIBILITY = {
    "ebitda_margin": (-1.0, 1.0),
    "net_margin": (-3.0, 1.0),
    "gross_margin": (-1.0, 1.0),
    "fcf_conversion": (-5.0, 5.0),
    "dividend_yield": (0.0, 0.25),
}

# Fraction of a sector's companies that must carry a metric before a persona
# that leans on it can be considered well supported.
COVERAGE_WARN_BELOW = 0.70


def run_checks(conn: sqlite3.Connection, run_id: int) -> int:
    findings = 0
    findings += _implausible_values(conn, run_id)
    findings += _sector_size(conn, run_id)
    findings += _persona_metric_coverage(conn, run_id)
    findings += _derivation_caveats(conn, run_id)
    conn.commit()
    return findings


def _implausible_values(conn: sqlite3.Connection, run_id: int) -> int:
    n = 0
    for metric, (lo, hi) in PLAUSIBILITY.items():
        rows = conn.execute(
            """SELECT c.ticker AS ticker, v.value AS value
               FROM v_company_latest_metrics v
               JOIN companies c ON c.id = v.company_id
               WHERE v.metric_code = ? AND (v.value < ? OR v.value > ?)""",
            (metric, lo, hi),
        ).fetchall()
        for r in rows:
            record_finding(
                conn, scope="company", ref=r["ticker"],
                check_name=f"implausible_{metric}", severity="error",
                detail=(f"{metric} for {r['ticker']} is {r['value']:.3f}, "
                        f"outside the plausible range [{lo}, {hi}]. The value "
                        f"is derived, so one of its inputs is wrong in the "
                        f"upstream snapshot. Do not cite it."),
                run_id=run_id,
            )
            n += 1
    return n


def _sector_size(conn: sqlite3.Connection, run_id: int) -> int:
    n = 0
    for sector_id, sector in load_sectors().items():
        count = int(conn.execute(
            "SELECT COUNT(*) FROM companies WHERE sector = ?", (sector_id,)
        ).fetchone()[0])
        if count == 0:
            record_finding(
                conn, scope="sector", ref=sector_id,
                check_name="empty_sector", severity="error",
                detail=(f"Sector '{sector.label}' matched no companies. Its "
                        f"GICS mapping in sectors.yaml is probably stale."),
                run_id=run_id)
            n += 1
        elif count < 8:
            record_finding(
                conn, scope="sector", ref=sector_id,
                check_name="thin_sector", severity="warn",
                detail=(f"Sector '{sector.label}' has only {count} companies. "
                        f"Sector medians and percentile ranks over a set this "
                        f"small are indicative, not statistically meaningful."),
                run_id=run_id)
            n += 1
    return n


def _persona_metric_coverage(conn: sqlite3.Connection, run_id: int) -> int:
    """Flag persona weights that the loaded data cannot actually support."""
    n = 0
    needed: set[str] = set()
    for p in load_personas().values():
        needed.update(w.metric for w in p.screen_weights)

    for sector_id in load_sectors():
        total = int(conn.execute(
            "SELECT COUNT(*) FROM companies WHERE sector = ?", (sector_id,)
        ).fetchone()[0])
        if not total:
            continue
        for metric in sorted(needed):
            have = int(conn.execute(
                """SELECT COUNT(DISTINCT v.company_id)
                   FROM v_company_latest_metrics v
                   JOIN companies c ON c.id = v.company_id
                   WHERE c.sector = ? AND v.metric_code = ?""",
                (sector_id, metric),
            ).fetchone()[0])
            ratio = have / total
            if ratio == 0:
                record_finding(
                    conn, scope="metric", ref=f"{sector_id}:{metric}",
                    check_name="metric_absent", severity="warn",
                    detail=(f"No company in '{sector_id}' carries '{metric}'. "
                            f"Personas weighted on it fall back to their "
                            f"remaining metrics, renormalised. The current "
                            f"ingest adapter does not supply this metric."),
                    run_id=run_id)
                n += 1
            elif ratio < COVERAGE_WARN_BELOW:
                record_finding(
                    conn, scope="metric", ref=f"{sector_id}:{metric}",
                    check_name="metric_sparse", severity="info",
                    detail=(f"'{metric}' covers {have}/{total} companies "
                            f"({ratio:.0%}) in '{sector_id}'. Rankings that "
                            f"lean on it are based on a partial field."),
                    run_id=run_id)
                n += 1
    return n


def _derivation_caveats(conn: sqlite3.Connection, run_id: int) -> int:
    """Record the systematic caveats that apply to whole classes of value."""
    derived_codes = [
        r["metric_code"] for r in conn.execute(
            "SELECT DISTINCT metric_code FROM company_metrics WHERE is_derived = 1"
            " ORDER BY metric_code")
    ]
    if not derived_codes:
        return 0

    record_finding(
        conn, scope="dataset", ref="derived_metrics",
        check_name="derived_from_ratios", severity="warn",
        detail=(
            "These metrics are computed, not reported: "
            + ", ".join(derived_codes) + ". Where revenue is reconstructed as "
            "market_cap / price_to_sales, any inconsistency between those two "
            "upstream fields propagates into every margin derived from it. "
            "Margins from this adapter are directionally useful for ranking "
            "companies against their own sector, and should not be quoted as "
            "the company's reported margin."),
        run_id=run_id)

    has_net_debt = int(conn.execute(
        "SELECT COUNT(*) FROM company_metrics WHERE metric_code = 'net_debt'"
    ).fetchone()[0])
    if not has_net_debt:
        record_finding(
            conn, scope="dataset", ref="ev_to_ebitda_proxy",
            check_name="ev_excludes_net_debt", severity="warn",
            detail=(
                "No balance-sheet data is loaded, so ev_to_ebitda_proxy is "
                "market cap over EBITDA and excludes net debt entirely. It "
                "therefore understates the true entry multiple for indebted "
                "businesses, which matters most for the PE persona. Run the "
                "EDGAR adapter to load debt and cash and get a real EV."),
            run_id=run_id)
    return 1 if has_net_debt else 2
