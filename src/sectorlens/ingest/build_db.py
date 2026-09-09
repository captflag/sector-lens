"""Build (or rebuild) the sector intelligence database.

    python -m sectorlens.ingest.build_db                      # default adapter
    python -m sectorlens.ingest.build_db --adapter edgar      # filing-grade data
    python -m sectorlens.ingest.build_db --sectors tech retail

The pipeline is: ingest facts -> compute sector benchmarks -> derive the
sector-relative metrics that personas key off -> run quality checks. The order
matters: `ebitda_margin_gap_to_sector` cannot exist before the sector medians
it is measured against.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import load_sectors
from ..db.store import (compute_relative_metrics, compute_sector_benchmarks,
                        connect, counts, finish_run, init_schema, start_run,
                        upsert_source)
from ..settings import get_settings
from .edgar import EdgarAdapter
from .public_datasets import PublicDatasetsAdapter
from .quality import run_checks

ADAPTERS = {
    "public": PublicDatasetsAdapter,
    "edgar": EdgarAdapter,
}


def build(adapter_name: str, db_path: Path, sectors: list[str] | None,
          limit_per_sector: int | None, fresh: bool) -> int:
    if adapter_name not in ADAPTERS:
        raise SystemExit(f"unknown adapter {adapter_name!r}; "
                         f"choose from {sorted(ADAPTERS)}")

    if fresh and db_path.exists():
        db_path.unlink()

    adapter = (EdgarAdapter(limit_per_sector=limit_per_sector)
               if adapter_name == "edgar" else PublicDatasetsAdapter())

    conn = connect(db_path)
    init_schema(conn)
    run_id = start_run(conn, adapter.key)

    print(f"[build] adapter={adapter.key} db={db_path}")
    try:
        result = adapter.run(conn, run_id, sectors)
    except Exception as exc:  # noqa: BLE001 - surface the failure in the run log
        finish_run(conn, run_id, status="failed", error=f"{type(exc).__name__}: {exc}")
        raise

    print(f"[build] ingested {result.companies} companies, "
          f"{result.metric_values} metric values, {result.signals} signals")
    for note in result.notes:
        print(f"[build] note: {note}")
    if result.skipped:
        print(f"[build] skipped {len(result.skipped)} records "
              f"(first 3: {result.skipped[:3]})")

    benchmarks = compute_sector_benchmarks(conn, run_id)
    print(f"[build] computed {benchmarks} sector benchmark rows")

    derived_source = upsert_source(
        conn, key="sectorlens_derived", name="Sector Lens derived metrics",
        adapter="build_db", publisher="this repository",
        notes="Sector-relative values computed from ingested facts at build "
              "time. Never a reported figure.")
    relative = compute_relative_metrics(conn, derived_source, run_id)
    print(f"[build] derived {relative} sector-relative metric values")

    findings = run_checks(conn, run_id)
    print(f"[build] recorded {findings} data-quality findings")

    summary = counts(conn)
    finish_run(conn, run_id, status="ok", companies=result.companies,
               metric_values=summary["metric_values"], signals=result.signals)

    print("\n[build] database summary")
    for key in ("companies", "metric_values", "signals", "benchmarks", "sources"):
        print(f"  {key:>14}: {summary[key]:,}")
    print(f"  {'by sector':>14}: {summary['by_sector']}")
    conn.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", default="public", choices=sorted(ADAPTERS),
                        help="'public' needs only GitHub; 'edgar' needs "
                             "data.sec.gov and gives filing-grade fundamentals")
    parser.add_argument("--db", type=Path, default=settings.resolved_db_path)
    parser.add_argument("--sectors", nargs="*", default=None,
                        choices=sorted(load_sectors()),
                        help="restrict the build to these sectors")
    parser.add_argument("--limit-per-sector", type=int, default=15,
                        help="EDGAR only: companies per sector (SEC rate limits)")
    parser.add_argument("--fresh", action="store_true",
                        help="delete the database first instead of upserting")
    args = parser.parse_args(argv)

    return build(args.adapter, args.db, args.sectors,
                 args.limit_per_sector, args.fresh)


if __name__ == "__main__":
    sys.exit(main())
