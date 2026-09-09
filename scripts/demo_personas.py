#!/usr/bin/env python3
"""Show that persona changes the evidence, not just the wording.

Asks one question about one sector as all three personas and prints the
rankings side by side. Runs without an API key -- the divergence happens in the
MCP screening tool, below the language model.

    python scripts/demo_personas.py --sector tech
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sectorlens.agent.mcp_client import open_toolbox  # noqa: E402
from sectorlens.config import load_personas, load_sectors  # noqa: E402
from sectorlens.formatting import pct_rank  # noqa: E402

PERSONAS = ["mutual_fund_analyst", "equity_analyst", "pe_analyst"]


def _build_date() -> str:
    """When this database was ingested.

    Printed with the table because the upstream market snapshot refreshes
    daily: the specific tickers below are a point-in-time result, while the
    property being demonstrated -- that the three personas do not converge --
    holds across builds. Without a date on the output, a reader comparing it to
    a figure in the README would reasonably conclude one of them is wrong.
    """
    from sectorlens.db.store import connect
    from sectorlens.settings import get_settings

    try:
        with connect(get_settings().resolved_db_path, read_only=True) as conn:
            row = conn.execute(
                "SELECT started_at FROM ingest_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return (row[0][:10] if row and row[0] else "unknown date")
    except Exception:  # noqa: BLE001 - a missing date must not break the demo
        return "unknown date"


async def run(sector: str, limit: int) -> None:
    sectors = load_sectors()
    personas = load_personas()
    print(f"\nSector: {sectors[sector].label}")
    print(f"Data:   built {_build_date()} from a daily-refreshed upstream "
          f"snapshot -- exact names drift between builds")
    print(f"Question: 'Is this sector a good place to put money to work, and "
          f"which names?'\n")

    tables: dict[str, list[dict]] = {}
    async with open_toolbox() as toolbox:
        for persona in PERSONAS:
            data = await toolbox.call_json(
                "screen_sector",
                {"sector": sector, "persona": persona, "limit": limit})
            tables[persona] = data.get("results", [])
            universe = data.get("universe_size", 0)

    width = 34
    header = "".join(f"{personas[p].short_label:<{width}}" for p in PERSONAS)
    print(header)
    print("-" * (width * len(PERSONAS)))
    for i in range(limit):
        row = ""
        for persona in PERSONAS:
            results = tables[persona]
            if i < len(results):
                r = results[i]
                cell = f"{i + 1}. {r['ticker']:<6} {r['score']:.2f}"
            else:
                cell = ""
            row += f"{cell:<{width}}"
        print(row)

    print(f"\nUniverse: {universe} companies, identical for all three.")
    overlap = set.intersection(*[
        {r["ticker"] for r in tables[p]} for p in PERSONAS])
    print(f"Names appearing in all three top-{limit}: "
          f"{sorted(overlap) if overlap else 'none'}")
    print("\nWhy they differ:")
    for persona in PERSONAS:
        top = tables[persona][0] if tables[persona] else None
        if top:
            driver = top["contributions"][0]
            print(f"  {personas[persona].short_label:<16} picks {top['ticker']:<6} "
                  f"mainly on {driver['metric']} "
                  f"({pct_rank(driver['percentile_in_sector'])}, "
                  f"weight {driver['weight']:.0%}, {driver['direction']} is better)")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sector", default="tech", choices=sorted(load_sectors()))
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(run(args.sector, args.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
