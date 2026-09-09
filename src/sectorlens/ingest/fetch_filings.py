"""Fetch and index annual filing text for the companies already loaded.

Run this after the numeric build, wherever you have outbound access to SEC:

    python -m sectorlens.ingest.fetch_filings --limit-per-sector 5 --years 1

It adds to the existing database rather than replacing it, so the numeric side
is untouched. SEC requires a real contact address -- set
SECTORLENS_SEC_USER_AGENT before running, or requests are throttled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import load_sectors
from ..db.store import connect, finish_run, record_finding, start_run
from ..settings import get_settings
from .filings import (ARCHIVE_URL, SUBMISSIONS_URL, build_client,
                      extract_section, html_to_text, recent_annual_filings,
                      register_source, sleep_for_rate_limit, store_filing,
                      SECTION_PATTERNS)


def fetch(db_path: Path, sectors: list[str] | None, limit_per_sector: int,
          years: int) -> int:
    conn = connect(db_path)
    run_id = start_run(conn, "filings")
    source_id = register_source(conn)

    where, params = "", []
    if sectors:
        placeholders = ",".join("?" * len(sectors))
        where = f" WHERE sector IN ({placeholders})"
        params = list(sectors)

    companies = conn.execute(
        f"""SELECT id, ticker, name, cik, sector FROM companies{where}
            ORDER BY sector, ticker""", params).fetchall()

    by_sector: dict[str, list] = {}
    for row in companies:
        if row["cik"]:
            by_sector.setdefault(row["sector"], []).append(row)

    total_filings = total_chunks = 0
    skipped: list[str] = []

    with build_client() as client:
        for sector, rows in by_sector.items():
            for row in rows[:limit_per_sector]:
                cik = str(row["cik"]).zfill(10)
                try:
                    sleep_for_rate_limit()
                    submissions = client.get(
                        SUBMISSIONS_URL.format(cik=cik)).raise_for_status().json()
                except Exception as exc:  # noqa: BLE001
                    skipped.append(f"{row['ticker']}: submissions — {exc}")
                    continue

                refs = recent_annual_filings(submissions, limit=years)
                if not refs:
                    skipped.append(f"{row['ticker']}: no 10-K in recent filings")
                    continue

                for ref in refs:
                    url = ARCHIVE_URL.format(
                        cik_int=int(cik), accession=ref.accession,
                        document=ref.document)
                    try:
                        sleep_for_rate_limit()
                        html = client.get(url).raise_for_status().text
                    except Exception as exc:  # noqa: BLE001
                        skipped.append(f"{row['ticker']} {ref.accession}: {exc}")
                        continue

                    text = html_to_text(html)
                    sections = {}
                    for name in SECTION_PATTERNS:
                        body = extract_section(text, name)
                        if body:
                            sections[name] = body

                    if not sections:
                        # Worth recording rather than silently dropping: it
                        # means the heading patterns missed this filer's layout.
                        record_finding(
                            conn, scope="company", ref=row["ticker"],
                            check_name="filing_sections_not_found",
                            severity="info",
                            detail=(f"No recognised item headings in "
                                    f"{ref.accession} ({len(text):,} chars of "
                                    f"text). The filing was fetched but not "
                                    f"indexed."),
                            run_id=run_id)
                        skipped.append(f"{row['ticker']} {ref.accession}: no sections")
                        continue

                    total_chunks += store_filing(
                        conn, company_id=row["id"], ref=ref, sections=sections,
                        source_id=source_id, run_id=run_id, url=url)
                    total_filings += 1
                    print(f"  {row['ticker']:<6} {ref.period_end or ref.filed_date}  "
                          f"{', '.join(f'{k}:{len(v):,}c' for k, v in sections.items())}")
                conn.commit()

    conn.commit()
    finish_run(conn, run_id, status="ok", metric_values=total_chunks)

    print(f"\n[filings] {total_filings} filings indexed, "
          f"{total_chunks} chunks written")
    if skipped:
        print(f"[filings] skipped {len(skipped)} (first 5):")
        for line in skipped[:5]:
            print(f"    {line}")
    coverage = conn.execute(
        """SELECT COUNT(DISTINCT f.company_id), COUNT(DISTINCT f.id),
                  COUNT(c.id)
           FROM filings f LEFT JOIN filing_chunks c ON c.filing_id = f.id"""
    ).fetchone()
    print(f"[filings] database now holds text for {coverage[0]} companies "
          f"across {coverage[1]} filings ({coverage[2]} chunks)")
    conn.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=settings.resolved_db_path)
    parser.add_argument("--sectors", nargs="*", default=None,
                        choices=sorted(load_sectors()))
    parser.add_argument("--limit-per-sector", type=int, default=5,
                        help="companies per sector; each costs two requests "
                             "per filing, so keep it modest")
    parser.add_argument("--years", type=int, default=1,
                        help="most recent 10-K filings per company")
    args = parser.parse_args(argv)

    if not args.db.exists():
        print("Build the numeric database first: "
              "python -m sectorlens.ingest.build_db", file=sys.stderr)
        return 2
    if "example.com" in settings.sec_user_agent:
        print("Set SECTORLENS_SEC_USER_AGENT to a real contact address first; "
              "SEC throttles requests without one.", file=sys.stderr)
        return 2

    return fetch(args.db, args.sectors, args.limit_per_sector, args.years)


if __name__ == "__main__":
    sys.exit(main())
