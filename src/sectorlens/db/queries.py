"""Read-only query layer behind the MCP tools.

Kept separate from the MCP server so the retrieval logic -- especially the
persona-weighted screen -- is unit-testable without a protocol round trip, and
so the server module stays a thin adapter.

Every query is parameterised. Tool arguments reach SQL only as bound values,
never as string interpolation, and the connection is opened read-only, so no
tool call can mutate the database.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any, Iterable, Sequence

from ..config import Persona, get_persona, load_sectors

# Guards against a caller (or a model) asking for the whole table at once.
MAX_LIMIT = 100
DEFAULT_LIMIT = 10

# A company must carry at least this share of a persona's total screen weight
# before it is ranked. Below it, a company with two of seven metrics could
# outrank a fully-covered peer purely by having less to be judged on.
MIN_WEIGHT_COVERAGE = 0.45


def _clamp(limit: int | None, default: int = DEFAULT_LIMIT) -> int:
    if not limit or limit < 1:
        return default
    return min(int(limit), MAX_LIMIT)


def _rows(conn: sqlite3.Connection, sql: str,
          params: Sequence[Any] = ()) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------

def list_sectors(conn: sqlite3.Connection) -> dict[str, Any]:
    configured = load_sectors()
    loaded = {
        r["sector"]: r["n"]
        for r in conn.execute(
            "SELECT sector, COUNT(*) n FROM companies GROUP BY sector")
    }
    return {
        "sectors": [
            {
                "id": sid,
                "label": s.label,
                "description": s.description,
                "companies_loaded": loaded.get(sid, 0),
            }
            for sid, s in configured.items()
        ],
        "note": "Only these sectors exist in this database. A question about "
                "any other sector cannot be answered from the data.",
    }


def list_companies(conn: sqlite3.Connection, sector: str,
                   limit: int | None = None) -> dict[str, Any]:
    if sector not in load_sectors():
        return {"sector": sector, "error": "unknown_sector",
                "valid_sectors": sorted(load_sectors())}
    rows = _rows(
        conn,
        """SELECT ticker, name, gics_sub_industry, hq_location, cik
           FROM companies WHERE sector = ? ORDER BY name LIMIT ?""",
        (sector, _clamp(limit, MAX_LIMIT)),
    )
    total = int(conn.execute(
        "SELECT COUNT(*) FROM companies WHERE sector = ?", (sector,)
    ).fetchone()[0])
    return {"sector": sector, "total_in_sector": total,
            "returned": len(rows), "companies": rows}


def find_company(conn: sqlite3.Connection, query: str) -> dict[str, Any]:
    """Resolve a free-text company reference against the loaded universe.

    Deliberately returns an explicit `in_database: false` rather than a nearest
    guess when nothing matches. The agent is required to say it holds no data
    on a company instead of reasoning about it from background knowledge, and
    that only works if the tool reports absence unambiguously.
    """
    q = (query or "").strip()
    if not q:
        return {"query": query, "in_database": False, "matches": []}

    exact = _rows(
        conn,
        """SELECT ticker, name, sector, gics_sub_industry
           FROM companies
           WHERE UPPER(ticker) = UPPER(?) OR UPPER(name) = UPPER(?)""",
        (q, q),
    )
    if exact:
        return {"query": query, "in_database": True, "match_type": "exact",
                "matches": exact}

    partial = _rows(
        conn,
        """SELECT ticker, name, sector, gics_sub_industry
           FROM companies
           WHERE name LIKE ? OR ticker LIKE ?
           ORDER BY LENGTH(name) LIMIT 8""",
        (f"%{q}%", f"%{q}%"),
    )
    if partial:
        return {"query": query, "in_database": True, "match_type": "partial",
                "matches": partial}

    return {
        "query": query,
        "in_database": False,
        "matches": [],
        "guidance": (
            f"No company matching {query!r} is loaded in this database. Say so "
            f"plainly. Do not describe this company from prior knowledge, "
            f"estimate its figures, or substitute a similar company without "
            f"saying that is what you are doing."
        ),
    }


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------

def get_company_profile(conn: sqlite3.Connection, ticker: str) -> dict[str, Any]:
    company = conn.execute(
        """SELECT id, ticker, name, sector, gics_sector, gics_sub_industry,
                  hq_location, cik, founded, index_added_date
           FROM companies WHERE UPPER(ticker) = UPPER(?)""",
        (ticker,),
    ).fetchone()
    if company is None:
        return find_company(conn, ticker)

    company = dict(company)
    facts = _rows(
        conn,
        """SELECT metric_code, metric_label, unit, value, period_end,
                  fiscal_period, is_derived, derivation, source_key,
                  source_name, source_url, source_retrieved_at
           FROM v_company_facts WHERE company_id = ?
           ORDER BY metric_code""",
        (company["id"],),
    )
    signals = _rows(
        conn,
        """SELECT s.signal_type, s.value_num, s.value_text, s.as_of,
                  src.name AS source_name, src.url AS source_url
           FROM company_signals s
           LEFT JOIN sources src ON src.id = s.source_id
           WHERE s.company_id = ? ORDER BY s.as_of DESC""",
        (company["id"],),
    )
    quality = _rows(
        conn,
        """SELECT check_name, severity, detail FROM data_quality_findings
           WHERE scope = 'company' AND ref = ?""",
        (company["ticker"],),
    )
    company.pop("id", None)
    return {
        "company": company,
        "metrics": facts,
        "signals": signals,
        "signals_held": sorted({s["signal_type"] for s in signals}),
        "data_quality_findings": quality,
    }


def get_company_signals(conn: sqlite3.Connection, ticker: str,
                        signal_type: str | None = None) -> dict[str, Any]:
    """Headcount / hiring style signals for one company.

    The hallucination-prone case. When nothing is held, the
    response says so explicitly and names what would populate it, so the agent
    has no reason to reach for background knowledge.
    """
    company = conn.execute(
        "SELECT id, ticker, name FROM companies WHERE UPPER(ticker) = UPPER(?)",
        (ticker,),
    ).fetchone()
    if company is None:
        return find_company(conn, ticker)

    sql = """SELECT s.signal_type, s.value_num, s.value_text, s.as_of,
                    src.name AS source_name, src.url AS source_url
             FROM company_signals s
             LEFT JOIN sources src ON src.id = s.source_id
             WHERE s.company_id = ?"""
    params: list[Any] = [company["id"]]
    if signal_type:
        sql += " AND s.signal_type = ?"
        params.append(signal_type)
    sql += " ORDER BY s.as_of DESC"

    signals = _rows(conn, sql, params)
    total_held = int(conn.execute(
        "SELECT COUNT(*) FROM company_signals WHERE signal_type = 'headcount'"
    ).fetchone()[0])

    out: dict[str, Any] = {
        "ticker": company["ticker"],
        "name": company["name"],
        "signal_type": signal_type,
        "signals": signals,
        "count": len(signals),
    }
    if not signals:
        out["guidance"] = (
            f"This database holds no {signal_type or 'workforce'} signal for "
            f"{company['ticker']}. State that directly. Do not estimate "
            f"headcount, infer it from market cap or revenue, or recall it "
            f"from general knowledge. "
            f"({total_held} companies in the database carry a headcount "
            f"signal; the currently loaded ingest adapter supplies headcount "
            f"only where a filer tags dei:EntityNumberOfEmployees, so gaps "
            f"are expected.)"
        )
    return out


def get_sector_benchmarks(conn: sqlite3.Connection, sector: str,
                          metrics: Iterable[str] | None = None) -> dict[str, Any]:
    if sector not in load_sectors():
        return {"sector": sector, "error": "unknown_sector",
                "valid_sectors": sorted(load_sectors())}
    sql = """SELECT b.metric_code, m.label AS metric_label, m.unit,
                    b.median, b.p25, b.p75, b.mean, b.n, b.computed_at
             FROM sector_benchmarks b JOIN metrics m ON m.code = b.metric_code
             WHERE b.sector = ?"""
    params: list[Any] = [sector]
    codes = [c for c in (metrics or []) if c]
    if codes:
        sql += f" AND b.metric_code IN ({','.join('?' * len(codes))})"
        params.extend(codes)
    sql += " ORDER BY b.metric_code"
    return {"sector": sector, "benchmarks": _rows(conn, sql, params),
            "note": "Medians and quartiles are computed across the companies "
                    "loaded for this sector only -- they are a peer-group "
                    "statistic, not a published index level."}


def compare_companies(conn: sqlite3.Connection, tickers: Sequence[str],
                      metrics: Sequence[str] | None = None) -> dict[str, Any]:
    if not tickers:
        return {"error": "no_tickers_supplied"}
    tickers = [t.strip().upper() for t in tickers if t and t.strip()][:12]
    placeholders = ",".join("?" * len(tickers))
    sql = f"""SELECT ticker, name, sector, metric_code, metric_label, unit,
                     value, period_end, is_derived
              FROM v_company_facts
              WHERE UPPER(ticker) IN ({placeholders})"""
    params: list[Any] = list(tickers)
    codes = [c for c in (metrics or []) if c]
    if codes:
        sql += f" AND metric_code IN ({','.join('?' * len(codes))})"
        params.extend(codes)
    sql += " ORDER BY ticker, metric_code"

    rows = _rows(conn, sql, params)
    found = {r["ticker"].upper() for r in rows}
    table: dict[str, dict[str, Any]] = {}
    for r in rows:
        entry = table.setdefault(r["ticker"], {"name": r["name"],
                                               "sector": r["sector"],
                                               "metrics": {}})
        entry["metrics"][r["metric_code"]] = {
            "value": r["value"], "unit": r["unit"],
            "period_end": r["period_end"], "is_derived": bool(r["is_derived"]),
        }
    return {"requested": tickers, "not_in_database": sorted(set(tickers) - found),
            "companies": table}


# ---------------------------------------------------------------------------
# Persona-weighted screening
# ---------------------------------------------------------------------------

def _percentile_rank(value: float, population: Sequence[float]) -> float:
    """Fraction of the population this value beats, ties counted at half.

    Percentile ranking rather than z-scoring because these metrics are heavily
    skewed -- a handful of mega-caps would dominate any mean/standard-deviation
    scheme and flatten the rest of the sector into noise.
    """
    n = len(population)
    if n <= 1:
        return 0.5
    below = sum(1 for v in population if v < value)
    ties = sum(1 for v in population if v == value)
    return (below + 0.5 * ties) / n


def screen_sector(conn: sqlite3.Connection, sector: str, persona: str,
                  limit: int | None = None) -> dict[str, Any]:
    """Rank a sector's companies through one persona's weighting.

    This is where persona stops being a matter of tone. The same sector and the
    same underlying rows produce a different ordering, and a different set of
    supporting numbers, for each persona -- before any language model is
    involved. Two weights in the PE persona deliberately invert the
    public-market ones (see personas.yaml), so the top of the PE list and the
    top of the mutual-fund list are genuinely different companies.
    """
    if sector not in load_sectors():
        return {"sector": sector, "error": "unknown_sector",
                "valid_sectors": sorted(load_sectors())}
    try:
        p: Persona = get_persona(persona)
    except KeyError as exc:
        return {"persona": persona, "error": "unknown_persona", "detail": str(exc)}

    weights = {w.metric: w for w in p.screen_weights}
    codes = list(weights)
    placeholders = ",".join("?" * len(codes))

    rows = _rows(
        conn,
        f"""SELECT company_id, ticker, name, gics_sub_industry, metric_code,
                   value, unit, period_end, is_derived
            FROM v_company_facts
            WHERE sector = ? AND metric_code IN ({placeholders})""",
        [sector, *codes],
    )
    if not rows:
        return {"sector": sector, "persona": persona, "results": [],
                "error": "no_data_for_sector"}

    by_company: dict[str, dict[str, Any]] = {}
    population: dict[str, list[float]] = {}
    for r in rows:
        entry = by_company.setdefault(
            r["ticker"], {"ticker": r["ticker"], "name": r["name"],
                          "gics_sub_industry": r["gics_sub_industry"],
                          "metrics": {}})
        entry["metrics"][r["metric_code"]] = r
        population.setdefault(r["metric_code"], []).append(float(r["value"]))

    scored: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    for entry in by_company.values():
        contributions: list[dict[str, Any]] = []
        weighted_sum = 0.0
        covered_weight = 0.0

        for code, w in weights.items():
            fact = entry["metrics"].get(code)
            if fact is None:
                continue
            value = float(fact["value"])
            pr = _percentile_rank(value, population[code])
            if not w.higher_is_better:
                pr = 1.0 - pr
            weighted_sum += w.weight * pr
            covered_weight += w.weight
            contributions.append({
                "metric": code,
                "value": value,
                "unit": fact["unit"],
                "percentile_in_sector": round(pr, 4),
                "weight": w.weight,
                "direction": w.direction,
                "contribution": round(w.weight * pr, 4),
                "is_derived": bool(fact["is_derived"]),
                "period_end": fact["period_end"],
            })

        record = {
            "ticker": entry["ticker"],
            "name": entry["name"],
            "gics_sub_industry": entry["gics_sub_industry"],
            "metric_coverage": round(covered_weight, 4),
            "contributions": sorted(contributions,
                                    key=lambda c: c["contribution"], reverse=True),
        }
        if covered_weight < MIN_WEIGHT_COVERAGE:
            record["reason"] = (
                f"only {covered_weight:.0%} of this persona's screen weight is "
                f"covered by available data (minimum {MIN_WEIGHT_COVERAGE:.0%})")
            excluded.append(record)
            continue
        # Renormalise so a company is scored on what it has, not punished for
        # gaps in the source data.
        record["score"] = round(weighted_sum / covered_weight, 4)
        scored.append(record)

    scored.sort(key=lambda r: r["score"], reverse=True)
    top = scored[: _clamp(limit)]
    for rank, record in enumerate(top, start=1):
        record["rank"] = rank

    return {
        "sector": sector,
        "persona": persona,
        "persona_label": p.label,
        "lens": p.lens,
        "ranking_basis": {
            "method": "weighted mean of within-sector percentile ranks, "
                      "renormalised over the metrics each company actually has",
            "weights": {m: {"weight": w.weight, "direction": w.direction}
                        for m, w in weights.items()},
            "rationale": p.rationale,
        },
        "universe_size": len(by_company),
        "ranked": len(scored),
        "results": top,
        "excluded_for_thin_data": excluded[:5],
        "caveat": "Scores are relative positions inside this sector and this "
                  "database only. They are a screen to focus attention, not a "
                  "recommendation, and they carry the data-quality caveats "
                  "reported by describe_data_coverage.",
    }


def describe_data_coverage(conn: sqlite3.Connection,
                           sector: str | None = None) -> dict[str, Any]:
    """What this database actually contains, and where it should not be trusted."""
    where, params = ("", [])
    if sector:
        if sector not in load_sectors():
            return {"error": "unknown_sector", "valid_sectors": sorted(load_sectors())}
        where, params = (" WHERE c.sector = ?", [sector])

    per_sector = _rows(
        conn,
        f"""SELECT c.sector, COUNT(DISTINCT c.id) AS companies,
                   COUNT(v.id) AS metric_values
            FROM companies c
            LEFT JOIN v_company_latest_metrics v ON v.company_id = c.id
            {where.replace('WHERE', 'WHERE') if where else ''}
            GROUP BY c.sector ORDER BY c.sector""",
        params,
    )
    sources = _rows(
        conn,
        """SELECT key, name, url, publisher, license, retrieved_at, notes
           FROM sources ORDER BY key""")
    runs = _rows(
        conn,
        """SELECT adapter, started_at, finished_at, status, companies,
                  metric_values, signals
           FROM ingest_runs ORDER BY id DESC LIMIT 5""")
    findings_sql = """SELECT scope, ref, check_name, severity, detail
                      FROM data_quality_findings
                      WHERE severity IN ('warn', 'error')"""
    findings_params: list[Any] = []
    if sector:
        findings_sql += " AND (scope = 'dataset' OR ref = ? OR ref LIKE ?)"
        findings_params = [sector, f"{sector}:%"]
    findings_sql += " ORDER BY CASE severity WHEN 'error' THEN 0 ELSE 1 END LIMIT 40"

    signals_by_type = _rows(
        conn,
        """SELECT signal_type, COUNT(*) AS n,
                  COUNT(DISTINCT company_id) AS companies
           FROM company_signals GROUP BY signal_type""")

    return {
        "scope": sector or "all sectors",
        "coverage_by_sector": per_sector,
        "signals": signals_by_type or [],
        "signals_note": (
            "No workforce or hiring signals are loaded."
            if not signals_by_type else
            "Workforce signals are present for a subset of companies only."),
        "sources": sources,
        "recent_ingest_runs": runs,
        "data_quality_findings": _rows(conn, findings_sql, findings_params),
        "how_to_use": (
            "Treat every warn/error finding as a live constraint on the answer. "
            "If a question depends on a metric flagged here, say so rather than "
            "quoting the number as if it were clean."),
    }


def get_metric_history(conn: sqlite3.Connection, ticker: str,
                       metric: str) -> dict[str, Any]:
    """One metric for one company across every period held.

    Separate from `get_company_profile`, which returns the latest value of
    everything. A question about direction -- is the margin expanding or
    eroding -- needs the series, and a level cannot answer it. Where only one
    period is held the response says so rather than letting a single point be
    read as a trend.
    """
    company = conn.execute(
        "SELECT id, ticker, name FROM companies WHERE UPPER(ticker) = UPPER(?)",
        (ticker,),
    ).fetchone()
    if company is None:
        return find_company(conn, ticker)

    from .metrics import METRICS_BY_CODE

    if metric not in METRICS_BY_CODE:
        return {"error": "unknown_metric", "metric": metric,
                "valid_metrics": sorted(METRICS_BY_CODE)}

    rows = _rows(
        conn,
        """SELECT cm.period_end, cm.fiscal_period, cm.value, cm.is_derived,
                  cm.derivation, s.name AS source_name
           FROM company_metrics cm
           LEFT JOIN sources s ON s.id = cm.source_id
           WHERE cm.company_id = ? AND cm.metric_code = ?
           ORDER BY COALESCE(cm.period_end, '0000-00-00')""",
        (company["id"], metric),
    )

    definition = METRICS_BY_CODE[metric]
    out: dict[str, Any] = {
        "ticker": company["ticker"],
        "name": company["name"],
        "metric": metric,
        "label": definition.label,
        "unit": definition.unit,
        "periods": rows,
        "count": len(rows),
    }

    if not rows:
        out["guidance"] = (
            f"No values of {metric!r} are held for {company['ticker']}. Say so "
            f"rather than estimating a direction.")
        return out

    if len(rows) == 1:
        out["guidance"] = (
            f"Only one period is held for {company['ticker']} "
            f"({rows[0]['period_end'] or 'undated'}), so this is a level and "
            f"not a trend. Do not describe it as rising or falling.")
        return out

    first, last = rows[0], rows[-1]
    change = last["value"] - first["value"]
    out["change"] = {
        "from_period": first["period_end"],
        "to_period": last["period_end"],
        "from_value": first["value"],
        "to_value": last["value"],
        "absolute_change": change,
        "direction": "rising" if change > 0 else "falling" if change < 0 else "flat",
    }
    return out


#: FTS5 has its own query grammar, so a natural-language question cannot be
#: passed through untouched -- an apostrophe or a bare "AND" is a syntax error,
#: and a hyphen means NOT. Words are extracted and re-quoted instead.
_FTS_TOKEN = re.compile(r"[A-Za-z0-9']{3,}")

#: Words that match half the corpus and rank nothing usefully.
_FTS_STOPWORDS = frozenset({
    "the", "and", "for", "are", "was", "were", "with", "that", "this", "from",
    "what", "which", "how", "why", "does", "did", "has", "have", "had", "you",
    "your", "our", "their", "its", "about", "any", "all", "can", "will",
    "would", "should", "there", "they", "them", "than", "then", "into",
})


def _fts_query(text: str) -> str:
    """Turn a question into a safe FTS5 query.

    Tokens are OR-ed rather than AND-ed: a question phrased in a user's words
    rarely shares every term with the passage that answers it, and BM25 already
    ranks documents matching more of them higher. Requiring all terms would
    mostly return nothing.
    """
    tokens = [t.lower() for t in _FTS_TOKEN.findall(text or "")]
    tokens = [t for t in tokens if t not in _FTS_STOPWORDS]
    seen: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.append(token)
    return " OR ".join(f'"{t}"' for t in seen[:24])


def search_filings(conn: sqlite3.Connection, query: str,
                   sector: str | None = None, ticker: str | None = None,
                   section: str | None = None,
                   limit: int | None = None) -> dict[str, Any]:
    """Search the narrative sections of annual filings.

    Returns passages with the company, filing date and section they came from,
    so a claim drawn from one can be attributed rather than asserted. Ranked by
    BM25.
    """
    # A database built before the text tables existed is still a valid
    # database; it simply holds no filings. Missing tables must read as "no
    # text loaded", not as a crash.
    try:
        total_chunks = int(conn.execute(
            "SELECT COUNT(*) FROM filing_chunks").fetchone()[0])
    except sqlite3.OperationalError:
        total_chunks = 0

    if not total_chunks:
        return {
            "query": query,
            "results": [],
            "count": 0,
            "guidance": (
                "No filing text is loaded in this database, so there is nothing "
                "to quote. Say that plainly rather than paraphrasing a filing "
                "from prior knowledge. Filing text is loaded separately with "
                "`python -m sectorlens.ingest.fetch_filings`."),
        }

    match = _fts_query(query)
    if not match:
        return {"query": query, "results": [], "count": 0,
                "guidance": "The question carried no searchable terms."}

    sql = """
        SELECT c.text AS text,
               c.section AS section,
               c.ordinal AS ordinal,
               co.ticker AS ticker,
               co.name AS name,
               co.sector AS sector,
               f.form AS form,
               f.filed_date AS filed_date,
               f.period_end AS period_end,
               f.source_url AS source_url,
               bm25(filing_chunks_fts) AS rank
        FROM filing_chunks_fts
        JOIN filing_chunks c ON c.id = filing_chunks_fts.rowid
        JOIN filings f ON f.id = c.filing_id
        JOIN companies co ON co.id = f.company_id
        WHERE filing_chunks_fts MATCH ?
    """
    params: list[Any] = [match]
    if sector:
        sql += " AND co.sector = ?"
        params.append(sector)
    if ticker:
        sql += " AND UPPER(co.ticker) = UPPER(?)"
        params.append(ticker)
    if section:
        sql += " AND c.section = ?"
        params.append(section)
    sql += " ORDER BY rank LIMIT ?"
    params.append(_clamp(limit, 6))

    rows = _rows(conn, sql, params)
    out: dict[str, Any] = {
        "query": query,
        "filters": {"sector": sector, "ticker": ticker, "section": section},
        "results": rows,
        "count": len(rows),
        "corpus_chunks": total_chunks,
    }
    if not rows:
        out["guidance"] = (
            "Nothing in the loaded filings matches this question. Say so "
            "rather than answering from general knowledge about the company.")
    else:
        out["note"] = (
            "These are passages a company wrote about itself in a filing. "
            "Quote or paraphrase them with the ticker and filing date "
            "attached, and do not present them as this system's own analysis.")
    return out


def describe_filing_coverage(conn: sqlite3.Connection) -> dict[str, Any]:
    """Which companies have filing text, and from when."""
    try:
        conn.execute("SELECT 1 FROM filings LIMIT 1")
    except sqlite3.OperationalError:
        return {"companies_with_text": 0, "filings": 0, "chunks": 0,
                "by_sector": [], "sections": [],
                "note": "This database predates the filing-text tables. "
                        "Rebuild it to add them."}
    return {
        "companies_with_text": int(conn.execute(
            "SELECT COUNT(DISTINCT company_id) FROM filings").fetchone()[0]),
        "filings": int(conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0]),
        "chunks": int(conn.execute(
            "SELECT COUNT(*) FROM filing_chunks").fetchone()[0]),
        "by_sector": _rows(conn, """
            SELECT co.sector, COUNT(DISTINCT f.company_id) AS companies,
                   COUNT(DISTINCT f.id) AS filings
            FROM filings f JOIN companies co ON co.id = f.company_id
            GROUP BY co.sector ORDER BY co.sector"""),
        "sections": _rows(conn, """
            SELECT section, COUNT(*) AS chunks FROM filing_chunks
            GROUP BY section ORDER BY section"""),
    }
