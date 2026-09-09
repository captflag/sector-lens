"""MCP server exposing the sector database as tools.

This is the protocol boundary. Nothing above this line
imports the database: the agent process holds no SQLite handle, no SQL and no
schema knowledge. It discovers what it can do by calling `list_tools` over MCP
and acts only through `call_tool`. Swapping SQLite for a warehouse, or moving
this server onto another host over streamable HTTP, would not touch the agent.

Run standalone (stdio):

    python -m sectorlens.mcp_server.server

The agent launches exactly this command as a subprocess -- see
`sectorlens.agent.mcp_client`.

Safety posture: the connection is opened read-only (SQLite `mode=ro`) and every
statement is parameterised, so a tool argument -- including one a model was
talked into producing -- cannot write to or drop anything.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer

from ..config import load_personas, load_sectors
from ..db import queries as q
from ..db.store import connect
from ..settings import get_settings

# Constrain the two enumerable arguments in the tool schema itself, built from
# the same YAML the agent reads -- so adding a sector still means editing one
# config file, and the model is told the valid values rather than left to guess
# from a description. A live run showed why this matters: the model called
# screen_sector with persona="private equity", got an error payload back, and
# burned a round trip recovering. An enum makes that call unrepresentable.
SectorId = Literal[tuple(load_sectors())]      # type: ignore[valid-type]
PersonaId = Literal[tuple(load_personas())]    # type: ignore[valid-type]

mcp = MCPServer(
    name="sectorlens-sector-intel",
    version="0.1.0",
    instructions=(
        "Sector intelligence over a curated database of public companies. "
        "Every number you cite must come from one of these tools. If a tool "
        "reports that a company or metric is absent, say so plainly rather "
        "than filling the gap from prior knowledge -- the database is the only "
        "source of truth you have here."
    ),
)


def _conn() -> sqlite3.Connection:
    """A fresh read-only connection per call.

    SQLite connections are not safe to share across threads, and the queries
    here are short; opening per call is cheaper than the bugs that come from
    pooling one across an async server.
    """
    return connect(get_settings().resolved_db_path, read_only=True)


@mcp.tool(
    description=(
        "List the sectors this database covers and how many companies are "
        "loaded for each. Call this first when you are unsure whether a "
        "sector is in scope -- no other sector can be answered from data."
    )
)
def list_sectors() -> dict[str, Any]:
    with _conn() as c:
        return q.list_sectors(c)


@mcp.tool(
    description=(
        "List the companies loaded for one sector. Use it to establish the "
        "universe you are reasoning over before making sector-level claims."
    )
)
def list_companies(sector: SectorId, limit: int = 50) -> dict[str, Any]:
    """Args:
    sector: sector id, e.g. tech, retail, manufacturing, logistics.
    limit: maximum companies to return (capped at 100).
    """
    with _conn() as c:
        return q.list_companies(c, sector, limit)


@mcp.tool(
    description=(
        "Check whether a company is in the database by ticker or name. Call "
        "this before discussing ANY specific company the user names. If it "
        "returns in_database=false, state that you hold no data on that "
        "company instead of answering from general knowledge."
    )
)
def find_company(query: str) -> dict[str, Any]:
    """Args:
    query: a ticker or company name, e.g. "NVDA" or "Union Pacific".
    """
    with _conn() as c:
        return q.find_company(c, query)


@mcp.tool(
    description=(
        "Full stored profile for one company: identifiers, every latest metric "
        "with its source, provenance and whether it was reported or derived, "
        "plus any data-quality findings recorded against it."
    )
)
def get_company_profile(ticker: str) -> dict[str, Any]:
    """Args:
    ticker: exchange ticker, e.g. "UPS".
    """
    with _conn() as c:
        return q.get_company_profile(c, ticker)


@mcp.tool(
    description=(
        "Workforce and hiring signals held for one company (headcount and "
        "similar). Returns an explicit empty result with guidance when nothing "
        "is held -- treat that as the answer, not as a prompt to estimate."
    )
)
def get_company_signals(ticker: str, signal_type: str = "headcount") -> dict[str, Any]:
    """Args:
    ticker: exchange ticker, e.g. "FDX".
    signal_type: signal to fetch; "headcount" is the one usually populated.
    """
    with _conn() as c:
        return q.get_company_signals(c, ticker, signal_type or None)


@mcp.tool(
    description=(
        "Rank a sector's companies through one persona's weighting. The "
        "weights differ per persona, so the ordering and the supporting "
        "metrics returned differ too -- this is the primary evidence source "
        "for any 'which companies' question. Each result carries the metric "
        "values, their percentile inside the sector, and each metric's "
        "contribution to the score, so the ranking can be explained rather "
        "than asserted."
    )
)
def screen_sector(sector: SectorId, persona: PersonaId, limit: int = 10) -> dict[str, Any]:
    """Args:
    sector: sector id, e.g. tech, retail, manufacturing, logistics.
    persona: one of mutual_fund_analyst, equity_analyst, pe_analyst.
    limit: how many ranked companies to return (capped at 100).
    """
    with _conn() as c:
        return q.screen_sector(c, sector, persona, limit)


@mcp.tool(
    description=(
        "Median and quartile values for a sector's metrics, computed across "
        "the loaded peer group. Use this whenever a claim is comparative "
        "('cheap', 'high margin', 'above average') so the comparison is "
        "anchored to a real number."
    )
)
def get_sector_benchmarks(sector: SectorId, metrics: list[str] | None = None) -> dict[str, Any]:
    """Args:
    sector: sector id.
    metrics: optional metric codes to restrict to, e.g. ["pe_ratio"].
    """
    with _conn() as c:
        return q.get_sector_benchmarks(c, sector, metrics)


@mcp.tool(
    description=(
        "Compare named companies side by side on the same metrics. Reports "
        "which requested tickers are not in the database."
    )
)
def compare_companies(tickers: list[str], metrics: list[str] | None = None) -> dict[str, Any]:
    """Args:
    tickers: tickers to compare, e.g. ["UPS", "FDX"].
    metrics: optional metric codes to restrict to.
    """
    with _conn() as c:
        return q.compare_companies(c, tickers, metrics)


@mcp.tool(
    description=(
        "The full history held for one metric on one company, oldest first, "
        "with the direction of travel. Use this for any question about whether "
        "something is improving or deteriorating -- a single latest value is a "
        "level, not a trend, and this tool says so explicitly when only one "
        "period exists."
    )
)
def get_metric_history(ticker: str, metric: str) -> dict[str, Any]:
    """Args:
    ticker: exchange ticker, e.g. "UNP".
    metric: metric code, e.g. "ebitda_margin" or "revenue_ttm".
    """
    with _conn() as c:
        return q.get_metric_history(c, ticker, metric)


@mcp.tool(
    description=(
        "What this database contains and where it should not be trusted: "
        "coverage per sector, the upstream sources with licences and retrieval "
        "times, recent ingest runs, and open data-quality findings. Call this "
        "when a question turns on data freshness or reliability, or when you "
        "need to qualify an answer honestly."
    )
)
def describe_data_coverage(sector: SectorId | None = None) -> dict[str, Any]:
    """Args:
    sector: optional sector id to narrow the report to.
    """
    with _conn() as c:
        return q.describe_data_coverage(c, sector)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
