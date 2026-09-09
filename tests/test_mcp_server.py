"""The MCP boundary itself.

These tests drive the server through a real `mcp.Client`, so tool discovery,
argument validation and result serialisation are all exercised over the
protocol rather than by calling the Python functions directly.
"""

from __future__ import annotations

import json

import pytest
from mcp import Client

from sectorlens.db.store import connect
from sectorlens.settings import Settings

EXPECTED_TOOLS = {
    "list_sectors", "list_companies", "find_company", "get_company_profile",
    "get_company_signals", "screen_sector", "get_sector_benchmarks",
    "compare_companies", "describe_data_coverage", "get_metric_history",
    "search_filings",
}


@pytest.fixture
def mcp_app(monkeypatch, fixture_db):
    """The real server module, pointed at the fixture database."""
    from sectorlens.mcp_server import server as server_module

    settings = Settings(db_path=fixture_db)
    monkeypatch.setattr(server_module, "get_settings", lambda: settings)
    return server_module.mcp


def _payload(result):
    if getattr(result, "structured_content", None):
        return result.structured_content
    return json.loads(result.content[0].text)


async def test_all_tools_are_discoverable(mcp_app):
    async with Client(mcp_app) as client:
        tools = (await client.list_tools()).tools
        assert {t.name for t in tools} == EXPECTED_TOOLS


async def test_every_tool_is_described_for_the_model(mcp_app):
    """Tool descriptions are the agent's only instructions on when to call
    what, so an undescribed tool is a silent behavioural bug."""
    async with Client(mcp_app) as client:
        for tool in (await client.list_tools()).tools:
            assert tool.description and len(tool.description) > 40, tool.name


async def test_screen_sector_round_trips(mcp_app):
    async with Client(mcp_app) as client:
        result = await client.call_tool(
            "screen_sector",
            {"sector": "tech", "persona": "pe_analyst", "limit": 3})
        data = _payload(result)
        assert data["persona"] == "pe_analyst"
        assert len(data["results"]) == 3
        assert data["results"][0]["rank"] == 1


async def test_personas_differ_across_the_protocol(mcp_app):
    """The differentiation must survive the protocol boundary, not just exist
    inside the query layer."""
    async with Client(mcp_app) as client:
        mf = _payload(await client.call_tool(
            "screen_sector",
            {"sector": "tech", "persona": "mutual_fund_analyst", "limit": 4}))
        pe = _payload(await client.call_tool(
            "screen_sector",
            {"sector": "tech", "persona": "pe_analyst", "limit": 4}))
    assert [r["ticker"] for r in mf["results"]] != [r["ticker"] for r in pe["results"]]


async def test_out_of_scope_company_over_the_protocol(mcp_app):
    async with Client(mcp_app) as client:
        data = _payload(await client.call_tool("find_company", {"query": "Ferrari"}))
    assert data["in_database"] is False


async def test_database_is_opened_read_only(fixture_db):
    """The tool surface must be incapable of mutating the database, not merely
    disinclined to."""
    conn = connect(fixture_db, read_only=True)
    with pytest.raises(Exception) as excinfo:
        conn.execute("DELETE FROM companies")
    assert "readonly" in str(excinfo.value).lower()
    conn.close()


async def test_enumerable_arguments_are_constrained_in_the_schema(mcp_app):
    """A live run had the model call screen_sector with persona="private
    equity", which cost a wasted round trip. Enumerating the valid values in
    the schema makes that call unrepresentable rather than merely discouraged."""
    async with Client(mcp_app) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}

    screen = tools["screen_sector"].input_schema["properties"]
    assert set(screen["persona"]["enum"]) == {
        "mutual_fund_analyst", "equity_analyst", "pe_analyst"}
    assert set(screen["sector"]["enum"]) == {
        "tech", "retail", "manufacturing", "logistics"}


async def test_a_tool_error_payload_is_recorded_as_a_failed_call(mcp_app):
    """The response's tool log is the one part a model cannot overstate, so a
    tool that reports a bad argument must not be logged as a success.

    MCP's own `is_error` does not cover this: the tool returns normally, with a
    payload describing the failure.
    """
    from sectorlens.agent.mcp_client import open_toolbox

    async with open_toolbox(mcp_app) as toolbox:
        payload = await toolbox.call_json(
            "screen_sector",
            {"sector": "logistics", "persona": "mutual_fund_analyst"})
        assert "error" not in payload
        assert toolbox.calls[-1].ok is True

        # A tool that validates its own arguments and answers with an error
        # payload -- a normal protocol response, so MCP's is_error is unset.
        text, ok = await toolbox.call("compare_companies", {"tickers": []})
        assert '"error"' in text
        assert ok is False, "an error payload must be recorded as a failed call"
        assert toolbox.calls[-1].ok is False

        # Schema rejection is a different path and must also count as failed.
        _, ok = await toolbox.call("get_sector_benchmarks", {"sector": "biotech"})
        assert ok is False
        assert toolbox.calls[-1].ok is False


async def test_absent_records_are_not_treated_as_errors(mcp_app):
    """`in_database: false` is a correct answer, not a tool failure."""
    from sectorlens.agent.mcp_client import open_toolbox

    async with open_toolbox(mcp_app) as toolbox:
        payload = await toolbox.call_json("find_company", {"query": "Ferrari"})
        assert payload["in_database"] is False
        assert toolbox.calls[-1].ok is True


async def test_metric_history_refuses_to_call_one_point_a_trend(mcp_app):
    """The fixture holds a single undated period, so a direction claim would be
    unfounded -- the tool has to say that rather than let the model infer one."""
    from sectorlens.agent.mcp_client import open_toolbox

    async with open_toolbox(mcp_app) as toolbox:
        payload = await toolbox.call_json(
            "get_metric_history", {"ticker": "BIGC", "metric": "ebitda_margin"})

    assert payload["count"] == 1
    assert "change" not in payload, "one period cannot yield a direction"
    assert "not a trend" in payload["guidance"]


async def test_metric_history_rejects_an_unknown_metric(mcp_app):
    from sectorlens.agent.mcp_client import open_toolbox

    async with open_toolbox(mcp_app) as toolbox:
        payload = await toolbox.call_json(
            "get_metric_history", {"ticker": "BIGC", "metric": "vibes"})
        assert payload["error"] == "unknown_metric"
        assert toolbox.calls[-1].ok is False
