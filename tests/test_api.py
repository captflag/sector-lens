"""The REST interface and the shape of its structured response."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sectorlens.agent.core import Agent
from sectorlens.settings import Settings


@pytest.fixture
def client(monkeypatch, fixture_db):
    from sectorlens.api import main as api_main
    from sectorlens.mcp_server import server as server_module

    settings = Settings(db_path=fixture_db, llm_provider="deterministic")
    monkeypatch.setattr(server_module, "get_settings", lambda: settings)
    monkeypatch.setattr(api_main, "get_settings", lambda: settings)
    monkeypatch.setattr(api_main, "_agent",
                        Agent(settings=settings, mcp_server=server_module.mcp))
    return TestClient(api_main.app)


def test_health_reports_configuration(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["database_present"] is True
    assert set(body["personas"]) == {"mutual_fund_analyst", "equity_analyst",
                                     "pe_analyst"}


def test_options_drives_the_ui_selectors(client):
    body = client.get("/v1/options").json()
    assert len(body["personas"]) == 3
    assert len(body["sectors"]) == 4


def test_ask_returns_structure_a_machine_can_consume(client):
    """A machine consumer needs more than prose: the values behind the answer,
    what was retrieved, and how much to trust it."""
    response = client.post("/v1/ask", json={
        "query": "Walk me through the margin profile of these companies.",
        "persona": "equity_analyst",
        "sector": "logistics",
    })
    assert response.status_code == 200
    body = response.json()

    assert body["answer"]
    assert body["persona"] == "equity_analyst"
    assert body["sector"] == "logistics"
    assert body["confidence"] in {"high", "medium", "low"}
    assert isinstance(body["companies_referenced"], list)
    assert body["companies_referenced"]
    assert body["evidence"], "callers need the values behind the answer"
    assert body["tool_calls"], "callers need the retrieval trace"
    assert body["data_sources"]

    evidence = body["evidence"][0]
    assert set(evidence) >= {"metric", "value", "is_derived"}


def test_persona_changes_the_api_response_for_one_question(client):
    payload = {"query": "Where would you deploy capital here?", "sector": "tech"}
    mf = client.post("/v1/ask", json={**payload,
                                      "persona": "mutual_fund_analyst"}).json()
    pe = client.post("/v1/ask", json={**payload, "persona": "pe_analyst"}).json()
    assert mf["companies_referenced"] != pe["companies_referenced"]
    assert mf["answer"] != pe["answer"]


@pytest.mark.parametrize("payload,field", [
    ({"query": "x", "persona": "day_trader", "sector": "tech"}, "persona"),
    ({"query": "x", "persona": "pe_analyst", "sector": "biotech"}, "sector"),
])
def test_invalid_selectors_are_rejected_with_the_valid_set(client, payload, field):
    response = client.post("/v1/ask", json=payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == f"unknown_{field}"
    assert detail["valid"]


def test_empty_query_is_rejected_by_validation(client):
    assert client.post("/v1/ask", json={
        "query": "", "persona": "pe_analyst", "sector": "tech"}).status_code == 422


def test_provider_failure_returns_an_actionable_502(monkeypatch, client):
    """A provider failure is an upstream problem with a specific remedy, not a
    bug in the request -- the caller should be told which, and what to do."""
    from sectorlens.api import main as api_main

    async def boom(_request):
        raise RuntimeError(
            "BadRequestError: Error code: 400 - {'type': 'error', 'error': "
            "{'type': 'invalid_request_error', 'message': 'Your credit balance "
            "is too low to access the Anthropic API.'}}")

    monkeypatch.setattr(api_main._agent, "ask", boom)
    response = client.post("/v1/ask", json={
        "query": "anything", "persona": "pe_analyst", "sector": "tech"})

    assert response.status_code == 502, "502 distinguishes upstream from local"
    detail = response.json()["detail"]
    assert detail["error"] == "insufficient_credit"
    assert "Plans & Billing" in detail["remedy"]
    assert detail["retryable"] is True
    # The raw provider text is still available for debugging.
    assert "credit balance" in detail["detail"]
