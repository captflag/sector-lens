"""The agent, end to end, over an in-process MCP session."""

from __future__ import annotations

import pytest

from sectorlens.agent.core import Agent, available_options
from sectorlens.agent.schemas import AgentRequest
from sectorlens.settings import Settings


@pytest.fixture
def agent(monkeypatch, fixture_db):
    from sectorlens.mcp_server import server as server_module

    settings = Settings(db_path=fixture_db, llm_provider="deterministic")
    monkeypatch.setattr(server_module, "get_settings", lambda: settings)
    return Agent(settings=settings, mcp_server=server_module.mcp)


async def test_agent_answers_and_records_its_retrieval(agent):
    response = await agent.ask(AgentRequest(
        query="Which companies here look like attractive buyout targets?",
        persona="pe_analyst", sector="tech"))

    assert response.answer
    assert response.persona == "pe_analyst"
    assert response.provider == "deterministic"
    assert response.companies_referenced
    assert response.evidence
    # The tool trace is observed by the harness, so it cannot be fabricated.
    assert any(c.tool == "screen_sector" for c in response.tool_calls)
    assert all(c.ok for c in response.tool_calls)


async def test_same_question_different_persona_cites_different_companies(agent):
    question = "Where would you put money to work in this sector?"
    mf = await agent.ask(AgentRequest(query=question,
                                      persona="mutual_fund_analyst", sector="tech"))
    pe = await agent.ask(AgentRequest(query=question,
                                      persona="pe_analyst", sector="tech"))
    assert mf.companies_referenced != pe.companies_referenced


async def test_out_of_scope_company_is_declared_not_answered(agent):
    response = await agent.ask(AgentRequest(
        query="What do you think about Ferrari?",
        persona="equity_analyst", sector="tech"))
    assert "Ferrari" in response.out_of_scope


async def test_headcount_question_is_answered_from_signals(agent):
    response = await agent.ask(AgentRequest(
        query="What is the most recent headcount signal you have for FRGT?",
        persona="pe_analyst", sector="logistics"))
    assert any(c.tool == "get_company_signals" for c in response.tool_calls)
    assert "310,000" in response.answer


async def test_missing_headcount_is_admitted_not_invented(agent):
    response = await agent.ask(AgentRequest(
        query="What is the most recent headcount signal you have for HAUL?",
        persona="pe_analyst", sector="logistics"))
    assert "no headcount" in response.answer.lower()
    assert response.confidence == "low"


async def test_answer_carries_data_caveats(agent):
    response = await agent.ask(AgentRequest(
        query="Summarise this sector.", persona="equity_analyst", sector="tech"))
    assert response.caveats
    assert response.data_sources


def test_available_options_lists_everything_selectable():
    options = available_options()
    assert len(options["personas"]) == 3
    assert len(options["sectors"]) == 4
    assert all("lens" in p for p in options["personas"])


@pytest.mark.parametrize("query", [
    "What is the most recent headcount signal you have for FRGT?",
    "How many people does FRGT employ?",
    "Has FRGT been hiring or shrinking its workforce?",
    "What does FRGT's payroll look like?",
    "Any staff numbers for FRGT?",
])
async def test_workforce_questions_route_to_the_signals_tool(agent, query):
    """A workforce question is a signals question however it is phrased.

    An earlier router matched only the noun forms and missed "how many people
    does X employ", which quietly answered with a sector screen instead. The
    failure mode is bad: the user asks about headcount and gets a confident
    answer about something else.
    """
    response = await agent.ask(AgentRequest(
        query=query, persona="pe_analyst", sector="logistics"))
    assert any(c.tool == "get_company_signals" for c in response.tool_calls), (
        f"{query!r} did not reach the signals tool")
