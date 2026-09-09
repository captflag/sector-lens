"""The Anthropic agentic loop, driven against a scripted Messages API.

These are the tests that would otherwise not exist: with no API key available,
this loop is the one part of the system that cannot be exercised end to end.
The fake supplies the responses; the MCP tools underneath are real, so a
scripted tool call really does query the fixture database.
"""

from __future__ import annotations

import json

import pytest

from sectorlens.agent.core import Agent
from sectorlens.agent.llm import AnthropicProvider
from sectorlens.agent.schemas import AgentRequest
from sectorlens.agent.mcp_client import open_toolbox
from sectorlens.config import get_persona, get_sector
from sectorlens.settings import Settings
from tests.fake_anthropic import (FakeAnthropic, FakeMessage, StopDetails,
                                  TextBlock, ToolUseBlock, answer_message,
                                  tool_message)


@pytest.fixture
def mcp_app(monkeypatch, fixture_db):
    from sectorlens.mcp_server import server as server_module

    monkeypatch.setattr(server_module, "get_settings",
                        lambda: Settings(db_path=fixture_db))
    return server_module.mcp


@pytest.fixture
def agent_with_script(monkeypatch, mcp_app, fixture_db):
    """An Agent whose provider is backed by the scripted fake API."""
    from sectorlens.agent import core as core_module

    def build(script):
        settings = Settings(db_path=fixture_db, llm_provider="anthropic")
        provider = AnthropicProvider(settings, client=FakeAnthropic(script))
        monkeypatch.setattr(core_module, "build_provider", lambda _s: provider)
        return Agent(settings=settings, mcp_server=mcp_app)

    return build


async def run_loop(mcp_app, script, *, max_rounds: int = 4,
                   query: str = "Which names look attractive here?",
                   persona: str = "pe_analyst", sector: str = "tech"):
    """Drive one full `answer()` call and hand back the result plus the fake."""
    client = FakeAnthropic(script)
    provider = AnthropicProvider(Settings(llm_provider="anthropic"), client=client)
    async with open_toolbox(mcp_app) as toolbox:
        answer = await provider.answer(
            query=query, persona=get_persona(persona), sector=get_sector(sector),
            toolbox=toolbox, max_rounds=max_rounds)
        return answer, client, list(toolbox.calls)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

async def test_answers_without_calling_tools(mcp_app):
    answer, client, tool_calls = await run_loop(mcp_app, [answer_message()])
    assert answer.answer == "Scripted analysis."
    assert answer.confidence == "medium"
    assert answer.evidence[0].metric == "ebitda_margin"
    assert tool_calls == []
    assert len(client.requests) == 1


async def test_tool_call_round_trips_to_the_real_database(mcp_app):
    answer, client, tool_calls = await run_loop(mcp_app, [
        tool_message("screen_sector",
                     {"sector": "tech", "persona": "pe_analyst", "limit": 3}),
        answer_message(),
    ])
    assert answer.answer == "Scripted analysis."
    assert [c.tool for c in tool_calls] == ["screen_sector"]
    assert tool_calls[0].ok

    # The second request must carry the assistant turn and a matching result.
    second = client.requests[1]
    assert second.roles == ["user", "assistant", "user"]
    result_block = second.messages[-1]["content"][0]
    assert result_block["type"] == "tool_result"
    assert result_block["tool_use_id"] == "toolu_1"
    payload = json.loads(result_block["content"])
    assert payload["persona"] == "pe_analyst"
    assert len(payload["results"]) == 3
    client.assert_roles_alternate()
    client.assert_tool_results_pair_with_tool_uses()


async def test_parallel_tool_uses_return_in_one_user_message(mcp_app):
    """Splitting parallel results across messages trains the model out of
    making parallel calls, so they must arrive together."""
    parallel = FakeMessage(
        content=[
            ToolUseBlock(name="list_sectors", input={}, id="toolu_a"),
            ToolUseBlock(name="get_sector_benchmarks",
                         input={"sector": "tech"}, id="toolu_b"),
        ],
        stop_reason="tool_use")
    answer, client, tool_calls = await run_loop(
        mcp_app, [parallel, answer_message()])

    assert [c.tool for c in tool_calls] == ["list_sectors", "get_sector_benchmarks"]
    results = client.requests[1].messages[-1]["content"]
    assert len(results) == 2
    assert {r["tool_use_id"] for r in results} == {"toolu_a", "toolu_b"}
    client.assert_tool_results_pair_with_tool_uses()


async def test_multiple_rounds_accumulate_history(mcp_app):
    answer, client, tool_calls = await run_loop(mcp_app, [
        tool_message("list_sectors", {}, block_id="t1"),
        tool_message("get_sector_benchmarks", {"sector": "tech"}, block_id="t2"),
        answer_message(),
    ])
    assert len(tool_calls) == 2
    assert client.requests[-1].roles == [
        "user", "assistant", "user", "assistant", "user"]
    client.assert_roles_alternate()


# ---------------------------------------------------------------------------
# Budget exhaustion -- the path that had a role-alternation bug
# ---------------------------------------------------------------------------

async def test_budget_exhaustion_forces_a_final_answer(mcp_app):
    """A model that keeps calling tools must still be made to answer, and the
    forced turn must not send two consecutive user messages."""
    answer, client, tool_calls = await run_loop(
        mcp_app,
        [tool_message("list_sectors", {}, block_id="t1"),
         tool_message("list_sectors", {}, block_id="t2"),
         answer_message()],
        max_rounds=2)

    # Two budgeted rounds of tools, then one forced final call.
    assert len(client.requests) == 3
    final = client.requests[-1]
    assert final.kwargs["tool_choice"] == {"type": "none"}, \
        "the forced call must disable tools or the model can loop again"
    client.assert_roles_alternate()

    # The nudge rides along with the last tool results, not as a new turn.
    last_content = final.messages[-1]["content"]
    assert final.messages[-1]["role"] == "user"
    assert any(b.get("type") == "text" and "tool budget" in b["text"]
               for b in last_content if isinstance(b, dict))
    assert any(b.get("type") == "tool_result"
               for b in last_content if isinstance(b, dict))
    assert answer.answer == "Scripted analysis."


async def test_budget_exhaustion_after_pause_turn_opens_a_new_user_turn(mcp_app):
    """After a pause_turn the last message is the assistant's, so the nudge is
    a fresh user message rather than an append."""
    paused = FakeMessage(content=[TextBlock(text="working")],
                         stop_reason="pause_turn")
    answer, client, _ = await run_loop(
        mcp_app, [paused, answer_message()], max_rounds=1)
    client.assert_roles_alternate()
    assert client.requests[-1].messages[-1]["role"] == "user"


# ---------------------------------------------------------------------------
# Degenerate model responses
# ---------------------------------------------------------------------------

async def test_refusal_is_surfaced_not_raised(mcp_app):
    refusal = FakeMessage(
        content=[], stop_reason="refusal",
        stop_details=StopDetails(explanation="declined for policy reasons"))
    answer, client, _ = await run_loop(mcp_app, [refusal])
    assert "declined for policy reasons" in answer.answer
    assert answer.confidence == "low"
    assert answer.caveats


async def test_pause_turn_is_resumed(mcp_app):
    paused = FakeMessage(content=[TextBlock(text="thinking")],
                         stop_reason="pause_turn")
    answer, client, _ = await run_loop(mcp_app, [paused, answer_message()])
    assert answer.answer == "Scripted analysis."
    assert len(client.requests) == 2
    assert client.requests[1].roles == ["user", "assistant"]


async def test_non_json_text_is_still_returned_to_the_user(mcp_app):
    """Structured output should guarantee JSON, but a parse failure must not
    lose an answer the user could still read."""
    prose = FakeMessage(content=[TextBlock(text="Just prose, no JSON here.")])
    answer, _, _ = await run_loop(mcp_app, [prose])
    assert answer.answer == "Just prose, no JSON here."
    assert answer.confidence == "low"
    assert any("structured schema" in c for c in answer.caveats)


async def test_empty_response_is_reported(mcp_app):
    answer, _, _ = await run_loop(mcp_app, [FakeMessage(content=[])])
    assert answer.confidence == "low"
    assert "no text" in answer.answer.lower()


async def test_failing_tool_is_reported_to_the_model_as_an_error(mcp_app):
    """A bad tool argument must come back marked as an error the model can
    recover from, not kill the request.

    `sector` is schema-constrained, so an invalid value is rejected before the
    query layer sees it. What matters either way is that the tool_result
    carries is_error and the call is logged as failed -- a live run showed the
    model passing persona="private equity" and the trace recording it as a
    success.
    """
    answer, client, tool_calls = await run_loop(mcp_app, [
        tool_message("screen_sector", {"sector": "nonexistent",
                                       "persona": "pe_analyst"}),
        answer_message(),
    ])
    result_block = client.requests[1].messages[-1]["content"][0]
    assert result_block["is_error"] is True
    assert tool_calls[0].ok is False
    assert answer.answer == "Scripted analysis."


async def test_a_handled_error_payload_also_marks_the_call_failed(mcp_app):
    """The subtler case: the tool returns normally and reports the problem in
    its payload, so MCP's own is_error is unset."""
    answer, client, tool_calls = await run_loop(mcp_app, [
        tool_message("compare_companies", {"tickers": []}),
        answer_message(),
    ])
    result_block = client.requests[1].messages[-1]["content"][0]
    payload = json.loads(result_block["content"])
    assert payload["error"] == "no_tickers_supplied"
    assert result_block["is_error"] is True
    assert tool_calls[0].ok is False


async def test_unknown_tool_name_does_not_crash_the_loop(mcp_app):
    answer, client, tool_calls = await run_loop(mcp_app, [
        tool_message("no_such_tool", {}),
        answer_message(),
    ])
    assert tool_calls[0].ok is False
    assert client.requests[1].messages[-1]["content"][0].get("is_error") is True
    assert answer.answer == "Scripted analysis."


# ---------------------------------------------------------------------------
# Optional-feature downgrades
# ---------------------------------------------------------------------------

async def test_beta_rejection_downgrades_and_stays_downgraded(mcp_app):
    rejection = TypeError("create() got an unexpected keyword argument 'fallbacks'")
    answer, client, _ = await run_loop(mcp_app, [
        rejection, tool_message("list_sectors", {}), answer_message()])

    assert answer.answer == "Scripted analysis."
    assert client.requests[0].used_beta is True
    # Every request after the rejection uses the stable endpoint.
    assert all(r.used_beta is False for r in client.requests[1:])


async def test_output_format_rejection_drops_the_schema_not_the_request(mcp_app):
    rejection = ValueError(
        "invalid_request_error: output_config.format json_schema is not "
        "supported for this model")
    answer, client, _ = await run_loop(mcp_app, [rejection, answer_message()])

    assert answer.answer == "Scripted analysis."
    assert "format" in client.requests[0].kwargs["output_config"]
    later = client.requests[-1].kwargs.get("output_config", {})
    assert "format" not in later, "the schema should be dropped, not the request"
    assert later.get("effort"), "effort must survive the downgrade"


async def test_auth_errors_reach_the_caller_intact(mcp_app, agent_with_script):
    """Only feature rejections downgrade -- everything else must propagate.

    Asserted through the Agent because that is where the MCP task group's
    ExceptionGroup wrapper is unwrapped; without that, callers see
    "unhandled errors in a TaskGroup" instead of the real cause.
    """
    agent = agent_with_script([
        RuntimeError("401 authentication_error: invalid x-api-key")])
    with pytest.raises(RuntimeError, match="invalid x-api-key"):
        await agent.ask(AgentRequest(query="hello", persona="pe_analyst",
                                     sector="tech"))


async def test_a_schema_shaped_network_error_is_not_mistaken_for_a_rejection(
        mcp_app, agent_with_script):
    """The downgrade heuristic needs a rejection signal, not merely a keyword
    that happens to appear in an unrelated failure."""
    agent = agent_with_script([
        ConnectionError("connection reset while sending schema payload")])
    with pytest.raises(ConnectionError):
        await agent.ask(AgentRequest(query="hello", persona="pe_analyst",
                                     sector="tech"))


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------

async def test_request_carries_model_tools_and_persona_system_prompt(mcp_app):
    _, client, _ = await run_loop(mcp_app, [answer_message()],
                                  persona="mutual_fund_analyst", sector="logistics")
    request = client.requests[0].kwargs
    assert request["model"] == "claude-opus-5"
    assert request["max_tokens"] > 0
    assert {t["name"] for t in request["tools"]} >= {"screen_sector", "find_company"}
    assert "Mutual Fund Analyst" in request["system"]
    assert "Logistics" in request["system"]
    # The evidence rules must reach the model on every request.
    assert "in_database=false" in request["system"]


async def test_structured_output_schema_is_strict(mcp_app):
    _, client, _ = await run_loop(mcp_app, [answer_message()])
    schema = client.requests[0].kwargs["output_config"]["format"]["schema"]
    assert schema["additionalProperties"] is False
    for name, definition in schema.get("$defs", {}).items():
        if "properties" in definition:
            assert definition["additionalProperties"] is False, name
            assert set(definition["required"]) == set(definition["properties"]), name
