"""The agent.

One implementation, reached two ways. The REST API and the Streamlit UI both
call `Agent.ask`; neither adds analysis of its own, so "same agent behind both
interfaces" is a structural fact rather than a convention someone has to keep.

Everything the agent knows about the data arrives through MCP. It holds no
database handle and no SQL.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from ..config import get_persona, get_sector, load_personas, load_sectors
from ..settings import Settings, get_settings
from .llm import build_provider
from .mcp_client import open_toolbox
from .schemas import AgentRequest, AgentResponse


class Agent:
    def __init__(self, settings: Settings | None = None,
                 mcp_server: Any | None = None) -> None:
        """
        Args:
            settings: overrides the process settings; handy in tests.
            mcp_server: an in-process `MCPServer` to talk to instead of
                spawning the server as a subprocess. Production leaves this
                None so the protocol crosses a real process boundary.
        """
        self.settings = settings or get_settings()
        self._mcp_server = mcp_server

    async def ask(self, request: AgentRequest) -> AgentResponse:
        started = time.perf_counter()
        persona = get_persona(request.persona)
        sector = get_sector(request.sector)
        provider = build_provider(self.settings)
        rounds = request.max_tool_rounds or self.settings.max_tool_rounds

        try:
            async with open_toolbox(self._mcp_server) as toolbox:
                answer = await provider.answer(
                    query=request.query, persona=persona, sector=sector,
                    toolbox=toolbox, max_rounds=rounds)
                sources = await self._collect_sources(toolbox)
                tool_calls = list(toolbox.calls)
        except BaseExceptionGroup as group:  # noqa: F821 - builtin on 3.11+
            # The MCP session runs under an anyio task group, which repackages
            # anything raised inside it as an ExceptionGroup. Left alone, a
            # plain auth failure reaches the API layer as "unhandled errors in
            # a TaskGroup (1 sub-exception)" and the real cause is invisible in
            # the HTTP response. Unwrap a lone leaf so callers see what
            # actually failed.
            raise _sole_cause(group) from None

        return AgentResponse.from_answer(
            answer,
            persona=persona.id,
            persona_label=persona.label,
            sector=sector.id,
            sector_label=sector.label,
            provider=provider.name,
            model=provider.model,
            tool_calls=tool_calls,
            data_sources=sources,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    def ask_sync(self, request: AgentRequest) -> AgentResponse:
        """Blocking entry point for Streamlit, which has no event loop."""
        return asyncio.run(self.ask(request))

    async def _collect_sources(self, toolbox: Any) -> list[str]:
        """Name the upstream sources behind this database, for attribution."""
        try:
            coverage = await toolbox.call_json("describe_data_coverage", {})
        except Exception:  # noqa: BLE001 - attribution must never fail a request
            return []
        # Drop the bookkeeping call so it does not look like retrieval the
        # model chose to do.
        if toolbox.calls and toolbox.calls[-1].tool == "describe_data_coverage":
            toolbox.calls.pop()
        return [f"{s['name']} ({s.get('license') or 'licence unstated'})"
                for s in coverage.get("sources", [])]


def _sole_cause(group: BaseException) -> BaseException:
    """Unwrap nested exception groups down to a single underlying error.

    Returns the group itself when it genuinely carries more than one failure,
    since collapsing those would hide information.
    """
    current = group
    while isinstance(current, BaseExceptionGroup) and len(current.exceptions) == 1:  # noqa: F821
        current = current.exceptions[0]
    return current


def available_options() -> dict[str, Any]:
    """Personas and sectors, for the UI selectors and the API discovery route."""
    return {
        "personas": [
            {"id": p.id, "label": p.label, "short_label": p.short_label,
             "lens": p.lens, "priority_metrics": list(p.priority_metrics)}
            for p in load_personas().values()
        ],
        "sectors": [
            {"id": s.id, "label": s.label, "description": s.description}
            for s in load_sectors().values()
        ],
    }
