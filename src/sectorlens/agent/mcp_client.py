"""The agent's side of the MCP boundary.

The agent has no database access. It launches the MCP server as a child
process, speaks JSON-RPC over its stdio, discovers the tool surface with
`list_tools`, and invokes tools with `call_tool`. Tool schemas are translated
into the shape the Anthropic Messages API expects, which is the only place the
two protocols meet.

Tests can pass an in-process `MCPServer` instead of spawning a subprocess; the
same `Client` type and the same protocol handle both, so the tested path and
the production path differ only in transport.
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from mcp import Client, StdioServerParameters

from ..settings import get_settings
from .schemas import ToolCallRecord

#: Cap on the characters of one tool result fed back to the model. Full
#: profiles and screens are large; without a cap a couple of calls can crowd
#: out the conversation. Truncation is announced in the payload so the model
#: knows it is seeing a prefix and can narrow its next call.
MAX_TOOL_RESULT_CHARS = 20_000


def server_parameters() -> StdioServerParameters:
    """Command that launches the MCP server as a subprocess.

    The database path is passed through the environment rather than argv so the
    child resolves exactly the same absolute path the parent did.
    """
    settings = get_settings()
    env = dict(os.environ)
    env["SECTORLENS_DB_PATH"] = str(settings.resolved_db_path)
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "sectorlens.mcp_server.server"],
        env=env,
    )


def _result_text(result: Any) -> str:
    """Normalise an MCP CallToolResult into JSON text for the model."""
    structured = getattr(result, "structured_content", None)
    if structured:
        return json.dumps(structured, default=str)
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts) if parts else "{}"


def _payload_is_error(text: str) -> bool:
    """Whether a tool's payload reports a failure it handled itself.

    MCP's `is_error` covers a tool that raised. It does not cover a tool that
    returned normally while reporting a bad argument -- `screen_sector` with an
    unknown persona answers with `{"error": "unknown_persona", ...}` and a
    perfectly successful protocol response.

    Without this, such a call was recorded `ok: true` and the result went back
    to the model with no `is_error` marker: the audit trail claimed a retrieval
    that never happened, and the model had to infer the failure from prose. The
    response's tool log is supposed to be the one part of the output a model
    cannot overstate, so it has to count these.

    An absent record is not an error: `find_company` returning
    `in_database: false` is a correct answer to a valid question, and carries no
    `error` key.
    """
    stripped = text.lstrip()
    if not stripped.startswith("{"):
        return False
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and bool(payload.get("error"))


def to_anthropic_tools(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    """Translate MCP tool descriptors into Messages API tool definitions."""
    tools: list[dict[str, Any]] = []
    for t in mcp_tools:
        schema = t.input_schema or {"type": "object", "properties": {}}
        tools.append({
            "name": t.name,
            "description": t.description or "",
            "input_schema": schema,
        })
    return tools


class McpToolbox:
    """A live MCP session plus the bookkeeping the response needs."""

    def __init__(self, client: Client) -> None:
        self._client = client
        self._tools: list[Any] = []
        self.calls: list[ToolCallRecord] = []

    async def discover(self) -> list[dict[str, Any]]:
        self._tools = list((await self._client.list_tools()).tools)
        return to_anthropic_tools(self._tools)

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self._tools]

    async def call(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Invoke one tool. Returns (payload_text, ok).

        A failing tool is reported back to the model as an error result rather
        than raised: the model can then choose a different tool or tell the
        user, which is a better outcome than the whole request dying.
        """
        started = time.perf_counter()
        try:
            result = await self._client.call_tool(name, arguments or {})
            text = _result_text(result)
            truncated = len(text) > MAX_TOOL_RESULT_CHARS
            if truncated:
                text = (text[:MAX_TOOL_RESULT_CHARS]
                        + '\n... [truncated: ask for a smaller limit or a '
                          'narrower query to see the rest]')
            ok = not getattr(result, "is_error", False) and not _payload_is_error(text)
            self.calls.append(ToolCallRecord(
                tool=name, arguments=arguments or {}, ok=ok,
                latency_ms=int((time.perf_counter() - started) * 1000),
                result_summary=f"{len(text)} chars"
                               + (" (truncated)" if truncated else ""),
            ))
            return text, ok
        except Exception as exc:  # noqa: BLE001 - surfaced to the model, not swallowed
            message = f"{type(exc).__name__}: {exc}"
            self.calls.append(ToolCallRecord(
                tool=name, arguments=arguments or {}, ok=False,
                latency_ms=int((time.perf_counter() - started) * 1000),
                result_summary="error", error=message))
            return json.dumps({"error": "tool_call_failed", "detail": message}), False

    async def call_json(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a tool and parse its payload. Used by the deterministic provider."""
        text, _ = await self.call(name, arguments)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"error": "unparseable_tool_result", "raw": text[:500]}
        return parsed if isinstance(parsed, dict) else {"result": parsed}


@asynccontextmanager
async def open_toolbox(server: Any | None = None) -> AsyncIterator[McpToolbox]:
    """Open an MCP session.

    `server` is None in production (spawns the server process) and an in-process
    `MCPServer` in tests.
    """
    target = server if server is not None else server_parameters()
    async with Client(target) as client:
        toolbox = McpToolbox(client)
        await toolbox.discover()
        yield toolbox
