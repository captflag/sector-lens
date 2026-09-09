"""REST interface.

This layer does no analysis. It validates input, calls `Agent.ask`, and
serialises the same `AgentResponse` the Streamlit UI renders -- so the two
interfaces cannot drift apart in behaviour, only in presentation.

Run:
    uvicorn sectorlens.api.main:app --reload
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..agent.core import Agent, available_options
from ..agent.errors import classify
from ..agent.schemas import AgentRequest, AgentResponse
from ..config import load_personas, load_sectors
from ..settings import get_settings

app = FastAPI(
    title="Sector Lens -- sector intelligence agent",
    version="0.1.0",
    description=(
        "One persona-configurable agent over an MCP tool boundary. Any of "
        "three personas can be combined with any loaded sector; the persona "
        "changes which companies are retrieved, not just how the answer reads."
    ),
)

# Wide open by design: no auth, no mutating routes, and read-only data. A
# deployment serving anything sensitive would pin origins and add auth.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_agent = Agent()


@app.get("/health", summary="Liveness and configuration probe")
async def health() -> dict[str, Any]:
    settings = get_settings()
    db = settings.resolved_db_path
    return {
        "status": "ok" if db.exists() else "database_missing",
        "database": str(db),
        "database_present": db.exists(),
        "llm_provider_configured": settings.llm_provider,
        "llm_provider_effective": settings.effective_provider(),
        "model": settings.model,
        "personas": sorted(load_personas()),
        "sectors": sorted(load_sectors()),
    }


@app.get("/v1/options", summary="Valid persona and sector values")
async def options() -> dict[str, Any]:
    return available_options()


@app.post("/v1/ask", response_model=AgentResponse,
          summary="Ask the agent a question as a given persona and sector")
async def ask(request: AgentRequest) -> AgentResponse:
    """The programmatic entry point.

    Returns the answer plus the structure another system needs to act on it:
    which companies were referenced, the specific values behind the argument,
    the data caveats that qualify it, anything asked about but not held, a
    confidence label, and the full MCP tool-call trace.
    """
    if request.persona not in load_personas():
        raise HTTPException(
            status_code=422,
            detail={"error": "unknown_persona", "valid": sorted(load_personas())})
    if request.sector not in load_sectors():
        raise HTTPException(
            status_code=422,
            detail={"error": "unknown_sector", "valid": sorted(load_sectors())})
    if not get_settings().resolved_db_path.exists():
        raise HTTPException(
            status_code=503,
            detail={"error": "database_missing",
                    "detail": "Build it with: python -m sectorlens.ingest.build_db"})

    try:
        return await _agent.ask(request)
    except Exception as exc:  # noqa: BLE001
        # Provider failures are operational, not bugs in the request, and each
        # has a specific remedy. Return that rather than a raw exception repr,
        # and use 502 so a caller can tell "the upstream model failed" from
        # "this service is broken".
        failure = classify(exc)
        raise HTTPException(
            status_code=502,
            detail={**failure.as_dict(), "detail": str(exc)}) from exc
