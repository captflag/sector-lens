"""Request and response contracts shared by every entry point.

The REST API, the Streamlit UI and the tests all speak these types, so the
"same agent behind both interfaces" requirement holds structurally rather than
by discipline: there is one `AgentResponse`, and the HTTP layer serialises it
rather than composing its own.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Confidence = Literal["high", "medium", "low"]


class Evidence(BaseModel):
    """One datum the answer leans on, traceable back to the database."""

    company: str | None = Field(
        default=None, description="Ticker the value belongs to, if company-level.")
    metric: str = Field(description="Metric code or signal name.")
    value: float | str | None = Field(default=None)
    unit: str | None = Field(default=None)
    context: str | None = Field(
        default=None,
        description="How the value should be read, e.g. its percentile in the "
                    "sector or how it compares to the sector median.")
    source: str | None = Field(default=None, description="Upstream source name.")
    as_of: str | None = Field(default=None)
    is_derived: bool = Field(
        default=False,
        description="True when computed by this system rather than reported.")


class ToolCallRecord(BaseModel):
    """One MCP round trip, recorded so a caller can audit the retrieval."""

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    latency_ms: int = 0
    result_summary: str = ""
    error: str | None = None


class AgentRequest(BaseModel):
    query: str = Field(min_length=1, description="The user's question.")
    persona: str = Field(description="mutual_fund_analyst | equity_analyst | pe_analyst")
    sector: str = Field(description="tech | retail | manufacturing | logistics")
    max_tool_rounds: int | None = Field(
        default=None, ge=1, le=12,
        description="Override the agentic tool-calling budget for this request.")


class AgentAnswer(BaseModel):
    """The part the model is responsible for producing.

    Kept separate from `AgentResponse` because everything else -- which tools
    ran, how long it took, which provider answered -- is observed by the
    harness, not asserted by the model. A model cannot claim it called a tool
    it did not call.
    """

    answer: str = Field(description="The analysis, written in the persona's voice.")
    companies_referenced: list[str] = Field(
        default_factory=list,
        description="Tickers actually discussed, all of which must exist in the "
                    "database.")
    evidence: list[Evidence] = Field(
        default_factory=list,
        description="The specific values the answer rests on.")
    caveats: list[str] = Field(
        default_factory=list,
        description="Data limitations that materially qualify the answer.")
    out_of_scope: list[str] = Field(
        default_factory=list,
        description="Companies or sectors the user raised that are absent from "
                    "the database and were therefore not analysed.")
    confidence: Confidence = Field(
        default="medium",
        description="high only when the answer rests on well-covered, "
                    "reported data; low when coverage is thin or the metrics "
                    "involved are flagged by data-quality findings.")


class AgentResponse(AgentAnswer):
    """The full, machine-consumable response returned by every interface."""

    persona: str
    persona_label: str
    sector: str
    sector_label: str
    provider: str
    model: str | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    elapsed_ms: int = 0
    data_sources: list[str] = Field(default_factory=list)

    @classmethod
    def from_answer(cls, answer: AgentAnswer, **meta: Any) -> "AgentResponse":
        return cls(**answer.model_dump(), **meta)


def _strictify(node: Any) -> Any:
    """Make a pydantic-generated schema acceptable as a strict output format.

    Structured output requires every object in the schema -- not just the root
    -- to declare `additionalProperties: false` and to list all its properties
    as `required`. Pydantic emits neither for nested models (`Evidence` came
    back with `required: ["metric"]` and no `additionalProperties`), which
    either trips a 400 or lets the model silently omit fields.

    Optional fields stay optional in Python: they are typed `X | None`, so a
    required-but-null value still validates on the way back in. The effect is
    that the model must decide about every field rather than quietly dropping
    the ones it has nothing to say about.
    """
    if isinstance(node, dict):
        out = {k: _strictify(v) for k, v in node.items()}
        if isinstance(out.get("properties"), dict):
            out["additionalProperties"] = False
            out["required"] = list(out["properties"])
        return out
    if isinstance(node, list):
        return [_strictify(v) for v in node]
    return node


#: JSON Schema handed to the model via `output_config.format`. Derived from the
#: pydantic model so the response contract cannot drift between what the model
#: is asked for and what the API returns.
def answer_json_schema() -> dict[str, Any]:
    return _strictify(AgentAnswer.model_json_schema())
