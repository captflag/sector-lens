# Architecture

Two views: what the pieces are, and what happens when someone asks a question.

---

## 1. Components

The load-bearing detail is the dashed line. Everything above it is the agent;
everything below is data access, in a separate process, reachable only by
JSON-RPC. The agent has no database handle, no SQL and no schema knowledge —
`grep -rl sqlite3 src/sectorlens/agent/` returns nothing.

```mermaid
flowchart TD
    subgraph clients["Interfaces — presentation only"]
        UI["Streamlit UI<br/>ui/streamlit_app.py"]
        API["REST API<br/>api/main.py"]
    end

    AGENT["<b>Agent.ask</b><br/>agent/core.py<br/><i>resolves persona + sector,<br/>owns the response contract</i>"]

    subgraph providers["Answer generation"]
        ANTHROPIC["AnthropicProvider<br/><i>manual agentic loop</i>"]
        DETERMINISTIC["DeterministicProvider<br/><i>no model — templates</i>"]
        CLAUDE(["Claude<br/>Messages API"])
    end

    TOOLBOX["McpToolbox<br/>agent/mcp_client.py<br/><i>discovers tools, records every call</i>"]
    MCP["<b>MCP server</b> — 11 tools<br/>mcp_server/server.py"]
    DB[("SQLite<br/><i>opened mode=ro</i>")]

    subgraph build["Build time — offline, separate lifecycle"]
        PUBLIC["public datasets<br/><i>index + market snapshot</i>"]
        EDGAR["SEC EDGAR XBRL<br/><i>multi-year fundamentals</i>"]
        FILINGS["SEC filing text<br/><i>risk factors, MD and A</i>"]
        QUALITY["quality checks<br/><i>caveats stored as rows</i>"]
    end

    UI --> AGENT
    API --> AGENT
    AGENT -- "same AgentResponse" --> UI
    AGENT -- "same AgentResponse" --> API

    AGENT -->|"reasons"| ANTHROPIC
    AGENT -->|"no key, or budget spent"| DETERMINISTIC
    ANTHROPIC <-->|"tool_use / tool_result"| CLAUDE

    ANTHROPIC --> TOOLBOX
    DETERMINISTIC --> TOOLBOX
    TOOLBOX -. "JSON-RPC over stdio<br/><b>process boundary</b>" .-> MCP
    MCP -->|"parameterised, read-only"| DB

    PUBLIC --> QUALITY
    EDGAR --> QUALITY
    FILINGS --> QUALITY
    QUALITY --> DB

    style MCP stroke-width:3px
```

**Why it is shaped this way**

| Decision | Consequence |
|---|---|
| Data access behind MCP, in its own process | Swapping SQLite for a warehouse, or moving the server to another host, touches no agent code |
| Both interfaces call one `Agent.ask` | The UI and API cannot drift apart in behaviour, only in presentation |
| Two providers behind one interface | The system runs, and demos, with no API key at all |
| Read-only connection, parameterised statements | No tool argument — including one a model was talked into producing — can write |
| Ingest is a separate lifecycle | The database is a build artifact; the agent never writes |

---

## 2. Request workflow

One question, end to end. The tool loop is the interesting part: the model
never touches data, it asks for it, and every request is recorded by the
harness rather than reported by the model.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant IF as UI / REST
    participant Agent as Agent.ask
    participant Tools as McpToolbox
    participant MCP as MCP server
    participant DB as SQLite read-only
    participant LLM as Claude

    User->>IF: question + persona + sector
    IF->>Agent: AgentRequest

    Agent->>Tools: open session
    Tools->>MCP: spawn subprocess, list_tools
    MCP-->>Tools: 11 tool schemas
    Note over Tools,MCP: sector and persona are enumerated in<br/>the schema — an invalid value is unrepresentable

    Agent->>LLM: system prompt (persona lens + evidence rules) + tools

    loop until no tool_use, or budget spent
        LLM-->>Agent: tool_use blocks
        Agent->>Tools: call_tool
        Tools->>MCP: JSON-RPC request
        MCP->>DB: parameterised query
        DB-->>MCP: rows
        MCP-->>Tools: result, or an explicit "not held"
        Tools-->>Agent: payload + ok/failed, latency
        Agent->>LLM: tool_result (is_error when the tool reported one)
    end

    Note over Agent,LLM: budget exhausted → one final call with<br/>tool_choice: none, so the turn cannot be spent<br/>on another tool call

    LLM-->>Agent: structured JSON (schema-constrained)
    Agent-->>IF: AgentResponse
    IF-->>User: answer + evidence + caveats + tool trace
```

**What the response carries, and who vouches for it**

| Field | Source | Can the model fabricate it? |
|---|---|---|
| `answer`, `evidence`, `caveats` | the model, constrained by a JSON schema | it is bounded by what the tools returned |
| `companies_referenced`, `out_of_scope` | the model | checked against the database by the eval |
| `tool_calls`, latency, `ok` | **the harness** | **no** — observed, not asserted |
| `provider`, `model`, `elapsed_ms` | **the harness** | **no** |

### Degradation paths

None of these fail the request; each is visible in the response.

```mermaid
flowchart LR
    REQ["request"] --> KEY{"API key<br/>configured?"}
    KEY -->|no| DET["deterministic provider<br/><i>retrieval + weighting intact</i>"]
    KEY -->|yes| BUDGET{"daily model<br/>budget left?"}
    BUDGET -->|no| DET
    BUDGET -->|yes| CALL["call Claude"]
    CALL --> ERR{"provider<br/>error?"}
    ERR -->|"no credit, bad key,<br/>rate limit, overload"| DIAG["diagnosis + remedy<br/>(HTTP 502 on the API)"]
    DIAG --> DET
    ERR -->|no| OK["model answer"]

    OK --> RESP["AgentResponse<br/><i>names which provider answered</i>"]
    DET --> RESP
```

---

## 3. Where persona bites

A persona is a YAML block, not a prompt string, and the layer that matters is
the first one — it changes which rows come back, before any model sees them.

```mermaid
flowchart TD
    P["personas.yaml"]

    P -->|"screen_weights"| R["<b>Retrieval</b><br/>screen_sector ranks the sector<br/>by this persona's weights"]
    P -->|"priority_metrics"| F["<b>Payload</b><br/>which fields attach<br/>to each company"]
    P -->|"lens, answer_sections,<br/>guardrails"| A["<b>Reasoning</b><br/>how the evidence<br/>is argued"]

    R --> EV["evidence set<br/><i>different per persona</i>"]
    F --> EV
    EV --> A
    A --> OUT["answer"]

    style R stroke-width:3px
```

Two weights are deliberately inverted, which is why the rankings diverge rather
than merely reorder:

| Metric | Mutual Fund | Private Equity | Why |
|---|---|---|---|
| `market_cap` | high is better | **low is better** | A long-only fund needs liquidity and index-relevant size; a sponsor needs something financeable |
| `ebitda_margin_gap_to_sector` | not weighted | **high is better** | A below-median margin is a quality problem to one lens and operational headroom to the other |
