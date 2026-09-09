# Sector Lens

**A persona-configurable analyst agent that cannot make things up about data it
does not have.**

Three financial analyst personas share one agent, one database and one set of
tools. Ask them the same question about the same sector and they return
different companies — because the persona's weighting is applied inside the
retrieval tool, below the language model, rather than in a prompt telling it to
sound different.

```
Sector: Technology — 64 companies, identical rows for all three
Data:   built 2026-09-09 from a daily-refreshed upstream snapshot

MF Analyst              Equity Analyst          PE Analyst
--------------------    --------------------    --------------------
1. MU     0.79          1. FSLR   0.86          1. HPQ    0.86
2. MSFT   0.79          2. MU     0.85          2. ACN    0.86
3. NVDA   0.77          3. WDC    0.85          3. CTSH   0.82
4. ORCL   0.75          4. VRSN   0.84          4. SMCI   0.80
5. ADBE   0.72          5. FICO   0.83          5. HPE    0.79

Companies appearing in all three top-5 lists: none
```

Reproduce with `make demo`. **No API key required** — the divergence happens in
the tool, not the model.

*Your run will differ from this table: the upstream market data refreshes daily,
so a rebuild moves the ordering. The last line is the claim; the tickers are
today's instance of it.*

---

## The problem it addresses

An LLM pointed at a database will answer questions it has no data for. It will
quote a headcount it never retrieved, call a company "cheap" without measuring
anything, and discuss a business that isn't in the dataset at all — fluently,
and with no signal to the reader that anything went wrong.

This treats that as a systems problem rather than a prompting one:

- **Absence is a first-class tool result.** Asking about a company that isn't
  loaded returns `in_database: false` with explicit guidance, not a nearest
  guess. Asking for a headcount nobody stored returns an empty result that says
  so and names what would populate it.
- **Every stored fact carries its source**, and every derived value carries the
  arithmetic that produced it, so the agent can tell a reported figure from an
  inferred one — and says which it is using.
- **Data-quality caveats are database rows**, recomputed on every build, so the
  agent cites them at answer time instead of a README asking you to take a
  paragraph on trust.
- **The tool trace is observed by the harness, not asserted by the model.** A
  model cannot claim it called a tool it did not call.
- **All of it is scored.** `make eval` grades answers on grounding, retrieval
  discipline and persona reasoning, and a grounding failure zeroes the case.

---

## Quickstart

Python 3.11+.

```bash
make install                 # venv + dependencies
cp .env.example .env         # add ANTHROPIC_API_KEY for reasoned answers
make db                      # build the database (~15s, needs GitHub only)
make test                    # 112 tests
make demo                    # persona divergence, no key needed
make eval                    # score answers against the case set

make api                     # REST on :8000  (/docs for OpenAPI)
make ui                      # Streamlit on :8501
```

<details>
<summary><b>Windows (PowerShell)</b></summary>

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .

copy .env.example .env
python -m sectorlens.ingest.build_db --fresh
python -m pytest tests -q
python scripts\demo_personas.py --sector tech
uvicorn sectorlens.api.main:app --port 8000
streamlit run src\sectorlens\ui\streamlit_app.py
```
</details>

**It runs with no credentials.** Without a key the agent falls back to a
deterministic provider that still goes through the same MCP tools and the same
persona weighting, composing the answer from templates instead of writing it
with a model. Retrieval, ranking, evidence, caveats and the tool trace are all
real; only the prose is templated. Every response names which provider produced
it.

---

## Architecture

```
   Streamlit UI ──┐
                  ├──> Agent.ask ──> LLM provider ──> Claude
   REST API ──────┘        │         (or deterministic)
                           │
                           │  MCP — JSON-RPC over stdio
                           ▼
                   MCP server (9 tools)     ← separate process
                           │
                           ▼
                   SQLite, opened read-only
```

Both interfaces call the same `Agent.ask` and serialise the same
`AgentResponse`, so "one agent, two interfaces" is structural rather than a
convention someone has to maintain.

**The agent holds no database handle.** It launches the MCP server as a child
process, discovers the tool surface with `list_tools`, and acts only through
`call_tool`. The claim is checkable rather than asserted:

```console
$ grep -rl sqlite3 src/sectorlens/agent/
(no matches)
```

Swapping SQLite for a warehouse, or moving the server to another host over
streamable HTTP, touches no agent code. The connection is opened with SQLite's
`mode=ro` URI and every statement is parameterised, so no tool argument — even
one a model was talked into producing — can write to it.

I used a **manual agentic loop** rather than an agent framework. The tools are
discovered over MCP at runtime and every round trip is recorded for the audit
trail; a framework would have hidden both behind an abstraction I would then
have had to fight.

### The tools

| Tool | Purpose |
|---|---|
| `list_sectors` | What is in scope at all |
| `list_companies` | The universe being reasoned over |
| `find_company` | **Scope check** — returns `in_database: false` explicitly |
| `get_company_profile` | Everything held on one company, with provenance |
| `get_company_signals` | Headcount/hiring, or an explicit "nothing held" |
| `screen_sector` | **Persona-weighted ranking** — the primary evidence source |
| `get_sector_benchmarks` | Medians and quartiles, to anchor comparative claims |
| `compare_companies` | Side by side, naming any ticker not in the database |
| `describe_data_coverage` | Sources, licences, ingest runs, open quality findings |

`sector` and `persona` are enumerated in the tool schema rather than described
in prose, so an invalid value is unrepresentable instead of merely discouraged.

---

## How persona changes retrieval

A persona is a YAML block, not a prompt string. Each carries three things that
bite at different layers:

| Layer | Field | Effect |
|---|---|---|
| Retrieval | `screen_weights` | Which companies come back from the database |
| Payload | `priority_metrics` | Which fields attach, and in what order |
| Reasoning | `lens`, `answer_sections`, `guardrails` | How the evidence is argued |

The first layer is the one that matters. `screen_sector` percentile-ranks each
metric **within the sector**, applies the persona's weights and directions, and
renormalises over the metrics each company actually has — so a company is
scored on what it has rather than punished for gaps in the source data.
Percentile ranks rather than z-scores, because these distributions are heavily
skewed and a handful of mega-caps would dominate any mean-and-deviation scheme.

**Two weights are deliberately inverted**, and they are why the lists diverge
rather than merely reorder:

| Metric | Mutual Fund | Private Equity | Why they disagree |
|---|---|---|---|
| `market_cap` | high is better | **low is better** | A long-only fund needs liquidity and index-relevant position sizes. A buyout shop needs an enterprise it can actually finance. |
| `ebitda_margin_gap_to_sector` | not weighted | **high is better** | A margin below the sector median reads as a quality problem to a public-market lens, and as the operational headroom that funds the return to a sponsor. |

The same database row therefore ranks near the top for one persona and near the
bottom for another. That inversion is asserted directly in the tests.

---

## Data and provenance

Two ingestion adapters write into one schema.

**`public` (default)** — S&P 500 constituents and a market snapshot, both
published under ODC-PDDL-1.0. Needs nothing but GitHub, so the build works in a
locked-down environment and the committed sample database is reproducible by
anyone. Not authoritative: the snapshot is undated upstream and several metrics
are reconstructed rather than reported.

**`edgar`** — annual XBRL company facts straight from SEC EDGAR: revenue, net
income, gross and operating income, cash, debt, operating cash flow, capex, and
headcount where a filer tags it. Restatements supersede originals by filing
date, and each metric resolves through an ordered list of concept aliases,
because companies tag the same quantity under different US-GAAP concepts. SEC
requires a contact string and caps clients at 10 req/s; the adapter enforces 8.

Current database: 163 companies across four sectors — tech 73, manufacturing
54, retail 22, logistics 14.

### Known caveats

Recorded as rows in `data_quality_findings`, recomputed every build, surfaced by
`describe_data_coverage` and returned by the API as structured `caveats`:

1. **The market snapshot is undated.** The latest index-add date among covered
   companies is a firm *lower* bound. No upper bound is claimed — ticker renames
   make the obvious inference unsound.
2. **Revenue is reconstructed** as `market_cap / price_to_sales`, so margins
   derived from it inherit any inconsistency between those fields. Sound for
   ranking within a sector; not the company's reported margin.
3. **`ev_to_ebitda_proxy` excludes net debt** under the default adapter, so it
   understates the entry multiple for indebted businesses. Never presented as a
   true EV/EBITDA. The EDGAR adapter fixes this.
4. **No headcount under the default adapter**, so a headcount question gets an
   honest "no signal held". That is the intended behaviour, not a gap.
5. **Logistics is thin** (14 companies). Medians over a set that small are
   indicative, not significant — and the build says so.

---

## Evaluation

Ranking divergence is unit-tested. That proves the personas *retrieve*
differently; it does not prove an answer reasons like the role or stayed inside
the data. `make eval` scores both.

```
grounding    1.00  ##################
discipline   1.00  ##################
persona      0.64  ############......
OVERALL      0.88  ################..
```

- **grounding** — invented companies, evidence attributed to unknown tickers, an
  undeclared out-of-scope company, a headcount asserted with no stored signal.
  These are *critical*: any one zeroes the case, because fluent invention is
  worse than an unhelpful answer.
- **discipline** — did it retrieve, cite, qualify, and calibrate its confidence?
- **persona** — did the prose engage with the concepts the role is defined by?

The case score is the mean of the three **dimension** scores, not of the
individual checks: grounding and discipline carry many more checks, so a flat
per-check mean would let a perfect grounding score hide prose that never reasons
like the role.

Scores are comparable only within a provider. The persona figure above is the
deterministic provider losing points by construction — it composes from
templates and cannot argue an operational thesis. **That gap is what the
language model adds, expressed as a number.**

Two things the harness does to avoid flattering itself, both of which it
initially failed:

1. **It does not grade its own scaffolding.** The PE persona's section heading
   is literally "Deal shape and entry multiple", so printing the heading scored
   a match on the `entry_multiple` concept while saying nothing about one.
   Headings are stripped before concept matching.
2. **Its checks are tested against bad answers.** The rubric is fed invented
   companies, fabricated headcounts and collapsed cross-persona rankings to
   confirm they fail — and honest phrasing to confirm they do not false-positive.

The eval has earned its keep: it caught a router that matched "employees" but
not "how many people does X employ", which quietly answered a headcount question
with a sector screen.

---

## Schema decisions

**Facts are stored long (EAV), not one column per metric.** Two adapters with
very different coverage — EDGAR exposes ~40 XBRL concepts, the snapshot exposes
9 — load into the same table with no migration, and the same table carries a
time series once multiple periods are ingested. The cost is clumsier ad-hoc SQL,
paid off by two pivot views. A `metrics` registry keeps the EAV self-describing
and stops adapters inventing near-duplicate metric names.

**Every fact carries a `source_id`.** The agent has to state where a number came
from and how stale it is, so provenance must be queryable per value, not per
database.

**Derived values are stored and flagged**, each with the `derivation` string
that produced it, so a reader can tell a reported figure from an inferred one —
and so can the agent.

Two supporting tables exist because a persona needs them: `sector_benchmarks`
(the mutual fund lens is defined as benchmark-relative, so the median must be a
stored, citable number rather than something a model estimates from a list it
was shown) and `company_signals` (workforce data is textual, irregular, and the
field a model is most tempted to invent).

---

## The API

```bash
curl -s localhost:8000/v1/ask -H 'content-type: application/json' -d '{
  "query": "Which companies look like attractive buyout targets?",
  "persona": "pe_analyst",
  "sector": "logistics"
}' | jq
```

Returns the answer plus what a machine needs to act on it:
`companies_referenced`, `evidence` (each value flagged reported or derived),
`caveats`, `out_of_scope`, `confidence`, and the full `tool_calls` trace with
per-call latency.

Provider failures return `502` with a diagnosis and a remedy — no credit,
rejected key, model unavailable, rate limit — rather than a raw exception.

---

## Roadmap

1. **EDGAR as the primary source, with multi-year history.** Fixes three of the
   five caveats above and turns "who is improving and who is under pressure"
   from a cross-section into a trajectory.
2. **Retrieval over filing text.** Every answer today is numeric, so the PE lens
   argues from margin gaps because that is all it has. Management commentary
   from a 10-K would let it argue an operational thesis from evidence — under
   the same provenance discipline the numeric side already has.
3. **An LLM judge on top of the eval.** Persona scoring is keyword-based, so it
   measures whether the right concepts appear, not whether the reasoning is any
   good.
4. **Prompt caching.** The system prompt and tool list are stable per
   persona/sector pair — a natural cache breakpoint.

## Limitations

- Sector membership follows GICS mechanically. `manufacturing` excludes
  "Electronic Manufacturing Services"; `retail` excludes restaurants.
  Defensible, documented, and arguable.
- Scores are relative positions inside one sector and one database. A screen to
  focus attention, not a recommendation.
- Nothing here is investment advice.
