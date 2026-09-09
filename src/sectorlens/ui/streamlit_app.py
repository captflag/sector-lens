"""Human-facing interface.

Presentation only. It imports the same `Agent` the REST API calls, so both
paths run identical retrieval and identical reasoning -- the UI just renders
the evidence, caveats and tool trace that the API returns as JSON.

Run:
    streamlit run src/sectorlens/ui/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run path/to/app.py` without installing the package first.
SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st  # noqa: E402

from sectorlens.agent.core import Agent, available_options  # noqa: E402
from sectorlens.agent.errors import classify  # noqa: E402
from sectorlens.agent.schemas import AgentRequest  # noqa: E402
from sectorlens.settings import get_settings  # noqa: E402

st.set_page_config(page_title="Sector Lens -- sector intelligence",
                   page_icon="*", layout="wide")

SAMPLE_QUESTIONS = [
    # Chosen to exercise different paths: a screen, a comparison, an anchored
    # comparative claim, and the two cases where the honest answer is "I don't
    # hold that".
    "Given what you can see, would you be adding exposure to this sector "
    "today or staying out?",
    "Screen this sector for a leveraged buyout. Which names could carry debt?",
    "Split these into names I could hold for five years and names to avoid.",
    "How wide is the margin dispersion here, and who sits at each end?",
    "Is this sector expensive right now, or is that just the headline names?",
    "Put UPS and FedEx side by side. Which is the better business?",
    "How many people does UPS employ, and has that number been moving?",
    "How does Ferrari stack up against the names you cover?",
]


def md(text: str) -> str:
    """Escape text before handing it to Streamlit's markdown renderer.

    Streamlit reads `$...$` as LaTeX, so a line carrying two dollar amounts --
    "median eps: $8 (IQR $3 to $12)" -- is silently swallowed into a maths
    block and the numbers vanish. Answers are full of currency, so every one
    of them has to be escaped on the way to the screen.

    Only the display layer needs this. The API returns the text unmodified,
    because a JSON consumer wants the real string.
    """
    return text.replace("$", r"\$")


@st.cache_resource
def get_agent() -> Agent:
    return Agent()


@st.cache_resource
def deterministic_agent() -> Agent:
    """A no-LLM agent, used when the provider is unreachable."""
    return Agent(settings=get_settings().model_copy(
        update={"llm_provider": "deterministic"}))


@st.cache_data
def get_options() -> dict:
    return available_options()


def main() -> None:
    settings = get_settings()
    options = get_options()
    personas = {p["label"]: p for p in options["personas"]}
    sectors = {s["label"]: s for s in options["sectors"]}

    st.title("Sector Lens")
    st.caption("One agent, three analyst personas, four sectors. All data "
               "reaches the agent through MCP tools.")

    with st.sidebar:
        st.header("Configuration")
        persona_label = st.selectbox("Persona", list(personas))
        sector_label = st.selectbox("Sector", list(sectors))
        persona, sector = personas[persona_label], sectors[sector_label]

        st.markdown("**Lens**")
        st.caption(persona["lens"])
        st.markdown("**Sector**")
        st.caption(sector["description"])

        st.divider()
        provider = settings.effective_provider()
        if provider == "anthropic":
            st.success(f"LLM: {settings.model}")
            st.caption("A key is configured. Whether the account can actually "
                       "serve a request is only known once one is made.")
        else:
            st.warning(
                "Running the deterministic provider (no API key found). "
                "Retrieval, persona weighting and the MCP boundary all work; "
                "the answer is composed from templates rather than written by "
                "a model. Set ANTHROPIC_API_KEY for full answers.")
        if not settings.resolved_db_path.exists():
            st.error("No database. Run: python -m sectorlens.ingest.build_db")

    st.subheader("Ask a question")
    sample = st.selectbox("Start from a sample question", ["(write my own)"]
                          + SAMPLE_QUESTIONS)
    default = "" if sample == "(write my own)" else sample
    query = st.text_area("Question", value=default, height=90,
                         placeholder="Ask about this sector...")

    if st.button("Ask", type="primary", disabled=not query.strip()):
        request = AgentRequest(query=query.strip(), persona=persona["id"],
                               sector=sector["id"])
        degraded = False
        with st.spinner(f"Thinking as a {persona['label']}..."):
            try:
                response = get_agent().ask_sync(request)
            except Exception as exc:  # noqa: BLE001
                failure = classify(exc)
                st.error(md(f"**{failure.title}**\n\n{failure.remedy}"))
                with st.expander("Full provider response"):
                    st.code(str(exc))
                if not failure.degradable:
                    return
                # The database and the MCP layer are unaffected, so answer from
                # them rather than leaving the user with only an error. The
                # response labels its own provider, so this is never passed off
                # as a model-written answer.
                st.info("Answering with the deterministic provider instead, so "
                        "you can still see the retrieval and the persona "
                        "weighting. The prose is templated, not reasoned.")
                degraded = True
                try:
                    response = deterministic_agent().ask_sync(request)
                except Exception as inner:  # noqa: BLE001
                    st.error(f"The deterministic fallback also failed: {inner}")
                    return

        st.markdown(f"### {response.persona_label} on {response.sector_label}")
        st.markdown(md(response.answer))

        cols = st.columns(4)
        cols[0].metric("Confidence", response.confidence)
        cols[1].metric("Companies cited", len(response.companies_referenced))
        cols[2].metric("Tool calls", len(response.tool_calls))
        cols[3].metric("Elapsed", f"{response.elapsed_ms / 1000:.1f}s")

        if response.out_of_scope:
            st.warning(md("Not in this database, so not analysed: "
                          + ", ".join(response.out_of_scope)))

        if response.evidence:
            st.markdown("#### Evidence")
            st.dataframe(
                [
                    {
                        "Company": e.company or "-",
                        "Metric": e.metric,
                        "Value": e.value,
                        "Unit": e.unit or "-",
                        "Context": e.context or "-",
                        "Derived": "yes" if e.is_derived else "no",
                    }
                    for e in response.evidence
                ],
                use_container_width=True, hide_index=True)

        if response.caveats:
            with st.expander(f"Data caveats ({len(response.caveats)})"):
                for c in response.caveats:
                    st.markdown(md(f"- {c}"))

        with st.expander(f"MCP tool trace ({len(response.tool_calls)} calls)"):
            st.caption("Every fact above arrived through one of these calls. "
                       "The agent holds no database connection of its own.")
            st.dataframe(
                [
                    {
                        "Tool": c.tool,
                        "Arguments": str(c.arguments),
                        "OK": c.ok,
                        "ms": c.latency_ms,
                        "Result": c.result_summary,
                    }
                    for c in response.tool_calls
                ],
                use_container_width=True, hide_index=True)

        with st.expander("Raw JSON (identical to POST /v1/ask)"):
            st.json(response.model_dump())

        if response.data_sources:
            st.caption("Sources: " + " | ".join(response.data_sources))


main()
