"""Filing-text ingestion and retrieval.

SEC is not reachable from CI, so the fetch is not exercised here. Everything
either side of it is: the HTML extraction, the item-heading heuristics, the
chunker, and BM25 retrieval with its provenance and its absence behaviour.
"""

from __future__ import annotations

import pytest

from sectorlens.db.queries import (_fts_query, describe_filing_coverage,
                                   search_filings)
from sectorlens.db.store import (connect, init_schema, start_run, upsert_company,
                                 upsert_source)
from sectorlens.ingest.filings import (FilingRef, chunk, extract_section,
                                       html_to_text, recent_annual_filings,
                                       store_filing)

RISK_BODY = (
    "Our results depend on freight demand, which is cyclical. "
    "A prolonged downturn in industrial production would reduce volumes. " * 40
)
MDNA_BODY = (
    "Operating margin improved to 15.2% from 12.8%, driven by pricing "
    "discipline and lower purchased transportation expense. " * 40
)

FILING_HTML = f"""
<html><head><style>.x{{color:red}}</style><title>10-K</title></head><body>
  <div>TABLE OF CONTENTS</div>
  <div>Item 1A. Risk Factors ....... 12</div>
  <div>Item 7. Management's Discussion and Analysis ....... 30</div>
  <div>Item 1A. Risk Factors</div><p>{RISK_BODY}</p>
  <div>Item 1B. Unresolved Staff Comments</div><p>None.</p>
  <div>Item 7. Management's Discussion and Analysis of Financial Condition</div>
  <p>{MDNA_BODY}</p>
  <div>Item 8. Financial Statements and Supplementary Data</div>
</body></html>
"""


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def test_script_and_style_never_reach_the_text():
    text = html_to_text(FILING_HTML)
    assert "color:red" not in text
    assert "freight demand" in text


def test_the_real_section_wins_over_its_table_of_contents_entry():
    """Item headings appear at least twice. The one-line contents entry must
    not be mistaken for the section itself."""
    text = html_to_text(FILING_HTML)
    body = extract_section(text, "risk_factors")
    assert body is not None
    assert len(body) > 2000, "picked the contents line instead of the section"
    assert "freight demand" in body
    assert "Unresolved Staff Comments" not in body, "ran past the terminator"


def test_sections_are_bounded_by_the_next_item():
    text = html_to_text(FILING_HTML)
    mdna = extract_section(text, "mdna")
    assert "pricing discipline" in mdna
    assert "Financial Statements" not in mdna


def test_a_filing_without_the_headings_yields_nothing():
    """Better to index nothing than to index the whole document as one section."""
    text = html_to_text("<html><body><p>" + "words " * 2000 + "</p></body></html>")
    assert extract_section(text, "risk_factors") is None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def test_chunks_overlap_so_a_boundary_sentence_stays_findable():
    pieces = chunk("A" * 5000, size=1000, overlap=200)
    assert len(pieces) > 1
    assert sum(len(p) for p in pieces) > 5000, "no overlap was applied"


def test_chunking_prefers_a_paragraph_boundary():
    text = ("alpha " * 150) + "\n\n" + ("beta " * 150)
    pieces = chunk(text, size=1000, overlap=50)
    assert pieces[0].endswith("alpha"), "split mid-paragraph"


def test_empty_text_produces_no_chunks():
    assert chunk("   \n  ") == []


# ---------------------------------------------------------------------------
# Filing selection
# ---------------------------------------------------------------------------

def test_only_annual_reports_are_selected():
    submissions = {"filings": {"recent": {
        "form": ["10-Q", "10-K", "8-K", "10-K"],
        "accessionNumber": ["0-1", "0-2", "0-3", "0-4"],
        "primaryDocument": ["q.htm", "k1.htm", "8k.htm", "k2.htm"],
        "filingDate": ["2025-05-01", "2025-02-01", "2024-11-01", "2024-02-01"],
        "reportDate": ["2025-03-31", "2024-12-31", "", "2023-12-31"],
    }}}
    refs = recent_annual_filings(submissions, limit=5)
    assert [r.document for r in refs] == ["k1.htm", "k2.htm"]
    assert refs[0].accession == "02", "dashes should be stripped for the URL"


def test_filings_without_a_primary_document_are_skipped():
    submissions = {"filings": {"recent": {
        "form": ["10-K"], "accessionNumber": ["0-1"],
        "primaryDocument": [""], "filingDate": ["2025-02-01"],
        "reportDate": ["2024-12-31"]}}}
    assert recent_annual_filings(submissions, limit=5) == []


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

@pytest.fixture
def corpus(tmp_path):
    conn = connect(tmp_path / "text.db")
    init_schema(conn)
    run_id = start_run(conn, "t")
    source_id = upsert_source(conn, key="t", name="Filing text", adapter="t")
    company_id = upsert_company(conn, ticker="FRGT", name="Freight Co",
                                sector="logistics", source_id=source_id)
    text = html_to_text(FILING_HTML)
    store_filing(
        conn, company_id=company_id,
        ref=FilingRef(accession="0001", form="10-K", filed_date="2025-02-14",
                      period_end="2024-12-31", document="k.htm"),
        sections={"risk_factors": extract_section(text, "risk_factors"),
                  "mdna": extract_section(text, "mdna")},
        source_id=source_id, run_id=run_id,
        url="https://www.sec.gov/Archives/edgar/data/1/0001/k.htm")
    conn.commit()
    yield conn
    conn.close()


def test_search_finds_a_passage_and_attributes_it(corpus):
    result = search_filings(corpus, "what drove the operating margin?")
    assert result["count"] > 0
    hit = result["results"][0]
    assert "pricing discipline" in hit["text"]
    # Provenance: a passage is only usable as evidence if it can be attributed.
    assert hit["ticker"] == "FRGT"
    assert hit["filed_date"] == "2025-02-14"
    assert hit["period_end"] == "2024-12-31"
    assert hit["section"] == "mdna"
    assert hit["source_url"].startswith("https://www.sec.gov/")


def test_search_can_be_restricted_to_a_section(corpus):
    result = search_filings(corpus, "freight demand", section="risk_factors")
    assert result["count"] > 0
    assert {r["section"] for r in result["results"]} == {"risk_factors"}


def test_a_question_with_no_match_says_so(corpus):
    result = search_filings(corpus, "cryptocurrency mining hardware")
    assert result["count"] == 0
    assert "rather than answering from general knowledge" in result["guidance"]


def test_an_empty_corpus_names_the_command_that_fills_it(tmp_path):
    conn = connect(tmp_path / "empty.db")
    init_schema(conn)
    result = search_filings(conn, "anything at all")
    assert result["count"] == 0
    assert "No filing text is loaded" in result["guidance"]
    assert "fetch_filings" in result["guidance"]
    conn.close()


def test_results_carry_a_note_against_passing_quotes_off_as_analysis(corpus):
    result = search_filings(corpus, "operating margin")
    assert "wrote about itself" in result["note"]


def test_a_database_predating_the_text_tables_does_not_crash(tmp_path):
    """An older database is still valid; it simply holds no filings."""
    conn = connect(tmp_path / "old.db")
    conn.execute("CREATE TABLE companies (id INTEGER PRIMARY KEY)")
    conn.commit()
    assert search_filings(conn, "anything")["count"] == 0
    assert describe_filing_coverage(conn)["chunks"] == 0
    conn.close()


@pytest.mark.parametrize("query", [
    "O'Reilly's supply-chain risk",     # apostrophe and hyphen
    "margins AND pricing OR costs",     # FTS operators as plain words
    'a "quoted" phrase',                # embedded quotes
    "NEAR(margin pricing)",             # a function call
])
def test_hostile_queries_do_not_break_the_search(corpus, query):
    """A question is not FTS5 syntax, and must never be treated as such."""
    result = search_filings(corpus, query)
    assert "error" not in result


def test_coverage_reports_what_is_indexed(corpus):
    coverage = describe_filing_coverage(corpus)
    assert coverage["companies_with_text"] == 1
    assert coverage["filings"] == 1
    assert coverage["chunks"] > 0
    assert {s["section"] for s in coverage["sections"]} == {"risk_factors", "mdna"}
