"""Adapter: narrative sections of annual filings from SEC EDGAR.

Everything else in this project is numeric, which caps what the agent can
argue. It can rank a company on a margin gap; it cannot say what management
attributed the gap to. This loads the parts of a 10-K where they say.

Two sections are taken by default -- risk factors and management's discussion
-- because those carry the reasoning rather than the tables, and the tables are
already covered by the XBRL adapter.

Network note: `www.sec.gov` and `data.sec.gov` are unreachable from some
locked-down environments, so this adapter runs where you have normal outbound
access. Its parsing is exercised against recorded filing HTML in the tests; the
fetch itself is the only part that needs the network.

    python -m sectorlens.ingest.fetch_filings --limit-per-sector 5
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Iterable

import httpx

from ..db.store import upsert_source, utcnow
from ..settings import get_settings

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{document}"

#: Sections worth retrieving. The keys are our stable names; the patterns match
#: how the item headings actually appear in filings, where spacing, casing and
#: punctuation all vary between filers.
SECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "risk_factors": re.compile(
        r"item\s*1a\.?\s*[-–—:]?\s*risk\s+factors", re.IGNORECASE),
    "mdna": re.compile(
        r"item\s*7\.?\s*[-–—:]?\s*management.{0,3}s\s+discussion", re.IGNORECASE),
}

#: Where each section ends: the heading of the item that follows it.
SECTION_TERMINATORS: dict[str, re.Pattern[str]] = {
    "risk_factors": re.compile(
        r"item\s*1b\.?\s*[-–—:]?\s*unresolved|item\s*2\.?\s*[-–—:]?\s*propert",
        re.IGNORECASE),
    "mdna": re.compile(
        r"item\s*7a\.?\s*[-–—:]?\s*quantitative|item\s*8\.?\s*[-–—:]?\s*financial",
        re.IGNORECASE),
}

CHUNK_CHARS = 1400
CHUNK_OVERLAP = 180
#: Below this a "section" is a table-of-contents entry, not the section itself.
MIN_SECTION_CHARS = 2000


class _TextExtractor(HTMLParser):
    """Strip tags to readable text. Standard library only, by choice.

    Filing HTML is enormous and irregular -- nested tables, inline styles,
    XBRL markup. A full DOM parser buys little here because the target is
    running prose, and avoiding the dependency keeps the ingest installable
    anywhere the rest of this runs.
    """

    _SKIP = {"script", "style", "head", "title"}
    _BREAK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "table"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skipping = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in self._SKIP:
            self._skipping += 1
        elif tag in self._BREAK:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skipping:
            self._skipping -= 1
        elif tag in self._BREAK:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skipping and data.strip():
            self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        raw = raw.replace("\xa0", " ")
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n\s*\n\s*\n+", "\n\n", raw)
        return raw.strip()


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return parser.text()


def extract_section(text: str, section: str) -> str | None:
    """Pull one item's prose out of a filing.

    Item headings appear at least twice -- once in the table of contents and
    once at the section itself -- so every start is tried and the longest
    resulting span wins. A table-of-contents hit yields a span of a few hundred
    characters and loses; the real section yields thousands.
    """
    start_pattern = SECTION_PATTERNS.get(section)
    end_pattern = SECTION_TERMINATORS.get(section)
    if start_pattern is None:
        return None

    best = ""
    for match in start_pattern.finditer(text):
        body_start = match.end()
        tail = text[body_start:]
        end = end_pattern.search(tail) if end_pattern else None
        candidate = tail[: end.start()] if end else tail
        if len(candidate) > len(best):
            best = candidate

    best = best.strip()
    return best if len(best) >= MIN_SECTION_CHARS else None


def chunk(text: str, size: int = CHUNK_CHARS,
          overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split prose into overlapping chunks, preferring paragraph boundaries.

    The overlap exists so a sentence spanning a boundary is still findable from
    either side; breaking on a blank line where one is nearby keeps chunks from
    starting mid-argument.
    """
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text.rfind("\n\n", start + size // 2, end)
            if window == -1:
                window = text.rfind(". ", start + size // 2, end)
                window = window + 1 if window != -1 else -1
            if window != -1:
                end = window
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


@dataclass
class FilingRef:
    accession: str
    form: str
    filed_date: str
    period_end: str
    document: str


def recent_annual_filings(submissions: dict, limit: int) -> list[FilingRef]:
    """Pick the most recent 10-K filings out of a submissions payload.

    The payload stores its filing history column-wise, as parallel arrays, so
    the fields are zipped back into records here.
    """
    recent = (submissions.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    out: list[FilingRef] = []
    for i, form in enumerate(forms):
        if form != "10-K":
            continue

        def field(name: str) -> str:
            values = recent.get(name) or []
            return values[i] if i < len(values) else ""

        accession = field("accessionNumber").replace("-", "")
        document = field("primaryDocument")
        if not accession or not document:
            continue
        out.append(FilingRef(
            accession=accession, form=form,
            filed_date=field("filingDate"), period_end=field("reportDate"),
            document=document))
        if len(out) >= limit:
            break
    return out


def store_filing(conn: sqlite3.Connection, *, company_id: int, ref: FilingRef,
                 sections: dict[str, str], source_id: int, run_id: int,
                 url: str) -> int:
    """Persist one filing and its chunked sections. Returns chunks written."""
    conn.execute(
        """INSERT INTO filings (company_id, accession, form, filed_date,
                                period_end, source_url, source_id, run_id,
                                retrieved_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(company_id, accession) DO UPDATE SET
             source_url=excluded.source_url, retrieved_at=excluded.retrieved_at""",
        (company_id, ref.accession, ref.form, ref.filed_date, ref.period_end,
         url, source_id, run_id, utcnow()),
    )
    filing_id = int(conn.execute(
        "SELECT id FROM filings WHERE company_id = ? AND accession = ?",
        (company_id, ref.accession)).fetchone()[0])

    # Replacing rather than merging keeps a re-ingest idempotent; the triggers
    # keep the FTS index in step.
    conn.execute("DELETE FROM filing_chunks WHERE filing_id = ?", (filing_id,))

    written = 0
    for section, body in sections.items():
        for ordinal, piece in enumerate(chunk(body)):
            conn.execute(
                """INSERT INTO filing_chunks
                     (filing_id, section, ordinal, text, char_count)
                   VALUES (?, ?, ?, ?, ?)""",
                (filing_id, section, ordinal, piece, len(piece)))
            written += 1
    return written


def register_source(conn: sqlite3.Connection) -> int:
    return upsert_source(
        conn, key="sec_edgar_filing_text",
        name="SEC EDGAR annual filing text",
        url="https://www.sec.gov/Archives/edgar/",
        publisher="U.S. Securities and Exchange Commission",
        license="Public domain (US Government work)", adapter="filings",
        notes="Risk factors and management's discussion, extracted from the "
              "primary 10-K document and chunked for retrieval.")


def build_client(timeout: float = 45.0) -> httpx.Client:
    """A client carrying the contact string SEC fair-access rules require."""
    return httpx.Client(
        timeout=timeout, follow_redirects=True,
        headers={"User-Agent": get_settings().sec_user_agent,
                 "Accept-Encoding": "gzip, deflate"})


def sleep_for_rate_limit(requests_per_second: float = 6.0) -> None:
    """Stay under SEC's 10 req/s ceiling. Two requests per filing, so this is
    deliberately slower than the numeric adapter."""
    time.sleep(1.0 / requests_per_second)
