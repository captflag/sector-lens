-- Sector Lens -- sector intelligence schema (SQLite)
--
-- Design notes (expanded in README):
--
--  * Facts live in a long/EAV table (`company_metrics`) rather than one wide
--    column-per-metric table. Two ingestion adapters with very different
--    coverage (SEC EDGAR exposes ~40 XBRL concepts; the public-dataset adapter
--    exposes 9) load into the same table with no schema migration, and the same
--    table naturally carries a time series once multiple periods are ingested.
--    The cost is clumsier ad-hoc SQL, which is paid off by the pivot views at
--    the bottom of this file.
--
--  * Every fact carries a `source_id`. Provenance is not a nice-to-have here:
--    the agent is required to say where a number came from and how stale it is,
--    so "which source, retrieved when" has to be queryable per value, not per
--    database.
--
--  * Derived values are stored, not recomputed at query time, but they are
--    flagged (`is_derived`) and carry the arithmetic that produced them
--    (`derivation`). The agent surfaces that flag so a reader can tell a
--    reported figure from an inferred one.

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- Provenance
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS sources (
    id            INTEGER PRIMARY KEY,
    key           TEXT NOT NULL UNIQUE,   -- stable slug, e.g. "sec_edgar_companyfacts"
    name          TEXT NOT NULL,
    url           TEXT,
    publisher     TEXT,
    license       TEXT,
    retrieved_at  TEXT NOT NULL,          -- ISO-8601 UTC
    adapter       TEXT NOT NULL,          -- ingest adapter that created this row
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS ingest_runs (
    id             INTEGER PRIMARY KEY,
    adapter        TEXT NOT NULL,
    started_at     TEXT NOT NULL,
    finished_at    TEXT,
    status         TEXT NOT NULL DEFAULT 'running',  -- running | ok | failed
    companies      INTEGER DEFAULT 0,
    metric_values  INTEGER DEFAULT 0,
    signals        INTEGER DEFAULT 0,
    error          TEXT
);

-- ---------------------------------------------------------------------------
-- Entities
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS companies (
    id                 INTEGER PRIMARY KEY,
    ticker             TEXT NOT NULL UNIQUE,
    name               TEXT NOT NULL,
    sector             TEXT NOT NULL,      -- canonical sector id from sectors.yaml
    gics_sector        TEXT,
    gics_sub_industry  TEXT,
    hq_location        TEXT,
    cik                TEXT,
    founded            TEXT,
    index_added_date   TEXT,
    source_id          INTEGER REFERENCES sources(id),
    updated_at         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_companies_sector ON companies(sector);
CREATE INDEX IF NOT EXISTS idx_companies_name   ON companies(name);

-- ---------------------------------------------------------------------------
-- Metric registry -- makes the EAV table self-describing
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS metrics (
    code             TEXT PRIMARY KEY,
    label            TEXT NOT NULL,
    unit             TEXT NOT NULL,        -- usd | ratio | percent | count | multiple
    description      TEXT,
    higher_is_better INTEGER               -- 1 / 0 / NULL where it depends on the lens
);

-- ---------------------------------------------------------------------------
-- Facts
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS company_metrics (
    id           INTEGER PRIMARY KEY,
    company_id   INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    metric_code  TEXT    NOT NULL REFERENCES metrics(code),
    value        REAL    NOT NULL,
    period_end   TEXT,                     -- ISO date the value describes
    fiscal_period TEXT,                    -- e.g. FY2017, TTM, snapshot
    is_derived   INTEGER NOT NULL DEFAULT 0,
    derivation   TEXT,                     -- arithmetic used, when is_derived = 1
    source_id    INTEGER REFERENCES sources(id),
    run_id       INTEGER REFERENCES ingest_runs(id),
    UNIQUE (company_id, metric_code, period_end, fiscal_period)
);

CREATE INDEX IF NOT EXISTS idx_cm_company ON company_metrics(company_id);
CREATE INDEX IF NOT EXISTS idx_cm_metric  ON company_metrics(metric_code);

-- Headcount / hiring and other non-financial signals. Split from
-- company_metrics because these are frequently textual, arrive at irregular
-- cadence, and they are the field a model is most tempted to invent:
-- the agent must be able to answer "I hold no headcount signal for X".
CREATE TABLE IF NOT EXISTS company_signals (
    id           INTEGER PRIMARY KEY,
    company_id   INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    signal_type  TEXT NOT NULL,            -- headcount | hiring | news | filing
    value_num    REAL,
    value_text   TEXT,
    as_of        TEXT,
    source_id    INTEGER REFERENCES sources(id),
    run_id       INTEGER REFERENCES ingest_runs(id),
    UNIQUE (company_id, signal_type, as_of)
);

CREATE INDEX IF NOT EXISTS idx_sig_company ON company_signals(company_id, signal_type);

-- Sector aggregates, computed at build time from the facts above. The mutual
-- fund persona is defined as benchmark-relative, so "the sector median" has to
-- be a first-class stored number the agent can cite, not something the LLM
-- estimates.
CREATE TABLE IF NOT EXISTS sector_benchmarks (
    id           INTEGER PRIMARY KEY,
    sector       TEXT NOT NULL,
    metric_code  TEXT NOT NULL REFERENCES metrics(code),
    median       REAL,
    p25          REAL,
    p75          REAL,
    mean         REAL,
    n            INTEGER NOT NULL,
    computed_at  TEXT NOT NULL,
    run_id       INTEGER REFERENCES ingest_runs(id),
    UNIQUE (sector, metric_code)
);

-- ---------------------------------------------------------------------------
-- Views
-- ---------------------------------------------------------------------------

-- Latest value per (company, metric). period_end NULLs sort last so that a
-- dated observation always beats an undated snapshot.
CREATE VIEW IF NOT EXISTS v_company_latest_metrics AS
SELECT cm.*
FROM company_metrics cm
JOIN (
    SELECT company_id, metric_code, MAX(COALESCE(period_end, '0000-00-00')) AS mx
    FROM company_metrics
    GROUP BY company_id, metric_code
) latest
  ON latest.company_id  = cm.company_id
 AND latest.metric_code = cm.metric_code
 AND COALESCE(cm.period_end, '0000-00-00') = latest.mx;

-- Flat, human-readable join used by most tool queries.
CREATE VIEW IF NOT EXISTS v_company_facts AS
SELECT
    c.id            AS company_id,
    c.ticker        AS ticker,
    c.name          AS name,
    c.sector        AS sector,
    c.gics_sub_industry AS gics_sub_industry,
    m.code          AS metric_code,
    m.label         AS metric_label,
    m.unit          AS unit,
    v.value         AS value,
    v.period_end    AS period_end,
    v.fiscal_period AS fiscal_period,
    v.is_derived    AS is_derived,
    v.derivation    AS derivation,
    s.key           AS source_key,
    s.name          AS source_name,
    s.url           AS source_url,
    s.retrieved_at  AS source_retrieved_at
FROM v_company_latest_metrics v
JOIN companies c ON c.id = v.company_id
JOIN metrics   m ON m.code = v.metric_code
LEFT JOIN sources s ON s.id = v.source_id;

-- ---------------------------------------------------------------------------
-- Data quality
-- ---------------------------------------------------------------------------
-- Known data-quality caveats live here rather than in the README, so the agent
-- can cite them at answer time and the API can return them as structured
-- caveats, instead of a reader having to take a paragraph on trust.
CREATE TABLE IF NOT EXISTS data_quality_findings (
    id          INTEGER PRIMARY KEY,
    scope       TEXT NOT NULL,        -- dataset | sector | company | metric
    ref         TEXT,                 -- ticker / sector id / metric code
    check_name  TEXT NOT NULL,
    severity    TEXT NOT NULL,        -- info | warn | error
    detail      TEXT NOT NULL,
    run_id      INTEGER REFERENCES ingest_runs(id),
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dq_scope ON data_quality_findings(scope, ref);

-- ---------------------------------------------------------------------------
-- Filing text
-- ---------------------------------------------------------------------------
-- Everything above this line is numeric, which caps what the agent can argue:
-- it can rank on a margin gap but cannot say what management attributed the
-- gap to. These tables hold narrative sections of annual filings so a claim
-- can cite what a company actually wrote.
--
-- The provenance discipline is the same as the numeric side. A retrieved
-- passage is useless as evidence unless the reader can see which company,
-- which filing, which section and which date it came from, so every chunk
-- resolves back to all four.

CREATE TABLE IF NOT EXISTS filings (
    id           INTEGER PRIMARY KEY,
    company_id   INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    accession    TEXT NOT NULL,          -- SEC accession number, the filing's id
    form         TEXT NOT NULL,          -- 10-K, 10-Q, ...
    filed_date   TEXT,
    period_end   TEXT,
    source_url   TEXT,
    source_id    INTEGER REFERENCES sources(id),
    run_id       INTEGER REFERENCES ingest_runs(id),
    retrieved_at TEXT NOT NULL,
    UNIQUE (company_id, accession)
);

CREATE INDEX IF NOT EXISTS idx_filings_company ON filings(company_id);

-- Chunks are the retrieval unit. `section` is the filing item they came from
-- (risk factors, MD&A), kept so a question about risk is not answered with a
-- passage from the business description.
CREATE TABLE IF NOT EXISTS filing_chunks (
    id          INTEGER PRIMARY KEY,
    filing_id   INTEGER NOT NULL REFERENCES filings(id) ON DELETE CASCADE,
    section     TEXT NOT NULL,
    ordinal     INTEGER NOT NULL,        -- position within the section
    text        TEXT NOT NULL,
    char_count  INTEGER NOT NULL,
    UNIQUE (filing_id, section, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_chunks_filing ON filing_chunks(filing_id);

-- Lexical index over the chunks.
--
-- FTS5/BM25 rather than embeddings, deliberately: it needs no API key, no
-- model download and no vector store, so the whole retrieval path stays
-- reproducible offline and deployable anywhere the rest of this runs. Filing
-- text is also unusually favourable to lexical search -- the useful queries
-- are full of specific terms (a segment name, "pricing pressure", a customer)
-- that dense retrieval tends to blur.
--
-- The honest limit: BM25 cannot match a paraphrase that shares no vocabulary
-- with the passage. Adding a dense index alongside and fusing the rankings is
-- the natural extension, and the schema does not stand in its way.
CREATE VIRTUAL TABLE IF NOT EXISTS filing_chunks_fts USING fts5(
    text,
    content='filing_chunks',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS filing_chunks_ai AFTER INSERT ON filing_chunks BEGIN
    INSERT INTO filing_chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS filing_chunks_ad AFTER DELETE ON filing_chunks BEGIN
    INSERT INTO filing_chunks_fts(filing_chunks_fts, rowid, text)
    VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS filing_chunks_au AFTER UPDATE ON filing_chunks BEGIN
    INSERT INTO filing_chunks_fts(filing_chunks_fts, rowid, text)
    VALUES ('delete', old.id, old.text);
    INSERT INTO filing_chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
