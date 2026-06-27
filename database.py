"""DuckDB schema creation and helper utilities."""

from __future__ import annotations

import threading
import time

import duckdb

from config import DB_PATH

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS stocks (
    code              VARCHAR PRIMARY KEY,
    type              VARCHAR,
    floor             VARCHAR,
    status            VARCHAR,
    company_name      VARCHAR,
    company_name_eng  VARCHAR,
    short_name        VARCHAR,
    listed_date       VARCHAR,
    delisted_date     VARCHAR,
    company_id        VARCHAR,
    tax_code          VARCHAR,
    isin              VARCHAR
);

CREATE TABLE IF NOT EXISTS companies (
    ticker VARCHAR PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS financial_models (
    model_type       VARCHAR,
    item_code        VARCHAR,
    model_type_name  VARCHAR,
    model_vn_desc    VARCHAR,
    model_en_desc    VARCHAR,
    company_form     VARCHAR,
    note             VARCHAR,
    code_list        VARCHAR,
    item_vn_name     VARCHAR,
    item_en_name     VARCHAR,
    display_order    INTEGER,
    display_level    INTEGER,
    form_type        VARCHAR
);

CREATE TABLE IF NOT EXISTS financial_statements (
    code           VARCHAR NOT NULL,
    item_code      VARCHAR,
    report_type    VARCHAR,
    model_type     VARCHAR,
    numeric_value  DOUBLE,
    fiscal_date    VARCHAR,
    created_date   VARCHAR,
    modified_date  VARCHAR
);

CREATE TABLE IF NOT EXISTS financial_ratios (
    code         VARCHAR NOT NULL,
    ratio_group  VARCHAR,
    report_date  VARCHAR,
    item_code    VARCHAR,
    ratio_code   VARCHAR,
    item_name    VARCHAR,
    value        DOUBLE
);

CREATE TABLE IF NOT EXISTS vietstock_documents (
    id              BIGINT PRIMARY KEY,
    ticker          VARCHAR NOT NULL,
    doc_type        VARCHAR NOT NULL,
    title           VARCHAR,
    full_name       VARCHAR,
    source          VARCHAR,
    published_date  VARCHAR,
    file_url        VARCHAR,
    file_info_id    BIGINT,
    synced_to_raw   BOOLEAN DEFAULT FALSE,
    raw_path        VARCHAR
);

CREATE TABLE IF NOT EXISTS annual_reports (
    ticker                         VARCHAR NOT NULL,
    year                           INTEGER NOT NULL,
    content                        VARCHAR NOT NULL,
    source_file                    VARCHAR NOT NULL,
    quality_checked_at             TIMESTAMP,
    quality_suspicious             BOOLEAN DEFAULT FALSE,
    suspicious_score               DOUBLE,
    single_char_token_ratio        DOUBLE,
    broken_spacing_pattern_count   INTEGER,
    average_token_length           DOUBLE,
    isolated_diacritic_token_count INTEGER,
    garbled_vietnamese_token_count INTEGER,
    garbled_vietnamese_token_ratio DOUBLE,
    affected_line_count            INTEGER,
    affected_line_ratio            DOUBLE,
    affected_region_count          INTEGER,
    quality_status                 VARCHAR,
    quality_reason                 VARCHAR,
    quality_evidence               VARCHAR,
    created_at                     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, year)
);

CREATE TABLE IF NOT EXISTS pipeline_files (
    ticker     VARCHAR NOT NULL,
    year       INTEGER NOT NULL,
    stage      VARCHAR NOT NULL,
    file_path  VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, year, stage)
);

CREATE SEQUENCE IF NOT EXISTS conversion_jobs_id_seq START 1;

CREATE TABLE IF NOT EXISTS conversion_jobs (
    id             INTEGER PRIMARY KEY DEFAULT nextval('conversion_jobs_id_seq'),
    ticker         VARCHAR NOT NULL,
    year           INTEGER NOT NULL,
    start_year     INTEGER,
    end_year       INTEGER,
    source_path    VARCHAR NOT NULL,
    output_dir     VARCHAR NOT NULL,
    status         VARCHAR NOT NULL DEFAULT 'pending',
    force_ocr      BOOLEAN NOT NULL DEFAULT FALSE,
    rerun_reason   VARCHAR,
    command        VARCHAR,
    log_path       VARCHAR,
    pid            INTEGER,
    error_message  VARCHAR,
    failed_step    VARCHAR,
    started_at     TIMESTAMP,
    completed_at   TIMESTAMP,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (ticker, year)
);

CREATE TABLE IF NOT EXISTS document_embeddings (
    ticker        VARCHAR NOT NULL,
    year          INTEGER NOT NULL,
    chunk_index   INTEGER NOT NULL,
    chunk_text    VARCHAR NOT NULL,
    token_count   INTEGER,
    embedding     FLOAT[],
    model         VARCHAR,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, year, chunk_index)
);

CREATE SEQUENCE IF NOT EXISTS inference_jobs_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS inference_results_id_seq START 1;

CREATE TABLE IF NOT EXISTS inference_jobs (
    id               INTEGER PRIMARY KEY DEFAULT nextval('inference_jobs_id_seq'),
    ticker           VARCHAR NOT NULL,
    year             INTEGER NOT NULL,
    status           VARCHAR NOT NULL DEFAULT 'pending',
    model            VARCHAR NOT NULL,
    top_k            INTEGER,
    categories_done  INTEGER DEFAULT 0,
    categories_total INTEGER DEFAULT 0,
    batch_id          VARCHAR,
    batch_submitted_at TIMESTAMP,
    batch_checked_at   TIMESTAMP,
    started_at       TIMESTAMP,
    completed_at     TIMESTAMP,
    error_message    VARCHAR,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (ticker, year, model)
);

CREATE TABLE IF NOT EXISTS inference_results (
    id            INTEGER PRIMARY KEY DEFAULT nextval('inference_results_id_seq'),
    ticker        VARCHAR NOT NULL,
    year          INTEGER NOT NULL,
    category_code VARCHAR NOT NULL,
    is_valid      BOOLEAN NOT NULL,
    reason        VARCHAR,
    top_chunks    VARCHAR,
    similarities  VARCHAR,
    model         VARCHAR NOT NULL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (ticker, year, category_code, model)
);

CREATE SEQUENCE IF NOT EXISTS proper_vn_jobs_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS proper_vn_results_id_seq START 1;

CREATE TABLE IF NOT EXISTS proper_vn_jobs (
    id               INTEGER PRIMARY KEY DEFAULT nextval('proper_vn_jobs_id_seq'),
    ticker           VARCHAR NOT NULL,
    year             INTEGER NOT NULL,
    status           VARCHAR NOT NULL DEFAULT 'pending',
    model            VARCHAR NOT NULL,
    top_k            INTEGER,
    indicators_done  INTEGER DEFAULT 0,
    indicators_total INTEGER DEFAULT 0,
    batch_id          VARCHAR,
    batch_submitted_at TIMESTAMP,
    batch_checked_at   TIMESTAMP,
    color            VARCHAR,
    s2_score         INTEGER,
    s2_max_score     INTEGER,
    started_at       TIMESTAMP,
    completed_at     TIMESTAMP,
    error_message    VARCHAR,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (ticker, year, model)
);

CREATE TABLE IF NOT EXISTS proper_vn_results (
    id             INTEGER PRIMARY KEY DEFAULT nextval('proper_vn_results_id_seq'),
    ticker         VARCHAR NOT NULL,
    year           INTEGER NOT NULL,
    indicator_code VARCHAR NOT NULL,
    is_present     BOOLEAN NOT NULL,
    evidence_level VARCHAR,
    reason         VARCHAR,
    top_chunks     VARCHAR,
    similarities   VARCHAR,
    model          VARCHAR NOT NULL,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (ticker, year, indicator_code, model)
);

CREATE SEQUENCE IF NOT EXISTS governance_jobs_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS governance_results_id_seq START 1;

CREATE TABLE IF NOT EXISTS governance_jobs (
    id            INTEGER PRIMARY KEY DEFAULT nextval('governance_jobs_id_seq'),
    ticker        VARCHAR NOT NULL,
    year          INTEGER NOT NULL,
    status        VARCHAR NOT NULL DEFAULT 'pending',
    model         VARCHAR NOT NULL,
    top_k         INTEGER,
    items_done    INTEGER DEFAULT 0,
    items_total   INTEGER DEFAULT 0,
    batch_id          VARCHAR,
    batch_submitted_at TIMESTAMP,
    batch_checked_at   TIMESTAMP,
    started_at    TIMESTAMP,
    completed_at  TIMESTAMP,
    error_message VARCHAR,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (ticker, year, model)
);

CREATE TABLE IF NOT EXISTS governance_results (
    id           INTEGER PRIMARY KEY DEFAULT nextval('governance_results_id_seq'),
    ticker       VARCHAR NOT NULL,
    year         INTEGER NOT NULL,
    item_code    VARCHAR NOT NULL,
    found        BOOLEAN NOT NULL,
    value_json   VARCHAR,
    details_json VARCHAR,
    reason       VARCHAR,
    top_chunks   VARCHAR,
    similarities VARCHAR,
    model        VARCHAR NOT NULL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (ticker, year, item_code, model)
);

CREATE TABLE IF NOT EXISTS reference_documents (
    doc_id        VARCHAR PRIMARY KEY,
    source_file   VARCHAR NOT NULL,
    title         VARCHAR,
    content       VARCHAR NOT NULL,
    content_hash  VARCHAR NOT NULL,
    updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reference_chunks (
    chunk_id      VARCHAR PRIMARY KEY,
    doc_id        VARCHAR NOT NULL,
    chunk_index   INTEGER NOT NULL,
    chunk_text    VARCHAR NOT NULL,
    token_count   INTEGER,
    FOREIGN KEY (doc_id) REFERENCES reference_documents(doc_id)
);

CREATE TABLE IF NOT EXISTS reference_embeddings (
    chunk_id      VARCHAR PRIMARY KEY,
    model         VARCHAR NOT NULL,
    dimensions    INTEGER NOT NULL,
    embedding     FLOAT[],
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chunk_id) REFERENCES reference_chunks(chunk_id)
);

CREATE TABLE IF NOT EXISTS reference_retrieval_logs (
    request_id        VARCHAR PRIMARY KEY,
    query             VARCHAR NOT NULL,
    framework         VARCHAR,
    indicator_code    VARCHAR,
    top_k             INTEGER,
    embedding_model   VARCHAR,
    llm_model         VARCHAR,
    response_json     VARCHAR,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS llm_model_presets (
    task_type    VARCHAR NOT NULL,
    model_name   VARCHAR NOT NULL,
    is_active    BOOLEAN NOT NULL DEFAULT TRUE,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (task_type, model_name)
);
"""

_MIGRATION_SQL = [
    "ALTER TABLE annual_reports ADD COLUMN IF NOT EXISTS quality_checked_at TIMESTAMP",
    "ALTER TABLE annual_reports ADD COLUMN IF NOT EXISTS quality_suspicious BOOLEAN",
    "ALTER TABLE annual_reports ADD COLUMN IF NOT EXISTS suspicious_score DOUBLE",
    "ALTER TABLE annual_reports ADD COLUMN IF NOT EXISTS single_char_token_ratio DOUBLE",
    "ALTER TABLE annual_reports ADD COLUMN IF NOT EXISTS broken_spacing_pattern_count INTEGER",
    "ALTER TABLE annual_reports ADD COLUMN IF NOT EXISTS average_token_length DOUBLE",
    "ALTER TABLE annual_reports ADD COLUMN IF NOT EXISTS isolated_diacritic_token_count INTEGER",
    "ALTER TABLE annual_reports ADD COLUMN IF NOT EXISTS garbled_vietnamese_token_count INTEGER",
    "ALTER TABLE annual_reports ADD COLUMN IF NOT EXISTS garbled_vietnamese_token_ratio DOUBLE",
    "ALTER TABLE annual_reports ADD COLUMN IF NOT EXISTS affected_line_count INTEGER",
    "ALTER TABLE annual_reports ADD COLUMN IF NOT EXISTS affected_line_ratio DOUBLE",
    "ALTER TABLE annual_reports ADD COLUMN IF NOT EXISTS affected_region_count INTEGER",
    "ALTER TABLE annual_reports ADD COLUMN IF NOT EXISTS quality_status VARCHAR",
    "ALTER TABLE annual_reports ADD COLUMN IF NOT EXISTS quality_reason VARCHAR",
    "ALTER TABLE annual_reports ADD COLUMN IF NOT EXISTS quality_evidence VARCHAR",
    "ALTER TABLE conversion_jobs ADD COLUMN IF NOT EXISTS force_ocr BOOLEAN",
    "ALTER TABLE conversion_jobs ADD COLUMN IF NOT EXISTS rerun_reason VARCHAR",
    "UPDATE annual_reports SET quality_suspicious = FALSE WHERE quality_suspicious IS NULL",
    "UPDATE annual_reports SET quality_status = 'pass' WHERE quality_status IS NULL",
    "UPDATE conversion_jobs SET force_ocr = FALSE WHERE force_ocr IS NULL",
    "ALTER TABLE reference_documents ADD COLUMN IF NOT EXISTS title VARCHAR",
    "ALTER TABLE reference_documents ADD COLUMN IF NOT EXISTS content_hash VARCHAR",
    "ALTER TABLE reference_documents ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP",
    "ALTER TABLE reference_chunks ADD COLUMN IF NOT EXISTS token_count INTEGER",
    "ALTER TABLE reference_embeddings ADD COLUMN IF NOT EXISTS dimensions INTEGER",
    "ALTER TABLE reference_retrieval_logs ADD COLUMN IF NOT EXISTS embedding_model VARCHAR",
    "ALTER TABLE document_embeddings ADD COLUMN IF NOT EXISTS model VARCHAR",
    "ALTER TABLE inference_jobs ADD COLUMN IF NOT EXISTS top_k INTEGER",
    "ALTER TABLE inference_jobs ADD COLUMN IF NOT EXISTS categories_done INTEGER",
    "ALTER TABLE inference_jobs ADD COLUMN IF NOT EXISTS categories_total INTEGER",
    "ALTER TABLE inference_jobs ADD COLUMN IF NOT EXISTS batch_id VARCHAR",
    "ALTER TABLE inference_jobs ADD COLUMN IF NOT EXISTS batch_submitted_at TIMESTAMP",
    "ALTER TABLE inference_jobs ADD COLUMN IF NOT EXISTS batch_checked_at TIMESTAMP",
    "ALTER TABLE proper_vn_jobs ADD COLUMN IF NOT EXISTS top_k INTEGER",
    "ALTER TABLE proper_vn_jobs ADD COLUMN IF NOT EXISTS indicators_done INTEGER",
    "ALTER TABLE proper_vn_jobs ADD COLUMN IF NOT EXISTS indicators_total INTEGER",
    "ALTER TABLE proper_vn_jobs ADD COLUMN IF NOT EXISTS batch_id VARCHAR",
    "ALTER TABLE proper_vn_jobs ADD COLUMN IF NOT EXISTS batch_submitted_at TIMESTAMP",
    "ALTER TABLE proper_vn_jobs ADD COLUMN IF NOT EXISTS batch_checked_at TIMESTAMP",
    "ALTER TABLE proper_vn_jobs ADD COLUMN IF NOT EXISTS color VARCHAR",
    "ALTER TABLE proper_vn_jobs ADD COLUMN IF NOT EXISTS s2_score INTEGER",
    "ALTER TABLE proper_vn_jobs ADD COLUMN IF NOT EXISTS s2_max_score INTEGER",
    "ALTER TABLE governance_jobs ADD COLUMN IF NOT EXISTS top_k INTEGER",
    "ALTER TABLE governance_jobs ADD COLUMN IF NOT EXISTS items_done INTEGER",
    "ALTER TABLE governance_jobs ADD COLUMN IF NOT EXISTS items_total INTEGER",
    "ALTER TABLE governance_jobs ADD COLUMN IF NOT EXISTS batch_id VARCHAR",
    "ALTER TABLE governance_jobs ADD COLUMN IF NOT EXISTS batch_submitted_at TIMESTAMP",
    "ALTER TABLE governance_jobs ADD COLUMN IF NOT EXISTS batch_checked_at TIMESTAMP",
    "ALTER TABLE llm_model_presets ADD COLUMN IF NOT EXISTS is_active BOOLEAN",
    "UPDATE llm_model_presets SET is_active = TRUE WHERE is_active IS NULL",
]

_INIT_DB_LOCK = threading.Lock()
_INIT_DB_DONE = False


def get_connection() -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection to the project database."""
    return duckdb.connect(str(DB_PATH))


def init_db(con: duckdb.DuckDBPyConnection | None = None) -> None:
    """Create all tables if they don't exist."""
    global _INIT_DB_DONE

    own = con is None
    if own:
        con = get_connection()

    try:
        with _INIT_DB_LOCK:
            if _INIT_DB_DONE:
                return

            for stmt in _SCHEMA_SQL.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    con.execute(stmt)

            retries = 5
            for attempt in range(retries):
                try:
                    for stmt in _MIGRATION_SQL:
                        con.execute(stmt)
                    break
                except duckdb.TransactionException as exc:
                    # Another concurrent writer may still be applying ALTER statements.
                    if (
                        "catalog write-write conflict on alter" in str(exc).lower()
                        and attempt < retries - 1
                    ):
                        time.sleep(0.05 * (attempt + 1))
                        continue
                    raise

            _INIT_DB_DONE = True
    finally:
        if own:
            con.close()


def ensure_vss_loaded(con: duckdb.DuckDBPyConnection) -> None:
    """Load DuckDB VSS extension, installing it on first use if needed."""
    try:
        con.execute("LOAD vss;")
        return
    except Exception as exc:
        msg = str(exc).lower()
        if "vss" not in msg:
            raise

    try:
        con.execute("INSTALL vss;")
        con.execute("LOAD vss;")
    except Exception as exc:
        raise RuntimeError(
            "Failed to load DuckDB VSS extension. "
            "Please ensure network access is available for 'INSTALL vss'."
        ) from exc


def ensure_company(con: duckdb.DuckDBPyConnection, ticker: str) -> None:
    """Insert a company ticker if it doesn't already exist."""
    con.execute(
        "INSERT INTO companies (ticker) VALUES (?) ON CONFLICT DO NOTHING",
        [ticker.upper()],
    )


def delete_company(
    con: duckdb.DuckDBPyConnection, ticker: str
) -> dict[str, int]:
    """Delete a company and all its related data across all tables.

    Also removes raw PDF and markdown directories from disk.

    Returns:
        Dict mapping table name -> number of rows deleted.
    """
    import shutil
    from config import RAW_DIR
    from config_marker import MARKDOWN_DIR

    t = ticker.upper()
    deleted: dict[str, int] = {}

    for table, col in [
        ("conversion_jobs", "ticker"),
        ("annual_reports", "ticker"),
        ("pipeline_files", "ticker"),
        ("vietstock_documents", "ticker"),
        ("financial_ratios", "code"),
        ("financial_statements", "code"),
    ]:
        result = con.execute(
            f"DELETE FROM {table} WHERE {col} = ? RETURNING *", [t]
        ).fetchall()
        deleted[table] = len(result)

    con.execute("DELETE FROM companies WHERE ticker = ?", [t])
    deleted["companies"] = 1

    # Clean up files on disk
    files_removed = 0
    for d in [RAW_DIR / t, MARKDOWN_DIR / t]:
        if d.is_dir():
            files_removed += sum(1 for _ in d.rglob("*") if _.is_file())
            shutil.rmtree(d)
    if files_removed:
        deleted["files"] = files_removed

    return deleted
