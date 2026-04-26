"""DuckDB schema creation and helper utilities."""

from __future__ import annotations

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
    ticker      VARCHAR NOT NULL,
    year        INTEGER NOT NULL,
    content     VARCHAR NOT NULL,
    source_file VARCHAR NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
"""


def get_connection() -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection to the project database."""
    return duckdb.connect(str(DB_PATH))


def init_db(con: duckdb.DuckDBPyConnection | None = None) -> None:
    """Create all tables if they don't exist."""
    own = con is None
    if own:
        con = get_connection()
    for stmt in _SCHEMA_SQL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            con.execute(stmt)
    if own:
        con.close()


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
