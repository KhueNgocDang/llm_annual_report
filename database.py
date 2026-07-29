from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import duckdb

from config import DB_PATH


def _schema_sql() -> str:
    return """
CREATE TABLE IF NOT EXISTS companies (
    ticker VARCHAR PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stocks (
    code VARCHAR PRIMARY KEY,
    type VARCHAR,
    floor VARCHAR,
    status VARCHAR,
    company_name VARCHAR,
    company_name_eng VARCHAR,
    short_name VARCHAR,
    listed_date VARCHAR,
    delisted_date VARCHAR,
    company_id VARCHAR,
    tax_code VARCHAR,
    isin VARCHAR,
    synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS financial_models (
    model_type VARCHAR,
    item_code VARCHAR,
    model_type_name VARCHAR,
    model_vn_desc VARCHAR,
    model_en_desc VARCHAR,
    company_form VARCHAR,
    note VARCHAR,
    code_list VARCHAR,
    item_vn_name VARCHAR,
    item_en_name VARCHAR,
    display_order INTEGER,
    display_level INTEGER,
    form_type VARCHAR,
    synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS financial_statements (
    code VARCHAR NOT NULL,
    item_code VARCHAR,
    report_type VARCHAR,
    model_type VARCHAR,
    numeric_value DOUBLE,
    fiscal_date VARCHAR,
    created_date VARCHAR,
    modified_date VARCHAR,
    synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS financial_ratios (
    code VARCHAR NOT NULL,
    ratio_group VARCHAR,
    report_date VARCHAR,
    item_code VARCHAR,
    ratio_code VARCHAR,
    item_name VARCHAR,
    value DOUBLE,
    synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vietstock_documents (
    id BIGINT,
    ticker VARCHAR NOT NULL,
    doc_type VARCHAR NOT NULL,
    title VARCHAR,
    full_name VARCHAR,
    source VARCHAR,
    published_date VARCHAR,
    file_url VARCHAR,
    file_info_id BIGINT,
    synced_to_raw BOOLEAN DEFAULT FALSE,
    raw_path VARCHAR,
    selection_method VARCHAR,
    selection_reason VARCHAR,
    sync_error VARCHAR,
    last_downloaded_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS annual_reports (
    ticker VARCHAR NOT NULL,
    year INTEGER NOT NULL,
    content VARCHAR NOT NULL,
    source_file VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, year)
);

CREATE TABLE IF NOT EXISTS financial_statement_reports (
    ticker VARCHAR NOT NULL,
    year INTEGER NOT NULL,
    content VARCHAR NOT NULL,
    source_file VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, year)
);

CREATE TABLE IF NOT EXISTS pipeline_files (
    ticker VARCHAR NOT NULL,
    year INTEGER NOT NULL,
    stage VARCHAR NOT NULL,
    file_path VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, year, stage)
);

CREATE TABLE IF NOT EXISTS llm_request_inputs (
    id BIGINT PRIMARY KEY,
    task_type VARCHAR NOT NULL,
    ticker VARCHAR,
    year INTEGER,
    category_code VARCHAR,
    model VARCHAR,
    request_body VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def get_connection(db_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    target = db_path or DB_PATH
    return duckdb.connect(str(target))


def init_db(db_path: Path | None = None) -> Path:
    target = db_path or DB_PATH
    con = duckdb.connect(str(target))
    try:
        con.execute(_schema_sql())
        _apply_schema_backfills(con)
    finally:
        con.close()
    return target


def _apply_schema_backfills(con: duckdb.DuckDBPyConnection) -> None:
    """Add columns that may be missing in older local schemas."""
    con.execute(
        "ALTER TABLE vietstock_documents ADD COLUMN IF NOT EXISTS selection_method VARCHAR"
    )
    con.execute(
        "ALTER TABLE vietstock_documents ADD COLUMN IF NOT EXISTS selection_reason VARCHAR"
    )
    con.execute(
        "ALTER TABLE vietstock_documents ADD COLUMN IF NOT EXISTS sync_error VARCHAR"
    )
    con.execute(
        "ALTER TABLE vietstock_documents ADD COLUMN IF NOT EXISTS last_downloaded_at TIMESTAMP"
    )
    _backfill_financial_statement_reports(con)


def _table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    return bool(
        con.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = ?
            """,
            [table_name],
        ).fetchone()[0]
    )


def _backfill_financial_statement_reports(con: duckdb.DuckDBPyConnection) -> None:
    """Migrate legacy bctc reports into financial statement table if needed."""
    if not _table_exists(con, "bctc_reports"):
        return
    con.execute(
        """
        INSERT INTO financial_statement_reports (ticker, year, content, source_file, created_at)
        SELECT ticker, year, content, source_file, created_at
        FROM bctc_reports
        ON CONFLICT (ticker, year) DO UPDATE SET
            content = EXCLUDED.content,
            source_file = EXCLUDED.source_file,
            created_at = EXCLUDED.created_at
        """
    )


@contextmanager
def connection_scope(db_path: Path | None = None) -> Iterator[duckdb.DuckDBPyConnection]:
    con = get_connection(db_path)
    try:
        yield con
    finally:
        con.close()


def ensure_company(ticker: str, db_path: Path | None = None) -> None:
    code = ticker.strip().upper()
    if not code:
        raise ValueError("Ticker cannot be empty")
    with connection_scope(db_path) as con:
        con.execute(
            """
            INSERT INTO companies (ticker)
            VALUES (?)
            ON CONFLICT (ticker) DO NOTHING
            """,
            [code],
        )


def delete_company(ticker: str, db_path: Path | None = None) -> None:
    code = ticker.strip().upper()
    if not code:
        return
    with connection_scope(db_path) as con:
        con.execute("DELETE FROM companies WHERE ticker = ?", [code])


def list_companies(db_path: Path | None = None) -> list[str]:
    with connection_scope(db_path) as con:
        rows = con.execute("SELECT ticker FROM companies ORDER BY ticker").fetchall()
    return [str(r[0]) for r in rows]
