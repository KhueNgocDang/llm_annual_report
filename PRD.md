# PRD: Vietnamese Stock Data & Annual Report Pipeline

## Overview

A tool with a web UI that:
1. Fetches all Vietnamese listed stocks from VNDirect
2. Fetches financial statements, ratios, and model metadata from VNDirect
3. Fetches annual report PDFs from Vietstock Finance, converts them to Markdown using `marker-pdf`
4. Stores everything in a DuckDB database. The UI allows users to manage companies, monitor pipeline progress, trigger individual stages, and browse loaded data.

## Problem

Vietnamese listed companies publish annual reports and financial data across multiple platforms (Vietstock, VNDirect). There is no single, structured, queryable local store that combines financial statements, ratios, and full-text annual report content for analysis.

## Goals

1. Fetch all Vietnamese listed stocks (HOSE, HNX, UPCOM) from VNDirect into a local `stocks` table
2. Fetch financial statement data, ratios, and model metadata from VNDirect per ticker
3. Fetch document listings from Vietstock for configured company tickers
4. Download annual report PDFs to a local raw layer
5. Convert PDFs to Markdown using `marker-pdf`
6. Store extracted Markdown content in a DuckDB table with metadata (ticker, year)
7. Provide a web UI (NiceGUI) to execute and monitor all pipelines

## Non-Goals

- PDF summarization or LLM-based analysis (out of scope for v1)

---

## Architecture

```
VNDirect API (api-finfo.vndirect.com.vn/v4)
        │
        ├──► [0. Sync Stocks]  ── /stocks
        │       → stocks table (all HOSE/HNX/UPCOM tickers)
        │
        ├──► [1a. Sync Financial Models]  ── /financial_models
        │       → financial_models table (line-item metadata)
        │
        ├──► [1b. Sync Financial Statements]  ── /financial_statements
        │       → financial_statements table (per ticker, annual)
        │
        └──► [1c. Sync Financial Ratios]  ── /ratios
                → financial_ratios table (per ticker, per report date)

Vietstock Finance API
        │
        ▼
  [2. Sync Document Listings]  ── vietstock_documents.py
        │  (fetches metadata for all configured tickers,
        │   stores in vietstock_documents table)
        ▼
  [3. Download PDFs]  ── vietstock_documents.py
        │  (downloads unsynced PDFs to data/raw/<TICKER>/)
        ▼
  Local PDF files (data/raw/<TICKER>/<title>.pdf)
        │
        ▼
  [4. Convert to Markdown]  ── marker-pdf
        │
        ▼
  Local Markdown files (data/output/<TICKER>/<title>.md)
        │
        ▼
  [5. Load into DuckDB]  ── duckdb
        │
        ▼
  DuckDB table: annual_reports(ticker, year, content, source_file, created_at)
```

---

## Detailed Requirements

### 0. Stock Listing Sync (implemented in `financial_data.py`)

- **Source**: `https://api-finfo.vndirect.com.vn/v4/stocks?q=type:stock~floor:HOSE,HNX,UPCOM&size=9999`
- **No auth required**
- **Behavior**:
  - Fetch all listed/delisted stocks across HOSE, HNX, and UPCOM exchanges in a single request
  - Store into the `stocks` table: `code`, `type`, `floor`, `status`, `company_name`, `company_name_eng`, `short_name`, `listed_date`, `delisted_date`, `company_id`, `tax_code`, `isin`
  - Full replace on each sync (delete + insert) since the dataset is small (~2000 rows)
  - This populates the universe of tickers available for financial data and document sync

### 1. Financial Data Sync (implemented in `financial_data.py`)

- **Source**: `api-finfo.vndirect.com.vn/v4`
  - `/financial_models` — metadata describing each line-item code
  - `/financial_statements` — annual financial statement values per company
  - `/ratios` — financial ratios per company per report date
- **No auth required** (public API, paginated via `page` + `size` params)
- **Behavior**:
  - **Financial Models**: Fetch all model metadata (line-item codes, names, display order). Full replace on sync.
  - **Financial Statements**: For each ticker, fetch annual statements for a configurable year range (e.g. 2015–2025). Replace per-ticker on sync.
  - **Financial Ratios**: For each ticker, fetch ratios for the same year range. Replace per-ticker on sync.
  - All three operations are paginated via `_paginated_get()` helper
  - Progress callback for UI integration

### 1a. Vietstock Document Sync (implemented in `vietstock_documents.py`)

- **Source**: `https://finance.vietstock.vn/data/getdocument` API
- **Auth**: Session-based — loads the documents page to obtain cookies and a `__RequestVerificationToken`, then POSTs to the API
- **Input**: Stock ticker code (e.g. `"VNM"`) and doc_type (`"2"` = annual reports)
- **Behavior**:
  - For each ticker in the `companies` table, call the Vietstock API to list available documents
  - Extract normalised fields: `id`, `title`, `full_name`, `source`, `published_date`, `file_url`, `file_info_id`
  - Upsert into a `vietstock_documents` DuckDB table (dedup by `id` and `file_info_id`)
  - Track download status via `synced_to_raw` boolean and `raw_path` columns

### 2. PDF Download
- **Destination**: `data/raw/<TICKER>/<sanitised_title>.pdf`
- **Behavior**:
  - Download all documents where `synced_to_raw = FALSE`
  - Mark as synced after successful download
  - Skip already-downloaded files
  - Progress callback for UI integration

### 3. PDF-to-Markdown Conversion

- **Tool**: `marker-pdf` (already in dependencies)
- **Input**: Each downloaded PDF in `data/raw/<TICKER>/`
- **Output**: Markdown file in `data/output/<TICKER>/`, e.g. `data/output/VNM/<title>.md`
- **Behavior**:
  - Process each PDF individually
  - Skip conversion if the output `.md` file already exists and the source PDF has not changed (mtime check)
  - Log progress (file being converted, success/failure)

### 4. DuckDB Storage

- **Database file**: `db.db` (project root)

**Tables**:

```sql
-- All Vietnamese listed stocks (from VNDirect)
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

-- Company registry (subset of stocks the user wants to track)
CREATE TABLE IF NOT EXISTS companies (
    ticker VARCHAR PRIMARY KEY
);

-- Financial model metadata (line-item definitions)
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

-- Annual financial statements per company
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

-- Financial ratios per company per report date
CREATE TABLE IF NOT EXISTS financial_ratios (
    code         VARCHAR NOT NULL,
    ratio_group  VARCHAR,
    report_date  VARCHAR,
    item_code    VARCHAR,
    ratio_code   VARCHAR,
    item_name    VARCHAR,
    value        DOUBLE
);

-- Document metadata from Vietstock
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

-- Extracted Markdown content from annual reports
CREATE TABLE IF NOT EXISTS annual_reports (
    ticker      VARCHAR NOT NULL,
    year        INTEGER NOT NULL,
    content     VARCHAR NOT NULL,
    source_file VARCHAR NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, year)
);
```

- **Behavior**:
  - On load, upsert (INSERT OR REPLACE) each Markdown file's content into `annual_reports`
  - `ticker` is from the folder/document metadata
  - `year` is extracted from the document title or published_date
  - `content` is the full Markdown text
  - `source_file` is the original PDF filename

### 5. Web UI (NiceGUI)

- **Framework**: NiceGUI
- **Entry point**: `uv run python main.py` — starts a local web server (default `http://localhost:8080`)

#### Layout

The UI is a single-page dashboard organised into **task cards** — one per data-loading operation. Each card is self-contained: it has its own trigger button, progress indicator, and result summary.

---

**Global Controls (top bar)**
- Manage tickers: add/remove company tickers to the `companies` table
- "Run Full Pipeline" button — runs all task cards sequentially top-to-bottom
- Year range selector (start year / end year) for financial data tasks

---

**Task Card: Sync Stocks**
- Button: "Sync Stocks"
- Fetches all HOSE/HNX/UPCOM tickers from VNDirect
- Monitor:
  - Spinner while request is in-flight
  - On completion: shows total rows fetched (e.g. "✅ 1,952 stocks synced")
  - On error: red badge with error message
- Result: row count badge + last synced timestamp

**Task Card: Sync Financial Models**
- Button: "Sync Models"
- Fetches all financial model metadata from VNDirect
- Monitor:
  - Spinner while fetching (may be multi-page)
  - Page counter: "Fetching page 2/5…"
  - On completion: "✅ 842 models synced"
  - On error: red badge
- Result: row count badge + last synced timestamp

**Task Card: Sync Financial Statements**
- Button: "Sync Statements"
- Iterates over all tickers in `companies` table
- Monitor:
  - Overall progress bar: `12 / 45 tickers`
  - Per-ticker status table (live-updating):
    | Ticker | Rows | Status |
    |--------|------|--------|
    | VNM | 1,240 | ✅ Done |
    | FPT | — | ⏳ Running |
    | HPG | — | ⬜ Pending |
  - On error for a ticker: row turns red with error text, continues to next
- Result: total rows across all tickers + count of succeeded/failed tickers

**Task Card: Sync Financial Ratios**
- Button: "Sync Ratios"
- Same monitoring pattern as Sync Financial Statements:
  - Overall progress bar per ticker
  - Per-ticker status table with row count
  - Error rows shown inline
- Result: total rows + succeeded/failed ticker count

**Task Card: Sync Document Listings (Vietstock)**
- Button: "Sync Listings"
- Iterates over all tickers in `companies` table, calls Vietstock API
- Monitor:
  - Overall progress bar: `5 / 20 tickers`
  - Per-ticker status table:
    | Ticker | Docs Found | Status |
    |--------|------------|--------|
    | VNM | 12 | ✅ Done |
    | FPT | — | ⏳ Running |
  - On error (e.g. CSRF token failure): row turns red, continues
- Result: total documents synced + succeeded/failed ticker count

**Task Card: Download PDFs**
- Button: "Download PDFs"
- Downloads all documents where `synced_to_raw = FALSE`
- Monitor:
  - Overall progress bar: `3 / 18 files`
  - Per-file status table:
    | Ticker | Title | Size | Status |
    |--------|-------|------|--------|
    | VNM | BCTN 2024 | 4.2 MB | ✅ Done |
    | FPT | Annual Report 2023 | — | ⏳ Downloading |
    | HPG | BCTN 2024 | — | ⬜ Pending |
  - On error: row turns red, continues to next file
- Result: files downloaded / total + total size

**Task Card: Convert to Markdown**
- Button: "Convert to MD"
- Runs `marker-pdf` on each downloaded PDF without an existing `.md` output
- Monitor:
  - Overall progress bar: `2 / 10 files`
  - Per-file status table:
    | Ticker | File | Status |
    |--------|------|--------|
    | VNM | BCTN_2024.pdf | ✅ Done |
    | FPT | BCTN_2023.pdf | ⏳ Converting |
    | HPG | BCTN_2024.pdf | ⏭ Skipped (exists) |
  - On error: row turns red with error detail
- Result: converted / skipped / failed counts

**Task Card: Load Markdown to DB**
- Button: "Load to DB"
- Reads each `.md` file from `data/output/` and upserts into `annual_reports`
- Monitor:
  - Overall progress bar: `5 / 8 files`
  - Per-file status table:
    | Ticker | Year | Content Length | Status |
    |--------|------|----------------|--------|
    | VNM | 2024 | 48,320 chars | ✅ Loaded |
    | FPT | 2023 | — | ⏳ Loading |
  - On error: row turns red
- Result: rows loaded / failed count

---

**Data Browser Panels** (below task cards)

- **Stocks Browser**: search/filter `stocks` table by ticker, floor, status
- **Financial Statements Browser**: select ticker → pivot by fiscal year, show line items
- **Financial Ratios Browser**: select ticker → show ratios by report date
- **Document Browser**: filter `vietstock_documents` by ticker, show title, published_date, sync status; click to view/download PDF
- **Annual Reports Browser**: show loaded reports (ticker, year, content length, created_at); click row to preview Markdown

---

#### Shared UI Patterns

All task cards follow the same monitoring contract:

1. **Idle state**: Button enabled, shows last-run timestamp and summary badge (if previously run)
2. **Running state**: Button disabled (all other task buttons also disabled to prevent concurrent runs), spinner + progress bar + live status table
3. **Completed state**: Green summary badge, result counts, timestamp updated
4. **Error state**: Red badge on failed items; successfully-processed items still shown as green; overall card shows partial-success summary (e.g. "38/45 tickers OK, 7 failed")

Implementation:
- Each task runs in a **background thread**
- UI updates via NiceGUI `ui.timer` (polling at ~500ms) or async binding
- Tasks emit progress events via a callback: `on_progress(item, status, detail)`
- The status table is backed by a reactive list; new rows are appended / updated in-place
- Log output area at the bottom of each card (collapsible) shows raw log lines

---

## Directory Structure

```
annual_report/
├── main.py                    # NiceGUI app entry point & pipeline orchestration
├── config.py                  # Shared config (RAW_DIR, OUTPUT_DIR, DB path)
├── database.py                # DuckDB schema setup & helpers (ensure_company, etc.)
├── financial_data.py          # VNDirect API client: stocks, statements, ratios, models
├── vietstock_documents.py     # Vietstock API client: list, download, sync
├── pyproject.toml
├── PRD.md
├── db.db                      # DuckDB database (gitignored)
├── data/
│   ├── raw/                   # Downloaded PDFs by ticker
│   │   ├── VNM/
│   │   │   └── Bao_cao_thuong_nien_2024.pdf
│   │   └── FPT/
│   │       └── BCTN_2023.pdf
│   └── output/                # Converted Markdown by ticker
│       ├── VNM/
│       │   └── Bao_cao_thuong_nien_2024.md
│       └── FPT/
│           └── BCTN_2023.md
```

---

## Error Handling

| Scenario | Behavior |
|---|---|
| VNDirect API returns empty data | Return 0 rows, log warning |
| VNDirect API pagination mismatch | Stop at last page, log |
| Vietstock API returns empty/non-JSON | Return empty list, log warning |
| CSRF token not found | Raise error — Vietstock may have changed page structure |
| File URL missing or invalid | Skip document, log warning |
| Download fails (network error) | Log error, continue with next file |
| marker-pdf fails on a PDF | Log error, continue with next file |
| DuckDB write failure | Log error, continue with next file |
| Duplicate documents (same file_info_id) | Kept via dedup logic, update existing record |

---

## Dependencies

Already declared in `pyproject.toml`:

| Package | Purpose |
|---|---|
| `requests` | HTTP client for VNDirect & Vietstock APIs |
| `beautifulsoup4` | Parse CSRF token from HTML |
| `pandas` | Data preview / manipulation |
| `marker-pdf` | PDF → Markdown conversion |
| `duckdb` | Local analytical database |
| `nicegui` | Web UI framework |

---

## Milestones

### M1: Project Scaffolding & Database Layer
- [x] `config.py` — shared constants (`RAW_DIR`, `OUTPUT_DIR`, `DB_PATH`)
- [ ] `database.py` — DuckDB schema creation (all 7 tables), `ensure_company()`, connection helper
- [ ] `.gitignore` — exclude `db.db`, `data/`, `token.json`, `credentials.json`, `__pycache__`
- **Exit criteria**: `uv run python -c "from database import init_db; init_db()"` creates `db.db` with all tables

### M2: Stock Listing Sync
- [ ] `financial_data.py` — `fetch_stocks()` function hitting VNDirect `/stocks` endpoint
- [ ] Store results in `stocks` table (full replace)
- **Exit criteria**: `stocks` table contains ~1,900+ rows after running `fetch_stocks()`

### M3: Financial Data Sync
- [x] `financial_data.py` — `fetch_financial_models()`, `fetch_financial_statements()`, `fetch_financial_ratios()`
- [ ] Wire up to iterate over `companies` table tickers
- [ ] Progress callback signatures for UI integration
- **Exit criteria**: For a test ticker (e.g. VNM), all three tables populated with data for 2015–2025

### M4: Vietstock Document Sync & PDF Download
- [x] `vietstock_documents.py` — `list_documents()`, `fetch_documents_preview()`, `sync_documents_to_db()`, `download_document_to_raw()`, `download_all_unsynced()`
- [ ] Verify end-to-end: sync listings → download PDFs for a test ticker
- **Exit criteria**: `vietstock_documents` table populated; PDFs downloaded to `data/raw/<TICKER>/`

### M5: PDF-to-Markdown Conversion
- [ ] Conversion module — call `marker-pdf` on each PDF in `data/raw/`
- [ ] Output to `data/output/<TICKER>/<title>.md`
- [ ] Skip logic (mtime check)
- **Exit criteria**: `.md` files generated for all downloaded PDFs; re-run skips existing

### M6: Load Markdown to DuckDB
- [ ] Load module — read `.md` files, parse ticker + year, upsert into `annual_reports`
- **Exit criteria**: `SELECT ticker, year, length(content) FROM annual_reports` returns rows for every converted file

### M7: NiceGUI App Shell
- [ ] `main.py` — NiceGUI app boots at `http://localhost:8080`
- [ ] Global controls: ticker management (add/remove), year range selector, "Run Full Pipeline" button
- [ ] Shared task-card component: button, spinner, progress bar, status table, result badge
- **Exit criteria**: App launches; clicking buttons triggers placeholder tasks; UI state transitions work (idle → running → completed)

### M8: Task Card — Sync Stocks
- [ ] Wire `fetch_stocks()` to task card with spinner + row count result
- **Exit criteria**: Clicking "Sync Stocks" in the UI populates `stocks` table and shows result badge

### M9: Task Card — Sync Financial Models
- [ ] Wire `fetch_financial_models()` with page counter + result badge
- **Exit criteria**: "Sync Models" card shows page progress and final row count

### M10: Task Card — Sync Financial Statements
- [ ] Wire `fetch_financial_statements()` per-ticker with progress bar + per-ticker status table
- **Exit criteria**: Per-ticker live progress visible; partial failures shown inline

### M11: Task Card — Sync Financial Ratios
- [ ] Wire `fetch_financial_ratios()` per-ticker (same pattern as M10)
- **Exit criteria**: Per-ticker progress + error handling visible in UI

### M12: Task Card — Sync Document Listings
- [ ] Wire `fetch_all_companies_documents()` with per-ticker progress
- **Exit criteria**: Vietstock doc listings synced with live progress in UI

### M13: Task Card — Download PDFs
- [ ] Wire `download_all_unsynced()` with per-file progress + size display
- **Exit criteria**: PDFs download with real-time file-by-file progress

### M14: Task Card — Convert to Markdown
- [ ] Wire marker-pdf conversion with per-file progress + skip indicator
- **Exit criteria**: Conversion runs with live status; skipped files shown as ⏭

### M15: Task Card — Load Markdown to DB
- [ ] Wire Markdown loading with per-file progress
- **Exit criteria**: `annual_reports` table populated; UI shows content length per loaded file

### M16: Data Browser Panels
- [ ] Stocks Browser — search/filter by ticker, floor, status
- [ ] Financial Statements Browser — select ticker, pivot by fiscal year
- [ ] Financial Ratios Browser — select ticker, show by report date
- [ ] Document Browser — filter by ticker, show sync status, link to PDF
- [ ] Annual Reports Browser — show loaded reports, click to preview Markdown
- **Exit criteria**: All 5 browser panels functional with data from previous milestones

### M17: Full Pipeline & Polish
- [ ] "Run Full Pipeline" executes M8–M15 sequentially
- [ ] Concurrent run prevention (all buttons disabled during any task)
- [ ] Persist config (last-used tickers, year range) across restarts
- [ ] Error summary at end of full pipeline run
- **Exit criteria**: Full end-to-end pipeline runs from empty state to browsable data in one click

---

## Success Criteria

1. `uv run python main.py` launches the NiceGUI dashboard at `http://localhost:8080`
2. "Sync Stocks" fetches ~2000 tickers from VNDirect into the `stocks` table
3. Adding a ticker and clicking "Sync Financials" fetches financial models, statements, and ratios from VNDirect
4. "Sync Listings" fetches document metadata from Vietstock into DuckDB
5. "Download PDFs" downloads all unsynced PDFs to `data/raw/<TICKER>/`
6. "Convert to MD" converts downloaded PDFs to Markdown via `marker-pdf`
7. "Load to DB" upserts Markdown content into the `annual_reports` table
8. "Run Full Pipeline" executes all stages sequentially
9. Progress is visible in real-time in the UI (per-ticker status + log output)
10. The Data Browser shows all loaded reports with correct ticker, year, and content
11. The Financial Data Browser allows browsing statements and ratios per ticker
12. Re-running the pipeline skips already-synced data where applicable
13. Individual stages can be triggered independently via the UI buttons
