# Annual Report Intelligence Platform (Rebuild)

This branch is a from-scratch implementation guided by `PRD_UNIFIED.md`.

## Quick Start

Default setup installs the lightweight Phase 0 stack.

1. Bootstrap directories and database schema:

```bash
uv run python -m rebuild_init
```

2. Start the NiceGUI app shell:

```bash
uv run python -m main
```

The app opens on the default NiceGUI local address.

2. Start the Dash frontend:

```bash
uv run python -m dash_app
```

Open http://127.0.0.1:8050 to use the task runner and output explorer dashboard.

3. Run Phase 2 markdown loading from terminal (optional):

```bash
uv run python -m load_reports --dataset all --start-year 2015 --end-year 2025
```

Example with ticker filter:

```bash
uv run python -m load_reports --dataset financial_statement --tickers "CAG, VOS" --start-year 2020 --end-year 2025
```

## Full Stack Dependencies

For OCR and advanced pipeline phases, install optional heavy dependencies:

```bash
uv sync --extra full
```

## Current Status

- Phase 0 foundation is implemented:
  - Environment/config bootstrap
  - Directory bootstrap
  - DuckDB schema initialization
  - Minimal app shell with health stats

- Phase 1 ingestion is in progress:
  - Company registry add/delete/list
  - Stocks sync from VNDirect
  - Financial models sync
  - Financial statements sync (all companies, year-range scoped)
  - Financial ratios sync (all companies, year-range scoped)
  - Vietstock document listing sync (single ticker + all companies)
  - Vietstock download resolver:
    - direct PDF download
    - ZIP extraction and candidate selection
    - RAR extraction fallback (rarfile/unrar/bsdtar)
    - optional LLM-assisted candidate selection
    - persisted selection method/reason and sync errors

- Phase 2 conversion/loading MVP is now available:
  - Markdown corpus loader for annual reports and financial statement reports
  - Idempotent upserts into `annual_reports` and `financial_statement_reports`
  - Lineage tracking in `pipeline_files` with stage keys:
    - `markdown_annual`
    - `markdown_financial_statement`
  - Ticker/year scoped loads from UI controls

- Simple output visualization is available in UI:
  - Dataset/ticker/year output filtering
  - Table view with source file and content preview
  - Selected row full-content preview panel

- Dash frontend is available for operations + visualization:
  - Task Runner tab to execute pipeline actions
  - Output Explorer tab with table and charts (by year, top tickers)
