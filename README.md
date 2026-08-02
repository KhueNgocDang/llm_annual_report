# Annual Report Intelligence Platform (Rebuild)

This branch is a from-scratch implementation guided by `PRD_UNIFIED.md`.

## Quick Start

Default setup installs the lightweight Phase 0 stack.

1. Install dependencies:

```bash
uv sync
```

2. Bootstrap directories and database schema:

```bash
uv run python -m rebuild_init
```

3. Open the marimo workspace for interactive data extraction:

```bash
uv run python -m main
```

This opens [marimo_app.py](marimo_app.py) in marimo (`marimo edit`) with parameter cells for:
- dataset selection (`all`, `annual`, `financial_statement`, `financial_rows`)
- ticker/year filters
- optional read-only SQL (`SELECT`/`WITH` only)
- optional CSV export path

The marimo app now also includes a DuckDB chat analyst flow:
- natural-language question to SQL suggestion
- schema-aware SQL generation templates
- optional OpenAI fallback (`enable_llm_fallback`, `force_llm`)
- safe SQL guardrails before execution
- automatic line/bar chart rendering for compatible result shapes
- lightweight chat history for follow-up analysis

For OpenAI fallback mode, set `OPENAI_API_KEY` in `.env`. The app reads this key directly from the project `.env` at runtime.

4. Run Phase 2 markdown loading from terminal (optional):

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
  - CLI bootstrap and health stats

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
  - Ticker/year scoped loads from CLI

- marimo workspace is available for data retrieval:
  - output dataset filtering by ticker/year
  - ad-hoc read-only SQL for deeper extraction
  - direct CSV export of query results

## MCP-Friendly SQL Templates

Use [mcp_financial_statement_queries.sql](mcp_financial_statement_queries.sql) for reusable queries with the DuckDB MCP server, including:

1. Discover item codes by keyword.
2. Single-item time series for one ticker.
3. Multi-item trend extraction by keyword.
4. Latest balance-sheet metric snapshot.
