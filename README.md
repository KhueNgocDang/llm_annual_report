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
