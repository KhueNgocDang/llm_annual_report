"""NiceGUI dashboard for the Vietnamese Stock Data & Annual Report Pipeline."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from nicegui import ui

from config import DEFAULT_END_YEAR, DEFAULT_START_YEAR
from converter import (
    list_nonstandard_input_files,
    preview_raw_input_dir_sync,
    resync_input_dir,
)
from database import get_connection, init_db, ensure_company, delete_company

# ---------------------------------------------------------------------------
# Task state model
# ---------------------------------------------------------------------------


@dataclass
class TaskRow:
    label: str
    detail: str = ""
    status: str = "pending"  # pending | running | done | skipped | error


@dataclass
class TaskState:
    running: bool = False
    rows: list[TaskRow] = field(default_factory=list)
    summary: str = ""
    last_run: str = ""
    error: str = ""
    progress: float = 0.0


# Global lock to prevent concurrent pipeline runs
_pipeline_lock = threading.Lock()


def _run_in_thread(fn: Callable, state: TaskState, refresh: Callable) -> None:
    """Execute *fn* in a background thread, guarded by the pipeline lock."""
    if not _pipeline_lock.acquire(blocking=False):
        state.error = "Another task is already running"
        refresh()
        return

    state.running = True
    state.rows.clear()
    state.summary = ""
    state.error = ""
    refresh()

    def _worker():
        try:
            fn()
        except Exception as exc:
            state.error = str(exc)
        finally:
            state.running = False
            state.last_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _pipeline_lock.release()
            refresh()

    threading.Thread(target=_worker, daemon=True).start()


# ---------------------------------------------------------------------------
# Task-card UI component
# ---------------------------------------------------------------------------


def task_card(
    title: str, state: TaskState, run_fn: Callable, refresh: Callable
):
    """Render a self-contained task card."""
    with ui.card().classes("w-full"):
        with ui.row().classes("items-center w-full justify-between"):
            ui.label(title).classes("text-lg font-bold")
            with ui.row().classes("items-center gap-2"):
                if state.last_run:
                    ui.label(f"Last: {state.last_run}").classes(
                        "text-xs text-gray-500"
                    )
                if state.running:
                    ui.spinner(size="sm")
                btn = ui.button(
                    "Run",
                    on_click=lambda: _run_in_thread(run_fn, state, refresh),
                )
                btn.props("dense")
                if state.running:
                    btn.disable()

        if state.error:
            ui.label(state.error).classes("text-red-500 text-sm")

        if state.summary:
            ui.label(state.summary).classes(
                "text-green-600 text-sm font-medium"
            )

        if state.rows:
            columns = [
                {
                    "name": "label",
                    "label": "Item",
                    "field": "label",
                    "align": "left",
                },
                {
                    "name": "detail",
                    "label": "Detail",
                    "field": "detail",
                    "align": "left",
                },
                {
                    "name": "status",
                    "label": "Status",
                    "field": "status",
                    "align": "center",
                },
            ]
            rows_data = [
                {
                    "label": r.label,
                    "detail": r.detail,
                    "status": {
                        "done": "✅",
                        "running": "⏳",
                        "pending": "⬜",
                        "skipped": "⏭",
                        "error": "❌",
                    }.get(r.status, r.status),
                }
                for r in state.rows
            ]
            ui.table(columns=columns, rows=rows_data, row_key="label").classes(
                "w-full"
            ).props("dense flat")


# ---------------------------------------------------------------------------
# Task implementations
# ---------------------------------------------------------------------------


def _make_sync_stocks(
    state: TaskState,
    refresh: Callable,
    *,
    force: Callable[[], bool] = lambda: False,
) -> Callable:
    def run():
        from financial_data import fetch_stocks

        con = get_connection()
        try:
            init_db(con)
            existing = con.execute("SELECT COUNT(*) FROM stocks").fetchone()[0]
            if existing > 0 and not force():
                state.summary = f"⏭ Already synced ({existing:,} stocks)"
                return
            count = fetch_stocks(con)
            state.summary = f"✅ {count:,} stocks synced"
        finally:
            con.close()

    return run


def _make_sync_models(
    state: TaskState,
    refresh: Callable,
    *,
    force: Callable[[], bool] = lambda: False,
) -> Callable:
    def run():
        from financial_data import fetch_financial_models

        con = get_connection()
        try:
            init_db(con)
            existing = con.execute(
                "SELECT COUNT(*) FROM financial_models"
            ).fetchone()[0]
            if existing > 0 and not force():
                state.summary = f"⏭ Already synced ({existing:,} models)"
                return
            count = fetch_financial_models(con)
            state.summary = f"✅ {count:,} models synced"
        finally:
            con.close()

    return run


def _make_sync_statements(
    state: TaskState,
    start_year: Callable[[], int],
    end_year: Callable[[], int],
    refresh: Callable,
    *,
    force: Callable[[], bool] = lambda: False,
) -> Callable:
    def run():
        from financial_data import fetch_financial_statements

        con = get_connection()
        try:
            init_db(con)
            tickers = con.execute(
                "SELECT ticker FROM companies ORDER BY ticker"
            ).fetchall()
            if not tickers:
                state.summary = "No companies configured"
                return

            # Per-ticker skip: find tickers that already have data
            existing_tickers = set()
            if not force():
                rows = con.execute(
                    "SELECT DISTINCT code FROM financial_statements"
                ).fetchall()
                existing_tickers = {r[0] for r in rows}

            state.rows = [TaskRow(label=t[0]) for t in tickers]
            refresh()

            total_rows = 0
            ok = 0
            fail = 0
            skipped = 0
            sy, ey = start_year(), end_year()
            for i, (ticker,) in enumerate(tickers):
                if ticker in existing_tickers:
                    cnt = con.execute(
                        "SELECT COUNT(*) FROM financial_statements WHERE code = ?",
                        [ticker],
                    ).fetchone()[0]
                    state.rows[i].detail = f"⏭ {cnt:,} rows"
                    state.rows[i].status = "skipped"
                    skipped += 1
                    refresh()
                    continue
                state.rows[i].status = "running"
                refresh()
                try:
                    count = fetch_financial_statements(con, ticker, sy, ey)
                    state.rows[i].detail = f"{count:,} rows"
                    state.rows[i].status = "done"
                    total_rows += count
                    ok += 1
                except Exception as exc:
                    state.rows[i].detail = str(exc)[:80]
                    state.rows[i].status = "error"
                    fail += 1
                refresh()
            if ok == 0 and fail == 0:
                state.summary = f"⏭ Already synced ({skipped} tickers)"
            else:
                state.summary = f"✅ {total_rows:,} rows | {ok} OK, {fail} failed, {skipped} skipped"
        finally:
            con.close()

    return run


def _make_sync_ratios(
    state: TaskState,
    start_year: Callable[[], int],
    end_year: Callable[[], int],
    refresh: Callable,
    *,
    force: Callable[[], bool] = lambda: False,
) -> Callable:
    def run():
        from financial_data import fetch_financial_ratios

        con = get_connection()
        try:
            init_db(con)
            tickers = con.execute(
                "SELECT ticker FROM companies ORDER BY ticker"
            ).fetchall()
            if not tickers:
                state.summary = "No companies configured"
                return

            # Per-ticker skip: find tickers that already have data
            existing_tickers = set()
            if not force():
                rows = con.execute(
                    "SELECT DISTINCT code FROM financial_ratios"
                ).fetchall()
                existing_tickers = {r[0] for r in rows}

            state.rows = [TaskRow(label=t[0]) for t in tickers]
            refresh()

            total_rows = 0
            ok = 0
            fail = 0
            skipped = 0
            sy, ey = start_year(), end_year()
            for i, (ticker,) in enumerate(tickers):
                if ticker in existing_tickers:
                    cnt = con.execute(
                        "SELECT COUNT(*) FROM financial_ratios WHERE code = ?",
                        [ticker],
                    ).fetchone()[0]
                    state.rows[i].detail = f"⏭ {cnt:,} rows"
                    state.rows[i].status = "skipped"
                    skipped += 1
                    refresh()
                    continue
                state.rows[i].status = "running"
                refresh()
                try:
                    count = fetch_financial_ratios(con, ticker, sy, ey)
                    state.rows[i].detail = f"{count:,} rows"
                    state.rows[i].status = "done"
                    total_rows += count
                    ok += 1
                except Exception as exc:
                    state.rows[i].detail = str(exc)[:80]
                    state.rows[i].status = "error"
                    fail += 1
                refresh()
            if ok == 0 and fail == 0:
                state.summary = f"⏭ Already synced ({skipped} tickers)"
            else:
                state.summary = f"✅ {total_rows:,} rows | {ok} OK, {fail} failed, {skipped} skipped"
        finally:
            con.close()

    return run


def _make_sync_listings(
    state: TaskState,
    start_year: Callable[[], int],
    end_year: Callable[[], int],
    refresh: Callable,
    *,
    force: Callable[[], bool] = lambda: False,
) -> Callable:
    def run():
        from vietstock_documents import fetch_all_companies_documents

        con = get_connection()
        try:
            init_db(con)
            tickers = con.execute(
                "SELECT ticker FROM companies ORDER BY ticker"
            ).fetchall()
            if not tickers:
                state.summary = "No companies configured"
                return
            # Per-ticker skip: find tickers that already have listings
            existing_tickers = set()
            if not force():
                rows = con.execute(
                    "SELECT DISTINCT ticker FROM vietstock_documents"
                ).fetchall()
                existing_tickers = {r[0] for r in rows}

            # Filter to only new tickers (unless force)
            new_tickers = [t for t in tickers if t[0] not in existing_tickers]
            if not new_tickers:
                total_existing = con.execute(
                    "SELECT COUNT(*) FROM vietstock_documents"
                ).fetchone()[0]
                state.summary = (
                    f"⏭ Already synced ({total_existing:,} documents)"
                )
                return

            state.rows = [TaskRow(label=t[0]) for t in tickers]
            refresh()

            ticker_index = {t[0]: i for i, t in enumerate(tickers)}

            # Mark existing tickers as skipped
            for t in tickers:
                if t[0] in existing_tickers:
                    idx = ticker_index[t[0]]
                    cnt = con.execute(
                        "SELECT COUNT(*) FROM vietstock_documents WHERE ticker = ?",
                        [t[0]],
                    ).fetchone()[0]
                    state.rows[idx].detail = f"⏭ {cnt} docs"
                    state.rows[idx].status = "skipped"
            refresh()

            def on_progress(ticker, count, error):
                idx = ticker_index.get(ticker)
                if idx is not None:
                    if error:
                        state.rows[idx].detail = str(error)[:80]
                        state.rows[idx].status = "error"
                    else:
                        state.rows[idx].detail = f"{count} docs"
                        state.rows[idx].status = "done"
                    refresh()

            results = fetch_all_companies_documents(
                con,
                tickers=[t[0] for t in new_tickers],
                start_year=start_year(),
                end_year=end_year(),
                on_progress=on_progress,
            )
            total = sum(results.values())
            ok = sum(1 for v in results.values() if v >= 0)
            skipped = len(existing_tickers)
            state.summary = (
                f"✅ {total:,} documents synced across {ok} tickers"
                + (f", {skipped} skipped" if skipped else "")
            )
        finally:
            con.close()

    return run


def _make_download_pdfs(
    state: TaskState,
    refresh: Callable,
    *,
    force: Callable[[], bool] = lambda: False,
) -> Callable:
    def run():
        from vietstock_documents import download_all_unsynced

        con = get_connection()
        try:
            init_db(con)
            pending = con.execute(
                "SELECT id, ticker, title FROM vietstock_documents "
                "WHERE synced_to_raw = FALSE AND file_url IS NOT NULL AND file_url != '' "
                "ORDER BY ticker"
            ).fetchall()
            if not pending:
                state.summary = "No files to download"
                return

            state.rows = [
                TaskRow(label=f"{t} – {title}") for _, t, title in pending
            ]
            refresh()

            row_map: dict[tuple, int] = {}
            for i, (doc_id, t, title) in enumerate(pending):
                row_map[(t, title)] = i

            downloaded = 0
            failed = 0

            def on_progress(ticker, title, path, error):
                nonlocal downloaded, failed
                idx = row_map.get((ticker, title))
                if idx is not None:
                    if error:
                        state.rows[idx].detail = str(error)[:80]
                        state.rows[idx].status = "error"
                        failed += 1
                    else:
                        state.rows[idx].detail = path or ""
                        state.rows[idx].status = "done"
                        downloaded += 1
                    refresh()

            download_all_unsynced(con, on_progress=on_progress)
            state.summary = f"✅ {downloaded} downloaded, {failed} failed"
        finally:
            con.close()

    return run


def _make_convert(
    state: TaskState,
    refresh: Callable,
    *,
    start_year: Callable[[], int | None] = lambda: None,
    end_year: Callable[[], int | None] = lambda: None,
    force: Callable[[], bool] = lambda: False,
) -> Callable:
    def run():
        from converter import create_jobs, run_pending_jobs, get_job_summary

        con = get_connection()
        try:
            init_db(con)

            # Skip only if there are no pending jobs and some completed (unless forced)
            summary = get_job_summary(con)
            if (
                summary.get("pending", 0) == 0
                and summary.get("completed", 0) > 0
                and not force()
            ):
                state.summary = (
                    f"⏭ Already converted ({summary['completed']:,} jobs completed). "
                    f"{summary.get('failed', 0)} failed"
                )
                return

            # Create jobs for any new staged PDFs
            created = create_jobs(
                con, start_year=start_year(), end_year=end_year()
            )
            if created:
                state.rows.append(
                    TaskRow(
                        label="Create jobs",
                        detail=f"{created} created",
                        status="done",
                    )
                )
                refresh()

            # Run all pending jobs
            def on_progress(job_id, ticker, year, status, error):
                row = TaskRow(
                    label=f"#{job_id} {ticker} {year}",
                    detail=error[:80] if error else "",
                    status="done" if status == "completed" else "error",
                )
                state.rows.append(row)
                refresh()

            results = run_pending_jobs(con, on_progress=on_progress)
            state.summary = (
                f"✅ {results['completed']} converted, "
                f"{results['failed']} failed "
                f"(of {results['total']} total)"
            )
        finally:
            con.close()

    return run


def _make_load(
    state: TaskState,
    refresh: Callable,
    *,
    force: Callable[[], bool] = lambda: False,
) -> Callable:
    def run():
        from loader import load_all

        con = get_connection()
        try:
            init_db(con)

            def on_progress(ticker, year, length, status, error):
                row = TaskRow(
                    label=f"{ticker} / {year or '?'}",
                    detail=(
                        f"{length:,} chars"
                        if length
                        else (str(error)[:80] if error else "")
                    ),
                    status="done" if status == "loaded" else "error",
                )
                state.rows.append(row)
                refresh()

            counts = load_all(con, on_progress=on_progress)
            state.summary = (
                f"✅ {counts['loaded']} loaded, {counts['failed']} failed"
            )
        finally:
            con.close()

    return run


# ---------------------------------------------------------------------------
# Shared navigation header
# ---------------------------------------------------------------------------

_NAV_ITEMS = [
    ("Home", "/"),
    ("Companies", "/companies"),
    ("Jobs", "/jobs"),
    ("Converter", "/converter"),
    ("Stocks", "/browse/stocks"),
    ("Financial", "/browse/financial"),
    ("Documents", "/browse/documents"),
    ("Reports", "/browse/reports"),
]


def _nav_header():
    """Render a consistent navigation bar at the top of every page."""
    with ui.header().classes("items-center gap-4"):
        ui.label("Annual Report Pipeline").classes("text-lg font-bold")
        for label, href in _NAV_ITEMS:
            ui.link(label, href).classes(
                "text-white no-underline hover:underline"
            )


# ---------------------------------------------------------------------------
# Home page — configuration + job generator
# ---------------------------------------------------------------------------

_INIT_JOBS = [
    ("stocks", "1. Sync Stocks"),
    ("models", "2. Sync Financial Models"),
]

_PIPELINE_JOBS = [
    ("statements", "3. Sync Financial Statements"),
    ("ratios", "4. Sync Financial Ratios"),
    ("listings", "5. Sync Document Listings"),
    ("download", "6. Download PDFs"),
    ("convert", "7. Convert to Markdown"),
    ("load", "8. Load Markdown to DB"),
]

_ALL_JOBS = _INIT_JOBS + _PIPELINE_JOBS


def _page_home_content():
    """Content for the home / configuration page."""
    init_db()

    with ui.column().classes("w-full max-w-5xl mx-auto gap-4 p-4"):
        ui.label("Configuration & Job Generator").classes("text-2xl font-bold")

        # --- Company summary ---
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center justify-between w-full"):
                ui.label("Companies").classes("text-lg font-bold")
                ui.link("Manage →", "/companies").classes(
                    "text-blue-600 no-underline hover:underline text-sm"
                )

            @ui.refreshable
            def company_chips():
                con = get_connection()
                try:
                    tickers = con.execute(
                        "SELECT ticker FROM companies ORDER BY ticker"
                    ).fetchall()
                finally:
                    con.close()
                if tickers:
                    with ui.row().classes("gap-2 flex-wrap"):
                        for (t,) in tickers:
                            ui.chip(t).props("outline")
                else:
                    ui.label("No companies added yet. ").classes(
                        "text-gray-500 text-sm"
                    )

            company_chips()

        # --- Job generator ---
        with ui.card().classes("w-full"):
            ui.label("Job Generator").classes("text-lg font-bold")

            with ui.row().classes("items-end gap-4 flex-wrap w-full"):
                exchange_sel = ui.select(
                    label="Exchanges",
                    options=["HOSE", "HNX", "UPCOM"],
                    multiple=True,
                    value=["HOSE"],
                ).props(
                    "dense options-dense use-chips outlined "
                    'style="min-width: 200px"'
                )

                years = list(range(2015, datetime.now().year + 1))
                start_yr = (
                    ui.select(
                        label="Start year",
                        options=years,
                        value=DEFAULT_START_YEAR,
                    )
                    .props("dense outlined")
                    .classes("w-32")
                )

                end_yr = (
                    ui.select(
                        label="End year",
                        options=years,
                        value=DEFAULT_END_YEAR,
                    )
                    .props("dense outlined")
                    .classes("w-32")
                )

            ui.separator()

            job_sel = (
                ui.select(
                    label="Select jobs to run",
                    options={k: v for k, v in _PIPELINE_JOBS},
                    multiple=True,
                    value=[k for k, _ in _PIPELINE_JOBS],
                )
                .props(
                    "dense options-dense use-chips outlined "
                    'style="min-width: 100%"'
                )
                .classes("w-full")
            )

            with ui.row().classes("gap-2 mt-2 flex-wrap"):

                def _go_run():
                    jobs = job_sel.value or []
                    exch = exchange_sel.value or []
                    sy = start_yr.value or DEFAULT_START_YEAR
                    ey = end_yr.value or DEFAULT_END_YEAR
                    params = (
                        f"?jobs={','.join(jobs)}"
                        f"&exchanges={','.join(exch)}"
                        f"&start_year={sy}"
                        f"&end_year={ey}"
                    )
                    ui.navigate.to(f"/jobs{params}")

                ui.button(
                    "Run Selected Jobs", on_click=_go_run, color="primary"
                )

                def _go_init():
                    exch = exchange_sel.value or []
                    sy = start_yr.value or DEFAULT_START_YEAR
                    ey = end_yr.value or DEFAULT_END_YEAR
                    init_keys = ",".join(k for k, _ in _INIT_JOBS)
                    params = (
                        f"?jobs={init_keys}"
                        f"&exchanges={','.join(exch)}"
                        f"&start_year={sy}"
                        f"&end_year={ey}"
                    )
                    ui.navigate.to(f"/jobs{params}")

                ui.button("Initial Setup", on_click=_go_init).props(
                    "outline"
                ).tooltip("Sync stocks & financial models (one-time)")

                def _go_pipeline():
                    exch = exchange_sel.value or []
                    sy = start_yr.value or DEFAULT_START_YEAR
                    ey = end_yr.value or DEFAULT_END_YEAR
                    pipe_keys = ",".join(k for k, _ in _PIPELINE_JOBS)
                    params = (
                        f"?jobs={pipe_keys}"
                        f"&exchanges={','.join(exch)}"
                        f"&start_year={sy}"
                        f"&end_year={ey}"
                    )
                    ui.navigate.to(f"/jobs{params}")

                ui.button("Run Pipeline", on_click=_go_pipeline).props(
                    "outline"
                ).tooltip(
                    "Statements, ratios, documents, download, convert, load"
                )

        # --- Quick stats ---
        with ui.card().classes("w-full"):
            ui.label("Database Summary").classes("text-lg font-bold")

            @ui.refreshable
            def stats_panel():
                con = get_connection()
                try:
                    tables = [
                        ("Stocks", "stocks"),
                        ("Companies", "companies"),
                        ("Financial Statements", "financial_statements"),
                        ("Financial Ratios", "financial_ratios"),
                        ("Financial Models", "financial_models"),
                        ("Vietstock Documents", "vietstock_documents"),
                        ("Annual Reports", "annual_reports"),
                    ]
                    with ui.row().classes("gap-4 flex-wrap"):
                        for label, table in tables:
                            count = con.execute(
                                f"SELECT COUNT(*) FROM {table}"
                            ).fetchone()[0]
                            with ui.column().classes("items-center"):
                                ui.label(f"{count:,}").classes(
                                    "text-xl font-bold"
                                )
                                ui.label(label).classes(
                                    "text-xs text-gray-500"
                                )
                finally:
                    con.close()

            stats_panel()
            ui.button("Refresh", on_click=lambda: stats_panel.refresh()).props(
                "dense flat"
            )


@ui.page("/")
def page_home():
    ui.dark_mode(False)
    _nav_header()
    _page_home_content()


# ---------------------------------------------------------------------------
# Companies page — add / remove stocks from the pipeline
# ---------------------------------------------------------------------------


@ui.page("/companies")
def page_companies():
    ui.dark_mode(False)
    _nav_header()
    init_db()

    with ui.column().classes("w-full max-w-5xl mx-auto gap-4 p-4"):
        ui.label("Company Management").classes("text-2xl font-bold")

        # --- Add company ---
        with ui.card().classes("w-full"):
            ui.label("Add Company").classes("text-lg font-bold")
            ui.label(
                "Search from the stocks database or type a ticker directly."
            ).classes("text-sm text-gray-500")

            with ui.row().classes("items-end gap-4 flex-wrap w-full"):
                search_input = (
                    ui.input(
                        "Search ticker or company name",
                        placeholder="e.g. VNM or Vinamilk",
                    )
                    .props("dense clearable")
                    .classes("w-64")
                )

                ui.button(
                    "Search",
                    on_click=lambda: search_results.refresh(),
                ).props("dense")

                # Direct add
                direct_input = (
                    ui.input("Or add ticker directly", placeholder="e.g. BVL")
                    .props("dense")
                    .classes("w-40")
                )

                def _direct_add():
                    val = direct_input.value
                    if val and val.strip():
                        con = get_connection()
                        try:
                            ensure_company(con, val.strip())
                        finally:
                            con.close()
                        direct_input.value = ""
                        ui.notify(f"Added {val.strip().upper()}")
                        company_table.refresh()

                ui.button("Add", on_click=_direct_add).props("dense")

            @ui.refreshable
            def search_results():
                q = (search_input.value or "").strip()
                if not q:
                    return
                con = get_connection()
                try:
                    rows = con.execute(
                        "SELECT s.code, s.floor, s.company_name, "
                        "  CASE WHEN c.ticker IS NOT NULL THEN TRUE ELSE FALSE END AS added "
                        "FROM stocks s "
                        "LEFT JOIN companies c ON s.code = c.ticker "
                        "WHERE s.code LIKE ? OR LOWER(s.company_name) LIKE LOWER(?) "
                        "ORDER BY s.code LIMIT 50",
                        [f"%{q.upper()}%", f"%{q}%"],
                    ).fetchall()
                finally:
                    con.close()

                if not rows:
                    ui.label("No matching stocks found.").classes(
                        "text-gray-500 text-sm"
                    )
                    return

                columns = [
                    {
                        "name": "code",
                        "label": "Ticker",
                        "field": "code",
                        "align": "left",
                    },
                    {
                        "name": "floor",
                        "label": "Exchange",
                        "field": "floor",
                        "align": "center",
                    },
                    {
                        "name": "company",
                        "label": "Company Name",
                        "field": "company",
                        "align": "left",
                    },
                    {
                        "name": "status",
                        "label": "Status",
                        "field": "status",
                        "align": "center",
                    },
                ]
                data = [
                    {
                        "code": r[0],
                        "floor": r[1] or "",
                        "company": r[2] or "",
                        "status": "✅ Added" if r[3] else "",
                    }
                    for r in rows
                ]

                ui.label(f"{len(data)} results").classes(
                    "text-xs text-gray-500"
                )

                table = (
                    ui.table(
                        columns=columns,
                        rows=data,
                        row_key="code",
                        selection="multiple",
                    )
                    .classes("w-full")
                    .props("dense flat")
                )

                def _add_selected():
                    selected = table.selected
                    if not selected:
                        ui.notify("No rows selected", type="warning")
                        return
                    con = get_connection()
                    try:
                        for row in selected:
                            ensure_company(con, row["code"])
                    finally:
                        con.close()
                    tickers = ", ".join(r["code"] for r in selected)
                    ui.notify(f"Added {tickers}")
                    table.selected.clear()
                    search_results.refresh()
                    company_table.refresh()

                ui.button("Add Selected", on_click=_add_selected).props(
                    "dense color=primary"
                )

            search_results()

        # --- Current companies ---
        with ui.card().classes("w-full"):
            ui.label("Active Companies").classes("text-lg font-bold")
            ui.label(
                "These companies are used by all pipeline jobs. "
                "Removing a company deletes all its data."
            ).classes("text-sm text-gray-500")

            @ui.refreshable
            def company_table():
                con = get_connection()
                try:
                    rows = con.execute(
                        """
                        SELECT
                            c.ticker,
                            s.floor,
                            s.company_name,
                            (SELECT COUNT(*) FROM financial_statements fs WHERE fs.code = c.ticker),
                            (SELECT COUNT(*) FROM financial_ratios fr WHERE fr.code = c.ticker),
                            (SELECT COUNT(*) FROM vietstock_documents vd WHERE vd.ticker = c.ticker),
                            (SELECT COUNT(*) FROM conversion_jobs cj WHERE cj.ticker = c.ticker),
                            (SELECT COUNT(*) FROM annual_reports ar WHERE ar.ticker = c.ticker)
                        FROM companies c
                        LEFT JOIN stocks s ON c.ticker = s.code
                        ORDER BY c.ticker
                        """
                    ).fetchall()
                finally:
                    con.close()

                if not rows:
                    ui.label("No companies added yet.").classes(
                        "text-gray-500 text-sm"
                    )
                    return

                columns = [
                    {
                        "name": "ticker",
                        "label": "Ticker",
                        "field": "ticker",
                        "align": "left",
                    },
                    {
                        "name": "exchange",
                        "label": "Exchange",
                        "field": "exchange",
                        "align": "center",
                    },
                    {
                        "name": "name",
                        "label": "Company Name",
                        "field": "name",
                        "align": "left",
                    },
                    {
                        "name": "statements",
                        "label": "Statements",
                        "field": "statements",
                        "align": "right",
                    },
                    {
                        "name": "ratios",
                        "label": "Ratios",
                        "field": "ratios",
                        "align": "right",
                    },
                    {
                        "name": "documents",
                        "label": "Documents",
                        "field": "documents",
                        "align": "right",
                    },
                    {
                        "name": "jobs",
                        "label": "Conv. Jobs",
                        "field": "jobs",
                        "align": "right",
                    },
                    {
                        "name": "reports",
                        "label": "Reports",
                        "field": "reports",
                        "align": "right",
                    },
                    {
                        "name": "action",
                        "label": "",
                        "field": "action",
                        "align": "center",
                    },
                ]
                data = [
                    {
                        "ticker": r[0],
                        "exchange": r[1] or "—",
                        "name": r[2] or "—",
                        "statements": f"{r[3]:,}",
                        "ratios": f"{r[4]:,}",
                        "documents": f"{r[5]:,}",
                        "jobs": f"{r[6]:,}",
                        "reports": f"{r[7]:,}",
                    }
                    for r in rows
                ]
                ui.label(f"{len(data)} companies").classes(
                    "text-xs text-gray-500"
                )

                tbl = (
                    ui.table(
                        columns=columns,
                        rows=data,
                        row_key="ticker",
                    )
                    .classes("w-full")
                    .props("dense flat")
                )

                # Add delete button per row via slot
                tbl.add_slot(
                    "body-cell-action",
                    r"""
                    <q-td :props="props">
                        <q-btn flat dense round icon="delete" color="negative" size="sm"
                               @click="$parent.$emit('delete', props.row)" />
                    </q-td>
                    """,
                )

                def _on_delete(e):
                    ticker = e.args["ticker"]

                    async def _do_delete():
                        c = get_connection()
                        try:
                            result = delete_company(c, ticker)
                        finally:
                            c.close()
                        details = ", ".join(
                            f"{k}: {v}"
                            for k, v in result.items()
                            if v > 0 and k != "companies"
                        )
                        msg = f"Deleted {ticker}"
                        if details:
                            msg += f" ({details})"
                        ui.notify(msg)
                        company_table.refresh()

                    with ui.dialog() as dlg, ui.card():
                        ui.label(f"Delete {ticker}?").classes(
                            "text-lg font-bold"
                        )
                        ui.label(
                            "This will permanently remove all related data: "
                            "financial statements, ratios, documents, "
                            "conversion jobs, reports, and files on disk."
                        ).classes("text-sm text-gray-600")
                        with ui.row().classes("justify-end gap-2 w-full"):
                            ui.button("Cancel", on_click=dlg.close).props(
                                "flat"
                            )

                            async def _confirm(d=dlg):
                                d.close()
                                await _do_delete()

                            ui.button("Delete", on_click=_confirm).props(
                                "color=negative"
                            )
                    dlg.open()

                tbl.on("delete", _on_delete)

            company_table()


# ---------------------------------------------------------------------------
# Jobs page — run selected tasks with progress
# ---------------------------------------------------------------------------


@ui.page("/jobs")
def page_jobs(
    jobs: str = "",
    exchanges: str = "",
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
):
    ui.dark_mode(False)
    _nav_header()
    init_db()

    job_keys = [j for j in jobs.split(",") if j] if jobs else []
    exchange_list = [e for e in exchanges.split(",") if e] if exchanges else []

    job_label_map = dict(_ALL_JOBS)

    states: dict[str, TaskState] = {}
    refreshers: dict[str, Callable] = {}

    with ui.column().classes("w-full max-w-5xl mx-auto gap-4 p-4"):
        ui.label("Job Runner").classes("text-2xl font-bold")

        if exchange_list:
            ui.label(
                f"Exchanges: {', '.join(exchange_list)} | "
                f"Years: {start_year}–{end_year}"
            ).classes("text-sm text-gray-600")

        if not job_keys:
            ui.label(
                "No jobs selected. Go to Home to configure and select jobs."
            ).classes("text-gray-500")
            return

        get_sy = lambda: start_year
        get_ey = lambda: end_year

        force_toggle = ui.switch("Force re-run (ignore existing data)").props(
            "dense"
        )
        get_force = lambda: force_toggle.value

        make_fns: dict[str, Callable] = {
            "stocks": lambda s, r: _make_sync_stocks(s, r, force=get_force),
            "models": lambda s, r: _make_sync_models(s, r, force=get_force),
            "statements": lambda s, r: _make_sync_statements(
                s, get_sy, get_ey, r, force=get_force
            ),
            "ratios": lambda s, r: _make_sync_ratios(
                s, get_sy, get_ey, r, force=get_force
            ),
            "listings": lambda s, r: _make_sync_listings(
                s, get_sy, get_ey, r, force=get_force
            ),
            "download": lambda s, r: _make_download_pdfs(
                s, r, force=get_force
            ),
            "convert": lambda s, r: _make_convert(
                s, r, start_year=get_sy, end_year=get_ey, force=get_force
            ),
            "load": lambda s, r: _make_load(s, r, force=get_force),
        }

        for key in job_keys:
            if key not in make_fns:
                continue
            label = job_label_map.get(key, key)
            st = TaskState()
            states[key] = st
            mk = make_fns[key]

            @ui.refreshable
            def panel(_title=label, _state=st, _mk=mk, _self=None):
                task_card(
                    _title,
                    _state,
                    _mk(_state, panel.refresh),
                    panel.refresh,
                )

            panel()
            refreshers[key] = panel.refresh

        ui.separator()

        def run_all_selected():
            def _run_all():
                for key in job_keys:
                    if key not in states:
                        continue
                    st = states[key]
                    rfn = refreshers.get(key, lambda: None)
                    mk = make_fns.get(key)
                    if not mk:
                        continue
                    st.running = True
                    st.rows.clear()
                    st.summary = ""
                    st.error = ""
                    rfn()
                    fn = mk(st, rfn)
                    try:
                        fn()
                    except Exception as exc:
                        st.error = str(exc)
                    finally:
                        st.running = False
                        st.last_run = datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    rfn()

            if not _pipeline_lock.acquire(blocking=False):
                ui.notify("Another task is already running", type="warning")
                return
            _pipeline_lock.release()
            _run_in_thread(_run_all, TaskState(), lambda: None)

        with ui.row().classes("gap-2"):
            ui.button(
                "Run All Selected",
                on_click=run_all_selected,
                color="primary",
            )
            ui.button("Back to Home", on_click=lambda: ui.navigate.to("/"))


# ---------------------------------------------------------------------------
# Browse pages
# ---------------------------------------------------------------------------


@ui.page("/browse/stocks")
def page_browse_stocks():
    ui.dark_mode(False)
    _nav_header()
    init_db()

    with ui.column().classes("w-full max-w-5xl mx-auto gap-4 p-4"):
        ui.label("Stocks Browser").classes("text-2xl font-bold")

        with ui.row().classes("items-end gap-2"):
            search = ui.input("Search ticker", placeholder="e.g. VNM").props(
                "dense clearable"
            )
            exchange_filter = (
                ui.select(
                    label="Exchange",
                    options=["All", "HOSE", "HNX", "UPCOM"],
                    value="All",
                )
                .props("dense outlined")
                .classes("w-36")
            )
            ui.button("Search", on_click=lambda: stocks_table.refresh()).props(
                "dense flat"
            )

        @ui.refreshable
        def stocks_table():
            con = get_connection()
            try:
                q = (search.value or "").strip().upper()
                exch = exchange_filter.value
                conditions = []
                params = []
                if q:
                    conditions.append("code LIKE ?")
                    params.append(f"%{q}%")
                if exch and exch != "All":
                    conditions.append("floor = ?")
                    params.append(exch)
                where = (
                    "WHERE " + " AND ".join(conditions) if conditions else ""
                )
                rows = con.execute(
                    f"SELECT code, floor, status, company_name "
                    f"FROM stocks {where} ORDER BY code LIMIT 200",
                    params,
                ).fetchall()
            finally:
                con.close()

            if not rows:
                ui.label("No stocks found").classes("text-gray-500")
                return

            columns = [
                {
                    "name": "code",
                    "label": "Code",
                    "field": "code",
                    "align": "left",
                },
                {
                    "name": "floor",
                    "label": "Floor",
                    "field": "floor",
                    "align": "center",
                },
                {
                    "name": "status",
                    "label": "Status",
                    "field": "status",
                    "align": "center",
                },
                {
                    "name": "company",
                    "label": "Company",
                    "field": "company",
                    "align": "left",
                },
            ]
            data = [
                {
                    "code": r[0],
                    "floor": r[1],
                    "status": r[2],
                    "company": r[3],
                }
                for r in rows
            ]
            ui.label(f"{len(data)} results").classes("text-xs text-gray-500")
            ui.table(columns=columns, rows=data, row_key="code").classes(
                "w-full"
            ).props("dense flat")

        stocks_table()


@ui.page("/browse/financial")
def page_browse_financial():
    ui.dark_mode(False)
    _nav_header()
    init_db()

    def _build_statement_table(ticker: str, model_type: str, con) -> None:
        """Build a pivot table for a financial statement type."""
        # Get available fiscal dates for this ticker + model_type
        dates = con.execute(
            "SELECT DISTINCT fiscal_date FROM financial_statements "
            "WHERE code = ? AND model_type = ? "
            "ORDER BY fiscal_date DESC",
            [ticker, model_type],
        ).fetchall()
        if not dates:
            ui.label(f"No data for {ticker}").classes("text-gray-500")
            return

        fiscal_dates = [d[0] for d in dates]

        # Pivot: join statements with models, group by item
        rows = con.execute(
            """
            SELECT fm.item_code, fm.item_vn_name, fm.item_en_name,
                   fm.display_order, fm.display_level,
                   fs.fiscal_date, fs.numeric_value
            FROM financial_models fm
            LEFT JOIN financial_statements fs
                ON fs.item_code = fm.item_code
                AND fs.model_type = fm.model_type
                AND fs.code = ?
            WHERE fm.model_type = ?
            ORDER BY fm.display_order, fs.fiscal_date DESC
            """,
            [ticker, model_type],
        ).fetchall()

        if not rows:
            ui.label("No data").classes("text-gray-500")
            return

        # Build pivot dict: item_code -> {fiscal_date: value}
        items: dict[str, dict] = {}
        for item_code, vn_name, en_name, order, level, fdate, val in rows:
            if item_code not in items:
                items[item_code] = {
                    "name": en_name or vn_name or "",
                    "order": order,
                    "level": level,
                    "values": {},
                }
            if fdate:
                items[item_code]["values"][fdate] = val

        sorted_items = sorted(items.values(), key=lambda x: x["order"])

        # Build columns: item name + year columns
        columns = [
            {
                "name": "item",
                "label": "Item",
                "field": "item",
                "align": "left",
                "headerStyle": "width: 300px; min-width: 300px",
                "style": "width: 300px; min-width: 300px",
            },
        ]
        for fd in fiscal_dates:
            year = fd[:4]
            columns.append(
                {
                    "name": f"y{year}",
                    "label": year,
                    "field": f"y{year}",
                    "align": "right",
                }
            )

        def _fmt(val):
            """Format value in billions VND."""
            if val is None or val == 0:
                return ""
            v = val / 1_000_000_000
            if abs(v) < 1:
                return f"{v:,.1f}"
            return f"{int(round(v)):,}"

        table_rows = []
        for item in sorted_items:
            level = item["level"]
            indent = "\u2003" * level  # em-space indentation
            prefix = "-" if level <= 1 else "+"
            if level == 0:
                prefix = ""
            name = f"{indent}{prefix}{item['name']}"

            row = {"item": name, "_level": level}
            for fd in fiscal_dates:
                year = fd[:4]
                row[f"y{year}"] = _fmt(item["values"].get(fd))
            table_rows.append(row)

        # Use HTML table for proper styling (bold headers, indentation)
        with ui.element("div").classes("w-full overflow-x-auto"):
            html_parts = [
                '<table style="width:100%; border-collapse:collapse; '
                'font-size:13px; font-family:monospace">'
            ]
            # Header
            html_parts.append("<thead><tr>")
            html_parts.append(
                '<th style="text-align:left; padding:6px 8px; '
                'border-bottom:2px solid #ccc; min-width:320px">'
                "Item</th>"
            )
            for fd in fiscal_dates:
                html_parts.append(
                    f'<th style="text-align:right; padding:6px 8px; '
                    f'border-bottom:2px solid #ccc; min-width:90px">'
                    f"{fd[:4]}</th>"
                )
            html_parts.append("</tr></thead><tbody>")

            for row in table_rows:
                level = row["_level"]
                is_bold = level <= 1
                indent_px = level * 16
                weight = "bold" if is_bold else "normal"
                bg = "#f5f5f5" if level == 0 else ""
                bg_style = f"background:{bg};" if bg else ""

                html_parts.append(f'<tr style="{bg_style}">')
                # Item name cell
                item_name = row["item"].replace("\u2003", "")
                prefix = ""
                if level == 0:
                    prefix = ""
                elif level <= 1:
                    prefix = ""
                else:
                    prefix = ""
                html_parts.append(
                    f'<td style="padding:3px 8px 3px {indent_px + 8}px; '
                    f"font-weight:{weight}; border-bottom:1px solid #eee; "
                    f'white-space:nowrap">{item_name.strip()}</td>'
                )
                # Value cells
                for fd in fiscal_dates:
                    year = fd[:4]
                    val = row.get(f"y{year}", "")
                    html_parts.append(
                        f'<td style="text-align:right; padding:3px 8px; '
                        f"font-weight:{weight}; "
                        f'border-bottom:1px solid #eee">{val}</td>'
                    )
                html_parts.append("</tr>")

            html_parts.append("</tbody></table>")
            ui.html("\n".join(html_parts))

    def _build_ratios_table(ticker: str, con) -> None:
        """Build a pivot table for financial ratios grouped by ratio_group."""
        dates = con.execute(
            "SELECT DISTINCT report_date FROM financial_ratios "
            "WHERE code = ? ORDER BY report_date DESC",
            [ticker],
        ).fetchall()
        if not dates:
            ui.label(f"No ratio data for {ticker}").classes("text-gray-500")
            return

        report_dates = [d[0] for d in dates]
        years = sorted(set(d[:4] for d in report_dates if d), reverse=True)

        rows = con.execute(
            """
            SELECT ratio_group, ratio_code, item_name, report_date, value
            FROM financial_ratios
            WHERE code = ?
            ORDER BY ratio_group, ratio_code, report_date DESC
            """,
            [ticker],
        ).fetchall()

        if not rows:
            ui.label("No data").classes("text-gray-500")
            return

        from collections import OrderedDict

        groups: dict[str, dict[str, dict]] = OrderedDict()
        for ratio_group, ratio_code, item_name, report_date, value in rows:
            grp = ratio_group or "Other"
            if grp not in groups:
                groups[grp] = {}
            if ratio_code not in groups[grp]:
                groups[grp][ratio_code] = {
                    "name": item_name or ratio_code or "",
                    "values": {},
                }
            if report_date:
                yr = report_date[:4]
                if yr not in groups[grp][ratio_code]["values"]:
                    groups[grp][ratio_code]["values"][yr] = value

        def _fmt_ratio(val):
            if val is None:
                return ""
            if abs(val) >= 1000:
                return f"{val:,.0f}"
            if abs(val) >= 1:
                return f"{val:,.2f}"
            return f"{val:,.4f}"

        with ui.element("div").classes("w-full overflow-x-auto"):
            html = [
                '<table style="width:100%; border-collapse:collapse; '
                'font-size:13px; font-family:monospace">'
            ]
            html.append("<thead><tr>")
            html.append(
                '<th style="text-align:left; padding:6px 8px; '
                'border-bottom:2px solid #ccc; min-width:280px">Ratio</th>'
            )
            for yr in years:
                html.append(
                    f'<th style="text-align:right; padding:6px 8px; '
                    f'border-bottom:2px solid #ccc; min-width:90px">{yr}</th>'
                )
            html.append("</tr></thead><tbody>")

            for grp_name, items in groups.items():
                col_span = 1 + len(years)
                html.append(
                    f'<tr style="background:#e8e8e8">'
                    f'<td colspan="{col_span}" style="padding:6px 8px; '
                    f'font-weight:bold; border-bottom:1px solid #ccc">'
                    f"{grp_name}</td></tr>"
                )
                for ratio_code, info in items.items():
                    html.append("<tr>")
                    html.append(
                        f'<td style="padding:3px 8px 3px 24px; '
                        f'border-bottom:1px solid #eee; white-space:nowrap">'
                        f'{info["name"]}</td>'
                    )
                    for yr in years:
                        val = _fmt_ratio(info["values"].get(yr))
                        html.append(
                            f'<td style="text-align:right; padding:3px 8px; '
                            f'border-bottom:1px solid #eee">{val}</td>'
                        )
                    html.append("</tr>")

            html.append("</tbody></table>")
            ui.html("\n".join(html))

    with ui.column().classes("w-full max-w-6xl mx-auto gap-4 p-4"):
        ui.label("Financial Data Browser").classes("text-2xl font-bold")
        ui.label("Values in billions VND").classes(
            "text-sm text-gray-500 -mt-2"
        )

        with ui.row().classes("items-end gap-2"):
            ticker_sel = ui.input(
                "Ticker", value="VNM", placeholder="e.g. VNM"
            ).props("dense")
            ui.button("Load", on_click=lambda: fin_tabs.refresh()).props(
                "dense"
            )

        @ui.refreshable
        def fin_tabs():
            t = (ticker_sel.value or "").strip().upper()
            if not t:
                ui.label("Enter a ticker above and click Load").classes(
                    "text-gray-500"
                )
                return

            con = get_connection()
            try:
                with ui.tabs().classes("w-full") as tabs:
                    tab_bs = ui.tab("balance_sheet", label="Balance Sheet")
                    tab_is = ui.tab("income", label="Income Statement")
                    tab_cf = ui.tab("cashflow", label="Cash Flow Statement")
                    tab_rt = ui.tab("ratios", label="Financial Ratios")

                with ui.tab_panels(tabs, value="balance_sheet").classes(
                    "w-full"
                ):
                    with ui.tab_panel("balance_sheet"):
                        _build_statement_table(t, "1.0", con)
                    with ui.tab_panel("income"):
                        _build_statement_table(t, "2.0", con)
                    with ui.tab_panel("cashflow"):
                        _build_statement_table(t, "3.0", con)
                    with ui.tab_panel("ratios"):
                        _build_ratios_table(t, con)
            finally:
                con.close()

        fin_tabs()


@ui.page("/browse/documents")
def page_browse_documents():
    ui.dark_mode(False)
    _nav_header()
    init_db()

    with ui.column().classes("w-full max-w-5xl mx-auto gap-4 p-4"):
        ui.label("Vietstock Documents Browser").classes("text-2xl font-bold")

        with ui.row().classes("items-end gap-2"):
            ticker_sel = ui.input(
                "Filter by ticker", placeholder="e.g. VNM"
            ).props("dense clearable")
            ui.button("Search", on_click=lambda: doc_table.refresh()).props(
                "dense flat"
            )

        @ui.refreshable
        def doc_table():
            con = get_connection()
            try:
                t = (ticker_sel.value or "").strip().upper()
                if t:
                    rows = con.execute(
                        "SELECT id, ticker, title, published_date, "
                        "synced_to_raw, raw_path "
                        "FROM vietstock_documents WHERE ticker = ? "
                        "ORDER BY published_date DESC LIMIT 200",
                        [t],
                    ).fetchall()
                else:
                    rows = con.execute(
                        "SELECT id, ticker, title, published_date, "
                        "synced_to_raw, raw_path "
                        "FROM vietstock_documents "
                        "ORDER BY ticker, published_date DESC LIMIT 200"
                    ).fetchall()
            finally:
                con.close()

            if not rows:
                ui.label("No documents found").classes("text-gray-500")
                return

            import re as _re

            columns = [
                {
                    "name": "ticker",
                    "label": "Ticker",
                    "field": "ticker",
                    "align": "left",
                },
                {
                    "name": "title",
                    "label": "Title",
                    "field": "title",
                    "align": "left",
                },
                {
                    "name": "year",
                    "label": "Year",
                    "field": "year",
                    "align": "center",
                },
                {
                    "name": "date",
                    "label": "Published",
                    "field": "date",
                    "align": "center",
                },
                {
                    "name": "synced",
                    "label": "Downloaded",
                    "field": "synced",
                    "align": "center",
                },
                {
                    "name": "action",
                    "label": "",
                    "field": "action",
                    "align": "center",
                },
            ]
            data = []
            for r in rows:
                m = _re.search(r"(\d{4})", r[2] or "")
                year = m.group(1) if m else ""
                data.append(
                    {
                        "id": r[0],
                        "ticker": r[1],
                        "title": r[2] or "",
                        "year": year,
                        "date": r[3] or "",
                        "synced": "✅" if r[4] else "❌",
                        "raw_path": r[5] or "",
                    }
                )
            ui.label(f"{len(data)} documents").classes("text-xs text-gray-500")

            tbl = (
                ui.table(
                    columns=columns,
                    rows=data,
                    row_key="id",
                    selection="multiple",
                )
                .classes("w-full")
                .props("dense flat")
            )

            # Per-row delete button
            tbl.add_slot(
                "body-cell-action",
                r"""
                <q-td :props="props">
                    <q-btn flat dense round icon="delete" color="negative" size="sm"
                           @click="$parent.$emit('delete', props.row)" />
                </q-td>
                """,
            )

            def _delete_doc(e):
                doc = e.args
                doc_id = doc["id"]
                ticker = doc["ticker"]
                title = doc["title"]

                async def _do_delete():
                    c = get_connection()
                    try:
                        # Delete related conversion job if exists
                        m = _re.search(r"(\d{4})", title or "")
                        if m:
                            year = int(m.group(1))
                            c.execute(
                                "DELETE FROM conversion_jobs WHERE ticker = ? AND year = ?",
                                [ticker, year],
                            )
                        # Delete the raw file if it exists
                        raw_path = doc.get("raw_path", "")
                        if raw_path:
                            from pathlib import Path

                            p = Path(raw_path)
                            if p.exists():
                                p.unlink()
                        c.execute(
                            "DELETE FROM vietstock_documents WHERE id = ?",
                            [doc_id],
                        )
                    finally:
                        c.close()
                    ui.notify(f"Deleted: {title}")
                    doc_table.refresh()

                with ui.dialog() as dlg, ui.card():
                    ui.label(f"Delete document?").classes("text-lg font-bold")
                    ui.label(f"{ticker} — {title}").classes(
                        "text-sm text-gray-600"
                    )
                    with ui.row().classes("justify-end gap-2 w-full"):
                        ui.button("Cancel", on_click=dlg.close).props("flat")

                        async def _confirm(d=dlg):
                            d.close()
                            await _do_delete()

                        ui.button("Delete", on_click=_confirm).props(
                            "color=negative"
                        )
                dlg.open()

            tbl.on("delete", _delete_doc)

            # Bulk delete selected
            def _delete_selected():
                selected = tbl.selected
                if not selected:
                    ui.notify("No rows selected", type="warning")
                    return

                async def _do_bulk_delete():
                    c = get_connection()
                    try:
                        for doc in selected:
                            m = _re.search(r"(\d{4})", doc.get("title", ""))
                            if m:
                                year = int(m.group(1))
                                c.execute(
                                    "DELETE FROM conversion_jobs WHERE ticker = ? AND year = ?",
                                    [doc["ticker"], year],
                                )
                            raw_path = doc.get("raw_path", "")
                            if raw_path:
                                from pathlib import Path

                                p = Path(raw_path)
                                if p.exists():
                                    p.unlink()
                            c.execute(
                                "DELETE FROM vietstock_documents WHERE id = ?",
                                [doc["id"]],
                            )
                    finally:
                        c.close()
                    ui.notify(f"Deleted {len(selected)} document(s)")
                    tbl.selected.clear()
                    doc_table.refresh()

                with ui.dialog() as dlg, ui.card():
                    ui.label(f"Delete {len(selected)} documents?").classes(
                        "text-lg font-bold"
                    )
                    ui.label(
                        "This will also delete associated conversion jobs and raw files."
                    ).classes("text-sm text-gray-600")
                    with ui.row().classes("justify-end gap-2 w-full"):
                        ui.button("Cancel", on_click=dlg.close).props("flat")

                        async def _confirm(d=dlg):
                            d.close()
                            await _do_bulk_delete()

                        ui.button("Delete", on_click=_confirm).props(
                            "color=negative"
                        )
                dlg.open()

            ui.button("Delete Selected", on_click=_delete_selected).props(
                "dense color=negative"
            )

        doc_table()


@ui.page("/browse/reports")
def page_browse_reports():
    ui.dark_mode(False)
    _nav_header()
    init_db()

    with ui.column().classes("w-full max-w-5xl mx-auto gap-4 p-4"):
        ui.label("Annual Reports Browser").classes("text-2xl font-bold")

        ui.button("Refresh", on_click=lambda: reports_table.refresh()).props(
            "dense flat"
        )

        @ui.refreshable
        def reports_table():
            con = get_connection()
            try:
                rows = con.execute(
                    "SELECT ticker, year, LENGTH(content) as chars, "
                    "source_file, created_at "
                    "FROM annual_reports ORDER BY ticker, year DESC"
                ).fetchall()
            finally:
                con.close()

            if not rows:
                ui.label("No reports loaded yet").classes("text-gray-500")
                return

            columns = [
                {
                    "name": "ticker",
                    "label": "Ticker",
                    "field": "ticker",
                    "align": "left",
                },
                {
                    "name": "year",
                    "label": "Year",
                    "field": "year",
                    "align": "center",
                },
                {
                    "name": "chars",
                    "label": "Content Length",
                    "field": "chars",
                    "align": "right",
                },
                {
                    "name": "source",
                    "label": "Source File",
                    "field": "source",
                    "align": "left",
                },
            ]
            data = [
                {
                    "ticker": r[0],
                    "year": r[1],
                    "chars": f"{r[2]:,}",
                    "source": r[3],
                }
                for r in rows
            ]
            ui.table(
                columns=columns, rows=data, row_key="ticker_year"
            ).classes("w-full").props("dense flat")

        reports_table()


# ---------------------------------------------------------------------------
# Converter page — track marker-pdf conversion jobs
# ---------------------------------------------------------------------------

_STATUS_ICONS = {
    "pending": "⬜",
    "running": "⏳",
    "completed": "✅",
    "failed": "❌",
    "cancelled": "🚫",
}


@ui.page("/converter")
def page_converter():
    ui.dark_mode(False)
    _nav_header()
    init_db()

    with ui.column().classes("w-full max-w-5xl mx-auto gap-4 p-4"):
        ui.label("Converter Management").classes("text-2xl font-bold")

        create_job_state: dict[str, list[str]] = {"tickers": []}
        current_year = datetime.now().year
        converter_years = list(range(2010, current_year + 1))

        # Create conversion jobs
        with ui.card().classes("w-full"):
            ui.label("Create Conversion Jobs").classes("text-lg font-bold")
            ui.label(
                "Select companies with downloaded PDFs and create or refresh their conversion jobs."
            ).classes("text-sm text-gray-500")

            with ui.row().classes("items-end gap-4 flex-wrap w-full"):
                ticker_sel = (
                    ui.select(
                        label="Companies",
                        options={},
                        multiple=True,
                        with_input=True,
                    )
                    .props(
                        "dense options-dense use-chips outlined "
                        'style="min-width: 340px"'
                    )
                    .classes("flex-1")
                )

                create_start_year = (
                    ui.select(
                        label="Start year",
                        options=converter_years,
                        value=DEFAULT_START_YEAR,
                    )
                    .props("dense outlined")
                    .classes("w-32")
                )

                create_end_year = (
                    ui.select(
                        label="End year",
                        options=converter_years,
                        value=DEFAULT_END_YEAR,
                    )
                    .props("dense outlined")
                    .classes("w-32")
                )

            create_jobs_summary = ui.label("").classes("text-sm text-gray-500")

            @ui.refreshable
            def create_job_picker():
                con = get_connection()
                try:
                    rows = con.execute(
                        """
                        SELECT
                            c.ticker,
                            (
                                SELECT COUNT(*)
                                FROM vietstock_documents vd
                                WHERE vd.ticker = c.ticker
                                  AND vd.synced_to_raw = TRUE
                                  AND vd.raw_path IS NOT NULL
                                  AND vd.raw_path != ''
                            ) AS synced_docs,
                            (
                                SELECT COUNT(*)
                                FROM conversion_jobs cj
                                WHERE cj.ticker = c.ticker
                            ) AS job_count
                        FROM companies c
                        ORDER BY c.ticker
                        """
                    ).fetchall()
                finally:
                    con.close()

                options = {
                    ticker: (
                        f"{ticker}"
                        + (
                            f" ({synced_docs} synced PDF{'s' if synced_docs != 1 else ''}, {job_count} job{'s' if job_count != 1 else ''})"
                        )
                    )
                    for ticker, synced_docs, job_count in rows
                    if synced_docs > 0
                }

                selected = [
                    ticker
                    for ticker in create_job_state["tickers"]
                    if ticker in options
                ]
                create_job_state["tickers"] = selected
                ticker_sel.options = options
                ticker_sel.value = selected
                ticker_sel.update()

                available = len(options)
                if available == 0:
                    create_jobs_summary.text = (
                        "No companies with downloaded PDFs are available yet."
                    )
                else:
                    create_jobs_summary.text = f"{available} companies available for conversion job creation"

            def _sync_selected_tickers(_=None):
                value = ticker_sel.value or []
                create_job_state["tickers"] = list(value)

            ticker_sel.on_value_change(_sync_selected_tickers)

            def _create_selected_jobs():
                selected = list(ticker_sel.value or [])
                if not selected:
                    ui.notify("Select at least one company", type="warning")
                    return

                start_year = int(create_start_year.value or DEFAULT_START_YEAR)
                end_year = int(create_end_year.value or DEFAULT_END_YEAR)
                if start_year > end_year:
                    ui.notify(
                        "Start year must be less than or equal to end year",
                        type="warning",
                    )
                    return

                from converter import create_jobs

                con = get_connection()
                try:
                    count = create_jobs(
                        con,
                        tickers=selected,
                        start_year=start_year,
                        end_year=end_year,
                    )
                finally:
                    con.close()

                ui.notify(
                    f"Created or refreshed {count} conversion job(s) for {', '.join(selected)}"
                )
                jobs_table.refresh()
                create_job_picker.refresh()

            with ui.row().classes("gap-2 mt-2 flex-wrap"):
                ui.button(
                    "Create Jobs for Selected Companies",
                    on_click=_create_selected_jobs,
                    color="primary",
                ).props("dense")
                ui.button(
                    "Refresh Company Options",
                    on_click=create_job_picker.refresh,
                ).props("dense outline")

            create_job_picker()

        # Resync input directory
        with ui.card().classes("w-full"):
            ui.label("Sync Imported Raw & Markdown").classes(
                "text-lg font-bold"
            )
            ui.label(
                "Only raw PDFs at or after the selected year will be added. Older raw PDFs will be listed below for cleanup."
            ).classes("text-sm text-gray-500")
            with ui.row().classes("items-end gap-4 flex-wrap w-full"):
                import_min_year = (
                    ui.select(
                        label="Minimum raw year",
                        options=converter_years,
                        value=DEFAULT_START_YEAR,
                    )
                    .props("dense outlined")
                    .classes("w-40")
                )
            sync_preview_summary = ui.label("").classes("text-sm")
            sync_preview_state: dict[str, list[dict[str, object]]] = {
                "rows": [],
                "older_rows": [],
            }

            @ui.refreshable
            def sync_preview_table():
                rows: list[dict[str, object]] = sync_preview_state["rows"]
                older_rows: list[dict[str, object]] = sync_preview_state[
                    "older_rows"
                ]
                if not rows and not older_rows:
                    ui.label(
                        "No new company/year additions detected."
                    ).classes("text-gray-500 text-sm")
                    return

                if rows:
                    ui.label("Candidates to add or update").classes(
                        "text-sm font-medium"
                    )
                    ui.table(
                        columns=[
                            {
                                "name": "source",
                                "label": "Source",
                                "field": "source",
                                "align": "left",
                            },
                            {
                                "name": "ticker",
                                "label": "Ticker",
                                "field": "ticker",
                                "align": "left",
                            },
                            {
                                "name": "year",
                                "label": "Year",
                                "field": "year",
                                "align": "left",
                            },
                            {
                                "name": "action",
                                "label": "Action",
                                "field": "action",
                                "align": "left",
                            },
                            {
                                "name": "path",
                                "label": "Path",
                                "field": "path",
                                "align": "left",
                            },
                        ],
                        rows=rows,
                        row_key="path",
                    ).classes("w-full").props("dense flat")

                if older_rows:
                    ui.separator()
                    ui.label(
                        f"Raw PDFs before {int(import_min_year.value or DEFAULT_START_YEAR)}"
                    ).classes("text-sm font-medium text-orange-700")
                    ui.table(
                        columns=[
                            {
                                "name": "ticker",
                                "label": "Ticker",
                                "field": "ticker",
                                "align": "left",
                            },
                            {
                                "name": "year",
                                "label": "Year",
                                "field": "year",
                                "align": "left",
                            },
                            {
                                "name": "path",
                                "label": "Path",
                                "field": "path",
                                "align": "left",
                            },
                        ],
                        rows=older_rows,
                        row_key="path",
                    ).classes("w-full").props("dense flat")

            def _preview_imported_files():
                from loader import preview_markdown_sync

                min_year = int(import_min_year.value or DEFAULT_START_YEAR)
                con = get_connection()
                try:
                    init_db(con)
                    raw_preview = preview_raw_input_dir_sync(
                        con, min_year=min_year
                    )
                    markdown_preview = preview_markdown_sync(con)
                finally:
                    con.close()

                raw_candidates = list(raw_preview["candidates"])
                markdown_candidates = list(markdown_preview["candidates"])
                combined_rows = sorted(
                    raw_candidates + markdown_candidates,
                    key=lambda row: (
                        str(row["ticker"]),
                        int(row["year"]),
                        str(row["source"]),
                    ),
                )
                sync_preview_state["rows"] = combined_rows
                sync_preview_state["older_rows"] = list(
                    raw_preview["older_than_min_year"]
                )

                company_parts: list[str] = []
                companies_to_add = sorted(
                    set(raw_preview["companies_to_add"])
                    | set(markdown_preview["companies_to_add"])
                )
                if companies_to_add:
                    company_parts.append(
                        "Companies to add: " + ", ".join(companies_to_add)
                    )

                years_to_add: dict[str, set[int]] = {}
                for source_preview in (raw_preview, markdown_preview):
                    for ticker, years in source_preview[
                        "years_to_add"
                    ].items():
                        years_to_add.setdefault(ticker, set()).update(years)

                if years_to_add:
                    company_parts.append(
                        "Years to add: "
                        + "; ".join(
                            f"{ticker} ({', '.join(str(year) for year in sorted(years))})"
                            for ticker, years in sorted(years_to_add.items())
                        )
                    )

                company_parts.append(
                    f"Raw candidates: {len(raw_candidates)} | Markdown candidates: {len(markdown_candidates)}"
                )
                if raw_preview["older_than_min_year"]:
                    company_parts.append(
                        "Older raw files skipped: "
                        f"{len(raw_preview['older_than_min_year'])}"
                    )
                if raw_preview["nonstandard"]:
                    company_parts.append(
                        f"Non-standard raw files: {len(raw_preview['nonstandard'])}"
                    )

                sync_preview_summary.text = " | ".join(company_parts)
                sync_preview_table.refresh()

            def _sync_imported_files():
                from loader import sync_markdown_files

                min_year = int(import_min_year.value or DEFAULT_START_YEAR)
                con = get_connection()
                try:
                    init_db(con)
                    raw_result = resync_input_dir(con=con, min_year=min_year)
                    markdown_result = sync_markdown_files(con)
                finally:
                    con.close()

                sync_preview_summary.text = (
                    f"Raw cutoff: {min_year}+"
                    f" | Skipped older raw files: {len(raw_result.get('older_than_min_year', []))}"
                    f" | Raw: {raw_result.get('created_companies', 0)} companies, "
                    f"{raw_result.get('created_jobs', 0)} jobs created, "
                    f"{raw_result.get('updated_jobs', 0)} jobs updated, "
                    f"{raw_result.get('updated_documents', 0)} documents updated"
                    f" | Markdown: {markdown_result.get('created_companies', 0)} companies, "
                    f"{markdown_result['loaded']} reports loaded, "
                    f"{markdown_result['failed']} failed"
                )
                ui.notify(sync_preview_summary.text)
                _preview_imported_files()
                create_job_picker.refresh()
                jobs_table.refresh()

            with ui.row().classes("gap-2 mt-2 flex-wrap"):
                ui.button(
                    "Preview Imported Files",
                    on_click=_preview_imported_files,
                    color="secondary",
                ).props("dense")
                ui.button(
                    "Sync Imported Files",
                    on_click=_sync_imported_files,
                    color="primary",
                ).props("dense")

            sync_preview_table()

        # Resync input directory
        with ui.card().classes("w-full"):
            ui.label("Resync Input Directory").classes("text-lg font-bold")
            ui.label(
                "This uses the same minimum raw year filter as the imported raw preview above."
            ).classes("text-sm text-gray-500")
            resync_result = ui.label("").classes("text-sm")

            def do_resync():
                min_year = int(import_min_year.value or DEFAULT_START_YEAR)
                result = resync_input_dir(min_year=min_year)
                msg = (
                    f"Minimum raw year {min_year}: "
                    f"scanned {len(result['added'])} standard raw PDFs, "
                    f"skipped {len(result.get('older_than_min_year', []))} older raw PDFs, "
                    f"created {result.get('created_companies', 0)} companies, "
                    f"created {result.get('created_jobs', 0)} jobs, "
                    f"updated {result.get('updated_documents', 0)} documents, "
                    f"updated {result.get('updated_jobs', 0)} jobs, "
                    f"found {len(result['nonstandard'])} non-standard files"
                )
                resync_result.text = msg
                ui.notify(msg)
                create_job_picker.refresh()
                jobs_table.refresh()

            ui.button("Resync Now", on_click=do_resync).props("color=primary")

        # List non-standard input files
        with ui.card().classes("w-full"):
            ui.label("Non-standard Raw Files").classes("text-lg font-bold")
            nonstandard_state: dict[str, list[str]] = {"items": []}

            @ui.refreshable
            def nonstandard_list():
                if not nonstandard_state["items"]:
                    ui.label("All raw files are standard PDFs.").classes(
                        "text-green-600"
                    )
                    return

                with ui.column().classes("gap-1"):
                    for path in nonstandard_state["items"]:
                        ui.label(path).classes("text-red-600 text-xs")

            def refresh_nonstandard():
                nonstandard_state["items"] = list_nonstandard_input_files()
                nonstandard_list.refresh()

            ui.button(
                "List Non-standard Files", on_click=refresh_nonstandard
            ).props("color=secondary")
            nonstandard_list()
            refresh_nonstandard()

        # --- Run / manage ---
        with ui.card().classes("w-full"):
            ui.label("Run & Manage").classes("text-lg font-bold")

            converter_state = TaskState()

            with ui.row().classes("gap-2 flex-wrap"):

                def _run_pending():
                    from converter import run_pending_jobs

                    def _execute():
                        con = get_connection()
                        try:
                            converter_state.rows.clear()

                            def on_progress(
                                job_id, ticker, year, status, error
                            ):
                                icon = _STATUS_ICONS.get(status, status)
                                row = TaskRow(
                                    label=f"#{job_id} {ticker} {year}",
                                    detail=error[:80] if error else "",
                                    status=(
                                        "done"
                                        if status == "completed"
                                        else "error"
                                    ),
                                )
                                converter_state.rows.append(row)
                                converter_run_panel.refresh()

                            results = run_pending_jobs(
                                con, on_progress=on_progress
                            )
                            converter_state.summary = (
                                f"✅ {results['completed']} completed, "
                                f"{results['failed']} failed "
                                f"(of {results['total']} total)"
                            )
                        finally:
                            con.close()
                        jobs_table.refresh()
                        converter_run_panel.refresh()

                    _run_in_thread(
                        _execute,
                        converter_state,
                        converter_run_panel.refresh,
                    )

                ui.button(
                    "Run Pending Jobs", on_click=_run_pending, color="primary"
                ).props("dense")

                def _reset_failed():
                    from converter import reset_failed_jobs

                    con = get_connection()
                    try:
                        count = reset_failed_jobs(con)
                    finally:
                        con.close()
                    if count == 0:
                        ui.notify(
                            "No failed, cancelled, or running jobs to reset",
                            type="warning",
                        )
                    else:
                        ui.notify(f"Reset {count} job(s) to pending")
                    jobs_table.refresh()

                ui.button("Reset Failed", on_click=_reset_failed).props(
                    "dense outline"
                )

                def _delete_all():
                    from converter import delete_all_jobs

                    con = get_connection()
                    try:
                        count = delete_all_jobs(con)
                    finally:
                        con.close()
                    ui.notify(f"Deleted {count} job(s)")
                    jobs_table.refresh()

                ui.button("Delete All Jobs", on_click=_delete_all).props(
                    "dense outline color=red"
                )

            @ui.refreshable
            def converter_run_panel():
                if converter_state.running:
                    with ui.row().classes("items-center gap-2"):
                        ui.spinner(size="sm")
                        ui.label("Converting...").classes("text-sm")

                if converter_state.error:
                    ui.label(converter_state.error).classes(
                        "text-red-500 text-sm"
                    )

                if converter_state.summary:
                    ui.label(converter_state.summary).classes(
                        "text-green-600 text-sm font-medium"
                    )

                if converter_state.rows:
                    columns = [
                        {
                            "name": "label",
                            "label": "Job",
                            "field": "label",
                            "align": "left",
                        },
                        {
                            "name": "detail",
                            "label": "Detail",
                            "field": "detail",
                            "align": "left",
                        },
                        {
                            "name": "status",
                            "label": "Status",
                            "field": "status",
                            "align": "center",
                        },
                    ]
                    rows_data = [
                        {
                            "label": r.label,
                            "detail": r.detail,
                            "status": {
                                "done": "✅",
                                "running": "⏳",
                                "pending": "⬜",
                                "error": "❌",
                            }.get(r.status, r.status),
                        }
                        for r in converter_state.rows
                    ]
                    ui.table(
                        columns=columns, rows=rows_data, row_key="label"
                    ).classes("w-full").props("dense flat")

            converter_run_panel()

        # --- Jobs table ---
        with ui.card().classes("w-full"):
            ui.label("All Conversion Jobs").classes("text-lg font-bold")

            with ui.row().classes("items-end gap-2"):
                filter_status = (
                    ui.select(
                        label="Status",
                        options=[
                            "all",
                            "pending",
                            "running",
                            "completed",
                            "failed",
                            "cancelled",
                        ],
                        value="all",
                    )
                    .props("dense outlined")
                    .classes("w-36")
                )
                ui.button(
                    "Refresh", on_click=lambda: jobs_table.refresh()
                ).props("dense flat")

            @ui.refreshable
            def jobs_table():
                con = get_connection()
                try:
                    status_filter = filter_status.value
                    if status_filter and status_filter != "all":
                        rows = con.execute(
                            """
                            SELECT id, ticker, year, status, error_message,
                                   failed_step, started_at, completed_at
                            FROM conversion_jobs
                            WHERE status = ?
                            ORDER BY id DESC
                            LIMIT 200
                            """,
                            [status_filter],
                        ).fetchall()
                    else:
                        rows = con.execute(
                            """
                            SELECT id, ticker, year, status, error_message,
                                   failed_step, started_at, completed_at
                            FROM conversion_jobs
                            ORDER BY id DESC
                            LIMIT 200
                            """
                        ).fetchall()
                finally:
                    con.close()

                if not rows:
                    ui.label("No jobs found").classes("text-gray-500")
                    return

                columns = [
                    {
                        "name": "id",
                        "label": "ID",
                        "field": "id",
                        "align": "left",
                    },
                    {
                        "name": "ticker",
                        "label": "Ticker",
                        "field": "ticker",
                        "align": "left",
                    },
                    {
                        "name": "year",
                        "label": "Year",
                        "field": "year",
                        "align": "center",
                    },
                    {
                        "name": "status",
                        "label": "Status",
                        "field": "status",
                        "align": "center",
                    },
                    {
                        "name": "failed_step",
                        "label": "Failed Step",
                        "field": "failed_step",
                        "align": "left",
                    },
                    {
                        "name": "error",
                        "label": "Error",
                        "field": "error",
                        "align": "left",
                    },
                    {
                        "name": "started",
                        "label": "Started",
                        "field": "started",
                        "align": "left",
                    },
                    {
                        "name": "completed",
                        "label": "Completed",
                        "field": "completed",
                        "align": "left",
                    },
                ]
                data = []
                for (
                    job_id,
                    ticker,
                    year,
                    status,
                    err,
                    step,
                    started,
                    completed,
                ) in rows:
                    icon = _STATUS_ICONS.get(status, status)
                    data.append(
                        {
                            "id": job_id,
                            "ticker": ticker,
                            "year": year,
                            "status": f"{icon} {status}",
                            "failed_step": step or "",
                            "error": (err or "")[:100],
                            "started": str(started)[:19] if started else "",
                            "completed": (
                                str(completed)[:19] if completed else ""
                            ),
                        }
                    )

                ui.table(columns=columns, rows=data, row_key="id").classes(
                    "w-full"
                ).props("dense flat")

            jobs_table()

        # --- Job log viewer ---
        with ui.card().classes("w-full"):
            ui.label("Job Log Viewer").classes("text-lg font-bold")

            with ui.row().classes("items-end gap-2"):
                log_job_id = (
                    ui.number("Job ID", value=1).props("dense").classes("w-28")
                )

                def _show_log():
                    log_area.refresh()

                ui.button("View Log", on_click=_show_log).props("dense flat")

            @ui.refreshable
            def log_area():
                from converter import get_job_log

                jid = int(log_job_id.value or 0)
                if jid <= 0:
                    ui.label("Enter a job ID above").classes(
                        "text-gray-500 text-sm"
                    )
                    return

                con = get_connection()
                try:
                    text = get_job_log(con, jid, tail=100)
                finally:
                    con.close()

                if text:
                    ui.code(text).classes(
                        "w-full max-h-96 overflow-auto text-xs"
                    )
                else:
                    ui.label(f"No log found for job #{jid}").classes(
                        "text-gray-500 text-sm"
                    )

            log_area()


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------


def main():
    init_db()
    ui.run(
        title="Annual Report Pipeline",
        host="0.0.0.0",
        port=8080,
        reload=False,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
