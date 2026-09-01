"""NiceGUI dashboard for the Vietnamese Stock Data & Annual Report Pipeline."""

from __future__ import annotations

import csv
import threading
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, TypedDict, cast

from nicegui import ui

from annual_inference_config import get_task_items
from config import (
    DEFAULT_END_YEAR,
    DEFAULT_START_YEAR,
    EMBEDDING_CHUNK_SIZE,
    EMBEDDING_MODEL,
    FINANCIAL_STATEMENT_MARKDOWN_DIR,
    INFERENCE_MODEL,
    INFERENCE_TOP_K,
    MARKDOWN_DIR,
    OUTPUT_DIR,
)
from converter import (
    list_nonstandard_input_files,
    preview_raw_input_dir_sync,
    resync_input_dir,
)
from database import (
    delete_company,
    ensure_company,
    ensure_vss_loaded,
    get_connection,
    init_db,
    sync_companies_from_stocks,
)
from sql_templates import (
    DATA_STUDIO_SQL_TEMPLATES,
    DEFAULT_DATA_STUDIO_SQL_TEMPLATE,
)

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


class ProcessingStatusRow(TypedDict):
    ticker: str
    year: int
    annual_emb_chunks: int
    financial_statement_emb_chunks: int
    annual_embedded: bool
    financial_statement_embedded: bool
    has_annual: bool
    has_financial_statement: bool
    edc_done: bool
    proper_done: bool
    gov_done: bool
    financial_statement_audit_done: bool


class TargetStatusRow(TypedDict):
    ticker: str
    in_companies: bool
    report_years: int
    embedded_years: int
    edc_jobs: int
    edc_pending: int
    edc_completed: int
    proper_jobs: int
    proper_pending: int
    proper_completed: int
    gov_jobs: int
    gov_pending: int
    gov_completed: int


class SqlConsoleState(TypedDict):
    rows: list[dict[str, str]]
    columns: list[dict[str, str]]
    error: str
    row_count: int


# Global lock to prevent concurrent pipeline runs
_pipeline_lock = threading.Lock()


def _run_in_thread(fn: Callable, state: TaskState, refresh: Callable) -> None:
    """Execute *fn* in a background thread, guarded by the pipeline lock."""
    if not _pipeline_lock.acquire(blocking=False):
        state.error = "Another task is already running"
        ui.notify("Another task is already running", type="warning")
        refresh()
        return

    state.running = True
    state.rows.clear()
    state.summary = ""
    state.error = ""
    state.progress = 0.0
    refresh()

    def _worker():
        try:
            fn()
        except Exception as exc:
            state.error = str(exc)
        finally:
            state.running = False
            state.progress = 0.0
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

        if state.running:
            ui.linear_progress(value=max(0.0, min(1.0, state.progress))).classes(
                "w-full"
            )

        if state.error:
            with ui.row().classes("items-center gap-2"):
                ui.label(state.error).classes("text-red-500 text-sm")
                ui.button(
                    "Clear",
                    on_click=lambda: (setattr(state, "error", ""), refresh()),
                ).props("dense flat")

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
    ticker_filter: Callable[[], list[str] | None] = lambda: None,
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
            selected = ticker_filter() or []
            if selected:
                selected_set = {t.upper() for t in selected}
                tickers = [t for t in tickers if str(t[0]).upper() in selected_set]
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
    ticker_filter: Callable[[], list[str] | None] = lambda: None,
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
            selected = ticker_filter() or []
            if selected:
                selected_set = {t.upper() for t in selected}
                tickers = [t for t in tickers if str(t[0]).upper() in selected_set]
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
    doc_type: Callable[[], str] = lambda: "2",
    ticker_filter: Callable[[], list[str] | None] = lambda: None,
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
            selected = ticker_filter() or []
            if selected:
                selected_set = {t.upper() for t in selected}
                tickers = [t for t in tickers if str(t[0]).upper() in selected_set]
            if not tickers:
                state.summary = "No companies configured"
                return
            # Per-ticker skip: find tickers that already have listings
            selected_doc_type = str(doc_type() or "2")

            existing_tickers = set()
            if not force():
                rows = con.execute(
                    "SELECT DISTINCT ticker FROM vietstock_documents WHERE doc_type = ?",
                    [selected_doc_type],
                ).fetchall()
                existing_tickers = {r[0] for r in rows}

            # Filter to only new tickers (unless force)
            new_tickers = [t for t in tickers if t[0] not in existing_tickers]
            if not new_tickers:
                total_existing = con.execute(
                    "SELECT COUNT(*) FROM vietstock_documents WHERE doc_type = ?",
                    [selected_doc_type],
                ).fetchone()[0]
                state.summary = (
                    f"⏭ Already synced ({total_existing:,} documents for type={selected_doc_type})"
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
                        "SELECT COUNT(*) FROM vietstock_documents WHERE ticker = ? AND doc_type = ?",
                        [t[0], selected_doc_type],
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
                doc_type=selected_doc_type,
                tickers=[t[0] for t in new_tickers],
                start_year=start_year(),
                end_year=end_year(),
                on_progress=on_progress,
            )
            total = sum(results.values())
            ok = sum(1 for v in results.values() if v >= 0)
            skipped = len(existing_tickers)
            state.summary = (
                f"✅ {total:,} documents synced across {ok} tickers (type={selected_doc_type})"
                + (f", {skipped} skipped" if skipped else "")
            )
        finally:
            con.close()

    return run


def _make_download_pdfs(
    state: TaskState,
    refresh: Callable,
    *,
    doc_type: Callable[[], str] = lambda: "2",
    ticker_filter: Callable[[], list[str] | None] = lambda: None,
    force: Callable[[], bool] = lambda: False,
) -> Callable:
    def run():
        from vietstock_documents import download_all_unsynced

        con = get_connection()
        try:
            init_db(con)
            selected = ticker_filter() or []
            selected_doc_type = str(doc_type() or "2")
            sql = (
                "SELECT id, ticker, title FROM vietstock_documents "
                "WHERE synced_to_raw = FALSE AND file_url IS NOT NULL AND file_url != '' "
                "AND doc_type = ? "
            )
            params: list[str] = [selected_doc_type]
            if selected:
                placeholders = ", ".join(["?"] * len(selected))
                sql += f"AND ticker IN ({placeholders}) "
                params.extend([t.upper() for t in selected])
            sql += "ORDER BY ticker"
            pending = con.execute(sql, params).fetchall()
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

            download_all_unsynced(
                con,
                doc_type=selected_doc_type,
                on_progress=on_progress,
            )
            state.summary = (
                f"✅ {downloaded} downloaded, {failed} failed (type={selected_doc_type})"
            )
        finally:
            con.close()

    return run


def _make_convert(
    state: TaskState,
    refresh: Callable,
    *,
    start_year: Callable[[], int | None] = lambda: None,
    end_year: Callable[[], int | None] = lambda: None,
    ticker_filter: Callable[[], list[str] | None] = lambda: None,
    doc_type: Callable[[], str] = lambda: "2",
    output_dir: Callable[[], str] = lambda: str(MARKDOWN_DIR),
    job_scope_label: Callable[[], str] = lambda: "Annual Reports",
    force: Callable[[], bool] = lambda: False,
) -> Callable:
    def run():
        from converter import create_jobs, run_job
        from vietstock_documents import DOC_TYPE_ANNUAL_REPORT

        con = get_connection()
        try:
            init_db(con)

            selected_doc_type = str(doc_type() or DOC_TYPE_ANNUAL_REPORT)
            selected_output_dir = str(output_dir() or MARKDOWN_DIR)
            scope_label = str(job_scope_label() or "Selected")

            selected = ticker_filter() or None
            sy = start_year()
            ey = end_year()

            # Create jobs for any new staged PDFs
            created = create_jobs(
                con,
                tickers=selected,
                start_year=sy,
                end_year=ey,
                doc_type=selected_doc_type,
                output_dir=selected_output_dir,
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

            summary_sql = (
                "SELECT "
                "  COUNT(*) FILTER (WHERE status = 'pending') AS pending, "
                "  COUNT(*) FILTER (WHERE status = 'completed') AS completed, "
                "  COUNT(*) FILTER (WHERE status = 'failed') AS failed "
                "FROM conversion_jobs WHERE output_dir = ? "
            )
            summary_params: list = [selected_output_dir]
            if selected:
                placeholders = ", ".join(["?"] * len(selected))
                summary_sql += f"AND ticker IN ({placeholders}) "
                summary_params.extend([t.upper() for t in selected])
            if sy is not None:
                summary_sql += "AND year >= ? "
                summary_params.append(int(sy))
            if ey is not None:
                summary_sql += "AND year <= ? "
                summary_params.append(int(ey))

            summary_row = con.execute(
                summary_sql,
                summary_params,
            ).fetchone() or (0, 0, 0)
            pending_count, completed_count, failed_count = summary_row
            if int(pending_count or 0) == 0 and int(completed_count or 0) > 0 and not force():
                state.summary = (
                    f"⏭ Already converted ({int(completed_count or 0):,} jobs completed). "
                    f"{int(failed_count or 0)} failed"
                )
                return

            # Run selected pending jobs
            job_sql = (
                "SELECT id, ticker, year FROM conversion_jobs "
                "WHERE status = 'pending' AND output_dir = ? "
            )
            params: list = [selected_output_dir]
            if selected:
                placeholders = ", ".join(["?"] * len(selected))
                job_sql += f"AND ticker IN ({placeholders}) "
                params.extend([t.upper() for t in selected])
            if sy is not None:
                job_sql += "AND year >= ? "
                params.append(int(sy))
            if ey is not None:
                job_sql += "AND year <= ? "
                params.append(int(ey))
            job_sql += "ORDER BY ticker, year"
            pending = con.execute(job_sql, params).fetchall()

            if not pending:
                state.summary = f"No {scope_label} conversion jobs to run for selected filters"
                return

            def on_progress(job_id, ticker, year, status, error):
                row = TaskRow(
                    label=f"[{scope_label}] #{job_id} {ticker} {year}",
                    detail=error[:80] if error else "",
                    status="done" if status == "completed" else "error",
                )
                state.rows.append(row)
                refresh()

            results = {"completed": 0, "failed": 0, "total": len(pending)}
            for job_id, ticker, year in pending:
                success = run_job(con, int(job_id))
                status = "completed" if success else "failed"
                results[status] += 1
                error = None
                if not success:
                    err_row = con.execute(
                        "SELECT error_message FROM conversion_jobs WHERE id = ?",
                        [job_id],
                    ).fetchone()
                    error = err_row[0] if err_row else None
                on_progress(int(job_id), str(ticker), int(year), status, error)

            state.summary = (
                f"✅ {scope_label}: {results['completed']} converted, "
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


def _make_convert_financial_statement(
    state: TaskState,
    refresh: Callable,
    *,
    start_year: Callable[[], int | None] = lambda: None,
    end_year: Callable[[], int | None] = lambda: None,
    ticker_filter: Callable[[], list[str] | None] = lambda: None,
    force: Callable[[], bool] = lambda: False,
) -> Callable:
    def run():
        from converter import convert_financial_statement_to_markdown

        con = get_connection()
        try:
            init_db(con)

            selected = ticker_filter() or None
            sy = start_year()
            ey = end_year()

            sql = """
                SELECT
                    vd.ticker,
                    TRY_CAST(regexp_extract(vd.title, '(\\d{4})', 1) AS INTEGER) AS year
                FROM vietstock_documents vd
                WHERE vd.doc_type = '1'
                  AND vd.synced_to_raw = TRUE
                  AND vd.raw_path IS NOT NULL
                  AND vd.raw_path != ''
                  AND TRY_CAST(regexp_extract(vd.title, '(\\d{4})', 1) AS INTEGER) IS NOT NULL
            """
            params: list = []
            if selected:
                placeholders = ", ".join(["?"] * len(selected))
                sql += f" AND vd.ticker IN ({placeholders})"
                params.extend([t.upper() for t in selected])
            if sy is not None:
                sql += " AND TRY_CAST(regexp_extract(vd.title, '(\\d{4})', 1) AS INTEGER) >= ?"
                params.append(int(sy))
            if ey is not None:
                sql += " AND TRY_CAST(regexp_extract(vd.title, '(\\d{4})', 1) AS INTEGER) <= ?"
                params.append(int(ey))
            sql += " GROUP BY vd.ticker, year ORDER BY vd.ticker, year"

            candidates = con.execute(sql, params).fetchall()
            if not candidates:
                state.summary = "No financial statements to convert for selected filters"
                return

            completed = 0
            failed = 0
            for ticker, year in candidates:
                try:
                    convert_financial_statement_to_markdown(
                        con,
                        ticker=str(ticker),
                        year=int(year),
                        output_dir=FINANCIAL_STATEMENT_MARKDOWN_DIR,
                    )
                    completed += 1
                    state.rows.append(
                        TaskRow(
                            label=f"[FINANCIAL_STATEMENT] {ticker} {int(year)}",
                            detail="converted",
                            status="done",
                        )
                    )
                except Exception as exc:
                    failed += 1
                    state.rows.append(
                        TaskRow(
                            label=f"[FINANCIAL_STATEMENT] {ticker} {int(year)}",
                            detail=str(exc)[:100],
                            status="error",
                        )
                    )
                refresh()

            state.summary = (
                f"✅ Financial statements: {completed} converted, {failed} failed "
                f"(of {len(candidates)} total)"
            )
        finally:
            con.close()

    return run


def _make_embed_reports(
    state: TaskState,
    refresh: Callable,
    *,
    ticker_filter: Callable[[], list[str] | None] = lambda: None,
    year_filter: Callable[[], list[int] | None] = lambda: None,
    force: Callable[[], bool] = lambda: False,
    embedding_model: Callable[[], str] = lambda: EMBEDDING_MODEL,
    chunk_size: Callable[[], int] = lambda: EMBEDDING_CHUNK_SIZE,
) -> Callable:
    def run():
        from llm_embeddings import embed_all_reports

        con = get_connection()
        try:
            init_db(con)

            def on_progress(ticker, year, out, i, total):
                status = "done"
                detail = f"{out.get('embedded', 0)} chunks"
                if out.get("skipped"):
                    status = "skipped"
                    detail = out.get("reason", "skipped")
                if out.get("failed"):
                    status = "error"
                    detail = out.get("reason", "failed")
                state.rows.append(
                    TaskRow(
                        label=f"{ticker} / {year}",
                        detail=str(detail),
                        status=status,
                    )
                )
                refresh()

            result = embed_all_reports(
                con,
                tickers=ticker_filter(),
                years=year_filter(),
                replace=force(),
                model=embedding_model(),
                chunk_size=max(1, int(chunk_size())),
                progress_callback=on_progress,
            )
            state.summary = (
                f"✅ {result['embedded_reports']} embedded, "
                f"{result['skipped_reports']} skipped, "
                f"{result['failed_reports']} failed "
                f"({result['embedded_chunks']:,} chunks)"
            )
        finally:
            con.close()

    return run


def _make_infer_edc(
    state: TaskState,
    refresh: Callable,
    *,
    ticker_filter: Callable[[], list[str] | None] = lambda: None,
    year_filter: Callable[[], list[int] | None] = lambda: None,
    force: Callable[[], bool] = lambda: False,
    inference_model: Callable[[], str] = lambda: INFERENCE_MODEL,
    embedding_model: Callable[[], str] = lambda: EMBEDDING_MODEL,
) -> Callable:
    def run():
        from llm_inference import infer_report

        con = get_connection()
        try:
            init_db(con)
            ensure_vss_loaded(con)
            sql = "SELECT DISTINCT ticker, year FROM document_embeddings WHERE 1=1 "
            params: list = []

            selected = ticker_filter() or []
            years = year_filter() or []
            if selected:
                placeholders = ", ".join(["?"] * len(selected))
                sql += f"AND ticker IN ({placeholders}) "
                params.extend([t.upper() for t in selected])
            if years:
                placeholders = ", ".join(["?"] * len(years))
                sql += f"AND year IN ({placeholders}) "
                params.extend([int(y) for y in years])
            sql += "ORDER BY ticker, year"

            reports = con.execute(sql, params).fetchall()
            result = {"evaluated": [], "skipped": [], "failed": []}
            for ticker, year in reports:
                try:
                    n = infer_report(
                        str(ticker),
                        int(year),
                        con=con,
                        replace=force(),
                        inference_model=inference_model(),
                        embedding_model=embedding_model(),
                    )
                    state.rows.append(
                        TaskRow(
                            label=f"{ticker} / {year}",
                            detail=f"{n} categories",
                            status="done" if n > 0 else "skipped",
                        )
                    )
                    if n > 0:
                        result["evaluated"].append((ticker, year, n))
                    else:
                        result["skipped"].append((ticker, year))
                except Exception as exc:
                    state.rows.append(
                        TaskRow(
                            label=f"{ticker} / {year}",
                            detail=str(exc)[:100],
                            status="error",
                        )
                    )
                    result["failed"].append((ticker, year, str(exc)))
                refresh()
        finally:
            con.close()

        state.summary = (
            f"✅ {len(result['evaluated'])} evaluated, "
            f"{len(result['skipped'])} skipped, "
            f"{len(result['failed'])} failed"
        )

    return run


def _make_infer_proper_vn(
    state: TaskState,
    refresh: Callable,
    *,
    ticker_filter: Callable[[], list[str] | None] = lambda: None,
    year_filter: Callable[[], list[int] | None] = lambda: None,
    force: Callable[[], bool] = lambda: False,
    inference_model: Callable[[], str] = lambda: INFERENCE_MODEL,
    embedding_model: Callable[[], str] = lambda: EMBEDDING_MODEL,
) -> Callable:
    def run():
        from llm_proper_vn import infer_proper_vn_report

        con = get_connection()
        try:
            init_db(con)
            ensure_vss_loaded(con)
            sql = "SELECT DISTINCT ticker, year FROM document_embeddings WHERE 1=1 "
            params: list = []

            selected = ticker_filter() or []
            years = year_filter() or []
            if selected:
                placeholders = ", ".join(["?"] * len(selected))
                sql += f"AND ticker IN ({placeholders}) "
                params.extend([t.upper() for t in selected])
            if years:
                placeholders = ", ".join(["?"] * len(years))
                sql += f"AND year IN ({placeholders}) "
                params.extend([int(y) for y in years])
            sql += "ORDER BY ticker, year"

            reports = con.execute(sql, params).fetchall()
            result = {"evaluated": [], "failed": []}
            for ticker, year in reports:
                try:
                    out = infer_proper_vn_report(
                        str(ticker),
                        int(year),
                        con=con,
                        replace=force(),
                        inference_model=inference_model(),
                        embedding_model=embedding_model(),
                    )
                    state.rows.append(
                        TaskRow(
                            label=f"{ticker} / {year}",
                            detail=f"Color: {out.get('color', '?')}",
                            status="done",
                        )
                    )
                    result["evaluated"].append((ticker, year, out.get("color")))
                except Exception as exc:
                    state.rows.append(
                        TaskRow(
                            label=f"{ticker} / {year}",
                            detail=str(exc)[:100],
                            status="error",
                        )
                    )
                    result["failed"].append((ticker, year, str(exc)))
                refresh()
        finally:
            con.close()

        state.summary = (
            f"✅ {len(result['evaluated'])} evaluated, "
            f"{len(result['failed'])} failed"
        )

    return run


def _make_extract_governance(
    state: TaskState,
    refresh: Callable,
    *,
    ticker_filter: Callable[[], list[str] | None] = lambda: None,
    year_filter: Callable[[], list[int] | None] = lambda: None,
    force: Callable[[], bool] = lambda: False,
    inference_model: Callable[[], str] = lambda: INFERENCE_MODEL,
    embedding_model: Callable[[], str] = lambda: EMBEDDING_MODEL,
) -> Callable:
    def run():
        from llm_governance import extract_governance

        con = get_connection()
        try:
            init_db(con)
            ensure_vss_loaded(con)
            sql = "SELECT DISTINCT ticker, year FROM document_embeddings WHERE 1=1 "
            params: list = []

            selected = ticker_filter() or []
            years = year_filter() or []
            if selected:
                placeholders = ", ".join(["?"] * len(selected))
                sql += f"AND ticker IN ({placeholders}) "
                params.extend([t.upper() for t in selected])
            if years:
                placeholders = ", ".join(["?"] * len(years))
                sql += f"AND year IN ({placeholders}) "
                params.extend([int(y) for y in years])
            sql += "ORDER BY ticker, year"

            reports = con.execute(sql, params).fetchall()
            result = {"evaluated": [], "skipped": [], "failed": []}
            for ticker, year in reports:
                try:
                    n = extract_governance(
                        str(ticker),
                        int(year),
                        con=con,
                        replace=force(),
                        inference_model=inference_model(),
                        embedding_model=embedding_model(),
                    )
                    state.rows.append(
                        TaskRow(
                            label=f"{ticker} / {year}",
                            detail=f"{n} items",
                            status="done" if n > 0 else "skipped",
                        )
                    )
                    if n > 0:
                        result["evaluated"].append((ticker, year, n))
                    else:
                        result["skipped"].append((ticker, year))
                except Exception as exc:
                    state.rows.append(
                        TaskRow(
                            label=f"{ticker} / {year}",
                            detail=str(exc)[:100],
                            status="error",
                        )
                    )
                    result["failed"].append((ticker, year, str(exc)))
                refresh()
        finally:
            con.close()

        state.summary = (
            f"✅ {len(result['evaluated'])} evaluated, "
            f"{len(result['skipped'])} skipped, "
            f"{len(result['failed'])} failed"
        )

    return run


def _make_extract_financial_statement_audit(
    state: TaskState,
    refresh: Callable,
    *,
    ticker_filter: Callable[[], list[str] | None] = lambda: None,
    year_filter: Callable[[], list[int] | None] = lambda: None,
    force: Callable[[], bool] = lambda: False,
    inference_model: Callable[[], str] = lambda: INFERENCE_MODEL,
    embedding_model: Callable[[], str] = lambda: EMBEDDING_MODEL,
) -> Callable:
    def run():
        from llm_financial_statement_audit import extract_financial_statement_audit_info

        con = get_connection()
        try:
            init_db(con)
            ensure_vss_loaded(con)
            sql = (
                "SELECT DISTINCT ticker, year "
                "FROM financial_statement_document_embeddings WHERE 1=1 "
            )
            params: list = []

            selected = ticker_filter() or []
            years = year_filter() or []
            if selected:
                placeholders = ", ".join(["?"] * len(selected))
                sql += f"AND ticker IN ({placeholders}) "
                params.extend([t.upper() for t in selected])
            if years:
                placeholders = ", ".join(["?"] * len(years))
                sql += f"AND year IN ({placeholders}) "
                params.extend([int(y) for y in years])
            sql += "ORDER BY ticker, year"

            reports = con.execute(sql, params).fetchall()
            result = {"evaluated": [], "skipped": [], "failed": []}
            for ticker, year in reports:
                try:
                    out = extract_financial_statement_audit_info(
                        str(ticker),
                        int(year),
                        con=con,
                        replace=force(),
                        top_k=INFERENCE_TOP_K,
                        inference_model=inference_model(),
                        embedding_model=embedding_model(),
                    )
                    found = bool(out.get("found"))
                    detail = (
                        out.get("audit_firm")
                        or out.get("audit_opinion")
                        or ("audit extracted" if found else "no audit evidence")
                    )
                    state.rows.append(
                        TaskRow(
                            label=f"{ticker} / {year}",
                            detail=str(detail),
                            status="done" if found else "skipped",
                        )
                    )
                    if found:
                        result["evaluated"].append((ticker, year))
                    else:
                        result["skipped"].append((ticker, year))
                except Exception as exc:
                    state.rows.append(
                        TaskRow(
                            label=f"{ticker} / {year}",
                            detail=str(exc)[:100],
                            status="error",
                        )
                    )
                    result["failed"].append((ticker, year, str(exc)))
                refresh()
        finally:
            con.close()

        state.summary = (
            f"✅ {len(result['evaluated'])} extracted, "
            f"{len(result['skipped'])} skipped, "
            f"{len(result['failed'])} failed"
        )

    return run


def _make_sync_company_history(
    state: TaskState,
    refresh: Callable,
    *,
    ticker_filter: Callable[[], list[str] | None] = lambda: None,
) -> Callable:
    def run():
        from company_history import sync_company_history

        con = get_connection()
        try:
            init_db(con)

            selected = ticker_filter() or []
            if selected:
                tickers = [str(t).strip().upper() for t in selected if str(t).strip()]
            else:
                tickers = [
                    str(row[0]).upper()
                    for row in con.execute(
                        "SELECT ticker FROM companies ORDER BY ticker"
                    ).fetchall()
                ]

            if not tickers:
                state.summary = "No tickers selected and no companies configured"
                return

            done = 0
            failed = 0
            for ticker in tickers:
                try:
                    result = sync_company_history(con, ticker=ticker)
                    first_event = result.get("first_event") or {}
                    first_year = first_event.get("first_event_year")
                    firm_age = first_event.get("firm_age")
                    detail = (
                        f"{result.get('event_count', 0)} events"
                        + (
                            f" | first_year={first_year}" if first_year is not None else ""
                        )
                        + (
                            f" | firm_age={firm_age}" if firm_age is not None else ""
                        )
                    )
                    state.rows.append(
                        TaskRow(
                            label=ticker,
                            detail=detail,
                            status="done",
                        )
                    )
                    done += 1
                except Exception as exc:
                    state.rows.append(
                        TaskRow(
                            label=ticker,
                            detail=str(exc)[:120],
                            status="error",
                        )
                    )
                    failed += 1
                refresh()

            state.summary = f"✅ {done} synced, {failed} failed"
        finally:
            con.close()

    return run


# ---------------------------------------------------------------------------
# Shared navigation header
# ---------------------------------------------------------------------------

_NAV_ITEMS = [
    ("Home", "/"),
    ("Companies", "/companies"),
    ("Job Matrix", "/job-matrix"),
    ("Data Studio", "/data-studio"),
    ("Company History", "/company-history"),
    ("Extract Items", "/extract-items"),
    ("Jobs", "/jobs"),
    ("LLM Tasks", "/llm-tasks"),
    ("LLM Query", "/llm-query"),
    ("LLM Inputs", "/llm-inputs"),
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
    ("listings_annual", "5. Sync Annual Report Listings"),
    (
        "listings_financial_statement",
        "6. Sync Financial Statement Listings (Audited Consolidated Financial Statements)",
    ),
    ("download_annual", "7. Download Annual Report PDFs"),
    ("download_financial_statement", "8. Download Financial Statement PDFs"),
    ("convert", "9. Convert Annual Reports to Markdown"),
    ("convert_financial_statement", "10. Convert Financial Statements to Markdown"),
    ("load", "11. Load Markdown to DB"),
    ("embed", "12. Embed Annual Reports"),
    ("infer_edc", "13. Infer EDC"),
    ("infer_proper", "14. Infer PROPER-VN"),
    ("infer_governance", "15. Extract Governance"),
    ("infer_financial_statement_audit", "16. Extract Financial Statement Audit"),
    ("company_history", "17. Sync Company History (Moc lich su)"),
]

_ALL_JOBS = _INIT_JOBS + _PIPELINE_JOBS


_TARGET_COMPANIES_DEFAULT_RAW = """
ASG
CCR
CDN
CLL
DL1
DVP
DXP
GIC
HAH
HMH
HTV
NCT
PCT
PDN
QNP
SCS
SFI
SGN
STG
TCL
TCO
TCT
TMS
VGP
VMS
VNF
VNL
VSA
VSC
VSM
VTP
WCS
"""


def _normalize_ticker_list(raw_text: str) -> list[str]:
    """Parse multiline ticker input and strip exchange suffixes (e.g. AAA.HM -> AAA)."""
    codes: list[str] = []
    seen: set[str] = set()
    for line in raw_text.splitlines():
        token = str(line).strip().upper()
        if not token:
            continue
        normalized = token.split(".", 1)[0].strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        codes.append(normalized)
    return codes


def _parse_year_list(raw_text: str) -> list[int]:
    """Parse comma/space/newline year input into sorted unique year values."""
    values: set[int] = set()
    for token in str(raw_text or "").replace(",", " ").split():
        token = token.strip()
        if not token:
            continue
        if not token.isdigit():
            continue
        year = int(token)
        if 1900 <= year <= 2100:
            values.add(year)
    return sorted(values)


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

            con = get_connection()
            try:
                company_rows = con.execute(
                    "SELECT ticker FROM companies ORDER BY ticker"
                ).fetchall()
            finally:
                con.close()
            company_options = {r[0]: r[0] for r in company_rows}

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

                ticker_sel = (
                    ui.select(
                        label="Tickers (optional)",
                        options=company_options,
                        multiple=True,
                        with_input=True,
                        value=[],
                    )
                    .props(
                        "dense options-dense use-chips outlined "
                        'style="min-width: 320px"'
                    )
                    .classes("flex-1")
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
                    tickers = ticker_sel.value or []
                    sy = start_yr.value or DEFAULT_START_YEAR
                    ey = end_yr.value or DEFAULT_END_YEAR
                    params = (
                        f"?jobs={','.join(jobs)}"
                        f"&exchanges={','.join(exch)}"
                        f"&tickers={','.join(tickers)}"
                        f"&start_year={sy}"
                        f"&end_year={ey}"
                    )
                    ui.navigate.to(f"/jobs{params}")

                ui.button(
                    "Run Selected Jobs", on_click=_go_run, color="primary"
                )

                def _go_init():
                    exch = exchange_sel.value or []
                    tickers = ticker_sel.value or []
                    sy = start_yr.value or DEFAULT_START_YEAR
                    ey = end_yr.value or DEFAULT_END_YEAR
                    init_keys = ",".join(k for k, _ in _INIT_JOBS)
                    params = (
                        f"?jobs={init_keys}"
                        f"&exchanges={','.join(exch)}"
                        f"&tickers={','.join(tickers)}"
                        f"&start_year={sy}"
                        f"&end_year={ey}"
                    )
                    ui.navigate.to(f"/jobs{params}")

                ui.button("Initial Setup", on_click=_go_init).props(
                    "outline"
                ).tooltip("Sync stocks & financial models (one-time)")

                def _go_pipeline():
                    exch = exchange_sel.value or []
                    tickers = ticker_sel.value or []
                    sy = start_yr.value or DEFAULT_START_YEAR
                    ey = end_yr.value or DEFAULT_END_YEAR
                    pipe_keys = ",".join(k for k, _ in _PIPELINE_JOBS)
                    params = (
                        f"?jobs={pipe_keys}"
                        f"&exchanges={','.join(exch)}"
                        f"&tickers={','.join(tickers)}"
                        f"&start_year={sy}"
                        f"&end_year={ey}"
                    )
                    ui.navigate.to(f"/jobs{params}")

                ui.button("Run Pipeline", on_click=_go_pipeline).props(
                    "outline"
                ).tooltip(
                    "Statements, ratios, annual listings/download, financial-statement listings/download, convert, load"
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
                        ("Embeddings", "document_embeddings"),
                        ("EDC Results", "inference_results"),
                        ("PROPER-VN Results", "proper_vn_results"),
                        ("Governance Results", "governance_results"),
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

            ui.separator().classes("my-2")
            ui.label("Document Tracking By Type").classes(
                "text-md font-semibold"
            )

            @ui.refreshable
            def document_tracking_panel():
                from vietstock_documents import DOC_TYPE_LABELS

                con = get_connection()
                try:
                    rows = con.execute(
                        """
                        SELECT
                            doc_type,
                            COUNT(*) AS total_docs,
                            SUM(CASE WHEN synced_to_raw THEN 1 ELSE 0 END) AS downloaded_docs,
                            SUM(CASE WHEN NOT synced_to_raw THEN 1 ELSE 0 END) AS pending_docs,
                            COUNT(DISTINCT ticker) AS tickers_covered,
                            MAX(published_date) AS latest_published
                        FROM vietstock_documents
                        GROUP BY doc_type
                        ORDER BY doc_type
                        """
                    ).fetchall()
                finally:
                    con.close()

                if not rows:
                    ui.label("No document listings synced yet.").classes(
                        "text-gray-500 text-sm"
                    )
                    return

                table_rows = []
                for doc_type, total, downloaded, pending, tickers, latest in rows:
                    table_rows.append(
                        {
                            "doc_type": DOC_TYPE_LABELS.get(
                                str(doc_type), str(doc_type)
                            ),
                            "total_docs": int(total or 0),
                            "downloaded_docs": int(downloaded or 0),
                            "pending_docs": int(pending or 0),
                            "tickers_covered": int(tickers or 0),
                            "latest_published": latest or "",
                        }
                    )

                columns = [
                    {
                        "name": "doc_type",
                        "label": "Document Type",
                        "field": "doc_type",
                        "align": "left",
                    },
                    {
                        "name": "tickers_covered",
                        "label": "Tickers",
                        "field": "tickers_covered",
                        "align": "center",
                    },
                    {
                        "name": "total_docs",
                        "label": "Total",
                        "field": "total_docs",
                        "align": "center",
                    },
                    {
                        "name": "downloaded_docs",
                        "label": "Downloaded",
                        "field": "downloaded_docs",
                        "align": "center",
                    },
                    {
                        "name": "pending_docs",
                        "label": "Pending Download",
                        "field": "pending_docs",
                        "align": "center",
                    },
                    {
                        "name": "latest_published",
                        "label": "Latest Published",
                        "field": "latest_published",
                        "align": "center",
                    },
                ]

                ui.table(
                    columns=columns,
                    rows=table_rows,
                    row_key="doc_type",
                ).classes("w-full").props("dense flat")

            document_tracking_panel()
            ui.button(
                "Refresh Tracking",
                on_click=lambda: document_tracking_panel.refresh(),
            ).props("dense flat")


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

        def _sync_all_companies() -> None:
            from financial_data import fetch_stocks

            con = get_connection()
            try:
                init_db(con)
                stocks_synced = fetch_stocks(con)
                result = sync_companies_from_stocks(con)
            finally:
                con.close()

            ui.notify(
                "Synced listed companies "
                f"({stocks_synced:,} stocks refreshed, {result['added']:,} added, "
                f"{result['existing']:,} already present)"
            )
            search_results.refresh()
            company_table.refresh()

        with ui.row().classes("items-end gap-4 flex-wrap w-full"):
            exchange_filter = (
                ui.select(
                    label="Exchange",
                    options=["All", "HOSE", "HNX", "UPCOM"],
                    value="All",
                    on_change=lambda e: (
                        search_results.refresh(),
                        company_table.refresh(),
                    ),
                )
                .props("dense outlined")
                .classes("w-36")
            )

            ui.button(
                "Sync All Companies",
                on_click=_sync_all_companies,
            ).props("dense color=primary")

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
                exch = exchange_filter.value
                con = get_connection()
                try:
                    conditions = [
                        "(s.code LIKE ? OR LOWER(s.company_name) LIKE LOWER(?))"
                    ]
                    params = [f"%{q.upper()}%", f"%{q}%"]
                    if exch and exch != "All":
                        conditions.append("s.floor = ?")
                        params.append(exch)
                    rows = con.execute(
                        "SELECT s.code, s.floor, s.company_name, "
                        "  CASE WHEN c.ticker IS NOT NULL THEN TRUE ELSE FALSE END AS added "
                        "FROM stocks s "
                        "LEFT JOIN companies c ON s.code = c.ticker "
                        f"WHERE {' AND '.join(conditions)} "
                        "ORDER BY s.code LIMIT 50",
                        params,
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
                exch = exchange_filter.value
                con = get_connection()
                try:
                    where = "WHERE s.floor = ?" if exch and exch != "All" else ""
                    params = [exch] if where else []
                    rows = con.execute("""
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
                    """ + where + """
                        ORDER BY c.ticker
                        """, params).fetchall()
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


@ui.page("/company-targets")
def page_company_targets():
    ui.dark_mode(False)
    _nav_header()
    init_db()

    state: dict[str, list[str]] = {
        "tickers": _normalize_ticker_list(_TARGET_COMPANIES_DEFAULT_RAW),
    }
    run_state = TaskState()

    with ui.column().classes("w-full max-w-6xl mx-auto gap-4 p-4"):
        ui.label("Target Company LLM Manager").classes("text-2xl font-bold")
        ui.label(
            "Manage target tickers, check readiness, and generate LLM extraction jobs."
        ).classes("text-sm text-gray-600")

        with ui.card().classes("w-full"):
            ui.label("Target List Input").classes("text-lg font-bold")
            ui.label(
                "Paste ticker list with or without suffix. The app normalizes to bare ticker codes."
            ).classes("text-sm text-gray-500")

            ticker_text = (
                ui.textarea(
                    label="Tickers",
                    value=_TARGET_COMPANIES_DEFAULT_RAW,
                )
                .props('autogrow outlined')
                .classes("w-full")
            )

            normalized_summary = ui.label("").classes("text-sm text-gray-600")

            @ui.refreshable
            def normalized_preview() -> None:
                tickers = list(state.get("tickers", []))
                normalized_summary.text = f"Normalized tickers: {len(tickers)}"
                if not tickers:
                    ui.label("No valid tickers parsed.").classes(
                        "text-xs text-gray-500"
                    )
                    return
                with ui.row().classes("gap-2 flex-wrap"):
                    for ticker in tickers:
                        ui.chip(str(ticker)).props("outline")

            def _normalize_input() -> None:
                state["tickers"] = _normalize_ticker_list(
                    str(ticker_text.value or "")
                )
                normalized_preview.refresh()
                target_status_table.refresh()

            def _add_missing_to_companies() -> None:
                tickers = list(state.get("tickers", []))
                if not tickers:
                    ui.notify("No tickers to add", type="warning")
                    return

                con = get_connection()
                try:
                    existing = {
                        r[0]
                        for r in con.execute(
                            "SELECT ticker FROM companies"
                        ).fetchall()
                    }
                    for ticker in tickers:
                        ensure_company(con, str(ticker))
                finally:
                    con.close()

                added = [t for t in tickers if t not in existing]
                ui.notify(
                    f"Added {len(added)} ticker(s), {len(tickers) - len(added)} already existed"
                )
                target_status_table.refresh()

            with ui.row().classes("gap-2"):
                ui.button("Normalize List", on_click=_normalize_input).props(
                    "dense color=primary"
                )
                ui.button(
                    "Add Missing to Companies",
                    on_click=_add_missing_to_companies,
                ).props("dense")

            normalized_preview()

        def _fetch_target_status_rows() -> list[TargetStatusRow]:
            tickers = list(state.get("tickers", []))
            if not tickers:
                return []

            values_sql = ", ".join(["(?)" for _ in tickers])
            sql = f"""
                WITH targets(ticker) AS (
                    SELECT * FROM (VALUES {values_sql}) AS v(ticker)
                ),
                comp AS (
                    SELECT ticker FROM companies
                ),
                rpt AS (
                    SELECT ticker, COUNT(DISTINCT year) AS report_years
                    FROM annual_reports
                    GROUP BY ticker
                ),
                emb AS (
                    SELECT ticker, COUNT(DISTINCT year) AS embedded_years
                    FROM document_embeddings
                    GROUP BY ticker
                ),
                ij AS (
                    SELECT
                        ticker,
                        COUNT(*) AS total_jobs,
                        SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending_jobs,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed_jobs
                    FROM inference_jobs
                    GROUP BY ticker
                ),
                pj AS (
                    SELECT
                        ticker,
                        COUNT(*) AS total_jobs,
                        SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending_jobs,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed_jobs
                    FROM proper_vn_jobs
                    GROUP BY ticker
                ),
                gj AS (
                    SELECT
                        ticker,
                        COUNT(*) AS total_jobs,
                        SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending_jobs,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed_jobs
                    FROM governance_jobs
                    GROUP BY ticker
                )
                SELECT
                    t.ticker,
                    CASE WHEN c.ticker IS NULL THEN FALSE ELSE TRUE END AS in_companies,
                    COALESCE(rpt.report_years, 0) AS report_years,
                    COALESCE(emb.embedded_years, 0) AS embedded_years,
                    COALESCE(ij.total_jobs, 0) AS edc_jobs,
                    COALESCE(ij.pending_jobs, 0) AS edc_pending,
                    COALESCE(ij.completed_jobs, 0) AS edc_completed,
                    COALESCE(pj.total_jobs, 0) AS proper_jobs,
                    COALESCE(pj.pending_jobs, 0) AS proper_pending,
                    COALESCE(pj.completed_jobs, 0) AS proper_completed,
                    COALESCE(gj.total_jobs, 0) AS gov_jobs,
                    COALESCE(gj.pending_jobs, 0) AS gov_pending,
                    COALESCE(gj.completed_jobs, 0) AS gov_completed
                FROM targets t
                LEFT JOIN comp c ON c.ticker = t.ticker
                LEFT JOIN rpt ON rpt.ticker = t.ticker
                LEFT JOIN emb ON emb.ticker = t.ticker
                LEFT JOIN ij ON ij.ticker = t.ticker
                LEFT JOIN pj ON pj.ticker = t.ticker
                LEFT JOIN gj ON gj.ticker = t.ticker
                ORDER BY t.ticker
            """

            con = get_connection()
            try:
                rows = con.execute(sql, tickers).fetchall()
            finally:
                con.close()

            out: list[TargetStatusRow] = []
            for row in rows:
                out.append(
                    {
                        "ticker": row[0],
                        "in_companies": bool(row[1]),
                        "report_years": int(row[2]),
                        "embedded_years": int(row[3]),
                        "edc_jobs": int(row[4]),
                        "edc_pending": int(row[5]),
                        "edc_completed": int(row[6]),
                        "proper_jobs": int(row[7]),
                        "proper_pending": int(row[8]),
                        "proper_completed": int(row[9]),
                        "gov_jobs": int(row[10]),
                        "gov_pending": int(row[11]),
                        "gov_completed": int(row[12]),
                    }
                )
            return out

        with ui.card().classes("w-full"):
            ui.label("Readiness & Job Status").classes("text-lg font-bold")
            status_summary = ui.label("").classes("text-sm text-gray-600")
            auto_refresh_targets = ui.switch(
                "Auto refresh status", value=True
            ).props("dense")

            @ui.refreshable
            def target_status_table() -> None:
                rows = _fetch_target_status_rows()
                if not rows:
                    ui.label("No target tickers loaded.").classes(
                        "text-xs text-gray-500"
                    )
                    status_summary.text = ""
                    return

                in_companies = sum(1 for r in rows if bool(r["in_companies"]))
                docs_ready = sum(1 for r in rows if int(r["report_years"]) > 0)
                emb_ready = sum(1 for r in rows if int(r["embedded_years"]) > 0)
                status_summary.text = (
                    f"In companies: {in_companies}/{len(rows)} | "
                    f"Documents ready: {docs_ready}/{len(rows)} | "
                    f"Embeddings ready: {emb_ready}/{len(rows)}"
                )

                columns = [
                    {"name": "ticker", "label": "Ticker", "field": "ticker", "align": "left"},
                    {"name": "in_companies", "label": "In Companies", "field": "in_companies", "align": "center"},
                    {"name": "report_years", "label": "Doc Years", "field": "report_years", "align": "right"},
                    {"name": "embedded_years", "label": "Embedded Years", "field": "embedded_years", "align": "right"},
                    {"name": "edc", "label": "EDC Jobs (P/C/T)", "field": "edc", "align": "center"},
                    {"name": "proper", "label": "PROPER Jobs (P/C/T)", "field": "proper", "align": "center"},
                    {"name": "gov", "label": "GOV Jobs (P/C/T)", "field": "gov", "align": "center"},
                ]
                table_rows = [
                    {
                        "ticker": r["ticker"],
                        "in_companies": "YES" if r["in_companies"] else "NO",
                        "report_years": r["report_years"],
                        "embedded_years": r["embedded_years"],
                        "edc": f"{r['edc_pending']}/{r['edc_completed']}/{r['edc_jobs']}",
                        "proper": f"{r['proper_pending']}/{r['proper_completed']}/{r['proper_jobs']}",
                        "gov": f"{r['gov_pending']}/{r['gov_completed']}/{r['gov_jobs']}",
                    }
                    for r in rows
                ]
                ui.table(
                    columns=columns,
                    rows=table_rows,
                    row_key="ticker",
                ).classes("w-full").props("dense flat")

            with ui.row().classes("gap-2"):
                ui.button(
                    "Refresh Status", on_click=lambda: target_status_table.refresh()
                ).props("dense outline")

            target_status_table()

        with ui.card().classes("w-full"):
            ui.label("Generate LLM Extraction Jobs").classes("text-lg font-bold")
            ui.label(
                "Jobs are generated only for ticker-year pairs that already have embeddings."
            ).classes("text-sm text-gray-500")

            with ui.row().classes("items-end gap-3 flex-wrap"):
                llm_model_input = (
                    ui.input("LLM model", value=INFERENCE_MODEL)
                    .props("dense outlined")
                    .classes("w-56")
                )
                replace_toggle = ui.switch("Reset completed/failed jobs").props(
                    "dense"
                )

            def _generate_target_jobs() -> None:
                from llm_financial_statement_audit import create_financial_statement_audit_jobs
                from llm_governance import create_governance_jobs
                from llm_inference import create_inference_jobs
                from llm_proper_vn import create_proper_vn_jobs

                rows = _fetch_target_status_rows()
                eligible = [
                    str(r["ticker"])
                    for r in rows
                    if int(r["embedded_years"]) > 0
                ]
                if not eligible:
                    ui.notify(
                        "No eligible tickers (embedded years = 0 for all).",
                        type="warning",
                    )
                    return

                model = str(llm_model_input.value or INFERENCE_MODEL)
                replace = bool(replace_toggle.value)

                con = get_connection()
                try:
                    init_db(con)
                    edc_created = create_inference_jobs(
                        con,
                        tickers=eligible,
                        replace=replace,
                        inference_model=model,
                    )
                    proper_created = create_proper_vn_jobs(
                        con,
                        tickers=eligible,
                        replace=replace,
                        inference_model=model,
                    )
                    gov_created = create_governance_jobs(
                        con,
                        tickers=eligible,
                        replace=replace,
                        inference_model=model,
                    )
                    financial_statement_audit_created = create_financial_statement_audit_jobs(
                        con,
                        tickers=eligible,
                        replace=replace,
                        inference_model=model,
                    )
                finally:
                    con.close()

                ui.notify(
                    "Created jobs - "
                    "EDC: "
                    f"{edc_created}, PROPER: {proper_created}, GOV: {gov_created}, "
                    f"FS_AUDIT: {financial_statement_audit_created}"
                )
                target_status_table.refresh()

            with ui.row().classes("gap-2"):
                ui.button(
                    "Generate LLM Jobs for Eligible Targets",
                    on_click=_generate_target_jobs,
                    color="primary",
                )

        with ui.card().classes("w-full"):
            ui.label("Run Pending Target Jobs").classes("text-lg font-bold")
            ui.label(
                "Submit pending jobs first, then reconcile already-running jobs to ingest completed batch outputs."
            ).classes("text-sm text-gray-500")

            def _get_pending_target_jobs(table_name: str, model: str) -> list[tuple[str, int]]:
                tickers = [str(t).upper() for t in state.get("tickers", [])]
                if not tickers:
                    return []

                placeholders = ", ".join(["?"] * len(tickers))
                sql = (
                    f"SELECT ticker, year FROM {table_name} "
                    "WHERE status = 'pending' AND model = ? "
                    f"AND ticker IN ({placeholders}) "
                    "ORDER BY ticker, year"
                )

                con = get_connection()
                try:
                    rows = con.execute(sql, [model, *tickers]).fetchall()
                finally:
                    con.close()
                return [(str(r[0]), int(r[1])) for r in rows]

            def _get_running_target_jobs(table_name: str, model: str) -> list[tuple[str, int]]:
                tickers = [str(t).upper() for t in state.get("tickers", [])]
                if not tickers:
                    return []

                placeholders = ", ".join(["?"] * len(tickers))
                sql = (
                    f"SELECT ticker, year FROM {table_name} "
                    "WHERE status = 'running' AND model = ? "
                    f"AND ticker IN ({placeholders}) "
                    "ORDER BY ticker, year"
                )

                con = get_connection()
                try:
                    rows = con.execute(sql, [model, *tickers]).fetchall()
                finally:
                    con.close()
                return [(str(r[0]), int(r[1])) for r in rows]

            def _make_run_pending_targets(kind: str) -> Callable:
                def run() -> None:
                    from llm_financial_statement_audit import (
                        extract_financial_statement_audit_info,
                    )
                    from llm_governance import extract_governance
                    from llm_inference import infer_report
                    from llm_proper_vn import infer_proper_vn_report

                    model = str(llm_model_input.value or INFERENCE_MODEL)
                    replace = bool(replace_toggle.value)

                    run_state.rows = []
                    run_state.summary = ""
                    run_status_panel.refresh()

                    kinds = [kind] if kind != "all" else ["edc", "proper", "gov", "fs_audit"]
                    labels = {
                        "edc": "EDC",
                        "proper": "PROPER-VN",
                        "gov": "Governance",
                        "fs_audit": "Financial Statement Audit",
                    }
                    tables = {
                        "edc": "inference_jobs",
                        "proper": "proper_vn_jobs",
                        "gov": "governance_jobs",
                        "fs_audit": "financial_statement_audit_jobs",
                    }

                    con = get_connection()
                    try:
                        ensure_vss_loaded(con)

                        total_done = 0
                        total_failed = 0
                        total_pending = 0
                        total_reconciled = 0

                        for current in kinds:
                            running_before = _get_running_target_jobs(tables[current], model)
                            pending = _get_pending_target_jobs(tables[current], model)
                            total_pending += len(pending)

                            if pending:
                                start_idx = len(run_state.rows)
                                for ticker, year in pending:
                                    run_state.rows.append(
                                        TaskRow(label=f"{labels[current]} {ticker}/{year}")
                                    )
                                run_status_panel.refresh()

                                for idx, (ticker, year) in enumerate(pending):
                                    row = run_state.rows[start_idx + idx]
                                    row.status = "running"
                                    run_status_panel.refresh()

                                    try:
                                        if current == "edc":
                                            n = infer_report(
                                                ticker,
                                                year,
                                                con=con,
                                                replace=replace,
                                                inference_model=model,
                                            )
                                            row.detail = f"{n} categories"
                                        elif current == "proper":
                                            cls = infer_proper_vn_report(
                                                ticker,
                                                year,
                                                con=con,
                                                replace=replace,
                                                inference_model=model,
                                            )
                                            row.detail = f"Color: {cls.get('color', '?')}"
                                        elif current == "fs_audit":
                                            out = extract_financial_statement_audit_info(
                                                ticker,
                                                year,
                                                con=con,
                                                replace=replace,
                                                inference_model=model,
                                            )
                                            fs_status = str(out.get("status") or "")
                                            if fs_status in {"submitted", "running"}:
                                                row.detail = "batch running on OpenAI"
                                            else:
                                                row.detail = (
                                                    out.get("audit_firm")
                                                    or out.get("audit_opinion")
                                                    or (
                                                        "audit extracted"
                                                        if out.get("found")
                                                        else "no audit evidence"
                                                    )
                                                )
                                        else:
                                            n = extract_governance(
                                                ticker,
                                                year,
                                                con=con,
                                                replace=replace,
                                                inference_model=model,
                                            )
                                            row.detail = f"{n} items"

                                        row.status = "done"
                                        total_done += 1
                                    except Exception as exc:
                                        row.status = "error"
                                        row.detail = str(exc)[:120]
                                        total_failed += 1
                                    run_status_panel.refresh()

                            if running_before:
                                start_idx = len(run_state.rows)
                                for ticker, year in running_before:
                                    run_state.rows.append(
                                        TaskRow(label=f"Sync {labels[current]} {ticker}/{year}")
                                    )
                                run_status_panel.refresh()

                                for idx, (ticker, year) in enumerate(running_before):
                                    row = run_state.rows[start_idx + idx]
                                    row.status = "running"
                                    run_status_panel.refresh()

                                    try:
                                        if current == "edc":
                                            n = infer_report(
                                                ticker,
                                                year,
                                                con=con,
                                                replace=False,
                                                inference_model=model,
                                            )
                                            row.detail = (
                                                f"Synced {n} categories"
                                                if n > 0
                                                else "Batch still running"
                                            )
                                        elif current == "proper":
                                            cls = infer_proper_vn_report(
                                                ticker,
                                                year,
                                                con=con,
                                                replace=False,
                                                inference_model=model,
                                            )
                                            row.detail = f"Color: {cls.get('color', '?')}"
                                        elif current == "fs_audit":
                                            out = extract_financial_statement_audit_info(
                                                ticker,
                                                year,
                                                con=con,
                                                replace=False,
                                                inference_model=model,
                                            )
                                            fs_status = str(out.get("status") or "")
                                            if fs_status in {"submitted", "running"}:
                                                row.detail = "batch still running"
                                            else:
                                                row.detail = (
                                                    out.get("audit_firm")
                                                    or out.get("audit_opinion")
                                                    or (
                                                        "audit extracted"
                                                        if out.get("found")
                                                        else "no audit evidence"
                                                    )
                                                )
                                        else:
                                            n = extract_governance(
                                                ticker,
                                                year,
                                                con=con,
                                                replace=False,
                                                inference_model=model,
                                            )
                                            row.detail = (
                                                f"Synced {n} items"
                                                if n > 0
                                                else "Batch still running"
                                            )

                                        row.status = "done"
                                        total_reconciled += 1
                                    except Exception as exc:
                                        row.status = "error"
                                        row.detail = str(exc)[:120]
                                        total_failed += 1
                                    run_status_panel.refresh()
                    finally:
                        con.close()

                    if total_pending == 0:
                        run_state.summary = (
                            f"⏭ No pending target jobs for model '{model}'. Reconciled {total_reconciled} running job(s)."
                        )
                    else:
                        run_state.summary = (
                            f"✅ Ran {total_pending} pending job(s): "
                            f"{total_done} done, {total_failed} failed | "
                            f"Reconciled {total_reconciled} running job(s)"
                        )

                    target_status_table.refresh()
                    run_status_panel.refresh()

                return run

            with ui.row().classes("gap-2 flex-wrap"):
                ui.button(
                    "Run Pending EDC",
                    on_click=lambda: _run_in_thread(
                        _make_run_pending_targets("edc"),
                        run_state,
                        lambda: run_status_panel.refresh(),
                    ),
                    color="primary",
                )
                ui.button(
                    "Run Pending PROPER-VN",
                    on_click=lambda: _run_in_thread(
                        _make_run_pending_targets("proper"),
                        run_state,
                        lambda: run_status_panel.refresh(),
                    ),
                )
                ui.button(
                    "Run Pending Governance",
                    on_click=lambda: _run_in_thread(
                        _make_run_pending_targets("gov"),
                        run_state,
                        lambda: run_status_panel.refresh(),
                    ),
                )
                ui.button(
                    "Run Pending Financial Statement Audit",
                    on_click=lambda: _run_in_thread(
                        _make_run_pending_targets("fs_audit"),
                        run_state,
                        lambda: run_status_panel.refresh(),
                    ),
                )
                ui.button(
                    "Run All Pending",
                    on_click=lambda: _run_in_thread(
                        _make_run_pending_targets("all"),
                        run_state,
                        lambda: run_status_panel.refresh(),
                    ),
                ).props("outline")

            @ui.refreshable
            def run_status_panel() -> None:
                if run_state.error:
                    with ui.row().classes("items-center gap-2"):
                        ui.label(run_state.error).classes("text-red-500 text-sm")
                        ui.button(
                            "Clear",
                            on_click=lambda: (
                                setattr(run_state, "error", ""),
                                run_status_panel.refresh(),
                            ),
                        ).props("dense flat")
                if run_state.summary:
                    ui.label(run_state.summary).classes(
                        "text-green-600 text-sm font-medium"
                    )
                if run_state.rows:
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
                                "skipped": "⏭",
                                "error": "❌",
                            }.get(r.status, r.status),
                        }
                        for r in run_state.rows
                    ]
                    ui.table(columns=columns, rows=rows_data, row_key="label").classes(
                        "w-full"
                    ).props("dense flat")

            run_status_panel()

        def _auto_refresh_company_targets() -> None:
            if not bool(auto_refresh_targets.value):
                return
            target_status_table.refresh()
            if run_state.running:
                run_status_panel.refresh()

        ui.timer(3.0, _auto_refresh_company_targets)


# ---------------------------------------------------------------------------
# Jobs page — run selected tasks with progress
# ---------------------------------------------------------------------------


@ui.page("/jobs")
def page_jobs(
    jobs: str = "",
    exchanges: str = "",
    tickers: str = "",
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
):
    ui.dark_mode(False)
    _nav_header()
    init_db()

    job_keys = [j for j in jobs.split(",") if j] if jobs else []
    exchange_list = [e for e in exchanges.split(",") if e] if exchanges else []
    ticker_list = [t.upper() for t in tickers.split(",") if t] if tickers else []

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
        if ticker_list:
            ui.label(f"Tickers: {', '.join(ticker_list)}").classes(
                "text-sm text-gray-600"
            )

        if not job_keys:
            ui.label(
                "No jobs selected. Go to Home to configure and select jobs."
            ).classes("text-gray-500")
            return

        get_sy = lambda: start_year
        get_ey = lambda: end_year
        get_years = lambda: list(range(int(start_year), int(end_year) + 1))
        get_tickers = lambda: ticker_list or None

        force_toggle = ui.switch("Force re-run (ignore existing data)").props(
            "dense"
        )
        get_force = lambda: force_toggle.value

        with ui.row().classes("items-end gap-3 flex-wrap w-full"):
            from vietstock_documents import (
                DOC_TYPE_AUDITED_CONSOLIDATED_FS,
                DOC_TYPE_ANNUAL_REPORT,
                DOC_TYPE_LABELS,
            )

            doc_type_select = (
                ui.select(
                    {
                        key: label
                        for key, label in DOC_TYPE_LABELS.items()
                    },
                    value=DOC_TYPE_ANNUAL_REPORT,
                    label="Vietstock document type",
                )
                .props("dense outlined")
                .classes("w-80")
            )
            embed_model_input = (
                ui.input("Embedding model", value=EMBEDDING_MODEL)
                .props("dense outlined")
                .classes("w-56")
            )
            infer_model_input = (
                ui.input("Inference model", value=INFERENCE_MODEL)
                .props("dense outlined")
                .classes("w-56")
            )
            chunk_size_input = (
                ui.number(
                    "Embedding chunk size",
                    value=EMBEDDING_CHUNK_SIZE,
                    min=32,
                    max=8192,
                    step=32,
                    format="%.0f",
                )
                .props("dense outlined")
                .classes("w-44")
            )

        get_embed_model = lambda: (embed_model_input.value or EMBEDDING_MODEL)
        get_infer_model = lambda: (infer_model_input.value or INFERENCE_MODEL)
        get_chunk_size = lambda: int(chunk_size_input.value or EMBEDDING_CHUNK_SIZE)
        get_doc_type = lambda: str(doc_type_select.value or DOC_TYPE_ANNUAL_REPORT)

        make_fns: dict[str, Callable] = {
            "stocks": lambda s, r: _make_sync_stocks(s, r, force=get_force),
            "models": lambda s, r: _make_sync_models(s, r, force=get_force),
            "statements": lambda s, r: _make_sync_statements(
                s,
                get_sy,
                get_ey,
                r,
                ticker_filter=get_tickers,
                force=get_force,
            ),
            "ratios": lambda s, r: _make_sync_ratios(
                s,
                get_sy,
                get_ey,
                r,
                ticker_filter=get_tickers,
                force=get_force,
            ),
            "listings": lambda s, r: _make_sync_listings(
                s,
                get_sy,
                get_ey,
                r,
                doc_type=get_doc_type,
                ticker_filter=get_tickers,
                force=get_force,
            ),
            "listings_annual": lambda s, r: _make_sync_listings(
                s,
                get_sy,
                get_ey,
                r,
                doc_type=lambda: DOC_TYPE_ANNUAL_REPORT,
                ticker_filter=get_tickers,
                force=get_force,
            ),
            "listings_financial_statement": lambda s, r: _make_sync_listings(
                s,
                get_sy,
                get_ey,
                r,
                doc_type=lambda: DOC_TYPE_AUDITED_CONSOLIDATED_FS,
                ticker_filter=get_tickers,
                force=get_force,
            ),
            "download": lambda s, r: _make_download_pdfs(
                s,
                r,
                doc_type=get_doc_type,
                ticker_filter=get_tickers,
                force=get_force,
            ),
            "download_annual": lambda s, r: _make_download_pdfs(
                s,
                r,
                doc_type=lambda: DOC_TYPE_ANNUAL_REPORT,
                ticker_filter=get_tickers,
                force=get_force,
            ),
            "download_financial_statement": lambda s, r: _make_download_pdfs(
                s,
                r,
                doc_type=lambda: DOC_TYPE_AUDITED_CONSOLIDATED_FS,
                ticker_filter=get_tickers,
                force=get_force,
            ),
            "convert": lambda s, r: _make_convert(
                s,
                r,
                start_year=get_sy,
                end_year=get_ey,
                ticker_filter=get_tickers,
                doc_type=lambda: DOC_TYPE_ANNUAL_REPORT,
                output_dir=lambda: str(MARKDOWN_DIR),
                job_scope_label=lambda: "Annual Reports",
                force=get_force,
            ),
            "convert_financial_statement": lambda s, r: _make_convert_financial_statement(
                s,
                r,
                start_year=get_sy,
                end_year=get_ey,
                ticker_filter=get_tickers,
                force=get_force,
            ),
            "load": lambda s, r: _make_load(s, r, force=get_force),
            "embed": lambda s, r: _make_embed_reports(
                s,
                r,
                ticker_filter=get_tickers,
                year_filter=get_years,
                force=get_force,
                embedding_model=get_embed_model,
                chunk_size=get_chunk_size,
            ),
            "infer_edc": lambda s, r: _make_infer_edc(
                s,
                r,
                ticker_filter=get_tickers,
                year_filter=get_years,
                force=get_force,
                inference_model=get_infer_model,
                embedding_model=get_embed_model,
            ),
            "infer_proper": lambda s, r: _make_infer_proper_vn(
                s,
                r,
                ticker_filter=get_tickers,
                year_filter=get_years,
                force=get_force,
                inference_model=get_infer_model,
                embedding_model=get_embed_model,
            ),
            "infer_governance": lambda s, r: _make_extract_governance(
                s,
                r,
                ticker_filter=get_tickers,
                year_filter=get_years,
                force=get_force,
                inference_model=get_infer_model,
                embedding_model=get_embed_model,
            ),
            "infer_financial_statement_audit": lambda s, r: _make_extract_financial_statement_audit(
                s,
                r,
                ticker_filter=get_tickers,
                year_filter=get_years,
                force=get_force,
                inference_model=get_infer_model,
                embedding_model=get_embed_model,
            ),
            "company_history": lambda s, r: _make_sync_company_history(
                s,
                r,
                ticker_filter=get_tickers,
            ),
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


@ui.page("/job-matrix")
def page_job_matrix():
    ui.dark_mode(False)
    _nav_header()
    init_db()

    from converter import convert_financial_statement_to_markdown, create_jobs, run_job
    from llm_embeddings import embed_report
    from llm_financial_statement_audit import extract_financial_statement_audit_info
    from llm_governance import extract_governance
    from llm_inference import infer_report
    from llm_proper_vn import infer_proper_vn_report
    from loader import sync_markdown_files
    from vietstock_documents import (
        DOC_TYPE_ANNUAL_REPORT,
        DOC_TYPE_AUDITED_CONSOLIDATED_FS,
        download_document_to_raw,
        fetch_all_companies_documents,
    )

    with ui.column().classes("w-full max-w-[1400px] mx-auto gap-4 p-4"):
        ui.label("Company-Year Job Matrix").classes("text-2xl font-bold")
        ui.label(
            "Track pipeline completion per company-year and run selected jobs directly from one grid."
        ).classes("text-sm text-gray-600")

        year_now = datetime.now().year
        year_options = list(range(DEFAULT_START_YEAR, year_now + 1))

        with ui.row().classes("items-end gap-3 flex-wrap w-full"):
            ticker_sel = (
                ui.select(
                    label="Tickers (optional)",
                    options={},
                    multiple=True,
                    with_input=True,
                )
                .props(
                    "dense options-dense use-chips outlined "
                    'style="min-width: 360px"'
                )
                .classes("flex-1")
            )
            start_year_input = (
                ui.select(
                    label="Start year",
                    options=year_options,
                    value=DEFAULT_START_YEAR,
                )
                .props("dense outlined")
                .classes("w-32")
            )
            end_year_input = (
                ui.select(
                    label="End year",
                    options=year_options,
                    value=DEFAULT_END_YEAR,
                )
                .props("dense outlined")
                .classes("w-32")
            )
            force_toggle = ui.switch("Force re-run").props("dense")

        with ui.row().classes("items-end gap-3 flex-wrap w-full"):
            inference_model_input = (
                ui.input("Inference model", value=INFERENCE_MODEL)
                .props("dense outlined")
                .classes("w-56")
            )
            ui.label(
                "Choose jobs to run on selected company-year rows:"
            ).classes("text-sm text-gray-600")

        job_options = {
            "listings_annual": "Sync Annual Listings",
            "listings_financial_statement": "Sync Financial Statement Listings",
            "download_annual": "Download Annual PDFs",
            "download_financial_statement": "Download Financial Statement PDFs",
            "convert": "Convert to Markdown",
            "convert_financial_statement": "Convert Financial Statements to Markdown",
            "load": "Load Markdown",
            "embed": "Embed",
            "infer_edc": "Infer EDC",
            "infer_proper": "Infer PROPER-VN",
            "infer_governance": "Extract Governance",
            "infer_financial_statement_audit": "Extract Financial Statement Audit",
        }
        run_jobs_sel = (
            ui.select(
                label="Jobs",
                options=job_options,
                multiple=True,
                value=[
                    "listings_annual",
                    "download_annual",
                    "convert",
                    "load",
                    "embed",
                    "infer_edc",
                    "infer_proper",
                    "infer_governance",
                    "infer_financial_statement_audit",
                ],
            )
            .props(
                "dense options-dense use-chips outlined "
                'style="min-width: 100%"'
            )
            .classes("w-full")
        )

        all_filtered_toggle = ui.switch("Use all filtered rows").props("dense")

        run_state = TaskState()

        def _selected_tickers() -> list[str] | None:
            values = ticker_sel.value or []
            return [str(t).strip().upper() for t in values if str(t).strip()] or None

        def _inference_model() -> str:
            return str(inference_model_input.value or INFERENCE_MODEL).strip()

        def _load_ticker_options() -> None:
            con = get_connection()
            try:
                rows = con.execute(
                    "SELECT ticker FROM companies ORDER BY ticker"
                ).fetchall()
            finally:
                con.close()
            ticker_sel.options = {r[0]: r[0] for r in rows}

        _load_ticker_options()

        @ui.refreshable
        def matrix_table():
            sy = int(start_year_input.value or DEFAULT_START_YEAR)
            ey = int(end_year_input.value or DEFAULT_END_YEAR)
            if sy > ey:
                sy, ey = ey, sy

            selected = _selected_tickers()

            con = get_connection()
            try:
                ticker_where_sql = ""
                ticker_where_params: list[str] = []
                if selected:
                    placeholders = ", ".join(["?"] * len(selected))
                    ticker_where_sql = f" AND u.ticker IN ({placeholders})"
                    ticker_where_params.extend(selected)

                rows = con.execute(
                    f"""
                    WITH universe AS (
                        SELECT DISTINCT ticker, year FROM annual_reports
                        WHERE year BETWEEN ? AND ?

                        UNION

                        SELECT DISTINCT ticker,
                               TRY_CAST(regexp_extract(title, '(\\d{{4}})', 1) AS INTEGER) AS year
                        FROM vietstock_documents
                        WHERE TRY_CAST(regexp_extract(title, '(\\d{{4}})', 1) AS INTEGER)
                              BETWEEN ? AND ?

                        UNION

                        SELECT DISTINCT ticker, year FROM conversion_jobs
                        WHERE year BETWEEN ? AND ?

                        UNION

                        SELECT DISTINCT ticker, year FROM document_embeddings
                        WHERE year BETWEEN ? AND ?
                    )
                    SELECT
                        u.ticker,
                        u.year,
                        EXISTS (
                            SELECT 1 FROM vietstock_documents vd
                            WHERE vd.ticker = u.ticker
                              AND vd.doc_type = '2'
                              AND TRY_CAST(regexp_extract(vd.title, '(\\d{{4}})', 1) AS INTEGER) = u.year
                        ) AS annual_listed,
                        EXISTS (
                            SELECT 1 FROM vietstock_documents vd
                            WHERE vd.ticker = u.ticker
                              AND vd.doc_type = '1'
                              AND TRY_CAST(regexp_extract(vd.title, '(\\d{{4}})', 1) AS INTEGER) = u.year
                                                ) AS financial_statement_listed,
                        EXISTS (
                            SELECT 1 FROM vietstock_documents vd
                            WHERE vd.ticker = u.ticker
                              AND vd.doc_type = '2'
                              AND vd.synced_to_raw = TRUE
                              AND TRY_CAST(regexp_extract(vd.title, '(\\d{{4}})', 1) AS INTEGER) = u.year
                        ) AS annual_downloaded,
                        EXISTS (
                            SELECT 1 FROM vietstock_documents vd
                            WHERE vd.ticker = u.ticker
                              AND vd.doc_type = '1'
                              AND vd.synced_to_raw = TRUE
                              AND TRY_CAST(regexp_extract(vd.title, '(\\d{{4}})', 1) AS INTEGER) = u.year
                                                ) AS financial_statement_downloaded,
                        EXISTS (
                            SELECT 1 FROM conversion_jobs cj
                            WHERE cj.ticker = u.ticker
                              AND cj.year = u.year
                              AND cj.status = 'completed'
                        ) AS converted,
                        EXISTS (
                            SELECT 1 FROM annual_reports ar
                            WHERE ar.ticker = u.ticker AND ar.year = u.year
                        ) AS loaded,
                        EXISTS (
                            SELECT 1 FROM document_embeddings de
                            WHERE de.ticker = u.ticker AND de.year = u.year
                        ) AS embedded,
                        EXISTS (
                            SELECT 1 FROM inference_jobs ij
                            WHERE ij.ticker = u.ticker AND ij.year = u.year
                              AND ij.status = 'completed'
                        ) AS edc_done,
                        EXISTS (
                            SELECT 1 FROM proper_vn_jobs pj
                            WHERE pj.ticker = u.ticker AND pj.year = u.year
                              AND pj.status = 'completed'
                        ) AS proper_done,
                        EXISTS (
                            SELECT 1 FROM governance_jobs gj
                            WHERE gj.ticker = u.ticker AND gj.year = u.year
                              AND gj.status = 'completed'
                        ) AS governance_done
                    FROM universe u
                    WHERE u.ticker IS NOT NULL
                      AND u.ticker != ''
                      AND u.year IS NOT NULL
                      AND u.year BETWEEN ? AND ?
                                            {ticker_where_sql}
                    ORDER BY u.ticker, u.year DESC
                    """,
                    [
                        sy,
                        ey,
                        sy,
                        ey,
                        sy,
                        ey,
                        sy,
                        ey,
                                                sy,
                                                ey,
                                                *ticker_where_params,
                    ],
                ).fetchall()
            finally:
                con.close()

            def _ok(flag: bool) -> str:
                return "✅" if bool(flag) else "❌"

            table_rows = [
                {
                    "key": f"{ticker}:{year}",
                    "ticker": ticker,
                    "year": year,
                    "annual_listed": _ok(annual_listed),
                    "financial_statement_listed": _ok(financial_statement_listed),
                    "annual_downloaded": _ok(annual_downloaded),
                    "financial_statement_downloaded": _ok(financial_statement_downloaded),
                    "converted": _ok(converted),
                    "loaded": _ok(loaded),
                    "embedded": _ok(embedded),
                    "edc_done": _ok(edc_done),
                    "proper_done": _ok(proper_done),
                    "governance_done": _ok(governance_done),
                }
                for (
                    ticker,
                    year,
                    annual_listed,
                    financial_statement_listed,
                    annual_downloaded,
                    financial_statement_downloaded,
                    converted,
                    loaded,
                    embedded,
                    edc_done,
                    proper_done,
                    governance_done,
                ) in rows
            ]

            columns = [
                {"name": "ticker", "label": "Ticker", "field": "ticker", "align": "left"},
                {"name": "year", "label": "Year", "field": "year", "align": "center"},
                {"name": "annual_listed", "label": "Annual Report Listed", "field": "annual_listed", "align": "center"},
                {"name": "financial_statement_listed", "label": "Financial Statement Listed", "field": "financial_statement_listed", "align": "center"},
                {"name": "annual_downloaded", "label": "Annual Downloaded", "field": "annual_downloaded", "align": "center"},
                {"name": "financial_statement_downloaded", "label": "Financial Statement Downloaded", "field": "financial_statement_downloaded", "align": "center"},
                {"name": "converted", "label": "Converted", "field": "converted", "align": "center"},
                {"name": "loaded", "label": "Loaded", "field": "loaded", "align": "center"},
                {"name": "embedded", "label": "Embedded", "field": "embedded", "align": "center"},
                {"name": "edc_done", "label": "EDC", "field": "edc_done", "align": "center"},
                {"name": "proper_done", "label": "PROPER", "field": "proper_done", "align": "center"},
                {"name": "governance_done", "label": "Governance", "field": "governance_done", "align": "center"},
            ]

            tbl = (
                ui.table(
                    columns=columns,
                    rows=table_rows,
                    row_key="key",
                    selection="multiple",
                    pagination={"rowsPerPage": 50},
                )
                .classes("w-full")
                .props("dense flat")
            )

            if table_rows:
                ui.label(f"{len(table_rows)} company-year rows").classes(
                    "text-xs text-gray-500"
                )
            else:
                ui.label("No rows found for selected filters").classes(
                    "text-gray-500"
                )

            def run_selected_rows():
                if bool(all_filtered_toggle.value):
                    selected_rows = list(table_rows)
                else:
                    selected_rows_raw = list(tbl.selected or [])
                    rows_by_key = {str(r.get("key", "")): r for r in table_rows}
                    selected_rows: list[dict] = []

                    # NiceGUI/QTable selection can yield row dicts or row keys.
                    for item in selected_rows_raw:
                        if isinstance(item, dict):
                            selected_rows.append(item)
                            continue
                        row = rows_by_key.get(str(item))
                        if row:
                            selected_rows.append(row)

                selected_jobs = list(run_jobs_sel.value or [])
                if not selected_rows:
                    msg = (
                        "No rows in current filtered result"
                        if bool(all_filtered_toggle.value)
                        else "No company-year rows selected"
                    )
                    ui.notify(msg, type="warning")
                    return
                if not selected_jobs:
                    ui.notify("No jobs selected", type="warning")
                    return

                ui.notify(
                    f"Starting {len(selected_rows)} row(s) x {len(selected_jobs)} job(s)",
                    type="info",
                )

                def _runner():
                    con = get_connection()
                    try:
                        init_db(con)
                        run_state.rows = []
                        run_state.summary = ""

                        total = len(selected_rows) * len(selected_jobs)
                        done = 0
                        failed = 0

                        for row in selected_rows:
                            ticker = str(row.get("ticker", "")).upper()
                            year = int(row.get("year", 0))
                            if not ticker or year <= 0:
                                failed += len(selected_jobs)
                                run_state.rows.append(
                                    TaskRow(
                                        label=str(row),
                                        detail="Invalid selected row payload",
                                        status="error",
                                    )
                                )
                                continue
                            for job_key in selected_jobs:
                                label = f"{ticker}-{year} | {job_options.get(job_key, job_key)}"
                                try:
                                    if job_key == "listings_annual":
                                        fetch_all_companies_documents(
                                            con,
                                            doc_type=DOC_TYPE_ANNUAL_REPORT,
                                            tickers=[ticker],
                                            start_year=year,
                                            end_year=year,
                                        )
                                    elif job_key == "listings_financial_statement":
                                        fetch_all_companies_documents(
                                            con,
                                            doc_type=DOC_TYPE_AUDITED_CONSOLIDATED_FS,
                                            tickers=[ticker],
                                            start_year=year,
                                            end_year=year,
                                        )
                                    elif job_key in {"download_annual", "download_financial_statement"}:
                                        doc_type = (
                                            DOC_TYPE_ANNUAL_REPORT
                                            if job_key == "download_annual"
                                            else DOC_TYPE_AUDITED_CONSOLIDATED_FS
                                        )
                                        doc_rows = con.execute(
                                            """
                                            SELECT id
                                            FROM vietstock_documents
                                            WHERE ticker = ?
                                              AND doc_type = ?
                                              AND synced_to_raw = FALSE
                                              AND file_url IS NOT NULL
                                              AND file_url != ''
                                              AND TRY_CAST(regexp_extract(title, '(\\d{4})', 1) AS INTEGER) = ?
                                            ORDER BY published_date DESC
                                            """,
                                            [ticker, doc_type, year],
                                        ).fetchall()
                                        for (doc_id,) in doc_rows:
                                            download_document_to_raw(con, int(doc_id))
                                    elif job_key == "convert":
                                        create_jobs(
                                            con,
                                            tickers=[ticker],
                                            years=[year],
                                        )
                                        job_row = con.execute(
                                            "SELECT id, status FROM conversion_jobs WHERE ticker = ? AND year = ?",
                                            [ticker, year],
                                        ).fetchone()
                                        if not job_row:
                                            raise RuntimeError("conversion job not found")
                                        job_id, status = job_row
                                        if bool(force_toggle.value) and status in {
                                            "completed",
                                            "failed",
                                            "cancelled",
                                        }:
                                            con.execute(
                                                """
                                                UPDATE conversion_jobs
                                                SET status = 'pending',
                                                    error_message = NULL,
                                                    failed_step = NULL,
                                                    started_at = NULL,
                                                    completed_at = NULL,
                                                    pid = NULL
                                                WHERE id = ?
                                                """,
                                                [job_id],
                                            )
                                            status = "pending"
                                        if status != "completed":
                                            ok = run_job(con, int(job_id))
                                            if not ok:
                                                raise RuntimeError("conversion failed")
                                    elif job_key == "convert_financial_statement":
                                        convert_financial_statement_to_markdown(
                                            con,
                                            ticker,
                                            year,
                                        )
                                    elif job_key == "load":
                                        sync_markdown_files(
                                            con,
                                            tickers=[ticker],
                                            years=[year],
                                        )
                                    elif job_key == "embed":
                                        embed_report(
                                            con,
                                            ticker,
                                            year,
                                            replace=bool(force_toggle.value),
                                        )
                                    elif job_key == "infer_edc":
                                        infer_report(
                                            ticker,
                                            year,
                                            con=con,
                                            replace=bool(force_toggle.value),
                                            inference_model=_inference_model(),
                                        )
                                    elif job_key == "infer_proper":
                                        infer_proper_vn_report(
                                            ticker,
                                            year,
                                            con=con,
                                            replace=bool(force_toggle.value),
                                            inference_model=_inference_model(),
                                        )
                                    elif job_key == "infer_governance":
                                        extract_governance(
                                            ticker,
                                            year,
                                            con=con,
                                            replace=bool(force_toggle.value),
                                            inference_model=_inference_model(),
                                        )
                                    elif job_key == "infer_financial_statement_audit":
                                        extract_financial_statement_audit_info(
                                            ticker,
                                            year,
                                            con=con,
                                            replace=bool(force_toggle.value),
                                            inference_model=_inference_model(),
                                        )

                                    run_state.rows.append(
                                        TaskRow(
                                            label=label,
                                            detail="OK",
                                            status="done",
                                        )
                                    )
                                    done += 1
                                except Exception as exc:
                                    run_state.rows.append(
                                        TaskRow(
                                            label=label,
                                            detail=str(exc)[:160],
                                            status="error",
                                        )
                                    )
                                    failed += 1
                                run_state.progress = done / max(total, 1)
                                run_status.refresh()

                        run_state.summary = (
                            f"✅ {done} succeeded, {failed} failed over {total} tasks"
                        )
                        matrix_table.refresh()
                    finally:
                        con.close()

                _run_in_thread(_runner, run_state, run_status.refresh)

            with ui.row().classes("gap-2 mt-2"):
                ui.button(
                    "Refresh Matrix",
                    on_click=lambda: matrix_table.refresh(),
                ).props("outline")
                ui.button(
                    "Select All Filtered",
                    on_click=lambda: (
                        setattr(tbl, "selected", list(table_rows)),
                        tbl.update(),
                        ui.notify(f"Selected {len(table_rows)} row(s)", type="info"),
                    ),
                ).props("outline")
                ui.button(
                    "Clear Selection",
                    on_click=lambda: (tbl.selected.clear(), tbl.update()),
                ).props("outline")
                ui.button(
                    "Run Selected Rows",
                    on_click=run_selected_rows,
                    color="primary",
                )

        matrix_table()

        @ui.refreshable
        def run_status():
            task_card(
                "Matrix Job Runner",
                run_state,
                lambda: None,
                run_status.refresh,
            )

        run_status()


@ui.page("/llm-tasks")
def page_llm_tasks():
    ui.dark_mode(False)
    _nav_header()
    init_db()

    with ui.column().classes("w-full max-w-6xl mx-auto gap-4 p-4"):
        ui.label("LLM Task Manager").classes("text-2xl font-bold")
        ui.label(
            "Run embedding and HyDE2 LLM extraction tasks with ticker/year filters."
        ).classes("text-sm text-gray-600")

        year_now = datetime.now().year
        year_options = list(range(DEFAULT_START_YEAR, year_now + 1))

        default_embed_models = [
            EMBEDDING_MODEL,
            "text-embedding-3-small",
            "text-embedding-3-large",
        ]
        default_infer_models = [
            INFERENCE_MODEL,
            "gpt-4.1-mini",
            "gpt-4.1",
            "gpt-4o-mini",
            "gpt-4o",
            "o3-mini",
            "o4-mini",
            "gpt-5-mini",
        ]
        EDC_TOTAL = 18
        PROPER_TOTAL = 7
        GOV_TOTAL = 4
        FINANCIAL_STATEMENT_AUDIT_TOTAL = 1
        GOVERNANCE_ANNUAL_ITEM_CODES = [
            "GOV_SHAREHOLDERS",
            "GOV_DIRECTORY",
            "GOV_EXECUTIVE",
            "GOV_SUPERVISORY",
        ]

        ticker_picker_state: dict[str, list[str]] = {"selected": []}
        model_store_state: dict[str, list[str]] = {
            "embedding": [],
            "inference": [],
        }

        def _ensure_model_presets() -> None:
            con = get_connection()
            try:
                init_db(con)
                for model in default_embed_models:
                    con.execute(
                        """
                        INSERT INTO llm_model_presets (task_type, model_name, is_active)
                        VALUES ('embedding', ?, TRUE)
                        ON CONFLICT (task_type, model_name) DO UPDATE SET is_active = TRUE
                        """,
                        [model],
                    )
                for model in default_infer_models:
                    con.execute(
                        """
                        INSERT INTO llm_model_presets (task_type, model_name, is_active)
                        VALUES ('inference', ?, TRUE)
                        ON CONFLICT (task_type, model_name) DO UPDATE SET is_active = TRUE
                        """,
                        [model],
                    )
            finally:
                con.close()

        def _load_model_presets() -> None:
            con = get_connection()
            try:
                init_db(con)
                emb_rows = con.execute(
                    "SELECT model_name FROM llm_model_presets "
                    "WHERE task_type = 'embedding' AND is_active = TRUE "
                    "ORDER BY created_at, model_name"
                ).fetchall()
                inf_rows = con.execute(
                    "SELECT model_name FROM llm_model_presets "
                    "WHERE task_type = 'inference' AND is_active = TRUE "
                    "ORDER BY created_at, model_name"
                ).fetchall()
            finally:
                con.close()
            model_store_state["embedding"] = [r[0] for r in emb_rows]
            model_store_state["inference"] = [r[0] for r in inf_rows]

        def _add_model_preset(task_type: str, model_name: str) -> None:
            value = model_name.strip()
            if not value:
                raise ValueError("Model name must not be empty")
            con = get_connection()
            try:
                init_db(con)
                con.execute(
                    """
                    INSERT INTO llm_model_presets (task_type, model_name, is_active)
                    VALUES (?, ?, TRUE)
                    ON CONFLICT (task_type, model_name) DO UPDATE SET is_active = TRUE
                    """,
                    [task_type, value],
                )
            finally:
                con.close()

        _ensure_model_presets()
        _load_model_presets()

        with ui.card().classes("w-full"):
            ui.label("Run Settings").classes("text-lg font-bold")

            with ui.row().classes("items-end gap-3 flex-wrap w-full"):
                ticker_sel = (
                    ui.select(
                        label="Tickers (optional)",
                        options={},
                        multiple=True,
                        with_input=True,
                    )
                    .props(
                        "dense options-dense use-chips outlined "
                        'style="min-width: 360px"'
                    )
                    .classes("flex-1")
                )

                start_year_input = (
                    ui.select(
                        label="Start year",
                        options=year_options,
                        value=DEFAULT_START_YEAR,
                    )
                    .props("dense outlined")
                    .classes("w-32")
                )

                end_year_input = (
                    ui.select(
                        label="End year",
                        options=year_options,
                        value=DEFAULT_END_YEAR,
                    )
                    .props("dense outlined")
                    .classes("w-32")
                )

                scope_sel = (
                    ui.select(
                        label="Processing scope",
                        options={
                            "all": "All reports",
                            "any_unprocessed": "Any unprocessed stage",
                            "needs_embedding": "Need embedding",
                            "needs_financial_statement_embedding": "Need financial-statement embedding",
                            "needs_edc": "Need EDC",
                            "needs_proper": "Need PROPER-VN",
                            "needs_governance": "Need governance",
                            "needs_financial_statement_audit": "Need financial-statement audit extraction",
                            "fully_processed": "Fully processed",
                        },
                        value="all",
                    )
                    .props("dense outlined")
                    .classes("w-60")
                )

                force_toggle = ui.switch("Force re-run").props("dense")

            ticker_summary = ui.label("").classes("text-xs text-gray-500")

            with ui.tabs().classes("w-full") as setting_tabs:
                ui.tab("embedding", label="Embedding")
                ui.tab("inference", label="Inference")

            with ui.tab_panels(setting_tabs, value="embedding").classes("w-full"):
                with ui.tab_panel("embedding"):
                    with ui.row().classes("items-end gap-2 flex-wrap"):
                        embed_model_input = (
                            ui.select(
                                label="Embedding model",
                                options=model_store_state["embedding"],
                                value=EMBEDDING_MODEL,
                            )
                            .props("dense outlined")
                            .classes("w-56")
                        )
                        chunk_size_input = (
                            ui.number(
                                "Chunk size",
                                value=EMBEDDING_CHUNK_SIZE,
                                min=32,
                                max=8192,
                                step=32,
                                format="%.0f",
                            )
                            .props("dense outlined")
                            .classes("w-32")
                        )
                        add_embed_model_input = (
                            ui.input("Add embedding model")
                            .props("dense outlined")
                            .classes("w-56")
                        )
                        embed_all_toggle = ui.switch("Embed all available documents")
                        embed_overwrite_toggle = ui.switch("Overwrite existing embeddings")
                        embed_target_sel = (
                            ui.select(
                                label="Embedding target",
                                options={
                                    "annual": "Annual reports",
                                    "financial_statement": "Financial statements",
                                    "both": "Both annual + financial statement",
                                },
                                value="annual",
                            )
                            .props("dense outlined")
                            .classes("w-64")
                        )
                    ui.label(
                        "When enabled, embedding ignores ticker/year/scope filters and runs for every report in the selected embedding target."
                    ).classes("text-xs text-gray-500")

                with ui.tab_panel("inference"):
                    with ui.row().classes("items-end gap-2 flex-wrap"):
                        infer_model_input = (
                            ui.select(
                                label="Inference model",
                                options=model_store_state["inference"],
                                value=INFERENCE_MODEL,
                            )
                            .props("dense outlined")
                            .classes("w-56")
                        )
                        top_k_input = (
                            ui.number(
                                "Top K",
                                value=INFERENCE_TOP_K,
                                min=1,
                                max=50,
                                step=1,
                                format="%.0f",
                            )
                            .props("dense outlined")
                            .classes("w-28")
                        )
                        add_infer_model_input = (
                            ui.input("Add inference model")
                            .props("dense outlined")
                            .classes("w-56")
                        )
                        edc_mode_sel = (
                            ui.select(
                                label="EDC mode",
                                options={
                                    "edc": "EDC (Strict)",
                                    "edc_alt": "EDC (Alternative)",
                                    "edc_alt_two": "EDC (Alternative 2)",
                                },
                                value="edc",
                            )
                            .props("dense outlined")
                            .classes("w-56")
                        )
                    ui.label(
                        "EDC alternatives use *_alt and *_alt_two checklist items and prime criteria embeddings before running."
                    ).classes("text-xs text-gray-500")

            def _refresh_model_selects() -> None:
                _load_model_presets()
                embed_model_input.options = model_store_state["embedding"]
                if embed_model_input.value not in model_store_state["embedding"]:
                    embed_model_input.value = (
                        model_store_state["embedding"][0]
                        if model_store_state["embedding"]
                        else EMBEDDING_MODEL
                    )
                embed_model_input.update()

                infer_model_input.options = model_store_state["inference"]
                if infer_model_input.value not in model_store_state["inference"]:
                    infer_model_input.value = (
                        model_store_state["inference"][0]
                        if model_store_state["inference"]
                        else INFERENCE_MODEL
                    )
                infer_model_input.update()

            def _add_embed_model() -> None:
                try:
                    _add_model_preset("embedding", str(add_embed_model_input.value or ""))
                    add_embed_model_input.value = ""
                    _refresh_model_selects()
                    ui.notify("Embedding model saved")
                except Exception as exc:
                    ui.notify(str(exc), type="warning")

            def _add_infer_model() -> None:
                try:
                    _add_model_preset("inference", str(add_infer_model_input.value or ""))
                    add_infer_model_input.value = ""
                    _refresh_model_selects()
                    ui.notify("Inference model saved")
                except Exception as exc:
                    ui.notify(str(exc), type="warning")

            with ui.row().classes("items-end gap-2 flex-wrap"):
                ui.button("Save Embedding Model", on_click=_add_embed_model).props(
                    "dense outline"
                )
                ui.button("Save Inference Model", on_click=_add_infer_model).props(
                    "dense outline"
                )
                ui.button("Reload Model Presets", on_click=_refresh_model_selects).props(
                    "dense outline"
                )

            def _refresh_tickers() -> None:
                con = get_connection()
                try:
                    rows = con.execute(
                        "SELECT ticker FROM companies ORDER BY ticker"
                    ).fetchall()
                finally:
                    con.close()

                options = {r[0]: r[0] for r in rows}
                selected = [
                    t for t in ticker_picker_state["selected"] if t in options
                ]
                ticker_picker_state["selected"] = selected
                ticker_sel.options = options
                ticker_sel.value = selected
                ticker_sel.update()
                ticker_summary.text = f"{len(options)} tickers available"

            def _sync_selected_tickers(_=None):
                ticker_picker_state["selected"] = list(ticker_sel.value or [])

            ticker_sel.on_value_change(_sync_selected_tickers)

            with ui.row().classes("gap-2 mt-2"):
                ui.button("Refresh Tickers", on_click=_refresh_tickers).props(
                    "dense outline"
                )

            _refresh_tickers()

        def _selected_tickers() -> list[str] | None:
            val = list(ticker_sel.value or [])
            return val or None

        def _selected_years() -> list[int]:
            sy = int(start_year_input.value or DEFAULT_START_YEAR)
            ey = int(end_year_input.value or DEFAULT_END_YEAR)
            if sy > ey:
                raise ValueError("Start year must be less than or equal to end year")
            return list(range(sy, ey + 1))

        def _selected_embed_model() -> str:
            return str(embed_model_input.value or EMBEDDING_MODEL)

        def _selected_infer_model() -> str:
            return str(infer_model_input.value or INFERENCE_MODEL)

        def _selected_chunk_size() -> int:
            return max(1, int(chunk_size_input.value or EMBEDDING_CHUNK_SIZE))

        def _selected_top_k() -> int:
            return max(1, int(top_k_input.value or 5))

        def _selected_edc_task_type() -> str:
            task_type = str(edc_mode_sel.value or "edc")
            return (
                task_type
                if task_type in {"edc", "edc_alt", "edc_alt_two"}
                else "edc"
            )

        def _selected_scope() -> str:
            return str(scope_sel.value or "all")

        def _selected_force() -> bool:
            return bool(force_toggle.value)

        def _selected_embed_all() -> bool:
            return bool(embed_all_toggle.value)

        def _selected_embed_overwrite() -> bool:
            return bool(embed_overwrite_toggle.value)

        def _selected_embed_target() -> str:
            return str(embed_target_sel.value or "annual")

        def _processing_rows(con) -> list[ProcessingStatusRow]:
            rows = con.execute(
                """
                WITH base AS (
                    SELECT ticker, year FROM annual_reports
                    UNION
                    SELECT ticker, year FROM financial_statement_reports
                ),
                emb AS (
                    SELECT ticker, year, COUNT(*) AS c
                    FROM document_embeddings
                    WHERE model = ?
                    GROUP BY ALL
                ),
                financial_statement_emb AS (
                    SELECT ticker, year, COUNT(*) AS c
                    FROM financial_statement_document_embeddings
                    WHERE model = ?
                    GROUP BY ALL
                ),
                edc AS (
                    SELECT ticker, year, COUNT(DISTINCT category_code) AS c
                    FROM inference_results_hyde2
                    WHERE model = ?
                    GROUP BY ALL
                ),
                proper AS (
                    SELECT ticker, year, COUNT(DISTINCT indicator_code) AS c
                    FROM proper_vn_results_hyde2
                    WHERE model = ?
                    GROUP BY ALL
                ),
                gov AS (
                    SELECT ticker, year, COUNT(DISTINCT item_code) AS c
                    FROM governance_results_hyde2
                    WHERE model = ? AND item_code != 'GOV_AUDIT'
                    GROUP BY ALL
                ),
                financial_statement_audit AS (
                    SELECT ticker, year, COUNT(*) AS c
                    FROM financial_statement_audit_results
                    WHERE model = ?
                    GROUP BY ALL
                )
                SELECT
                    b.ticker,
                    b.year,
                    COALESCE(emb.c, 0) AS annual_emb_chunks,
                    COALESCE(financial_statement_emb.c, 0) AS financial_statement_emb_chunks,
                    EXISTS (
                        SELECT 1 FROM annual_reports ar
                        WHERE ar.ticker = b.ticker AND ar.year = b.year
                    ) AS has_annual,
                    EXISTS (
                        SELECT 1 FROM financial_statement_reports br
                        WHERE br.ticker = b.ticker AND br.year = b.year
                    ) AS has_financial_statement,
                    COALESCE(edc.c, 0) AS edc_count,
                    COALESCE(proper.c, 0) AS proper_count,
                    COALESCE(gov.c, 0) AS gov_count,
                    COALESCE(financial_statement_audit.c, 0) AS financial_statement_audit_count
                FROM base b
                LEFT JOIN emb ON emb.ticker = b.ticker AND emb.year = b.year
                LEFT JOIN financial_statement_emb ON financial_statement_emb.ticker = b.ticker AND financial_statement_emb.year = b.year
                LEFT JOIN edc ON edc.ticker = b.ticker AND edc.year = b.year
                LEFT JOIN proper ON proper.ticker = b.ticker AND proper.year = b.year
                LEFT JOIN gov ON gov.ticker = b.ticker AND gov.year = b.year
                LEFT JOIN financial_statement_audit ON financial_statement_audit.ticker = b.ticker AND financial_statement_audit.year = b.year
                ORDER BY b.ticker, b.year
                """,
                [
                    _selected_embed_model(),
                    _selected_embed_model(),
                    _selected_infer_model(),
                    _selected_infer_model(),
                    _selected_infer_model(),
                    _selected_infer_model(),
                ],
            ).fetchall()

            return [
                {
                    "ticker": r[0],
                    "year": int(r[1]),
                    "annual_emb_chunks": int(r[2]),
                    "financial_statement_emb_chunks": int(r[3]),
                    "annual_embedded": int(r[2]) > 0,
                    "financial_statement_embedded": int(r[3]) > 0,
                    "has_annual": bool(r[4]),
                    "has_financial_statement": bool(r[5]),
                    "edc_done": int(r[6]) >= EDC_TOTAL,
                    "proper_done": int(r[7]) >= PROPER_TOTAL,
                    "gov_done": int(r[8]) >= GOV_TOTAL,
                    "financial_statement_audit_done": int(r[9]) >= FINANCIAL_STATEMENT_AUDIT_TOTAL,
                }
                for r in rows
            ]

        def _row_matches_scope(row: ProcessingStatusRow, scope: str) -> bool:
            annual_embedded = bool(row["annual_embedded"])
            financial_statement_embedded = bool(row["financial_statement_embedded"])
            has_annual = bool(row["has_annual"])
            has_financial_statement = bool(row["has_financial_statement"])
            edc_done = bool(row["edc_done"])
            proper_done = bool(row["proper_done"])
            gov_done = bool(row["gov_done"])
            financial_statement_audit_done = bool(row["financial_statement_audit_done"])
            if scope == "all":
                return True
            if scope == "needs_embedding":
                return has_annual and not annual_embedded
            if scope == "needs_financial_statement_embedding":
                return has_financial_statement and not financial_statement_embedded
            if scope == "needs_edc":
                return has_annual and annual_embedded and not edc_done
            if scope == "needs_proper":
                return has_annual and annual_embedded and not proper_done
            if scope == "needs_governance":
                return has_annual and annual_embedded and not gov_done
            if scope == "needs_financial_statement_audit":
                return has_financial_statement and financial_statement_embedded and not financial_statement_audit_done
            if scope == "fully_processed":
                annual_ok = (not has_annual) or (
                    annual_embedded and edc_done and proper_done and gov_done
                )
                financial_statement_ok = (not has_financial_statement) or (
                    financial_statement_embedded and financial_statement_audit_done
                )
                return annual_ok and financial_statement_ok
            if scope == "any_unprocessed":
                annual_ok = (not has_annual) or (
                    annual_embedded and edc_done and proper_done and gov_done
                )
                financial_statement_ok = (not has_financial_statement) or (
                    financial_statement_embedded and financial_statement_audit_done
                )
                return not (annual_ok and financial_statement_ok)
            return True

        def _target_pairs(con) -> list[tuple[str, int]]:
            rows = _processing_rows(con)
            selected_tickers = _selected_tickers()
            years = set(_selected_years())
            scope = _selected_scope()
            filtered = rows
            if selected_tickers:
                tset = {t.upper() for t in selected_tickers}
                filtered = [r for r in filtered if r["ticker"] in tset]
            filtered = [r for r in filtered if int(r["year"]) in years]
            filtered = [r for r in filtered if _row_matches_scope(r, scope)]
            return [(str(r["ticker"]), int(r["year"])) for r in filtered]

        def _all_report_pairs(con) -> list[tuple[str, int]]:
            rows = con.execute(
                "SELECT DISTINCT ticker, year FROM annual_reports ORDER BY ticker, year"
            ).fetchall()
            return [(str(r[0]), int(r[1])) for r in rows]

        def _all_financial_statement_pairs(con) -> list[tuple[str, int]]:
            rows = con.execute(
                "SELECT DISTINCT ticker, year FROM financial_statement_reports ORDER BY ticker, year"
            ).fetchall()
            return [(str(r[0]), int(r[1])) for r in rows]

        def _target_financial_statement_pairs(con) -> list[tuple[str, int]]:
            rows = _processing_rows(con)
            selected_tickers = _selected_tickers()
            years = set(_selected_years())
            scope = _selected_scope()
            filtered = [r for r in rows if bool(r["has_financial_statement"])]
            if selected_tickers:
                tset = {t.upper() for t in selected_tickers}
                filtered = [r for r in filtered if r["ticker"] in tset]
            filtered = [r for r in filtered if int(r["year"]) in years]
            filtered = [r for r in filtered if _row_matches_scope(r, scope)]
            return [(str(r["ticker"]), int(r["year"])) for r in filtered]

        embed_state = TaskState()
        embed_financial_statement_state = TaskState()
        fetch_batch_state = TaskState()
        infer_edc_state = TaskState()
        infer_proper_state = TaskState()
        infer_gov_state = TaskState()
        infer_financial_statement_audit_state = TaskState()

        def _filtered_processing_rows() -> list[ProcessingStatusRow]:
            con = get_connection()
            try:
                init_db(con)
                rows = _processing_rows(con)
            finally:
                con.close()

            selected_tickers = _selected_tickers()
            years = set(_selected_years())
            scope = _selected_scope()
            if selected_tickers:
                tset = {t.upper() for t in selected_tickers}
                rows = [r for r in rows if r["ticker"] in tset]
            rows = [r for r in rows if int(r["year"]) in years]
            rows = [r for r in rows if _row_matches_scope(r, scope)]
            return rows

        @ui.refreshable
        def embedding_status_panel():
            rows = _filtered_processing_rows()

            with ui.card().classes("w-full"):
                ui.label("Embedding Status by Ticker/Year").classes(
                    "text-lg font-bold"
                )
                if not rows:
                    ui.label("No rows for current filters").classes("text-gray-500")
                else:
                    ui.label(f"{len(rows)} ticker/year rows in scope").classes(
                        "text-xs text-gray-500"
                    )
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
                                "align": "center",
                            },
                            {
                                "name": "annual_chunks",
                                "label": "Annual Chunks",
                                "field": "annual_chunks",
                                "align": "right",
                            },
                            {
                                "name": "annual_embedded",
                                "label": "Annual Embedded",
                                "field": "annual_embedded",
                                "align": "center",
                            },
                            {
                                "name": "financial_statement_chunks",
                                "label": "Financial Statement Chunks",
                                "field": "financial_statement_chunks",
                                "align": "right",
                            },
                            {
                                "name": "financial_statement_embedded",
                                "label": "Financial Statement Embedded",
                                "field": "financial_statement_embedded",
                                "align": "center",
                            },
                        ],
                        rows=[
                            {
                                "ticker": r["ticker"],
                                "year": r["year"],
                                "annual_chunks": r["annual_emb_chunks"],
                                "annual_embedded": "✅" if r["annual_embedded"] else "⬜",
                                "financial_statement_chunks": r["financial_statement_emb_chunks"],
                                "financial_statement_embedded": "✅" if r["financial_statement_embedded"] else "⬜",
                            }
                            for r in rows
                        ],
                        row_key="ticker_year",
                    ).classes("w-full").props("dense flat")

            with ui.row().classes("gap-2"):
                ui.button("Refresh Embedding Status", on_click=embedding_status_panel.refresh).props(
                    "dense outline"
                )

        @ui.refreshable
        def processing_status_panel():
            rows = _filtered_processing_rows()

            with ui.card().classes("w-full"):
                ui.label("Processing Status by Ticker/Year").classes(
                    "text-lg font-bold"
                )
                if not rows:
                    ui.label("No rows for current filters").classes("text-gray-500")
                else:
                    ui.label(f"{len(rows)} ticker/year rows in scope").classes(
                        "text-xs text-gray-500"
                    )
                    ui.table(
                        columns=[
                            {"name": "ticker", "label": "Ticker", "field": "ticker", "align": "left"},
                            {"name": "year", "label": "Year", "field": "year", "align": "center"},
                            {"name": "annual_embed", "label": "Annual Embed", "field": "annual_embed", "align": "center"},
                            {"name": "financial_statement_embed", "label": "Financial Statement Embed", "field": "financial_statement_embed", "align": "center"},
                            {"name": "edc", "label": "EDC", "field": "edc", "align": "center"},
                            {"name": "proper", "label": "PROPER", "field": "proper", "align": "center"},
                            {"name": "gov", "label": "Governance", "field": "gov", "align": "center"},
                            {"name": "financial_statement_audit", "label": "Financial Statement Audit", "field": "financial_statement_audit", "align": "center"},
                        ],
                        rows=[
                            {
                                "ticker": r["ticker"],
                                "year": r["year"],
                                "annual_embed": "✅" if r["annual_embedded"] else "⬜",
                                "financial_statement_embed": "✅" if r["financial_statement_embedded"] else "⬜",
                                "edc": "✅" if r["edc_done"] else "⬜",
                                "proper": "✅" if r["proper_done"] else "⬜",
                                "gov": "✅" if r["gov_done"] else "⬜",
                                "financial_statement_audit": "✅" if r["financial_statement_audit_done"] else "⬜",
                            }
                            for r in rows
                        ],
                        row_key="ticker_year",
                    ).classes("w-full").props("dense flat")

            with ui.row().classes("gap-2"):
                ui.button("Refresh Status", on_click=processing_status_panel.refresh).props(
                    "dense outline"
                )

        def _refresh_status_panels() -> None:
            embedding_status_panel.refresh()
            processing_status_panel.refresh()

        def _make_llm_embed_task(
            state: TaskState,
            refresh: Callable,
            *,
            forced_target: str | None = None,
        ) -> Callable:
            def run():
                from llm_embeddings import embed_report
                from llm_financial_statement_audit import (
                    sync_financial_statement_reports_from_markdown,
                    embed_financial_statement_report,
                )

                con = get_connection()
                try:
                    init_db(con)

                    embed_target = forced_target or _selected_embed_target()
                    annual_pairs: list[tuple[str, int]] = []
                    financial_statement_pairs: list[tuple[str, int]] = []

                    if embed_target in {"annual", "both"}:
                        annual_pairs = (
                            _all_report_pairs(con)
                            if _selected_embed_all()
                            else _target_pairs(con)
                        )

                    if embed_target in {"financial_statement", "both"}:
                        sync_financial_statement_reports_from_markdown(
                            con,
                            tickers=_selected_tickers(),
                            years=_selected_years(),
                        )
                        financial_statement_pairs = (
                            _all_financial_statement_pairs(con)
                            if _selected_embed_all()
                            else _target_financial_statement_pairs(con)
                        )

                    if not annual_pairs and not financial_statement_pairs:
                        state.summary = (
                            "No reports match selected filters for embedding target"
                        )
                        return

                    embedded_reports = 0
                    embedded_chunks = 0
                    skipped_reports = 0
                    failed_reports = 0

                    for ticker, year in annual_pairs:
                        try:
                            out = embed_report(
                                con,
                                ticker,
                                year,
                                replace=_selected_embed_overwrite(),
                                model=_selected_embed_model(),
                                chunk_size=_selected_chunk_size(),
                            )
                            status = "done"
                            detail = f"{out.get('embedded', 0)} chunks"
                            if out.get("skipped"):
                                status = "skipped"
                                detail = out.get("reason", "skipped")
                                skipped_reports += 1
                            elif out.get("failed"):
                                status = "error"
                                detail = out.get("reason", "failed")
                                failed_reports += 1
                            else:
                                embedded_reports += 1
                                embedded_chunks += int(out.get("embedded", 0))
                        except Exception as exc:
                            status = "error"
                            detail = str(exc)[:100]
                            failed_reports += 1

                        state.rows.append(
                            TaskRow(
                                label=f"[ANNUAL] {ticker} / {year}",
                                detail=str(detail),
                                status=status,
                            )
                        )
                        refresh()

                    for ticker, year in financial_statement_pairs:
                        try:
                            out = embed_financial_statement_report(
                                con,
                                ticker,
                                year,
                                replace=_selected_embed_overwrite(),
                                model=_selected_embed_model(),
                                chunk_size=_selected_chunk_size(),
                            )
                            status = "done"
                            detail = f"{out.get('embedded', 0)} chunks"
                            if out.get("skipped"):
                                status = "skipped"
                                detail = out.get("reason", "skipped")
                                skipped_reports += 1
                            elif out.get("failed"):
                                status = "error"
                                detail = out.get("reason", "failed")
                                failed_reports += 1
                            else:
                                embedded_reports += 1
                                embedded_chunks += int(out.get("embedded", 0))
                        except Exception as exc:
                            status = "error"
                            detail = str(exc)[:100]
                            failed_reports += 1

                        state.rows.append(
                            TaskRow(
                                label=f"[FINANCIAL_STATEMENT] {ticker} / {year}",
                                detail=str(detail),
                                status=status,
                            )
                        )
                        refresh()

                    state.summary = (
                        f"✅ {embedded_reports} embedded, "
                        f"{skipped_reports} skipped, "
                        f"{failed_reports} failed "
                        f"({embedded_chunks:,} chunks)"
                    )
                finally:
                    con.close()
                _refresh_status_panels()

            return run

        def _make_fetch_batch_outputs_task(
            state: TaskState, refresh: Callable
        ) -> Callable:
            def run():
                from llm_financial_statement_audit import extract_financial_statement_audit_info
                from llm_governance import extract_governance
                from llm_inference import infer_report
                from llm_proper_vn import infer_proper_vn_report
                from hyde_retrieval import precompute_hyde2_embeddings

                con = get_connection()
                try:
                    init_db(con)
                    ensure_vss_loaded(con)

                    allowed_pairs = set(_target_pairs(con))
                    if not allowed_pairs:
                        state.summary = "No reports match current filters"
                        return

                    model = _selected_infer_model()
                    top_k = _selected_top_k()
                    embedding_model = _selected_embed_model()
                    edc_task_type = _selected_edc_task_type()
                    edc_item_configs = (
                        get_task_items(edc_task_type) if edc_task_type != "edc" else None
                    )

                    edc_jobs = con.execute(
                        """
                        SELECT ticker, year
                        FROM inference_jobs
                        WHERE model = ? AND batch_id IS NOT NULL
                          AND status IN ('running', 'pending')
                        ORDER BY ticker, year
                        """,
                        [model],
                    ).fetchall()
                    proper_jobs = con.execute(
                        """
                        SELECT ticker, year
                        FROM proper_vn_jobs
                        WHERE model = ? AND batch_id IS NOT NULL
                          AND status IN ('running', 'pending')
                        ORDER BY ticker, year
                        """,
                        [model],
                    ).fetchall()
                    gov_jobs = con.execute(
                        """
                        SELECT ticker, year
                        FROM governance_jobs
                        WHERE model = ? AND batch_id IS NOT NULL
                          AND status IN ('running', 'pending')
                        ORDER BY ticker, year
                        """,
                        [model],
                    ).fetchall()
                    financial_statement_audit_jobs = con.execute(
                        """
                        SELECT ticker, year
                        FROM financial_statement_audit_jobs
                        WHERE model = ? AND batch_id IS NOT NULL
                          AND status IN ('running', 'pending')
                        ORDER BY ticker, year
                        """,
                        [model],
                    ).fetchall()

                    edc_pairs = [
                        (str(t), int(y))
                        for t, y in edc_jobs
                        if (str(t), int(y)) in allowed_pairs
                    ]
                    proper_pairs = [
                        (str(t), int(y))
                        for t, y in proper_jobs
                        if (str(t), int(y)) in allowed_pairs
                    ]
                    gov_pairs = [
                        (str(t), int(y))
                        for t, y in gov_jobs
                        if (str(t), int(y)) in allowed_pairs
                    ]
                    financial_statement_audit_pairs = [
                        (str(t), int(y))
                        for t, y in financial_statement_audit_jobs
                        if (str(t), int(y)) in allowed_pairs
                    ]

                    total_jobs = (
                        len(edc_pairs)
                        + len(proper_pairs)
                        + len(gov_pairs)
                        + len(financial_statement_audit_pairs)
                    )
                    if total_jobs == 0:
                        state.summary = (
                            "No running/pending batch jobs with batch_id found for current filters"
                        )
                        return

                    if edc_pairs:
                        # Prime selected EDC criteria before fetching EDC batch outputs.
                        prime_row = TaskRow(
                            label=f"Prime EDC criteria ({edc_task_type})",
                            detail="Preparing HyDE2 embeddings for checklist descriptions...",
                            status="running",
                        )
                        state.rows.append(prime_row)
                        refresh()
                        try:
                            prime_items = get_task_items(edc_task_type)
                            precompute_hyde2_embeddings(
                                [it["description"] for it in prime_items],
                                embedding_model=embedding_model,
                            )
                            prime_row.status = "done"
                            prime_row.detail = f"Primed {len(prime_items)} checklist items"
                        except Exception as exc:
                            prime_row.status = "error"
                            prime_row.detail = str(exc)[:120]
                        refresh()

                    fetched = 0
                    waiting = 0
                    failed = 0
                    processed = 0

                    def _begin_step(item_type: str, ticker: str, year: int) -> int:
                        state.rows.append(
                            TaskRow(
                                label=f"{item_type} {ticker} / {year}",
                                detail="Fetching batch output...",
                                status="running",
                            )
                        )
                        refresh()
                        return len(state.rows) - 1

                    def _finish_step(
                        row_index: int,
                        status: str,
                        detail: str,
                    ) -> None:
                        nonlocal processed
                        state.rows[row_index].status = status
                        state.rows[row_index].detail = detail
                        processed += 1
                        state.progress = (
                            float(processed) / float(total_jobs)
                            if total_jobs > 0
                            else 0.0
                        )
                        refresh()

                    for ticker, year in edc_pairs:
                        row_idx = _begin_step("EDC", ticker, year)
                        try:
                            n = infer_report(
                                ticker,
                                year,
                                con=con,
                                replace=False,
                                top_k=top_k,
                                inference_model=model,
                                embedding_model=embedding_model,
                                item_configs=edc_item_configs,
                            )
                            post = con.execute(
                                "SELECT status, batch_id FROM inference_jobs "
                                "WHERE ticker = ? AND year = ? AND model = ?",
                                [ticker, year, model],
                            ).fetchone()
                            if post and post[1]:
                                status = "skipped"
                                detail = "batch still running on OpenAI"
                                waiting += 1
                            elif post and str(post[0]) == "failed":
                                status = "error"
                                detail = "batch fetch failed"
                                failed += 1
                            else:
                                status = "done"
                                detail = f"fetched {n} categories"
                                fetched += 1
                        except Exception as exc:
                            status = "error"
                            detail = str(exc)[:120]
                            failed += 1
                        _finish_step(row_idx, status, detail)

                    for ticker, year in proper_pairs:
                        row_idx = _begin_step("PROPER", ticker, year)
                        try:
                            out = infer_proper_vn_report(
                                ticker,
                                year,
                                con=con,
                                replace=False,
                                top_k=top_k,
                                inference_model=model,
                                embedding_model=embedding_model,
                            )
                            post = con.execute(
                                "SELECT status, batch_id, color FROM proper_vn_jobs "
                                "WHERE ticker = ? AND year = ? AND model = ?",
                                [ticker, year, model],
                            ).fetchone()
                            if post and post[1]:
                                status = "skipped"
                                detail = "batch still running on OpenAI"
                                waiting += 1
                            elif post and str(post[0]) == "failed":
                                status = "error"
                                detail = "batch fetch failed"
                                failed += 1
                            else:
                                color = str((post[2] if post else None) or out.get("color") or "")
                                status = "done"
                                detail = f"fetched output (color={color or 'n/a'})"
                                fetched += 1
                        except Exception as exc:
                            status = "error"
                            detail = str(exc)[:120]
                            failed += 1
                        _finish_step(row_idx, status, detail)

                    for ticker, year in gov_pairs:
                        row_idx = _begin_step("GOV", ticker, year)
                        try:
                            n = extract_governance(
                                ticker,
                                year,
                                con=con,
                                replace=False,
                                top_k=top_k,
                                inference_model=model,
                                embedding_model=embedding_model,
                            )
                            post = con.execute(
                                "SELECT status, batch_id FROM governance_jobs "
                                "WHERE ticker = ? AND year = ? AND model = ?",
                                [ticker, year, model],
                            ).fetchone()
                            if post and post[1]:
                                status = "skipped"
                                detail = "batch still running on OpenAI"
                                waiting += 1
                            elif post and str(post[0]) == "failed":
                                status = "error"
                                detail = "batch fetch failed"
                                failed += 1
                            else:
                                status = "done"
                                detail = f"fetched {n} items"
                                fetched += 1
                        except Exception as exc:
                            status = "error"
                            detail = str(exc)[:120]
                            failed += 1
                        _finish_step(row_idx, status, detail)

                    for ticker, year in financial_statement_audit_pairs:
                        row_idx = _begin_step("FINANCIAL_STATEMENT_AUDIT", ticker, year)
                        try:
                            out = extract_financial_statement_audit_info(
                                ticker,
                                year,
                                con=con,
                                replace=False,
                                top_k=top_k,
                                inference_model=model,
                                embedding_model=embedding_model,
                            )
                            fs_status = str(out.get("status") or "")
                            if fs_status in {"submitted", "running"}:
                                status = "skipped"
                                detail = "batch still running on OpenAI"
                                waiting += 1
                            else:
                                status = "done"
                                detail = (
                                    out.get("audit_firm")
                                    or out.get("audit_opinion")
                                    or (
                                        "audit extracted"
                                        if out.get("found")
                                        else "no audit evidence"
                                    )
                                )
                                fetched += 1
                        except Exception as exc:
                            status = "error"
                            detail = str(exc)[:120]
                            failed += 1
                        _finish_step(row_idx, status, detail)

                    state.summary = (
                        f"Checked {total_jobs} batch job(s): "
                        f"{fetched} fetched, {waiting} still running, {failed} failed"
                    )
                finally:
                    con.close()
                _refresh_status_panels()

            return run

        def _make_llm_infer_edc_task(
            state: TaskState, refresh: Callable
        ) -> Callable:
            def run():
                from llm_inference import infer_report
                from hyde_retrieval import precompute_hyde2_embeddings

                con = get_connection()
                try:
                    init_db(con)
                    ensure_vss_loaded(con)
                    edc_task_type = _selected_edc_task_type()
                    edc_item_configs = (
                        get_task_items(edc_task_type) if edc_task_type != "edc" else None
                    )

                    # Prime selected EDC criteria embeddings first so retrieval is ready.
                    prime_items = get_task_items(edc_task_type)
                    state.rows.append(
                        TaskRow(
                            label=f"Prime EDC criteria ({edc_task_type})",
                            detail="Preparing HyDE2 embeddings for checklist descriptions...",
                            status="running",
                        )
                    )
                    refresh()
                    try:
                        precompute_hyde2_embeddings(
                            [it["description"] for it in prime_items],
                            embedding_model=_selected_embed_model(),
                        )
                        state.rows[-1].status = "done"
                        state.rows[-1].detail = f"Primed {len(prime_items)} checklist items"
                    except Exception as exc:
                        state.rows[-1].status = "error"
                        state.rows[-1].detail = str(exc)[:100]
                        state.summary = "Failed while priming EDC criteria"
                        refresh()
                        return
                    refresh()

                    reports = _target_pairs(con)
                    reports = [
                        (t, y)
                        for t, y in reports
                        if con.execute(
                            "SELECT 1 FROM document_embeddings "
                            "WHERE ticker = ? AND year = ? LIMIT 1",
                            [t, y],
                        ).fetchone()
                    ]
                    if not reports:
                        state.summary = "No embedded reports match the selected filter"
                        return

                    ok = 0
                    fail = 0
                    skip = 0
                    for ticker, year in reports:
                        try:
                            n = infer_report(
                                ticker,
                                year,
                                con=con,
                                replace=_selected_force(),
                                top_k=_selected_top_k(),
                                inference_model=_selected_infer_model(),
                                embedding_model=_selected_embed_model(),
                                item_configs=edc_item_configs,
                            )
                            if n > 0:
                                status = "done"
                                detail = f"{n} categories"
                                ok += 1
                            else:
                                status = "skipped"
                                detail = "no new categories"
                                skip += 1
                        except Exception as exc:
                            status = "error"
                            detail = str(exc)[:100]
                            fail += 1

                        state.rows.append(
                            TaskRow(
                                label=f"{ticker} / {year}",
                                detail=detail,
                                status=status,
                            )
                        )
                        refresh()

                    mode_labels = {
                        "edc": "EDC (Strict)",
                        "edc_alt": "EDC (Alternative)",
                        "edc_alt_two": "EDC (Alternative 2)",
                    }
                    mode_label = mode_labels.get(edc_task_type, "EDC (Strict)")
                    state.summary = (
                        f"[{mode_label}] ✅ {ok} evaluated, {skip} skipped, {fail} failed"
                    )
                finally:
                    con.close()
                _refresh_status_panels()

            return run

        def _make_llm_infer_proper_task(
            state: TaskState, refresh: Callable
        ) -> Callable:
            def run():
                from llm_proper_vn import infer_proper_vn_report

                con = get_connection()
                try:
                    init_db(con)
                    ensure_vss_loaded(con)
                    reports = _target_pairs(con)
                    reports = [
                        (t, y)
                        for t, y in reports
                        if con.execute(
                            "SELECT 1 FROM document_embeddings "
                            "WHERE ticker = ? AND year = ? LIMIT 1",
                            [t, y],
                        ).fetchone()
                    ]
                    if not reports:
                        state.summary = "No embedded reports match the selected filter"
                        return

                    ok = 0
                    fail = 0
                    for ticker, year in reports:
                        try:
                            out = infer_proper_vn_report(
                                ticker,
                                year,
                                con=con,
                                replace=_selected_force(),
                                top_k=_selected_top_k(),
                                inference_model=_selected_infer_model(),
                                embedding_model=_selected_embed_model(),
                            )
                            state.rows.append(
                                TaskRow(
                                    label=f"{ticker} / {year}",
                                    detail=f"Color: {out.get('color', '?')}",
                                    status="done",
                                )
                            )
                            ok += 1
                        except Exception as exc:
                            state.rows.append(
                                TaskRow(
                                    label=f"{ticker} / {year}",
                                    detail=str(exc)[:100],
                                    status="error",
                                )
                            )
                            fail += 1
                        refresh()

                    state.summary = f"✅ {ok} evaluated, {fail} failed"
                finally:
                    con.close()
                _refresh_status_panels()

            return run

        def _make_llm_infer_governance_task(
            state: TaskState, refresh: Callable
        ) -> Callable:
            def run():
                from llm_governance import extract_governance

                con = get_connection()
                try:
                    init_db(con)
                    ensure_vss_loaded(con)
                    target_reports = _target_pairs(con)
                    if not target_reports:
                        state.summary = "No reports match the selected filter"
                        return

                    reports: list[tuple[str, int]] = []
                    missing_embeddings: list[tuple[str, int]] = []
                    for ticker, year in target_reports:
                        has_embeddings = con.execute(
                            "SELECT 1 FROM document_embeddings "
                            "WHERE ticker = ? AND year = ? LIMIT 1",
                            [ticker, year],
                        ).fetchone()
                        if has_embeddings:
                            reports.append((ticker, year))
                        else:
                            missing_embeddings.append((ticker, year))

                    for ticker, year in missing_embeddings:
                        state.rows.append(
                            TaskRow(
                                label=f"{ticker} / {year}",
                                detail="missing embeddings; run 'Embed Annual Reports' first",
                                status="skipped",
                            )
                        )
                        refresh()

                    if not reports:
                        state.summary = (
                            "No embedded reports to run governance extraction. "
                            f"{len(missing_embeddings)} report(s) skipped for missing embeddings."
                        )
                        return

                    ok = 0
                    fail = 0
                    skip = len(missing_embeddings)
                    for ticker, year in reports:
                        try:
                            n = extract_governance(
                                ticker,
                                year,
                                con=con,
                                replace=_selected_force(),
                                top_k=_selected_top_k(),
                                inference_model=_selected_infer_model(),
                                embedding_model=_selected_embed_model(),
                                item_codes=GOVERNANCE_ANNUAL_ITEM_CODES,
                            )
                            if n > 0:
                                status = "done"
                                detail = f"{n} items"
                                ok += 1
                            else:
                                status = "skipped"
                                detail = "no new items"
                                skip += 1
                        except Exception as exc:
                            status = "error"
                            detail = str(exc)[:100]
                            fail += 1

                        state.rows.append(
                            TaskRow(
                                label=f"{ticker} / {year}",
                                detail=detail,
                                status=status,
                            )
                        )
                        refresh()

                    state.summary = (
                        f"✅ {ok} evaluated, {skip} skipped, {fail} failed"
                    )
                finally:
                    con.close()
                _refresh_status_panels()

            return run

        def _make_llm_extract_financial_statement_audit_task(
            state: TaskState, refresh: Callable
        ) -> Callable:
            def run():
                from llm_financial_statement_audit import (
                    extract_financial_statement_audit_info,
                )

                con = get_connection()
                try:
                    init_db(con)
                    ensure_vss_loaded(con)

                    reports = _target_financial_statement_pairs(con)
                    reports = [
                        (t, y)
                        for t, y in reports
                        if con.execute(
                            "SELECT 1 FROM financial_statement_document_embeddings "
                            "WHERE ticker = ? AND year = ? LIMIT 1",
                            [t, y],
                        ).fetchone()
                    ]
                    if not reports:
                        state.summary = "No embedded financial statements match the selected filter"
                        return

                    ok = 0
                    fail = 0
                    skip = 0
                    for ticker, year in reports:
                        try:
                            out = extract_financial_statement_audit_info(
                                ticker,
                                year,
                                con=con,
                                replace=_selected_force(),
                                top_k=_selected_top_k(),
                                inference_model=_selected_infer_model(),
                                embedding_model=_selected_embed_model(),
                            )
                            if out.get("found"):
                                status = "done"
                                detail = (
                                    out.get("audit_firm")
                                    or out.get("audit_opinion")
                                    or "audit extracted"
                                )
                                ok += 1
                            else:
                                status = "skipped"
                                detail = str(out.get("reason") or "no audit evidence found")[:100]
                                skip += 1
                        except Exception as exc:
                            status = "error"
                            detail = str(exc)[:100]
                            fail += 1

                        state.rows.append(
                            TaskRow(
                                label=f"[FINANCIAL_STATEMENT AUDIT] {ticker} / {year}",
                                detail=detail,
                                status=status,
                            )
                        )
                        refresh()

                    state.summary = (
                        f"✅ {ok} extracted, {skip} skipped, {fail} failed"
                    )
                finally:
                    con.close()
                _refresh_status_panels()

            return run

        @ui.refreshable
        def embed_panel():
            task_card(
                "1. Embed Reports (Annual/Financial Statement)",
                embed_state,
                _make_llm_embed_task(embed_state, embed_panel.refresh),
                embed_panel.refresh,
            )

        @ui.refreshable
        def embed_financial_statement_panel():
            task_card(
                "1B. Embed Financial Statements Only",
                embed_financial_statement_state,
                _make_llm_embed_task(
                    embed_financial_statement_state,
                    embed_financial_statement_panel.refresh,
                    forced_target="financial_statement",
                ),
                embed_financial_statement_panel.refresh,
            )

        @ui.refreshable
        def infer_edc_panel():
            task_card(
                "2. Infer EDC",
                infer_edc_state,
                _make_llm_infer_edc_task(infer_edc_state, infer_edc_panel.refresh),
                infer_edc_panel.refresh,
            )

        @ui.refreshable
        def infer_proper_panel():
            task_card(
                "3. Infer PROPER-VN",
                infer_proper_state,
                _make_llm_infer_proper_task(
                    infer_proper_state, infer_proper_panel.refresh
                ),
                infer_proper_panel.refresh,
            )

        @ui.refreshable
        def infer_gov_panel():
            task_card(
                "4. Extract Governance",
                infer_gov_state,
                _make_llm_infer_governance_task(
                    infer_gov_state, infer_gov_panel.refresh
                ),
                infer_gov_panel.refresh,
            )

        @ui.refreshable
        def infer_financial_statement_audit_panel():
            task_card(
                "5. Extract Financial Statement Audit",
                infer_financial_statement_audit_state,
                _make_llm_extract_financial_statement_audit_task(
                    infer_financial_statement_audit_state,
                    infer_financial_statement_audit_panel.refresh,
                ),
                infer_financial_statement_audit_panel.refresh,
            )

        @ui.refreshable
        def fetch_batch_panel():
            def _refresh_fetch_batch_views():
                fetch_batch_panel.refresh()
                processing_status_panel.refresh()

            task_card(
                "0. Sync Running Batch Outputs",
                fetch_batch_state,
                _make_fetch_batch_outputs_task(
                    fetch_batch_state,
                    _refresh_fetch_batch_views,
                ),
                _refresh_fetch_batch_views,
            )

        with ui.tabs().classes("w-full") as task_tabs:
            ui.tab("embed_tasks", label="Embedding Tasks")
            ui.tab("infer_tasks", label="Inference Tasks")

        with ui.tab_panels(task_tabs, value="embed_tasks").classes("w-full"):
            with ui.tab_panel("embed_tasks"):
                embedding_status_panel()
                embed_panel()
                embed_financial_statement_panel()

            with ui.tab_panel("infer_tasks"):
                processing_status_panel()
                with ui.row().classes("gap-2"):
                    ui.button(
                        "Run Pending Governance",
                        on_click=lambda: _run_in_thread(
                            _make_llm_infer_governance_task(
                                infer_gov_state,
                                infer_gov_panel.refresh,
                            ),
                            infer_gov_state,
                            infer_gov_panel.refresh,
                        ),
                        color="primary",
                    )
                    ui.button(
                        "Sync Governance Batch Outputs",
                        on_click=lambda: _run_in_thread(
                            _make_fetch_batch_outputs_task(
                                fetch_batch_state,
                                fetch_batch_panel.refresh,
                            ),
                            fetch_batch_state,
                            fetch_batch_panel.refresh,
                        ),
                    ).props("outline")
                fetch_batch_panel()
                infer_edc_panel()
                infer_proper_panel()
                infer_gov_panel()
                infer_financial_statement_audit_panel()


@ui.page("/llm-query")
def page_llm_query():
    ui.dark_mode(False)
    _nav_header()
    init_db()

    with ui.column().classes("w-full max-w-6xl mx-auto gap-4 p-4"):
        ui.label("LLM Extracted Items Query").classes("text-2xl font-bold")
        ui.label(
            "Query extracted outputs from EDC, PROPER-VN, annual Governance, and financial-statement audit pipelines."
        ).classes("text-sm text-gray-600")

        with ui.card().classes("w-full"):
            ui.label("Query Filters").classes("text-lg font-bold")
            with ui.row().classes("items-end gap-3 flex-wrap w-full"):
                dataset_sel = (
                    ui.select(
                        label="Dataset",
                        options={
                            "edc": "EDC Results (HyDE2)",
                            "edc_alt": "EDC Alternative Results (HyDE2)",
                            "edc_alt_two": "EDC Alternative 2 Results (HyDE2)",
                            "proper": "PROPER-VN Results",
                            "governance": "Governance Results (Annual, no GOV_AUDIT)",
                            "financial_statement_audit": "Financial Statement Audit Results",
                        },
                        value="edc",
                    )
                    .props("dense outlined")
                    .classes("w-56")
                )

                ticker_input = (
                    ui.input("Ticker (optional)", placeholder="e.g. AAA")
                    .props("dense clearable outlined")
                    .classes("w-40")
                )

                year_input = (
                    ui.number(
                        "Year (optional)",
                        min=2000,
                        max=2100,
                        step=1,
                        format="%.0f",
                    )
                    .props("dense outlined")
                    .classes("w-32")
                )

                model_input = (
                    ui.input("Model (optional)")
                    .props("dense clearable outlined")
                    .classes("w-56")
                )

                limit_input = (
                    ui.number("Limit", value=200, min=1, max=1000, step=1)
                    .props("dense outlined")
                    .classes("w-28")
                )

            ui.button("Run Query", on_click=lambda: query_result_panel.refresh()).props(
                "dense color=primary"
            )

        @ui.refreshable
        def query_result_panel():
            dataset = str(dataset_sel.value or "edc")
            ticker = (ticker_input.value or "").strip().upper()
            year_val = year_input.value
            model = (model_input.value or "").strip()
            limit = int(limit_input.value or 200)

            conditions: list[str] = []
            params: list = []
            if ticker:
                conditions.append("ticker = ?")
                params.append(ticker)
            if year_val is not None and str(year_val).strip() != "":
                conditions.append("year = ?")
                params.append(int(year_val))
            if model:
                conditions.append("model = ?")
                params.append(model)

            where = ""
            if conditions:
                where = " WHERE " + " AND ".join(conditions)

            con = get_connection()
            try:
                def _has_rows(table_name: str) -> bool:
                    row = con.execute(
                        f"SELECT 1 FROM {table_name} LIMIT 1"
                    ).fetchone()
                    return row is not None

                edc_table = (
                    "inference_results_hyde2"
                    if _has_rows("inference_results_hyde2")
                    else "inference_results"
                )
                proper_table = (
                    "proper_vn_results_hyde2"
                    if _has_rows("proper_vn_results_hyde2")
                    else "proper_vn_results"
                )
                governance_table = (
                    "governance_results_hyde2"
                    if _has_rows("governance_results_hyde2")
                    else "governance_results"
                )

                if dataset in {"edc", "edc_alt", "edc_alt_two"}:
                    if dataset == "edc":
                        edc_suffix_filter = "category_code NOT LIKE ? AND category_code NOT LIKE ?"
                        edc_suffix_params = ["%_alt", "%_alt_two"]
                    elif dataset == "edc_alt":
                        edc_suffix_filter = "category_code LIKE ? AND category_code NOT LIKE ?"
                        edc_suffix_params = ["%_alt", "%_alt_two"]
                    else:
                        edc_suffix_filter = "category_code LIKE ?"
                        edc_suffix_params = ["%_alt_two"]

                    rows = con.execute(
                        f"""
                        SELECT ticker, year, category_code, is_valid, reason, model, created_at
                        FROM {edc_table}
                        {where}{' AND' if where else ' WHERE'} {edc_suffix_filter}
                        ORDER BY ticker, year DESC, category_code
                        LIMIT ?
                        """,
                        params + edc_suffix_params + [limit],
                    ).fetchall()

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
                            "name": "code",
                            "label": "Category",
                            "field": "code",
                            "align": "left",
                        },
                        {
                            "name": "valid",
                            "label": "Valid",
                            "field": "valid",
                            "align": "center",
                        },
                        {
                            "name": "reason",
                            "label": "Reason",
                            "field": "reason",
                            "align": "left",
                        },
                        {
                            "name": "model",
                            "label": "Model",
                            "field": "model",
                            "align": "left",
                        },
                    ]
                    data = [
                        {
                            "id": i,
                            "ticker": r[0],
                            "year": r[1],
                            "code": r[2],
                            "valid": "1" if r[3] else "0",
                            "reason": (r[4] or "")[:140],
                            "model": r[5],
                            "payload": {
                                "ticker": r[0],
                                "year": r[1],
                                "category_code": r[2],
                                "is_valid": bool(r[3]),
                                "reason": r[4],
                                "model": r[5],
                                "created_at": str(r[6]),
                            },
                        }
                        for i, r in enumerate(rows)
                    ]
                elif dataset == "proper":
                    rows = con.execute(
                        f"""
                        SELECT ticker, year, indicator_code, is_present,
                               evidence_level, reason, model, created_at
                        FROM {proper_table}
                        {where}
                        ORDER BY ticker, year DESC, indicator_code
                        LIMIT ?
                        """,
                        params + [limit],
                    ).fetchall()

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
                            "name": "code",
                            "label": "Indicator",
                            "field": "code",
                            "align": "left",
                        },
                        {
                            "name": "present",
                            "label": "Present",
                            "field": "present",
                            "align": "center",
                        },
                        {
                            "name": "level",
                            "label": "Evidence",
                            "field": "level",
                            "align": "left",
                        },
                        {
                            "name": "reason",
                            "label": "Reason",
                            "field": "reason",
                            "align": "left",
                        },
                        {
                            "name": "model",
                            "label": "Model",
                            "field": "model",
                            "align": "left",
                        },
                    ]
                    data = [
                        {
                            "id": i,
                            "ticker": r[0],
                            "year": r[1],
                            "code": r[2],
                            "present": "1" if r[3] else "0",
                            "level": r[4] or "",
                            "reason": (r[5] or "")[:140],
                            "model": r[6],
                            "payload": {
                                "ticker": r[0],
                                "year": r[1],
                                "indicator_code": r[2],
                                "is_present": bool(r[3]),
                                "evidence_level": r[4],
                                "reason": r[5],
                                "model": r[6],
                                "created_at": str(r[7]),
                            },
                        }
                        for i, r in enumerate(rows)
                    ]
                elif dataset == "governance":
                    rows = con.execute(
                        f"""
                        SELECT ticker, year, item_code, found,
                               value_json, details_json, reason, model, created_at
                        FROM {governance_table}
                        {where}{' AND' if where else ' WHERE'} item_code != 'GOV_AUDIT'
                        ORDER BY ticker, year DESC, item_code
                        LIMIT ?
                        """,
                        params + [limit],
                    ).fetchall()

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
                            "name": "code",
                            "label": "Item",
                            "field": "code",
                            "align": "left",
                        },
                        {
                            "name": "found",
                            "label": "Found",
                            "field": "found",
                            "align": "center",
                        },
                        {
                            "name": "reason",
                            "label": "Reason",
                            "field": "reason",
                            "align": "left",
                        },
                        {
                            "name": "model",
                            "label": "Model",
                            "field": "model",
                            "align": "left",
                        },
                    ]
                    data = []
                    for i, r in enumerate(rows):
                        try:
                            value_json = json.loads(r[4]) if r[4] else None
                        except Exception:
                            value_json = r[4]
                        try:
                            details_json = json.loads(r[5]) if r[5] else []
                        except Exception:
                            details_json = r[5]

                        data.append(
                            {
                                "id": i,
                                "ticker": r[0],
                                "year": r[1],
                                "code": r[2],
                                "found": "1" if r[3] else "0",
                                "reason": (r[6] or "")[:140],
                                "model": r[7],
                                "payload": {
                                    "ticker": r[0],
                                    "year": r[1],
                                    "item_code": r[2],
                                    "found": bool(r[3]),
                                    "value_json": value_json,
                                    "details_json": details_json,
                                    "reason": r[6],
                                    "model": r[7],
                                    "created_at": str(r[8]),
                                },
                            }
                        )
                else:
                    rows = con.execute(
                        f"""
                        SELECT ticker, year, found,
                               audit_firm, audit_opinion, signing_auditor_names,
                               reason, model, created_at
                        FROM financial_statement_audit_results
                        {where}
                        ORDER BY ticker, year DESC
                        LIMIT ?
                        """,
                        params + [limit],
                    ).fetchall()

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
                            "name": "found",
                            "label": "Found",
                            "field": "found",
                            "align": "center",
                        },
                        {
                            "name": "audit_firm",
                            "label": "Audit Firm",
                            "field": "audit_firm",
                            "align": "left",
                        },
                        {
                            "name": "audit_opinion",
                            "label": "Opinion",
                            "field": "audit_opinion",
                            "align": "left",
                        },
                        {
                            "name": "model",
                            "label": "Model",
                            "field": "model",
                            "align": "left",
                        },
                    ]
                    data = []
                    for i, r in enumerate(rows):
                        try:
                            signing_names = json.loads(r[5]) if r[5] else []
                        except Exception:
                            signing_names = r[5]
                        data.append(
                            {
                                "id": i,
                                "ticker": r[0],
                                "year": r[1],
                                "found": "1" if r[2] else "0",
                                "audit_firm": (r[3] or "")[:100],
                                "audit_opinion": r[4] or "",
                                "model": r[7],
                                "payload": {
                                    "ticker": r[0],
                                    "year": r[1],
                                    "found": bool(r[2]),
                                    "audit_firm": r[3],
                                    "audit_opinion": r[4],
                                    "signing_auditor_names": signing_names,
                                    "reason": r[6],
                                    "model": r[7],
                                    "created_at": str(r[8]),
                                },
                            }
                        )
            finally:
                con.close()

            if not data:
                if dataset in {"edc_alt", "edc_alt_two"}:
                    strict_where = (
                        f"{where}{' AND' if where else ' WHERE'} "
                        "category_code NOT LIKE ? AND category_code NOT LIKE ?"
                    )
                    con = get_connection()
                    try:
                        strict_row = con.execute(
                            f"SELECT COUNT(*) FROM {edc_table} {strict_where}",
                            params + ["%_alt", "%_alt_two"],
                        ).fetchone()
                    finally:
                        con.close()
                    strict_count = int(strict_row[0]) if strict_row else 0
                    if strict_count > 0:
                        alt_mode_label = (
                            "EDC (Alternative 2)"
                            if dataset == "edc_alt_two"
                            else "EDC (Alternative)"
                        )
                        ui.label(
                            "No rows found for these filters in "
                            f"{alt_mode_label}. Strict EDC rows exist; run Infer EDC "
                            f"with mode {alt_mode_label}, then Sync Running Batch Outputs."
                        ).classes("text-amber-700")
                        return
                ui.label("No extracted items found for current filters").classes(
                    "text-gray-500"
                )
                return

            ui.label(f"{len(data)} row(s)").classes("text-xs text-gray-500")
            table = (
                ui.table(
                    columns=columns,
                    rows=data,
                    row_key="id",
                    selection="single",
                )
                .classes("w-full")
                .props("dense flat")
            )

            payload_area = ui.code("Select one row, then click View Payload").classes(
                "w-full max-h-96 overflow-auto text-xs"
            )

            def _view_payload():
                selected = table.selected
                if not selected:
                    ui.notify("Select one row first", type="warning")
                    return
                payload = selected[0].get("payload", {})
                payload_area.set_content(
                    json.dumps(payload, indent=2, ensure_ascii=False)
                )

            ui.button("View Payload", on_click=_view_payload).props(
                "dense outline"
            )

        query_result_panel()


@ui.page("/llm-inputs")
def page_llm_inputs():
    ui.dark_mode(False)
    _nav_header()
    init_db()

    with ui.column().classes("w-full max-w-6xl mx-auto gap-4 p-4"):
        ui.label("LLM Request Inputs").classes("text-2xl font-bold")
        ui.label(
            "Inspect exact request payloads/prompts that were sent to OpenAI."
        ).classes("text-sm text-gray-600")

        with ui.card().classes("w-full"):
            ui.label("Filters").classes("text-lg font-bold")
            with ui.row().classes("items-end gap-3 flex-wrap w-full"):
                task_type_sel = (
                    ui.select(
                        label="Task Type",
                        options={
                            "all": "All",
                            "edc": "EDC",
                            "edc_alt": "EDC (Alternative)",
                            "edc_alt_two": "EDC (Alternative 2)",
                            "proper_vn": "PROPER-VN",
                            "governance": "Governance",
                            "financial_statement_audit": "Financial Statement Audit",
                        },
                        value="all",
                    )
                    .props("dense outlined")
                    .classes("w-44")
                )

                ticker_input = (
                    ui.input("Ticker")
                    .props("dense clearable outlined")
                    .classes("w-32")
                )
                year_input = (
                    ui.number("Year", min=2000, max=2100, step=1, format="%.0f")
                    .props("dense clearable outlined")
                    .classes("w-28")
                )
                model_input = (
                    ui.input("Model")
                    .props("dense clearable outlined")
                    .classes("w-48")
                )
                batch_input = (
                    ui.input("Batch ID")
                    .props("dense clearable outlined")
                    .classes("w-72")
                )
                item_code_input = (
                    ui.input("Item code")
                    .props("dense clearable outlined")
                    .classes("w-40")
                )
                limit_input = (
                    ui.number("Limit", value=200, min=1, max=5000, step=1)
                    .props("dense outlined")
                    .classes("w-28")
                )

            ui.button(
                "Run Query",
                on_click=lambda: llm_inputs_panel.refresh(),
                color="primary",
            ).props("dense")

        @ui.refreshable
        def llm_inputs_panel():
            conditions: list[str] = []
            params: list = []

            task_type = str(task_type_sel.value or "all")
            ticker = str(ticker_input.value or "").strip().upper()
            year_val = year_input.value
            model = str(model_input.value or "").strip()
            batch_id = str(batch_input.value or "").strip()
            item_code = str(item_code_input.value or "").strip()
            limit = int(limit_input.value or 200)

            if task_type != "all":
                conditions.append("task_type = ?")
                params.append(task_type)
            if ticker:
                conditions.append("ticker = ?")
                params.append(ticker)
            if year_val is not None and str(year_val).strip() != "":
                conditions.append("year = ?")
                params.append(int(year_val))
            if model:
                conditions.append("model = ?")
                params.append(model)
            if batch_id:
                conditions.append("batch_id = ?")
                params.append(batch_id)
            if item_code:
                conditions.append("item_code = ?")
                params.append(item_code)

            where = ""
            if conditions:
                where = " WHERE " + " AND ".join(conditions)

            con = get_connection()
            try:
                rows = con.execute(
                    f"""
                    SELECT id, task_type, ticker, year, item_code, model, batch_id,
                           request_body, prompt_text, created_at
                    FROM llm_request_inputs
                    {where}
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                    """,
                    params + [limit],
                ).fetchall()
            finally:
                con.close()

            if not rows:
                ui.label("No request inputs found for current filters").classes(
                    "text-gray-500"
                )
                return

            table_rows = [
                {
                    "id": r[0],
                    "task_type": r[1],
                    "ticker": r[2] or "",
                    "year": r[3] or "",
                    "item_code": r[4] or "",
                    "model": r[5],
                    "batch_id": r[6] or "",
                    "created_at": str(r[9]) if r[9] is not None else "",
                    "request_body": r[7] or "",
                    "prompt_text": r[8] or "",
                }
                for r in rows
            ]

            table = ui.table(
                columns=[
                    {"name": "id", "label": "ID", "field": "id", "align": "right"},
                    {
                        "name": "task_type",
                        "label": "Task",
                        "field": "task_type",
                        "align": "left",
                    },
                    {
                        "name": "ticker",
                        "label": "Ticker",
                        "field": "ticker",
                        "align": "left",
                    },
                    {"name": "year", "label": "Year", "field": "year", "align": "center"},
                    {
                        "name": "item_code",
                        "label": "Item",
                        "field": "item_code",
                        "align": "left",
                    },
                    {
                        "name": "model",
                        "label": "Model",
                        "field": "model",
                        "align": "left",
                    },
                    {
                        "name": "created_at",
                        "label": "Created",
                        "field": "created_at",
                        "align": "left",
                    },
                ],
                rows=table_rows,
                row_key="id",
                selection="single",
            ).classes("w-full").props("dense flat")

            preview_area = ui.code("Select one row then click a preview button").classes(
                "w-full max-h-96 overflow-auto text-xs"
            )

            def _selected_row() -> dict | None:
                selected = table.selected
                if not selected:
                    ui.notify("Select one row first", type="warning")
                    return None
                return selected[0]

            def _view_request_body() -> None:
                row = _selected_row()
                if row is None:
                    return
                preview_area.set_content(str(row.get("request_body") or ""))

            def _view_prompt_text() -> None:
                row = _selected_row()
                if row is None:
                    return
                preview_area.set_content(str(row.get("prompt_text") or ""))

            with ui.row().classes("gap-2"):
                ui.button("View Request Body", on_click=_view_request_body).props(
                    "dense outline"
                )
                ui.button("View Prompt Text", on_click=_view_prompt_text).props(
                    "dense outline"
                )

        llm_inputs_panel()



@ui.page("/data-studio")
def page_data_studio():
    ui.dark_mode(False)
    _nav_header()
    init_db()

    with ui.column().classes("w-full max-w-6xl mx-auto gap-4 p-4"):
        ui.label("Data Studio").classes("text-2xl font-bold")
        ui.label(
            "Run SQL analytics and monitor annual report gaps."
        ).classes("text-sm text-gray-600")

        with ui.tabs().classes("w-full") as ds_tabs:
            ui.tab("sql", label="DuckDB SQL")
            ui.tab("report_gap", label="Report Gap")

        with ui.tab_panels(ds_tabs, value="sql").classes("w-full"):
            with ui.tab_panel("sql"):
                with ui.card().classes("w-full"):
                    ui.label("DuckDB SQL").classes("text-lg font-bold")
                    ui.label(
                        "Run read-only DuckDB queries. Choose a pre-made SQL template from the dropdown and load it into the editor."
                    ).classes("text-sm text-gray-600")

                    sql_templates: dict[str, str] = DATA_STUDIO_SQL_TEMPLATES
                    default_template = DEFAULT_DATA_STUDIO_SQL_TEMPLATE

                    sql_state: SqlConsoleState = {
                        "rows": [],
                        "columns": [],
                        "error": "",
                        "row_count": 0,
                    }

                    with ui.row().classes("items-end gap-2 flex-wrap w-full"):
                        template_select = (
                            ui.select(
                                options=list(sql_templates.keys()),
                                value=default_template,
                                label="SQL Template",
                            )
                            .props("dense outlined")
                            .classes("w-72")
                        )

                    sql_input = (
                        ui.textarea(
                            "SQL",
                            value=sql_templates[default_template],
                        )
                        .props("outlined autogrow")
                        .classes("w-full")
                    )

                    def _load_selected_template() -> None:
                        template_name = str(template_select.value or default_template)
                        sql_input.value = sql_templates.get(
                            template_name,
                            sql_templates[default_template],
                        )
                        sql_input.update()

                    def _is_read_only_query(query: str) -> tuple[bool, str]:
                        q = str(query or "").strip()
                        if not q:
                            return False, "Query is empty"

                        # Keep to one statement only.
                        trimmed = q.rstrip()
                        if ";" in trimmed.rstrip(";"):
                            return False, "Only one SQL statement is allowed"

                        lowered = q.lstrip().lower()
                        allowed_prefixes = (
                            "select",
                            "with",
                            "show",
                            "describe",
                            "summarize",
                            "pragma",
                            "explain",
                        )
                        if not lowered.startswith(allowed_prefixes):
                            return False, (
                                "Only read-only SQL is allowed "
                                "(SELECT/WITH/SHOW/DESCRIBE/SUMMARIZE/PRAGMA/EXPLAIN)"
                            )
                        return True, ""

                    @ui.refreshable
                    def sql_result_panel() -> None:
                        error = sql_state["error"]
                        if error:
                            ui.label(error).classes("text-red-600 text-sm")
                            return

                        columns = sql_state["columns"]
                        rows = sql_state["rows"]
                        row_count = sql_state["row_count"]

                        if not columns:
                            ui.label("Run a query to see results.").classes(
                                "text-gray-500 text-sm"
                            )
                            return

                        ui.label(f"{row_count} row(s)").classes("text-xs text-gray-500")
                        ui.table(
                            columns=columns,
                            rows=rows,
                            row_key="_row_id",
                        ).classes("w-full").props("dense flat")

                    def _run_sql_query() -> None:
                        query = str(sql_input.value or "").strip()
                        ok, message = _is_read_only_query(query)
                        if not ok:
                            sql_state["error"] = message
                            sql_state["columns"] = []
                            sql_state["rows"] = []
                            sql_state["row_count"] = 0
                            sql_result_panel.refresh()
                            return

                        con = get_connection(read_only=True)
                        try:
                            cursor = con.execute(query)
                            description = cursor.description or []
                            col_names = [str(col[0]) for col in description]
                            data_rows = cursor.fetchall()
                        except Exception as exc:
                            sql_state["error"] = f"Query failed: {exc}"
                            sql_state["columns"] = []
                            sql_state["rows"] = []
                            sql_state["row_count"] = 0
                            sql_result_panel.refresh()
                            return
                        finally:
                            con.close()

                        table_columns = [
                            {
                                "name": col,
                                "label": col,
                                "field": col,
                                "align": "left",
                            }
                            for col in col_names
                        ]
                        table_rows = [
                            {
                                "_row_id": i,
                                **{
                                    col_names[j]: (
                                        "" if value is None else str(value)
                                    )
                                    for j, value in enumerate(row)
                                },
                            }
                            for i, row in enumerate(data_rows)
                        ]

                        sql_state["error"] = ""
                        sql_state["columns"] = table_columns
                        sql_state["rows"] = table_rows
                        sql_state["row_count"] = len(table_rows)
                        sql_result_panel.refresh()

                    def _export_sql_result_csv() -> None:
                        rows = sql_state["rows"]
                        columns = sql_state["columns"]
                        if not rows or not columns:
                            ui.notify("Run a query first to export results", type="warning")
                            return

                        export_dir = Path(OUTPUT_DIR) / "data_studio_exports"
                        export_dir.mkdir(parents=True, exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        export_path = export_dir / f"data_studio_query_{timestamp}.csv"

                        field_names = [str(col.get("name") or "") for col in columns]
                        field_names = [name for name in field_names if name and name != "_row_id"]

                        with export_path.open("w", newline="", encoding="utf-8") as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=field_names)
                            writer.writeheader()
                            for row in rows:
                                writer.writerow({key: str(row.get(key, "")) for key in field_names})

                        ui.notify(f"Exported CSV: {export_path}", type="positive")
                        ui.download(str(export_path))

                    with ui.row().classes("gap-2"):
                        ui.button(
                            "Load SQL Template",
                            on_click=_load_selected_template,
                        ).props("dense outline")
                        ui.button(
                            "Run DuckDB Query",
                            on_click=_run_sql_query,
                            color="primary",
                        ).props("dense")
                        ui.button(
                            "Export Results to CSV",
                            on_click=_export_sql_result_csv,
                            color="secondary",
                        ).props("dense outline")
                        ui.button(
                            "Clear Results",
                            on_click=lambda: (
                                sql_state.update(
                                    {
                                        "rows": [],
                                        "columns": [],
                                        "error": "",
                                        "row_count": 0,
                                    }
                                ),
                                sql_result_panel.refresh(),
                            ),
                        ).props("dense outline")

                    sql_result_panel()

            with ui.tab_panel("report_gap"):
                with ui.column().classes("w-full gap-4"):
                    ui.label("Annual Report Gap Dashboard").classes("text-lg font-bold")
                    ui.label(
                        "Find firms missing their latest expected annual report and queue a rerun for that ticker/year."
                    ).classes("text-sm text-gray-600")

                    with ui.card().classes("w-full"):
                        ui.label("Filters").classes("text-lg font-bold")
                        with ui.row().classes("items-end gap-3 flex-wrap w-full"):
                            ticker_input = (
                                ui.input("Ticker contains")
                                .props("dense clearable outlined")
                                .classes("w-48")
                            )
                            only_missing_toggle = ui.switch(
                                "Only missing", value=True
                            ).props("dense")
                        summary_label = ui.label("").classes("text-sm text-gray-600")

                    @ui.refreshable
                    def gap_table() -> None:
                        ticker_filter = (ticker_input.value or "").strip().upper()

                        con = get_connection()
                        try:
                            sql = """
                                WITH all_tickers AS (
                                    SELECT ticker FROM companies
                                    UNION
                                    SELECT DISTINCT ticker FROM annual_reports
                                    UNION
                                    SELECT DISTINCT ticker FROM conversion_jobs
                                    UNION
                                    SELECT DISTINCT ticker FROM vietstock_documents
                                ),
                                doc_years AS (
                                    SELECT
                                        ticker,
                                        MAX(CAST(regexp_extract(title, '(\\d{4})', 1) AS INTEGER)) AS doc_latest_year
                                    FROM vietstock_documents
                                    WHERE regexp_extract(title, '(\\d{4})', 1) != ''
                                    GROUP BY ticker
                                ),
                                job_years AS (
                                    SELECT ticker, MAX(year) AS job_latest_year
                                    FROM conversion_jobs
                                    GROUP BY ticker
                                ),
                                loaded_years AS (
                                    SELECT ticker, MAX(year) AS loaded_latest_year
                                    FROM annual_reports
                                    GROUP BY ticker
                                )
                                SELECT
                                    t.ticker,
                                    GREATEST(
                                        COALESCE(d.doc_latest_year, 0),
                                        COALESCE(j.job_latest_year, 0)
                                    ) AS expected_latest_year,
                                    COALESCE(l.loaded_latest_year, 0) AS loaded_latest_year,
                                    cj.status AS expected_job_status,
                                    cj.source_path AS expected_source_path
                                FROM all_tickers t
                                LEFT JOIN doc_years d ON d.ticker = t.ticker
                                LEFT JOIN job_years j ON j.ticker = t.ticker
                                LEFT JOIN loaded_years l ON l.ticker = t.ticker
                                LEFT JOIN conversion_jobs cj
                                  ON cj.ticker = t.ticker
                                 AND cj.year = GREATEST(
                                    COALESCE(d.doc_latest_year, 0),
                                    COALESCE(j.job_latest_year, 0)
                                 )
                                WHERE GREATEST(
                                        COALESCE(d.doc_latest_year, 0),
                                        COALESCE(j.job_latest_year, 0)
                                    ) > 0
                                ORDER BY t.ticker
                            """
                            rows = con.execute(sql).fetchall()
                        finally:
                            con.close()

                        prepared: list[dict[str, object]] = []
                        for i, row in enumerate(rows):
                            ticker = str(row[0])
                            expected_latest_year = int(row[1] or 0)
                            loaded_latest_year = int(row[2] or 0)
                            expected_job_status = str(row[3] or "")
                            expected_source_path = str(row[4] or "")
                            is_missing = loaded_latest_year < expected_latest_year

                            if ticker_filter and ticker_filter not in ticker:
                                continue
                            if bool(only_missing_toggle.value) and not is_missing:
                                continue

                            prepared.append(
                                {
                                    "id": i,
                                    "ticker": ticker,
                                    "expected_year": expected_latest_year,
                                    "loaded_year": loaded_latest_year if loaded_latest_year > 0 else "",
                                    "missing": "✅" if is_missing else "",
                                    "missing_year": expected_latest_year if is_missing else "",
                                    "job_status": expected_job_status,
                                    "has_source": "✅" if expected_source_path else "",
                                    "payload": {
                                        "ticker": ticker,
                                        "expected_latest_year": expected_latest_year,
                                        "loaded_latest_year": loaded_latest_year,
                                        "missing_year": expected_latest_year,
                                        "expected_job_status": expected_job_status,
                                        "expected_source_path": expected_source_path,
                                        "is_missing": is_missing,
                                    },
                                }
                            )

                        missing_count = sum(1 for row in prepared if row["missing"] == "✅")
                        summary_label.text = (
                            f"{len(prepared)} firm(s) shown | {missing_count} firm(s) missing latest expected annual report"
                        )

                        if not prepared:
                            ui.label("No firms match current filters").classes("text-gray-500")
                            return

                        columns = [
                            {"name": "ticker", "label": "Ticker", "field": "ticker", "align": "left"},
                            {"name": "expected_year", "label": "Expected Latest Year", "field": "expected_year", "align": "center"},
                            {"name": "loaded_year", "label": "Loaded Latest Year", "field": "loaded_year", "align": "center"},
                            {"name": "missing", "label": "Missing", "field": "missing", "align": "center"},
                            {"name": "missing_year", "label": "Rerun Year", "field": "missing_year", "align": "center"},
                            {"name": "job_status", "label": "Job Status", "field": "job_status", "align": "center"},
                            {"name": "has_source", "label": "Has Source", "field": "has_source", "align": "center"},
                            {"name": "action", "label": "", "field": "action", "align": "center"},
                        ]

                        table = (
                            ui.table(
                                columns=columns,
                                rows=prepared,
                                row_key="id",
                                selection="single",
                            )
                            .classes("w-full")
                            .props("dense flat")
                        )

                        table.add_slot(
                            "body-cell-action",
                            r"""
                            <q-td :props="props">
                                <q-btn flat dense round icon="refresh" color="primary" size="sm"
                                       :disable="!props.row.missing_year"
                                       @click="$parent.$emit('rerun', props.row)" />
                            </q-td>
                            """,
                        )

                        payload_area = ui.code("Select one row, then click View Row").classes(
                            "w-full max-h-72 overflow-auto text-xs"
                        )

                        def _view_row_payload() -> None:
                            selected = table.selected
                            if not selected:
                                ui.notify("Select one row first", type="warning")
                                return
                            payload = selected[0].get("payload", {})
                            payload_area.set_content(
                                json.dumps(payload, indent=2, ensure_ascii=False)
                            )

                        def _queue_rerun_for_row(row: dict) -> None:
                            from config_marker import MARKDOWN_DIR
                            from converter import create_jobs

                            ticker = str(row.get("ticker") or "").upper()
                            year_val = row.get("missing_year")
                            if not ticker or not year_val:
                                ui.notify("Row has no missing year to rerun", type="warning")
                                return

                            year = int(year_val)
                            con = get_connection()
                            try:
                                init_db(con)
                                ensure_company(con, ticker)

                                create_jobs(
                                    con,
                                    tickers=[ticker],
                                    years=[year],
                                    start_year=year,
                                    end_year=year,
                                )

                                existing_job = con.execute(
                                    "SELECT source_path FROM conversion_jobs "
                                    "WHERE ticker = ? AND year = ?",
                                    [ticker, year],
                                ).fetchone()

                                source_path = (
                                    str(existing_job[0])
                                    if existing_job and existing_job[0]
                                    else ""
                                )
                                if not source_path:
                                    doc_row = con.execute(
                                        """
                                        SELECT raw_path
                                        FROM vietstock_documents
                                        WHERE ticker = ?
                                          AND synced_to_raw = TRUE
                                          AND raw_path IS NOT NULL
                                          AND raw_path != ''
                                          AND regexp_extract(title, '(\\d{4})', 1) = ?
                                        ORDER BY published_date DESC
                                        LIMIT 1
                                        """,
                                        [ticker, str(year)],
                                    ).fetchone()
                                    source_path = str(doc_row[0]) if doc_row and doc_row[0] else ""

                                if not source_path:
                                    raise ValueError(
                                        f"No raw source file found for {ticker}/{year}. Sync or download source PDF first."
                                    )

                                con.execute(
                                    """
                                    INSERT INTO conversion_jobs (
                                        id, ticker, year, start_year, end_year, source_path, output_dir, status
                                    ) VALUES (
                                        nextval('conversion_jobs_id_seq'), ?, ?, ?, ?, ?, ?, 'pending'
                                    )
                                    ON CONFLICT (ticker, year) DO UPDATE SET
                                        source_path = EXCLUDED.source_path,
                                        output_dir = EXCLUDED.output_dir,
                                        status = 'pending',
                                        rerun_reason = ?,
                                        command = NULL,
                                        log_path = NULL,
                                        pid = NULL,
                                        error_message = NULL,
                                        failed_step = NULL,
                                        started_at = NULL,
                                        completed_at = NULL
                                    """,
                                    [
                                        ticker,
                                        year,
                                        year,
                                        year,
                                        source_path,
                                        str(MARKDOWN_DIR),
                                        f"Manual rerun from report-gap dashboard ({datetime.now().isoformat(timespec='seconds')})",
                                    ],
                                )

                                con.execute(
                                    "DELETE FROM annual_reports WHERE ticker = ? AND year = ?",
                                    [ticker, year],
                                )
                                con.execute(
                                    "DELETE FROM document_embeddings WHERE ticker = ? AND year = ?",
                                    [ticker, year],
                                )
                                con.execute(
                                    "DELETE FROM inference_results WHERE ticker = ? AND year = ?",
                                    [ticker, year],
                                )
                                con.execute(
                                    "DELETE FROM proper_vn_results WHERE ticker = ? AND year = ?",
                                    [ticker, year],
                                )
                                con.execute(
                                    "DELETE FROM governance_results WHERE ticker = ? AND year = ?",
                                    [ticker, year],
                                )

                                con.execute(
                                    """
                                    UPDATE inference_jobs
                                    SET status = 'pending',
                                        categories_done = 0,
                                        batch_id = NULL,
                                        batch_submitted_at = NULL,
                                        batch_checked_at = NULL,
                                        started_at = NULL,
                                        completed_at = NULL,
                                        error_message = NULL
                                    WHERE ticker = ? AND year = ?
                                    """,
                                    [ticker, year],
                                )
                                con.execute(
                                    """
                                    UPDATE proper_vn_jobs
                                    SET status = 'pending',
                                        indicators_done = 0,
                                        batch_id = NULL,
                                        batch_submitted_at = NULL,
                                        batch_checked_at = NULL,
                                        color = NULL,
                                        s2_score = NULL,
                                        s2_max_score = NULL,
                                        started_at = NULL,
                                        completed_at = NULL,
                                        error_message = NULL
                                    WHERE ticker = ? AND year = ?
                                    """,
                                    [ticker, year],
                                )
                                con.execute(
                                    """
                                    UPDATE governance_jobs
                                    SET status = 'pending',
                                        items_done = 0,
                                        batch_id = NULL,
                                        batch_submitted_at = NULL,
                                        batch_checked_at = NULL,
                                        started_at = NULL,
                                        completed_at = NULL,
                                        error_message = NULL
                                    WHERE ticker = ? AND year = ?
                                    """,
                                    [ticker, year],
                                )
                            finally:
                                con.close()

                            ui.notify(
                                f"Queued rerun for {ticker}/{year} (convert -> load -> embed -> infer)",
                                type="positive",
                            )
                            gap_table.refresh()

                        table.on("rerun", lambda e: _queue_rerun_for_row(e.args))

                        with ui.row().classes("gap-2"):
                            ui.button("Refresh", on_click=gap_table.refresh).props("dense outline")
                            ui.button("View Row", on_click=_view_row_payload).props("dense outline")

                    gap_table()


@ui.page("/company-history")
def page_company_history():
    ui.dark_mode(False)
    _nav_header()
    init_db()

    with ui.column().classes("w-full max-w-6xl mx-auto gap-4 p-4"):
        ui.label("Company History").classes("text-2xl font-bold")
        ui.label(
            "Fetch and store Moc lich su events from Vietstock profile pages; compute firm age from first event."
        ).classes("text-sm text-gray-600")

        with ui.card().classes("w-full"):
            ui.label("Sync Company History").classes("text-lg font-bold")

            with ui.row().classes("items-end gap-3 flex-wrap w-full"):
                ticker_input = (
                    ui.input(
                        "Tickers",
                        placeholder="SRF,VNM or newline-separated",
                        value="",
                    )
                    .props("dense outlined clearable")
                    .classes("w-80")
                )

            sync_summary = ui.label("").classes("text-sm text-gray-600")

            @ui.refreshable
            def history_summary_table() -> None:
                ticker_filter = _normalize_ticker_list(str(ticker_input.value or ""))

                con = get_connection()
                try:
                    sql = """
                        SELECT ticker, first_event_year, first_event_date, first_event_text, firm_age, event_count, fetched_at
                        FROM company_history_summary
                    """
                    params: list = []
                    if ticker_filter:
                        placeholders = ", ".join(["?"] * len(ticker_filter))
                        sql += f" WHERE ticker IN ({placeholders})"
                        params.extend([t.upper() for t in ticker_filter])
                    sql += " ORDER BY ticker"
                    rows = con.execute(sql, params).fetchall()
                finally:
                    con.close()

                if not rows:
                    ui.label("No company history summary rows found.").classes("text-gray-500")
                    return

                ui.table(
                    columns=[
                        {"name": "ticker", "label": "Ticker", "field": "ticker", "align": "left"},
                        {"name": "first_year", "label": "First Year", "field": "first_year", "align": "center"},
                        {"name": "firm_age", "label": "Firm Age", "field": "firm_age", "align": "center"},
                        {"name": "event_count", "label": "Events", "field": "event_count", "align": "center"},
                        {"name": "first_event", "label": "First Event", "field": "first_event", "align": "left"},
                        {"name": "fetched_at", "label": "Fetched At", "field": "fetched_at", "align": "left"},
                    ],
                    rows=[
                        {
                            "id": i,
                            "ticker": r[0],
                            "first_year": r[1] if r[1] is not None else "",
                            "firm_age": r[4] if r[4] is not None else "",
                            "event_count": r[5] if r[5] is not None else 0,
                            "first_event": (r[3] or "")[:120],
                            "fetched_at": str(r[6]) if r[6] is not None else "",
                        }
                        for i, r in enumerate(rows)
                    ],
                    row_key="id",
                ).classes("w-full").props("dense flat")

            def _run_history_sync() -> None:
                from company_history import sync_company_history, sync_company_history_many

                tickers = _normalize_ticker_list(str(ticker_input.value or ""))

                con = get_connection()
                try:
                    init_db(con)
                    if tickers:
                        results = sync_company_history_many(con, tickers=tickers)
                    else:
                        company_rows = con.execute(
                            "SELECT ticker FROM companies ORDER BY ticker"
                        ).fetchall()
                        all_tickers = [str(row[0]).upper() for row in company_rows]
                        if not all_tickers:
                            ui.notify("No tickers in companies table", type="warning")
                            return
                        results = [
                            sync_company_history(con, ticker=ticker)
                            for ticker in all_tickers
                        ]
                except Exception as exc:
                    ui.notify(f"Company history sync failed: {exc}", type="negative")
                    return
                finally:
                    con.close()

                synced = len(results)
                with_events = sum(1 for r in results if int(r.get("event_count", 0)) > 0)
                sync_summary.text = (
                    f"Synced {synced} ticker(s) | {with_events} ticker(s) with events"
                )
                ui.notify(sync_summary.text, type="positive")
                history_summary_table.refresh()

            with ui.row().classes("gap-2"):
                ui.button(
                    "Run Company History Sync",
                    on_click=_run_history_sync,
                    color="primary",
                ).props("dense")
                ui.button(
                    "Refresh Summary",
                    on_click=history_summary_table.refresh,
                ).props("dense outline")

            history_summary_table()


@ui.page("/extract-items")
def page_extract_items():
    ui.dark_mode(False)
    _nav_header()
    init_db()

    with ui.column().classes("w-full max-w-6xl mx-auto gap-4 p-4"):
        ui.label("Extracted Items Browser").classes("text-2xl font-bold")
        ui.label(
            "Browse extracted EDC, PROPER-VN, annual Governance items, and financial-statement audit items across ticker/year with filters."
        ).classes("text-sm text-gray-600")

        with ui.card().classes("w-full"):
            ui.label("Filters").classes("text-lg font-bold")
            with ui.row().classes("items-end gap-3 flex-wrap w-full"):
                dataset_sel = (
                    ui.select(
                        label="Dataset",
                        options={
                            "all": "All",
                            "edc": "EDC",
                            "edc_alt": "EDC (Alternative)",
                            "edc_alt_two": "EDC (Alternative 2)",
                            "proper": "PROPER-VN",
                            "governance": "Governance (Annual, no GOV_AUDIT)",
                            "financial_statement_audit": "Financial Statement Audit",
                        },
                        value="all",
                    )
                    .props("dense outlined")
                    .classes("w-40")
                )

                ticker_input = (
                    ui.input("Ticker", placeholder="e.g. VNM")
                    .props("dense clearable outlined")
                    .classes("w-32")
                )

                year_input = (
                    ui.number(
                        "Year",
                        min=2000,
                        max=2100,
                        step=1,
                        format="%.0f",
                    )
                    .props("dense clearable outlined")
                    .classes("w-28")
                )

                code_input = (
                    ui.input("Item code contains")
                    .props("dense clearable outlined")
                    .classes("w-48")
                )

                status_sel = (
                    ui.select(
                        label="Status",
                        options={
                            "all": "All",
                            "positive": "Positive (1)",
                            "negative": "Negative (0)",
                        },
                        value="all",
                    )
                    .props("dense outlined")
                    .classes("w-44")
                )

                model_input = (
                    ui.input("Model")
                    .props("dense clearable outlined")
                    .classes("w-48")
                )

                text_input = (
                    ui.input("Reason/text contains")
                    .props("dense clearable outlined")
                    .classes("w-64")
                )

                limit_input = (
                    ui.number("Limit", value=500, min=1, max=5000, step=1)
                    .props("dense outlined")
                    .classes("w-28")
                )

            ui.button(
                "Run Query",
                on_click=lambda: extract_items_panel.refresh(),
                color="primary",
            ).props("dense")

        @ui.refreshable
        def extract_items_panel():
            dataset = str(dataset_sel.value or "all")
            ticker = (ticker_input.value or "").strip().upper()
            year_val = year_input.value
            item_code = (code_input.value or "").strip().lower()
            status = str(status_sel.value or "all")
            model = (model_input.value or "").strip()
            text = (text_input.value or "").strip().lower()
            limit = int(limit_input.value or 500)

            def _build_where(
                code_col: str,
                status_col: str,
                text_expr: str,
            ) -> tuple[str, list]:
                conditions: list[str] = []
                params: list = []

                if ticker:
                    conditions.append("ticker = ?")
                    params.append(ticker)
                if year_val is not None and str(year_val).strip() != "":
                    conditions.append("year = ?")
                    params.append(int(year_val))
                if model:
                    conditions.append("model = ?")
                    params.append(model)
                if item_code:
                    conditions.append(f"LOWER({code_col}) LIKE ?")
                    params.append(f"%{item_code}%")
                if status == "positive":
                    conditions.append(f"{status_col} = TRUE")
                elif status == "negative":
                    conditions.append(f"{status_col} = FALSE")
                if text:
                    conditions.append(f"LOWER({text_expr}) LIKE ?")
                    params.append(f"%{text}%")

                where = ""
                if conditions:
                    where = " WHERE " + " AND ".join(conditions)
                return where, params

            edc_where, edc_params = _build_where(
                code_col="category_code",
                status_col="is_valid",
                text_expr="COALESCE(reason, '')",
            )
            proper_where, proper_params = _build_where(
                code_col="indicator_code",
                status_col="is_present",
                text_expr="COALESCE(reason, '') || ' ' || COALESCE(evidence_level, '')",
            )
            gov_where, gov_params = _build_where(
                code_col="item_code",
                status_col="found",
                text_expr="COALESCE(reason, '') || ' ' || COALESCE(value_json, '') || ' ' || COALESCE(details_json, '')",
            )

            con = get_connection()
            try:
                def _has_rows(table_name: str) -> bool:
                    row = con.execute(
                        f"SELECT 1 FROM {table_name} LIMIT 1"
                    ).fetchone()
                    return row is not None

                edc_table = (
                    "inference_results_hyde2"
                    if _has_rows("inference_results_hyde2")
                    else "inference_results"
                )
                proper_table = (
                    "proper_vn_results_hyde2"
                    if _has_rows("proper_vn_results_hyde2")
                    else "proper_vn_results"
                )
                governance_table = (
                    "governance_results_hyde2"
                    if _has_rows("governance_results_hyde2")
                    else "governance_results"
                )
            finally:
                con.close()

            edc_sql = f"""
                SELECT
                    'EDC' AS dataset,
                    ticker,
                    year,
                    category_code AS item_code,
                    CASE WHEN is_valid THEN '1' ELSE '0' END AS status,
                    '' AS extra,
                    reason,
                    model,
                    created_at
                FROM {edc_table}
                {edc_where}{' AND' if edc_where else ' WHERE'} category_code NOT LIKE '%_alt' AND category_code NOT LIKE '%_alt_two'
            """

            edc_alt_sql = f"""
                SELECT
                    'EDC_ALT' AS dataset,
                    ticker,
                    year,
                    category_code AS item_code,
                    CASE WHEN is_valid THEN '1' ELSE '0' END AS status,
                    '' AS extra,
                    reason,
                    model,
                    created_at
                FROM {edc_table}
                {edc_where}{' AND' if edc_where else ' WHERE'} category_code LIKE '%_alt' AND category_code NOT LIKE '%_alt_two'
            """

            edc_alt_two_sql = f"""
                SELECT
                    'EDC_ALT_TWO' AS dataset,
                    ticker,
                    year,
                    category_code AS item_code,
                    CASE WHEN is_valid THEN '1' ELSE '0' END AS status,
                    '' AS extra,
                    reason,
                    model,
                    created_at
                FROM {edc_table}
                {edc_where}{' AND' if edc_where else ' WHERE'} category_code LIKE '%_alt_two'
            """

            proper_sql = f"""
                SELECT
                    'PROPER_VN' AS dataset,
                    ticker,
                    year,
                    indicator_code AS item_code,
                    CASE WHEN is_present THEN '1' ELSE '0' END AS status,
                    COALESCE(evidence_level, '') AS extra,
                    reason,
                    model,
                    created_at
                FROM {proper_table}
                {proper_where}
            """

            gov_sql = f"""
                SELECT
                    'GOVERNANCE' AS dataset,
                    ticker,
                    year,
                    item_code,
                    CASE WHEN found THEN '1' ELSE '0' END AS status,
                    COALESCE(SUBSTR(value_json, 1, 160), '') AS extra,
                    reason,
                    model,
                    created_at
                FROM {governance_table}
                {gov_where}{' AND' if gov_where else ' WHERE'} item_code != 'GOV_AUDIT'
            """

            financial_statement_where, financial_statement_params = _build_where(
                code_col="COALESCE(audit_firm, '')",
                status_col="found",
                text_expr="COALESCE(reason, '') || ' ' || COALESCE(audit_firm, '') || ' ' || COALESCE(audit_opinion, '')",
            )

            financial_statement_sql = f"""
                SELECT
                    'FINANCIAL_STATEMENT_AUDIT' AS dataset,
                    ticker,
                    year,
                    COALESCE(audit_firm, '') AS item_code,
                    CASE WHEN found THEN '1' ELSE '0' END AS status,
                    COALESCE(audit_opinion, '') AS extra,
                    reason,
                    model,
                    created_at
                FROM financial_statement_audit_results
                {financial_statement_where}
            """

            if dataset == "edc":
                sql = f"""
                    {edc_sql}
                    ORDER BY ticker, year DESC, item_code
                    LIMIT ?
                """
                params = edc_params + [limit]
            elif dataset == "edc_alt":
                sql = f"""
                    {edc_alt_sql}
                    ORDER BY ticker, year DESC, item_code
                    LIMIT ?
                """
                params = edc_params + [limit]
            elif dataset == "edc_alt_two":
                sql = f"""
                    {edc_alt_two_sql}
                    ORDER BY ticker, year DESC, item_code
                    LIMIT ?
                """
                params = edc_params + [limit]
            elif dataset == "proper":
                sql = f"""
                    {proper_sql}
                    ORDER BY ticker, year DESC, item_code
                    LIMIT ?
                """
                params = proper_params + [limit]
            elif dataset == "governance":
                sql = f"""
                    {gov_sql}
                    ORDER BY ticker, year DESC, item_code
                    LIMIT ?
                """
                params = gov_params + [limit]
            elif dataset == "financial_statement_audit":
                sql = f"""
                    {financial_statement_sql}
                    ORDER BY ticker, year DESC, item_code
                    LIMIT ?
                """
                params = financial_statement_params + [limit]
            else:
                sql = f"""
                    {edc_sql}
                    UNION ALL
                    {edc_alt_sql}
                    UNION ALL
                    {edc_alt_two_sql}
                    UNION ALL
                    {proper_sql}
                    UNION ALL
                    {gov_sql}
                    UNION ALL
                    {financial_statement_sql}
                    ORDER BY ticker, year DESC, dataset, item_code
                    LIMIT ?
                """
                params = (
                    edc_params
                    + edc_params
                    + edc_params
                    + proper_params
                    + gov_params
                    + financial_statement_params
                    + [limit]
                )

            con = get_connection()
            try:
                rows = con.execute(sql, params).fetchall()
            finally:
                con.close()

            if not rows:
                ui.label("No extracted items found for current filters").classes(
                    "text-gray-500"
                )
                return

            columns = [
                {
                    "name": "dataset",
                    "label": "Dataset",
                    "field": "dataset",
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
                    "name": "item_code",
                    "label": "Item Code",
                    "field": "item_code",
                    "align": "left",
                },
                {
                    "name": "status",
                    "label": "Status",
                    "field": "status",
                    "align": "center",
                },
                {
                    "name": "extra",
                    "label": "Extra",
                    "field": "extra",
                    "align": "left",
                },
                {
                    "name": "reason",
                    "label": "Reason",
                    "field": "reason",
                    "align": "left",
                },
                {
                    "name": "model",
                    "label": "Model",
                    "field": "model",
                    "align": "left",
                },
            ]

            data = [
                {
                    "id": i,
                    "dataset": r[0],
                    "ticker": r[1],
                    "year": r[2],
                    "item_code": r[3],
                    "status": r[4],
                    "extra": (r[5] or "")[:160],
                    "reason": (r[6] or "")[:180],
                    "model": r[7] or "",
                    "payload": {
                        "dataset": r[0],
                        "ticker": r[1],
                        "year": r[2],
                        "item_code": r[3],
                        "status": r[4],
                        "extra": r[5],
                        "reason": r[6],
                        "model": r[7],
                        "created_at": str(r[8]),
                    },
                }
                for i, r in enumerate(rows)
            ]

            ui.label(f"{len(data)} row(s)").classes("text-xs text-gray-500")
            table = (
                ui.table(
                    columns=columns,
                    rows=data,
                    row_key="id",
                    selection="single",
                )
                .classes("w-full")
                .props("dense flat")
            )

            payload_area = ui.code("Select one row, then click View Payload").classes(
                "w-full max-h-96 overflow-auto text-xs"
            )

            def _view_payload():
                selected = table.selected
                if not selected:
                    ui.notify("Select one row first", type="warning")
                    return
                payload = selected[0].get("payload", {})
                payload_area.set_content(
                    json.dumps(payload, indent=2, ensure_ascii=False)
                )

            ui.button("View Payload", on_click=_view_payload).props(
                "dense outline"
            )

        extract_items_panel()


@ui.page("/report-gap")
def page_report_gap():
    ui.navigate.to("/data-studio")
    return

    ui.dark_mode(False)
    _nav_header()
    init_db()

    with ui.column().classes("w-full max-w-6xl mx-auto gap-4 p-4"):
        ui.label("Annual Report Gap Dashboard").classes("text-2xl font-bold")
        ui.label(
            "Find firms missing their latest expected annual report and queue a rerun for that ticker/year."
        ).classes("text-sm text-gray-600")

        with ui.card().classes("w-full"):
            ui.label("Filters").classes("text-lg font-bold")
            with ui.row().classes("items-end gap-3 flex-wrap w-full"):
                ticker_input = (
                    ui.input("Ticker contains")
                    .props("dense clearable outlined")
                    .classes("w-48")
                )
                only_missing_toggle = ui.switch("Only missing", value=True).props(
                    "dense"
                )
            summary_label = ui.label("").classes("text-sm text-gray-600")

        @ui.refreshable
        def gap_table():
            ticker_filter = (ticker_input.value or "").strip().upper()

            con = get_connection()
            try:
                sql = """
                    WITH all_tickers AS (
                        SELECT ticker FROM companies
                        UNION
                        SELECT DISTINCT ticker FROM annual_reports
                        UNION
                        SELECT DISTINCT ticker FROM conversion_jobs
                        UNION
                        SELECT DISTINCT ticker FROM vietstock_documents
                    ),
                    doc_years AS (
                        SELECT
                            ticker,
                            MAX(CAST(regexp_extract(title, '(\\d{4})', 1) AS INTEGER)) AS doc_latest_year
                        FROM vietstock_documents
                        WHERE regexp_extract(title, '(\\d{4})', 1) != ''
                        GROUP BY ticker
                    ),
                    job_years AS (
                        SELECT ticker, MAX(year) AS job_latest_year
                        FROM conversion_jobs
                        GROUP BY ticker
                    ),
                    loaded_years AS (
                        SELECT ticker, MAX(year) AS loaded_latest_year
                        FROM annual_reports
                        GROUP BY ticker
                    )
                    SELECT
                        t.ticker,
                        GREATEST(
                            COALESCE(d.doc_latest_year, 0),
                            COALESCE(j.job_latest_year, 0)
                        ) AS expected_latest_year,
                        COALESCE(l.loaded_latest_year, 0) AS loaded_latest_year,
                        cj.status AS expected_job_status,
                        cj.source_path AS expected_source_path
                    FROM all_tickers t
                    LEFT JOIN doc_years d ON d.ticker = t.ticker
                    LEFT JOIN job_years j ON j.ticker = t.ticker
                    LEFT JOIN loaded_years l ON l.ticker = t.ticker
                    LEFT JOIN conversion_jobs cj
                      ON cj.ticker = t.ticker
                     AND cj.year = GREATEST(
                        COALESCE(d.doc_latest_year, 0),
                        COALESCE(j.job_latest_year, 0)
                     )
                    WHERE GREATEST(
                            COALESCE(d.doc_latest_year, 0),
                            COALESCE(j.job_latest_year, 0)
                        ) > 0
                    ORDER BY t.ticker
                """
                rows = con.execute(sql).fetchall()
            finally:
                con.close()

            prepared: list[dict[str, object]] = []
            for i, row in enumerate(rows):
                ticker = str(row[0])
                expected_latest_year = int(row[1] or 0)
                loaded_latest_year = int(row[2] or 0)
                expected_job_status = str(row[3] or "")
                expected_source_path = str(row[4] or "")
                is_missing = loaded_latest_year < expected_latest_year

                if ticker_filter and ticker_filter not in ticker:
                    continue
                if bool(only_missing_toggle.value) and not is_missing:
                    continue

                prepared.append(
                    {
                        "id": i,
                        "ticker": ticker,
                        "expected_year": expected_latest_year,
                        "loaded_year": loaded_latest_year if loaded_latest_year > 0 else "",
                        "missing": "✅" if is_missing else "",
                        "missing_year": expected_latest_year if is_missing else "",
                        "job_status": expected_job_status,
                        "has_source": "✅" if expected_source_path else "",
                        "payload": {
                            "ticker": ticker,
                            "expected_latest_year": expected_latest_year,
                            "loaded_latest_year": loaded_latest_year,
                            "missing_year": expected_latest_year,
                            "expected_job_status": expected_job_status,
                            "expected_source_path": expected_source_path,
                            "is_missing": is_missing,
                        },
                    }
                )

            missing_count = sum(1 for row in prepared if row["missing"] == "✅")
            summary_label.text = (
                f"{len(prepared)} firm(s) shown | {missing_count} firm(s) missing latest expected annual report"
            )

            if not prepared:
                ui.label("No firms match current filters").classes("text-gray-500")
                return

            columns = [
                {
                    "name": "ticker",
                    "label": "Ticker",
                    "field": "ticker",
                    "align": "left",
                },
                {
                    "name": "expected_year",
                    "label": "Expected Latest Year",
                    "field": "expected_year",
                    "align": "center",
                },
                {
                    "name": "loaded_year",
                    "label": "Loaded Latest Year",
                    "field": "loaded_year",
                    "align": "center",
                },
                {
                    "name": "missing",
                    "label": "Missing",
                    "field": "missing",
                    "align": "center",
                },
                {
                    "name": "missing_year",
                    "label": "Rerun Year",
                    "field": "missing_year",
                    "align": "center",
                },
                {
                    "name": "job_status",
                    "label": "Job Status",
                    "field": "job_status",
                    "align": "center",
                },
                {
                    "name": "has_source",
                    "label": "Has Source",
                    "field": "has_source",
                    "align": "center",
                },
                {
                    "name": "action",
                    "label": "",
                    "field": "action",
                    "align": "center",
                },
            ]

            table = (
                ui.table(
                    columns=columns,
                    rows=prepared,
                    row_key="id",
                    selection="single",
                )
                .classes("w-full")
                .props("dense flat")
            )

            table.add_slot(
                "body-cell-action",
                r"""
                <q-td :props="props">
                    <q-btn flat dense round icon="refresh" color="primary" size="sm"
                           :disable="!props.row.missing_year"
                           @click="$parent.$emit('rerun', props.row)" />
                </q-td>
                """,
            )

            payload_area = ui.code("Select one row, then click View Row").classes(
                "w-full max-h-72 overflow-auto text-xs"
            )

            def _view_row_payload() -> None:
                selected = table.selected
                if not selected:
                    ui.notify("Select one row first", type="warning")
                    return
                payload = selected[0].get("payload", {})
                payload_area.set_content(
                    json.dumps(payload, indent=2, ensure_ascii=False)
                )

            def _queue_rerun_for_row(row: dict) -> None:
                from config_marker import MARKDOWN_DIR
                from converter import create_jobs

                ticker = str(row.get("ticker") or "").upper()
                year_val = row.get("missing_year")
                if not ticker or not year_val:
                    ui.notify("Row has no missing year to rerun", type="warning")
                    return

                year = int(year_val)
                con = get_connection()
                try:
                    init_db(con)
                    ensure_company(con, ticker)

                    create_jobs(
                        con,
                        tickers=[ticker],
                        years=[year],
                        start_year=year,
                        end_year=year,
                    )

                    existing_job = con.execute(
                        "SELECT source_path FROM conversion_jobs "
                        "WHERE ticker = ? AND year = ?",
                        [ticker, year],
                    ).fetchone()

                    source_path = str(existing_job[0]) if existing_job and existing_job[0] else ""
                    if not source_path:
                        doc_row = con.execute(
                            """
                            SELECT raw_path
                            FROM vietstock_documents
                            WHERE ticker = ?
                              AND synced_to_raw = TRUE
                              AND raw_path IS NOT NULL
                              AND raw_path != ''
                              AND regexp_extract(title, '(\\d{4})', 1) = ?
                            ORDER BY published_date DESC
                            LIMIT 1
                            """,
                            [ticker, str(year)],
                        ).fetchone()
                        source_path = str(doc_row[0]) if doc_row and doc_row[0] else ""

                    if not source_path:
                        raise ValueError(
                            f"No raw source file found for {ticker}/{year}. Sync or download source PDF first."
                        )

                    con.execute(
                        """
                        INSERT INTO conversion_jobs (
                            id, ticker, year, start_year, end_year, source_path, output_dir, status
                        ) VALUES (
                            nextval('conversion_jobs_id_seq'), ?, ?, ?, ?, ?, ?, 'pending'
                        )
                        ON CONFLICT (ticker, year) DO UPDATE SET
                            source_path = EXCLUDED.source_path,
                            output_dir = EXCLUDED.output_dir,
                            status = 'pending',
                            rerun_reason = ?,
                            command = NULL,
                            log_path = NULL,
                            pid = NULL,
                            error_message = NULL,
                            failed_step = NULL,
                            started_at = NULL,
                            completed_at = NULL
                        """,
                        [
                            ticker,
                            year,
                            year,
                            year,
                            source_path,
                            str(MARKDOWN_DIR),
                            f"Manual rerun from report-gap dashboard ({datetime.now().isoformat(timespec='seconds')})",
                        ],
                    )

                    con.execute(
                        "DELETE FROM annual_reports WHERE ticker = ? AND year = ?",
                        [ticker, year],
                    )
                    con.execute(
                        "DELETE FROM document_embeddings WHERE ticker = ? AND year = ?",
                        [ticker, year],
                    )
                    con.execute(
                        "DELETE FROM inference_results WHERE ticker = ? AND year = ?",
                        [ticker, year],
                    )
                    con.execute(
                        "DELETE FROM proper_vn_results WHERE ticker = ? AND year = ?",
                        [ticker, year],
                    )
                    con.execute(
                        "DELETE FROM governance_results WHERE ticker = ? AND year = ?",
                        [ticker, year],
                    )

                    con.execute(
                        """
                        UPDATE inference_jobs
                        SET status = 'pending',
                            categories_done = 0,
                            batch_id = NULL,
                            batch_submitted_at = NULL,
                            batch_checked_at = NULL,
                            started_at = NULL,
                            completed_at = NULL,
                            error_message = NULL
                        WHERE ticker = ? AND year = ?
                        """,
                        [ticker, year],
                    )
                    con.execute(
                        """
                        UPDATE proper_vn_jobs
                        SET status = 'pending',
                            indicators_done = 0,
                            batch_id = NULL,
                            batch_submitted_at = NULL,
                            batch_checked_at = NULL,
                            color = NULL,
                            s2_score = NULL,
                            s2_max_score = NULL,
                            started_at = NULL,
                            completed_at = NULL,
                            error_message = NULL
                        WHERE ticker = ? AND year = ?
                        """,
                        [ticker, year],
                    )
                    con.execute(
                        """
                        UPDATE governance_jobs
                        SET status = 'pending',
                            items_done = 0,
                            batch_id = NULL,
                            batch_submitted_at = NULL,
                            batch_checked_at = NULL,
                            started_at = NULL,
                            completed_at = NULL,
                            error_message = NULL
                        WHERE ticker = ? AND year = ?
                        """,
                        [ticker, year],
                    )
                finally:
                    con.close()

                ui.notify(
                    f"Queued rerun for {ticker}/{year} (convert -> load -> embed -> infer)",
                    type="positive",
                )
                gap_table.refresh()

            table.on("rerun", lambda e: _queue_rerun_for_row(e.args))

            with ui.row().classes("gap-2"):
                ui.button("Refresh", on_click=gap_table.refresh).props("dense outline")
                ui.button("View Row", on_click=_view_row_payload).props("dense outline")

        gap_table()


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
            from vietstock_documents import (
                DOC_TYPE_ANNUAL_REPORT,
                DOC_TYPE_AUDITED_CONSOLIDATED_FS,
                DOC_TYPE_LABELS,
            )

            doc_type_sel = ui.select(
                {
                    "": "All document types",
                    DOC_TYPE_AUDITED_CONSOLIDATED_FS: DOC_TYPE_LABELS[DOC_TYPE_AUDITED_CONSOLIDATED_FS],
                    DOC_TYPE_ANNUAL_REPORT: DOC_TYPE_LABELS[DOC_TYPE_ANNUAL_REPORT],
                },
                value="",
                label="Doc type",
            ).props("dense clearable")
            ui.button("Search", on_click=lambda: doc_table.refresh()).props(
                "dense flat"
            )

        @ui.refreshable
        def doc_table():
            con = get_connection()
            try:
                t = (ticker_sel.value or "").strip().upper()
                selected_doc_type = str(doc_type_sel.value or "").strip()
                where_parts: list[str] = []
                params: list[str] = []
                if t:
                    where_parts.append("ticker = ?")
                    params.append(t)
                if selected_doc_type:
                    where_parts.append("doc_type = ?")
                    params.append(selected_doc_type)

                sql = (
                    "SELECT id, ticker, doc_type, title, published_date, synced_to_raw, raw_path "
                    "FROM vietstock_documents "
                )
                if where_parts:
                    sql += "WHERE " + " AND ".join(where_parts) + " "
                sql += "ORDER BY ticker, published_date DESC LIMIT 200"
                rows = con.execute(sql, params).fetchall()
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
                    "name": "doc_type",
                    "label": "Doc Type",
                    "field": "doc_type",
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
                m = _re.search(r"(\d{4})", r[3] or "")
                year = m.group(1) if m else ""
                data.append(
                    {
                        "id": r[0],
                        "ticker": r[1],
                        "doc_type": DOC_TYPE_LABELS.get(str(r[2]), str(r[2])),
                        "title": r[3] or "",
                        "year": year,
                        "date": r[4] or "",
                        "synced": "✅" if r[5] else "❌",
                        "raw_path": r[6] or "",
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

        viewer_state = {
            "ticker": "",
            "year": None,
            "content": "",
            "meta": "",
            "source_file": "",
            "extract_items": [],
            "loaded": False,
        }

        def _list_report_rows() -> list[tuple]:
            con = get_connection()
            try:
                return con.execute(
                    "SELECT ticker, year, LENGTH(content) as chars, "
                    "source_file, created_at "
                    "FROM annual_reports ORDER BY ticker, year DESC"
                ).fetchall()
            finally:
                con.close()

        def _list_report_options() -> dict[str, str]:
            rows = _list_report_rows()
            return {
                f"{ticker}|{year}": f"{ticker} - {year}"
                for ticker, year, *_ in rows
            }

        with ui.tabs().classes("w-full") as report_tabs:
            ui.tab("reports_table", label="Loaded Reports")
            ui.tab("reports_viewer", label="Markdown Viewer")

        with ui.tab_panels(report_tabs, value="reports_table").classes("w-full"):
            with ui.tab_panel("reports_table"):
                ui.button(
                    "Refresh",
                    on_click=lambda: (
                        reports_table.refresh(),
                        _refresh_report_selector(),
                    ),
                ).props("dense flat")

                @ui.refreshable
                def reports_table():
                    rows = _list_report_rows()

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

            with ui.tab_panel("reports_viewer"):
                with ui.row().classes("items-end gap-2 flex-wrap w-full"):
                    report_selector = (
                        ui.select(
                            label="Loaded report",
                            options={},
                            with_input=True,
                        )
                        .props("dense outlined")
                        .classes("w-64")
                    )

                    def _load_selected_report() -> None:
                        selected = str(report_selector.value or "")
                        if "|" not in selected:
                            ui.notify(
                                "Select a ticker-year report first",
                                type="warning",
                            )
                            return

                        ticker, year_str = selected.split("|", 1)
                        year = int(year_str)

                        con = get_connection()
                        try:
                            row = con.execute(
                                "SELECT content, source_file, created_at "
                                "FROM annual_reports WHERE ticker = ? AND year = ?",
                                [ticker, year],
                            ).fetchone()

                            extract_rows = con.execute(
                                """
                                SELECT
                                    dataset,
                                    item_code,
                                    status,
                                    extra,
                                    reason,
                                    model,
                                    created_at
                                FROM (
                                    SELECT
                                        'EDC' AS dataset,
                                        category_code AS item_code,
                                        CASE WHEN is_valid THEN '1' ELSE '0' END AS status,
                                        '' AS extra,
                                        COALESCE(reason, '') AS reason,
                                        COALESCE(model, '') AS model,
                                        created_at
                                    FROM inference_results
                                    WHERE ticker = ? AND year = ?

                                    UNION ALL

                                    SELECT
                                        'PROPER_VN' AS dataset,
                                        indicator_code AS item_code,
                                        CASE WHEN is_present THEN '1' ELSE '0' END AS status,
                                        COALESCE(evidence_level, '') AS extra,
                                        COALESCE(reason, '') AS reason,
                                        COALESCE(model, '') AS model,
                                        created_at
                                    FROM proper_vn_results
                                    WHERE ticker = ? AND year = ?

                                    UNION ALL

                                    SELECT
                                        'GOVERNANCE' AS dataset,
                                        item_code,
                                        CASE WHEN found THEN '1' ELSE '0' END AS status,
                                        COALESCE(SUBSTR(value_json, 1, 160), '') AS extra,
                                        COALESCE(reason, '') AS reason,
                                        COALESCE(model, '') AS model,
                                        created_at
                                    FROM governance_results
                                    WHERE ticker = ? AND year = ?
                                ) all_items
                                ORDER BY dataset, item_code, created_at DESC
                                """,
                                [
                                    ticker,
                                    year,
                                    ticker,
                                    year,
                                    ticker,
                                    year,
                                ],
                            ).fetchall()
                        finally:
                            con.close()

                        if not row:
                            viewer_state["ticker"] = ""
                            viewer_state["year"] = None
                            viewer_state["content"] = ""
                            viewer_state["meta"] = ""
                            viewer_state["source_file"] = ""
                            viewer_state["extract_items"] = []
                            viewer_state["loaded"] = False
                            report_preview.refresh()
                            ui.notify("Report not found", type="warning")
                            return

                        content = str(row[0] or "")
                        source_file = str(row[1] or "")
                        created_at = row[2] or ""
                        extracted_items = [
                            {
                                "id": i,
                                "dataset": str(r[0]),
                                "item_code": str(r[1]),
                                "status": str(r[2]),
                                "extra": str(r[3] or ""),
                                "reason": str(r[4] or ""),
                                "model": str(r[5] or ""),
                                "created_at": str(r[6] or ""),
                            }
                            for i, r in enumerate(extract_rows)
                        ]

                        viewer_state["ticker"] = ticker
                        viewer_state["year"] = year
                        viewer_state["content"] = content
                        viewer_state["source_file"] = source_file
                        viewer_state["extract_items"] = extracted_items
                        viewer_state["meta"] = (
                            f"{ticker} {year} | {len(content):,} chars | Loaded at {created_at}"
                        )
                        viewer_state["loaded"] = True
                        report_preview.refresh()

                    def _refresh_report_selector() -> None:
                        options = _list_report_options()
                        report_selector.options = options
                        if not options:
                            report_selector.value = None
                        elif report_selector.value not in options:
                            report_selector.value = next(iter(options.keys()))
                        report_selector.update()

                    ui.button("Load", on_click=_load_selected_report).props(
                        "dense color=primary"
                    )
                    ui.button(
                        "Refresh List", on_click=_refresh_report_selector
                    ).props("dense flat")

                @ui.refreshable
                def report_preview():
                    if not viewer_state["loaded"]:
                        ui.label("Select a loaded report and click Load").classes(
                            "text-gray-500"
                        )
                        return

                    ui.label(str(viewer_state["meta"])).classes(
                        "text-xs text-gray-500"
                    )
                    if viewer_state["source_file"]:
                        ui.label(f"Source: {viewer_state['source_file']}").classes(
                            "text-xs text-gray-500"
                        )
                    ui.separator()

                    with ui.row().classes("w-full items-start gap-4"):
                        with ui.column().classes("w-full lg:w-8/12"):
                            ui.markdown(str(viewer_state["content"])).classes("w-full")

                        with ui.column().classes("w-full lg:w-4/12"):
                            items = list(viewer_state["extract_items"])
                            ui.label(
                                f"Extracted Items ({len(items)})"
                            ).classes("text-sm font-bold")

                            if not items:
                                ui.label(
                                    "No extracted items for this ticker/year yet"
                                ).classes("text-gray-500 text-sm")
                                return

                            table = (
                                ui.table(
                                    columns=[
                                        {
                                            "name": "dataset",
                                            "label": "Dataset",
                                            "field": "dataset",
                                            "align": "left",
                                        },
                                        {
                                            "name": "item_code",
                                            "label": "Code",
                                            "field": "item_code",
                                            "align": "left",
                                        },
                                        {
                                            "name": "status",
                                            "label": "Status",
                                            "field": "status",
                                            "align": "center",
                                        },
                                    ],
                                    rows=items,
                                    row_key="id",
                                    selection="single",
                                )
                                .classes("w-full")
                                .props("dense flat")
                            )

                            payload_area = ui.code(
                                "Select an extracted item row, then click View"
                            ).classes("w-full max-h-72 overflow-auto text-xs")

                            def _view_sidebar_payload():
                                selected = table.selected
                                if not selected:
                                    ui.notify("Select an extracted item", type="warning")
                                    return
                                payload_area.set_content(
                                    json.dumps(
                                        selected[0],
                                        indent=2,
                                        ensure_ascii=False,
                                    )
                                )

                            ui.button(
                                "View",
                                on_click=_view_sidebar_payload,
                            ).props("dense outline")

                _refresh_report_selector()
                report_preview()


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
    from vietstock_documents import (
        DOC_TYPE_ANNUAL_REPORT,
        DOC_TYPE_AUDITED_CONSOLIDATED_FS,
        DOC_TYPE_LABELS,
    )

    def _output_dir_for_doc_type(doc_type: str) -> Path:
        if str(doc_type) == DOC_TYPE_AUDITED_CONSOLIDATED_FS:
            return FINANCIAL_STATEMENT_MARKDOWN_DIR
        return MARKDOWN_DIR

    ui.dark_mode(False)
    _nav_header()
    init_db()

    from converter import (
        convert_financial_statement_to_markdown,
        create_jobs,
        get_job_log,
        run_job,
    )
    from vietstock_documents import (
        download_all_unsynced,
        fetch_all_companies_documents,
    )

    current_year = datetime.now().year

    with ui.column().classes("w-full max-w-6xl mx-auto gap-4 p-4"):
        ui.label("Converter").classes("text-2xl font-bold")
        ui.label(
            "Simple document and marker-pdf conversion manager for Annual Reports and Financial Statements."
        ).classes("text-sm text-gray-600")

        live_state: dict[str, str | bool | int] = {
            "running": False,
            "label": "",
            "elapsed": 0,
            "last_update": "",
            "message": "Idle",
        }

        @ui.refreshable
        def live_status_panel() -> None:
            with ui.card().classes("w-full"):
                ui.label("Live Conversion Status").classes("text-sm font-semibold")
                if bool(live_state["running"]):
                    ui.label(
                        f"⏳ {str(live_state['label'])} | elapsed={int(cast(int, live_state['elapsed']))}s | last update={str(live_state['last_update'])}"
                    ).classes("text-sm text-blue-700")
                    ui.spinner(size="sm")
                else:
                    ui.label(f"{str(live_state['message'])}").classes("text-sm text-gray-600")

        def _run_conversion_background(
            label: str,
            work: Callable[[Callable[[int], None]], str],
            on_done: Callable[[], object] | None = None,
        ) -> None:
            if bool(live_state["running"]):
                ui.notify("Another conversion is already running", type="warning")
                return

            live_state["running"] = True
            live_state["label"] = label
            live_state["elapsed"] = 0
            live_state["last_update"] = datetime.now().strftime("%H:%M:%S")
            live_state["message"] = f"Starting {label}..."
            live_status_panel.refresh()

            def _heartbeat(elapsed: int) -> None:
                live_state["elapsed"] = int(elapsed)
                live_state["last_update"] = datetime.now().strftime("%H:%M:%S")
                live_state["message"] = f"[heartbeat] {label} is still running"
                live_status_panel.refresh()

            def _worker() -> None:
                try:
                    message = work(_heartbeat)
                    live_state["message"] = message
                except Exception as exc:
                    live_state["message"] = f"❌ {label} failed: {exc}"
                finally:
                    live_state["running"] = False
                    live_state["last_update"] = datetime.now().strftime("%H:%M:%S")
                    live_status_panel.refresh()
                    if on_done is not None:
                        on_done()

            threading.Thread(target=_worker, daemon=True).start()

        live_status_panel()

        with ui.row().classes("items-end gap-3 flex-wrap w-full"):
            ticker_filter_input = (
                ui.input("Ticker contains")
                .props("dense clearable outlined")
                .classes("w-52")
            )
            start_year_input = (
                ui.number(
                    "Start year",
                    value=DEFAULT_START_YEAR,
                    min=2010,
                    max=current_year,
                    step=1,
                    format="%.0f",
                )
                .props("dense outlined")
                .classes("w-32")
            )
            end_year_input = (
                ui.number(
                    "End year",
                    value=DEFAULT_END_YEAR,
                    min=2010,
                    max=current_year,
                    step=1,
                    format="%.0f",
                )
                .props("dense outlined")
                .classes("w-32")
            )
            select_all_reports_toggle = ui.switch(
                "Select all filtered reports",
                value=True,
            ).props("dense")
            skip_processed_toggle = ui.switch(
                "Skip processed files",
                value=True,
            ).props("dense")
            manual_financial_statement_fallback_toggle = ui.switch(
                "Financial statement: use manually extracted/renamed PDF fallback",
                value=False,
            ).props("dense")

        def _selected_year_bounds() -> tuple[int, int]:
            sy = int(start_year_input.value or DEFAULT_START_YEAR)
            ey = int(end_year_input.value or DEFAULT_END_YEAR)
            if sy > ey:
                sy, ey = ey, sy
            return sy, ey

        def _ticker_filter_sql(column_name: str = "ticker") -> tuple[str, list[str]]:
            ticker_filter = str(ticker_filter_input.value or "").strip().upper()
            if not ticker_filter:
                return "", []
            return f" AND {column_name} LIKE ?", [f"%{ticker_filter}%"]

        with ui.tabs().classes("w-full") as converter_tabs:
            ui.tab("annual", label="Annual Reports")
            ui.tab("financial_statement", label="Financial Statement")

        with ui.tab_panels(converter_tabs, value="annual").classes("w-full"):
            with ui.tab_panel("annual"):
                @ui.refreshable
                def annual_panel() -> None:
                    sy, ey = _selected_year_bounds()
                    ticker_where, ticker_params = _ticker_filter_sql("d.ticker")

                    con = get_connection()
                    try:
                        rows = con.execute(
                            f"""
                            WITH annual_docs AS (
                                SELECT
                                    ticker,
                                    TRY_CAST(regexp_extract(title, '(\\d{{4}})', 1) AS INTEGER) AS year,
                                    SUM(CASE WHEN synced_to_raw THEN 1 ELSE 0 END) AS raw_count,
                                    COUNT(*) AS doc_count
                                FROM vietstock_documents
                                WHERE doc_type = ?
                                GROUP BY ticker, year
                            )
                            SELECT
                                d.ticker,
                                d.year,
                                d.doc_count,
                                d.raw_count,
                                cj.id,
                                cj.status,
                                cj.started_at,
                                cj.completed_at
                            FROM annual_docs d
                            LEFT JOIN conversion_jobs cj
                              ON cj.ticker = d.ticker
                             AND cj.year = d.year
                             AND cj.output_dir = ?
                            WHERE d.year IS NOT NULL
                              AND d.year BETWEEN ? AND ?
                              {ticker_where}
                            ORDER BY d.ticker, d.year DESC
                            """,
                            [
                                DOC_TYPE_ANNUAL_REPORT,
                                str(MARKDOWN_DIR),
                                sy,
                                ey,
                                *ticker_params,
                            ],
                        ).fetchall()
                    finally:
                        con.close()

                    table_rows: list[dict[str, object]] = []
                    for ticker, year, doc_count, raw_count, job_id, status, started, completed in rows:
                        ticker_key = str(ticker).upper()
                        year_key = int(year)
                        has_markdown = "❌"
                        ticker_dir = MARKDOWN_DIR / ticker_key
                        if ticker_dir.exists():
                            year_text = str(year_key)
                            for md_path in ticker_dir.rglob("*.md"):
                                haystack = " ".join(
                                    (
                                        md_path.name,
                                        md_path.stem,
                                        md_path.parent.name,
                                        md_path.as_posix(),
                                    )
                                )
                                if year_text in haystack:
                                    has_markdown = "✅"
                                    break

                        table_rows.append(
                            {
                                "id": f"{ticker_key}-{year_key}",
                                "ticker": ticker_key,
                                "year": year_key,
                                "docs": int(doc_count or 0),
                                "raw": int(raw_count or 0),
                                "job_id": int(job_id) if job_id is not None else "",
                                "job_status": str(status or ""),
                                "markdown": has_markdown,
                                "started": str(started)[:19] if started else "",
                                "completed": str(completed)[:19] if completed else "",
                            }
                        )

                    with ui.row().classes("gap-2 flex-wrap"):
                        def _sync_annual_listings() -> None:
                            tickers: list[str] | None = None
                            ticker_filter = str(ticker_filter_input.value or "").strip().upper()
                            if ticker_filter:
                                con1 = get_connection()
                                try:
                                    ticker_rows = con1.execute(
                                        "SELECT ticker FROM companies WHERE ticker LIKE ? ORDER BY ticker",
                                        [f"%{ticker_filter}%"],
                                    ).fetchall()
                                finally:
                                    con1.close()
                                tickers = [str(row[0]).upper() for row in ticker_rows]
                            con2 = get_connection()
                            try:
                                out = fetch_all_companies_documents(
                                    con2,
                                    doc_type=DOC_TYPE_ANNUAL_REPORT,
                                    tickers=tickers,
                                    start_year=sy,
                                    end_year=ey,
                                )
                            finally:
                                con2.close()
                            ui.notify(f"Annual listings synced for {len(out)} ticker(s)", type="positive")
                            annual_panel.refresh()

                        def _download_annual_unsynced() -> None:
                            con3 = get_connection()
                            try:
                                out = download_all_unsynced(
                                    con3,
                                    doc_type=DOC_TYPE_ANNUAL_REPORT,
                                )
                            finally:
                                con3.close()
                            total = sum(len(v) for v in out.values())
                            ui.notify(f"Downloaded {total} annual PDF(s)", type="positive")
                            annual_panel.refresh()

                        ui.button("Sync Annual Listings", on_click=_sync_annual_listings, color="primary").props("dense")
                        ui.button("Download Unsynced Annual PDFs", on_click=_download_annual_unsynced).props("dense outline")

                    if not table_rows:
                        ui.label("No annual rows match current filters").classes("text-gray-500 text-sm")
                        return

                    ui.label(f"{len(table_rows)} annual row(s)").classes("text-xs text-gray-500")
                    annual_table = ui.table(
                        columns=[
                            {"name": "ticker", "label": "Ticker", "field": "ticker", "align": "left"},
                            {"name": "year", "label": "Year", "field": "year", "align": "center"},
                            {"name": "docs", "label": "Listings", "field": "docs", "align": "center"},
                            {"name": "raw", "label": "Raw PDFs", "field": "raw", "align": "center"},
                            {"name": "job_id", "label": "Job ID", "field": "job_id", "align": "center"},
                            {"name": "job_status", "label": "Job Status", "field": "job_status", "align": "center"},
                            {"name": "markdown", "label": "Markdown", "field": "markdown", "align": "center"},
                        ],
                        rows=table_rows,
                        row_key="id",
                        selection="multiple",
                    ).classes("w-full").props("dense flat")

                    annual_preview = ui.code("Select one row for actions or details").classes(
                        "w-full max-h-72 overflow-auto text-xs"
                    )

                    def _selected_annual_row() -> dict[str, object] | None:
                        selected = annual_table.selected
                        if not selected:
                            ui.notify("Select one annual row first", type="warning")
                            return None
                        return selected[0]

                    def _selected_annual_rows() -> list[dict[str, object]]:
                        if bool(select_all_reports_toggle.value):
                            rows = list(table_rows)
                        else:
                            rows = list(annual_table.selected or [])
                            if not rows:
                                ui.notify("Select one or more annual rows first", type="warning")
                                return []

                        if bool(skip_processed_toggle.value):
                            rows = [
                                r
                                for r in rows
                                if str(r.get("markdown") or "") != "✅"
                                and str(r.get("job_status") or "").lower() != "completed"
                            ]
                        return rows

                    def _create_refresh_selected_annual_job() -> None:
                        row = _selected_annual_row()
                        if not row:
                            return
                        ticker = str(row.get("ticker") or "").upper()
                        year = int(str(row.get("year") or "0"))
                        con4 = get_connection()
                        try:
                            count = create_jobs(
                                con4,
                                tickers=[ticker],
                                years=[year],
                                start_year=year,
                                end_year=year,
                                doc_type=DOC_TYPE_ANNUAL_REPORT,
                                output_dir=MARKDOWN_DIR,
                            )
                        finally:
                            con4.close()
                        ui.notify(f"Created/refreshed {count} annual job(s) for {ticker}/{year}", type="positive")
                        annual_panel.refresh()

                    def _run_selected_annual_job() -> None:
                        row = _selected_annual_row()
                        if not row:
                            return
                        ticker = str(row.get("ticker") or "").upper()
                        year = int(str(row.get("year") or "0"))

                        def _work(on_hb: Callable[[int], None]) -> str:
                            con5 = get_connection()
                            try:
                                create_jobs(
                                    con5,
                                    tickers=[ticker],
                                    years=[year],
                                    start_year=year,
                                    end_year=year,
                                    doc_type=DOC_TYPE_ANNUAL_REPORT,
                                    output_dir=MARKDOWN_DIR,
                                )
                                job_row = con5.execute(
                                    """
                                    SELECT id
                                    FROM conversion_jobs
                                    WHERE ticker = ? AND year = ? AND output_dir = ?
                                    ORDER BY id DESC
                                    LIMIT 1
                                    """,
                                    [ticker, year, str(MARKDOWN_DIR)],
                                ).fetchone()
                                if not job_row:
                                    raise RuntimeError("Conversion job not found")
                                success = run_job(
                                    con5,
                                    int(job_row[0]),
                                    on_heartbeat=on_hb,
                                )
                            finally:
                                con5.close()
                            return (
                                f"✅ Annual conversion completed for {ticker}/{year}"
                                if success
                                else f"❌ Annual conversion failed for {ticker}/{year}"
                            )

                        _run_conversion_background(
                            f"Annual {ticker}/{year}",
                            _work,
                            on_done=annual_panel.refresh,
                        )

                    def _run_pending_annual_jobs() -> None:
                        annual_targets = _selected_annual_rows()
                        if not annual_targets:
                            if bool(skip_processed_toggle.value):
                                ui.notify("No annual rows left after skipping processed files", type="warning")
                            return

                        def _work(on_hb: Callable[[int], None]) -> str:
                            con6 = get_connection()
                            try:
                                ok = 0
                                failed = 0
                                for row in annual_targets:
                                    ticker = str(row.get("ticker") or "").upper()
                                    year = int(str(row.get("year") or "0"))
                                    create_jobs(
                                        con6,
                                        tickers=[ticker],
                                        years=[year],
                                        start_year=year,
                                        end_year=year,
                                        doc_type=DOC_TYPE_ANNUAL_REPORT,
                                        output_dir=MARKDOWN_DIR,
                                    )
                                    job_row = con6.execute(
                                        """
                                        SELECT id
                                        FROM conversion_jobs
                                        WHERE ticker = ? AND year = ? AND output_dir = ?
                                        ORDER BY id DESC
                                        LIMIT 1
                                        """,
                                        [ticker, year, str(MARKDOWN_DIR)],
                                    ).fetchone()
                                    if not job_row:
                                        failed += 1
                                        continue
                                    if run_job(con6, int(job_row[0]), on_heartbeat=on_hb):
                                        ok += 1
                                    else:
                                        failed += 1
                            finally:
                                con6.close()
                            return f"Annual batch conversion finished: {ok} completed, {failed} failed"

                        _run_conversion_background(
                            "Annual batch conversion",
                            _work,
                            on_done=annual_panel.refresh,
                        )

                    def _view_selected_annual_log() -> None:
                        row = _selected_annual_row()
                        if not row:
                            return
                        job_id = row.get("job_id")
                        if not job_id:
                            annual_preview.set_content("No conversion job/log for selected row")
                            return
                        con7 = get_connection()
                        try:
                            text = get_job_log(
                                con7,
                                int(str(job_id)),
                                tail=200,
                            ) or "No log found"
                        finally:
                            con7.close()
                        annual_preview.set_content(text)

                    def _open_selected_annual_output() -> None:
                        row = _selected_annual_row()
                        if not row:
                            return
                        ticker = str(row.get("ticker") or "").upper()
                        year = int(str(row.get("year") or "0"))
                        folder = MARKDOWN_DIR / ticker
                        if not folder.exists():
                            annual_preview.set_content(f"Folder not found: {folder}")
                            return
                        year_text = str(year)
                        matches: list[str] = []
                        for md_path in folder.rglob("*.md"):
                            haystack = " ".join((md_path.name, md_path.stem, md_path.parent.name, md_path.as_posix()))
                            if year_text in haystack:
                                matches.append(str(md_path))
                        annual_preview.set_content(
                            f"Output folder: {folder}\n\n" + ("\n".join(matches) if matches else "No markdown files found for selected year")
                        )

                    with ui.row().classes("gap-2 mt-2 flex-wrap"):
                        ui.button("Create/Refresh Selected Job", on_click=_create_refresh_selected_annual_job, color="primary").props("dense")
                        ui.button("Run Selected Job", on_click=_run_selected_annual_job, color="primary").props("dense")
                        ui.button("Run Annual Batch", on_click=_run_pending_annual_jobs).props("dense outline")
                        ui.button("View Selected Log", on_click=_view_selected_annual_log).props("dense outline")
                        ui.button("Open Output Folder", on_click=_open_selected_annual_output).props("dense outline")

                annual_panel()

            with ui.tab_panel("financial_statement"):
                @ui.refreshable
                def financial_statement_panel() -> None:
                    sy, ey = _selected_year_bounds()
                    ticker_where, ticker_params = _ticker_filter_sql()

                    con = get_connection()
                    try:
                        rows = con.execute(
                            f"""
                            WITH financial_statement_docs AS (
                                SELECT
                                    ticker,
                                    TRY_CAST(regexp_extract(title, '(\\d{{4}})', 1) AS INTEGER) AS year,
                                    SUM(CASE WHEN synced_to_raw THEN 1 ELSE 0 END) AS raw_count,
                                    COUNT(*) AS doc_count
                                FROM vietstock_documents
                                WHERE doc_type = ?
                                GROUP BY ticker, year
                            )
                            SELECT ticker, year, doc_count, raw_count
                            FROM financial_statement_docs
                            WHERE year IS NOT NULL
                              AND year BETWEEN ? AND ?
                              {ticker_where}
                            ORDER BY ticker, year DESC
                            """,
                            [
                                DOC_TYPE_AUDITED_CONSOLIDATED_FS,
                                sy,
                                ey,
                                *ticker_params,
                            ],
                        ).fetchall()
                    finally:
                        con.close()

                    from config_marker import LOGS_DIR

                    table_rows: list[dict[str, object]] = []
                    for ticker, year, doc_count, raw_count in rows:
                        ticker_key = str(ticker).upper()
                        year_key = int(year)
                        has_markdown = "❌"
                        ticker_dir = FINANCIAL_STATEMENT_MARKDOWN_DIR / ticker_key
                        if ticker_dir.exists():
                            year_text = str(year_key)
                            for md_path in ticker_dir.rglob("*.md"):
                                haystack = " ".join(
                                    (
                                        md_path.name,
                                        md_path.stem,
                                        md_path.parent.name,
                                        md_path.as_posix(),
                                    )
                                )
                                if year_text in haystack:
                                    has_markdown = "✅"
                                    break

                        latest_log = ""
                        latest_result = ""
                        pattern = f"financial_statement_{ticker_key}_{year_key}_*.log"
                        logs = sorted(
                            LOGS_DIR.glob(pattern),
                            key=lambda p: p.stat().st_mtime,
                            reverse=True,
                        )
                        if logs:
                            latest_log = logs[0].name
                            text = logs[0].read_text(errors="replace")
                            m = re.search(r"Exit code:\\s*(\\d+)", text)
                            if m:
                                latest_result = "✅ success" if int(m.group(1)) == 0 else f"❌ fail ({m.group(1)})"

                        table_rows.append(
                            {
                                "id": f"{ticker_key}-{year_key}",
                                "ticker": ticker_key,
                                "year": year_key,
                                "docs": int(doc_count or 0),
                                "raw": int(raw_count or 0),
                                "markdown": has_markdown,
                                "latest_result": latest_result,
                                "latest_log": latest_log,
                            }
                        )

                    with ui.row().classes("gap-2 flex-wrap"):
                        def _sync_financial_statement_listings() -> None:
                            tickers: list[str] | None = None
                            ticker_filter = str(ticker_filter_input.value or "").strip().upper()
                            if ticker_filter:
                                con1 = get_connection()
                                try:
                                    ticker_rows = con1.execute(
                                        "SELECT ticker FROM companies WHERE ticker LIKE ? ORDER BY ticker",
                                        [f"%{ticker_filter}%"],
                                    ).fetchall()
                                finally:
                                    con1.close()
                                tickers = [str(row[0]).upper() for row in ticker_rows]
                            con2 = get_connection()
                            try:
                                out = fetch_all_companies_documents(
                                    con2,
                                    doc_type=DOC_TYPE_AUDITED_CONSOLIDATED_FS,
                                    tickers=tickers,
                                    start_year=sy,
                                    end_year=ey,
                                )
                            finally:
                                con2.close()
                            ui.notify(f"Financial statement listings synced for {len(out)} ticker(s)", type="positive")
                            financial_statement_panel.refresh()

                        def _download_financial_statement_unsynced() -> None:
                            con3 = get_connection()
                            try:
                                out = download_all_unsynced(
                                    con3,
                                    doc_type=DOC_TYPE_AUDITED_CONSOLIDATED_FS,
                                )
                            finally:
                                con3.close()
                            total = sum(len(v) for v in out.values())
                            ui.notify(f"Downloaded {total} financial statement PDF(s)", type="positive")
                            financial_statement_panel.refresh()

                        ui.button("Sync Financial Statement Listings", on_click=_sync_financial_statement_listings, color="primary").props("dense")
                        ui.button("Download Unsynced Financial Statement PDFs", on_click=_download_financial_statement_unsynced).props("dense outline")

                    if not table_rows:
                        ui.label("No financial-statement rows match current filters").classes("text-gray-500 text-sm")
                        return

                    ui.label(f"{len(table_rows)} financial-statement row(s)").classes("text-xs text-gray-500")
                    financial_statement_table = ui.table(
                        columns=[
                            {"name": "ticker", "label": "Ticker", "field": "ticker", "align": "left"},
                            {"name": "year", "label": "Year", "field": "year", "align": "center"},
                            {"name": "docs", "label": "Listings", "field": "docs", "align": "center"},
                            {"name": "raw", "label": "Raw PDFs", "field": "raw", "align": "center"},
                            {"name": "markdown", "label": "Markdown", "field": "markdown", "align": "center"},
                            {"name": "latest_result", "label": "Latest Result", "field": "latest_result", "align": "center"},
                            {"name": "latest_log", "label": "Latest Log", "field": "latest_log", "align": "left"},
                        ],
                        rows=table_rows,
                        row_key="id",
                        selection="multiple",
                    ).classes("w-full").props("dense flat")

                    financial_statement_preview = ui.code("Select one financial-statement row for actions or details").classes(
                        "w-full max-h-72 overflow-auto text-xs"
                    )

                    def _selected_financial_statement_rows_selected_only(
                        *,
                        skip_processed: bool = False,
                    ) -> list[dict[str, object]]:
                        selected = list(financial_statement_table.selected or [])
                        if not selected:
                            ui.notify("Select one or more financial-statement rows first", type="warning")
                            return []

                        rows_by_id = {
                            str(r.get("id") or ""): r for r in table_rows
                        }
                        resolved_rows: list[dict[str, object]] = []
                        for item in selected:
                            if isinstance(item, dict):
                                resolved_rows.append(item)
                                continue
                            row = rows_by_id.get(str(item))
                            if row:
                                resolved_rows.append(row)

                        if skip_processed and bool(skip_processed_toggle.value):
                            resolved_rows = [
                                r
                                for r in resolved_rows
                                if str(r.get("markdown") or "") != "✅"
                            ]

                        return resolved_rows

                    def _selected_financial_statement_row() -> dict[str, object] | None:
                        rows = _selected_financial_statement_rows_selected_only()
                        if not rows:
                            return None
                        return rows[0]

                    def _selected_financial_statement_rows() -> list[dict[str, object]]:
                        if bool(select_all_reports_toggle.value):
                            rows = list(table_rows)
                        else:
                            rows = list(financial_statement_table.selected or [])
                            if not rows:
                                ui.notify("Select one or more financial-statement rows first", type="warning")
                                return []

                        if bool(skip_processed_toggle.value):
                            rows = [
                                r
                                for r in rows
                                if str(r.get("markdown") or "") != "✅"
                            ]
                        return rows

                    def _convert_selected_financial_statement_rows() -> None:
                        selected_rows = _selected_financial_statement_rows_selected_only(
                            skip_processed=True
                        )
                        if not selected_rows:
                            if bool(skip_processed_toggle.value):
                                ui.notify("No selected financial-statement rows left after skipping processed files", type="warning")
                            return

                        def _work(on_hb: Callable[[int], None]) -> str:
                            ok = 0
                            failed = 0
                            con4 = get_connection()
                            try:
                                for row in selected_rows:
                                    ticker = str(row.get("ticker") or "").upper()
                                    year = int(str(row.get("year") or "0"))
                                    try:
                                        convert_financial_statement_to_markdown(
                                            con4,
                                            ticker=ticker,
                                            year=year,
                                            output_dir=FINANCIAL_STATEMENT_MARKDOWN_DIR,
                                            allow_manual_updated_files=bool(
                                                manual_financial_statement_fallback_toggle.value
                                            ),
                                            on_heartbeat=on_hb,
                                        )
                                        ok += 1
                                    except Exception:
                                        failed += 1
                            finally:
                                con4.close()
                            return f"Selected financial-statement conversion finished: {ok} completed, {failed} failed"

                        _run_conversion_background(
                            f"Financial statement selected batch ({len(selected_rows)} rows)",
                            _work,
                            on_done=financial_statement_panel.refresh,
                        )

                    def _convert_filtered_financial_statement_rows() -> None:
                        financial_statement_targets = _selected_financial_statement_rows()
                        if not financial_statement_targets:
                            if bool(skip_processed_toggle.value):
                                ui.notify("No financial-statement rows left after skipping processed files", type="warning")
                            return

                        def _work(on_hb: Callable[[int], None]) -> str:
                            ok = 0
                            failed = 0
                            con5 = get_connection()
                            try:
                                for row in financial_statement_targets:
                                    ticker = str(row.get("ticker") or "").upper()
                                    year = int(str(row.get("year") or "0"))
                                    try:
                                        convert_financial_statement_to_markdown(
                                            con5,
                                            ticker=ticker,
                                            year=year,
                                            output_dir=FINANCIAL_STATEMENT_MARKDOWN_DIR,
                                            allow_manual_updated_files=bool(
                                                manual_financial_statement_fallback_toggle.value
                                            ),
                                            on_heartbeat=on_hb,
                                        )
                                        ok += 1
                                    except Exception:
                                        failed += 1
                            finally:
                                con5.close()
                            return f"Financial statement conversion finished: {ok} completed, {failed} failed"

                        _run_conversion_background(
                            "Financial statement filtered conversion",
                            _work,
                            on_done=financial_statement_panel.refresh,
                        )

                    def _view_selected_financial_statement_log() -> None:
                        row = _selected_financial_statement_row()
                        if not row:
                            return
                        ticker = str(row.get("ticker") or "").upper()
                        year = int(str(row.get("year") or "0"))
                        pattern = f"financial_statement_{ticker}_{year}_*.log"
                        logs = sorted(
                            LOGS_DIR.glob(pattern),
                            key=lambda p: p.stat().st_mtime,
                            reverse=True,
                        )
                        if not logs:
                            financial_statement_preview.set_content("No log found for selected row")
                            return
                        financial_statement_preview.set_content(logs[0].read_text(errors="replace"))

                    def _open_selected_financial_statement_output() -> None:
                        row = _selected_financial_statement_row()
                        if not row:
                            return
                        ticker = str(row.get("ticker") or "").upper()
                        year = int(str(row.get("year") or "0"))
                        folder = FINANCIAL_STATEMENT_MARKDOWN_DIR / ticker
                        if not folder.exists():
                            financial_statement_preview.set_content(f"Folder not found: {folder}")
                            return
                        year_text = str(year)
                        matches: list[str] = []
                        for md_path in folder.rglob("*.md"):
                            haystack = " ".join((md_path.name, md_path.stem, md_path.parent.name, md_path.as_posix()))
                            if year_text in haystack:
                                matches.append(str(md_path))
                        financial_statement_preview.set_content(
                            f"Output folder: {folder}\n\n" + ("\n".join(matches) if matches else "No markdown files found for selected year")
                        )

                    with ui.row().classes("gap-2 mt-2 flex-wrap"):
                        ui.button("Convert Selected Financial Statements", on_click=_convert_selected_financial_statement_rows, color="primary").props("dense")
                        ui.button("Run Financial Statement Batch", on_click=_convert_filtered_financial_statement_rows).props("dense outline")
                        ui.button("View Selected Log", on_click=_view_selected_financial_statement_log).props("dense outline")
                        ui.button("Open Output Folder", on_click=_open_selected_financial_statement_output).props("dense outline")

                financial_statement_panel()

        with ui.row().classes("gap-2"):
            ui.button("Refresh Annual", on_click=lambda: annual_panel.refresh()).props("dense outline")
            ui.button("Refresh Financial Statement", on_click=lambda: financial_statement_panel.refresh()).props("dense outline")

        ticker_filter_input.on_value_change(lambda _: (annual_panel.refresh(), financial_statement_panel.refresh()))
        start_year_input.on_value_change(lambda _: (annual_panel.refresh(), financial_statement_panel.refresh()))
        end_year_input.on_value_change(lambda _: (annual_panel.refresh(), financial_statement_panel.refresh()))

    return

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
