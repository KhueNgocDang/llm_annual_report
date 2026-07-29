from __future__ import annotations

from datetime import datetime

from nicegui import ui

from config import DB_PATH, DEFAULT_END_YEAR, DEFAULT_START_YEAR, bootstrap_directories
from database import (
    connection_scope,
    delete_company,
    ensure_company,
    init_db,
    list_companies,
)
from financial_data import sync_stocks
from financial_data import (
    sync_financial_models,
    sync_financial_ratios_all,
    sync_financial_statements_all,
)
from vietstock_documents import (
    download_all_unsynced,
    sync_documents_for_all_companies,
    sync_documents_for_ticker,
    upsert_document_stub,
)


def _bootstrap_now() -> None:
    bootstrap_directories()
    init_db()
    ui.notify("Bootstrap complete", type="positive")


def _stats_text() -> str:
    try:
        with connection_scope() as con:
            company_count = con.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
            doc_count = con.execute("SELECT COUNT(*) FROM vietstock_documents").fetchone()[0]
            report_count = con.execute("SELECT COUNT(*) FROM annual_reports").fetchone()[0]
            return (
                f"companies={company_count} | documents={doc_count} | "
                f"annual_reports={report_count}"
            )
    except Exception as exc:
        return f"Database not ready: {exc}"


def _refresh_companies_table(table: ui.table) -> None:
    table.rows = [{"ticker": t} for t in list_companies()]
    table.update()


def _add_companies(raw_text: str, table: ui.table) -> None:
    added = 0
    for token in raw_text.replace(",", " ").split():
        code = token.strip().upper()
        if not code:
            continue
        ensure_company(code)
        added += 1
    _refresh_companies_table(table)
    ui.notify(f"Added/kept {added} tickers", type="positive")


def _sync_stocks_now(stats: ui.label) -> None:
    n = sync_stocks()
    stats.set_text(_stats_text())
    ui.notify(f"Synced {n} stocks", type="positive")


def _sync_models_now(stats: ui.label) -> None:
    n = sync_financial_models()
    stats.set_text(_stats_text())
    ui.notify(f"Synced {n} financial model rows", type="positive")


def _sync_statements_now(stats: ui.label, start_year: int, end_year: int) -> None:
    result = sync_financial_statements_all(start_year=start_year, end_year=end_year)
    stats.set_text(_stats_text())
    ui.notify(
        (
            f"Statements sync: rows={result['rows']} "
            f"tickers_ok={result['tickers_done']} failed={result['tickers_failed']}"
        ),
        type="positive" if result["tickers_failed"] == 0 else "warning",
    )


def _sync_ratios_now(stats: ui.label, start_year: int, end_year: int) -> None:
    result = sync_financial_ratios_all(start_year=start_year, end_year=end_year)
    stats.set_text(_stats_text())
    ui.notify(
        (
            f"Ratios sync: rows={result['rows']} "
            f"tickers_ok={result['tickers_done']} failed={result['tickers_failed']}"
        ),
        type="positive" if result["tickers_failed"] == 0 else "warning",
    )


def _queue_stub_document(
    ticker: str,
    doc_type: str,
    title: str,
    file_url: str,
    doc_id: str,
) -> None:
    doc_id_int = int(doc_id.strip())
    upsert_document_stub(
        doc_id=doc_id_int,
        ticker=ticker,
        doc_type=doc_type,
        title=title,
        file_url=file_url,
    )
    ui.notify(f"Queued document id={doc_id_int}", type="positive")


def _download_unsynced_now(stats: ui.label, use_llm: bool) -> None:
    result = download_all_unsynced(limit=200, use_llm_selection=use_llm)
    stats.set_text(_stats_text())
    ui.notify(
        f"Downloaded {result['done']}/{result['total']} (failed={result['failed']})",
        type="positive" if result["failed"] == 0 else "warning",
    )


def _sync_docs_one_now(stats: ui.label, ticker: str, doc_type: str) -> None:
    n = sync_documents_for_ticker(ticker=ticker, doc_type=doc_type)
    stats.set_text(_stats_text())
    ui.notify(f"Synced {n} document rows for {ticker.upper()}", type="positive")


def _sync_docs_all_now(stats: ui.label, doc_type: str) -> None:
    result = sync_documents_for_all_companies(doc_type=doc_type)
    stats.set_text(_stats_text())
    ui.notify(
        (
            f"Listings sync done: rows={result['rows']} "
            f"tickers_ok={result['tickers_done']} failed={result['tickers_failed']}"
        ),
        type="positive" if result["tickers_failed"] == 0 else "warning",
    )


def build_ui() -> None:
    ui.page_title("Annual Report Intelligence Platform")

    with ui.column().classes("w-full max-w-4xl mx-auto p-6 gap-4"):
        ui.label("Annual Report Intelligence Platform").classes("text-3xl font-bold")
        ui.label("Rebuild foundation is running.").classes("text-gray-600")

        with ui.card().classes("w-full"):
            ui.label("Phase 0 Controls").classes("text-lg font-semibold")
            with ui.row().classes("items-center gap-2"):
                ui.button("Bootstrap directories + DB", on_click=_bootstrap_now)
                ui.button("Refresh stats", on_click=lambda: stats.set_text(_stats_text()))

            stats = ui.label(_stats_text()).classes("text-sm text-gray-700")
            ui.label(f"DB path: {DB_PATH}").classes("text-xs text-gray-500")
            ui.label(
                f"Last rendered: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ).classes("text-xs text-gray-500")

        with ui.card().classes("w-full"):
            ui.label("Phase 1 - Company Registry").classes("text-lg font-semibold")
            company_input = ui.input("Tickers (space/comma separated)")
            company_input.props("clearable")

            company_table = ui.table(
                columns=[
                    {"name": "ticker", "label": "Ticker", "field": "ticker", "align": "left"}
                ],
                rows=[{"ticker": t} for t in list_companies()],
                row_key="ticker",
            ).classes("w-full")

            with ui.row().classes("gap-2"):
                ui.button(
                    "Add Tickers",
                    on_click=lambda: _add_companies(company_input.value or "", company_table),
                )

                ui.button(
                    "Delete Selected",
                    on_click=lambda: (
                        [delete_company(str(r["ticker"])) for r in (company_table.selected or [])],
                        _refresh_companies_table(company_table),
                    ),
                )

                ui.button(
                    "Sync Stocks from VNDirect",
                    on_click=lambda: _sync_stocks_now(stats),
                )

            company_table.props("selection=multiple")

        with ui.card().classes("w-full"):
            ui.label("Phase 1 - Financial Data Sync").classes("text-lg font-semibold")

            with ui.row().classes("gap-2"):
                start_year = ui.number("Start Year", value=DEFAULT_START_YEAR, step=1).classes("w-40")
                end_year = ui.number("End Year", value=DEFAULT_END_YEAR, step=1).classes("w-40")

            with ui.row().classes("gap-2"):
                ui.button("Sync Financial Models", on_click=lambda: _sync_models_now(stats))
                ui.button(
                    "Sync Financial Statements (All Companies)",
                    on_click=lambda: _sync_statements_now(
                        stats,
                        int(start_year.value or DEFAULT_START_YEAR),
                        int(end_year.value or DEFAULT_END_YEAR),
                    ),
                )
                ui.button(
                    "Sync Financial Ratios (All Companies)",
                    on_click=lambda: _sync_ratios_now(
                        stats,
                        int(start_year.value or DEFAULT_START_YEAR),
                        int(end_year.value or DEFAULT_END_YEAR),
                    ),
                )

        with ui.card().classes("w-full"):
            ui.label("Phase 1 - Vietstock Download Prototype").classes("text-lg font-semibold")
            ui.label(
                "Use this to queue document URLs and test robust download resolution for PDF/ZIP/RAR inputs."
            ).classes("text-sm text-gray-600")

            with ui.row().classes("w-full gap-2"):
                doc_id_input = ui.input("Document ID", value="1").classes("w-32")
                ticker_input = ui.input("Ticker", value="VNM").classes("w-32")
                doc_type_input = ui.input("Doc Type", value="2").classes("w-32")

            title_input = ui.input("Title", value="Bao cao thuong nien 2024").classes("w-full")
            url_input = ui.input("File URL").classes("w-full")
            use_llm = ui.checkbox("Use LLM for archive candidate selection", value=False)

            with ui.row().classes("gap-2"):
                ui.button(
                    "Sync Listings (Ticker)",
                    on_click=lambda: _sync_docs_one_now(
                        stats,
                        ticker=str(ticker_input.value or "VNM"),
                        doc_type=str(doc_type_input.value or "2"),
                    ),
                )
                ui.button(
                    "Sync Listings (All Companies)",
                    on_click=lambda: _sync_docs_all_now(
                        stats,
                        doc_type=str(doc_type_input.value or "2"),
                    ),
                )
                ui.button(
                    "Queue Stub Doc",
                    on_click=lambda: _queue_stub_document(
                        ticker=ticker_input.value or "",
                        doc_type=doc_type_input.value or "2",
                        title=title_input.value or "",
                        file_url=url_input.value or "",
                        doc_id=doc_id_input.value or "1",
                    ),
                )
                ui.button(
                    "Download Unsynced",
                    on_click=lambda: _download_unsynced_now(stats, bool(use_llm.value)),
                )


def main() -> None:
    bootstrap_directories()
    init_db()
    build_ui()
    ui.run(title="Annual Report Intelligence Platform", reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
