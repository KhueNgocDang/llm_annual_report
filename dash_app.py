from __future__ import annotations

from datetime import datetime
import re
import unicodedata

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html

from config import DEFAULT_END_YEAR, DEFAULT_START_YEAR, bootstrap_directories
from database import connection_scope, init_db
from financial_data import (
    sync_financial_models,
    sync_financial_ratios_all,
    sync_financial_statements_all,
    sync_stocks,
)
from report_loader import (
    load_annual_reports_from_markdown,
    load_financial_statement_reports_from_markdown,
)
from vietstock_documents import (
    download_all_unsynced,
    sync_documents_for_all_companies,
    sync_documents_for_ticker,
)


def _stats() -> dict[str, int]:
    with connection_scope() as con:
        return {
            "companies": _scalar_int(con, "SELECT COUNT(*) FROM companies"),
            "documents": _scalar_int(con, "SELECT COUNT(*) FROM vietstock_documents"),
            "annual_reports": _scalar_int(con, "SELECT COUNT(*) FROM annual_reports"),
            "financial_statement_reports": _scalar_int(con, "SELECT COUNT(*) FROM financial_statement_reports"),
        }


def _scalar_int(con, sql: str) -> int:
    row = con.execute(sql).fetchone()
    if not row:
        return 0
    return int(row[0] or 0)


def _parse_tickers(raw_text: str) -> list[str] | None:
    tickers = [token.strip().upper() for token in raw_text.replace(",", " ").split() if token.strip()]
    return tickers or None


def _parse_year(raw_text: str) -> int | None:
    text = raw_text.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _query_outputs(
    dataset: str,
    ticker_filter: str,
    year_filter: int | None,
    limit: int,
) -> list[dict[str, str]]:
    ticker = ticker_filter.strip().upper()

    if dataset == "all":
        sql = """
            SELECT dataset, ticker, year, source_file, content
            FROM (
                SELECT 'annual' AS dataset, ticker, year, source_file, content
                FROM annual_reports
                UNION ALL
                SELECT 'financial_statement' AS dataset, ticker, year, source_file, content
                FROM financial_statement_reports
            ) q
            WHERE (? = '' OR ticker = ?)
              AND (? IS NULL OR year = ?)
            ORDER BY year DESC, ticker
            LIMIT ?
        """
        params: list[object] = [ticker, ticker, year_filter, year_filter, limit]
    else:
        table = "annual_reports" if dataset == "annual" else "financial_statement_reports"
        sql = f"""
            SELECT ? AS dataset, ticker, year, source_file, content
            FROM {table}
            WHERE (? = '' OR ticker = ?)
              AND (? IS NULL OR year = ?)
            ORDER BY year DESC, ticker
            LIMIT ?
        """
        params = [dataset, ticker, ticker, year_filter, year_filter, limit]

    with connection_scope() as con:
        rows = con.execute(sql, params).fetchall()

    out: list[dict[str, str]] = []
    for row in rows:
        content = str(row[4] or "")
        out.append(
            {
                "dataset": str(row[0]),
                "ticker": str(row[1]),
                "year": str(row[2]),
                "source_file": str(row[3]),
                "preview": content[:280].replace("\n", " "),
                "full_content": content,
            }
        )
    return out


def _empty_fig(title: str):
    fig = px.scatter(title=title)
    fig.update_layout(
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": "No data",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 16},
            }
        ],
    )
    return fig


def _build_figures(rows: list[dict[str, str]]):
    if not rows:
        return _empty_fig("Outputs by Year"), _empty_fig("Top Tickers")

    df = pd.DataFrame(rows)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    if df.empty:
        return _empty_fig("Outputs by Year"), _empty_fig("Top Tickers")

    year_df = (
        df.groupby(["dataset", "year"], as_index=False)
        .size()
        .rename(columns={"size": "records"})
        .sort_values(["year", "dataset"])
    )
    fig_year = px.bar(
        year_df,
        x="year",
        y="records",
        color="dataset",
        barmode="group",
        title="Outputs by Year",
        color_discrete_map={"annual": "#1d4ed8", "financial_statement": "#0f766e"},
    )
    fig_year.update_layout(template="plotly_white", legend_title_text="Dataset")

    top_ticker_df = (
        df.groupby(["ticker", "dataset"], as_index=False)
        .size()
        .rename(columns={"size": "records"})
    )
    top_ticker_df = top_ticker_df.sort_values("records", ascending=False).head(20)
    fig_ticker = px.bar(
        top_ticker_df,
        x="records",
        y="ticker",
        color="dataset",
        orientation="h",
        title="Top Tickers by Output Count",
        color_discrete_map={"annual": "#1d4ed8", "financial_statement": "#0f766e"},
    )
    fig_ticker.update_layout(template="plotly_white", legend_title_text="Dataset")
    return fig_year, fig_ticker


def _query_financial_statement_series(
    ticker: str,
    keyword: str,
    start_year: int | None,
    end_year: int | None,
    limit: int,
) -> list[dict[str, object]]:
    code = ticker.strip().upper()
    if not code:
        return []

    keyword_terms = _keyword_terms(keyword)
    sql = """
        WITH model_names AS (
            SELECT
                item_code,
                MAX(NULLIF(item_vn_name, '')) AS item_vn_name,
                MAX(NULLIF(item_en_name, '')) AS item_en_name
            FROM financial_models
            GROUP BY item_code
        )
        SELECT
            fs.code AS ticker,
            fs.item_code,
            COALESCE(mn.item_en_name, mn.item_vn_name, fs.item_code) AS item_name,
            TRY_CAST(SUBSTR(fs.fiscal_date, 1, 4) AS INTEGER) AS year,
            fs.fiscal_date,
            fs.numeric_value
        FROM financial_statements fs
        LEFT JOIN model_names mn ON mn.item_code = fs.item_code
        WHERE fs.code = ?
          AND fs.numeric_value IS NOT NULL
          AND TRY_CAST(SUBSTR(fs.fiscal_date, 1, 4) AS INTEGER) IS NOT NULL
          AND (? IS NULL OR TRY_CAST(SUBSTR(fs.fiscal_date, 1, 4) AS INTEGER) >= ?)
          AND (? IS NULL OR TRY_CAST(SUBSTR(fs.fiscal_date, 1, 4) AS INTEGER) <= ?)
        ORDER BY year, fs.item_code
        LIMIT ?
    """
    params: list[object] = [
        code,
        start_year,
        start_year,
        end_year,
        end_year,
        limit,
    ]
    with connection_scope() as con:
        rows = con.execute(sql, params).fetchall()

    out: list[dict[str, object]] = []
    for row in rows:
        item_code = _format_item_code(row[1])
        item_name = str(row[2] or item_code)
        searchable_text = _normalize_text(f"{item_code} {item_name}")
        if keyword_terms and not any(term in searchable_text for term in keyword_terms):
            continue
        out.append(
            {
                "ticker": str(row[0]),
                "item_code": item_code,
                "item_name": item_name,
                "item_label": f"{item_code} | {item_name}",
                "year": int(row[3]),
                "fiscal_date": str(row[4] or ""),
                "value": float(row[5]),
            }
        )
    return out


def _normalize_text(value: str) -> str:
    lowered = (value or "").strip().lower()
    decomposed = unicodedata.normalize("NFKD", lowered)
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", without_marks)


def _keyword_terms(raw_keyword: str) -> list[str]:
    normalized = _normalize_text(raw_keyword)
    if not normalized:
        return []

    terms = {normalized}
    alias_rules = {
        "tai san": ["asset", "assets"],
        "tong tai san": ["total asset", "asset"],
        "no phai tra": ["liabil", "liability"],
        "von chu so huu": ["equity", "owner"],
        "doanh thu": ["revenue", "sales"],
        "loi nhuan": ["profit", "income"],
        "tien": ["cash"],
    }
    for pattern, extras in alias_rules.items():
        if pattern in normalized:
            terms.update(extras)
    return sorted(terms)


def _format_item_code(raw_item_code: object) -> str:
    text = str(raw_item_code or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _empty_fs_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=["ticker", "item_code", "item_name", "item_label", "year", "fiscal_date", "value"])


def _build_financial_statement_figures(rows: list[dict[str, object]], selected_item_codes: list[str]):
    if not rows:
        return _empty_fig("Item Trends"), _empty_fig("Latest Snapshot")

    df = pd.DataFrame(rows)
    if df.empty:
        return _empty_fig("Item Trends"), _empty_fig("Latest Snapshot")

    filtered = df if not selected_item_codes else df[df["item_code"].isin(selected_item_codes)]
    if filtered.empty:
        return _empty_fig("Item Trends"), _empty_fig("Latest Snapshot")

    fig_trend = px.line(
        filtered,
        x="year",
        y="value",
        color="item_label",
        markers=True,
        title="Financial Statement Item Trends",
    )
    fig_trend.update_layout(template="plotly_white", legend_title_text="Item")

    latest_df = (
        filtered.sort_values(["item_code", "year"]).groupby("item_code", as_index=False).tail(1)
    )
    latest_df = latest_df.sort_values("value", ascending=False).head(25)
    fig_latest = px.bar(
        latest_df,
        x="value",
        y="item_label",
        color="item_label",
        orientation="h",
        title="Latest Value Snapshot",
    )
    fig_latest.update_layout(template="plotly_white", showlegend=False)
    return fig_trend, fig_latest


def _build_financial_statement_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    latest = df.sort_values(["item_code", "year"]).groupby("item_code", as_index=False).tail(1)
    latest = latest.sort_values("value", ascending=False)
    return [
        {
            "item_code": str(r["item_code"]),
            "item_name": str(r["item_name"]),
            "year": int(r["year"]),
            "latest_value": float(r["value"]),
        }
        for _, r in latest.iterrows()
    ]


def _card(title: str, value_id: str) -> html.Div:
    return html.Div(
        [
            html.Div(title, style={"fontSize": "13px", "color": "#475569"}),
            html.Div(id=value_id, style={"fontSize": "26px", "fontWeight": 700, "color": "#0f172a"}),
        ],
        style={
            "padding": "14px 16px",
            "border": "1px solid #dbe4ee",
            "borderRadius": "12px",
            "background": "#f8fbff",
            "minWidth": "180px",
        },
    )


bootstrap_directories()
init_db()
initial_stats = _stats()

app = Dash(__name__)
app.title = "Annual Report Dashboard"

app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("Annual Report Intelligence Dashboard", style={"margin": "0"}),
                html.Div(
                    f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    style={"color": "#64748b", "fontSize": "13px"},
                ),
            ],
            style={"marginBottom": "14px"},
        ),
        html.Div(
            [
                _card("Companies", "kpi-companies"),
                _card("Documents", "kpi-documents"),
                _card("Annual Reports", "kpi-annual"),
                _card("Financial Statements", "kpi-financial"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "16px"},
        ),
        dcc.Tabs(
            value="tab-tasks",
            children=[
                dcc.Tab(
                    label="Task Runner",
                    value="tab-tasks",
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H3("Pipeline Actions"),
                                        html.Div(
                                            [
                                                html.Button("Bootstrap Directories + DB", id="task-bootstrap", n_clicks=0),
                                                html.Button("Sync Stocks", id="task-sync-stocks", n_clicks=0),
                                                html.Button("Sync Financial Models", id="task-sync-models", n_clicks=0),
                                            ],
                                            style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginBottom": "10px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Div("Year Range", style={"fontWeight": 600}),
                                                html.Div(
                                                    [
                                                        dcc.Input(
                                                            id="task-start-year",
                                                            type="number",
                                                            value=DEFAULT_START_YEAR,
                                                            style={"width": "130px"},
                                                        ),
                                                        dcc.Input(
                                                            id="task-end-year",
                                                            type="number",
                                                            value=DEFAULT_END_YEAR,
                                                            style={"width": "130px"},
                                                        ),
                                                        html.Button(
                                                            "Sync Financial Statements",
                                                            id="task-sync-statements",
                                                            n_clicks=0,
                                                        ),
                                                        html.Button(
                                                            "Sync Financial Ratios",
                                                            id="task-sync-ratios",
                                                            n_clicks=0,
                                                        ),
                                                    ],
                                                    style={"display": "flex", "gap": "8px", "alignItems": "center", "flexWrap": "wrap"},
                                                ),
                                            ],
                                            style={"marginBottom": "12px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Div("Vietstock Listing + Download", style={"fontWeight": 600}),
                                                html.Div(
                                                    [
                                                        dcc.Input(id="task-doc-ticker", type="text", value="VNM", placeholder="Ticker", style={"width": "110px"}),
                                                        dcc.Input(id="task-doc-type", type="text", value="2", placeholder="Doc Type", style={"width": "90px"}),
                                                        html.Button("Sync Listings (Ticker)", id="task-sync-docs-one", n_clicks=0),
                                                        html.Button("Sync Listings (All)", id="task-sync-docs-all", n_clicks=0),
                                                        dcc.Checklist(
                                                            id="task-use-llm",
                                                            options=[{"label": "Use LLM archive selector", "value": "llm"}],
                                                            value=[],
                                                            style={"minWidth": "210px"},
                                                        ),
                                                        html.Button("Download Unsynced", id="task-download-unsynced", n_clicks=0),
                                                    ],
                                                    style={"display": "flex", "gap": "8px", "alignItems": "center", "flexWrap": "wrap"},
                                                ),
                                            ],
                                            style={"marginBottom": "12px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Div("Markdown Loading", style={"fontWeight": 600}),
                                                html.Div(
                                                    [
                                                        dcc.Input(
                                                            id="task-loader-tickers",
                                                            type="text",
                                                            placeholder="Tickers filter (optional)",
                                                            style={"width": "240px"},
                                                        ),
                                                        html.Button("Load Annual Markdown", id="task-load-annual", n_clicks=0),
                                                        html.Button(
                                                            "Load Financial Statement Markdown",
                                                            id="task-load-financial",
                                                            n_clicks=0,
                                                        ),
                                                    ],
                                                    style={"display": "flex", "gap": "8px", "alignItems": "center", "flexWrap": "wrap"},
                                                ),
                                            ]
                                        ),
                                    ],
                                    style={
                                        "padding": "14px",
                                        "border": "1px solid #dbe4ee",
                                        "borderRadius": "12px",
                                        "background": "#ffffff",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.H3("Task Log"),
                                        html.Pre(
                                            id="task-log",
                                            children="Ready.",
                                            style={
                                                "height": "260px",
                                                "overflow": "auto",
                                                "background": "#0f172a",
                                                "color": "#e2e8f0",
                                                "padding": "12px",
                                                "borderRadius": "10px",
                                                "fontSize": "12px",
                                            },
                                        ),
                                    ],
                                    style={
                                        "padding": "14px",
                                        "border": "1px solid #dbe4ee",
                                        "borderRadius": "12px",
                                        "background": "#ffffff",
                                        "marginTop": "12px",
                                    },
                                ),
                            ],
                            style={"padding": "12px 4px"},
                        )
                    ],
                ),
                dcc.Tab(
                    label="Output Explorer",
                    value="tab-output",
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dcc.Dropdown(
                                            id="output-dataset",
                                            options=[
                                                {"label": "All", "value": "all"},
                                                {"label": "Annual", "value": "annual"},
                                                {"label": "Financial Statement", "value": "financial_statement"},
                                            ],
                                            value="all",
                                            style={"width": "220px"},
                                        ),
                                        dcc.Input(
                                            id="output-ticker",
                                            type="text",
                                            placeholder="Ticker (optional)",
                                            style={"width": "170px"},
                                        ),
                                        dcc.Input(
                                            id="output-year",
                                            type="text",
                                            placeholder="Year (optional)",
                                            style={"width": "150px"},
                                        ),
                                        dcc.Input(
                                            id="output-limit",
                                            type="number",
                                            value=100,
                                            min=1,
                                            max=2000,
                                            style={"width": "110px"},
                                        ),
                                        html.Button("Refresh", id="output-refresh", n_clicks=0),
                                    ],
                                    style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginBottom": "10px"},
                                ),
                                html.Div(id="output-summary", style={"color": "#334155", "marginBottom": "8px"}),
                                dash_table.DataTable(
                                    id="output-table",
                                    columns=[
                                        {"name": "dataset", "id": "dataset"},
                                        {"name": "ticker", "id": "ticker"},
                                        {"name": "year", "id": "year"},
                                        {"name": "source_file", "id": "source_file"},
                                        {"name": "preview", "id": "preview"},
                                        {"name": "full_content", "id": "full_content"},
                                    ],
                                    hidden_columns=["full_content"],
                                    data=[],
                                    page_size=12,
                                    row_selectable="single",
                                    style_table={"overflowX": "auto"},
                                    style_cell={"textAlign": "left", "maxWidth": "520px", "whiteSpace": "normal"},
                                    style_header={"backgroundColor": "#e2e8f0", "fontWeight": "bold"},
                                ),
                                html.Div(
                                    [
                                        dcc.Graph(id="output-by-year", style={"height": "360px"}),
                                        dcc.Graph(id="output-top-tickers", style={"height": "360px"}),
                                    ],
                                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px"},
                                ),
                                html.H4("Selected Content"),
                                dcc.Textarea(
                                    id="output-preview",
                                    readOnly=True,
                                    value="",
                                    style={"width": "100%", "height": "260px", "fontFamily": "monospace"},
                                ),
                            ],
                            style={"padding": "12px 4px"},
                        )
                    ],
                ),
                dcc.Tab(
                    label="Financial Statement Explorer",
                    value="tab-financial",
                    children=[
                        html.Div(
                            [
                                dcc.Store(id="fs-rows-store", data=[]),
                                html.Div(
                                    [
                                        dcc.Input(
                                            id="fs-ticker",
                                            type="text",
                                            value="VNM",
                                            placeholder="Ticker",
                                            style={"width": "120px"},
                                        ),
                                        dcc.Input(
                                            id="fs-keyword",
                                            type="text",
                                            value="tong tai san",
                                            placeholder="Keyword (e.g. total asset)",
                                            style={"width": "260px"},
                                        ),
                                        dcc.Input(
                                            id="fs-start-year",
                                            type="number",
                                            value=DEFAULT_START_YEAR,
                                            style={"width": "120px"},
                                        ),
                                        dcc.Input(
                                            id="fs-end-year",
                                            type="number",
                                            value=DEFAULT_END_YEAR,
                                            style={"width": "120px"},
                                        ),
                                        dcc.Input(
                                            id="fs-limit",
                                            type="number",
                                            value=3000,
                                            min=100,
                                            max=50000,
                                            style={"width": "120px"},
                                        ),
                                        html.Button("Load Items", id="fs-load", n_clicks=0),
                                    ],
                                    style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginBottom": "10px"},
                                ),
                                html.Div(id="fs-summary", style={"color": "#334155", "marginBottom": "8px"}),
                                dcc.Dropdown(
                                    id="fs-item-select",
                                    options=[],
                                    value=[],
                                    multi=True,
                                    placeholder="Select one or more item codes to chart",
                                    style={"marginBottom": "10px"},
                                ),
                                html.Div(
                                    [
                                        dcc.Graph(id="fs-trend-chart", style={"height": "380px"}),
                                        dcc.Graph(id="fs-latest-chart", style={"height": "380px"}),
                                    ],
                                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px"},
                                ),
                                dash_table.DataTable(
                                    id="fs-summary-table",
                                    columns=[
                                        {"name": "item_code", "id": "item_code"},
                                        {"name": "item_name", "id": "item_name"},
                                        {"name": "year", "id": "year"},
                                        {"name": "latest_value", "id": "latest_value", "type": "numeric", "format": {"specifier": ",.2f"}},
                                    ],
                                    data=[],
                                    page_size=12,
                                    sort_action="native",
                                    filter_action="native",
                                    style_table={"overflowX": "auto"},
                                    style_cell={"textAlign": "left", "maxWidth": "420px", "whiteSpace": "normal"},
                                    style_header={"backgroundColor": "#e2e8f0", "fontWeight": "bold"},
                                ),
                            ],
                            style={"padding": "12px 4px"},
                        )
                    ],
                ),
            ],
        ),
    ],
    style={
        "maxWidth": "1300px",
        "margin": "0 auto",
        "padding": "20px",
        "fontFamily": "Verdana, 'DejaVu Sans', sans-serif",
        "background": "linear-gradient(180deg, #f8fbff 0%, #eef4fb 100%)",
        "minHeight": "100vh",
    },
)


@app.callback(
    Output("task-log", "children"),
    Output("kpi-companies", "children"),
    Output("kpi-documents", "children"),
    Output("kpi-annual", "children"),
    Output("kpi-financial", "children"),
    Input("task-bootstrap", "n_clicks"),
    Input("task-sync-stocks", "n_clicks"),
    Input("task-sync-models", "n_clicks"),
    Input("task-sync-statements", "n_clicks"),
    Input("task-sync-ratios", "n_clicks"),
    Input("task-sync-docs-one", "n_clicks"),
    Input("task-sync-docs-all", "n_clicks"),
    Input("task-download-unsynced", "n_clicks"),
    Input("task-load-annual", "n_clicks"),
    Input("task-load-financial", "n_clicks"),
    State("task-start-year", "value"),
    State("task-end-year", "value"),
    State("task-doc-ticker", "value"),
    State("task-doc-type", "value"),
    State("task-use-llm", "value"),
    State("task-loader-tickers", "value"),
    prevent_initial_call=True,
)
def run_task(
    _bootstrap_clicks: int,
    _stocks_clicks: int,
    _models_clicks: int,
    _statements_clicks: int,
    _ratios_clicks: int,
    _docs_one_clicks: int,
    _docs_all_clicks: int,
    _download_clicks: int,
    _load_annual_clicks: int,
    _load_financial_clicks: int,
    start_year: int,
    end_year: int,
    doc_ticker: str,
    doc_type: str,
    llm_flags: list[str],
    loader_tickers: str,
):
    trigger = ctx.triggered_id
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        if start_year is None:
            start_year = DEFAULT_START_YEAR
        if end_year is None:
            end_year = DEFAULT_END_YEAR

        if trigger == "task-bootstrap":
            bootstrap_directories()
            init_db()
            message = "Bootstrap completed"
        elif trigger == "task-sync-stocks":
            count = sync_stocks()
            message = f"Synced stocks: {count}"
        elif trigger == "task-sync-models":
            count = sync_financial_models()
            message = f"Synced financial models: {count}"
        elif trigger == "task-sync-statements":
            result = sync_financial_statements_all(start_year=start_year, end_year=end_year)
            message = (
                "Financial statements sync done: "
                f"rows={result['rows']} ok={result['tickers_done']} failed={result['tickers_failed']}"
            )
        elif trigger == "task-sync-ratios":
            result = sync_financial_ratios_all(start_year=start_year, end_year=end_year)
            message = (
                "Financial ratios sync done: "
                f"rows={result['rows']} ok={result['tickers_done']} failed={result['tickers_failed']}"
            )
        elif trigger == "task-sync-docs-one":
            ticker = (doc_ticker or "VNM").strip().upper()
            dtype = (doc_type or "2").strip()
            count = sync_documents_for_ticker(ticker=ticker, doc_type=dtype)
            message = f"Synced listings for {ticker}: {count}"
        elif trigger == "task-sync-docs-all":
            dtype = (doc_type or "2").strip()
            result = sync_documents_for_all_companies(doc_type=dtype)
            message = (
                "Synced listings for all companies: "
                f"rows={result['rows']} ok={result['tickers_done']} failed={result['tickers_failed']}"
            )
        elif trigger == "task-download-unsynced":
            use_llm = "llm" in (llm_flags or [])
            result = download_all_unsynced(limit=200, use_llm_selection=use_llm)
            message = f"Downloaded unsynced: done={result['done']} failed={result['failed']} total={result['total']}"
        elif trigger == "task-load-annual":
            result = load_annual_reports_from_markdown(
                tickers=_parse_tickers(loader_tickers or ""),
                start_year=int(start_year),
                end_year=int(end_year),
            )
            message = (
                "Loaded annual markdown: "
                f"loaded={result['loaded']} failed={result['failed']} pairs={result['pairs']}"
            )
        elif trigger == "task-load-financial":
            result = load_financial_statement_reports_from_markdown(
                tickers=_parse_tickers(loader_tickers or ""),
                start_year=int(start_year),
                end_year=int(end_year),
            )
            message = (
                "Loaded financial statement markdown: "
                f"loaded={result['loaded']} failed={result['failed']} pairs={result['pairs']}"
            )
        else:
            message = "No task selected"

        line = f"[{now}] OK: {message}"
    except Exception as exc:
        line = f"[{now}] ERROR ({trigger}): {exc}"

    stats = _stats()
    return (
        line,
        str(stats["companies"]),
        str(stats["documents"]),
        str(stats["annual_reports"]),
        str(stats["financial_statement_reports"]),
    )


@app.callback(
    Output("output-table", "data"),
    Output("output-summary", "children"),
    Output("output-by-year", "figure"),
    Output("output-top-tickers", "figure"),
    Input("output-refresh", "n_clicks"),
    State("output-dataset", "value"),
    State("output-ticker", "value"),
    State("output-year", "value"),
    State("output-limit", "value"),
)
def refresh_outputs(
    _refresh_clicks: int,
    dataset: str,
    ticker: str,
    year_text: str,
    limit_value: int,
):
    year_filter = _parse_year(year_text or "")
    if (year_text or "").strip() and year_filter is None:
        return [], "Year filter must be a valid integer.", _empty_fig("Outputs by Year"), _empty_fig("Top Tickers")

    safe_limit = 100 if limit_value is None else max(1, min(int(limit_value), 2000))
    rows = _query_outputs(
        dataset=(dataset or "all"),
        ticker_filter=(ticker or ""),
        year_filter=year_filter,
        limit=safe_limit,
    )
    fig_year, fig_ticker = _build_figures(rows)
    summary = f"Rows: {len(rows)} | dataset={dataset or 'all'} | ticker={(ticker or '').upper() or '*'}"
    return rows, summary, fig_year, fig_ticker


@app.callback(
    Output("output-preview", "value"),
    Input("output-table", "selected_rows"),
    State("output-table", "data"),
)
def preview_output(selected_rows: list[int] | None, rows: list[dict[str, str]] | None):
    if not selected_rows or not rows:
        return ""
    index = selected_rows[0]
    if index < 0 or index >= len(rows):
        return ""
    return str(rows[index].get("full_content") or "")


@app.callback(
    Output("kpi-companies", "children", allow_duplicate=True),
    Output("kpi-documents", "children", allow_duplicate=True),
    Output("kpi-annual", "children", allow_duplicate=True),
    Output("kpi-financial", "children", allow_duplicate=True),
    Input("output-refresh", "n_clicks"),
    prevent_initial_call=True,
)
def refresh_kpis(_refresh_clicks: int):
    stats = _stats()
    return (
        str(stats["companies"]),
        str(stats["documents"]),
        str(stats["annual_reports"]),
        str(stats["financial_statement_reports"]),
    )


@app.callback(
    Output("fs-rows-store", "data"),
    Output("fs-item-select", "options"),
    Output("fs-item-select", "value"),
    Output("fs-summary", "children"),
    Output("fs-summary-table", "data"),
    Output("fs-trend-chart", "figure"),
    Output("fs-latest-chart", "figure"),
    Input("fs-load", "n_clicks"),
    State("fs-ticker", "value"),
    State("fs-keyword", "value"),
    State("fs-start-year", "value"),
    State("fs-end-year", "value"),
    State("fs-limit", "value"),
    prevent_initial_call=True,
)
def load_financial_statement_items(
    _clicks: int,
    ticker: str,
    keyword: str,
    start_year: int,
    end_year: int,
    limit_value: int,
):
    safe_ticker = (ticker or "").strip().upper()
    if not safe_ticker:
        return [], [], [], "Ticker is required.", [], _empty_fig("Item Trends"), _empty_fig("Latest Snapshot")

    if start_year is not None and end_year is not None and int(start_year) > int(end_year):
        return [], [], [], "Invalid year range.", [], _empty_fig("Item Trends"), _empty_fig("Latest Snapshot")

    safe_limit = 3000 if limit_value is None else max(100, min(int(limit_value), 50000))
    rows = _query_financial_statement_series(
        ticker=safe_ticker,
        keyword=keyword or "",
        start_year=int(start_year) if start_year is not None else None,
        end_year=int(end_year) if end_year is not None else None,
        limit=safe_limit,
    )

    if not rows:
        summary = f"No financial statement rows found for ticker={safe_ticker}."
        return [], [], [], summary, [], _empty_fig("Item Trends"), _empty_fig("Latest Snapshot")

    options = []
    seen_codes: set[str] = set()
    for row in rows:
        code = str(row["item_code"])
        if code in seen_codes:
            continue
        seen_codes.add(code)
        options.append({"label": str(row["item_label"]), "value": code})

    selected_codes = [opt["value"] for opt in options[: min(8, len(options))]]
    fig_trend, fig_latest = _build_financial_statement_figures(rows, selected_codes)
    summary_table = _build_financial_statement_summary(rows)
    summary = (
        f"Ticker={safe_ticker} | rows={len(rows)} | items={len(options)} "
        f"| keyword={keyword or '*'}"
    )
    return rows, options, selected_codes, summary, summary_table, fig_trend, fig_latest


@app.callback(
    Output("fs-trend-chart", "figure", allow_duplicate=True),
    Output("fs-latest-chart", "figure", allow_duplicate=True),
    Input("fs-item-select", "value"),
    State("fs-rows-store", "data"),
    prevent_initial_call=True,
)
def update_financial_statement_charts(selected_item_codes: list[str], rows: list[dict[str, object]]):
    fig_trend, fig_latest = _build_financial_statement_figures(rows or [], selected_item_codes or [])
    return fig_trend, fig_latest


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
