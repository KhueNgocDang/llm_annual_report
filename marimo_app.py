from __future__ import annotations

from datetime import datetime
import re

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")

CHAT_HISTORY: list[dict[str, str]] = []


def _safe_select_sql(raw_sql: str) -> tuple[bool, str]:
    text = (raw_sql or "").strip()
    if not text:
        return False, "Empty SQL"

    lowered = text.lower()
    if ";" in text[:-1]:
        return False, "Only one SQL statement is allowed"

    if not (lowered.startswith("select") or lowered.startswith("with")):
        return False, "Only SELECT/WITH queries are allowed"

    forbidden = re.compile(
        r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|copy|call|pragma)\b",
        re.IGNORECASE,
    )
    if forbidden.search(text):
        return False, "Potentially destructive SQL keyword detected"

    return True, text.rstrip(";")


def _ensure_limit(sql: str, limit: int) -> str:
    if re.search(r"\blimit\b", sql, flags=re.IGNORECASE):
        return sql
    return f"{sql.rstrip()}\nLIMIT {int(limit)}"


def _schema_summary() -> tuple[str, list[tuple[str, str, str]]]:
    from database import connection_scope

    with connection_scope() as con:
        rows = con.execute(
            """
            SELECT table_schema, table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = 'main'
              AND table_name IN (
                'companies',
                'annual_reports',
                'financial_statement_reports',
                'financial_statements',
                'financial_models',
                'vietstock_documents'
              )
            ORDER BY table_name, ordinal_position
            """
        ).fetchall()

    by_table: dict[str, list[str]] = {}
    for _schema, table, column in rows:
        by_table.setdefault(str(table), []).append(str(column))

    parts = []
    for table, cols in by_table.items():
        parts.append(f"{table}({', '.join(cols)})")

    return "\n".join(parts), [(str(r[0]), str(r[1]), str(r[2])) for r in rows]


def _extract_ticker(question: str, ticker_hint: str) -> str:
    hint = (ticker_hint or "").strip().upper()
    if hint:
        return hint
    candidates = re.findall(r"\b[A-Z]{3}\b", question.upper())
    if candidates:
        return candidates[0]
    return ""


def _extract_item_code(question: str) -> str:
    m = re.search(r"\b(\d{1,4}(?:\.\d{1,2})?)\b", question)
    return m.group(1) if m else ""


def _extract_sql_from_text(text: str) -> str:
    if not text.strip():
        return ""

    fenced = re.search(r"```sql\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    generic_fence = re.search(r"```\s*(.*?)```", text, flags=re.DOTALL)
    if generic_fence:
        return generic_fence.group(1).strip()

    return text.strip()


def _llm_sql_fallback(
    question: str,
    schema_text: str,
    ticker_hint: str,
    start_year: int,
    end_year: int,
    row_limit: int,
) -> tuple[bool, str, str]:
    import os

    from config import INFERENCE_MODEL, ensure_env_loaded

    ensure_env_loaded()
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return False, "", "OPENAI_API_KEY is missing in .env"

    try:
        from openai import OpenAI
    except Exception as exc:
        return False, "", f"OpenAI SDK unavailable: {exc}"

    system_prompt = (
        "You generate one DuckDB SQL query only. "
        "Constraints: SELECT/WITH only; no DDL/DML; no PRAGMA/ATTACH/DETACH/COPY/CALL; one statement. "
        "Prefer placeholders only when matching this style: (? = '' OR ticker = ?) and year BETWEEN ? AND ?. "
        "If using financial_statements years, use TRY_CAST(SUBSTR(fiscal_date, 1, 4) AS INTEGER). "
        "Return SQL only, no explanation."
    )

    user_prompt = f"""
Schema:
{schema_text}

User question: {question}
Ticker hint: {ticker_hint or '(none)'}
Start year: {start_year}
End year: {end_year}
Row limit: {row_limit}
""".strip()

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=INFERENCE_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as exc:
        return False, "", f"OpenAI request failed: {exc}"

    content = ""
    choices = getattr(response, "choices", None)
    if choices:
        msg = getattr(choices[0], "message", None)
        if msg is not None:
            content = str(getattr(msg, "content", "") or "")

    candidate_sql = _extract_sql_from_text(content)
    ok, status = _safe_select_sql(candidate_sql)
    if not ok:
        return False, "", f"LLM SQL blocked: {status}"

    final_sql = _ensure_limit(status, row_limit)
    return True, final_sql, "LLM-generated SQL"


def _suggest_sql(
    question: str,
    ticker_hint: str,
    start_year: int,
    end_year: int,
    row_limit: int,
) -> tuple[str, str, bool]:
    q = (question or "").strip()
    ql = q.lower()
    ticker = _extract_ticker(q, ticker_hint)
    item_code = _extract_item_code(q)

    if any(k in ql for k in ("coverage", "bao phu", "coverage by year", "do phu")) and "year" in ql:
        sql = """
WITH annual AS (
  SELECT year, COUNT(*) AS annual_reports
  FROM annual_reports
  GROUP BY year
), fs_reports AS (
  SELECT year, COUNT(*) AS financial_statement_reports
  FROM financial_statement_reports
  GROUP BY year
), fs_rows AS (
  SELECT TRY_CAST(SUBSTR(fiscal_date, 1, 4) AS INTEGER) AS year, COUNT(*) AS financial_statement_rows
  FROM financial_statements
  WHERE TRY_CAST(SUBSTR(fiscal_date, 1, 4) AS INTEGER) IS NOT NULL
  GROUP BY year
)
SELECT
  COALESCE(a.year, fr.year, fs.year) AS year,
  COALESCE(a.annual_reports, 0) AS annual_reports,
  COALESCE(fr.financial_statement_reports, 0) AS financial_statement_reports,
  COALESCE(fs.financial_statement_rows, 0) AS financial_statement_rows
FROM annual a
FULL OUTER JOIN fs_reports fr ON fr.year = a.year
FULL OUTER JOIN fs_rows fs ON fs.year = COALESCE(a.year, fr.year)
WHERE COALESCE(a.year, fr.year, fs.year) BETWEEN ? AND ?
ORDER BY year
"""
        return (
            sql,
            "Coverage by year across annual reports, financial statement reports, and financial statement rows.",
            True,
        )

    if "top" in ql and "ticker" in ql and any(k in ql for k in ("financial", "statement", "bctc")):
        sql = """
SELECT
  code AS ticker,
  COUNT(*) AS rows_count
FROM financial_statements
WHERE TRY_CAST(SUBSTR(fiscal_date, 1, 4) AS INTEGER) BETWEEN ? AND ?
GROUP BY code
ORDER BY rows_count DESC
"""
        return sql, "Top tickers by financial statement row count in selected years.", True

    if "top" in ql and "ticker" in ql:
        sql = """
SELECT ticker, SUM(records) AS total_records
FROM (
  SELECT ticker, COUNT(*) AS records
  FROM annual_reports
  WHERE year BETWEEN ? AND ?
  GROUP BY ticker
  UNION ALL
  SELECT ticker, COUNT(*) AS records
  FROM financial_statement_reports
  WHERE year BETWEEN ? AND ?
  GROUP BY ticker
) t
GROUP BY ticker
ORDER BY total_records DESC
"""
        return sql, "Top tickers by total report records (annual + financial statement reports).", True

    if any(k in ql for k in ("trend", "timeseries", "time series", "xu huong")) and (ticker or item_code):
        sql = """
SELECT
  code AS ticker,
  TRY_CAST(SUBSTR(fiscal_date, 1, 4) AS INTEGER) AS year,
  item_code,
  AVG(numeric_value) AS avg_value
FROM financial_statements
WHERE (? = '' OR code = ?)
  AND (? = '' OR CAST(item_code AS VARCHAR) = ?)
  AND TRY_CAST(SUBSTR(fiscal_date, 1, 4) AS INTEGER) BETWEEN ? AND ?
  AND numeric_value IS NOT NULL
GROUP BY code, year, item_code
ORDER BY year
"""
        return (
            sql,
            "Trend for a ticker/item_code from financial_statements (yearly average numeric_value).",
            True,
        )

    if any(k in ql for k in ("annual", "annual reports", "bao cao thuong nien")):
        sql = """
SELECT ticker, year, source_file
FROM annual_reports
WHERE (? = '' OR ticker = ?)
  AND year BETWEEN ? AND ?
ORDER BY year DESC, ticker
"""
        return sql, "List annual report records with ticker/year filters.", True

    if any(k in ql for k in ("financial statement report", "markdown bctc", "financial_statement_reports")):
        sql = """
SELECT ticker, year, source_file
FROM financial_statement_reports
WHERE (? = '' OR ticker = ?)
  AND year BETWEEN ? AND ?
ORDER BY year DESC, ticker
"""
        return sql, "List financial statement report records with ticker/year filters.", True

    sql = """
SELECT dataset, ticker, year, source_file
FROM (
  SELECT 'annual' AS dataset, ticker, year, source_file FROM annual_reports
  UNION ALL
  SELECT 'financial_statement' AS dataset, ticker, year, source_file FROM financial_statement_reports
) q
WHERE (? = '' OR ticker = ?)
  AND year BETWEEN ? AND ?
ORDER BY year DESC, ticker
"""
    return sql, "Default output explorer query (annual + financial statement reports).", False


def _run_sql_with_params(
    sql: str,
    question: str,
    ticker_hint: str,
    start_year: int,
    end_year: int,
    row_limit: int,
):
    from database import connection_scope

    ticker = _extract_ticker(question, ticker_hint)
    item_code = _extract_item_code(question)

    sql_limited = _ensure_limit(sql, row_limit)
    ok, safe_sql = _safe_select_sql(sql_limited)
    if not ok:
        return False, safe_sql, None

    params: list[object]
    lowered = safe_sql.lower()
    if "(? = '' or ticker = ?)" in lowered and "financial_statements" not in lowered:
        params = [ticker, ticker, start_year, end_year]
    elif "(? = '' or code = ?)" in lowered and "item_code" not in lowered:
        params = [ticker, ticker, start_year, end_year]
    elif "(? = '' or code = ?)" in lowered and "item_code" in lowered:
        params = [ticker, ticker, item_code, item_code, start_year, end_year]
    elif "year between ? and ?" in lowered and "union all" in lowered and "total_records" in lowered:
        params = [start_year, end_year, start_year, end_year]
    elif "between ? and ?" in lowered:
        params = [start_year, end_year]
    else:
        params = []

    try:
        with connection_scope() as con:
            df = con.execute(safe_sql, params).fetchdf()
        return True, safe_sql, df
    except Exception as exc:
        return False, f"SQL error: {exc}", None


def _chart_plan(df) -> tuple[str, str, str]:
    if df is None or df.empty:
        return "none", "", ""

    numeric_cols = [c for c in df.columns if str(df[c].dtype).startswith(("int", "float"))]
    if not numeric_cols:
        return "none", "", ""

    year_like = [c for c in df.columns if c.lower() in {"year", "fiscal_year"}]
    if year_like:
        x = year_like[0]
        y_candidates = [c for c in numeric_cols if c != x]
        if y_candidates:
            return "line", x, y_candidates[0]

    cat_cols = [c for c in df.columns if c not in numeric_cols]
    if cat_cols:
        return "bar", cat_cols[0], numeric_cols[0]

    return "none", "", ""


@app.cell
def _():
    import marimo as mo
    from config import DB_PATH, bootstrap_directories
    from database import init_db

    bootstrap_directories()
    init_db()

    mo.md(
        f"""
# DuckDB Chat Analyst (marimo)

Database: `{DB_PATH}`

Type a question, review the suggested SQL, optionally edit it, then run.
"""
    )
    return mo


@app.cell
def _():
    from config import DEFAULT_END_YEAR, DEFAULT_START_YEAR

    question = "coverage by year"
    ticker_hint = ""
    start_year = DEFAULT_START_YEAR
    end_year = DEFAULT_END_YEAR
    row_limit = 300
    enable_llm_fallback = False
    force_llm = False

    # Optional manual override: if not empty, this SQL is executed instead of the suggested SQL.
    sql_override = ""

    return (
        question,
        ticker_hint,
        start_year,
        end_year,
        row_limit,
        enable_llm_fallback,
        force_llm,
        sql_override,
    )


@app.cell
def _(mo):
    schema_text, schema_rows = _schema_summary()
    mo.md("## Schema Context")
    mo.md("```\n" + schema_text + "\n```")
    return schema_rows, schema_text


@app.cell
def _(
    enable_llm_fallback,
    end_year,
    force_llm,
    question,
    row_limit,
    schema_text,
    sql_override,
    start_year,
    ticker_hint,
):
    suggested_sql, explanation, matched_template = _suggest_sql(
        question=question,
        ticker_hint=ticker_hint,
        start_year=int(start_year),
        end_year=int(end_year),
        row_limit=int(row_limit),
    )

    used_llm = False
    llm_status = ""
    llm_sql = ""
    should_try_llm = bool(enable_llm_fallback) and (bool(force_llm) or not matched_template)

    if should_try_llm:
        ok_llm, llm_value, llm_note = _llm_sql_fallback(
            question=question,
            schema_text=schema_text,
            ticker_hint=ticker_hint,
            start_year=int(start_year),
            end_year=int(end_year),
            row_limit=int(row_limit),
        )
        if ok_llm:
            used_llm = True
            llm_sql = llm_value
            llm_status = llm_note
        else:
            llm_status = llm_note

    sql_candidate = llm_sql if used_llm else suggested_sql
    sql_to_run = (sql_override or "").strip() or sql_candidate
    source = "llm" if used_llm else "template"
    return explanation, llm_status, source, sql_to_run, suggested_sql


@app.cell
def _(explanation, llm_status, mo, source, sql_to_run, suggested_sql):
    mo.md("## Assistant")
    mo.md(f"**Intent:** {explanation}")
    mo.md(f"**SQL Source:** {source}")
    if llm_status:
        mo.md(f"**LLM Status:** {llm_status}")
    mo.md("**Suggested SQL**")
    mo.md("```sql\n" + suggested_sql.strip() + "\n```")
    mo.md("**SQL To Execute**")
    mo.md("```sql\n" + sql_to_run.strip() + "\n```")


@app.cell
def _(end_year, question, row_limit, sql_to_run, start_year, ticker_hint):
    ok, sql_status, df_result = _run_sql_with_params(
        sql=sql_to_run,
        question=question,
        ticker_hint=ticker_hint,
        start_year=int(start_year),
        end_year=int(end_year),
        row_limit=int(row_limit),
    )
    return df_result, ok, sql_status


@app.cell
def _(df_result, mo, ok, sql_status):
    mo.md("## Query Result")
    if not ok:
        mo.md(f"Execution blocked/failed: {sql_status}")
    else:
        mo.md(f"Executed SQL (safe):\n```sql\n{sql_status}\n```")
        mo.md(f"Rows returned: **{len(df_result)}**")
        df_result


@app.cell
def _(df_result, mo, ok):
    mo.md("## Visualization")
    fig = None
    if not ok or df_result is None or df_result.empty:
        mo.md("No chart data available.")
    else:
        kind, x_col, y_col = _chart_plan(df_result)
        if kind == "none":
            mo.md("No automatic chart candidate found (table shown above).")
        else:
            try:
                import plotly.express as px
            except Exception:
                mo.md(
                    "Plotly is not installed. Run `uv add plotly` (or add to dependencies) for auto charts."
                )
            else:
                chart_df = df_result.copy()
                if kind == "bar":
                    chart_df = chart_df.head(25)
                    fig = px.bar(chart_df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                else:
                    color = "ticker" if "ticker" in chart_df.columns else None
                    fig = px.line(
                        chart_df,
                        x=x_col,
                        y=y_col,
                        color=color,
                        markers=True,
                        title=f"{y_col} trend by {x_col}",
                    )

    if fig is not None:
        fig
    return fig


@app.cell
def _(df_result, end_year, ok, question, sql_status, start_year, ticker_hint):
    if ok:
        signature = f"{question.strip()}|{start_year}|{end_year}|{ticker_hint.strip().upper()}|{sql_status}"
        if not CHAT_HISTORY or CHAT_HISTORY[-1].get("signature") != signature:
            CHAT_HISTORY.append(
                {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "question": question.strip(),
                    "ticker_hint": ticker_hint.strip().upper(),
                    "rows": str(0 if df_result is None else len(df_result)),
                    "sql": sql_status,
                    "signature": signature,
                }
            )

        if len(CHAT_HISTORY) > 20:
            del CHAT_HISTORY[:-20]

    return CHAT_HISTORY


@app.cell
def _(CHAT_HISTORY, mo):
    import pandas as pd

    mo.md("## Chat History")
    hist_df = None
    if not CHAT_HISTORY:
        mo.md("No history yet.")
    else:
        hist_df = pd.DataFrame(CHAT_HISTORY)
        if "signature" in hist_df.columns:
            hist_df = hist_df.drop(columns=["signature"])
        hist_df
    return hist_df


if __name__ == "__main__":
    app.run()
