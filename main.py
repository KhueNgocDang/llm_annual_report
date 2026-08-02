from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from config import bootstrap_directories
from database import connection_scope, init_db


def _stats_text() -> str:
    with connection_scope() as con:
        row = con.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM companies) AS companies,
                (SELECT COUNT(*) FROM vietstock_documents) AS documents,
                (SELECT COUNT(*) FROM annual_reports) AS annual_reports,
                (SELECT COUNT(*) FROM financial_statement_reports) AS financial_statement_reports,
                (SELECT COUNT(*) FROM financial_statements) AS financial_statement_rows
            """
        ).fetchone()
    if not row:
        return "No data"
    return (
        f"companies={int(row[0] or 0)} | "
        f"documents={int(row[1] or 0)} | "
        f"annual_reports={int(row[2] or 0)} | "
        f"financial_statement_reports={int(row[3] or 0)} | "
        f"financial_statement_rows={int(row[4] or 0)}"
    )


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
        r"\\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|copy|call|pragma)\\b",
        re.IGNORECASE,
    )
    if forbidden.search(text):
        return False, "Potentially destructive SQL keyword detected"

    return True, text.rstrip(";")


def _run_query(sql: str, limit: int) -> int:
    ok, safe_sql = _safe_select_sql(sql)
    if not ok:
        print(f"Query blocked: {safe_sql}")
        return 2

    with connection_scope() as con:
        df = con.execute(safe_sql).fetchdf()
    if limit > 0:
        df = df.head(limit)
    print(df.to_string(index=False))
    return 0


def _open_marimo() -> int:
    app_file = Path(__file__).resolve().parent / "marimo_app.py"
    result = subprocess.run(["marimo", "edit", str(app_file)], check=False)
    return int(result.returncode)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main",
        description="CLI entrypoint for annual_report. Defaults to opening marimo app.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="marimo",
        choices=["marimo", "stats", "query", "init"],
        help="Action to run.",
    )
    parser.add_argument(
        "--sql",
        default="",
        help="SQL text for the query command.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Row cap when printing query results.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    bootstrap_directories()
    init_db()

    if args.command == "init":
        print("Bootstrap complete")
        print(_stats_text())
        return 0

    if args.command == "stats":
        print(_stats_text())
        return 0

    if args.command == "query":
        if not args.sql.strip():
            parser.error("--sql is required for command=query")
        return _run_query(args.sql, args.limit)

    return _open_marimo()


if __name__ in {"__main__", "__mp_main__"}:
    sys.exit(main())
