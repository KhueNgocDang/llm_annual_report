from __future__ import annotations

import argparse
import json
import sys

from config import DEFAULT_END_YEAR, DEFAULT_START_YEAR, bootstrap_directories
from database import init_db
from report_loader import (
    load_annual_reports_from_markdown,
    load_financial_statement_reports_from_markdown,
)


def _parse_tickers(raw: str) -> list[str] | None:
    tickers = [token.strip().upper() for token in raw.replace(",", " ").split() if token.strip()]
    return tickers or None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="load_reports",
        description="Load markdown reports into DuckDB tables.",
    )
    parser.add_argument(
        "--dataset",
        choices=["annual", "financial_statement", "all"],
        default="all",
        help="Which dataset to load.",
    )
    parser.add_argument(
        "--tickers",
        default="",
        help="Optional ticker filter, comma or space separated.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_START_YEAR,
        help="Inclusive start year filter.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=DEFAULT_END_YEAR,
        help="Inclusive end year filter.",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Return non-zero exit code when any file fails to load.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.start_year > args.end_year:
        parser.error("--start-year cannot be greater than --end-year")

    bootstrap_directories()
    init_db()

    tickers = _parse_tickers(args.tickers)
    summaries: list[dict[str, object]] = []

    if args.dataset in {"annual", "all"}:
        summaries.append(
            load_annual_reports_from_markdown(
                tickers=tickers,
                start_year=args.start_year,
                end_year=args.end_year,
            )
        )

    if args.dataset in {"financial_statement", "all"}:
        summaries.append(
            load_financial_statement_reports_from_markdown(
                tickers=tickers,
                start_year=args.start_year,
                end_year=args.end_year,
            )
        )

    output = {
        "dataset": args.dataset,
        "start_year": args.start_year,
        "end_year": args.end_year,
        "tickers": tickers or [],
        "summaries": summaries,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))

    failed_total = sum(int(s.get("failed", 0)) for s in summaries)
    if args.fail_on_error and failed_total > 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
