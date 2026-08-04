#!/usr/bin/env python3
"""Resume targeted financial-statement markdown reconversion.

This script uses the existing issue inventory at
``data/output/financial_statement_markdown_partial_issues.csv`` to build a
target list of ticker/year pairs that need rerun attention because they were
missing, duplicated, or suspiciously small.

Default behavior is conservative and monitor-friendly:
- leaves GPU enabled by default; use --no-allow-gpu to force CPU execution
- removes stale year-matching output folders only for duplicate/suspicious rows
- continues on individual failures and writes a run results CSV
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from config import FINANCIAL_STATEMENT_MARKDOWN_DIR
from converter import convert_financial_statement_to_markdown
from database import get_connection, init_db

ISSUES_CSV = Path("data/output/financial_statement_markdown_partial_issues.csv")
RESULTS_DIR = Path("data/output")
RESULTS_GLOB = "financial_statement_reconversion_results_*.csv"


@dataclass(frozen=True)
class Target:
    ticker: str
    year: int
    reason: str
    cleanup_existing: bool


def _parse_years(raw: str) -> list[int]:
    value = (raw or "").strip()
    if not value:
        return []
    return [int(part) for part in value.split("|") if part.strip()]


def _load_targets(csv_path: Path) -> list[Target]:
    targets: dict[tuple[str, int], Target] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ticker = str(row.get("ticker") or "").strip().upper()
            if not ticker:
                continue

            for year in _parse_years(str(row.get("missing_years") or "")):
                key = (ticker, year)
                targets.setdefault(
                    key,
                    Target(
                        ticker=ticker,
                        year=year,
                        reason="missing",
                        cleanup_existing=False,
                    ),
                )

            for year in _parse_years(str(row.get("duplicate_years") or "")):
                key = (ticker, year)
                targets[key] = Target(
                    ticker=ticker,
                    year=year,
                    reason="duplicate",
                    cleanup_existing=True,
                )

            for year in _parse_years(
                str(row.get("suspiciously_small_years") or "")
            ):
                key = (ticker, year)
                prior = targets.get(key)
                reason = "suspicious_small"
                if prior is not None and prior.reason != reason:
                    reason = f"{prior.reason}+suspicious_small"
                targets[key] = Target(
                    ticker=ticker,
                    year=year,
                    reason=reason,
                    cleanup_existing=True,
                )

    return sorted(targets.values(), key=lambda item: (item.ticker, item.year))


def _load_completed_targets(results_dir: Path) -> set[tuple[str, int]]:
    completed: set[tuple[str, int]] = set()
    for csv_path in sorted(results_dir.glob(RESULTS_GLOB)):
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    status = str(row.get("status") or "").strip().lower()
                    if status != "ok":
                        continue

                    ticker = str(row.get("ticker") or "").strip().upper()
                    year_raw = str(row.get("year") or "").strip()
                    if not ticker or not year_raw:
                        continue

                    try:
                        year = int(year_raw)
                    except ValueError:
                        continue

                    completed.add((ticker, year))
        except FileNotFoundError:
            continue

    return completed


def _cleanup_year_outputs(ticker: str, year: int) -> list[str]:
    ticker_dir = Path(FINANCIAL_STATEMENT_MARKDOWN_DIR) / ticker
    if not ticker_dir.exists():
        return []

    removed: list[str] = []
    year_text = str(year)
    for child in sorted(ticker_dir.iterdir()):
        if not child.is_dir():
            continue
        if year_text not in child.name:
            continue
        shutil.rmtree(child)
        removed.append(str(child))
    return removed


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resume targeted financial-statement reconversion"
    )
    parser.add_argument(
        "--issues-csv",
        type=Path,
        default=ISSUES_CSV,
        help="CSV file listing ticker/year issues to rerun",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        help="Optional ticker filter",
    )
    parser.add_argument(
        "--years",
        nargs="*",
        type=int,
        help="Optional year filter",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of targets to run (0 = no limit)",
    )
    parser.add_argument(
        "--allow-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use GPU for marker conversion when available; pass --no-allow-gpu to force CPU",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the target plan without converting anything",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore prior successful reconversion results and rerun all selected targets",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not args.issues_csv.exists():
        raise FileNotFoundError(f"Issues CSV not found: {args.issues_csv}")

    if not args.allow_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    targets = _load_targets(args.issues_csv)

    if args.tickers:
        ticker_filter = {value.strip().upper() for value in args.tickers if value.strip()}
        targets = [target for target in targets if target.ticker in ticker_filter]
    if args.years:
        year_filter = set(args.years)
        targets = [target for target in targets if target.year in year_filter]

    resumed_targets = 0
    if not args.no_resume:
        completed_targets = _load_completed_targets(RESULTS_DIR)
        if completed_targets:
            before_count = len(targets)
            targets = [
                target
                for target in targets
                if (target.ticker, target.year) not in completed_targets
            ]
            resumed_targets = before_count - len(targets)

    if args.limit and args.limit > 0:
        targets = targets[: args.limit]

    print(f"targets={len(targets)}")
    if resumed_targets:
        print(f"resume_skip_ok={resumed_targets}")
    for target in targets:
        print(
            f"PLAN\t{target.ticker}\t{target.year}\t{target.reason}"
            f"\tcleanup={int(target.cleanup_existing)}"
        )

    if args.dry_run:
        return 0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"financial_statement_reconversion_results_{timestamp}.csv"

    con = get_connection()
    try:
        init_db(con)
        with results_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "ticker",
                    "year",
                    "reason",
                    "cleanup_existing",
                    "removed_paths",
                    "status",
                    "detail",
                    "finished_at",
                ],
            )
            writer.writeheader()

            for index, target in enumerate(targets, start=1):
                removed_paths: list[str] = []
                print(
                    f"RUN\t{index}/{len(targets)}\t{target.ticker}\t{target.year}\t{target.reason}"
                )
                try:
                    if target.cleanup_existing:
                        removed_paths = _cleanup_year_outputs(target.ticker, target.year)
                        if removed_paths:
                            print(
                                f"CLEAN\t{target.ticker}\t{target.year}\t{len(removed_paths)}"
                            )

                    out_dir = convert_financial_statement_to_markdown(
                        con,
                        target.ticker,
                        target.year,
                    )
                    status = "ok"
                    detail = str(out_dir)
                    print(
                        f"OK\t{target.ticker}\t{target.year}\t{detail}"
                    )
                except Exception as exc:
                    status = "error"
                    detail = str(exc)
                    print(
                        f"ERR\t{target.ticker}\t{target.year}\t{detail}"
                    )

                writer.writerow(
                    {
                        "ticker": target.ticker,
                        "year": target.year,
                        "reason": target.reason,
                        "cleanup_existing": int(target.cleanup_existing),
                        "removed_paths": "|".join(removed_paths),
                        "status": status,
                        "detail": detail,
                        "finished_at": datetime.now().isoformat(timespec="seconds"),
                    }
                )
                fh.flush()
    finally:
        con.close()

    print(f"results_csv={results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())