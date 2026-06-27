"""Quick smoke commands for LLM embedding and inference flows."""

from __future__ import annotations

import argparse
import json

from company_history import sync_company_history, sync_company_history_many
from config import MARKDOWN_DIR, OUTPUT_DIR, ensure_env_loaded
from database import get_connection, init_db
from llm_embeddings import embed_all_reports, embed_report
from llm_governance import create_governance_jobs, extract_governance
from llm_inference import create_inference_jobs, infer_report
from llm_proper_vn import create_proper_vn_jobs, infer_proper_vn_report

ensure_env_loaded()


def cmd_embed_all(args: argparse.Namespace) -> None:
    con = get_connection()
    try:
        init_db(con)
        result = embed_all_reports(
            con,
            replace=args.replace,
            tickers=args.tickers,
            years=args.years,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        con.close()


def cmd_embed_one(args: argparse.Namespace) -> None:
    con = get_connection()
    try:
        init_db(con)
        result = embed_report(
            con,
            args.ticker,
            args.year,
            replace=args.replace,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        con.close()


def cmd_create_jobs(args: argparse.Namespace) -> None:
    con = get_connection()
    try:
        init_db(con)
        result = {
            "inference_jobs": create_inference_jobs(
                con, tickers=args.tickers, years=args.years, replace=args.replace
            ),
            "proper_vn_jobs": create_proper_vn_jobs(
                con, tickers=args.tickers, years=args.years, replace=args.replace
            ),
            "governance_jobs": create_governance_jobs(
                con, tickers=args.tickers, years=args.years, replace=args.replace
            ),
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        con.close()


def cmd_infer_one(args: argparse.Namespace) -> None:
    con = get_connection()
    try:
        init_db(con)
        edc_count = infer_report(
            args.ticker,
            args.year,
            con=con,
            replace=args.replace,
            inference_model=args.model,
        )
        proper = infer_proper_vn_report(
            args.ticker,
            args.year,
            con=con,
            replace=args.replace,
            inference_model=args.model,
        )
        gov_count = extract_governance(
            args.ticker,
            args.year,
            con=con,
            replace=args.replace,
            inference_model=args.model,
        )
        print(
            json.dumps(
                {
                    "ticker": args.ticker.upper(),
                    "year": args.year,
                    "model": args.model,
                    "edc_categories": edc_count,
                    "proper_vn": proper,
                    "governance_items": gov_count,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    finally:
        con.close()


def cmd_rescan_markdown(args: argparse.Namespace) -> None:
    from loader import preview_markdown_sync, sync_markdown_files

    source_dirs = [MARKDOWN_DIR]
    if args.include_output:
        source_dirs = [OUTPUT_DIR, MARKDOWN_DIR]

    con = get_connection()
    try:
        init_db(con)
        preview = preview_markdown_sync(
            con,
            source_dirs=source_dirs,
            tickers=args.tickers,
            years=args.years,
        )
        if args.preview_only:
            print(json.dumps(preview, indent=2, ensure_ascii=False))
            return

        result = sync_markdown_files(
            con,
            source_dirs=source_dirs,
            tickers=args.tickers,
            years=args.years,
        )
        print(
            json.dumps(
                {
                    "source_dirs": [str(path) for path in source_dirs],
                    "scanned": preview["scanned"],
                    "candidates": len(preview["candidates"]),
                    "companies_to_add": preview["companies_to_add"],
                    "years_to_add": preview["years_to_add"],
                    "loaded": result["loaded"],
                    "failed": result["failed"],
                    "created_companies": result["created_companies"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    finally:
        con.close()


def cmd_sync_company_history(args: argparse.Namespace) -> None:
    con = get_connection()
    try:
        init_db(con)
        if args.ticker:
            result = sync_company_history(
                con,
                ticker=args.ticker,
                section_name=args.section_name,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return

        tickers = [str(t).strip().upper() for t in (args.tickers or []) if str(t).strip()]
        if not tickers:
            raise ValueError("Provide --ticker or --tickers")

        result = sync_company_history_many(
            con,
            tickers=tickers,
            section_name=args.section_name,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        con.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke commands for LLM flows")
    sub = parser.add_subparsers(dest="command", required=True)

    p_embed_all = sub.add_parser("embed-all", help="Embed all annual reports")
    p_embed_all.add_argument("--tickers", nargs="*")
    p_embed_all.add_argument("--years", nargs="*", type=int)
    p_embed_all.add_argument("--replace", action="store_true")
    p_embed_all.set_defaults(func=cmd_embed_all)

    p_embed_one = sub.add_parser("embed-one", help="Embed one annual report")
    p_embed_one.add_argument("--ticker", required=True)
    p_embed_one.add_argument("--year", required=True, type=int)
    p_embed_one.add_argument("--replace", action="store_true")
    p_embed_one.set_defaults(func=cmd_embed_one)

    p_jobs = sub.add_parser("create-jobs", help="Create inference/proper/governance jobs")
    p_jobs.add_argument("--tickers", nargs="*")
    p_jobs.add_argument("--years", nargs="*", type=int)
    p_jobs.add_argument("--replace", action="store_true")
    p_jobs.set_defaults(func=cmd_create_jobs)

    p_infer = sub.add_parser(
        "infer-one",
        help="Run one ticker-year inference for EDC, PROPER-VN, and governance",
    )
    p_infer.add_argument("--ticker", required=True)
    p_infer.add_argument("--year", required=True, type=int)
    p_infer.add_argument("--model", default="gpt-4.1-mini")
    p_infer.add_argument("--replace", action="store_true")
    p_infer.set_defaults(func=cmd_infer_one)

    p_rescan = sub.add_parser(
        "rescan-markdown",
        help="Rescan markdown files on disk and upsert annual_reports",
    )
    p_rescan.add_argument("--tickers", nargs="*")
    p_rescan.add_argument("--years", nargs="*", type=int)
    p_rescan.add_argument(
        "--include-output",
        action="store_true",
        help="Also scan data/output in addition to data/markdown",
    )
    p_rescan.add_argument(
        "--preview-only",
        action="store_true",
        help="Print preview only without updating database",
    )
    p_rescan.set_defaults(func=cmd_rescan_markdown)

    p_history = sub.add_parser(
        "sync-company-history",
        help="Fetch Vietstock company profile milestones (Moc lich su) and store firm age",
    )
    p_history.add_argument("--ticker")
    p_history.add_argument("--tickers", nargs="*")
    p_history.add_argument(
        "--section-name",
        default="ho-so-doanh-nghiep",
        help="Vietstock /view section name payload (default: ho-so-doanh-nghiep)",
    )
    p_history.set_defaults(func=cmd_sync_company_history)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
