"""Quick smoke commands for LLM embedding and inference flows."""

from __future__ import annotations

import argparse
import json

from config import ensure_env_loaded
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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
