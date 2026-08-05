#!/usr/bin/env python3
"""Audit converted financial-statement markdown and rerun suspicious outputs."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path

from config import FINANCIAL_STATEMENT_MARKDOWN_DIR, RAW_DIR
from converter import (
    _resolve_source_pdf,
    convert_financial_statement_source_to_markdown,
)
from loader import collect_markdown_files
from markdown_quality import score_markdown_quality
from vietstock_documents import (
    _financial_statement_title_quality_score,
    _is_preferred_financial_statement_title,
)

RESULTS_DIR = Path("data/output")
_FINANCIAL_STATEMENT_SECTION_MARKERS = (
    "bang can doi ke toan",
    "bao cao ket qua hoat dong kinh doanh",
    "bao cao luu chuyen tien te",
    "luu chuyen tien te",
    "thuyet minh bao cao tai chinh",
)
_DISCLOSURE_MARKERS = (
    "cong bo thong tin",
    "nghi quyet",
    "kinh gui",
    "uy ban chung khoan",
    "so giao dich chung khoan",
)
_MIN_FINANCIAL_STATEMENT_TOKEN_COUNT = 3000


def _normalize_vi_text(value: str) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d")


def _apply_financial_statement_content_checks(
    row: dict[str, str | int | float | bool],
    content: str,
) -> dict[str, str | int | float | bool]:
    normalized = _normalize_vi_text(content)
    matched_sections = [
        marker
        for marker in _FINANCIAL_STATEMENT_SECTION_MARKERS
        if marker in normalized
    ]
    matched_disclosures = [
        marker for marker in _DISCLOSURE_MARKERS if marker in normalized
    ]
    token_count = int(row["token_count"])

    if matched_sections:
        row["financial_statement_section_count"] = len(matched_sections)
        row["financial_statement_disclosure_marker_count"] = len(matched_disclosures)
        return row

    if token_count >= _MIN_FINANCIAL_STATEMENT_TOKEN_COUNT and not matched_disclosures:
        row["financial_statement_section_count"] = 0
        row["financial_statement_disclosure_marker_count"] = 0
        return row

    reasons = [
        part
        for part in str(row.get("suspicious_reason") or "").split(",")
        if part
    ]
    if "non_financial_statement_content" not in reasons:
        reasons.append("non_financial_statement_content")

    evidence_parts = [str(row.get("quality_evidence") or "").strip()]
    evidence_parts.append(
        "content_check="
        f"sections={len(matched_sections)} disclosures={len(matched_disclosures)} tokens={token_count}"
    )
    if matched_disclosures:
        evidence_parts.append("disclosure_markers=" + ",".join(matched_disclosures[:5]))

    row["quality_status"] = "fail"
    row["suspicious"] = True
    row["suspicious_reason"] = ",".join(reasons)
    row["quality_evidence"] = " || ".join(part for part in evidence_parts if part)
    row["suspicious_score"] = max(float(row["suspicious_score"]), 0.95)
    row["financial_statement_section_count"] = len(matched_sections)
    row["financial_statement_disclosure_marker_count"] = len(matched_disclosures)
    return row


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


def _find_financial_statement_source_path(ticker: str, year: int) -> Path:
    ticker_dir = Path(RAW_DIR) / ticker
    if not ticker_dir.exists():
        raise FileNotFoundError(f"Raw folder not found for {ticker}")

    candidates: list[tuple[int, int, int, str, Path]] = []
    for candidate in sorted(ticker_dir.iterdir()):
        if not candidate.is_file():
            continue
        if str(year) not in candidate.name:
            continue
        try:
            resolved = _resolve_source_pdf(candidate, ticker=ticker, year=year)
        except Exception:
            continue

        size = 0
        try:
            size = int(resolved.stat().st_size)
        except Exception:
            size = 0

        title = candidate.stem
        candidates.append(
            (
                1 if _is_preferred_financial_statement_title(title) else 0,
                _financial_statement_title_quality_score(title),
                size,
                candidate.name,
                resolved,
            )
        )

    if not candidates:
        raise FileNotFoundError(
            f"No readable raw financial-statement source file found for {ticker}-{year}"
        )

    candidates.sort(reverse=True)
    return candidates[0][-1]


def _score_entries(
    tickers: list[str] | None = None,
    years: list[int] | None = None,
    suspicious_score_threshold: float = 0.45,
) -> list[dict[str, str | int | float | bool]]:
    rows: list[dict[str, str | int | float | bool]] = []
    for entry in collect_markdown_files(
        source_dirs=[FINANCIAL_STATEMENT_MARKDOWN_DIR],
        tickers=tickers,
        years=years,
    ):
        path = Path(str(entry["path"]))
        content = path.read_text(encoding="utf-8")
        quality = score_markdown_quality(
            content,
            suspicious_score_threshold=suspicious_score_threshold,
        )
        row = {
            **entry,
            **quality.to_dict(),
        }
        rows.append(
            _apply_financial_statement_content_checks(row, content)
        )

    status_rank = {"fail": 0, "warning": 1, "pass": 2}
    rows.sort(
        key=lambda entry: (
            status_rank.get(str(entry["quality_status"]), 3),
            -int(entry["garbled_vietnamese_token_count"]),
            -float(entry["garbled_vietnamese_token_ratio"]),
            -float(entry["suspicious_score"]),
            str(entry["ticker"]),
            int(entry["year"]),
        )
    )
    return rows


def _write_audit_csv(
    rows: list[dict[str, str | int | float | bool]],
    output_path: Path,
) -> None:
    fieldnames = [
        "ticker",
        "year",
        "path",
        "source",
        "source_file",
        "quality_status",
        "suspicious_score",
        "suspicious",
        "suspicious_reason",
        "token_count",
        "single_char_token_count",
        "single_char_token_ratio",
        "broken_spacing_pattern_count",
        "average_token_length",
        "isolated_diacritic_token_count",
        "garbled_vietnamese_token_count",
        "garbled_vietnamese_token_ratio",
        "financial_statement_section_count",
        "financial_statement_disclosure_marker_count",
        "affected_line_count",
        "affected_line_ratio",
        "affected_region_count",
        "quality_evidence",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _select_rerun_candidates(
    rows: list[dict[str, str | int | float | bool]],
    rerun_statuses: set[str],
    min_garbled_token_count: int,
    limit: int,
) -> list[dict[str, str | int | float | bool]]:
    selected: list[dict[str, str | int | float | bool]] = []
    for row in rows:
        status = str(row["quality_status"])
        if status not in rerun_statuses:
            continue
        garbled_count = int(row["garbled_vietnamese_token_count"])
        if status != "fail" and garbled_count < min_garbled_token_count:
            continue
        selected.append(row)
        if limit > 0 and len(selected) >= limit:
            break
    return selected


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit financial-statement markdown quality and rerun suspicious outputs"
    )
    parser.add_argument("--tickers", nargs="*", help="Optional ticker filter")
    parser.add_argument("--years", nargs="*", type=int, help="Optional year filter")
    parser.add_argument(
        "--suspicious-score-threshold",
        type=float,
        default=0.45,
        help="Suspicious score threshold passed to markdown_quality",
    )
    parser.add_argument(
        "--rerun-suspicious",
        action="store_true",
        help="Force-OCR rerun suspicious financial statements after auditing",
    )
    parser.add_argument(
        "--rerun-statuses",
        nargs="*",
        choices=["fail", "warning"],
        default=["fail", "warning"],
        help="Quality statuses eligible for rerun when --rerun-suspicious is enabled",
    )
    parser.add_argument(
        "--min-garbled-token-count",
        type=int,
        default=20,
        help="Minimum garbled-token count required before rerunning warning-level files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of suspicious files to rerun (0 = no limit)",
    )
    parser.add_argument(
        "--allow-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use GPU for reruns when available; pass --no-allow-gpu to force CPU",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.allow_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audit_path = RESULTS_DIR / f"financial_statement_quality_audit_{timestamp}.csv"

    tickers = [value.strip().upper() for value in (args.tickers or []) if value.strip()]
    years = list(args.years or [])
    rows = _score_entries(
        tickers=tickers or None,
        years=years or None,
        suspicious_score_threshold=args.suspicious_score_threshold,
    )
    _write_audit_csv(rows, audit_path)

    fail_count = sum(1 for row in rows if row["quality_status"] == "fail")
    warning_count = sum(1 for row in rows if row["quality_status"] == "warning")
    pass_count = sum(1 for row in rows if row["quality_status"] == "pass")
    print(f"audited={len(rows)}")
    print(f"quality_fail={fail_count}")
    print(f"quality_warning={warning_count}")
    print(f"quality_pass={pass_count}")
    print(f"audit_csv={audit_path}")

    for row in rows[:20]:
        if row["quality_status"] == "pass":
            continue
        print(
            "SUSPECT\t"
            f"{row['ticker']}\t{row['year']}\t{row['quality_status']}\t"
            f"score={float(row['suspicious_score']):.3f}\t"
            f"garbled={int(row['garbled_vietnamese_token_count'])}\t"
            f"reason={row['suspicious_reason']}"
        )

    if not args.rerun_suspicious:
        return 0

    rerun_candidates = _select_rerun_candidates(
        rows,
        rerun_statuses=set(args.rerun_statuses),
        min_garbled_token_count=int(args.min_garbled_token_count),
        limit=int(args.limit),
    )
    print(f"rerun_candidates={len(rerun_candidates)}")

    rerun_path = RESULTS_DIR / f"financial_statement_quality_rerun_{timestamp}.csv"
    with rerun_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "ticker",
                "year",
                "before_quality_status",
                "before_suspicious_score",
                "before_garbled_vietnamese_token_count",
                "removed_paths",
                "status",
                "detail",
                "after_quality_status",
                "after_suspicious_score",
                "after_garbled_vietnamese_token_count",
                "after_path",
                "finished_at",
            ],
        )
        writer.writeheader()

        for index, row in enumerate(rerun_candidates, start=1):
            ticker = str(row["ticker"])
            year = int(row["year"])
            print(
                f"RERUN\t{index}/{len(rerun_candidates)}\t{ticker}\t{year}\t"
                f"before={row['quality_status']}\tgarbled={int(row['garbled_vietnamese_token_count'])}"
            )
            removed_paths: list[str] = []
            status = "ok"
            detail = ""
            after_quality_status = ""
            after_suspicious_score = ""
            after_garbled_count = ""
            after_path = ""
            try:
                removed_paths = _cleanup_year_outputs(ticker, year)
                source_path = _find_financial_statement_source_path(ticker, year)
                convert_financial_statement_source_to_markdown(
                    source_path,
                    ticker=ticker,
                    year=year,
                    output_dir=FINANCIAL_STATEMENT_MARKDOWN_DIR,
                    force_ocr_override=True,
                    rerun_reason=(
                        "quality_audit:"
                        f"{row['quality_status']}:{row['suspicious_reason']}"
                    ),
                )

                rescored = _score_entries(
                    tickers=[ticker],
                    years=[year],
                    suspicious_score_threshold=args.suspicious_score_threshold,
                )
                if not rescored:
                    raise RuntimeError("rerun finished but no markdown file was found")
                refreshed = rescored[0]
                after_quality_status = str(refreshed["quality_status"])
                after_suspicious_score = f"{float(refreshed['suspicious_score']):.6f}"
                after_garbled_count = str(
                    int(refreshed["garbled_vietnamese_token_count"])
                )
                after_path = str(refreshed["path"])
                detail = after_path
            except Exception as exc:
                status = "error"
                detail = str(exc)

            writer.writerow(
                {
                    "ticker": ticker,
                    "year": year,
                    "before_quality_status": row["quality_status"],
                    "before_suspicious_score": f"{float(row['suspicious_score']):.6f}",
                    "before_garbled_vietnamese_token_count": int(
                        row["garbled_vietnamese_token_count"]
                    ),
                    "removed_paths": "|".join(removed_paths),
                    "status": status,
                    "detail": detail,
                    "after_quality_status": after_quality_status,
                    "after_suspicious_score": after_suspicious_score,
                    "after_garbled_vietnamese_token_count": after_garbled_count,
                    "after_path": after_path,
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            fh.flush()

    print(f"rerun_csv={rerun_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())