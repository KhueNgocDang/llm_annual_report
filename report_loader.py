from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import FINANCIAL_STATEMENT_MARKDOWN_DIR, MARKDOWN_DIR
from database import connection_scope

_YEAR_RE = re.compile(r"(19\d{2}|20\d{2})")


@dataclass
class _Candidate:
    path: Path
    ticker: str
    year: int
    score: tuple[int, int]


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalized_text(value: str) -> str:
    lowered = _strip_accents(value).lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return lowered.strip()


def _extract_year(text: str) -> int | None:
    m = _YEAR_RE.search(text)
    if not m:
        return None
    year = int(m.group(1))
    if 1900 <= year <= 2100:
        return year
    return None


def _candidate_score(path: Path, stage: str) -> tuple[int, int]:
    text = _normalized_text(path.as_posix())
    score = 0

    if stage == "markdown_annual":
        if "annual report" in text or "annual report" in text.replace("_", " "):
            score += 35
        if "bao cao thuong nien" in text:
            score += 45
    elif stage == "markdown_financial_statement":
        if "bctc" in text:
            score += 35
        if "kiem toan" in text:
            score += 25
        if "hop nhat" in text:
            score += 10

    if "pdf" in text:
        score += 5

    if any(k in text for k in ("thuyet minh", "appendix", "phu luc", "notice", "dieu chinh", "bo sung")):
        score -= 40

    size_kb = int(path.stat().st_size // 1024)
    return (score, size_kb)


def _iter_candidates(root: Path, stage: str) -> tuple[list[_Candidate], int]:
    candidates: list[_Candidate] = []
    skipped = 0
    for path in root.rglob("*.md"):
        rel = path.relative_to(root)
        if len(rel.parts) < 2:
            skipped += 1
            continue

        ticker = rel.parts[0].strip().upper()
        if not ticker:
            skipped += 1
            continue

        year = _extract_year(" ".join(rel.parts))
        if year is None:
            skipped += 1
            continue

        candidates.append(
            _Candidate(
                path=path,
                ticker=ticker,
                year=year,
                score=_candidate_score(path, stage),
            )
        )
    return candidates, skipped


def _pick_best_by_ticker_year(candidates: list[_Candidate]) -> list[_Candidate]:
    chosen: dict[tuple[str, int], _Candidate] = {}
    for item in candidates:
        key = (item.ticker, item.year)
        current = chosen.get(key)
        if current is None or item.score > current.score:
            chosen[key] = item
    return list(chosen.values())


def _upsert_report(
    *,
    table: str,
    stage: str,
    ticker: str,
    year: int,
    content: str,
    source_file: str,
) -> None:
    if table not in {"annual_reports", "financial_statement_reports"}:
        raise ValueError(f"Unsupported report table: {table}")

    with connection_scope() as con:
        con.execute(
            f"""
            INSERT INTO {table} (ticker, year, content, source_file)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (ticker, year) DO UPDATE SET
                content = EXCLUDED.content,
                source_file = EXCLUDED.source_file
            """,
            [ticker, year, content, source_file],
        )

        con.execute(
            """
            INSERT INTO pipeline_files (ticker, year, stage, file_path)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (ticker, year, stage) DO UPDATE SET
                file_path = EXCLUDED.file_path
            """,
            [ticker, year, stage, source_file],
        )


def _load_markdown_into_table(
    *,
    root: Path,
    table: str,
    stage: str,
    tickers: list[str] | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
) -> dict[str, Any]:
    candidates, skipped = _iter_candidates(root, stage)
    selected = _pick_best_by_ticker_year(candidates)

    ticker_filter = {t.upper() for t in (tickers or [])}
    loaded = 0
    failed = 0
    filtered_out = 0

    for item in selected:
        if ticker_filter and item.ticker not in ticker_filter:
            filtered_out += 1
            continue
        if start_year is not None and item.year < start_year:
            filtered_out += 1
            continue
        if end_year is not None and item.year > end_year:
            filtered_out += 1
            continue

        try:
            content = item.path.read_text(encoding="utf-8", errors="ignore").strip()
            if not content:
                failed += 1
                continue
            _upsert_report(
                table=table,
                stage=stage,
                ticker=item.ticker,
                year=item.year,
                content=content,
                source_file=str(item.path),
            )
            loaded += 1
        except Exception:
            failed += 1

    return {
        "root": str(root),
        "table": table,
        "stage": stage,
        "scanned_files": len(candidates),
        "pairs": len(selected),
        "loaded": loaded,
        "failed": failed,
        "skipped": skipped,
        "filtered_out": filtered_out,
    }


def load_annual_reports_from_markdown(
    tickers: list[str] | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
) -> dict[str, Any]:
    return _load_markdown_into_table(
        root=MARKDOWN_DIR,
        table="annual_reports",
        stage="markdown_annual",
        tickers=tickers,
        start_year=start_year,
        end_year=end_year,
    )


def load_financial_statement_reports_from_markdown(
    tickers: list[str] | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
) -> dict[str, Any]:
    return _load_markdown_into_table(
        root=FINANCIAL_STATEMENT_MARKDOWN_DIR,
        table="financial_statement_reports",
        stage="markdown_financial_statement",
        tickers=tickers,
        start_year=start_year,
        end_year=end_year,
    )
