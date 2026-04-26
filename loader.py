"""Load converted Markdown files into the annual_reports DuckDB table."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, Optional

import duckdb

from config import OUTPUT_DIR
from config_marker import MARKDOWN_DIR

logger = logging.getLogger(__name__)

_YEAR_RE = re.compile(r"(20\d{2})")


def _extract_year(text: str) -> int | None:
    """Try to extract a 4-digit year from a string."""
    m = _YEAR_RE.search(text)
    return int(m.group(1)) if m else None


def _markdown_source_dirs(
    source_dirs: list[str | Path] | None = None,
) -> list[Path]:
    dirs = source_dirs or [OUTPUT_DIR, MARKDOWN_DIR]
    seen: set[Path] = set()
    normalized: list[Path] = []
    for value in dirs:
        path = Path(value)
        if path in seen:
            continue
        seen.add(path)
        normalized.append(path)
    return normalized


def collect_markdown_files(
    source_dirs: list[str | Path] | None = None,
) -> list[dict[str, str | int]]:
    """Collect unique markdown documents from output/ and markdown/."""
    entries: list[dict[str, str | int]] = []
    seen_keys: set[tuple[str, int]] = set()

    for base_dir in _markdown_source_dirs(source_dirs):
        if not base_dir.exists():
            continue

        for md_path in sorted(base_dir.rglob("*.md")):
            rel_path = md_path.relative_to(base_dir)
            if not rel_path.parts:
                continue

            ticker = rel_path.parts[0].upper()
            year = (
                _extract_year(md_path.stem)
                or _extract_year(md_path.parent.name)
                or _extract_year(str(rel_path))
            )
            if year is None:
                continue

            key = (ticker, year)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            entries.append(
                {
                    "ticker": ticker,
                    "year": year,
                    "path": str(md_path),
                    "source": base_dir.name,
                    "source_file": str(Path(base_dir.name) / rel_path),
                }
            )

    return entries


def preview_markdown_sync(
    con: duckdb.DuckDBPyConnection,
    source_dirs: list[str | Path] | None = None,
) -> dict:
    """Preview company/year additions implied by markdown files on disk."""
    entries = collect_markdown_files(source_dirs=source_dirs)
    existing_companies = {
        row[0]
        for row in con.execute("SELECT ticker FROM companies").fetchall()
    }
    existing_reports = {
        (ticker, year): source_file
        for ticker, year, source_file in con.execute(
            "SELECT ticker, year, source_file FROM annual_reports"
        ).fetchall()
    }

    companies_to_add: set[str] = set()
    years_to_add: dict[str, set[int]] = {}
    candidates: list[dict[str, str | int]] = []

    for entry in entries:
        ticker = str(entry["ticker"])
        year = int(entry["year"])

        if ticker not in existing_companies:
            companies_to_add.add(ticker)

        current_source = existing_reports.get((ticker, year))
        if current_source is None:
            action = "create_report"
            years_to_add.setdefault(ticker, set()).add(year)
        else:
            action = "refresh_report"

        candidates.append(
            {
                **entry,
                "action": action,
            }
        )

    return {
        "scanned": len(entries),
        "candidates": candidates,
        "companies_to_add": sorted(companies_to_add),
        "years_to_add": {
            ticker: sorted(years)
            for ticker, years in sorted(years_to_add.items())
        },
    }


def sync_markdown_files(
    con: duckdb.DuckDBPyConnection,
    source_dirs: list[str | Path] | None = None,
    on_progress: Optional[
        Callable[[str, int | None, int, str, Optional[Exception]], None]
    ] = None,
) -> dict[str, int]:
    """Upsert markdown files from output/ and markdown/ into annual_reports."""
    from database import ensure_company

    counts = {"loaded": 0, "failed": 0, "created_companies": 0}
    preview = preview_markdown_sync(con, source_dirs=source_dirs)

    for ticker in preview["companies_to_add"]:
        ensure_company(con, ticker)
        counts["created_companies"] += 1

    for entry in preview["candidates"]:
        ticker = str(entry["ticker"])
        year = int(entry["year"])
        path = Path(str(entry["path"]))
        source_file = str(entry["source_file"])
        try:
            content = path.read_text(encoding="utf-8")
            con.execute(
                """
                INSERT INTO annual_reports (ticker, year, content, source_file)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (ticker, year) DO UPDATE SET
                    content = EXCLUDED.content,
                    source_file = EXCLUDED.source_file
                """,
                [ticker, year, content, source_file],
            )
            con.execute(
                """
                UPDATE conversion_jobs
                SET status = 'completed',
                    error_message = NULL,
                    failed_step = NULL,
                    pid = NULL,
                    started_at = COALESCE(started_at, get_current_timestamp()),
                    completed_at = COALESCE(completed_at, get_current_timestamp())
                WHERE ticker = ? AND year = ?
                """,
                [ticker, year],
            )
            counts["loaded"] += 1
            if on_progress:
                on_progress(ticker, year, len(content), "loaded", None)
        except Exception as exc:
            counts["failed"] += 1
            logger.error("Failed to load %s: %s", path, exc)
            if on_progress:
                on_progress(ticker, year, 0, "error", exc)

    return counts


def load_all(
    con: duckdb.DuckDBPyConnection,
    on_progress: Optional[
        Callable[[str, int | None, int, str, Optional[Exception]], None]
    ] = None,
) -> dict[str, int]:
    """Read all .md files from data/output/ and upsert into annual_reports.

    Args:
        con: DuckDB connection.
        on_progress: callback(ticker, year, content_length, status, error)
            status is one of: "loaded", "error"

    Returns:
        Dict with keys 'loaded', 'failed'.
    """
    counts = sync_markdown_files(con, on_progress=on_progress)
    return {"loaded": counts["loaded"], "failed": counts["failed"]}
