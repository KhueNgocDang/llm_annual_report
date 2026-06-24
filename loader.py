"""Load converted Markdown files into the annual_reports DuckDB table."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, Optional

import duckdb

from config import OUTPUT_DIR
from config_marker import MARKDOWN_DIR
from markdown_quality import score_markdown_quality

logger = logging.getLogger(__name__)

_YEAR_RE = re.compile(r"(20\d{2})")


def _annual_report_quality_params(
    content: str,
) -> dict[str, float | int | bool | str]:
    quality = score_markdown_quality(content)
    return {
        "quality_suspicious": quality.suspicious,
        "suspicious_score": quality.suspicious_score,
        "single_char_token_ratio": quality.single_char_token_ratio,
        "broken_spacing_pattern_count": quality.broken_spacing_pattern_count,
        "average_token_length": quality.average_token_length,
        "isolated_diacritic_token_count": quality.isolated_diacritic_token_count,
        "garbled_vietnamese_token_count": quality.garbled_vietnamese_token_count,
        "garbled_vietnamese_token_ratio": quality.garbled_vietnamese_token_ratio,
        "affected_line_count": quality.affected_line_count,
        "affected_line_ratio": quality.affected_line_ratio,
        "affected_region_count": quality.affected_region_count,
        "quality_status": quality.quality_status,
        "quality_reason": quality.suspicious_reason,
        "quality_evidence": quality.quality_evidence,
    }


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


def _resolve_markdown_source_file(source_file: str | Path) -> Path | None:
    source_path = Path(source_file)
    if source_path.is_absolute() and source_path.exists():
        return source_path

    if source_path.exists():
        return source_path

    for base_dir in _markdown_source_dirs():
        if source_path.parts and source_path.parts[0] == base_dir.name:
            candidate = base_dir.parent / source_path
            if candidate.exists():
                return candidate

    return None


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


def find_suspicious_markdown_files(
    source_dirs: list[str | Path] | None = None,
    suspicious_score_threshold: float = 0.45,
) -> list[dict[str, str | int | float | bool]]:
    """Score markdown files on disk and return suspicious reports first."""
    suspicious_entries: list[dict[str, str | int | float | bool]] = []

    for entry in collect_markdown_files(source_dirs=source_dirs):
        path = Path(str(entry["path"]))
        content = path.read_text(encoding="utf-8")
        quality = score_markdown_quality(
            content,
            suspicious_score_threshold=suspicious_score_threshold,
        )
        if not quality.suspicious:
            continue

        suspicious_entries.append(
            {
                **entry,
                **quality.to_dict(),
            }
        )

    suspicious_entries.sort(
        key=lambda entry: (
            float(entry["suspicious_score"]),
            int(entry["broken_spacing_pattern_count"]),
            float(entry["single_char_token_ratio"]),
        ),
        reverse=True,
    )
    return suspicious_entries


def audit_annual_report_quality(
    con: duckdb.DuckDBPyConnection,
    ticker: str | None = None,
    year: int | None = None,
    suspicious_score_threshold: float = 0.45,
) -> dict[str, int]:
    """Re-score loaded annual reports and persist quality signals in DuckDB."""
    query = """
        SELECT ticker, year, content, source_file
        FROM annual_reports
        WHERE 1 = 1
    """
    params: list[str | int] = []
    if ticker:
        query += " AND ticker = ?"
        params.append(ticker.upper())
    if year is not None:
        query += " AND year = ?"
        params.append(year)
    query += " ORDER BY ticker, year"

    rows = con.execute(query, params).fetchall()
    checked = 0
    flagged = 0
    warnings = 0
    failed = 0

    for report_ticker, report_year, content, source_file in rows:
        current_content = content
        if source_file:
            source_path = _resolve_markdown_source_file(source_file)
            if source_path is not None:
                current_content = source_path.read_text(encoding="utf-8")

        quality = score_markdown_quality(
            current_content,
            suspicious_score_threshold=suspicious_score_threshold,
        )
        if quality.quality_status != "pass":
            flagged += 1
        if quality.quality_status == "warning":
            warnings += 1
        if quality.quality_status == "fail":
            failed += 1
        checked += 1
        con.execute(
            """
            UPDATE annual_reports
            SET content = ?,
                quality_checked_at = get_current_timestamp(),
                quality_suspicious = ?,
                suspicious_score = ?,
                single_char_token_ratio = ?,
                broken_spacing_pattern_count = ?,
                average_token_length = ?,
                isolated_diacritic_token_count = ?,
                garbled_vietnamese_token_count = ?,
                garbled_vietnamese_token_ratio = ?,
                affected_line_count = ?,
                affected_line_ratio = ?,
                affected_region_count = ?,
                quality_status = ?,
                quality_reason = ?,
                quality_evidence = ?
            WHERE ticker = ? AND year = ?
            """,
            [
                current_content,
                quality.quality_status != "pass",
                quality.suspicious_score,
                quality.single_char_token_ratio,
                quality.broken_spacing_pattern_count,
                quality.average_token_length,
                quality.isolated_diacritic_token_count,
                quality.garbled_vietnamese_token_count,
                quality.garbled_vietnamese_token_ratio,
                quality.affected_line_count,
                quality.affected_line_ratio,
                quality.affected_region_count,
                quality.quality_status,
                quality.suspicious_reason,
                quality.quality_evidence,
                report_ticker,
                report_year,
            ],
        )

    return {
        "checked": checked,
        "flagged": flagged,
        "warnings": warnings,
        "failed": failed,
    }


def get_force_ocr_candidates(
    con: duckdb.DuckDBPyConnection,
    limit: int | None = None,
    min_garbled_token_count: int | None = None,
) -> list[dict[str, str | int | float | bool]]:
    """Return loaded reports whose quality signals need audit review."""
    query = """
        SELECT
            ar.ticker,
            ar.year,
            ar.source_file,
            ar.suspicious_score,
            ar.single_char_token_ratio,
            ar.broken_spacing_pattern_count,
            ar.average_token_length,
            ar.isolated_diacritic_token_count,
            ar.garbled_vietnamese_token_count,
            ar.garbled_vietnamese_token_ratio,
            ar.affected_line_count,
            ar.affected_line_ratio,
            ar.affected_region_count,
            ar.quality_status,
            ar.quality_reason,
            ar.quality_evidence,
            cj.status,
            cj.force_ocr
        FROM annual_reports ar
        LEFT JOIN conversion_jobs cj
          ON cj.ticker = ar.ticker AND cj.year = ar.year
        WHERE ar.quality_status IS NOT NULL
    """
    params: list[int] = []
    if min_garbled_token_count is not None:
        query += """
            AND (
                ar.quality_status != 'pass'
                OR COALESCE(ar.garbled_vietnamese_token_count, 0) >= ?
            )
        """
        params.append(min_garbled_token_count)
    else:
        query += " AND ar.quality_status != 'pass'"

    query += """
        ORDER BY
            CASE ar.quality_status
                WHEN 'fail' THEN 0
                WHEN 'warning' THEN 1
                ELSE 2
            END,
            ar.garbled_vietnamese_token_count DESC NULLS LAST,
            ar.garbled_vietnamese_token_ratio DESC NULLS LAST,
            ar.suspicious_score DESC NULLS LAST,
            ar.broken_spacing_pattern_count DESC NULLS LAST,
            ar.single_char_token_ratio DESC NULLS LAST,
            ar.ticker,
            ar.year
    """
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    rows = con.execute(query, params).fetchall()

    return [
        {
            "ticker": ticker,
            "year": report_year,
            "source_file": source_file,
            "suspicious_score": suspicious_score,
            "single_char_token_ratio": single_char_token_ratio,
            "broken_spacing_pattern_count": broken_spacing_pattern_count,
            "average_token_length": average_token_length,
            "isolated_diacritic_token_count": isolated_diacritic_token_count,
            "garbled_vietnamese_token_count": garbled_vietnamese_token_count,
            "garbled_vietnamese_token_ratio": garbled_vietnamese_token_ratio,
            "affected_line_count": affected_line_count,
            "affected_line_ratio": affected_line_ratio,
            "affected_region_count": affected_region_count,
            "quality_status": quality_status or "pass",
            "quality_reason": quality_reason or "",
            "quality_evidence": quality_evidence or "",
            "job_status": job_status or "missing",
            "job_force_ocr": (
                job_force_ocr if job_force_ocr is not None else False
            ),
        }
        for (
            ticker,
            report_year,
            source_file,
            suspicious_score,
            single_char_token_ratio,
            broken_spacing_pattern_count,
            average_token_length,
            isolated_diacritic_token_count,
            garbled_vietnamese_token_count,
            garbled_vietnamese_token_ratio,
            affected_line_count,
            affected_line_ratio,
            affected_region_count,
            quality_status,
            quality_reason,
            quality_evidence,
            job_status,
            job_force_ocr,
        ) in rows
    ]


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
            quality = _annual_report_quality_params(content)
            con.execute(
                """
                INSERT INTO annual_reports (
                    ticker,
                    year,
                    content,
                    source_file,
                    quality_checked_at,
                    quality_suspicious,
                    suspicious_score,
                    single_char_token_ratio,
                    broken_spacing_pattern_count,
                    average_token_length,
                    isolated_diacritic_token_count,
                    garbled_vietnamese_token_count,
                    garbled_vietnamese_token_ratio,
                    affected_line_count,
                    affected_line_ratio,
                    affected_region_count,
                    quality_status,
                    quality_reason,
                    quality_evidence
                )
                VALUES (
                    $ticker,
                    $year,
                    $content,
                    $source_file,
                    get_current_timestamp(),
                    $quality_suspicious,
                    $suspicious_score,
                    $single_char_token_ratio,
                    $broken_spacing_pattern_count,
                    $average_token_length,
                    $isolated_diacritic_token_count,
                    $garbled_vietnamese_token_count,
                    $garbled_vietnamese_token_ratio,
                    $affected_line_count,
                    $affected_line_ratio,
                    $affected_region_count,
                    $quality_status,
                    $quality_reason,
                    $quality_evidence
                )
                ON CONFLICT (ticker, year) DO UPDATE SET
                    content = EXCLUDED.content,
                    source_file = EXCLUDED.source_file,
                    quality_checked_at = EXCLUDED.quality_checked_at,
                    quality_suspicious = EXCLUDED.quality_suspicious,
                    suspicious_score = EXCLUDED.suspicious_score,
                    single_char_token_ratio = EXCLUDED.single_char_token_ratio,
                    broken_spacing_pattern_count = EXCLUDED.broken_spacing_pattern_count,
                    average_token_length = EXCLUDED.average_token_length,
                    isolated_diacritic_token_count = EXCLUDED.isolated_diacritic_token_count,
                    garbled_vietnamese_token_count = EXCLUDED.garbled_vietnamese_token_count,
                    garbled_vietnamese_token_ratio = EXCLUDED.garbled_vietnamese_token_ratio,
                    affected_line_count = EXCLUDED.affected_line_count,
                    affected_line_ratio = EXCLUDED.affected_line_ratio,
                    affected_region_count = EXCLUDED.affected_region_count,
                    quality_status = EXCLUDED.quality_status,
                    quality_reason = EXCLUDED.quality_reason,
                    quality_evidence = EXCLUDED.quality_evidence
                """,
                {
                    "ticker": ticker,
                    "year": year,
                    "content": content,
                    "source_file": source_file,
                    "quality_suspicious": quality["quality_suspicious"],
                    "suspicious_score": quality["suspicious_score"],
                    "single_char_token_ratio": quality["single_char_token_ratio"],
                    "broken_spacing_pattern_count": quality[
                        "broken_spacing_pattern_count"
                    ],
                    "average_token_length": quality["average_token_length"],
                    "isolated_diacritic_token_count": quality[
                        "isolated_diacritic_token_count"
                    ],
                    "garbled_vietnamese_token_count": quality[
                        "garbled_vietnamese_token_count"
                    ],
                    "garbled_vietnamese_token_ratio": quality[
                        "garbled_vietnamese_token_ratio"
                    ],
                    "affected_line_count": quality["affected_line_count"],
                    "affected_line_ratio": quality["affected_line_ratio"],
                    "affected_region_count": quality["affected_region_count"],
                    "quality_status": quality["quality_status"],
                    "quality_reason": quality["quality_reason"],
                    "quality_evidence": quality["quality_evidence"],
                },
            )
            con.execute(
                """
                UPDATE conversion_jobs
                SET status = 'completed',
                    force_ocr = CASE
                        WHEN ? THEN TRUE
                        ELSE force_ocr
                    END,
                    rerun_reason = CASE
                        WHEN ? THEN COALESCE(NULLIF(?, ''), rerun_reason)
                        ELSE rerun_reason
                    END,
                    error_message = NULL,
                    failed_step = NULL,
                    pid = NULL,
                    started_at = COALESCE(started_at, get_current_timestamp()),
                    completed_at = COALESCE(completed_at, get_current_timestamp())
                WHERE ticker = ? AND year = ?
                """,
                [
                    quality["quality_status"] == "fail",
                    quality["quality_status"] == "fail",
                    str(quality["quality_reason"]),
                    ticker,
                    year,
                ],
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
