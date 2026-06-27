"""
Converter module — create, run, and track marker-pdf OCR conversion jobs.

Each job converts a single PDF to markdown using marker_single with --force_ocr.
Jobs are tracked in the conversion_jobs table in DuckDB.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, TypedDict

import duckdb

from config import OUTPUT_DIR, RAW_DIR
from config_marker import DATA_DIR, MARKDOWN_DIR, LOGS_DIR, MARKER_EXTRA_ARGS
from database import ensure_company, get_connection

import shutil

# ---------------------------------------------------------------------------
# Input directory sync and validation
# ---------------------------------------------------------------------------

_RAW_REPORT_PATTERN = r"^Báo cáo thường niên năm (\d{4})\s*\.pdf$"


class RawInputEntry(TypedDict):
    ticker: str
    year: int
    path: str


def _has_markdown_output(
    ticker: str,
    year: int,
    output_dir: str | Path | None = None,
) -> bool:
    """Return whether a markdown output already exists for ticker/year."""
    search_dirs: list[Path] = []
    if output_dir is not None:
        search_dirs.append(Path(output_dir))
    for base_dir in (MARKDOWN_DIR, OUTPUT_DIR):
        if base_dir not in search_dirs:
            search_dirs.append(base_dir)

    year_text = str(year)
    ticker_upper = ticker.upper()

    for base_dir in search_dirs:
        ticker_dir = base_dir / ticker_upper
        if not ticker_dir.exists():
            continue
        for md_path in ticker_dir.rglob("*.md"):
            haystack = " ".join(
                (
                    md_path.name,
                    md_path.stem,
                    md_path.parent.name,
                    md_path.as_posix(),
                )
            )
            if year_text in haystack:
                return True
    return False


def _scan_raw_input_dir(
    input_dir: str | Path | None = None,
    valid_pattern: str = _RAW_REPORT_PATTERN,
) -> tuple[list[RawInputEntry], list[str]]:
    """Scan raw input files and infer ticker/year from standard annual-report PDFs."""
    input_dir = Path(input_dir or RAW_DIR)
    valid_re = re.compile(valid_pattern)
    entries: list[RawInputEntry] = []
    nonstandard: list[str] = []

    if not input_dir.exists():
        return entries, nonstandard

    for ticker_dir in sorted(input_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name.upper()
        for entry in sorted(ticker_dir.iterdir()):
            if not entry.is_file():
                continue
            if entry.suffix.lower() != ".pdf":
                nonstandard.append(str(entry))
                continue
            match = valid_re.match(entry.name)
            if not match:
                nonstandard.append(str(entry))
                continue
            entries.append(
                RawInputEntry(
                    ticker=ticker,
                    year=int(match.group(1)),
                    path=str(entry),
                )
            )

    return entries, nonstandard


def preview_raw_input_dir_sync(
    con: duckdb.DuckDBPyConnection,
    input_dir: str | Path | None = None,
    valid_pattern: str = _RAW_REPORT_PATTERN,
    min_year: int | None = None,
) -> dict:
    """Preview company/year additions implied by raw PDFs on disk."""
    entries, nonstandard = _scan_raw_input_dir(
        input_dir=input_dir,
        valid_pattern=valid_pattern,
    )

    existing_companies = {
        row[0]
        for row in con.execute("SELECT ticker FROM companies").fetchall()
    }
    existing_jobs = {
        (ticker, year): source_path
        for ticker, year, source_path in con.execute(
            "SELECT ticker, year, source_path FROM conversion_jobs"
        ).fetchall()
    }

    companies_to_add: set[str] = set()
    years_to_add: dict[str, set[int]] = {}
    candidates: list[dict[str, str | int]] = []
    older_than_min_year: list[dict[str, str | int]] = []

    for entry in entries:
        ticker = str(entry["ticker"])
        year = int(entry["year"])
        path = str(entry["path"])

        if min_year is not None and year < min_year:
            older_than_min_year.append(
                {
                    "source": "raw",
                    "ticker": ticker,
                    "year": year,
                    "path": path,
                    "action": "skip_old",
                }
            )
            continue

        if ticker not in existing_companies:
            companies_to_add.add(ticker)

        current_source = existing_jobs.get((ticker, year))
        if current_source is None:
            action = "create_job"
            years_to_add.setdefault(ticker, set()).add(year)
        elif current_source != path:
            action = "update_job"
        else:
            continue

        candidates.append(
            {
                "source": "raw",
                "ticker": ticker,
                "year": year,
                "path": path,
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
        "older_than_min_year": older_than_min_year,
        "nonstandard": nonstandard,
    }


def resync_input_dir(
    con: duckdb.DuckDBPyConnection | None = None,
    input_dir: str | Path | None = None,
    valid_pattern: str = _RAW_REPORT_PATTERN,
    min_year: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Resync document and job source paths with the current raw input dir.

    The authoritative input layer for conversion is ``data/raw/<ticker>/*.pdf``.
    This scans that directory for standard annual-report filenames and updates both
    ``vietstock_documents.raw_path`` and ``conversion_jobs.source_path`` to match
    the current files on disk. If a raw PDF exists without a conversion job yet,
    a pending job is created so imported files can enter the pipeline.
    """
    entries, nonstandard = _scan_raw_input_dir(
        input_dir=input_dir,
        valid_pattern=valid_pattern,
    )
    added = [str(entry["path"]) for entry in entries]
    updated_documents = 0
    created_companies = 0
    created_jobs = 0
    updated_jobs = 0
    if not entries and not nonstandard:
        return {
            "added": added,
            "nonstandard": nonstandard,
            "older_than_min_year": [],
            "created_companies": 0,
            "created_jobs": 0,
            "updated_documents": 0,
            "updated_jobs": 0,
        }

    own = con is None and not dry_run
    active_con = con or (get_connection() if not dry_run else None)

    try:
        preview = (
            preview_raw_input_dir_sync(
                active_con,
                input_dir=input_dir,
                valid_pattern=valid_pattern,
                min_year=min_year,
            )
            if active_con is not None
            else {
                "companies_to_add": [],
                "candidates": [],
                "older_than_min_year": [],
            }
        )

        if dry_run:
            return {
                "added": added,
                "nonstandard": nonstandard,
                "older_than_min_year": list(preview["older_than_min_year"]),
                "created_companies": len(preview["companies_to_add"]),
                "created_jobs": sum(
                    1
                    for item in preview["candidates"]
                    if item["action"] == "create_job"
                ),
                "updated_documents": 0,
                "updated_jobs": sum(
                    1
                    for item in preview["candidates"]
                    if item["action"] == "update_job"
                ),
            }

        assert active_con is not None

        for ticker in preview["companies_to_add"]:
            ensure_company(active_con, ticker)
            created_companies += 1

        for entry in entries:
            ticker = str(entry["ticker"])
            year = int(entry["year"])
            path_str = str(entry["path"])

            doc_result = active_con.execute(
                """
                UPDATE vietstock_documents
                SET raw_path = ?, synced_to_raw = TRUE
                WHERE ticker = ?
                  AND regexp_extract(title, '(\\d{4})', 1) = ?
                  AND (raw_path IS NULL OR raw_path != ? OR synced_to_raw = FALSE)
                RETURNING id
                """,
                [path_str, ticker, str(year), path_str],
            ).fetchall()
            updated_documents += len(doc_result)

        for item in preview["candidates"]:
            ticker = str(item["ticker"])
            year = int(item["year"])
            path_str = str(item["path"])
            action = str(item["action"])
            target_status = (
                "completed"
                if _has_markdown_output(ticker, year, MARKDOWN_DIR)
                else "pending"
            )

            active_con.execute(
                """
                INSERT INTO conversion_jobs (
                    id, ticker, year, start_year, end_year, source_path, output_dir, status
                )
                VALUES (
                    nextval('conversion_jobs_id_seq'), ?, ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT (ticker, year) DO UPDATE SET
                    start_year = EXCLUDED.start_year,
                    end_year = EXCLUDED.end_year,
                    source_path = EXCLUDED.source_path,
                    output_dir = EXCLUDED.output_dir,
                    status = CASE
                        WHEN EXCLUDED.status = 'completed' THEN 'completed'
                        WHEN conversion_jobs.source_path != EXCLUDED.source_path THEN 'pending'
                        ELSE conversion_jobs.status
                    END,
                    command = CASE
                        WHEN EXCLUDED.status = 'completed' THEN NULL
                        WHEN conversion_jobs.source_path != EXCLUDED.source_path THEN NULL
                        ELSE conversion_jobs.command
                    END,
                    log_path = CASE
                        WHEN EXCLUDED.status = 'completed' THEN conversion_jobs.log_path
                        WHEN conversion_jobs.source_path != EXCLUDED.source_path THEN NULL
                        ELSE conversion_jobs.log_path
                    END,
                    pid = CASE
                        WHEN EXCLUDED.status = 'completed' THEN NULL
                        WHEN conversion_jobs.source_path != EXCLUDED.source_path THEN NULL
                        ELSE conversion_jobs.pid
                    END,
                    error_message = CASE
                        WHEN EXCLUDED.status = 'completed' THEN NULL
                        WHEN conversion_jobs.source_path != EXCLUDED.source_path THEN NULL
                        ELSE conversion_jobs.error_message
                    END,
                    failed_step = CASE
                        WHEN EXCLUDED.status = 'completed' THEN NULL
                        WHEN conversion_jobs.source_path != EXCLUDED.source_path THEN NULL
                        ELSE conversion_jobs.failed_step
                    END,
                    started_at = CASE
                        WHEN EXCLUDED.status = 'completed' THEN COALESCE(conversion_jobs.started_at, get_current_timestamp())
                        WHEN conversion_jobs.source_path != EXCLUDED.source_path THEN NULL
                        ELSE conversion_jobs.started_at
                    END,
                    completed_at = CASE
                        WHEN EXCLUDED.status = 'completed' THEN COALESCE(conversion_jobs.completed_at, get_current_timestamp())
                        WHEN conversion_jobs.source_path != EXCLUDED.source_path THEN NULL
                        ELSE conversion_jobs.completed_at
                    END
                """,
                [
                    ticker,
                    year,
                    year,
                    year,
                    path_str,
                    str(MARKDOWN_DIR),
                    target_status,
                ],
            )

            if action == "create_job":
                created_jobs += 1
            elif action == "update_job":
                updated_jobs += 1
    finally:
        if own and active_con is not None:
            active_con.close()

    return {
        "added": added,
        "nonstandard": nonstandard,
        "older_than_min_year": list(preview["older_than_min_year"]),
        "created_companies": created_companies,
        "created_jobs": created_jobs,
        "updated_documents": updated_documents,
        "updated_jobs": updated_jobs,
    }


def list_nonstandard_input_files(
    input_dir: str | Path | None = None,
    valid_pattern: str = r"^Báo cáo thường niên năm \d{4}\s*\.pdf$",
) -> list[str]:
    """List raw inputs that are either non-PDF files or mismatched PDFs."""
    input_dir = Path(input_dir or RAW_DIR)
    nonstandard: list[str] = []
    valid_re = re.compile(valid_pattern)
    if not input_dir.exists():
        return nonstandard

    for ticker_dir in sorted(input_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        for entry in sorted(ticker_dir.iterdir()):
            if not entry.is_file():
                continue
            if entry.suffix.lower() != ".pdf":
                nonstandard.append(str(entry))
                continue
            if not valid_re.match(entry.name):
                nonstandard.append(str(entry))
    return nonstandard


# ---------------------------------------------------------------------------
# Job creation
# ---------------------------------------------------------------------------


def create_jobs(
    con: duckdb.DuckDBPyConnection,
    tickers: list[str] | None = None,
    years: list[int] | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    output_dir: str | Path | None = None,
) -> int:
    """Create pending conversion jobs from downloaded PDFs.

    Sources from vietstock_documents where synced_to_raw = TRUE.
    Only creates jobs for PDFs that don't already have a job entry.
    Extracts year from the document title (e.g. "Báo cáo thường niên năm 2024").

    Args:
        con: DuckDB connection.
        tickers: Optional filter — only these tickers.
        years: Optional filter — only these specific years.
        start_year: Optional — only years >= start_year.
        end_year: Optional — only years <= end_year.
        output_dir: Override output directory (default: MARKDOWN_DIR).

    Returns:
        Number of jobs created.
    """
    output_dir = str(output_dir or MARKDOWN_DIR)

    query = """
        SELECT vd.ticker, vd.title, vd.raw_path
        FROM vietstock_documents vd
        WHERE vd.synced_to_raw = TRUE
          AND vd.raw_path IS NOT NULL
          AND vd.raw_path != ''
    """
    params: list = []

    if tickers:
        placeholders = ", ".join(["?" for _ in tickers])
        query += f" AND vd.ticker IN ({placeholders})"
        params.extend([t.upper() for t in tickers])

    query += " ORDER BY vd.ticker, vd.title"

    rows = con.execute(query, params).fetchall()

    count = 0
    for ticker, title, raw_path in rows:
        # Extract year from title (last 4-digit number)
        m = re.search(r"(\d{4})", title or "")
        if not m:
            continue
        year = int(m.group(1))

        # Apply year filters
        if years and year not in years:
            continue
        if start_year is not None and year < start_year:
            continue
        if end_year is not None and year > end_year:
            continue

        target_status = (
            "completed"
            if _has_markdown_output(ticker, year, output_dir)
            else "pending"
        )

        con.execute(
            """
            INSERT INTO conversion_jobs (id, ticker, year, start_year, end_year, source_path, output_dir, status)
            VALUES (nextval('conversion_jobs_id_seq'), ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (ticker, year) DO UPDATE SET
                start_year = EXCLUDED.start_year,
                end_year = EXCLUDED.end_year,
                source_path = EXCLUDED.source_path,
                output_dir = EXCLUDED.output_dir,
                status = CASE
                    WHEN EXCLUDED.status = 'completed' THEN 'completed'
                    ELSE conversion_jobs.status
                END,
                error_message = CASE
                    WHEN EXCLUDED.status = 'completed' THEN NULL
                    ELSE conversion_jobs.error_message
                END,
                failed_step = CASE
                    WHEN EXCLUDED.status = 'completed' THEN NULL
                    ELSE conversion_jobs.failed_step
                END,
                pid = CASE
                    WHEN EXCLUDED.status = 'completed' THEN NULL
                    ELSE conversion_jobs.pid
                END,
                completed_at = CASE
                    WHEN EXCLUDED.status = 'completed' THEN COALESCE(conversion_jobs.completed_at, get_current_timestamp())
                    ELSE conversion_jobs.completed_at
                END
            """,
            [
                ticker,
                year,
                start_year,
                end_year,
                raw_path,
                output_dir,
                target_status,
            ],
        )
        count += 1

    return count


# ---------------------------------------------------------------------------
# PDF pre-check
# ---------------------------------------------------------------------------


def _pdf_needs_ocr(
    pdf_path: str, sample_pages: int = 5, min_chars: int = 100
) -> bool:
    """Check if a PDF needs OCR by sampling pages for extractable text.

    Samples up to `sample_pages` pages and checks if the average text length
    per sampled page is below `min_chars`. If so, the PDF is likely scanned
    images and needs OCR.

    Returns True if OCR should be used, False if text is already extractable.
    """
    import fitz

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return True  # Can't open → fall back to OCR

    try:
        total_pages = len(doc)
        if total_pages == 0:
            return True

        pages_to_check = min(sample_pages, total_pages)
        # Sample evenly across the document (skip first page which may be a cover image)
        step = max(1, total_pages // pages_to_check)
        indices = [
            min(i * step, total_pages - 1) for i in range(pages_to_check)
        ]

        total_chars = 0
        for idx in indices:
            text = doc[idx].get_text("text") or ""
            if not isinstance(text, str):
                text = str(text)
            total_chars += len(text.strip())

        avg_chars = total_chars / pages_to_check
        return avg_chars < min_chars
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------


def run_job(con: duckdb.DuckDBPyConnection, job_id: int) -> bool:
    """Run a single conversion job using marker_single.

    Streams stdout/stderr to a log file in real-time.
    Returns True if successful, False otherwise.
    """
    row = con.execute(
        "SELECT id, ticker, year, source_path, output_dir, force_ocr, rerun_reason FROM conversion_jobs WHERE id = ?",
        [job_id],
    ).fetchone()

    if not row:
        return False

    (
        _,
        ticker,
        year,
        source_path,
        output_dir,
        force_ocr_override,
        rerun_reason,
    ) = row

    # Ensure the source file has a .pdf extension (some downloads have .zip/.rar etc.)
    src = Path(source_path)
    if src.suffix.lower() != ".pdf":
        pdf_path = src.with_suffix(".pdf")
        if src.exists() and not pdf_path.exists():
            src.rename(pdf_path)
        source_path = str(pdf_path)

    # Build output directory per ticker
    out_dir = Path(output_dir) / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare log file
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"job_{job_id}.log"

    cmd = [
        "marker_single",
        source_path,
        "--output_dir",
        str(out_dir),
    ]

    # Pre-check: only use force_ocr if the PDF is image-based (no extractable text)
    needs_ocr = _pdf_needs_ocr(source_path)
    should_force_ocr = bool(force_ocr_override) or needs_ocr

    # Append extra marker args from config
    for flag, value in MARKER_EXTRA_ARGS.items():
        # Skip force_ocr if the PDF already has extractable text
        if flag == "force_ocr" and not should_force_ocr:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{flag}")
        else:
            cmd.extend([f"--{flag}", str(value)])

    cmd_str = " ".join(cmd)

    # Mark as running
    con.execute(
        """
        UPDATE conversion_jobs
        SET status = 'running', started_at = get_current_timestamp(),
            error_message = NULL, command = ?, log_path = ?
        WHERE id = ?
        """,
        [cmd_str, str(log_path), job_id],
    )

    try:
        with open(log_path, "w") as log_file:
            log_file.write(f"=== Job #{job_id}: {ticker} {year} ===\n")
            log_file.write(
                f"OCR mode: {'force_ocr (manual override)' if force_ocr_override else 'force_ocr (image-based PDF)' if needs_ocr else 'text extraction (text-based PDF)'}\n"
            )
            if rerun_reason:
                log_file.write(f"Rerun reason: {rerun_reason}\n")
            log_file.write(f"Command: {cmd_str}\n")
            log_file.write(f"Started: {datetime.now().isoformat()}\n")
            log_file.write("=" * 60 + "\n\n")
            log_file.flush()

            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )

            # Store PID for monitoring
            con.execute(
                "UPDATE conversion_jobs SET pid = ? WHERE id = ?",
                [proc.pid, job_id],
            )

            proc.wait()

            log_file.write(f"\n{'=' * 60}\n")
            log_file.write(f"Exit code: {proc.returncode}\n")
            log_file.write(f"Finished: {datetime.now().isoformat()}\n")
            log_file.flush()

        if proc.returncode == 0:
            con.execute(
                """
                UPDATE conversion_jobs
                SET status = 'completed', completed_at = get_current_timestamp(), pid = NULL
                WHERE id = ?
                """,
                [job_id],
            )
            return True
        else:
            # Read last portion of log for error message
            error_msg = _tail_log(log_path, max_chars=2000)
            failed_step = _detect_failed_step(log_path)
            con.execute(
                """
                UPDATE conversion_jobs
                SET status = 'failed', completed_at = get_current_timestamp(),
                    error_message = ?, failed_step = ?, pid = NULL
                WHERE id = ?
                """,
                [error_msg, failed_step, job_id],
            )
            return False

    except Exception as e:
        # Write exception to log file
        try:
            with open(log_path, "a") as lf:
                lf.write(f"\n\n!!! EXCEPTION: {e}\n")
        except Exception:
            pass

        failed_step = (
            _detect_failed_step(log_path) if log_path.exists() else None
        )
        con.execute(
            """
            UPDATE conversion_jobs
            SET status = 'failed', completed_at = get_current_timestamp(),
                error_message = ?, failed_step = ?, pid = NULL
            WHERE id = ?
            """,
            [str(e)[:2000], failed_step, job_id],
        )
        return False


# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------


def _tail_log(log_path: Path, max_chars: int = 2000) -> str:
    """Read the last `max_chars` characters from a log file."""
    try:
        text = log_path.read_text(errors="replace")
        if len(text) > max_chars:
            return "...\n" + text[-max_chars:]
        return text
    except Exception:
        return ""


# Known marker_single processing step patterns (tqdm labels & tracebacks)
_STEP_PATTERNS: list[tuple[str, str]] = [
    (r"Recognizing layout", "Recognizing layout"),
    (r"Running OCR Error Detection", "OCR error detection"),
    (r"Detecting bboxes", "Detecting bboxes"),
    (r"Recognizing Text", "Recognizing text (OCR)"),
    (r"Recognizing tables", "Recognizing tables"),
    (r"PdfiumError|Failed to load document", "Loading PDF"),
    (r"Traceback", "Python exception"),
]


def _detect_failed_step(log_path: Path) -> str | None:
    """Parse a job log and return the last processing step that was active."""
    try:
        text = log_path.read_text(errors="replace")
    except Exception:
        return None

    last_step: str | None = None
    for pattern, label in _STEP_PATTERNS:
        if re.search(pattern, text):
            last_step = label

    return last_step


def get_job_log(
    con: duckdb.DuckDBPyConnection,
    job_id: int,
    tail: int | None = None,
) -> str | None:
    """Read the log file for a job.

    Args:
        con: DuckDB connection.
        job_id: The job ID.
        tail: If set, return only the last N lines.

    Returns:
        Log contents as string, or None if no log exists.
    """
    row = con.execute(
        "SELECT log_path FROM conversion_jobs WHERE id = ?",
        [job_id],
    ).fetchone()

    if not row or not row[0]:
        # Fallback — check if log file exists by convention
        fallback = LOGS_DIR / f"job_{job_id}.log"
        if fallback.exists():
            log_path = fallback
        else:
            return None
    else:
        log_path = Path(row[0])

    if not log_path.exists():
        return None

    try:
        text = log_path.read_text(errors="replace")
        if tail is not None:
            lines = text.splitlines()
            text = "\n".join(lines[-tail:])
        return text
    except Exception:
        return None


def run_pending_jobs(
    con: duckdb.DuckDBPyConnection,
    on_progress: Optional[
        Callable[[int, str, int, str, Optional[str]], None]
    ] = None,
) -> dict[str, int]:
    """Run all pending jobs sequentially.

    Args:
        con: DuckDB connection.
        on_progress: Optional callback(job_id, ticker, year, status, error).

    Returns:
        Dict with counts: {"completed": N, "failed": N, "total": N}.
    """
    rows = con.execute("""
        SELECT id, ticker, year
        FROM conversion_jobs
        WHERE status = 'pending'
        ORDER BY ticker, year
        """).fetchall()

    results = {"completed": 0, "failed": 0, "total": len(rows)}

    for job_id, ticker, year in rows:
        success = run_job(con, job_id)
        status = "completed" if success else "failed"
        results[status] += 1

        if on_progress:
            error = None
            if not success:
                err_row = con.execute(
                    "SELECT error_message FROM conversion_jobs WHERE id = ?",
                    [job_id],
                ).fetchone()
                error = err_row[0] if err_row else None
            on_progress(job_id, ticker, year, status, error)

    return results


# ---------------------------------------------------------------------------
# Job management
# ---------------------------------------------------------------------------


def cancel_job(con: duckdb.DuckDBPyConnection, job_id: int) -> bool:
    """Cancel a running job by killing its process."""
    row = con.execute(
        "SELECT pid, status FROM conversion_jobs WHERE id = ?",
        [job_id],
    ).fetchone()

    if not row:
        return False

    pid, status = row

    if status == "running" and pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    con.execute(
        """
        UPDATE conversion_jobs
        SET status = 'cancelled', completed_at = get_current_timestamp(), pid = NULL
        WHERE id = ?
        """,
        [job_id],
    )
    return True


def reset_failed_jobs(con: duckdb.DuckDBPyConnection) -> int:
    """Reset all failed/cancelled/running jobs back to pending."""
    row = con.execute("""
        SELECT COUNT(*)
        FROM conversion_jobs
        WHERE status IN ('failed', 'cancelled', 'running')
        """).fetchone()
    count = int(row[0]) if row else 0

    if count == 0:
        return 0

    con.execute("""
        UPDATE conversion_jobs
        SET status = 'pending', started_at = NULL, completed_at = NULL,
            error_message = NULL, failed_step = NULL, pid = NULL
        WHERE status IN ('failed', 'cancelled', 'running')
        """)
    return count


def delete_all_jobs(con: duckdb.DuckDBPyConnection) -> int:
    """Delete all conversion jobs."""
    row = con.execute("SELECT COUNT(*) FROM conversion_jobs").fetchone()
    count = row[0] if row else 0
    con.execute("DELETE FROM conversion_jobs")
    return count


def get_job_summary(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Get counts of jobs by status."""
    rows = con.execute("""
        SELECT status, COUNT(*) AS cnt
        FROM conversion_jobs
        GROUP BY status
        """).fetchall()

    summary = {
        "pending": 0,
        "running": 0,
        "completed": 0,
        "failed": 0,
        "cancelled": 0,
    }
    for status, cnt in rows:
        summary[status] = cnt
    summary["total"] = sum(summary.values())
    return summary


def _remove_markdown_outputs_for_year(
    ticker: str,
    year: int,
    output_dirs: list[str | Path] | None = None,
) -> int:
    """Delete existing markdown files for a ticker/year so rerun can overwrite."""
    targets = output_dirs or [MARKDOWN_DIR, OUTPUT_DIR]
    removed = 0
    year_text = str(year)
    ticker_upper = ticker.upper()

    for base in targets:
        ticker_dir = Path(base) / ticker_upper
        if not ticker_dir.exists():
            continue
        for md_path in ticker_dir.rglob("*.md"):
            haystack = " ".join(
                (
                    md_path.name,
                    md_path.stem,
                    md_path.parent.name,
                    md_path.as_posix(),
                )
            )
            if year_text not in haystack:
                continue
            try:
                md_path.unlink()
                removed += 1
            except FileNotFoundError:
                continue

    return removed


def queue_single_rerun_overwrite(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    year: int,
    rerun_reason: str | None = None,
) -> dict[str, int | str]:
    """Queue a pending conversion job for one ticker/year and overwrite old markdown output."""
    ticker = str(ticker or "").strip().upper()
    year = int(year)
    if not ticker:
        raise ValueError("Ticker is required")

    source_row = con.execute(
        "SELECT source_path FROM conversion_jobs WHERE ticker = ? AND year = ?",
        [ticker, year],
    ).fetchone()
    source_path = str(source_row[0]) if source_row and source_row[0] else ""

    if not source_path:
        doc_row = con.execute(
            """
            SELECT raw_path
            FROM vietstock_documents
            WHERE ticker = ?
              AND synced_to_raw = TRUE
              AND raw_path IS NOT NULL
              AND raw_path != ''
              AND regexp_extract(title, '(\\d{4})', 1) = ?
            ORDER BY published_date DESC
            LIMIT 1
            """,
            [ticker, str(year)],
        ).fetchone()
        source_path = str(doc_row[0]) if doc_row and doc_row[0] else ""

    if not source_path:
        raise ValueError(
            f"No raw source file found for {ticker}/{year}. Sync or download source PDF first."
        )

    removed_outputs = _remove_markdown_outputs_for_year(ticker, year)

    reason = (
        rerun_reason
        or f"Manual overwrite rerun ({datetime.now().isoformat(timespec='seconds')})"
    )

    queued = con.execute(
        """
        INSERT INTO conversion_jobs (
            id, ticker, year, start_year, end_year, source_path, output_dir, status
        ) VALUES (
            nextval('conversion_jobs_id_seq'), ?, ?, ?, ?, ?, ?, 'pending'
        )
        ON CONFLICT (ticker, year) DO UPDATE SET
            source_path = EXCLUDED.source_path,
            output_dir = EXCLUDED.output_dir,
            status = 'pending',
            rerun_reason = ?,
            command = NULL,
            log_path = NULL,
            pid = NULL,
            error_message = NULL,
            failed_step = NULL,
            started_at = NULL,
            completed_at = NULL
        RETURNING id
        """,
        [
            ticker,
            year,
            year,
            year,
            source_path,
            str(MARKDOWN_DIR),
            reason,
        ],
    ).fetchone()

    return {
        "ticker": ticker,
        "year": year,
        "job_id": int(queued[0]) if queued and queued[0] is not None else -1,
        "removed_outputs": removed_outputs,
    }


def queue_force_ocr_reruns(
    con: duckdb.DuckDBPyConnection,
    limit: int | None = None,
) -> dict[str, int]:
    """Queue fail-level loaded reports for rerun with a force-OCR override."""
    query = """
        SELECT ar.ticker, ar.year, ar.quality_reason
        FROM annual_reports ar
        INNER JOIN conversion_jobs cj
          ON cj.ticker = ar.ticker AND cj.year = ar.year
        WHERE ar.quality_status = 'fail'
        ORDER BY
            ar.suspicious_score DESC NULLS LAST,
            ar.broken_spacing_pattern_count DESC NULLS LAST,
            ar.single_char_token_ratio DESC NULLS LAST,
            ar.ticker,
            ar.year
    """
    if limit is not None:
        query += " LIMIT ?"
        rows = con.execute(query, [limit]).fetchall()
    else:
        rows = con.execute(query).fetchall()

    queued = 0
    for ticker, year, quality_reason in rows:
        updated = con.execute(
            """
            UPDATE conversion_jobs
            SET status = 'pending',
                force_ocr = TRUE,
                rerun_reason = COALESCE(NULLIF(?, ''), rerun_reason),
                error_message = NULL,
                failed_step = NULL,
                pid = NULL,
                started_at = NULL,
                completed_at = NULL
            WHERE ticker = ? AND year = ?
            RETURNING id
            """,
            [quality_reason, ticker, year],
        ).fetchall()
        queued += len(updated)

    return {"queued": queued, "matched": len(rows)}


def queue_force_ocr_high_garbled_reruns(
    con: duckdb.DuckDBPyConnection,
    min_garbled_token_count: int = 500,
    limit: int | None = None,
) -> dict[str, int]:
    """Queue loaded reports with high garbled-token counts for force-OCR reruns."""
    query = """
        SELECT ar.ticker, ar.year, ar.quality_reason, ar.garbled_vietnamese_token_count
        FROM annual_reports ar
        INNER JOIN conversion_jobs cj
          ON cj.ticker = ar.ticker AND cj.year = ar.year
        WHERE COALESCE(ar.garbled_vietnamese_token_count, 0) >= ?
        ORDER BY
            ar.garbled_vietnamese_token_count DESC NULLS LAST,
            ar.garbled_vietnamese_token_ratio DESC NULLS LAST,
            ar.ticker,
            ar.year
    """
    params: list[int] = [min_garbled_token_count]
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    rows = con.execute(query, params).fetchall()

    queued = 0
    for ticker, year, quality_reason, garbled_count in rows:
        rerun_reason = (
            quality_reason or f"high_garbled_token_count:{garbled_count}"
        )
        updated = con.execute(
            """
            UPDATE conversion_jobs
            SET status = 'pending',
                force_ocr = TRUE,
                rerun_reason = COALESCE(NULLIF(?, ''), rerun_reason),
                error_message = NULL,
                failed_step = NULL,
                pid = NULL,
                started_at = NULL,
                completed_at = NULL
            WHERE ticker = ? AND year = ?
            RETURNING id
            """,
            [rerun_reason, ticker, year],
        ).fetchall()
        queued += len(updated)

    return {
        "queued": queued,
        "matched": len(rows),
        "threshold": min_garbled_token_count,
    }


def get_tickers_without_jobs(
    con: duckdb.DuckDBPyConnection,
) -> list[dict]:
    """Return tickers that have synced PDFs but no conversion jobs.

    Returns a list of dicts with keys: ticker, doc_count (number of synced
    documents without a matching conversion job).
    """
    rows = con.execute("""
        SELECT vd.ticker, COUNT(*) AS doc_count
        FROM vietstock_documents vd
        WHERE vd.synced_to_raw = TRUE
          AND vd.raw_path IS NOT NULL
          AND vd.raw_path != ''
          AND NOT EXISTS (
              SELECT 1 FROM conversion_jobs cj
              WHERE cj.ticker = vd.ticker
                AND cj.year = CAST(regexp_extract(vd.title, '(\\d{4})', 1) AS INTEGER)
          )
        GROUP BY vd.ticker
        ORDER BY vd.ticker
        """).fetchall()

    return [{"ticker": r[0], "doc_count": r[1]} for r in rows]
