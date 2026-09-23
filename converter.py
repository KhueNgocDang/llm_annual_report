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
import time
import unicodedata
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, TypedDict

import duckdb

from config import FINANCIAL_STATEMENT_MARKDOWN_DIR, OUTPUT_DIR, RAW_DIR
from config_marker import DATA_DIR, MARKDOWN_DIR, LOGS_DIR, MARKER_EXTRA_ARGS
from database import ensure_company, get_connection
from vietstock_documents import (
    DOC_TYPE_ANNUAL_REPORT,
    DOC_TYPE_AUDITED_CONSOLIDATED_FS,
    _financial_statement_title_quality_score,
    _is_preferred_financial_statement_title,
)

import shutil

# ---------------------------------------------------------------------------
# Input directory sync and validation
# ---------------------------------------------------------------------------

_RAW_REPORT_PATTERN = r"^Báo cáo thường niên năm (\d{4})\s*\.pdf$"


def _build_marker_single_command() -> list[str]:
    """Reuse the active repo venv when present; otherwise prefer uv-managed execution."""
    project_root = Path(__file__).resolve().parent
    local_marker = project_root / ".venv" / "bin" / "marker_single"
    active_venv = os.environ.get("VIRTUAL_ENV")
    if active_venv and Path(active_venv).resolve() == (project_root / ".venv").resolve():
        if local_marker.exists() and os.access(local_marker, os.X_OK):
            return [str(local_marker)]

    uv_executable = shutil.which("uv")
    if uv_executable:
        return [
            uv_executable,
            "run",
            "--project",
            str(project_root),
            "marker_single",
        ]

    if local_marker.exists() and os.access(local_marker, os.X_OK):
        return [str(local_marker)]

    global_marker = shutil.which("marker_single")
    if global_marker:
        return [global_marker]

    raise FileNotFoundError(
        "marker_single was not found. Run `uv sync` in the project root to install the repo-managed CLI."
    )


def _job_kind_from_output_dir(output_dir: str | Path) -> str:
    """Infer converter job kind from output directory path."""
    try:
        resolved = Path(output_dir).resolve()
    except Exception:
        resolved = Path(output_dir)
    return (
        "financial_statement"
        if resolved == Path(FINANCIAL_STATEMENT_MARKDOWN_DIR).resolve()
        else "annual"
    )


def _default_output_dir_for_doc_type(doc_type: str) -> Path:
    """Map Vietstock doc_type to converter output root directory."""
    if str(doc_type) == DOC_TYPE_AUDITED_CONSOLIDATED_FS:
        return Path(FINANCIAL_STATEMENT_MARKDOWN_DIR)
    return Path(MARKDOWN_DIR)


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
    doc_type: str = DOC_TYPE_ANNUAL_REPORT,
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
    if str(doc_type) != DOC_TYPE_ANNUAL_REPORT:
        raise ValueError(
            "conversion_jobs supports annual reports only; use convert_financial_statement_to_markdown for financial-statement management"
        )

    output_dir = str(output_dir or _default_output_dir_for_doc_type(doc_type))

    query = """
        SELECT vd.ticker, vd.title, vd.raw_path
        FROM vietstock_documents vd
        WHERE vd.synced_to_raw = TRUE
          AND vd.raw_path IS NOT NULL
          AND vd.raw_path != ''
                    AND vd.doc_type = ?
    """
    params: list = [str(doc_type)]

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


def _find_manual_pdf_candidate(
    search_dir: Path,
    *,
    ticker: str | None = None,
    year: int | None = None,
    preferred_stem: str | None = None,
) -> Path | None:
    """Find a likely manually extracted PDF in *search_dir*.

    Ranking priority:
    - filename contains year
    - filename contains ticker
    - filename contains original source stem
    If there is only one PDF, use it directly.
    """
    if not search_dir.exists() or not search_dir.is_dir():
        return None

    pdfs = sorted(path for path in search_dir.glob("*.pdf") if path.is_file())
    if not pdfs:
        return None
    if len(pdfs) == 1:
        return pdfs[0]

    ticker_text = str(ticker or "").strip().upper()
    year_text = str(int(year)) if year is not None else ""
    stem_text = str(preferred_stem or "").strip().lower()

    def _score(path: Path) -> tuple[int, int]:
        name_upper = path.name.upper()
        name_lower = path.name.lower()
        score = 0
        if year_text and year_text in name_upper:
            score += 5
        if ticker_text and ticker_text in name_upper:
            score += 3
        if stem_text and stem_text in name_lower:
            score += 2
        # Tie-break by shorter filename as a weak proxy for canonical doc file.
        return (score, -len(path.name))

    ranked = sorted(pdfs, key=_score, reverse=True)
    best = ranked[0]
    if _score(best)[0] <= 0:
        return None
    return best


def _normalize_archive_member_name(value: str) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d")


def _archive_pdf_language_score(filename: str) -> int:
    """Score PDF member names to prefer Vietnamese-language report files."""
    name = Path(filename).name.lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", name) if t]
    token_set = set(tokens)

    score = 0
    if "vi" in token_set or "vietnamese" in token_set:
        score += 10
    if "en" in token_set or "english" in token_set:
        score -= 10

    vi_markers = {
        "bao",
        "cao",
        "tai",
        "chinh",
        "hop",
        "nhat",
        "kiem",
        "toan",
        "thuyet",
        "minh",
        "viet",
        "nam",
    }
    score += sum(1 for marker in vi_markers if marker in token_set)

    en_markers = {
        "audited",
        "consolidated",
        "financial",
        "statements",
        "statement",
        "report",
        "notes",
        "note",
        "english",
    }
    score -= sum(1 for marker in en_markers if marker in token_set)
    return score


def _extract_pdf_from_rar(src: Path) -> Path:
    """Extract the best PDF candidate from a RAR archive into _extracted/."""
    unrar = shutil.which("unrar")
    if not unrar:
        raise ValueError(
            "RAR archive support requires `unrar` to be installed"
        )

    list_proc = subprocess.run(
        [unrar, "lb", str(src)],
        check=False,
        capture_output=True,
        text=True,
    )
    if list_proc.returncode != 0:
        detail = (list_proc.stderr or list_proc.stdout or "").strip()
        raise ValueError(f"Failed to inspect RAR archive {src}: {detail}")

    pdf_members: list[str] = []
    for line in list_proc.stdout.splitlines():
        member_name = line.strip()
        if member_name.lower().endswith(".pdf"):
            pdf_members.append(member_name)

    if not pdf_members:
        raise ValueError(f"No PDF found inside RAR: {src}")

    extract_dir = src.parent / "_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    member_rows: list[tuple[tuple[int, int, int, int], str, bytes]] = []
    for member_name in pdf_members:
        extract_proc = subprocess.run(
            [unrar, "p", "-inul", str(src), member_name],
            check=False,
            capture_output=True,
        )
        if extract_proc.returncode != 0:
            continue
        pdf_bytes = extract_proc.stdout
        if not pdf_bytes:
            continue
        page_count = _pdf_page_count_from_bytes(pdf_bytes) or 0
        member_stem = Path(member_name).stem
        title_score = _financial_statement_title_quality_score(member_stem)
        preferred_title = 1 if _is_preferred_financial_statement_title(member_stem) else 0
        rank = (
            preferred_title,
            title_score,
            page_count,
            _archive_pdf_language_score(member_name),
        )
        member_rows.append((rank, member_name, pdf_bytes))

    if not member_rows:
        raise ValueError(f"No readable PDF found inside RAR: {src}")

    member_rows.sort(key=lambda item: (item[0], len(item[2])), reverse=True)
    _, member_name, pdf_bytes = member_rows[0]

    safe_member_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(member_name).stem).strip("_")
    target_stem = safe_member_stem or src.stem
    target_pdf = extract_dir / f"{target_stem}.pdf"
    with open(target_pdf, "wb") as wf:
        wf.write(pdf_bytes)
    return target_pdf


def _pdf_page_count_from_bytes(pdf_bytes: bytes) -> int | None:
    try:
        import fitz

        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return int(doc.page_count)
    except Exception:
        return None


def _resolve_source_pdf(
    source_path: str | Path,
    *,
    ticker: str | None = None,
    year: int | None = None,
    allow_manual_updated_files: bool = False,
) -> Path:
    """Resolve a conversion source into a readable PDF path.

    Supports:
    - direct PDF files
    - ZIP archives containing at least one PDF
    - sibling .pdf next to non-pdf source names
    - optional fallback to manually extracted/renamed PDFs in source folder
    """
    src = Path(source_path)

    search_dirs: list[Path] = []
    if src.parent.exists() and src.parent.is_dir():
        search_dirs.append(src.parent)
    extracted_dir = src.parent / "_extracted"
    if extracted_dir.exists() and extracted_dir.is_dir():
        search_dirs.append(extracted_dir)

    if not src.exists():
        if allow_manual_updated_files:
            candidate = None
            for search_dir in search_dirs:
                candidate = _find_manual_pdf_candidate(
                    search_dir,
                    ticker=ticker,
                    year=year,
                    preferred_stem=src.stem,
                )
                if candidate is not None:
                    return candidate
        raise FileNotFoundError(f"Source file not found: {src}")

    if src.suffix.lower() == ".pdf":
        return src

    sibling_pdf = src.with_suffix(".pdf")
    if sibling_pdf.exists():
        return sibling_pdf

    if src.suffix.lower() == ".zip":
        extract_dir = src.parent / "_extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src, "r") as zf:
            nested_archive_members = [
                info
                for info in zf.infolist()
                if not info.is_dir()
                and info.filename.lower().endswith((".zip", ".rar", ".7z"))
            ]
            pdf_members = [
                info
                for info in zf.infolist()
                if not info.is_dir() and info.filename.lower().endswith(".pdf")
            ]
            if not pdf_members:
                raise ValueError(f"No PDF found inside ZIP: {src}")

            # Filter out tiny supplemental PDFs before applying language preference.
            max_size = max(info.file_size for info in pdf_members)
            primary_members = [
                info
                for info in pdf_members
                if info.file_size >= max(1, int(max_size * 0.6))
            ]
            if not primary_members:
                primary_members = pdf_members

            member_rows: list[tuple[tuple[int, int, int, int, int], zipfile.ZipInfo, bytes]] = []
            for info in primary_members:
                with zf.open(info) as rf:
                    pdf_bytes = rf.read()
                page_count = _pdf_page_count_from_bytes(pdf_bytes) or 0
                title_score = _financial_statement_title_quality_score(
                    Path(info.filename).stem
                )
                preferred_title = 1 if _is_preferred_financial_statement_title(
                    Path(info.filename).stem
                ) else 0
                rank = (
                    preferred_title,
                    title_score,
                    page_count,
                    _archive_pdf_language_score(info.filename),
                    info.file_size,
                )
                member_rows.append((rank, info, pdf_bytes))

            member_rows.sort(key=lambda item: item[0], reverse=True)
            best_rank, member, pdf_bytes = member_rows[0]

            largest_nested_archive = max(
                (info.file_size for info in nested_archive_members),
                default=0,
            )
            if (
                largest_nested_archive > int(member.file_size * 2)
                and best_rank[2] <= 3
                and best_rank[0] == 0
            ):
                raise ValueError(
                    "ZIP contains only short attachment PDFs and a larger nested archive; "
                    "extract the main report PDF manually before conversion"
                )

            target_pdf = extract_dir / f"{src.stem}.pdf"
            with open(target_pdf, "wb") as wf:
                wf.write(pdf_bytes)
            return target_pdf

    if src.suffix.lower() == ".rar":
        return _extract_pdf_from_rar(src)

    if allow_manual_updated_files:
        candidate = None
        for search_dir in search_dirs:
            candidate = _find_manual_pdf_candidate(
                search_dir,
                ticker=ticker,
                year=year,
                preferred_stem=src.stem,
            )
            if candidate is not None:
                return candidate

    raise ValueError(f"Unsupported source format for marker conversion: {src}")


def _wait_with_log_heartbeat(
    proc: subprocess.Popen,
    log_file,
    heartbeat_seconds: float = 8.0,
    heartbeat_callback: Callable[[int], None] | None = None,
) -> int:
    """Wait for a subprocess while appending periodic liveness markers to log."""
    started_at = time.monotonic()
    next_heartbeat_at = started_at + heartbeat_seconds

    while True:
        return_code = proc.poll()
        if return_code is not None:
            return int(return_code)

        now = time.monotonic()
        if now >= next_heartbeat_at:
            elapsed = int(now - started_at)
            log_file.write(
                f"[heartbeat] marker is still running... elapsed={elapsed}s\n"
            )
            log_file.flush()
            if heartbeat_callback is not None:
                try:
                    heartbeat_callback(elapsed)
                except Exception:
                    pass
            next_heartbeat_at = now + heartbeat_seconds

        # Keep loop lightweight while preserving frequent heartbeat updates.
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------


def run_job(
    con: duckdb.DuckDBPyConnection,
    job_id: int,
    on_heartbeat: Callable[[int], None] | None = None,
) -> bool:
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

    source_pdf = _resolve_source_pdf(source_path)
    source_path = str(source_pdf)

    # Build output directory per ticker
    out_dir = Path(output_dir) / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare log file
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    job_kind = _job_kind_from_output_dir(output_dir)
    log_path = LOGS_DIR / f"{job_kind}_job_{job_id}.log"

    cmd = [
        *_build_marker_single_command(),
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
            log_file.write(
                f"=== {job_kind.upper()} Job #{job_id}: {ticker} {year} ===\n"
            )
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
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )

            # Store PID for monitoring
            con.execute(
                "UPDATE conversion_jobs SET pid = ? WHERE id = ?",
                [proc.pid, job_id],
            )

            return_code = _wait_with_log_heartbeat(
                proc,
                log_file,
                heartbeat_callback=on_heartbeat,
            )

            log_file.write(f"\n{'=' * 60}\n")
            log_file.write(f"Exit code: {return_code}\n")
            log_file.write(f"Finished: {datetime.now().isoformat()}\n")
            log_file.flush()

        if return_code == 0:
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


def convert_financial_statement_source_to_markdown(
    source_path: str | Path,
    *,
    ticker: str,
    year: int,
    output_dir: str | Path | None = None,
    allow_manual_updated_files: bool = False,
    force_ocr_override: bool = False,
    rerun_reason: str | None = None,
    doc_id: int | None = None,
    on_heartbeat: Callable[[int], None] | None = None,
) -> Path:
    """Convert one known financial-statement source file to markdown."""
    ticker = str(ticker).upper()
    year = int(year)
    source_file = _resolve_source_pdf(
        source_path,
        ticker=ticker,
        year=year,
        allow_manual_updated_files=allow_manual_updated_files,
    )
    out_root = Path(output_dir or FINANCIAL_STATEMENT_MARKDOWN_DIR)
    out_dir = out_root / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_id = str(int(doc_id)) if doc_id is not None else "manual"
    log_path = LOGS_DIR / f"financial_statement_{ticker}_{year}_{log_id}.log"

    cmd = [
        *_build_marker_single_command(),
        str(source_file),
        "--output_dir",
        str(out_dir),
    ]

    needs_ocr = _pdf_needs_ocr(str(source_file))
    should_force_ocr = bool(force_ocr_override) or needs_ocr

    for flag, value in MARKER_EXTRA_ARGS.items():
        # Keep OCR behavior consistent with conversion jobs.
        if flag == "force_ocr" and not should_force_ocr:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{flag}")
        else:
            cmd.extend([f"--{flag}", str(value)])

    with open(log_path, "w") as log_file:
        log_file.write(
            f"=== Financial statement marker conversion: {ticker} {year} (doc_id={log_id}) ===\n"
        )
        log_file.write(
            f"OCR mode: {'force_ocr (manual override)' if force_ocr_override else 'force_ocr (image-based PDF)' if needs_ocr else 'text extraction (text-based PDF)'}\n"
        )
        if rerun_reason:
            log_file.write(f"Rerun reason: {rerun_reason}\n")
        log_file.write(f"Command: {' '.join(cmd)}\n")
        log_file.write(f"Started: {datetime.now().isoformat()}\n")
        log_file.write("=" * 60 + "\n\n")
        log_file.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        return_code = _wait_with_log_heartbeat(
            proc,
            log_file,
            heartbeat_callback=on_heartbeat,
        )

        log_file.write(f"\n{'=' * 60}\n")
        log_file.write(f"Exit code: {return_code}\n")
        log_file.write(f"Finished: {datetime.now().isoformat()}\n")
        log_file.flush()

    if return_code != 0:
        raise RuntimeError(
            "marker_single failed for financial statement "
            f"{ticker}-{year}; see {log_path}"
        )

    return out_dir


def convert_financial_statement_to_markdown(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    year: int,
    output_dir: str | Path | None = None,
    allow_manual_updated_files: bool = False,
    force_ocr_override: bool = False,
    rerun_reason: str | None = None,
    on_heartbeat: Callable[[int], None] | None = None,
) -> Path:
    """Convert latest synced financial-statement PDF for ticker/year to markdown.

    This writes to a dedicated financial-statement markdown directory to avoid clashing with
    annual-report markdown ingestion.
    """
    rows = con.execute(
        """
        SELECT id, raw_path, title
        FROM vietstock_documents
        WHERE ticker = ?
          AND doc_type = '1'
          AND synced_to_raw = TRUE
          AND raw_path IS NOT NULL
          AND raw_path != ''
          AND TRY_CAST(regexp_extract(title, '(\\d{4})', 1) AS INTEGER) = ?
        ORDER BY published_date DESC, id DESC
        """,
        [str(ticker).upper(), int(year)],
    ).fetchall()

    if not rows:
        raise ValueError(
            f"No synced financial statement PDF found for {str(ticker).upper()}-{int(year)}"
        )

    best_candidate: tuple[int, int, int, int, int, str, Path] | None = None
    best_doc_id: int | None = None

    for idx, (candidate_id, candidate_source_path, candidate_title) in enumerate(rows):
        try:
            resolved = _resolve_source_pdf(
                str(candidate_source_path),
                ticker=str(ticker).upper(),
                year=int(year),
                allow_manual_updated_files=allow_manual_updated_files,
            )
        except Exception:
            continue

        size = 0
        try:
            size = int(resolved.stat().st_size)
        except Exception:
            size = 0

        title_score = _financial_statement_title_quality_score(
            str(candidate_title or "")
        )
        preferred_title = 1 if _is_preferred_financial_statement_title(
            str(candidate_title or "")
        ) else 0

        rank = (
            preferred_title,
            title_score,
            size,
            -idx,
            int(candidate_id),
            str(candidate_source_path),
            resolved,
        )
        if best_candidate is None or rank > best_candidate:
            best_candidate = rank
            best_doc_id = int(candidate_id)

    if best_candidate is None or best_doc_id is None:
        raise ValueError(
            f"No readable financial statement source file found for {str(ticker).upper()}-{int(year)}"
        )

    return convert_financial_statement_source_to_markdown(
        best_candidate[-1],
        ticker=str(ticker).upper(),
        year=int(year),
        output_dir=output_dir,
        allow_manual_updated_files=allow_manual_updated_files,
        force_ocr_override=force_ocr_override,
        rerun_reason=rerun_reason,
        doc_id=best_doc_id,
        on_heartbeat=on_heartbeat,
    )


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
