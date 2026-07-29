from __future__ import annotations

import json
import re
import shutil
import subprocess
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from config import RAW_DIR
from database import connection_scope

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

try:
    import rarfile
except Exception:  # pragma: no cover
    rarfile = None  # type: ignore[assignment]


_YEAR_RE = re.compile(r"(20\d{2}|19\d{2})")
_VIETSTOCK_BASE = "https://finance.vietstock.vn"
_VIETSTOCK_DOC_ENDPOINT = f"{_VIETSTOCK_BASE}/data/getdocument"


@dataclass
class DownloadSelection:
    selected_path: Path
    method: str
    reason: str


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _slug_filename(value: str, fallback: str = "document") -> str:
    lowered = _strip_accents(value).lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = lowered.strip("_")
    return lowered or fallback


def _extract_year(text: str) -> int | None:
    m = _YEAR_RE.search(text)
    if not m:
        return None
    try:
        year = int(m.group(1))
        if 1900 <= year <= 2100:
            return year
    except ValueError:
        return None
    return None


def _quality_score(path: Path, ticker: str, year: int | None) -> tuple[int, int]:
    name = _strip_accents(path.name).lower()
    size_score = int(path.stat().st_size // 1024)
    score = 0

    # Prefer vietnamese versions when both VI and EN files are in an archive.
    if any(k in name for k in ("bao_cao", "thuong_nien", "kiem_toan", "hop_nhat", "vn")):
        score += 50
    if any(k in name for k in ("english", "en_", "_en", "eng")):
        score -= 15

    if ticker.lower() in name:
        score += 20
    if year and str(year) in name:
        score += 20

    # Down-rank annex-like files.
    if any(k in name for k in ("phu_luc", "appendix", "notice", "thong_bao", "dieu_chinh")):
        score -= 25

    # Favor pdf payloads likely to be the main report.
    if path.suffix.lower() == ".pdf":
        score += 40

    return (score, size_score)


def _choose_with_llm(candidates: list[Path], ticker: str, year: int | None) -> tuple[int | None, str]:
    if OpenAI is None:
        return None, "OpenAI client unavailable"

    client = OpenAI()
    payload = [
        {
            "index": i,
            "name": p.name,
            "size_bytes": p.stat().st_size,
            "suffix": p.suffix.lower(),
        }
        for i, p in enumerate(candidates)
    ]
    prompt = {
        "task": "choose_primary_annual_report_file",
        "ticker": ticker,
        "year": year,
        "instruction": (
            "Pick the single best annual report PDF candidate. Prefer Vietnamese main report, "
            "avoid annex/notice/adjustment files. Return JSON: {index:int, reason:str}."
        ),
        "candidates": payload,
    }
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=json.dumps(prompt, ensure_ascii=False),
        temperature=0,
    )
    text = getattr(resp, "output_text", "") or ""
    try:
        obj = json.loads(text)
        idx = int(obj.get("index"))
        if 0 <= idx < len(candidates):
            return idx, str(obj.get("reason") or "llm selected")
    except Exception:
        pass
    return None, "LLM did not return valid selection"


def select_best_candidate(
    candidates: list[Path],
    ticker: str,
    year: int | None,
    use_llm: bool = False,
) -> DownloadSelection:
    if not candidates:
        raise ValueError("No candidate files to select")

    ranked = sorted(candidates, key=lambda p: _quality_score(p, ticker, year), reverse=True)

    if use_llm and len(ranked) > 1:
        idx, llm_reason = _choose_with_llm(ranked, ticker, year)
        if idx is not None:
            return DownloadSelection(
                selected_path=ranked[idx],
                method="llm",
                reason=llm_reason,
            )

    best = ranked[0]
    return DownloadSelection(
        selected_path=best,
        method="heuristic",
        reason="Selected by score(size, language hints, ticker/year, and exclusions)",
    )


def _resolve_archive(
    archive_path: Path,
    ticker: str,
    year: int | None,
    use_llm: bool,
) -> DownloadSelection:
    extract_root = archive_path.parent / "_extracted" / archive_path.stem
    extract_root.mkdir(parents=True, exist_ok=True)

    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_root)
    elif archive_path.suffix.lower() == ".rar":
        _extract_rar(archive_path, extract_root)
    else:
        raise RuntimeError(f"Unsupported archive extension: {archive_path.suffix}")

    candidates = [
        p
        for p in extract_root.rglob("*")
        if p.is_file() and p.suffix.lower() == ".pdf" and p.stat().st_size > 50_000
    ]
    if not candidates:
        raise RuntimeError("No suitable PDF candidates found after archive extraction")

    return select_best_candidate(candidates, ticker=ticker, year=year, use_llm=use_llm)


def _extract_rar(archive_path: Path, output_dir: Path) -> None:
    if rarfile is not None:
        try:
            with rarfile.RarFile(str(archive_path)) as rf:
                rf.extractall(path=str(output_dir))
            return
        except Exception:
            pass

    unrar_cmd = shutil.which("unrar")
    bsdtar_cmd = shutil.which("bsdtar")

    if unrar_cmd:
        subprocess.run(
            [unrar_cmd, "x", "-o+", str(archive_path), str(output_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        return

    if bsdtar_cmd:
        subprocess.run(
            [bsdtar_cmd, "-xf", str(archive_path), "-C", str(output_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        return

    raise RuntimeError(
        "Cannot extract RAR: install unrar or bsdtar, or add python rarfile backend"
    )


def _extract_vietstock_token(session: requests.Session, ticker: str = "VNM") -> str | None:
    profile_url = f"{_VIETSTOCK_BASE}/{ticker.lower()}-ho-so-doanh-nghiep.htm"
    resp = session.get(profile_url, timeout=40)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    token_input = soup.find("input", {"name": "__RequestVerificationToken"})
    if token_input and token_input.get("value"):
        return str(token_input.get("value"))
    return None


def _normalize_document_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("data", "Data", "rows", "Rows", "items", "Items"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                return [row for row in candidate if isinstance(row, dict)]
        if isinstance(payload.get("documents"), list):
            return [row for row in payload["documents"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _row_int(row: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        try:
            return int(str(value).strip())
        except Exception:
            continue
    return None


def _row_text(row: dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def fetch_documents_for_ticker(
    ticker: str,
    doc_type: str = "2",
    max_pages: int = 30,
    page_size: int = 20,
) -> list[dict[str, Any]]:
    """Fetch document rows for a ticker with Vietstock's paginated endpoint."""
    ticker = ticker.strip().upper()
    if not ticker:
        return []

    session = requests.Session()
    token = _extract_vietstock_token(session, ticker=ticker)

    out: list[dict[str, Any]] = []
    seen_keys: set[tuple[int | None, int | None, str]] = set()

    for page in range(1, max_pages + 1):
        form: dict[str, Any] = {
            "code": ticker,
            "type": doc_type,
            "page": page,
            "pageSize": page_size,
        }
        if token:
            form["__RequestVerificationToken"] = token

        resp = session.post(_VIETSTOCK_DOC_ENDPOINT, data=form, timeout=45)
        resp.raise_for_status()
        payload = resp.json()
        rows = _normalize_document_rows(payload)
        if not rows:
            break

        before = len(out)
        for row in rows:
            doc_id = _row_int(row, "id", "ID", "documentId")
            info_id = _row_int(row, "file_info_id", "fileInfoId", "fileId")
            title = _row_text(row, "title", "Title", "name", "Name", default="")
            dedup_key = (doc_id, info_id, title)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            out.append(
                {
                    "id": doc_id,
                    "ticker": ticker,
                    "doc_type": doc_type,
                    "title": title,
                    "full_name": _row_text(row, "full_name", "fullName", "FullName"),
                    "source": _row_text(row, "source", "Source"),
                    "published_date": _row_text(row, "published_date", "publishedDate", "date"),
                    "file_url": _row_text(row, "file_url", "fileUrl", "url", "downloadUrl"),
                    "file_info_id": info_id,
                }
            )

        if len(out) == before:
            break
        if len(rows) < page_size:
            break

    return out


def sync_documents_for_ticker(ticker: str, doc_type: str = "2") -> int:
    rows = fetch_documents_for_ticker(ticker=ticker, doc_type=doc_type)
    if not rows:
        return 0

    upserted = 0
    with connection_scope() as con:
        for row in rows:
            doc_id = row.get("id")
            if doc_id is None:
                continue
            con.execute(
                """
                INSERT INTO vietstock_documents (
                    id,
                    ticker,
                    doc_type,
                    title,
                    full_name,
                    source,
                    published_date,
                    file_url,
                    file_info_id,
                    synced_to_raw,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, FALSE, CURRENT_TIMESTAMP)
                ON CONFLICT (id) DO UPDATE SET
                    ticker = EXCLUDED.ticker,
                    doc_type = EXCLUDED.doc_type,
                    title = EXCLUDED.title,
                    full_name = EXCLUDED.full_name,
                    source = EXCLUDED.source,
                    published_date = EXCLUDED.published_date,
                    file_url = EXCLUDED.file_url,
                    file_info_id = EXCLUDED.file_info_id,
                    updated_at = CURRENT_TIMESTAMP
                """,
                [
                    doc_id,
                    row.get("ticker"),
                    row.get("doc_type"),
                    row.get("title"),
                    row.get("full_name"),
                    row.get("source"),
                    row.get("published_date"),
                    row.get("file_url"),
                    row.get("file_info_id"),
                ],
            )
            upserted += 1
    return upserted


def sync_documents_for_all_companies(doc_type: str = "2") -> dict[str, int]:
    done = 0
    failed = 0
    rows = 0
    with connection_scope() as con:
        companies = [str(r[0]) for r in con.execute("SELECT ticker FROM companies ORDER BY ticker").fetchall()]

    for ticker in companies:
        try:
            rows += sync_documents_for_ticker(ticker=ticker, doc_type=doc_type)
            done += 1
        except Exception:
            failed += 1
    return {"tickers_done": done, "tickers_failed": failed, "rows": rows}


def upsert_document_stub(
    *,
    doc_id: int,
    ticker: str,
    doc_type: str,
    title: str,
    file_url: str,
    published_date: str | None = None,
) -> None:
    """Insert or update a document row for manual/prototype testing."""
    with connection_scope() as con:
        con.execute(
            """
            INSERT INTO vietstock_documents (
                id, ticker, doc_type, title, published_date, file_url, synced_to_raw, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, FALSE, CURRENT_TIMESTAMP)
            ON CONFLICT (id) DO UPDATE SET
                ticker = EXCLUDED.ticker,
                doc_type = EXCLUDED.doc_type,
                title = EXCLUDED.title,
                published_date = EXCLUDED.published_date,
                file_url = EXCLUDED.file_url,
                updated_at = CURRENT_TIMESTAMP
            """,
            [doc_id, ticker.upper(), doc_type, title, published_date, file_url],
        )


def list_unsynced_documents(limit: int = 200) -> list[dict[str, Any]]:
    with connection_scope() as con:
        rows = con.execute(
            """
            SELECT id, ticker, doc_type, COALESCE(title, ''), COALESCE(file_url, ''), COALESCE(published_date, '')
            FROM vietstock_documents
            WHERE COALESCE(synced_to_raw, FALSE) = FALSE
              AND COALESCE(file_url, '') <> ''
            ORDER BY ticker, published_date DESC, id DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()

    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": int(r[0]),
                "ticker": str(r[1]),
                "doc_type": str(r[2]),
                "title": str(r[3]),
                "file_url": str(r[4]),
                "published_date": str(r[5]),
            }
        )
    return out


def _download_file(url: str, destination: Path, timeout_seconds: int = 90) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout_seconds) as resp:
        resp.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if chunk:
                    handle.write(chunk)


def _target_pdf_name(ticker: str, year: int | None, title: str) -> str:
    if year:
        return f"{ticker.upper()}_annual_report_{year}.pdf"
    return f"{ticker.upper()}_{_slug_filename(title, 'annual_report')}.pdf"


def download_document_to_raw(doc: dict[str, Any], use_llm_selection: bool = False) -> DownloadSelection:
    ticker = str(doc.get("ticker") or "").strip().upper()
    if not ticker:
        raise ValueError("Document ticker is required")

    title = str(doc.get("title") or "document")
    file_url = str(doc.get("file_url") or "").strip()
    if not file_url:
        raise ValueError("Document file_url is required")

    year = _extract_year(title) or _extract_year(str(doc.get("published_date") or ""))
    ticker_dir = RAW_DIR / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    parsed = urlparse(file_url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in {".pdf", ".zip", ".rar"}:
        suffix = ".pdf"

    downloaded_name = f"download_{doc.get('id')}{suffix}"
    downloaded_path = ticker_dir / downloaded_name
    _download_file(file_url, downloaded_path)

    if downloaded_path.suffix.lower() == ".pdf":
        final_path = ticker_dir / _target_pdf_name(ticker, year, title)
        shutil.copy2(downloaded_path, final_path)
        return DownloadSelection(final_path, "direct", "Direct PDF download")

    selection = _resolve_archive(
        downloaded_path,
        ticker=ticker,
        year=year,
        use_llm=use_llm_selection,
    )
    final_path = ticker_dir / _target_pdf_name(ticker, year, title)
    shutil.copy2(selection.selected_path, final_path)
    return DownloadSelection(
        selected_path=final_path,
        method=selection.method,
        reason=selection.reason,
    )


def mark_document_synced(doc_id: int, raw_path: Path, method: str, reason: str) -> None:
    with connection_scope() as con:
        con.execute(
            """
            UPDATE vietstock_documents
            SET
                synced_to_raw = TRUE,
                raw_path = ?,
                selection_method = ?,
                selection_reason = ?,
                sync_error = NULL,
                last_downloaded_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            [str(raw_path), method, reason, doc_id],
        )


def mark_document_failed(doc_id: int, error: str) -> None:
    with connection_scope() as con:
        con.execute(
            """
            UPDATE vietstock_documents
            SET
                synced_to_raw = FALSE,
                sync_error = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            [error[:800], doc_id],
        )


def download_all_unsynced(limit: int = 200, use_llm_selection: bool = False) -> dict[str, int]:
    done = 0
    failed = 0
    docs = list_unsynced_documents(limit=limit)
    for doc in docs:
        try:
            selection = download_document_to_raw(doc, use_llm_selection=use_llm_selection)
            mark_document_synced(
                doc_id=int(doc["id"]),
                raw_path=selection.selected_path,
                method=selection.method,
                reason=selection.reason,
            )
            done += 1
        except Exception as exc:
            mark_document_failed(int(doc["id"]), str(exc))
            failed += 1
    return {"done": done, "failed": failed, "total": len(docs)}
