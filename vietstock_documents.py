import os
import re
import unicodedata
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from typing import Callable, Optional

import duckdb
import pandas as pd

from config import RAW_DIR

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
)

DOC_TYPE_ANNUAL_REPORT = "2"
DOC_TYPE_AUDITED_CONSOLIDATED_FS = "1"

DOC_TYPE_LABELS: dict[str, str] = {
    DOC_TYPE_AUDITED_CONSOLIDATED_FS: "Audited Consolidated Financial Statements",
    DOC_TYPE_ANNUAL_REPORT: "Annual Reports",
}

DEFAULT_PAGE_SIZE = 20


def _extract_year_from_title(title: str) -> int | None:
    """Extract 4-digit year from a document title."""
    m = re.search(r"(\d{4})", str(title or ""))
    if not m:
        return None
    year = int(m.group(1))
    return year if 2000 <= year <= 2100 else None


def _bctc_title_quality_score(title: str) -> int:
    """Score BCTC titles so full audited reports rank above adjustment notices."""
    text = _normalize_vi_text(title or "")
    score = 0

    # De-prioritize adjustment/notice style attachments.
    for marker in (
        "dieu chinh",
        "dinh chinh",
        "giai trinh",
        "bo sung",
        "thong bao",
        "phu luc",
    ):
        if marker in text:
            score -= 20

    # Prefer likely full audited financial statements.
    for marker in (
        "bao cao tai chinh",
        "bctc",
        "hop nhat",
        "kiem toan",
    ):
        if marker in text:
            score += 4

    return score


def _normalize_vi_text(value: str) -> str:
    """Normalize Vietnamese text for accent-insensitive matching."""
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    # Handle Vietnamese specific letter after accent stripping.
    return text.replace("đ", "d")


def _is_audited_annual_doc(doc: dict) -> bool:
    """Keep audited annual financial statements (consolidated or standalone)."""
    title = str(doc.get("Title") or doc.get("R_Title") or "")
    full_name = str(doc.get("FullName") or doc.get("R_FullName") or "")
    text = _normalize_vi_text(f"{title} {full_name}")

    has_financial_report = (
        "bao cao tai chinh" in text or "bctc" in text
    )
    has_audited = "kiem toan" in text
    has_yearly = " nam " in f" {text} " or any(
        str(y) in text for y in range(2000, 2101)
    )

    # Exclude interim/reviewed/parent-company docs.
    blocked_keywords = [
        "soat xet",
        "quy",
        "6 thang",
        "9 thang",
        "cong ty me",
        "rieng",
    ]
    has_blocked = any(k in text for k in blocked_keywords)

    return (
        has_financial_report
        and has_audited
        and has_yearly
        and not has_blocked
    )


def _list_documents_page(
    code: str = "VNM",
    doc_type: str = DOC_TYPE_ANNUAL_REPORT,
    *,
    page: int = 1,
    year: int | None = None,
) -> list[dict]:
    """List available PDF documents for a given stock code on Vietstock.

    Args:
        code: Stock ticker symbol (e.g. "VNM").
        doc_type: Vietstock document type identifier.
        page: 1-based page index for Vietstock listing API.
        year: Optional year filter sent to Vietstock.

    Returns:
        Parsed JSON response from the Vietstock API.
    """
    session = requests.Session()

    # 1) Load the documents page to obtain cookies and the CSRF token
    page_url = (
        f"https://finance.vietstock.vn/{code}/tai-tai-lieu.htm?doctype={doc_type}"
    )
    headers_page = {"User-Agent": USER_AGENT}

    response = session.get(page_url, headers=headers_page)
    response.raise_for_status()

    # 2) Extract __RequestVerificationToken from the HTML form
    soup = BeautifulSoup(response.text, "html.parser")
    token_input = soup.find("input", {"name": "__RequestVerificationToken"})

    if not token_input:
        raise RuntimeError(
            "Could not find __RequestVerificationToken in the page HTML. "
            "Vietstock may have changed its page structure."
        )

    verification_token = token_input["value"]

    # 3) POST to the API endpoint using the same session (cookies included)
    api_url = "https://finance.vietstock.vn/data/getdocument"

    headers_post = {
        "User-Agent": USER_AGENT,
        "Accept": "*/*",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Origin": "https://finance.vietstock.vn",
        "Referer": page_url,
        "X-Requested-With": "XMLHttpRequest",
    }

    data = {
        "code": code,
        "type": doc_type,
        "page": str(max(1, int(page))),
        "__RequestVerificationToken": verification_token,
    }
    if year is not None:
        data["year"] = str(int(year))

    post_response = session.post(api_url, headers=headers_post, data=data)
    post_response.raise_for_status()

    # The response might be plain text / HTML instead of JSON
    content_type = post_response.headers.get("Content-Type", "")
    raw_text = post_response.text.strip()

    # If response is empty or not JSON-like, return empty
    if not raw_text or (not raw_text.startswith(("{", "["))):
        return []

    result = post_response.json()

    # The API may return a list directly, or wrap it in a dict (e.g. {"Data": [...]})
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        # Try every key that holds a list
        for key, val in result.items():
            if (
                isinstance(val, list)
                and len(val) > 0
                and isinstance(val[0], dict)
            ):
                return val
        # Maybe the dict itself is a single document
        return [result]
    return []


def list_documents(
    code: str = "VNM",
    doc_type: str = DOC_TYPE_ANNUAL_REPORT,
    *,
    year: int | None = None,
) -> list[dict]:
    """List available documents for one ticker and one document type.

    Uses Vietstock pagination and returns a de-duplicated flat list.
    """
    first_page = _list_documents_page(code=code, doc_type=doc_type, page=1, year=year)
    if not first_page:
        return []

    # Vietstock returns TotalRow on each item when available.
    total_rows = 0
    total_raw = first_page[0].get("TotalRow") if isinstance(first_page[0], dict) else 0
    try:
        total_rows = int(total_raw or 0)
    except (TypeError, ValueError):
        total_rows = 0

    if total_rows > 0:
        total_pages = max(1, (total_rows + DEFAULT_PAGE_SIZE - 1) // DEFAULT_PAGE_SIZE)
    else:
        total_pages = 1

    docs = list(first_page)
    for p in range(2, total_pages + 1):
        page_docs = _list_documents_page(
            code=code,
            doc_type=doc_type,
            page=p,
            year=year,
        )
        if not page_docs:
            break
        docs.extend(page_docs)

    # De-duplicate by FileInfoID or title+url fallback.
    deduped: list[dict] = []
    seen: set[str] = set()
    for doc in docs:
        file_info = doc.get("FileInfoID")
        if file_info is not None and str(file_info).strip() != "":
            key = f"fid:{file_info}"
        else:
            key = f"fallback:{doc.get('Title', '')}|{doc.get('Url', '')}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)

    if str(doc_type) == DOC_TYPE_AUDITED_CONSOLIDATED_FS:
        deduped = [
            doc for doc in deduped if _is_audited_annual_doc(doc)
        ]

    return deduped


def _extract_doc_fields(doc: dict, index: int = 0) -> dict | None:
    """Extract normalised fields from a raw Vietstock document dict."""
    # Try multiple possible key patterns (case-insensitive search)
    doc_id = None
    for key in doc:
        key_lower = key.lower()
        if "documentid" in key_lower or key_lower == "id":
            val = doc[key]
            if val is not None:
                doc_id = val
                break

    # Last resort: generate a stable ID from the document content
    if doc_id is None:
        doc_id = abs(hash(str(sorted(doc.items())))) % (10**9)

    def _find(doc: dict, *candidates: str) -> str:
        """Case-insensitive key lookup."""
        lower_map = {k.lower(): k for k in doc}
        for c in candidates:
            real_key = lower_map.get(c.lower())
            if real_key and doc.get(real_key):
                return str(doc[real_key])
        return ""

    return {
        "id": doc_id,
        "title": _find(doc, "R_Title", "Title", "title"),
        "full_name": _find(
            doc, "R_FullName", "FullName", "fullname", "full_name"
        ),
        "source": _find(doc, "R_Source", "Source", "source"),
        "published_date": _find(
            doc,
            "R_DateTime",
            "DateTime",
            "datetime",
            "date",
            "published_date",
            "UpdateTime",
            "LastUpdate",
        ),
        "file_url": _find(
            doc, "R_FileURL", "FileURL", "fileurl", "file_url", "Url", "url"
        ),
        "file_info_id": _find(doc, "FileInfoID", "R_FileInfoID", "fileinfoid"),
    }


def fetch_documents_preview(
    code: str = "VNM",
    doc_type: str = DOC_TYPE_ANNUAL_REPORT,
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """Fetch documents and return a preview DataFrame, extracted rows, and raw list.

    Returns:
        (preview_df, extracted_rows, raw_docs)
    """
    raw_docs = list_documents(code=code, doc_type=doc_type)

    if not raw_docs:
        return pd.DataFrame(), [], raw_docs

    rows = []
    for i, doc in enumerate(raw_docs):
        extracted = _extract_doc_fields(doc, index=i)
        if extracted:
            rows.append(extracted)

    if rows:
        return pd.DataFrame(rows), rows, raw_docs

    # Fallback: just dump raw dicts so the user can see what the API returned
    return pd.DataFrame(raw_docs), [], raw_docs


def save_extracted_to_db(
    con: duckdb.DuckDBPyConnection,
    extracted_rows: list[dict],
    code: str = "VNM",
    doc_type: str = DOC_TYPE_ANNUAL_REPORT,
) -> int:
    """Save pre-extracted document rows into the database.

    Uses INSERT ... ON CONFLICT to merge duplicates based on id.
    Also skips rows that would duplicate (ticker, doc_type, file_info_id).

    Returns:
        Number of rows inserted or updated.
    """
    from database import ensure_company

    ensure_company(con, code)

    count = 0
    for fields in extracted_rows:
        file_info_id = (
            int(fields["file_info_id"]) if fields.get("file_info_id") else None
        )

        # Skip if a record with same ticker + doc_type + file_info_id already exists
        if file_info_id is not None:
            existing_dup = con.execute(
                """
                SELECT id FROM vietstock_documents
                WHERE ticker = ? AND doc_type = ? AND file_info_id = ? AND id != ?
                """,
                [code.upper(), doc_type, file_info_id, fields["id"]],
            ).fetchone()
            if existing_dup:
                # Update the existing record instead of inserting a duplicate
                con.execute(
                    """
                    UPDATE vietstock_documents
                    SET title = ?, full_name = ?, source = ?, published_date = ?,
                        file_url = ?
                    WHERE id = ?
                    """,
                    [
                        fields["title"],
                        fields["full_name"],
                        fields["source"],
                        fields["published_date"],
                        fields["file_url"],
                        existing_dup[0],
                    ],
                )
                count += 1
                continue

        existing = con.execute(
            "SELECT id FROM vietstock_documents WHERE id = ?", [fields["id"]]
        ).fetchone()

        if existing:
            con.execute(
                """
                UPDATE vietstock_documents
                SET title = ?, full_name = ?, source = ?, published_date = ?,
                    file_url = ?, file_info_id = ?
                WHERE id = ?
                """,
                [
                    fields["title"],
                    fields["full_name"],
                    fields["source"],
                    fields["published_date"],
                    fields["file_url"],
                    file_info_id,
                    fields["id"],
                ],
            )
        else:
            con.execute(
                """
                INSERT INTO vietstock_documents
                    (id, ticker, doc_type, title, full_name, source,
                     published_date, file_url, file_info_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    fields["id"],
                    code.upper(),
                    doc_type,
                    fields["title"],
                    fields["full_name"],
                    fields["source"],
                    fields["published_date"],
                    fields["file_url"],
                    file_info_id,
                ],
            )
        count += 1

    return count


def save_documents_to_db(
    con: duckdb.DuckDBPyConnection,
    raw_docs: list[dict],
    code: str = "VNM",
    doc_type: str = DOC_TYPE_ANNUAL_REPORT,
) -> int:
    """Save previously fetched documents into the database.

    Returns:
        Number of rows inserted or updated.
    """
    extracted_rows = []
    for doc in raw_docs:
        fields = _extract_doc_fields(doc)
        if fields:
            extracted_rows.append(fields)
    return save_extracted_to_db(con, extracted_rows, code, doc_type)


def sync_documents_to_db(
    con: duckdb.DuckDBPyConnection,
    code: str = "VNM",
    doc_type: str = DOC_TYPE_ANNUAL_REPORT,
) -> int:
    """Fetch documents from Vietstock and upsert them into the database.

    Returns:
        Number of rows inserted or updated.
    """
    raw_docs = list_documents(code=code, doc_type=doc_type)
    if not raw_docs:
        return 0
    return save_documents_to_db(con, raw_docs, code=code, doc_type=doc_type)


def download_document_to_raw(
    con: duckdb.DuckDBPyConnection,
    doc_id: int,
) -> Path:
    """Download a single document PDF to data/raw/ and mark it as synced.

    Returns:
        Path to the downloaded file.
    """
    row = con.execute(
        "SELECT ticker, file_url, title FROM vietstock_documents WHERE id = ?",
        [doc_id],
    ).fetchone()

    if not row:
        raise ValueError(f"Document {doc_id} not found in database.")

    ticker, file_url, title = row

    if not file_url:
        raise ValueError(f"Document {doc_id} has no file URL.")

    # Build download path: data/raw/<TICKER>/<safe_name>.<ext>
    dest_dir = RAW_DIR / ticker
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Sanitise filename from title or fall back to doc_id
    safe_name = "".join(
        c if c.isalnum() or c in " _-" else "_" for c in (title or str(doc_id))
    )

    # Download
    resp = requests.get(
        file_url, headers={"User-Agent": USER_AGENT}, stream=True
    )
    resp.raise_for_status()

    # Detect actual file type from Content-Type header or URL
    content_type = (
        resp.headers.get("Content-Type", "").lower().split(";")[0].strip()
    )
    _MIME_TO_EXT = {
        "application/pdf": ".pdf",
        "application/zip": ".zip",
        "application/x-zip-compressed": ".zip",
        "application/x-rar-compressed": ".rar",
        "application/vnd.rar": ".rar",
        "application/x-7z-compressed": ".7z",
        "application/octet-stream": None,  # ambiguous, check URL
    }
    ext = _MIME_TO_EXT.get(content_type)
    if ext is None:
        # Fall back to URL extension
        from urllib.parse import urlparse

        url_path = urlparse(file_url).path.lower()
        for known_ext in (
            ".zip",
            ".rar",
            ".7z",
            ".pdf",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
        ):
            if url_path.endswith(known_ext):
                ext = known_ext
                break
        else:
            ext = ".pdf"  # default

    dest_path = dest_dir / f"{safe_name}{ext}"

    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    # Update database
    con.execute(
        "UPDATE vietstock_documents SET synced_to_raw = TRUE, raw_path = ? WHERE id = ?",
        [str(dest_path), doc_id],
    )

    return dest_path


def delete_duplicates_by_file_info_id(con: duckdb.DuckDBPyConnection) -> int:
    """Delete duplicate vietstock_documents rows keeping lowest id per file_info_id.

    Returns:
        Number of rows deleted.
    """
    cursor = con.execute(
        """
        DELETE FROM vietstock_documents
        WHERE id IN (
            SELECT d.id
            FROM vietstock_documents d
            INNER JOIN (
                SELECT file_info_id, MIN(id) AS keep_id
                FROM vietstock_documents
                WHERE file_info_id IS NOT NULL
                GROUP BY file_info_id
                HAVING COUNT(*) > 1
            ) dup ON d.file_info_id = dup.file_info_id AND d.id != dup.keep_id
        )
        """
    )

    # For DELETE, DuckDB does not return a result set; use rowcount instead.
    return int(getattr(cursor, "rowcount", 0) or 0)


import re as _re


def _extract_year_from_title(title: str) -> int | None:
    """Extract the 4-digit year from a document title."""
    m = _re.search(r"(\d{4})", title or "")
    return int(m.group(1)) if m else None


def fetch_all_companies_documents(
    con: duckdb.DuckDBPyConnection,
    doc_type: str = DOC_TYPE_ANNUAL_REPORT,
    tickers: list[str] | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    on_progress: Optional[
        Callable[[str, int, Optional[Exception]], None]
    ] = None,
) -> dict[str, int]:
    """Fetch documents from Vietstock for all companies in the database.

    Args:
        con: DuckDB connection.
        doc_type: Vietstock document type.
        tickers: Optional list of tickers to fetch. If None, fetches all companies.
        start_year: Optional — only save documents with year >= start_year.
        end_year: Optional — only save documents with year <= end_year.
        on_progress: Optional callback(ticker, count, error) for progress reporting.

    Returns:
        Dict mapping ticker -> number of documents saved.
    """
    from database import ensure_company

    if tickers is None:
        tickers_rows = con.execute(
            "SELECT ticker FROM companies ORDER BY ticker"
        ).fetchall()
        ticker_list = [t[0] for t in tickers_rows]
    else:
        ticker_list = tickers

    results = {}
    for ticker in ticker_list:
        try:
            years: list[int | None]
            if start_year is not None and end_year is not None:
                lower = min(start_year, end_year)
                upper = max(start_year, end_year)
                years = list(range(lower, upper + 1))
            elif start_year is not None:
                years = [start_year]
            elif end_year is not None:
                years = [end_year]
            else:
                years = [None]

            raw_docs: list[dict] = []
            for y in years:
                raw_docs.extend(list_documents(code=ticker, doc_type=doc_type, year=y))

            # De-duplicate after combining years.
            unique_docs: list[dict] = []
            seen_keys: set[str] = set()
            for doc in raw_docs:
                file_info = doc.get("FileInfoID")
                if file_info is not None and str(file_info).strip() != "":
                    key = f"fid:{file_info}"
                else:
                    key = f"fallback:{doc.get('Title', '')}|{doc.get('Url', '')}"
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                unique_docs.append(doc)
            raw_docs = unique_docs

            if not raw_docs:
                results[ticker] = 0
                if on_progress:
                    on_progress(ticker, 0, None)
                continue

            extracted = []
            for i, doc in enumerate(raw_docs):
                fields = _extract_doc_fields(doc, index=i)
                if fields:
                    # Apply year filter based on title
                    year = _extract_year_from_title(fields.get("title", ""))
                    if year is not None:
                        if start_year is not None and year < start_year:
                            continue
                        if end_year is not None and year > end_year:
                            continue
                    extracted.append(fields)

            count = save_extracted_to_db(
                con, extracted, code=ticker, doc_type=doc_type
            )
            results[ticker] = count
            if on_progress:
                on_progress(ticker, count, None)
        except Exception as e:
            results[ticker] = 0
            if on_progress:
                on_progress(ticker, 0, e)

    return results


def download_all_unsynced(
    con: duckdb.DuckDBPyConnection,
    doc_type: str | None = None,
    on_progress: Optional[
        Callable[[str, str, Optional[str], Optional[Exception]], None]
    ] = None,
) -> dict[str, list[str]]:
    """Download all unsynced documents to the raw layer.

    Args:
        con: DuckDB connection.
        on_progress: Optional callback(ticker, title, path, error) for progress reporting.

    Returns:
        Dict mapping ticker -> list of downloaded file paths.
    """
    sql = (
        "SELECT id, ticker, title, published_date "
        "FROM vietstock_documents "
        "WHERE synced_to_raw = FALSE AND file_url IS NOT NULL AND file_url != '' "
    )
    params: list[str] = []
    if doc_type:
        sql += "AND doc_type = ? "
        params.append(str(doc_type))
    sql += "ORDER BY ticker, published_date DESC, id DESC"
    rows = con.execute(sql, params).fetchall()

    if str(doc_type or "") == DOC_TYPE_AUDITED_CONSOLIDATED_FS:
        # Prefer full audited BCTC over adjustment notices within the same ticker/year.
        ranked_rows: list[tuple[int, str, str, str | None]] = []
        by_group: dict[tuple[str, int], list[tuple[int, str, str, str | None]]] = {}
        no_year_rows: list[tuple[int, str, str, str | None]] = []

        for row in rows:
            doc_id, ticker, title, published_date = row
            year = _extract_year_from_title(str(title or ""))
            item = (int(doc_id), str(ticker), str(title or ""), published_date)
            if year is None:
                no_year_rows.append(item)
            else:
                by_group.setdefault((str(ticker), int(year)), []).append(item)

        for group_rows in by_group.values():
            group_rows.sort(
                key=lambda r: (
                    _bctc_title_quality_score(r[2]),
                    str(r[3] or ""),
                    r[0],
                ),
                reverse=True,
            )
            ranked_rows.extend(group_rows)

        # Keep deterministic behavior for rows with missing year.
        no_year_rows.sort(key=lambda r: (r[1], str(r[3] or ""), r[0]), reverse=True)
        rows = ranked_rows + no_year_rows

    results: dict[str, list[str]] = {}
    for doc_id, ticker, title, _published_date in rows:
        try:
            path = download_document_to_raw(con, doc_id)
            results.setdefault(ticker, []).append(str(path))
            if on_progress:
                on_progress(ticker, title, str(path), None)
        except Exception as e:
            results.setdefault(ticker, [])
            if on_progress:
                on_progress(ticker, title, None, e)

    return results


if __name__ == "__main__":
    import json

    documents = list_documents()
    print(json.dumps(documents, indent=2, ensure_ascii=False))
