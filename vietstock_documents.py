import os
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


def list_documents(code: str = "VNM", doc_type: str = "2") -> list[dict]:
    """List available PDF documents for a given stock code on Vietstock.

    Args:
        code: Stock ticker symbol (e.g. "VNM").
        doc_type: Document type identifier ("2" = annual reports).

    Returns:
        Parsed JSON response from the Vietstock API.
    """
    session = requests.Session()

    # 1) Load the documents page to obtain cookies and the CSRF token
    page_url = (
        f"https://finance.vietstock.vn/{code}/documents.htm?doctype={doc_type}"
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
        "__RequestVerificationToken": verification_token,
    }

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
            doc, "R_DateTime", "DateTime", "datetime", "date", "published_date"
        ),
        "file_url": _find(
            doc, "R_FileURL", "FileURL", "fileurl", "file_url", "Url", "url"
        ),
        "file_info_id": _find(doc, "FileInfoID", "R_FileInfoID", "fileinfoid"),
    }


def fetch_documents_preview(
    code: str = "VNM", doc_type: str = "2"
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
    doc_type: str = "2",
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
    doc_type: str = "2",
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
    doc_type: str = "2",
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
    doc_type: str = "2",
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
        doc_type: Document type ("2" = annual reports).
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
            raw_docs = list_documents(code=ticker, doc_type=doc_type)
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
    rows = con.execute(
        """
        SELECT id, ticker, title
        FROM vietstock_documents
        WHERE synced_to_raw = FALSE AND file_url IS NOT NULL AND file_url != ''
        ORDER BY ticker, published_date DESC
        """
    ).fetchall()

    results: dict[str, list[str]] = {}
    for doc_id, ticker, title in rows:
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
