"""
Fetch financial statement data from VNDirect API and store in DuckDB.

Data sources (api-finfo.vndirect.com.vn/v4):
  - /financial_models   – metadata describing each line-item code
  - /financial_statements – annual financial statement values per company
  - /ratios             – financial ratios per company per report date
"""

from __future__ import annotations

import logging
from typing import Generator

import duckdb
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api-finfo.vndirect.com.vn/v4"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
    ),
}
_PAGE_SIZE = 5000
_TIMEOUT = 120


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fiscal_dates(start_year: int, end_year: int) -> str:
    """Comma-separated fiscal dates ``YYYY-12-31`` for a year range."""
    return ",".join(f"{y}-12-31" for y in range(start_year, end_year + 1))


def _paginated_get(url: str) -> Generator[list[dict], None, None]:
    """Yield pages of ``data`` from a paginated VNDirect endpoint."""
    page = 1
    while True:
        sep = "&" if "?" in url else "?"
        paged_url = f"{url}{sep}page={page}"
        resp = requests.get(paged_url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data", [])
        if not data:
            break
        yield data
        total_pages = payload.get("totalPages", 1)
        if page >= total_pages:
            break
        page += 1


# ---------------------------------------------------------------------------
# Stock Listing
# ---------------------------------------------------------------------------


def fetch_stocks(con: duckdb.DuckDBPyConnection) -> int:
    """Fetch all Vietnamese listed stocks from VNDirect and replace the stocks table.

    Returns the number of rows written.
    """
    url = f"{BASE_URL}/stocks?q=type:stock~floor:HOSE,HNX,UPCOM&size=9999"
    resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    rows = payload.get("data", [])

    if not rows:
        return 0

    con.execute("DELETE FROM stocks")
    con.executemany(
        """
        INSERT INTO stocks (
            code, type, floor, status, company_name, company_name_eng,
            short_name, listed_date, delisted_date, company_id, tax_code, isin
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                r.get("code"),
                r.get("type"),
                r.get("floor"),
                r.get("status"),
                r.get("companyName"),
                r.get("companyNameEng"),
                r.get("shortName"),
                r.get("listedDate"),
                r.get("delistedDate"),
                r.get("companyId"),
                r.get("taxCode"),
                r.get("isin"),
            )
            for r in rows
        ],
    )
    return len(rows)


# ---------------------------------------------------------------------------
# Financial Models (metadata)
# ---------------------------------------------------------------------------


def fetch_financial_models(con: duckdb.DuckDBPyConnection) -> int:
    """Fetch all financial-model metadata and upsert into ``financial_models``.

    Returns the number of rows written.
    """
    url = f"{BASE_URL}/financial_models?size={_PAGE_SIZE}"
    rows: list[dict] = []
    for page_data in _paginated_get(url):
        rows.extend(page_data)

    if not rows:
        return 0

    con.execute("DELETE FROM financial_models")
    con.executemany(
        """
        INSERT INTO financial_models (
            model_type, item_code, model_type_name, model_vn_desc,
            model_en_desc, company_form, note, code_list,
            item_vn_name, item_en_name, display_order, display_level,
            form_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                r.get("modelType"),
                r.get("itemCode"),
                r.get("modelTypeName"),
                r.get("modelVnDesc"),
                r.get("modelEnDesc"),
                r.get("companyForm"),
                r.get("note"),
                r.get("codeList"),
                r.get("itemVnName"),
                r.get("itemEnName"),
                r.get("displayOrder"),
                r.get("displayLevel"),
                r.get("formType"),
            )
            for r in rows
        ],
    )
    return len(rows)


# ---------------------------------------------------------------------------
# Financial Statements
# ---------------------------------------------------------------------------


def fetch_financial_statements(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    start_year: int,
    end_year: int,
) -> int:
    """Fetch annual financial statements for *ticker* and upsert.

    Returns the number of rows written.
    """
    fiscal = _fiscal_dates(start_year, end_year)
    url = (
        f"{BASE_URL}/financial_statements"
        f"?q=code:{ticker}~reportType:ANNUAL~fiscalDate:{fiscal}"
        f"&sort=fiscalDate&size={_PAGE_SIZE}"
    )
    rows: list[dict] = []
    for page_data in _paginated_get(url):
        rows.extend(page_data)

    if not rows:
        return 0

    # Replace existing data for this ticker (clean refresh)
    con.execute("DELETE FROM financial_statements WHERE code = ?", [ticker])
    con.executemany(
        """
        INSERT INTO financial_statements (
            code, item_code, report_type, model_type,
            numeric_value, fiscal_date, created_date, modified_date
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                r.get("code"),
                r.get("itemCode"),
                r.get("reportType"),
                r.get("modelType"),
                r.get("numericValue"),
                r.get("fiscalDate"),
                r.get("createdDate"),
                r.get("modifiedDate"),
            )
            for r in rows
        ],
    )
    return len(rows)


# ---------------------------------------------------------------------------
# Financial Ratios
# ---------------------------------------------------------------------------


def fetch_financial_ratios(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    start_year: int,
    end_year: int,
) -> int:
    """Fetch financial ratios for *ticker* and upsert.

    Returns the number of rows written.
    """
    dates = _fiscal_dates(start_year, end_year)
    url = (
        f"{BASE_URL}/ratios"
        f"?q=code:{ticker}~reportDate:{dates}"
        f"&size={_PAGE_SIZE}"
    )
    rows: list[dict] = []
    for page_data in _paginated_get(url):
        rows.extend(page_data)

    if not rows:
        return 0

    con.execute("DELETE FROM financial_ratios WHERE code = ?", [ticker])
    con.executemany(
        """
        INSERT INTO financial_ratios (
            code, ratio_group, report_date, item_code,
            ratio_code, item_name, value
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                r.get("code"),
                r.get("group"),
                r.get("reportDate"),
                r.get("itemCode"),
                r.get("ratioCode"),
                r.get("itemName"),
                r.get("value"),
            )
            for r in rows
        ],
    )
    return len(rows)
