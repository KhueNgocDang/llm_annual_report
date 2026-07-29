from __future__ import annotations

from typing import Any

import requests

from config import DEFAULT_END_YEAR, DEFAULT_START_YEAR
from database import connection_scope

_STOCKS_URL = "https://api-finfo.vndirect.com.vn/v4/stocks"
_STOCKS_QUERY = "type:stock~floor:HOSE,HNX,UPCOM"
_BASE_V4 = "https://api-finfo.vndirect.com.vn/v4"


def fetch_stocks(timeout_seconds: int = 45) -> list[dict[str, Any]]:
    params = {
        "q": _STOCKS_QUERY,
        "size": 9999,
    }
    resp = requests.get(_STOCKS_URL, params=params, timeout=timeout_seconds)
    resp.raise_for_status()

    payload = resp.json()
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    return data


def sync_stocks() -> int:
    rows = fetch_stocks()
    prepared: list[tuple[Any, ...]] = []

    for item in rows:
        prepared.append(
            (
                item.get("code"),
                item.get("type"),
                item.get("floor"),
                item.get("status"),
                item.get("companyName"),
                item.get("companyNameEng"),
                item.get("shortName"),
                item.get("listedDate"),
                item.get("delistedDate"),
                item.get("companyId"),
                item.get("taxCode"),
                item.get("isin"),
            )
        )

    with connection_scope() as con:
        con.execute("DELETE FROM stocks")
        if prepared:
            con.executemany(
                """
                INSERT INTO stocks (
                    code,
                    type,
                    floor,
                    status,
                    company_name,
                    company_name_eng,
                    short_name,
                    listed_date,
                    delisted_date,
                    company_id,
                    tax_code,
                    isin
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                prepared,
            )
    return len(prepared)


def _paginated_get(
    endpoint: str,
    *,
    q: str | None = None,
    page_size: int = 500,
    max_pages: int = 200,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        params: dict[str, Any] = {"page": page, "size": page_size}
        if q:
            params["q"] = q
        resp = requests.get(f"{_BASE_V4}/{endpoint}", params=params, timeout=45)
        resp.raise_for_status()
        payload = resp.json()
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list) or not rows:
            break
        out.extend([r for r in rows if isinstance(r, dict)])
        if len(rows) < page_size:
            break
    return out


def sync_financial_models() -> int:
    rows = _paginated_get("financial_models", page_size=1000)
    prepared: list[tuple[Any, ...]] = []
    for item in rows:
        prepared.append(
            (
                item.get("modelType"),
                item.get("itemCode"),
                item.get("modelTypeName"),
                item.get("modelVnDesc"),
                item.get("modelEnDesc"),
                item.get("companyForm"),
                item.get("note"),
                item.get("codeList"),
                item.get("itemVnName"),
                item.get("itemEnName"),
                item.get("displayOrder"),
                item.get("displayLevel"),
                item.get("formType"),
            )
        )

    with connection_scope() as con:
        con.execute("DELETE FROM financial_models")
        if prepared:
            con.executemany(
                """
                INSERT INTO financial_models (
                    model_type,
                    item_code,
                    model_type_name,
                    model_vn_desc,
                    model_en_desc,
                    company_form,
                    note,
                    code_list,
                    item_vn_name,
                    item_en_name,
                    display_order,
                    display_level,
                    form_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                prepared,
            )
    return len(prepared)


def _statement_query(ticker: str, start_year: int, end_year: int) -> str:
    return (
        f"code:{ticker.upper()}~reportType:ANNUAL"
        f"~fiscalDate:gte:{start_year}-01-01~fiscalDate:lte:{end_year}-12-31"
    )


def _ratio_query(ticker: str, start_year: int, end_year: int) -> str:
    return (
        f"code:{ticker.upper()}"
        f"~reportDate:gte:{start_year}-01-01~reportDate:lte:{end_year}-12-31"
    )


def sync_financial_statements_for_ticker(
    ticker: str,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
) -> int:
    rows = _paginated_get(
        "financial_statements",
        q=_statement_query(ticker, start_year, end_year),
        page_size=1000,
    )
    prepared: list[tuple[Any, ...]] = []
    for item in rows:
        prepared.append(
            (
                item.get("code") or ticker.upper(),
                item.get("itemCode"),
                item.get("reportType"),
                item.get("modelType"),
                item.get("numericValue"),
                item.get("fiscalDate"),
                item.get("createdDate"),
                item.get("modifiedDate"),
            )
        )

    with connection_scope() as con:
        con.execute("DELETE FROM financial_statements WHERE code = ?", [ticker.upper()])
        if prepared:
            con.executemany(
                """
                INSERT INTO financial_statements (
                    code,
                    item_code,
                    report_type,
                    model_type,
                    numeric_value,
                    fiscal_date,
                    created_date,
                    modified_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                prepared,
            )
    return len(prepared)


def sync_financial_ratios_for_ticker(
    ticker: str,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
) -> int:
    rows = _paginated_get(
        "ratios",
        q=_ratio_query(ticker, start_year, end_year),
        page_size=1000,
    )
    prepared: list[tuple[Any, ...]] = []
    for item in rows:
        prepared.append(
            (
                item.get("code") or ticker.upper(),
                item.get("ratioGroup"),
                item.get("reportDate"),
                item.get("itemCode"),
                item.get("ratioCode"),
                item.get("itemName"),
                item.get("value"),
            )
        )

    with connection_scope() as con:
        con.execute("DELETE FROM financial_ratios WHERE code = ?", [ticker.upper()])
        if prepared:
            con.executemany(
                """
                INSERT INTO financial_ratios (
                    code,
                    ratio_group,
                    report_date,
                    item_code,
                    ratio_code,
                    item_name,
                    value
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                prepared,
            )
    return len(prepared)


def sync_financial_statements_all(
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
) -> dict[str, int]:
    with connection_scope() as con:
        tickers = [str(r[0]) for r in con.execute("SELECT ticker FROM companies ORDER BY ticker").fetchall()]

    done = 0
    failed = 0
    rows = 0
    for ticker in tickers:
        try:
            rows += sync_financial_statements_for_ticker(ticker, start_year, end_year)
            done += 1
        except Exception:
            failed += 1
    return {"tickers_done": done, "tickers_failed": failed, "rows": rows}


def sync_financial_ratios_all(
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
) -> dict[str, int]:
    with connection_scope() as con:
        tickers = [str(r[0]) for r in con.execute("SELECT ticker FROM companies ORDER BY ticker").fetchall()]

    done = 0
    failed = 0
    rows = 0
    for ticker in tickers:
        try:
            rows += sync_financial_ratios_for_ticker(ticker, start_year, end_year)
            done += 1
        except Exception:
            failed += 1
    return {"tickers_done": done, "tickers_failed": failed, "rows": rows}
