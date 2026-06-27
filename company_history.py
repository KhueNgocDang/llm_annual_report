"""Fetch and persist Vietstock company history milestones (Moc lich su)."""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Any

import duckdb
import requests
from bs4 import BeautifulSoup

from database import ensure_company

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
)

_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_DATE_RE = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2})\b")
_CATEGORIES_RE = re.compile(r"_categories\s*=\s*\[(.*?)\]", re.S)


def _normalize_text(text: str) -> str:
    text = " ".join(str(text or "").split())
    return text.strip()


def _normalize_for_match(text: str) -> str:
    raw = _normalize_text(text).lower()
    decomposed = unicodedata.normalize("NFD", raw)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return stripped


def fetch_company_history_html(ticker: str, section_name: str = "ho-so-doanh-nghiep") -> str:
    """Fetch HTML snippet for a company profile section from Vietstock /view endpoint."""
    code = str(ticker or "").strip().upper()
    if not code:
        raise ValueError("Ticker is required")

    session = requests.Session()
    referer = f"https://finance.vietstock.vn/{code}/documents.htm?doctype=2"

    # Prime session cookies with a normal page load.
    warmup = session.get(
        referer,
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    warmup.raise_for_status()

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html, */*; q=0.01",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Origin": "https://finance.vietstock.vn",
        "Referer": referer,
        "X-Requested-With": "XMLHttpRequest",
    }
    payload = {
        "name": section_name,
        "code": code,
    }

    response = session.post(
        "https://finance.vietstock.vn/view",
        headers=headers,
        data=payload,
        timeout=30,
    )
    response.raise_for_status()
    return response.text


def _extract_event_fields(text: str) -> tuple[int | None, str | None, str]:
    clean = _normalize_text(text)
    date_match = _DATE_RE.search(clean)
    event_date = date_match.group(1) if date_match else None

    year_match = _YEAR_RE.search(clean)
    event_year = int(year_match.group(1)) if year_match else None

    return event_year, event_date, clean


def extract_moc_lich_su_events(html_text: str) -> list[dict[str, Any]]:
    """Parse company history events from HTML snippet, focused on Moc lich su section."""
    soup = BeautifulSoup(html_text or "", "html.parser")
    heading = soup.select_one(".company-history__header-title")

    if heading is not None:
        heading_text_norm = _normalize_for_match(heading.get_text(" ", strip=True))
        if "moc lich su" not in heading_text_norm:
            heading = None

    if heading is None:
        for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "strong", "b", "div", "span", "p"]):
            text_norm = _normalize_for_match(tag.get_text(" ", strip=True))
            if "moc lich su" in text_norm:
                heading = tag
                break

    raw_events: list[str] = []

    # Extract milestone timeline from inline script categories when present.
    for script in soup.find_all("script"):
        script_text = script.string or script.get_text("\n", strip=True)
        if not script_text:
            continue
        match = _CATEGORIES_RE.search(script_text)
        if not match:
            continue
        labels = re.findall(r'"([^\"]+)"', match.group(1))
        for label in labels:
            clean_label = _normalize_text(label)
            if clean_label:
                raw_events.append(clean_label)

    def _append_event(text: str) -> None:
        clean = _normalize_text(text)
        if clean:
            raw_events.append(clean)

    if heading is not None:
        # First try direct list/table blocks near the heading.
        parent = heading.parent if heading.parent is not None else heading
        for li in parent.find_all("li"):
            _append_event(li.get_text(" ", strip=True))

        for tr in parent.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            line = " - ".join([c for c in cells if c])
            _append_event(line)

        # Many Vietstock profiles render the timeline under a dedicated
        # company-history body container (paragraph per milestone).
        container = heading.find_parent(
            lambda tag: tag
            and tag.name == "div"
            and "company-history__container" in (tag.get("class") or [])
        )
        if container is not None:
            body = container.select_one("div.company-history__body")
            if body is not None:
                for p in body.find_all("p"):
                    _append_event(p.get_text(" ", strip=True))

        # Always scan nearby sibling blocks because some profiles contain
        # complete milestones in HTML paragraphs while script categories keep
        # only a truncated subset (for example a single month/year label).
        sibling = heading
        steps = 0
        while sibling is not None and steps < 15:
            sibling = sibling.find_next_sibling()
            steps += 1
            if sibling is None:
                break
            if sibling.name in {"script", "style"}:
                continue
            if sibling.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                break
            if sibling.name in {"ul", "ol"}:
                for li in sibling.find_all("li"):
                    _append_event(li.get_text(" ", strip=True))
            elif sibling.name == "table":
                for tr in sibling.find_all("tr"):
                    cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
                    line = " - ".join([c for c in cells if c])
                    _append_event(line)
            else:
                text = sibling.get_text(" ", strip=True)
                if _YEAR_RE.search(text) and "var _" not in text:
                    _append_event(text)

    # Fallback: line-based extraction if section structure is unexpected.
    if not raw_events:
        for script in soup(["script", "style"]):
            script.extract()
        for line in soup.get_text("\n").splitlines():
            text = _normalize_text(line)
            if len(text) < 8:
                continue
            if _YEAR_RE.search(text):
                raw_events.append(text)

    dedup: list[str] = []
    seen: set[str] = set()
    for event in raw_events:
        key = event.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(event)

    events: list[dict[str, Any]] = []
    for i, text in enumerate(dedup, start=1):
        year, date, clean = _extract_event_fields(text)
        events.append(
            {
                "event_order": i,
                "event_year": year,
                "event_date": date,
                "event_text": clean,
            }
        )

    return events


def _compute_first_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not events:
        return None

    def _rank(item: dict[str, Any]) -> tuple[int, int]:
        year = item.get("event_year")
        year_rank = int(year) if year is not None else 9999
        return year_rank, int(item.get("event_order") or 9999)

    first = sorted(events, key=_rank)[0]
    first_year = first.get("event_year")
    if isinstance(first_year, int):
        firm_age = max(0, datetime.now().year - first_year)
    else:
        firm_age = None

    return {
        "first_event_year": first_year,
        "first_event_date": first.get("event_date"),
        "first_event_text": first.get("event_text"),
        "firm_age": firm_age,
    }


def sync_company_history(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    section_name: str = "ho-so-doanh-nghiep",
) -> dict[str, Any]:
    """Fetch, parse, and persist company history events + firm age for one ticker."""
    code = str(ticker or "").strip().upper()
    if not code:
        raise ValueError("Ticker is required")

    ensure_company(con, code)

    html = fetch_company_history_html(code, section_name=section_name)
    events = extract_moc_lich_su_events(html)
    first_event = _compute_first_event(events)

    con.execute("DELETE FROM company_history_events WHERE ticker = ?", [code])
    for event in events:
        con.execute(
            """
            INSERT INTO company_history_events (
                ticker,
                event_order,
                event_year,
                event_date,
                event_text,
                section_name,
                fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, get_current_timestamp())
            """,
            [
                code,
                int(event["event_order"]),
                event.get("event_year"),
                event.get("event_date"),
                str(event.get("event_text") or ""),
                section_name,
            ],
        )

    first_year = first_event.get("first_event_year") if first_event else None
    first_date = first_event.get("first_event_date") if first_event else None
    first_text = first_event.get("first_event_text") if first_event else None
    firm_age = first_event.get("firm_age") if first_event else None

    con.execute(
        """
        INSERT INTO company_history_summary (
            ticker,
            first_event_year,
            first_event_date,
            first_event_text,
            firm_age,
            event_count,
            fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, get_current_timestamp())
        ON CONFLICT (ticker) DO UPDATE SET
            first_event_year = EXCLUDED.first_event_year,
            first_event_date = EXCLUDED.first_event_date,
            first_event_text = EXCLUDED.first_event_text,
            firm_age = EXCLUDED.firm_age,
            event_count = EXCLUDED.event_count,
            fetched_at = EXCLUDED.fetched_at
        """,
        [
            code,
            first_year,
            first_date,
            first_text,
            firm_age,
            len(events),
        ],
    )

    return {
        "ticker": code,
        "events": events,
        "event_count": len(events),
        "first_event": first_event,
    }


def sync_company_history_many(
    con: duckdb.DuckDBPyConnection,
    tickers: list[str],
    section_name: str = "ho-so-doanh-nghiep",
) -> list[dict[str, Any]]:
    """Run history sync for many tickers and return per-ticker results."""
    results: list[dict[str, Any]] = []
    for ticker in tickers:
        code = str(ticker or "").strip().upper()
        if not code:
            continue
        results.append(sync_company_history(con, code, section_name=section_name))
    return results
