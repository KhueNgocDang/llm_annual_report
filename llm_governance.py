"""
Governance extraction module — RAG-based extraction of shareholders,
board of directors, and audit information from annual reports.

Extracts structured data that can be used to compute:
  - Audit firm
  - Proportion of ownership held by foreign shareholders
  - Proportion of ownership held by state shareholders
  - Number of shares owned by institutions
  - Number of outstanding shares
  - Number of woman board members
  - Number of foreign board members
  - Number of independent board members
  - Chair of the board cum director (CEO duality)
  - Total members of the board of directors and commissioners
  - Existence of an internal audit committee

Results are stored in ``governance_results``; batch progress is tracked
in ``governance_jobs``.
"""

from __future__ import annotations

import json
import logging
import time

import duckdb

from config import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    GOVERNANCE_EXTRACTION_ITEMS,
    INFERENCE_MODEL,
    INFERENCE_TOP_K,
)
from database import ensure_vss_loaded, get_connection
from embedder import _get_client, get_embeddings

logger = logging.getLogger(__name__)

try:
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None


MAX_PROMPT_CONTENT_TOKENS = 120000


def _is_token_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "requested" in msg
        and "token" in msg
        and ("max" in msg or "maximum" in msg)
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_governance_items() -> list[dict[str, str]]:
    """Return all governance extraction items."""
    return GOVERNANCE_EXTRACTION_ITEMS


def get_governance_groups() -> dict[str, list[dict]]:
    """Return governance items grouped by their group name."""
    groups: dict[str, list[dict]] = {}
    for item in GOVERNANCE_EXTRACTION_ITEMS:
        groups.setdefault(item["group"], []).append(item)
    return groups


# ---------------------------------------------------------------------------
# Category embedding cache
# ---------------------------------------------------------------------------

_gov_category_embeddings: dict[str, list[float]] | None = None


def _get_gov_category_embeddings(
    model: str | None = None,
    dimensions: int | None = None,
) -> dict[str, list[float]]:
    global _gov_category_embeddings
    if _gov_category_embeddings is not None:
        return _gov_category_embeddings

    model = model or EMBEDDING_MODEL
    dimensions = dimensions or EMBEDDING_DIMENSIONS

    descriptions = [
        item["description"] for item in GOVERNANCE_EXTRACTION_ITEMS
    ]
    codes = [item["code"] for item in GOVERNANCE_EXTRACTION_ITEMS]

    embeddings = get_embeddings(
        descriptions, model=model, dimensions=dimensions
    )
    _gov_category_embeddings = dict(zip(codes, embeddings))
    return _gov_category_embeddings


def reset_gov_category_embeddings_cache() -> None:
    global _gov_category_embeddings
    _gov_category_embeddings = None


# ---------------------------------------------------------------------------
# RAG retrieval
# ---------------------------------------------------------------------------


def retrieve_chunks_for_gov_item(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    year: int,
    item_code: str,
    *,
    top_k: int | None = None,
    model: str | None = None,
    dimensions: int | None = None,
) -> list[dict]:
    """Retrieve top-k most relevant chunks for a governance extraction item."""
    top_k = top_k or INFERENCE_TOP_K
    dimensions = dimensions or EMBEDDING_DIMENSIONS

    cat_embeddings = _get_gov_category_embeddings(
        model=model, dimensions=dimensions
    )
    if item_code not in cat_embeddings:
        raise ValueError(f"Unknown governance item code: {item_code}")

    query_emb = cat_embeddings[item_code]

    sql = f"""
        SELECT
            chunk_index,
            chunk_text,
            token_count,
            list_cosine_distance(embedding::FLOAT[], ?::FLOAT[]) AS distance
        FROM document_embeddings
        WHERE ticker = ? AND year = ?
        ORDER BY distance ASC
        LIMIT ?
    """
    rows = con.execute(
        sql, [query_emb, ticker.upper(), year, top_k]
    ).fetchall()
    return [
        {
            "chunk_index": r[0],
            "chunk_text": r[1],
            "token_count": r[2],
            "distance": r[3],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# LLM evaluation prompts
# ---------------------------------------------------------------------------

_GOV_EXTRACT_PROMPT = """You are a structured data extraction assistant specializing in Vietnamese annual reports (Báo cáo thường niên). You will be given text from an annual report and a specific extraction task.

Your job is to extract ALL information related to the task — list every person, entity, and data point you can find. Be exhaustive: if 7 board members are mentioned, list all 7 with all available details for each.

Rules:
- Extract every individual/entity mentioned that is relevant to the task.
- For each person, extract ALL available fields (name, position, gender, nationality, shares, dates, etc.).
- If a field is not mentioned for a person, use `null` for that field — do NOT skip the person.
- Translate Vietnamese names to their original form (keep Vietnamese diacritics).
- Infer gender from Vietnamese names when not explicitly stated (e.g., "Nguyễn Thị ..." → female, "Nguyễn Văn ..." → male).
- Return ONLY a valid JSON object — no additional text.

The JSON must have these properties:

- `"value"`: an object with summary/aggregate statistics computed from the extracted details.
- `"details"`: an array of ALL extracted records. Each record is an object with relevant fields. Every person/entity must be a separate entry.
- `"reason"`: brief summary of what was extracted and from which section of the report.

All JSON properties must always be present.

TEXT DATA:
```
{content}
```

EXTRACTION TASK:
```
{criteria}
```

EXPECTED OUTPUT FORMAT:
```
{output_format}
```"""

# Per-item output format hints
_OUTPUT_FORMATS: dict[str, str] = {
    "GOV_SHAREHOLDERS": """{
  "value": {
    "total_outstanding_shares": 123456789,
    "total_treasury_shares": 0,
    "foreign_ownership_pct": 5.2,
    "state_ownership_pct": 30.0,
    "institutional_shares": 50000000,
    "individual_shares": 73456789
  },
  "details": [
    {"name": "Nguyen Van A", "shares": 1000000, "ownership_pct": 2.5, "type": "individual", "nationality": "Vietnamese", "notes": "Major shareholder"},
    {"name": "SCIC", "shares": 30000000, "ownership_pct": 30.0, "type": "state", "nationality": "Vietnamese", "notes": "State Capital Investment Corporation"},
    {"name": "Dragon Capital", "shares": 5000000, "ownership_pct": 5.2, "type": "foreign_institution", "nationality": "British Virgin Islands", "notes": "Foreign fund"}
  ],
  "reason": "Extracted from section 'Cơ cấu cổ đông' ..."
}""",
    "GOV_BOARD": """{
  "value": {
    "total_members": 7,
    "women_count": 2,
    "men_count": 5,
    "foreign_count": 1,
    "independent_count": 3,
    "non_executive_count": 4,
    "executive_count": 3,
    "chair_name": "Nguyen Van A",
    "chair_is_ceo": false,
    "ceo_name": "Tran Van B"
  },
  "details": [
    {"name": "Nguyen Van A", "position": "Chủ tịch HĐQT / Chairman", "gender": "male", "is_independent": false, "is_executive": false, "is_foreign": false, "nationality": "Vietnamese", "date_of_birth": "1965-03-15", "appointment_date": "2020-06-15", "term_end": "2025-06-15", "education": "MBA", "shares_owned": 500000, "ownership_pct": 1.2, "notes": "Also serves as..."},
    {"name": "Tran Thi C", "position": "Thành viên HĐQT độc lập / Independent Member", "gender": "female", "is_independent": true, "is_executive": false, "is_foreign": false, "nationality": "Vietnamese", "date_of_birth": null, "appointment_date": "2021-04-20", "term_end": null, "education": null, "shares_owned": 0, "ownership_pct": 0, "notes": ""}
  ],
  "reason": "Extracted 7 board members from section 'Hội đồng quản trị' ..."
}""",
    "GOV_AUDIT": """{
  "value": {
    "external_audit_firm": "Công ty TNHH Deloitte Việt Nam",
    "external_audit_firm_en": "Deloitte Vietnam Co., Ltd.",
    "audit_opinion": "unqualified",
    "has_internal_audit_committee": true,
    "internal_audit_committee_size": 3,
    "audit_fee": null
  },
  "details": [
    {"name": "Nguyen Van D", "role": "internal_audit_committee_chair", "gender": "male", "is_independent": true, "appointment_date": "2020-06-15", "notes": ""},
    {"name": "Le Thi E", "role": "internal_audit_committee_member", "gender": "female", "is_independent": true, "appointment_date": "2020-06-15", "notes": ""},
    {"name": "Công ty TNHH Deloitte Việt Nam", "role": "external_auditor", "gender": null, "is_independent": null, "appointment_date": null, "notes": "Signed by auditor Pham Van F"}
  ],
  "reason": "Extracted from 'Báo cáo kiểm toán' and 'Ủy ban Kiểm toán' sections ..."
}""",
    "GOV_SUPERVISORY": """{
  "value": {
    "total_members": 3,
    "women_count": 1,
    "men_count": 2,
    "independent_count": 2
  },
  "details": [
    {"name": "Nguyen Van G", "position": "Trưởng Ban Kiểm soát / Head", "gender": "male", "is_independent": true, "date_of_birth": null, "appointment_date": "2020-06-15", "term_end": null, "education": null, "shares_owned": 0, "notes": ""},
    {"name": "Pham Thi H", "position": "Thành viên / Member", "gender": "female", "is_independent": true, "date_of_birth": null, "appointment_date": "2020-06-15", "term_end": null, "education": null, "shares_owned": 1000, "notes": ""}
  ],
  "reason": "Extracted from 'Ban Kiểm soát' section ..."
}""",
    "GOV_COMPANY_INFO": """{
  "value": {
    "founding_year": 1995,
    "full_name_vi": "Công ty Cổ phần Vận tải Biển Việt Nam",
    "full_name_en": "Vietnam Ocean Shipping Joint Stock Company",
    "ticker": "VOS",
    "listing_date": "2007-06-20",
    "main_business": "Vận tải biển / Ocean shipping"
  },
  "details": [
    {"field": "founding_year", "value": 1995, "source": "Giấy phép thành lập số ... ngày 01/07/1995"},
    {"field": "listing_date", "value": "2007-06-20", "source": "Niêm yết trên HoSE ngày 20/06/2007"}
  ],
  "reason": "Extracted from 'Giới thiệu công ty' / company overview section ..."
}""",
}


def evaluate_gov_item(
    chunks: list[dict],
    item_code: str,
    item_description: str,
    *,
    model: str | None = None,
) -> dict:
    """Send retrieved chunks to the LLM for governance data extraction.

    Returns dict with keys: ``found``, ``value``, ``details``, ``reason``.
    """
    model = model or INFERENCE_MODEL
    client = _get_client()

    def _estimate_tokens(text: str) -> int:
        if tiktoken is not None:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        return max(1, len(text) // 4)

    def _truncate_text_tokens(text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        if tiktoken is not None:
            enc = tiktoken.get_encoding("cl100k_base")
            toks = enc.encode(text)
            if len(toks) <= max_tokens:
                return text
            return enc.decode(toks[:max_tokens])
        words = text.split()
        if len(words) <= max_tokens:
            return text
        return " ".join(words[:max_tokens])

    selected_chunks: list[str] = []
    used_tokens = 0
    for chunk in chunks:
        text = str(chunk.get("chunk_text") or "")
        if not text:
            continue
        token_count = int(chunk.get("token_count") or _estimate_tokens(text))
        if used_tokens + token_count <= MAX_PROMPT_CONTENT_TOKENS:
            selected_chunks.append(text)
            used_tokens += token_count
            continue
        remain = MAX_PROMPT_CONTENT_TOKENS - used_tokens
        if remain > 0 and not selected_chunks:
            selected_chunks.append(_truncate_text_tokens(text, remain))
            used_tokens += remain
        break

    if not selected_chunks:
        selected_chunks = [""]

    content = "\n".join(selected_chunks)
    output_format = _OUTPUT_FORMATS.get(
        item_code,
        '{"found": true/false, "value": ..., "details": [...], "reason": "..."}',
    )
    prompt = _GOV_EXTRACT_PROMPT.format(
        content=content,
        criteria=item_description,
        output_format=output_format,
    )

    _is_reasoning = model.startswith(("o1", "o3", "o4", "gpt-5"))

    api_kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    if _is_reasoning:
        api_kwargs["reasoning_effort"] = "low"
    else:
        api_kwargs["temperature"] = 0

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(**api_kwargs)
            break
        except Exception as exc:
            last_exc = exc
            if attempt < 2 and _is_token_limit_error(exc):
                new_limit = max(1024, int(_estimate_tokens(content) * 0.7))
                content = _truncate_text_tokens(content, new_limit)
                prompt = _GOV_EXTRACT_PROMPT.format(
                    content=content,
                    criteria=item_description,
                    output_format=output_format,
                )
                api_kwargs["messages"] = [{"role": "user", "content": prompt}]
                logger.warning(
                    "Prompt too large for %s; shrinking and retrying (attempt %d/3)",
                    model,
                    attempt + 1,
                )
                continue
            if (
                attempt < 2
                and "could not parse the JSON body" in str(exc).lower()
            ):
                logger.warning(
                    "Transient API error (attempt %d/3), retrying: %s",
                    attempt + 1,
                    exc,
                )
                time.sleep(2**attempt)
            else:
                raise
    else:
        raise last_exc  # type: ignore[misc]

    raw = response.choices[0].message.content or ""
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "value": None,
            "details": [],
            "reason": f"Failed to parse LLM response: {raw}",
        }

    details = result.get("details", [])
    value = result.get("value")
    # Derive 'found' from whether we got any details or non-null value
    found = bool(details) or (value is not None and value != {})

    return {
        "found": found,
        "value": value,
        "details": details,
        "reason": str(result.get("reason", "")),
    }


# ---------------------------------------------------------------------------
# Computed governance variables
# ---------------------------------------------------------------------------


def compute_governance_variables(
    results: dict[str, dict],
) -> dict[str, object]:
    """Compute derived governance variables from extraction results.

    Parameters
    ----------
    results : dict
        Mapping of item code → extraction result dict.

    Returns
    -------
    dict
        Computed variables ready for analysis/export.
    """
    variables: dict[str, object] = {}

    # --- From GOV_SHAREHOLDERS ---
    sh = results.get("GOV_SHAREHOLDERS", {})
    sh_val = sh.get("value") or {}
    sh_details = sh.get("details") or []
    if isinstance(sh_val, dict):
        variables["total_outstanding_shares"] = sh_val.get(
            "total_outstanding_shares"
        )
        variables["total_treasury_shares"] = sh_val.get(
            "total_treasury_shares"
        )
        variables["foreign_ownership_pct"] = sh_val.get(
            "foreign_ownership_pct"
        )
        variables["state_ownership_pct"] = sh_val.get("state_ownership_pct")
        variables["institutional_shares"] = sh_val.get("institutional_shares")
        variables["individual_shares"] = sh_val.get("individual_shares")
    variables["shareholder_count"] = len(sh_details) if sh_details else None

    # --- From GOV_BOARD ---
    bd = results.get("GOV_BOARD", {})
    bd_val = bd.get("value") or {}
    bd_details = bd.get("details") or []
    if isinstance(bd_val, dict):
        variables["total_board_members"] = bd_val.get("total_members")
        variables["women_board_members"] = bd_val.get("women_count")
        variables["men_board_members"] = bd_val.get("men_count")
        variables["foreign_board_members"] = bd_val.get("foreign_count")
        variables["independent_board_members"] = bd_val.get(
            "independent_count"
        )
        variables["non_executive_board_members"] = bd_val.get(
            "non_executive_count"
        )
        variables["executive_board_members"] = bd_val.get("executive_count")
        variables["chair_name"] = bd_val.get("chair_name")
        variables["chair_is_ceo"] = bd_val.get("chair_is_ceo")
        variables["ceo_name"] = bd_val.get("ceo_name")
    # Fallback: count from details if value fields missing
    if bd_details and not variables.get("total_board_members"):
        variables["total_board_members"] = len(bd_details)
    if bd_details and variables.get("women_board_members") is None:
        variables["women_board_members"] = sum(
            1
            for d in bd_details
            if str(d.get("gender", "")).lower() == "female"
        )
    if bd_details and variables.get("foreign_board_members") is None:
        variables["foreign_board_members"] = sum(
            1 for d in bd_details if d.get("is_foreign")
        )
    if bd_details and variables.get("independent_board_members") is None:
        variables["independent_board_members"] = sum(
            1 for d in bd_details if d.get("is_independent")
        )

    # --- From GOV_AUDIT ---
    au = results.get("GOV_AUDIT", {})
    au_val = au.get("value") or {}
    if isinstance(au_val, dict):
        variables["audit_firm"] = (
            au_val.get("external_audit_firm")
            or au_val.get("external_audit_firm_en")
            or au_val.get("audit_firm")
        )
        variables["audit_opinion"] = au_val.get("audit_opinion")
        variables["has_internal_audit_committee"] = au_val.get(
            "has_internal_audit_committee"
        )
        variables["internal_audit_committee_size"] = au_val.get(
            "internal_audit_committee_size"
        )

    # --- From GOV_SUPERVISORY ---
    sv = results.get("GOV_SUPERVISORY", {})
    sv_val = sv.get("value") or {}
    sv_details = sv.get("details") or []
    if isinstance(sv_val, dict):
        variables["total_supervisory_members"] = sv_val.get("total_members")
        variables["supervisory_women_count"] = sv_val.get("women_count")
        variables["supervisory_independent_count"] = sv_val.get(
            "independent_count"
        )
    if sv_details and not variables.get("total_supervisory_members"):
        variables["total_supervisory_members"] = len(sv_details)

    # --- From GOV_COMPANY_INFO ---
    ci = results.get("GOV_COMPANY_INFO", {})
    ci_val = ci.get("value") or {}
    if isinstance(ci_val, dict):
        variables["founding_year"] = ci_val.get("founding_year")
        variables["full_name_vi"] = ci_val.get("full_name_vi")
        variables["full_name_en"] = ci_val.get("full_name_en")
        variables["listing_date"] = ci_val.get("listing_date")
        variables["main_business"] = ci_val.get("main_business")

    return variables


# ---------------------------------------------------------------------------
# Job management
# ---------------------------------------------------------------------------


def create_governance_jobs(
    con: duckdb.DuckDBPyConnection,
    *,
    tickers: list[str] | None = None,
    years: list[int] | None = None,
    replace: bool = False,
    inference_model: str | None = None,
) -> int:
    """Create pending governance extraction jobs for embedded reports."""
    inference_model = inference_model or INFERENCE_MODEL

    rows = con.execute(
        "SELECT DISTINCT ticker, year FROM document_embeddings"
    ).fetchall()
    embedded = [(r[0], r[1]) for r in rows]

    if tickers:
        upper = {t.upper() for t in tickers}
        embedded = [(t, y) for t, y in embedded if t in upper]
    if years:
        embedded = [(t, y) for t, y in embedded if y in years]

    total_items = len(GOVERNANCE_EXTRACTION_ITEMS)
    created = 0
    for ticker, year in embedded:
        existing = con.execute(
            "SELECT status FROM governance_jobs "
            "WHERE ticker = ? AND year = ? AND model = ?",
            [ticker, year, inference_model],
        ).fetchone()

        if existing is None:
            con.execute(
                """
                INSERT INTO governance_jobs
                    (id, ticker, year, model, status, items_total)
                VALUES (nextval('governance_jobs_id_seq'), ?, ?, ?, 'pending', ?)
                """,
                [ticker, year, inference_model, total_items],
            )
            created += 1
        elif replace and existing[0] in ("completed", "failed"):
            con.execute(
                """
                UPDATE governance_jobs
                SET status = 'pending', error_message = NULL,
                    started_at = NULL, completed_at = NULL,
                    items_done = 0
                WHERE ticker = ? AND year = ? AND model = ?
                """,
                [ticker, year, inference_model],
            )
            created += 1

    return created


def get_governance_jobs(
    con: duckdb.DuckDBPyConnection,
    *,
    status: str | None = None,
    model: str | None = None,
) -> list[dict]:
    """Return governance jobs, optionally filtered."""
    query = "SELECT * FROM governance_jobs"
    conditions: list[str] = []
    params: list = []
    if status:
        conditions.append("status = ?")
        params.append(status)
    if model:
        conditions.append("model = ?")
        params.append(model)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY ticker, year"

    rows = con.execute(query, params).fetchall()
    columns = [
        "id",
        "ticker",
        "year",
        "status",
        "model",
        "top_k",
        "items_done",
        "items_total",
        "started_at",
        "completed_at",
        "error_message",
        "created_at",
    ]
    return [dict(zip(columns, row)) for row in rows]


def get_governance_job_counts(
    con: duckdb.DuckDBPyConnection,
    *,
    model: str | None = None,
) -> dict[str, int]:
    query = "SELECT status, COUNT(*) FROM governance_jobs"
    params: list = []
    if model:
        query += " WHERE model = ?"
        params.append(model)
    query += " GROUP BY status"
    rows = con.execute(query, params).fetchall()
    return {r[0]: r[1] for r in rows}


def reset_failed_governance_jobs(
    con: duckdb.DuckDBPyConnection,
    *,
    model: str | None = None,
) -> int:
    query = (
        "UPDATE governance_jobs SET status = 'pending', error_message = NULL, "
        "started_at = NULL, completed_at = NULL, items_done = 0 "
        "WHERE status = 'failed'"
    )
    params: list = []
    if model:
        query += " AND model = ?"
        params.append(model)
    con.execute(query, params)
    return 1


def sync_governance_job_states(
    con: duckdb.DuckDBPyConnection,
    *,
    model: str | None = None,
) -> dict[str, int]:
    """Sync governance job statuses with actual results in DB."""
    total_items = len(GOVERNANCE_EXTRACTION_ITEMS)

    query = (
        "SELECT id, ticker, year, model, status, items_done "
        "FROM governance_jobs WHERE status != 'completed'"
    )
    params: list = []
    if model:
        query += " AND model = ?"
        params.append(model)

    jobs = con.execute(query, params).fetchall()
    n_completed = 0
    n_updated = 0

    for job_id, ticker, year, job_model, status, done in jobs:
        row = con.execute(
            "SELECT COUNT(*) FROM governance_results "
            "WHERE ticker = ? AND year = ? AND model = ?",
            [ticker, year, job_model],
        ).fetchone()
        result_count = row[0] if row else 0

        if result_count >= total_items:
            con.execute(
                """
                UPDATE governance_jobs
                SET status = 'completed',
                    completed_at = get_current_timestamp(),
                    items_done = ?, items_total = ?,
                    error_message = NULL
                WHERE id = ?
                """,
                [total_items, total_items, job_id],
            )
            n_completed += 1
        elif result_count > 0 and result_count != done:
            con.execute(
                """
                UPDATE governance_jobs
                SET items_done = ?, items_total = ?
                WHERE id = ?
                """,
                [result_count, total_items, job_id],
            )
            n_updated += 1

    return {"completed": n_completed, "updated": n_updated}


def delete_governance_jobs(
    con: duckdb.DuckDBPyConnection,
    *,
    ticker: str | None = None,
    year: int | None = None,
    status: str | None = None,
    model: str | None = None,
) -> None:
    conditions: list[str] = []
    params: list = []
    if ticker:
        conditions.append("ticker = ?")
        params.append(ticker.upper())
    if year is not None:
        conditions.append("year = ?")
        params.append(year)
    if status:
        conditions.append("status = ?")
        params.append(status)
    if model:
        conditions.append("model = ?")
        params.append(model)

    where = ""
    if conditions:
        where = " WHERE " + " AND ".join(conditions)

    con.execute(f"DELETE FROM governance_results{where}", params)
    con.execute(f"DELETE FROM governance_jobs{where}", params)


# ---------------------------------------------------------------------------
# Results helpers
# ---------------------------------------------------------------------------


def get_governance_results(
    con: duckdb.DuckDBPyConnection,
    *,
    ticker: str | None = None,
    year: int | None = None,
    model: str | None = None,
) -> list[dict]:
    conditions: list[str] = []
    params: list = []
    if ticker:
        conditions.append("ticker = ?")
        params.append(ticker.upper())
    if year is not None:
        conditions.append("year = ?")
        params.append(year)
    if model:
        conditions.append("model = ?")
        params.append(model)

    where = ""
    if conditions:
        where = " WHERE " + " AND ".join(conditions)

    rows = con.execute(
        f"""
        SELECT id, ticker, year, item_code, found, value_json,
               details_json, reason, top_chunks, similarities,
               model, created_at
        FROM governance_results
        {where}
        ORDER BY ticker, year, item_code
        """,
        params,
    ).fetchall()

    columns = [
        "id",
        "ticker",
        "year",
        "item_code",
        "found",
        "value_json",
        "details_json",
        "reason",
        "top_chunks",
        "similarities",
        "model",
        "created_at",
    ]
    return [dict(zip(columns, row)) for row in rows]


# ---------------------------------------------------------------------------
# Core: extract governance data for a single report
# ---------------------------------------------------------------------------


def extract_governance(
    ticker: str,
    year: int,
    con: duckdb.DuckDBPyConnection | None = None,
    *,
    replace: bool = False,
    top_k: int | None = None,
    inference_model: str | None = None,
    embedding_model: str | None = None,
    dimensions: int | None = None,
    item_codes: list[str] | None = None,
) -> int:
    """Run governance extraction for a single annual report.

    For each governance item:
    1. Retrieve top-k chunks via vector similarity (RAG)
    2. Send to LLM for structured extraction
    3. Store result in ``governance_results``

    Returns the number of items extracted.
    """
    own_con = con is None
    if own_con:
        con = get_connection()
        ensure_vss_loaded(con)

    ticker = ticker.upper()
    top_k = top_k or INFERENCE_TOP_K
    inference_model = inference_model or INFERENCE_MODEL

    # Determine items to extract
    if item_codes:
        valid_codes = {it["code"] for it in GOVERNANCE_EXTRACTION_ITEMS}
        unknown = set(item_codes) - valid_codes
        if unknown:
            raise ValueError(f"Unknown governance item codes: {unknown}")
        items_to_eval = [
            it
            for it in GOVERNANCE_EXTRACTION_ITEMS
            if it["code"] in item_codes
        ]
    else:
        items_to_eval = GOVERNANCE_EXTRACTION_ITEMS

    # Ensure job row exists
    existing_job = con.execute(
        "SELECT status FROM governance_jobs "
        "WHERE ticker = ? AND year = ? AND model = ?",
        [ticker, year, inference_model],
    ).fetchone()

    total_items = len(GOVERNANCE_EXTRACTION_ITEMS)

    if existing_job is None:
        con.execute(
            """
            INSERT INTO governance_jobs
                (id, ticker, year, model, status, items_total)
            VALUES (nextval('governance_jobs_id_seq'), ?, ?, ?, 'pending', ?)
            """,
            [ticker, year, inference_model, total_items],
        )

    # Mark job as running
    con.execute(
        """
        UPDATE governance_jobs
        SET status = 'running', started_at = get_current_timestamp(),
            top_k = ?, items_total = ?,
            items_done = 0, error_message = NULL
        WHERE ticker = ? AND year = ? AND model = ?
        """,
        [top_k, total_items, ticker, year, inference_model],
    )

    try:
        # Check embeddings exist
        row = con.execute(
            "SELECT COUNT(*) FROM document_embeddings "
            "WHERE ticker = ? AND year = ?",
            [ticker, year],
        ).fetchone()
        emb_count = row[0] if row else 0

        if emb_count == 0:
            raise ValueError(
                f"No embeddings found for {ticker}/{year}. "
                "Embed the report first."
            )

        if replace:
            con.execute(
                "DELETE FROM governance_results "
                "WHERE ticker = ? AND year = ? AND model = ?",
                [ticker, year, inference_model],
            )

        done = 0
        for item in items_to_eval:
            code = item["code"]

            if not replace:
                existing = con.execute(
                    "SELECT 1 FROM governance_results "
                    "WHERE ticker = ? AND year = ? AND item_code = ? "
                    "AND model = ?",
                    [ticker, year, code, inference_model],
                ).fetchone()
                if existing:
                    done += 1
                    continue

            # Use more chunks for governance extraction (richer context)
            gov_top_k = max(top_k, 10)
            chunks = retrieve_chunks_for_gov_item(
                con,
                ticker,
                year,
                code,
                top_k=gov_top_k,
                model=embedding_model,
                dimensions=dimensions,
            )

            if not chunks:
                result = {
                    "found": False,
                    "value": None,
                    "details": [],
                    "reason": "No relevant text chunks found in the report.",
                }
            else:
                result = evaluate_gov_item(
                    chunks,
                    code,
                    item["description"],
                    model=inference_model,
                )

            similarities_json = json.dumps(
                [round(c["distance"], 6) for c in chunks]
            )
            top_chunks_json = json.dumps([c["chunk_index"] for c in chunks])
            value_json = json.dumps(result.get("value"), ensure_ascii=False)
            details_json = json.dumps(
                result.get("details", []), ensure_ascii=False
            )

            con.execute(
                """
                INSERT INTO governance_results
                    (id, ticker, year, item_code, found, value_json,
                     details_json, reason, top_chunks, similarities, model)
                VALUES
                    (nextval('governance_results_id_seq'),
                     ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (ticker, year, item_code, model) DO UPDATE SET
                    found = EXCLUDED.found,
                    value_json = EXCLUDED.value_json,
                    details_json = EXCLUDED.details_json,
                    reason = EXCLUDED.reason,
                    top_chunks = EXCLUDED.top_chunks,
                    similarities = EXCLUDED.similarities,
                    created_at = get_current_timestamp()
                """,
                [
                    ticker,
                    year,
                    code,
                    result["found"],
                    value_json,
                    details_json,
                    result["reason"],
                    top_chunks_json,
                    similarities_json,
                    inference_model,
                ],
            )

            done += 1

            con.execute(
                """
                UPDATE governance_jobs
                SET items_done = ?
                WHERE ticker = ? AND year = ? AND model = ?
                """,
                [done, ticker, year, inference_model],
            )

        # Check completion
        total_results = con.execute(
            "SELECT COUNT(*) FROM governance_results "
            "WHERE ticker = ? AND year = ? AND model = ?",
            [ticker, year, inference_model],
        ).fetchone()
        total_done = total_results[0] if total_results else 0

        if total_done >= total_items:
            con.execute(
                """
                UPDATE governance_jobs
                SET status = 'completed',
                    completed_at = get_current_timestamp(),
                    items_done = ?
                WHERE ticker = ? AND year = ? AND model = ?
                """,
                [total_done, ticker, year, inference_model],
            )
        else:
            con.execute(
                """
                UPDATE governance_jobs
                SET items_done = ?,
                    status = CASE WHEN status = 'running' THEN 'pending'
                                  ELSE status END
                WHERE ticker = ? AND year = ? AND model = ?
                """,
                [total_done, ticker, year, inference_model],
            )

        logger.info(
            "Governance extraction complete for %s/%d — %d items extracted",
            ticker,
            year,
            done,
        )

        if own_con:
            con.close()
        return done

    except Exception as exc:
        con.execute(
            """
            UPDATE governance_jobs
            SET status = 'failed', completed_at = get_current_timestamp(),
                error_message = ?
            WHERE ticker = ? AND year = ? AND model = ?
            """,
            [str(exc), ticker, year, inference_model],
        )
        logger.error(
            "Governance extraction failed for %s/%d: %s", ticker, year, exc
        )
        if own_con:
            con.close()
        raise


# ---------------------------------------------------------------------------
# Batch operations
# ---------------------------------------------------------------------------


def extract_governance_all(
    *,
    replace: bool = False,
    progress_callback=None,
    top_k: int | None = None,
    inference_model: str | None = None,
    embedding_model: str | None = None,
    dimensions: int | None = None,
) -> dict:
    """Run governance extraction for every pending job."""
    con = get_connection()
    ensure_vss_loaded(con)

    _model = inference_model or INFERENCE_MODEL

    create_governance_jobs(con, replace=replace, inference_model=_model)

    pending = get_governance_jobs(con, status="pending", model=_model)
    results: dict = {"evaluated": [], "skipped": [], "failed": []}
    total = len(pending)

    for i, job in enumerate(pending):
        ticker, year = job["ticker"], job["year"]
        n = 0

        try:
            n = extract_governance(
                ticker,
                year,
                con=con,
                replace=replace,
                top_k=top_k,
                inference_model=inference_model,
                embedding_model=embedding_model,
                dimensions=dimensions,
            )
            if n > 0:
                results["evaluated"].append((ticker, year, n))
            else:
                results["skipped"].append((ticker, year))
        except Exception as exc:
            logger.error(
                "Governance extraction failed for %s/%d: %s",
                ticker,
                year,
                exc,
            )
            results["failed"].append((ticker, year, str(exc)))

        if progress_callback:
            progress_callback(ticker, year, n, i + 1, total)

    con.close()
    return results


def extract_governance_item_all_reports(
    item_codes: list[str],
    *,
    tickers: list[str] | None = None,
    years: list[int] | None = None,
    replace: bool = False,
    progress_callback=None,
    top_k: int | None = None,
    inference_model: str | None = None,
    embedding_model: str | None = None,
    dimensions: int | None = None,
) -> dict:
    """Run specific governance items across all embedded reports."""
    con = get_connection()
    ensure_vss_loaded(con)

    rows = con.execute(
        "SELECT DISTINCT ticker, year FROM document_embeddings "
        "ORDER BY ticker, year"
    ).fetchall()
    reports = [(r[0], r[1]) for r in rows]

    if tickers:
        upper = {t.upper() for t in tickers}
        reports = [(t, y) for t, y in reports if t in upper]
    if years:
        reports = [(t, y) for t, y in reports if y in years]

    results: dict = {"evaluated": [], "skipped": [], "failed": []}
    total = len(reports)

    for i, (ticker, year) in enumerate(reports):
        n = 0
        try:
            n = extract_governance(
                ticker,
                year,
                con=con,
                replace=replace,
                top_k=top_k,
                inference_model=inference_model,
                embedding_model=embedding_model,
                dimensions=dimensions,
                item_codes=item_codes,
            )
            if n > 0:
                results["evaluated"].append((ticker, year, n))
            else:
                results["skipped"].append((ticker, year))
        except Exception as exc:
            logger.error(
                "Governance extraction failed for %s/%d: %s",
                ticker,
                year,
                exc,
            )
            results["failed"].append((ticker, year, str(exc)))

        if progress_callback:
            progress_callback(ticker, year, n, i + 1, total)

    con.close()
    return results
