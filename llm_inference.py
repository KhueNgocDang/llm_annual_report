"""
Inference module — RAG-based environmental disclosure checklist evaluation.

For each company annual report, retrieves the most relevant embedded chunks
per checklist category using DuckDB vector similarity (cosine distance),
re-ranks them, and sends the top-k chunks to GPT for validation.

Results are stored in ``inference_results``; batch progress is tracked in
``inference_jobs``.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime

import duckdb
from openai import OpenAI

from annual_inference_config import get_task_items
from config import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    ENV_JSON_PATH,
    INFERENCE_MODEL,
    INFERENCE_TEMPERATURE,
    INFERENCE_TOP_K,
)
from database import ensure_vss_loaded, get_connection
from embedder import _get_client, get_embeddings
from hyde_retrieval import build_hyde2_embedding, precompute_hyde2_embeddings
from llm_batch_api import (
    get_batch_output_map,
    get_batch_status,
    run_chat_json_batch,
    submit_chat_json_batch,
)

logger = logging.getLogger(__name__)

try:
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None


MAX_PROMPT_CONTENT_TOKENS = 120000
INFERENCE_RESULTS_TABLE = "inference_results"
INFERENCE_RESULTS_WRITE_TABLE = "inference_results_hyde2"


def _is_token_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "requested" in msg
        and "token" in msg
        and ("max" in msg or "maximum" in msg)
    )


# ---------------------------------------------------------------------------
# Checklist helpers
# ---------------------------------------------------------------------------


def get_checklist() -> list[dict[str, str]]:
    """Return the full Environmental Disclosure Checklist."""
    return get_task_items("edc")


def get_checklist_groups() -> dict[str, list[dict]]:
    """Return checklist items grouped by their group name."""
    checklist_items = get_task_items("edc")
    groups: dict[str, list[dict]] = {}
    for item in checklist_items:
        groups.setdefault(item["group"], []).append(item)
    return groups


# ---------------------------------------------------------------------------
# Category embedding cache (embed each checklist description once)
# ---------------------------------------------------------------------------

_category_embeddings: dict[str, list[float]] | None = None


def _get_category_embeddings(
    model: str | None = None,
    dimensions: int | None = None,
    item_configs: list[dict[str, str]] | None = None,
) -> dict[str, list[float]]:
    """Return a dict mapping category code → embedding vector.

    Embeddings are cached in-process so they're only computed once.
    """
    global _category_embeddings
    if item_configs is None and _category_embeddings is not None:
        return _category_embeddings

    model = model or EMBEDDING_MODEL
    dimensions = dimensions or EMBEDDING_DIMENSIONS
    checklist_items = get_task_items("edc", item_configs)

    # Criteria are static (not ticker/year-specific), so precompute HyDE upfront.
    precompute_hyde2_embeddings(
        [item["description"] for item in checklist_items],
        embedding_model=model,
        dimensions=dimensions,
    )

    embeddings: dict[str, list[float]] = {}
    for item in checklist_items:
        code = item["code"]
        description = item["description"]
        embeddings[code] = build_hyde2_embedding(
            description,
            embedding_model=model,
            dimensions=dimensions,
        )
    if item_configs is None:
        _category_embeddings = embeddings
    return embeddings


def reset_category_embeddings_cache() -> None:
    """Clear the cached category embeddings (e.g. after model change)."""
    global _category_embeddings
    _category_embeddings = None


# ---------------------------------------------------------------------------
# RAG retrieval — vector similarity search per category
# ---------------------------------------------------------------------------


def retrieve_chunks_for_category(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    year: int,
    category_code: str,
    *,
    top_k: int | None = None,
    model: str | None = None,
    dimensions: int | None = None,
    item_configs: list[dict[str, str]] | None = None,
) -> list[dict]:
    """Retrieve the top-k most relevant chunks for a checklist category.

    Uses DuckDB ``array_cosine_distance`` for vector similarity,
    returning chunks sorted by ascending distance (most similar first).

    Returns list of dicts with keys: chunk_index, chunk_text, token_count,
    distance.
    """
    top_k = top_k or INFERENCE_TOP_K
    dimensions = dimensions or EMBEDDING_DIMENSIONS

    cat_embeddings = _get_category_embeddings(
        model=model, dimensions=dimensions, item_configs=item_configs
    )
    if category_code not in cat_embeddings:
        raise ValueError(f"Unknown category code: {category_code}")

    query_emb = cat_embeddings[category_code]

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
# LLM evaluation
# ---------------------------------------------------------------------------

_EVAL_PROMPT = """You are a helpful assistant designed to validate the content of sections in an annual report. The content to validate is related to climate change. You will be given a single row of Vietnamese text, and your task is to determine whether the data matches the provided criteria.

- Translate the text to English.

**Return only a JSON object** with the following properties:

- `"is_valid"`: a boolean (`true` or `false`) indicating whether the text matches the criteria.
- `"reason"`: Provide a brief explanation on why the text data is valid or not.
- `"citations"`: an array of `chunk_index` integers that support your answer. Use only indices that appear in the input block.

All JSON properties must always be present.

Do not include any additional text or explanations outside the JSON object.

TEXT DATA:
```
{content}
```

CRITERIA
```
{criteria}
```"""


def _format_reason_with_citations(reason: str, citations: list[int]) -> str:
    if not citations:
        return reason
    unique_sorted = sorted(set(citations))
    refs = ", ".join(str(i) for i in unique_sorted)
    suffix = f" [citations: {refs}]"
    if suffix in reason:
        return reason
    return f"{reason}{suffix}".strip()


def evaluate_category(
    chunks: list[dict],
    category_code: str,
    category_description: str,
    *,
    model: str | None = None,
) -> dict:
    """Send retrieved chunks to the LLM for checklist validation.

    Returns a dict with keys: ``is_valid`` (bool), ``reason`` (str).
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
        chunk_index = int(chunk.get("chunk_index") or -1)
        token_count = int(chunk.get("token_count") or _estimate_tokens(text))
        if used_tokens + token_count <= MAX_PROMPT_CONTENT_TOKENS:
            selected_chunks.append(f"[chunk_index={chunk_index}]\n{text}")
            used_tokens += token_count
            continue
        remain = MAX_PROMPT_CONTENT_TOKENS - used_tokens
        if remain > 0 and not selected_chunks:
            truncated = _truncate_text_tokens(text, remain)
            selected_chunks.append(f"[chunk_index={chunk_index}]\n{truncated}")
            used_tokens += remain
        break

    if not selected_chunks:
        selected_chunks = [""]

    content = "\n".join(selected_chunks)
    prompt = _EVAL_PROMPT.format(
        content=content, criteria=category_description
    )

    # Reasoning models (gpt-5*, o-series) don't support temperature;
    # use reasoning_effort instead. Non-reasoning models use configured temperature.
    _is_reasoning = model.startswith(("o1", "o3", "o4", "gpt-5"))

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            response_map = run_chat_json_batch(
                client=client,
                model=model,
                prompts={"eval": prompt},
                is_reasoning=_is_reasoning,
                temperature=INFERENCE_TEMPERATURE,
            )
            raw = response_map["eval"]
            break
        except Exception as exc:
            last_exc = exc
            if attempt < 2 and _is_token_limit_error(exc):
                new_limit = max(1024, int(_estimate_tokens(content) * 0.7))
                content = _truncate_text_tokens(content, new_limit)
                prompt = _EVAL_PROMPT.format(
                    content=content, criteria=category_description
                )
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

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "is_valid": False,
            "reason": f"Failed to parse LLM response: {raw}",
            "citations": [],
        }

    citations = [
        int(c)
        for c in (result.get("citations") or [])
        if isinstance(c, int) or (isinstance(c, str) and c.strip().lstrip("-").isdigit())
    ]
    reason = _format_reason_with_citations(
        str(result.get("reason", "")),
        citations,
    )

    return {
        "is_valid": bool(result.get("is_valid", False)),
        "reason": reason,
        "citations": citations,
    }


def _build_eval_prompt(chunks: list[dict], category_description: str) -> str:
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
        chunk_index = int(chunk.get("chunk_index") or -1)
        token_count = int(chunk.get("token_count") or _estimate_tokens(text))
        if used_tokens + token_count <= MAX_PROMPT_CONTENT_TOKENS:
            selected_chunks.append(f"[chunk_index={chunk_index}]\n{text}")
            used_tokens += token_count
            continue
        remain = MAX_PROMPT_CONTENT_TOKENS - used_tokens
        if remain > 0 and not selected_chunks:
            truncated = _truncate_text_tokens(text, remain)
            selected_chunks.append(f"[chunk_index={chunk_index}]\n{truncated}")
        break

    if not selected_chunks:
        selected_chunks = [""]

    content = "\n".join(selected_chunks)
    return _EVAL_PROMPT.format(content=content, criteria=category_description)


def _parse_eval_raw(raw: str) -> dict:
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "is_valid": False,
            "reason": f"Failed to parse LLM response: {raw}",
            "citations": [],
        }

    citations = [
        int(c)
        for c in (result.get("citations") or [])
        if isinstance(c, int) or (isinstance(c, str) and c.strip().lstrip("-").isdigit())
    ]
    reason = _format_reason_with_citations(
        str(result.get("reason", "")),
        citations,
    )

    return {
        "is_valid": bool(result.get("is_valid", False)),
        "reason": reason,
        "citations": citations,
    }


def _save_edc_request_inputs(
    con: duckdb.DuckDBPyConnection,
    *,
    ticker: str,
    year: int,
    model: str,
    batch_id: str,
    prompts: dict[str, str],
) -> None:
    """Save exact EDC request inputs sent to OpenAI for audit/debug."""
    if prompts and all(str(code).endswith("_alt_two") for code in prompts.keys()):
        task_type = "edc_alt_two"
    elif prompts and all(str(code).endswith("_alt") for code in prompts.keys()):
        task_type = "edc_alt"
    else:
        task_type = "edc"
    rows: list[tuple] = []
    for code, prompt in prompts.items():
        request_body = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            },
            ensure_ascii=False,
        )
        rows.append(
            (
                task_type,
                ticker,
                year,
                code,
                model,
                batch_id,
                request_body,
                prompt,
            )
        )

    if not rows:
        return

    con.executemany(
        """
        INSERT INTO llm_request_inputs (
            task_type,
            ticker,
            year,
            item_code,
            model,
            batch_id,
            request_body,
            prompt_text
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


# ---------------------------------------------------------------------------
# Inference job management
# ---------------------------------------------------------------------------


def create_inference_jobs(
    con: duckdb.DuckDBPyConnection,
    *,
    tickers: list[str] | None = None,
    years: list[int] | None = None,
    replace: bool = False,
    inference_model: str | None = None,
    item_configs: list[dict[str, str]] | None = None,
) -> int:
    """Create pending inference jobs for embedded reports.

    Only reports that already have embeddings are eligible.
    Jobs are keyed by (ticker, year, model) so different models coexist.
    Returns the number of new jobs created.
    """
    inference_model = inference_model or INFERENCE_MODEL

    # Find reports with embeddings
    rows = con.execute(
        "SELECT DISTINCT ticker, year FROM document_embeddings"
    ).fetchall()
    embedded = [(r[0], r[1]) for r in rows]

    # Optional filters
    if tickers:
        upper = {t.upper() for t in tickers}
        embedded = [(t, y) for t, y in embedded if t in upper]
    if years:
        embedded = [(t, y) for t, y in embedded if y in years]

    created = 0
    total_cats = len(get_task_items("edc", item_configs))
    for ticker, year in embedded:
        existing = con.execute(
            "SELECT status FROM inference_jobs "
            "WHERE ticker = ? AND year = ? AND model = ?",
            [ticker, year, inference_model],
        ).fetchone()

        if existing is None:
            con.execute(
                """
                INSERT INTO inference_jobs
                    (id, ticker, year, model, status, categories_total)
                VALUES (nextval('inference_jobs_id_seq'), ?, ?, ?, 'pending', ?)
                """,
                [ticker, year, inference_model, total_cats],
            )
            created += 1
        elif replace and existing[0] in ("completed", "failed"):
            con.execute(
                """
                UPDATE inference_jobs
                SET status = 'pending', error_message = NULL,
                    started_at = NULL, completed_at = NULL,
                    categories_done = 0,
                    batch_id = NULL,
                    batch_submitted_at = NULL,
                    batch_checked_at = NULL
                WHERE ticker = ? AND year = ? AND model = ?
                """,
                [ticker, year, inference_model],
            )
            created += 1

    return created


def get_inference_jobs(
    con: duckdb.DuckDBPyConnection,
    *,
    status: str | None = None,
    model: str | None = None,
) -> list[dict]:
    """Return inference jobs, optionally filtered by status and/or model."""
    query = "SELECT * FROM inference_jobs"
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
        "categories_done",
        "categories_total",
        "batch_id",
        "batch_submitted_at",
        "batch_checked_at",
        "started_at",
        "completed_at",
        "error_message",
        "created_at",
    ]
    return [dict(zip(columns, row)) for row in rows]


def get_inference_job_counts(
    con: duckdb.DuckDBPyConnection,
    *,
    model: str | None = None,
) -> dict[str, int]:
    """Return ``{status: count}``, optionally filtered by model."""
    query = "SELECT status, COUNT(*) FROM inference_jobs"
    params: list = []
    if model:
        query += " WHERE model = ?"
        params.append(model)
    query += " GROUP BY status"
    rows = con.execute(query, params).fetchall()
    return {r[0]: r[1] for r in rows}


def reset_failed_inference_jobs(
    con: duckdb.DuckDBPyConnection,
    *,
    model: str | None = None,
) -> int:
    """Reset failed inference jobs back to pending, optionally for a model."""
    query = (
        "UPDATE inference_jobs SET status = 'pending', error_message = NULL, "
        "started_at = NULL, completed_at = NULL, categories_done = 0, "
        "batch_id = NULL, batch_submitted_at = NULL, batch_checked_at = NULL "
        "WHERE status = 'failed'"
    )
    params: list = []
    if model:
        query += " AND model = ?"
        params.append(model)
    con.execute(query, params)
    return 1


def sync_inference_job_states(
    con: duckdb.DuckDBPyConnection,
    *,
    model: str | None = None,
    item_configs: list[dict[str, str]] | None = None,
) -> dict[str, int]:
    """Sync inference job statuses with actual results in the database.

    For each non-completed job, counts the matching rows in
    ``inference_results`` (by ticker, year, model).  If all categories are
    present the job is marked **completed**; if some are present the job is
    marked **running** with ``categories_done`` updated; otherwise it stays
    as-is.

    Returns ``{"completed": N, "updated": M}`` — how many jobs were
    flipped to completed and how many had their progress updated.
    """
    total_cats = len(get_task_items("edc", item_configs))

    query = (
        "SELECT id, ticker, year, model, status, categories_done "
        "FROM inference_jobs WHERE status != 'completed'"
    )
    params: list = []
    if model:
        query += " AND model = ?"
        params.append(model)

    jobs = con.execute(query, params).fetchall()

    n_completed = 0
    n_updated = 0

    for job_id, ticker, year, job_model, status, cats_done in jobs:
        # Count results that exist for this (ticker, year, model)
        row = con.execute(
            f"SELECT COUNT(*) FROM {INFERENCE_RESULTS_WRITE_TABLE} "
            "WHERE ticker = ? AND year = ? AND model = ?",
            [ticker, year, job_model],
        ).fetchone()
        result_count = row[0] if row else 0

        if result_count >= total_cats:
            # All categories present → mark completed
            con.execute(
                """
                UPDATE inference_jobs
                SET status = 'completed',
                    completed_at = get_current_timestamp(),
                    categories_done = ?,
                    categories_total = ?,
                    error_message = NULL
                WHERE id = ?
                """,
                [total_cats, total_cats, job_id],
            )
            n_completed += 1
        elif result_count > 0 and result_count != cats_done:
            # Partial results — update progress
            con.execute(
                """
                UPDATE inference_jobs
                SET categories_done = ?, categories_total = ?
                WHERE id = ?
                """,
                [result_count, total_cats, job_id],
            )
            n_updated += 1

    return {"completed": n_completed, "updated": n_updated}


def delete_inference_jobs(
    con: duckdb.DuckDBPyConnection,
    *,
    ticker: str | None = None,
    year: int | None = None,
    status: str | None = None,
    model: str | None = None,
) -> None:
    """Delete inference jobs (and associated results) with optional filters."""
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

    # Also delete corresponding inference_results for HyDe-2 writes.
    con.execute(f"DELETE FROM {INFERENCE_RESULTS_WRITE_TABLE}{where}", params)
    con.execute(f"DELETE FROM inference_jobs{where}", params)


# ---------------------------------------------------------------------------
# Inference results helpers
# ---------------------------------------------------------------------------


def get_inference_results(
    con: duckdb.DuckDBPyConnection,
    *,
    ticker: str | None = None,
    year: int | None = None,
    model: str | None = None,
) -> list[dict]:
    """Return inference results, optionally filtered by ticker/year/model."""
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
        SELECT id, ticker, year, category_code, is_valid, reason,
               top_chunks, similarities, model, created_at
        FROM {INFERENCE_RESULTS_TABLE}
        {where}
        ORDER BY ticker, year, category_code
        """,
        params,
    ).fetchall()

    columns = [
        "id",
        "ticker",
        "year",
        "category_code",
        "is_valid",
        "reason",
        "top_chunks",
        "similarities",
        "model",
        "created_at",
    ]
    return [dict(zip(columns, row)) for row in rows]


# ---------------------------------------------------------------------------
# Core: run inference for a single report (all categories)
# ---------------------------------------------------------------------------


def infer_report(
    ticker: str,
    year: int,
    con: duckdb.DuckDBPyConnection | None = None,
    *,
    replace: bool = False,
    top_k: int | None = None,
    inference_model: str | None = None,
    embedding_model: str | None = None,
    dimensions: int | None = None,
    category_codes: list[str] | None = None,
    item_configs: list[dict[str, str]] | None = None,
) -> int:
    """Run checklist inference for a single annual report.

    For each category:
    1. Retrieve top-k chunks via vector similarity (RAG)
    2. Re-rank by cosine distance (DuckDB handles this natively)
    3. Send top chunks + criteria to GPT for validation
    4. Store the result in ``inference_results``

    Parameters
    ----------
    category_codes : list[str] | None
        If provided, only evaluate these specific checklist items
        (e.g. ``["CC1", "GHG3"]``).  When *None* (default), all items
        are evaluated.

    Returns the number of categories evaluated.
    """
    own_con = con is None
    if own_con:
        con = get_connection()
        ensure_vss_loaded(con)

    ticker = ticker.upper()
    top_k = top_k or INFERENCE_TOP_K
    inference_model = inference_model or INFERENCE_MODEL
    checklist_items = get_task_items("edc", item_configs)

    # Determine which checklist items to evaluate
    if category_codes:
        valid_codes = {it["code"] for it in checklist_items}
        unknown = set(category_codes) - valid_codes
        if unknown:
            raise ValueError(f"Unknown category codes: {unknown}")
        items_to_eval = [
            it for it in checklist_items if it["code"] in category_codes
        ]
    else:
        items_to_eval = checklist_items

    # ── Ensure an inference_jobs row exists ───────────────────────────────
    existing_job = con.execute(
        "SELECT status, batch_id FROM inference_jobs "
        "WHERE ticker = ? AND year = ? AND model = ?",
        [ticker, year, inference_model],
    ).fetchone()

    total_cats = len(checklist_items)

    if existing_job is None:
        con.execute(
            """
            INSERT INTO inference_jobs
                (id, ticker, year, model, status, categories_total)
            VALUES (nextval('inference_jobs_id_seq'), ?, ?, ?, 'pending', ?)
            """,
            [ticker, year, inference_model, total_cats],
        )

    # Mark job as running
    con.execute(
        """
        UPDATE inference_jobs
        SET status = 'running', started_at = get_current_timestamp(),
            top_k = ?, categories_total = ?,
            categories_done = 0, error_message = NULL
        WHERE ticker = ? AND year = ? AND model = ?
        """,
        [top_k, total_cats, ticker, year, inference_model],
    )

    try:
        batch_id = str(existing_job[1]) if existing_job and existing_job[1] else None
        client = _get_client()

        if batch_id:
            batch_status = get_batch_status(client, batch_id)
            con.execute(
                """
                UPDATE inference_jobs
                SET batch_checked_at = get_current_timestamp(),
                    status = CASE
                        WHEN ? = 'completed' THEN status
                        WHEN ? IN ('failed', 'expired', 'cancelled') THEN 'failed'
                        ELSE 'running'
                    END
                WHERE ticker = ? AND year = ? AND model = ?
                """,
                [batch_status, batch_status, ticker, year, inference_model],
            )

            if batch_status in {"validating", "in_progress", "finalizing", "cancelling"}:
                return 0

            if batch_status in {"failed", "expired", "cancelled"}:
                msg = f"Batch {batch_id} ended with status={batch_status}"
                con.execute(
                    """
                    UPDATE inference_jobs
                    SET status = 'failed', completed_at = get_current_timestamp(),
                        error_message = ?, batch_id = NULL
                    WHERE ticker = ? AND year = ? AND model = ?
                    """,
                    [msg, ticker, year, inference_model],
                )
                raise RuntimeError(msg)

            output_map = get_batch_output_map(client, batch_id)
            done_now = 0
            by_code = {it["code"]: it for it in checklist_items}

            for code, raw in output_map.items():
                item = by_code.get(code)
                if item is None:
                    continue

                chunks = retrieve_chunks_for_category(
                    con,
                    ticker,
                    year,
                    code,
                    top_k=top_k,
                    model=embedding_model,
                    dimensions=dimensions,
                    item_configs=checklist_items,
                )
                result = _parse_eval_raw(raw)

                similarities_json = json.dumps([round(c["distance"], 6) for c in chunks])
                top_chunks_json = json.dumps([c["chunk_index"] for c in chunks])

                con.execute(
                    """
                    INSERT INTO inference_results_hyde2
                        (id, ticker, year, category_code, is_valid, reason,
                         top_chunks, similarities, model)
                    VALUES
                        (nextval('inference_results_id_seq'),
                         ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (ticker, year, category_code, model) DO UPDATE SET
                        is_valid = EXCLUDED.is_valid,
                        reason = EXCLUDED.reason,
                        top_chunks = EXCLUDED.top_chunks,
                        similarities = EXCLUDED.similarities,
                        created_at = get_current_timestamp()
                    """,
                    [
                        ticker,
                        year,
                        code,
                        result["is_valid"],
                        result["reason"],
                        top_chunks_json,
                        similarities_json,
                        inference_model,
                    ],
                )
                done_now += 1

            total_results = con.execute(
                f"SELECT COUNT(*) FROM {INFERENCE_RESULTS_WRITE_TABLE} "
                "WHERE ticker = ? AND year = ? AND model = ?",
                [ticker, year, inference_model],
            ).fetchone()
            total_done = total_results[0] if total_results else 0

            if total_done >= total_cats:
                con.execute(
                    """
                    UPDATE inference_jobs
                    SET status = 'completed', completed_at = get_current_timestamp(),
                        categories_done = ?, batch_id = NULL
                    WHERE ticker = ? AND year = ? AND model = ?
                    """,
                    [total_done, ticker, year, inference_model],
                )
            else:
                con.execute(
                    """
                    UPDATE inference_jobs
                    SET status = 'pending', categories_done = ?,
                        batch_id = NULL
                    WHERE ticker = ? AND year = ? AND model = ?
                    """,
                    [total_done, ticker, year, inference_model],
                )

            return done_now

        # Check that embeddings exist for this report
        row = con.execute(
            "SELECT COUNT(*) FROM document_embeddings WHERE ticker = ? AND year = ?",
            [ticker, year],
        ).fetchone()
        emb_count = row[0] if row else 0

        if emb_count == 0:
            raise ValueError(
                f"No embeddings found for {ticker}/{year}. "
                "Embed the report first."
            )

        # Optionally clear previous results for this model
        if replace:
            con.execute(
                f"DELETE FROM {INFERENCE_RESULTS_WRITE_TABLE} "
                "WHERE ticker = ? AND year = ? AND model = ?",
                [ticker, year, inference_model],
            )

        done = 0
        prompts: dict[str, str] = {}
        for item in items_to_eval:
            code = item["code"]

            # Skip if result already exists for this model (unless replacing)
            if not replace:
                existing = con.execute(
                    f"SELECT 1 FROM {INFERENCE_RESULTS_WRITE_TABLE} "
                    "WHERE ticker = ? AND year = ? AND category_code = ? "
                    "AND model = ?",
                    [ticker, year, code, inference_model],
                ).fetchone()
                if existing:
                    done += 1
                    continue

            # 1. RAG retrieval + re-ranking via cosine distance
            chunks = retrieve_chunks_for_category(
                con,
                ticker,
                year,
                code,
                top_k=top_k,
                model=embedding_model,
                dimensions=dimensions,
                item_configs=checklist_items,
            )

            if not chunks:
                # No chunks — mark as not valid
                result = {
                    "is_valid": False,
                    "reason": "No relevant text chunks found in the report.",
                }
                similarities_json = json.dumps([round(c["distance"], 6) for c in chunks])
                top_chunks_json = json.dumps([c["chunk_index"] for c in chunks])
                con.execute(
                    """
                    INSERT INTO inference_results_hyde2
                        (id, ticker, year, category_code, is_valid, reason,
                         top_chunks, similarities, model)
                    VALUES
                        (nextval('inference_results_id_seq'),
                         ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (ticker, year, category_code, model) DO UPDATE SET
                        is_valid = EXCLUDED.is_valid,
                        reason = EXCLUDED.reason,
                        top_chunks = EXCLUDED.top_chunks,
                        similarities = EXCLUDED.similarities,
                        created_at = get_current_timestamp()
                    """,
                    [
                        ticker,
                        year,
                        code,
                        result["is_valid"],
                        result["reason"],
                        top_chunks_json,
                        similarities_json,
                        inference_model,
                    ],
                )
                done += 1
                continue

            prompts[code] = _build_eval_prompt(chunks, item["description"])

        # Submit all LLM requests for this report in one async batch.
        if prompts:
            is_reasoning = inference_model.startswith(("o1", "o3", "o4", "gpt-5"))
            new_batch_id = submit_chat_json_batch(
                client=client,
                model=inference_model,
                prompts=prompts,
                is_reasoning=is_reasoning,
                temperature=INFERENCE_TEMPERATURE,
            )
            _save_edc_request_inputs(
                con,
                ticker=ticker,
                year=year,
                model=inference_model,
                batch_id=new_batch_id,
                prompts=prompts,
            )
            con.execute(
                """
                UPDATE inference_jobs
                SET status = 'running',
                    categories_done = ?,
                    batch_id = ?,
                    batch_submitted_at = get_current_timestamp(),
                    batch_checked_at = get_current_timestamp()
                WHERE ticker = ? AND year = ? AND model = ?
                """,
                [done, new_batch_id, ticker, year, inference_model],
            )
            logger.info(
                "Submitted inference batch %s for %s/%d (%d categories)",
                new_batch_id,
                ticker,
                year,
                len(prompts),
            )
            if own_con:
                con.close()
            return done + len(prompts)

        # Mark job completed (or update progress if partial run)
        total_results = con.execute(
            f"SELECT COUNT(*) FROM {INFERENCE_RESULTS_WRITE_TABLE} "
            "WHERE ticker = ? AND year = ? AND model = ?",
            [ticker, year, inference_model],
        ).fetchone()
        total_done = total_results[0] if total_results else 0

        if total_done >= total_cats:
            con.execute(
                """
                UPDATE inference_jobs
                SET status = 'completed', completed_at = get_current_timestamp(),
                    categories_done = ?,
                    batch_id = NULL
                WHERE ticker = ? AND year = ? AND model = ?
                """,
                [total_done, ticker, year, inference_model],
            )
        else:
            # Partial run — update progress but keep running/pending status
            con.execute(
                """
                UPDATE inference_jobs
                SET categories_done = ?,
                    status = CASE WHEN status = 'running' THEN 'pending'
                                  ELSE status END,
                    batch_id = NULL
                WHERE ticker = ? AND year = ? AND model = ?
                """,
                [total_done, ticker, year, inference_model],
            )

        logger.info(
            "Inference complete for %s/%d — %d categories evaluated",
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
            UPDATE inference_jobs
            SET status = 'failed', completed_at = get_current_timestamp(),
                error_message = ?
            WHERE ticker = ? AND year = ? AND model = ?
            """,
            [str(exc), ticker, year, inference_model],
        )
        logger.error("Inference failed for %s/%d: %s", ticker, year, exc)
        if own_con:
            con.close()
        raise


# ---------------------------------------------------------------------------
# Batch: run inference for all pending jobs
# ---------------------------------------------------------------------------


def infer_all(
    *,
    replace: bool = False,
    progress_callback=None,
    top_k: int | None = None,
    inference_model: str | None = None,
    embedding_model: str | None = None,
    dimensions: int | None = None,
    item_configs: list[dict[str, str]] | None = None,
) -> dict:
    """Run inference for every pending inference job.

    Parameters
    ----------
    replace : bool
        Re-evaluate categories that already have results.
    progress_callback : callable, optional
        Called with ``(ticker, year, categories_done, i, total)`` after each
        report.
    top_k, inference_model, embedding_model, dimensions :
        Override config defaults.

    Returns
    -------
    dict
        Summary with keys ``evaluated``, ``skipped``, ``failed``.
    """
    con = get_connection()
    ensure_vss_loaded(con)

    _model = inference_model or INFERENCE_MODEL

    # Ensure jobs exist for all embedded reports (scoped to model)
    create_inference_jobs(
        con,
        replace=replace,
        inference_model=_model,
        item_configs=item_configs,
    )

    pending = get_inference_jobs(con, status="pending", model=_model)
    results = {"evaluated": [], "skipped": [], "failed": []}
    total = len(pending)

    for i, job in enumerate(pending):
        ticker, year = job["ticker"], job["year"]
        n = 0

        try:
            n = infer_report(
                ticker,
                year,
                con=con,
                replace=replace,
                top_k=top_k,
                inference_model=inference_model,
                embedding_model=embedding_model,
                dimensions=dimensions,
                item_configs=item_configs,
            )
            if n > 0:
                results["evaluated"].append((ticker, year, n))
            else:
                results["skipped"].append((ticker, year))
        except Exception as exc:
            logger.error("Inference failed for %s/%d: %s", ticker, year, exc)
            results["failed"].append((ticker, year, str(exc)))

        if progress_callback:
            progress_callback(ticker, year, n, i + 1, total)

    con.close()
    return results


def infer_category_all_reports(
    category_codes: list[str],
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
    """Run inference for specific checklist item(s) across all embedded reports.

    Parameters
    ----------
    category_codes : list[str]
        Checklist codes to evaluate (e.g. ``["CC1"]``).
    tickers, years :
        Optional filters to restrict which reports are processed.
    replace : bool
        Re-evaluate categories that already have results.
    progress_callback : callable, optional
        Called with ``(ticker, year, categories_done, i, total)`` after each
        report.

    Returns
    -------
    dict
        Summary with keys ``evaluated``, ``skipped``, ``failed``.
    """
    con = get_connection()
    ensure_vss_loaded(con)

    # Find all embedded reports
    rows = con.execute(
        "SELECT DISTINCT ticker, year FROM document_embeddings ORDER BY ticker, year"
    ).fetchall()
    reports = [(r[0], r[1]) for r in rows]

    # Apply optional filters
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
            n = infer_report(
                ticker,
                year,
                con=con,
                replace=replace,
                top_k=top_k,
                inference_model=inference_model,
                embedding_model=embedding_model,
                dimensions=dimensions,
                category_codes=category_codes,
            )
            if n > 0:
                results["evaluated"].append((ticker, year, n))
            else:
                results["skipped"].append((ticker, year))
        except Exception as exc:
            logger.error("Inference failed for %s/%d: %s", ticker, year, exc)
            results["failed"].append((ticker, year, str(exc)))

        if progress_callback:
            progress_callback(ticker, year, n, i + 1, total)

    con.close()
    return results
