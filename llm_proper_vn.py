"""
PROPER-VN Inference module — RAG-based environmental rating using
the PROPER-VN Simplified Adaptation Framework.

Two-stage evaluation:
  Stage 1  Regulatory compliance (→ Black / Red / Compliant)
  Stage 2  Beyond-compliance indicators (→ Blue / Green / Gold)

Results are stored in ``proper_vn_results``; batch progress is tracked
in ``proper_vn_jobs``.
"""

from __future__ import annotations

import json
import logging
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
    PROPER_VN_STAGE1_ITEMS,
    PROPER_VN_STAGE2_ITEMS,
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
PROPER_RESULTS_TABLE = "proper_vn_results"
PROPER_RESULTS_WRITE_TABLE = "proper_vn_results_hyde2"


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


def get_proper_vn_items() -> list[dict[str, str]]:
    """Return all PROPER-VN checklist items (Stage 1 + Stage 2)."""
    return get_task_items("proper_vn")


def get_proper_vn_groups() -> dict[str, list[dict]]:
    """Return PROPER-VN items grouped by stage."""
    proper_items = get_task_items("proper_vn")
    groups: dict[str, list[dict]] = {}
    for item in proper_items:
        groups.setdefault(item["group"], []).append(item)
    return groups


# ---------------------------------------------------------------------------
# Category embedding cache
# ---------------------------------------------------------------------------

_proper_category_embeddings: dict[str, list[float]] | None = None


def _get_proper_category_embeddings(
    model: str | None = None,
    dimensions: int | None = None,
    item_configs: list[dict[str, str]] | None = None,
) -> dict[str, list[float]]:
    global _proper_category_embeddings
    if item_configs is None and _proper_category_embeddings is not None:
        return _proper_category_embeddings

    model = model or EMBEDDING_MODEL
    dimensions = dimensions or EMBEDDING_DIMENSIONS
    proper_items = get_task_items("proper_vn", item_configs)

    # Criteria are static (not ticker/year-specific), so precompute HyDE upfront.
    precompute_hyde2_embeddings(
        [item["description"] for item in proper_items],
        embedding_model=model,
        dimensions=dimensions,
    )

    embeddings: dict[str, list[float]] = {}
    for item in proper_items:
        code = item["code"]
        description = item["description"]
        embeddings[code] = build_hyde2_embedding(
            description,
            embedding_model=model,
            dimensions=dimensions,
        )
    if item_configs is None:
        _proper_category_embeddings = embeddings
    return _proper_category_embeddings


def reset_proper_category_embeddings_cache() -> None:
    global _proper_category_embeddings
    _proper_category_embeddings = None


# ---------------------------------------------------------------------------
# RAG retrieval
# ---------------------------------------------------------------------------


def retrieve_chunks_for_indicator(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    year: int,
    indicator_code: str,
    *,
    top_k: int | None = None,
    model: str | None = None,
    dimensions: int | None = None,
    item_configs: list[dict[str, str]] | None = None,
) -> list[dict]:
    """Retrieve top-k most relevant chunks for a PROPER-VN indicator."""
    top_k = top_k or INFERENCE_TOP_K
    dimensions = dimensions or EMBEDDING_DIMENSIONS

    cat_embeddings = _get_proper_category_embeddings(
        model=model, dimensions=dimensions, item_configs=item_configs
    )
    if indicator_code not in cat_embeddings:
        raise ValueError(f"Unknown PROPER-VN indicator: {indicator_code}")

    query_emb = cat_embeddings[indicator_code]

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

_PROPER_EVAL_PROMPT = """You are a helpful assistant designed to evaluate annual reports against PROPER-VN environmental indicators. You will be given Vietnamese text from an annual report and a specific environmental indicator to check.

- Translate the text to English.

**Return only a JSON object** with the following properties:

- `"is_present"`: a boolean (`true` or `false`) indicating whether evidence for this indicator is found in the text.
- `"evidence_level"`: one of `"none"`, `"basic_mention"`, or `"quantified"`.
  - `"none"` — no relevant information found.
  - `"basic_mention"` — the topic is mentioned but without specific data.
  - `"quantified"` — specific numbers, targets, or detailed evidence provided.
- `"reason"`: A brief explanation supporting your assessment.

All JSON properties must always be present.

Do not include any additional text or explanations outside the JSON object.

TEXT DATA:
```
{content}
```

INDICATOR:
```
{criteria}
```"""


def evaluate_proper_indicator(
    chunks: list[dict],
    indicator_code: str,
    indicator_description: str,
    *,
    model: str | None = None,
) -> dict:
    """Send retrieved chunks to the LLM for PROPER-VN indicator evaluation.

    Returns dict with keys: ``is_present``, ``evidence_level``, ``reason``.
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
    prompt = _PROPER_EVAL_PROMPT.format(
        content=content, criteria=indicator_description
    )

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
                prompt = _PROPER_EVAL_PROMPT.format(
                    content=content, criteria=indicator_description
                )
                logger.warning(
                    "Prompt too large for %s; shrinking and retrying (attempt %d/3)",
                    model,
                    attempt + 1,
                )
                continue
            raise
    else:
        raise last_exc  # type: ignore[misc]

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "is_present": False,
            "evidence_level": "none",
            "reason": f"Failed to parse LLM response: {raw}",
        }

    return {
        "is_present": bool(result.get("is_present", False)),
        "evidence_level": str(result.get("evidence_level", "none")),
        "reason": str(result.get("reason", "")),
    }


def _build_proper_prompt(chunks: list[dict], indicator_description: str) -> str:
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
        break

    if not selected_chunks:
        selected_chunks = [""]

    content = "\n".join(selected_chunks)
    return _PROPER_EVAL_PROMPT.format(
        content=content, criteria=indicator_description
    )


def _save_proper_request_inputs(
    con: duckdb.DuckDBPyConnection,
    *,
    ticker: str,
    year: int,
    model: str,
    batch_id: str,
    prompts: dict[str, str],
) -> None:
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
                "proper_vn",
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


def _parse_proper_raw(raw: str) -> dict:
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "is_present": False,
            "evidence_level": "none",
            "reason": f"Failed to parse LLM response: {raw}",
        }

    return {
        "is_present": bool(result.get("is_present", False)),
        "evidence_level": str(result.get("evidence_level", "none")),
        "reason": str(result.get("reason", "")),
    }


# ---------------------------------------------------------------------------
# Color classification logic
# ---------------------------------------------------------------------------

# Evidence-level scoring for Stage 2 indicators
_EVIDENCE_SCORES = {"none": 0, "basic_mention": 1, "quantified": 2}


def classify_proper_color(
    stage1_results: dict[str, dict],
    stage2_results: dict[str, dict],
) -> dict:
    """Determine the PROPER-VN color rating from indicator results.

    Parameters
    ----------
    stage1_results : dict
        Mapping of S1 indicator codes → evaluation result dicts.
    stage2_results : dict
        Mapping of S2 indicator codes → evaluation result dicts.

    Returns
    -------
    dict
        ``color``, ``stage``, ``reason``, ``s2_score``, ``s2_max_score``.
    """
    # Stage 1: Check for violations
    violation = stage1_results.get("S1_VIOLATION", {})
    minor_nc = stage1_results.get("S1_MINOR_NC", {})
    compliance = stage1_results.get("S1_COMPLIANCE", {})

    if violation.get("is_present"):
        return {
            "color": "Black",
            "stage": 1,
            "reason": f"Serious violation detected: {violation.get('reason', '')}",
            "s2_score": 0,
            "s2_max_score": len(PROPER_VN_STAGE2_ITEMS) * 2,
        }

    if minor_nc.get("is_present") and not compliance.get("is_present"):
        return {
            "color": "Red",
            "stage": 1,
            "reason": f"Minor non-compliance without stated compliance: {minor_nc.get('reason', '')}",
            "s2_score": 0,
            "s2_max_score": len(PROPER_VN_STAGE2_ITEMS) * 2,
        }

    # Stage 2: Beyond-compliance scoring
    s2_max = len(PROPER_VN_STAGE2_ITEMS) * 2  # max 2 per indicator
    s2_score = 0
    for code, res in stage2_results.items():
        level = res.get("evidence_level", "none")
        s2_score += _EVIDENCE_SCORES.get(level, 0)

    # Classification thresholds
    # Gold:  ≥75% of max score
    # Green: ≥37.5% of max score
    # Blue:  everything else (compliant but low engagement)
    gold_threshold = s2_max * 0.75
    green_threshold = s2_max * 0.375

    if s2_score >= gold_threshold:
        color = "Gold"
        reason = (
            f"Strong beyond-compliance engagement (score {s2_score}/{s2_max})"
        )
    elif s2_score >= green_threshold:
        color = "Green"
        reason = f"Moderate beyond-compliance engagement (score {s2_score}/{s2_max})"
    else:
        color = "Blue"
        reason = f"Compliant with low beyond-compliance engagement (score {s2_score}/{s2_max})"

    return {
        "color": color,
        "stage": 2,
        "reason": reason,
        "s2_score": s2_score,
        "s2_max_score": s2_max,
    }


# ---------------------------------------------------------------------------
# Job management
# ---------------------------------------------------------------------------


def create_proper_vn_jobs(
    con: duckdb.DuckDBPyConnection,
    *,
    tickers: list[str] | None = None,
    years: list[int] | None = None,
    replace: bool = False,
    inference_model: str | None = None,
    item_configs: list[dict[str, str]] | None = None,
) -> int:
    """Create pending PROPER-VN jobs for embedded reports."""
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

    total_indicators = len(get_task_items("proper_vn", item_configs))
    created = 0
    for ticker, year in embedded:
        existing = con.execute(
            "SELECT status FROM proper_vn_jobs "
            "WHERE ticker = ? AND year = ? AND model = ?",
            [ticker, year, inference_model],
        ).fetchone()

        if existing is None:
            con.execute(
                """
                INSERT INTO proper_vn_jobs
                    (id, ticker, year, model, status, indicators_total)
                VALUES (nextval('proper_vn_jobs_id_seq'), ?, ?, ?, 'pending', ?)
                """,
                [ticker, year, inference_model, total_indicators],
            )
            created += 1
        elif replace and existing[0] in ("completed", "failed"):
            con.execute(
                """
                UPDATE proper_vn_jobs
                SET status = 'pending', error_message = NULL,
                    started_at = NULL, completed_at = NULL,
                    indicators_done = 0, color = NULL,
                    batch_id = NULL,
                    batch_submitted_at = NULL,
                    batch_checked_at = NULL
                WHERE ticker = ? AND year = ? AND model = ?
                """,
                [ticker, year, inference_model],
            )
            created += 1

    return created


def get_proper_vn_jobs(
    con: duckdb.DuckDBPyConnection,
    *,
    status: str | None = None,
    model: str | None = None,
) -> list[dict]:
    """Return PROPER-VN jobs, optionally filtered."""
    query = "SELECT * FROM proper_vn_jobs"
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
        "indicators_done",
        "indicators_total",
        "batch_id",
        "batch_submitted_at",
        "batch_checked_at",
        "color",
        "s2_score",
        "s2_max_score",
        "started_at",
        "completed_at",
        "error_message",
        "created_at",
    ]
    return [dict(zip(columns, row)) for row in rows]


def get_proper_vn_job_counts(
    con: duckdb.DuckDBPyConnection,
    *,
    model: str | None = None,
) -> dict[str, int]:
    query = "SELECT status, COUNT(*) FROM proper_vn_jobs"
    params: list = []
    if model:
        query += " WHERE model = ?"
        params.append(model)
    query += " GROUP BY status"
    rows = con.execute(query, params).fetchall()
    return {r[0]: r[1] for r in rows}


def reset_failed_proper_vn_jobs(
    con: duckdb.DuckDBPyConnection,
    *,
    model: str | None = None,
) -> int:
    query = (
        "UPDATE proper_vn_jobs SET status = 'pending', error_message = NULL, "
        "started_at = NULL, completed_at = NULL, indicators_done = 0, "
        "color = NULL, batch_id = NULL, batch_submitted_at = NULL, "
        "batch_checked_at = NULL WHERE status = 'failed'"
    )
    params: list = []
    if model:
        query += " AND model = ?"
        params.append(model)
    con.execute(query, params)
    return 1


def sync_proper_vn_job_states(
    con: duckdb.DuckDBPyConnection,
    *,
    model: str | None = None,
    item_configs: list[dict[str, str]] | None = None,
) -> dict[str, int]:
    """Sync PROPER-VN job statuses with actual results in DB."""
    total_indicators = len(get_task_items("proper_vn", item_configs))

    query = (
        "SELECT id, ticker, year, model, status, indicators_done "
        "FROM proper_vn_jobs WHERE status != 'completed'"
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
            f"SELECT COUNT(*) FROM {PROPER_RESULTS_WRITE_TABLE} "
            "WHERE ticker = ? AND year = ? AND model = ?",
            [ticker, year, job_model],
        ).fetchone()
        result_count = row[0] if row else 0

        if result_count >= total_indicators:
            con.execute(
                """
                UPDATE proper_vn_jobs
                SET status = 'completed',
                    completed_at = get_current_timestamp(),
                    indicators_done = ?, indicators_total = ?,
                    error_message = NULL
                WHERE id = ?
                """,
                [total_indicators, total_indicators, job_id],
            )
            n_completed += 1
        elif result_count > 0 and result_count != done:
            con.execute(
                """
                UPDATE proper_vn_jobs
                SET indicators_done = ?, indicators_total = ?
                WHERE id = ?
                """,
                [result_count, total_indicators, job_id],
            )
            n_updated += 1

    return {"completed": n_completed, "updated": n_updated}


def delete_proper_vn_jobs(
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

    con.execute(f"DELETE FROM {PROPER_RESULTS_WRITE_TABLE}{where}", params)
    con.execute(f"DELETE FROM proper_vn_jobs{where}", params)


# ---------------------------------------------------------------------------
# Results helpers
# ---------------------------------------------------------------------------


def get_proper_vn_results(
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
        SELECT id, ticker, year, indicator_code, is_present,
               evidence_level, reason, top_chunks, similarities,
               model, created_at
        FROM {PROPER_RESULTS_TABLE}
        {where}
        ORDER BY ticker, year, indicator_code
        """,
        params,
    ).fetchall()

    columns = [
        "id",
        "ticker",
        "year",
        "indicator_code",
        "is_present",
        "evidence_level",
        "reason",
        "top_chunks",
        "similarities",
        "model",
        "created_at",
    ]
    return [dict(zip(columns, row)) for row in rows]


# ---------------------------------------------------------------------------
# Core: run PROPER-VN inference for a single report
# ---------------------------------------------------------------------------


def infer_proper_vn_report(
    ticker: str,
    year: int,
    con: duckdb.DuckDBPyConnection | None = None,
    *,
    replace: bool = False,
    top_k: int | None = None,
    inference_model: str | None = None,
    embedding_model: str | None = None,
    dimensions: int | None = None,
    item_configs: list[dict[str, str]] | None = None,
) -> dict:
    """Run PROPER-VN inference for a single annual report.

    Evaluates all Stage 1 + Stage 2 indicators, stores individual results
    in ``proper_vn_results``, classifies the final PROPER-VN color, and
    updates the job row.

    Returns the classification dict (color, stage, reason, s2_score, …).
    """
    own_con = con is None
    if own_con:
        con = get_connection()
        ensure_vss_loaded(con)

    ticker = ticker.upper()
    top_k = top_k or INFERENCE_TOP_K
    inference_model = inference_model or INFERENCE_MODEL
    proper_items = get_task_items("proper_vn", item_configs)
    total_indicators = len(proper_items)

    # ── Ensure a job row exists ──────────────────────────────────────────
    existing_job = con.execute(
        "SELECT status, batch_id FROM proper_vn_jobs "
        "WHERE ticker = ? AND year = ? AND model = ?",
        [ticker, year, inference_model],
    ).fetchone()

    if existing_job is None:
        con.execute(
            """
            INSERT INTO proper_vn_jobs
                (id, ticker, year, model, status, indicators_total)
            VALUES (nextval('proper_vn_jobs_id_seq'), ?, ?, ?, 'pending', ?)
            """,
            [ticker, year, inference_model, total_indicators],
        )

    # Mark running
    con.execute(
        """
        UPDATE proper_vn_jobs
        SET status = 'running', started_at = get_current_timestamp(),
            top_k = ?, indicators_total = ?,
            indicators_done = 0, error_message = NULL, color = NULL
        WHERE ticker = ? AND year = ? AND model = ?
        """,
        [top_k, total_indicators, ticker, year, inference_model],
    )

    try:
        batch_id = str(existing_job[1]) if existing_job and existing_job[1] else None
        client = _get_client()

        if batch_id:
            batch_status = get_batch_status(client, batch_id)
            con.execute(
                """
                UPDATE proper_vn_jobs
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
                return {
                    "color": "BatchSubmitted",
                    "stage": 0,
                    "reason": f"Waiting for OpenAI batch {batch_id}",
                    "s2_score": 0,
                    "s2_max_score": len(PROPER_VN_STAGE2_ITEMS) * 2,
                }

            if batch_status in {"failed", "expired", "cancelled"}:
                msg = f"Batch {batch_id} ended with status={batch_status}"
                con.execute(
                    """
                    UPDATE proper_vn_jobs
                    SET status = 'failed', completed_at = get_current_timestamp(),
                        error_message = ?, batch_id = NULL
                    WHERE ticker = ? AND year = ? AND model = ?
                    """,
                    [msg, ticker, year, inference_model],
                )
                raise RuntimeError(msg)

            output_map = get_batch_output_map(client, batch_id)
            by_code = {it["code"]: it for it in proper_items}

            for code, raw in output_map.items():
                if code not in by_code:
                    continue
                chunks = retrieve_chunks_for_indicator(
                    con,
                    ticker,
                    year,
                    code,
                    top_k=top_k,
                    model=embedding_model,
                    dimensions=dimensions,
                    item_configs=proper_items,
                )
                result = _parse_proper_raw(raw)
                similarities_json = json.dumps([round(c["distance"], 6) for c in chunks])
                top_chunks_json = json.dumps([c["chunk_index"] for c in chunks])

                con.execute(
                    """
                    INSERT INTO proper_vn_results_hyde2
                        (id, ticker, year, indicator_code, is_present,
                         evidence_level, reason, top_chunks, similarities, model)
                    VALUES
                        (nextval('proper_vn_results_id_seq'),
                         ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (ticker, year, indicator_code, model) DO UPDATE SET
                        is_present = EXCLUDED.is_present,
                        evidence_level = EXCLUDED.evidence_level,
                        reason = EXCLUDED.reason,
                        top_chunks = EXCLUDED.top_chunks,
                        similarities = EXCLUDED.similarities,
                        created_at = get_current_timestamp()
                    """,
                    [
                        ticker,
                        year,
                        code,
                        result["is_present"],
                        result["evidence_level"],
                        result["reason"],
                        top_chunks_json,
                        similarities_json,
                        inference_model,
                    ],
                )

            rows = con.execute(
                "SELECT indicator_code, is_present, evidence_level, reason "
                f"FROM {PROPER_RESULTS_WRITE_TABLE} WHERE ticker = ? AND year = ? AND model = ?",
                [ticker, year, inference_model],
            ).fetchall()
            all_indicator_results = {
                str(r[0]): {
                    "is_present": bool(r[1]),
                    "evidence_level": str(r[2]),
                    "reason": str(r[3]),
                }
                for r in rows
            }
            total_done = len(all_indicator_results)

            if total_done >= total_indicators:
                stage1_results = {
                    c: all_indicator_results[c]
                    for c in [i["code"] for i in PROPER_VN_STAGE1_ITEMS]
                    if c in all_indicator_results
                }
                stage2_results = {
                    c: all_indicator_results[c]
                    for c in [i["code"] for i in PROPER_VN_STAGE2_ITEMS]
                    if c in all_indicator_results
                }
                classification = classify_proper_color(stage1_results, stage2_results)
                con.execute(
                    """
                    UPDATE proper_vn_jobs
                    SET status = 'completed', completed_at = get_current_timestamp(),
                        indicators_done = ?, color = ?,
                        s2_score = ?, s2_max_score = ?,
                        batch_id = NULL
                    WHERE ticker = ? AND year = ? AND model = ?
                    """,
                    [
                        total_done,
                        classification["color"],
                        classification["s2_score"],
                        classification["s2_max_score"],
                        ticker,
                        year,
                        inference_model,
                    ],
                )
                if own_con:
                    con.close()
                return classification

            con.execute(
                """
                UPDATE proper_vn_jobs
                SET status = 'pending', indicators_done = ?, batch_id = NULL
                WHERE ticker = ? AND year = ? AND model = ?
                """,
                [total_done, ticker, year, inference_model],
            )
            if own_con:
                con.close()
            return {
                "color": "Pending",
                "stage": 0,
                "reason": "Partial PROPER-VN results loaded from batch output",
                "s2_score": 0,
                "s2_max_score": len(PROPER_VN_STAGE2_ITEMS) * 2,
            }

        # Check embeddings exist
        row = con.execute(
            "SELECT COUNT(*) FROM document_embeddings "
            "WHERE ticker = ? AND year = ?",
            [ticker, year],
        ).fetchone()
        if (row[0] if row else 0) == 0:
            raise ValueError(
                f"No embeddings found for {ticker}/{year}. "
                "Embed the report first."
            )

        # Optionally clear previous results for this model
        if replace:
            con.execute(
                f"DELETE FROM {PROPER_RESULTS_WRITE_TABLE} "
                "WHERE ticker = ? AND year = ? AND model = ?",
                [ticker, year, inference_model],
            )

        done = 0
        all_indicator_results: dict[str, dict] = {}
        prompts: dict[str, str] = {}

        for item in proper_items:
            code = item["code"]

            # Skip if result exists (unless replacing)
            if not replace:
                existing = con.execute(
                    "SELECT is_present, evidence_level, reason "
                    f"FROM {PROPER_RESULTS_WRITE_TABLE} "
                    "WHERE ticker = ? AND year = ? AND indicator_code = ? "
                    "AND model = ?",
                    [ticker, year, code, inference_model],
                ).fetchone()
                if existing:
                    all_indicator_results[code] = {
                        "is_present": existing[0],
                        "evidence_level": existing[1],
                        "reason": existing[2],
                    }
                    done += 1
                    continue

            # RAG retrieval
            chunks = retrieve_chunks_for_indicator(
                con,
                ticker,
                year,
                code,
                top_k=top_k,
                model=embedding_model,
                dimensions=dimensions,
                item_configs=proper_items,
            )

            if not chunks:
                result = {
                    "is_present": False,
                    "evidence_level": "none",
                    "reason": "No relevant text chunks found in the report.",
                }
                all_indicator_results[code] = result
                similarities_json = json.dumps([round(c["distance"], 6) for c in chunks])
                top_chunks_json = json.dumps([c["chunk_index"] for c in chunks])
                con.execute(
                    """
                    INSERT INTO proper_vn_results_hyde2
                        (id, ticker, year, indicator_code, is_present,
                         evidence_level, reason, top_chunks, similarities, model)
                    VALUES
                        (nextval('proper_vn_results_id_seq'),
                         ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (ticker, year, indicator_code, model) DO UPDATE SET
                        is_present = EXCLUDED.is_present,
                        evidence_level = EXCLUDED.evidence_level,
                        reason = EXCLUDED.reason,
                        top_chunks = EXCLUDED.top_chunks,
                        similarities = EXCLUDED.similarities,
                        created_at = get_current_timestamp()
                    """,
                    [
                        ticker,
                        year,
                        code,
                        result["is_present"],
                        result["evidence_level"],
                        result["reason"],
                        top_chunks_json,
                        similarities_json,
                        inference_model,
                    ],
                )
                done += 1
                continue
            else:
                prompts[code] = _build_proper_prompt(chunks, item["description"])

        if prompts:
            is_reasoning = inference_model.startswith(("o1", "o3", "o4", "gpt-5"))
            new_batch_id = submit_chat_json_batch(
                client=client,
                model=inference_model,
                prompts=prompts,
                is_reasoning=is_reasoning,
                temperature=INFERENCE_TEMPERATURE,
            )
            _save_proper_request_inputs(
                con,
                ticker=ticker,
                year=year,
                model=inference_model,
                batch_id=new_batch_id,
                prompts=prompts,
            )
            con.execute(
                """
                UPDATE proper_vn_jobs
                SET status = 'running', indicators_done = ?,
                    batch_id = ?,
                    batch_submitted_at = get_current_timestamp(),
                    batch_checked_at = get_current_timestamp()
                WHERE ticker = ? AND year = ? AND model = ?
                """,
                [done, new_batch_id, ticker, year, inference_model],
            )
            if own_con:
                con.close()
            return {
                "color": "BatchSubmitted",
                "stage": 0,
                "reason": f"Submitted OpenAI batch {new_batch_id} ({len(prompts)} indicators)",
                "s2_score": 0,
                "s2_max_score": len(PROPER_VN_STAGE2_ITEMS) * 2,
            }

        # ── Classify color ───────────────────────────────────────────────
        stage1_results = {
            c: all_indicator_results[c]
            for c in [i["code"] for i in PROPER_VN_STAGE1_ITEMS]
            if c in all_indicator_results
        }
        stage2_results = {
            c: all_indicator_results[c]
            for c in [i["code"] for i in PROPER_VN_STAGE2_ITEMS]
            if c in all_indicator_results
        }
        classification = classify_proper_color(stage1_results, stage2_results)

        # Update job with final color
        con.execute(
            """
            UPDATE proper_vn_jobs
            SET status = 'completed', completed_at = get_current_timestamp(),
                indicators_done = ?, color = ?,
                s2_score = ?, s2_max_score = ?
            WHERE ticker = ? AND year = ? AND model = ?
            """,
            [
                done,
                classification["color"],
                classification["s2_score"],
                classification["s2_max_score"],
                ticker,
                year,
                inference_model,
            ],
        )

        logger.info(
            "PROPER-VN complete for %s/%d — color: %s (%d indicators)",
            ticker,
            year,
            classification["color"],
            done,
        )

        if own_con:
            con.close()
        return classification

    except Exception as exc:
        con.execute(
            """
            UPDATE proper_vn_jobs
            SET status = 'failed', completed_at = get_current_timestamp(),
                error_message = ?
            WHERE ticker = ? AND year = ? AND model = ?
            """,
            [str(exc), ticker, year, inference_model],
        )
        logger.error("PROPER-VN failed for %s/%d: %s", ticker, year, exc)
        if own_con:
            con.close()
        raise


# ---------------------------------------------------------------------------
# Batch: run PROPER-VN for all pending jobs
# ---------------------------------------------------------------------------


def infer_proper_vn_all(
    *,
    replace: bool = False,
    progress_callback=None,
    top_k: int | None = None,
    inference_model: str | None = None,
    embedding_model: str | None = None,
    dimensions: int | None = None,
    item_configs: list[dict[str, str]] | None = None,
) -> dict:
    """Run PROPER-VN inference for every pending job.

    Returns summary with keys ``evaluated``, ``skipped``, ``failed``.
    """
    con = get_connection()
    ensure_vss_loaded(con)

    _model = inference_model or INFERENCE_MODEL
    create_proper_vn_jobs(
        con,
        replace=replace,
        inference_model=_model,
        item_configs=item_configs,
    )

    pending = get_proper_vn_jobs(con, status="pending", model=_model)
    results = {"evaluated": [], "skipped": [], "failed": []}
    total = len(pending)

    for i, job in enumerate(pending):
        ticker, year = job["ticker"], job["year"]
        classification = None

        try:
            classification = infer_proper_vn_report(
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
            results["evaluated"].append(
                (ticker, year, classification["color"])
            )
        except Exception as exc:
            logger.error("PROPER-VN failed for %s/%d: %s", ticker, year, exc)
            results["failed"].append((ticker, year, str(exc)))

        if progress_callback:
            color = classification["color"] if classification else "?"
            progress_callback(ticker, year, color, i + 1, total)

    con.close()
    return results
