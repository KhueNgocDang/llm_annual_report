"""Financial-statement audit extraction pipeline.

This module is intentionally separate from annual-report governance extraction.
It uses dedicated DuckDB tables:
- financial_statement_reports
- financial_statement_document_embeddings
- financial_statement_audit_results
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import duckdb

from config import (
    FINANCIAL_STATEMENT_MARKDOWN_DIR,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EMBEDDING_CHUNK_OVERLAP,
    EMBEDDING_CHUNK_SIZE,
    INFERENCE_MODEL,
    INFERENCE_TEMPERATURE,
    INFERENCE_TOP_K,
)
from database import ensure_vss_loaded, get_connection
from embedder import _get_client, get_embeddings
from llm_batch_api import run_chat_json_batch
from llm_embeddings import chunk_markdown_content

logger = logging.getLogger(__name__)

try:
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None


MAX_PROMPT_CONTENT_TOKENS = 120000
_FINANCIAL_STATEMENT_AUDIT_QUERY = (
    "Trich xuat thong tin kiem toan doc lap trong bao cao tai chinh: ten cong ty kiem toan, "
    "y kien kiem toan (chap nhan toan phan/ngoai tru/tu choi/khong dua ra y kien), "
    "va kiem toan vien ky bao cao."
)

_FINANCIAL_STATEMENT_AUDIT_PROMPT = """You are a structured data extraction assistant for Vietnamese audited consolidated financial statements.

Extract external audit information from the provided financial-statement text.

Return ONLY a JSON object with exactly these keys:
- "found": boolean
- "external_audit_firm": string or null
- "external_audit_firm_en": string or null
- "audit_opinion": one of ["unqualified", "qualified", "adverse", "disclaimer", null]
- "signing_auditor_names": array of strings
- "details": array of objects (firm/auditor evidence records)
- "reason": short explanation

Rules:
- Focus only on independent external audit report sections.
- Exclude internal audit committee, board, supervisory board, or management roster info.

TEXT DATA:
```
{content}
```
"""

_audit_query_embedding: list[float] | None = None


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


def _extract_year(text: str) -> int | None:
    m = re.search(r"(20\d{2})", text)
    return int(m.group(1)) if m else None


def _safe_json_load(raw: Any, default: Any) -> Any:
    if raw is None:
        return default
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(str(raw))
    except Exception:
        return default


def _get_audit_query_embedding(
    model: str | None = None,
    dimensions: int | None = None,
) -> list[float]:
    global _audit_query_embedding
    if _audit_query_embedding is not None:
        return _audit_query_embedding

    vectors = get_embeddings(
        [_FINANCIAL_STATEMENT_AUDIT_QUERY],
        model=model or EMBEDDING_MODEL,
        dimensions=dimensions or EMBEDDING_DIMENSIONS,
    )
    _audit_query_embedding = vectors[0]
    return _audit_query_embedding


def sync_financial_statement_reports_from_markdown(
    con: duckdb.DuckDBPyConnection,
    *,
    source_dir: str | Path = FINANCIAL_STATEMENT_MARKDOWN_DIR,
    tickers: list[str] | None = None,
    years: list[int] | None = None,
) -> dict[str, int]:
    """Load financial-statement markdown files into financial_statement_reports."""
    source_path = Path(source_dir)
    if not source_path.exists():
        return {"loaded": 0, "failed": 0}

    ticker_filter = {t.upper() for t in (tickers or [])}
    year_filter = set(years or [])

    loaded = 0
    failed = 0

    for md_path in sorted(source_path.rglob("*.md")):
        try:
            rel = md_path.relative_to(source_path)
            if not rel.parts:
                continue

            ticker = rel.parts[0].upper()
            if ticker_filter and ticker not in ticker_filter:
                continue

            year = (
                _extract_year(md_path.stem)
                or _extract_year(md_path.parent.name)
                or _extract_year(str(rel))
            )
            if year is None:
                continue
            if year_filter and year not in year_filter:
                continue

            content = md_path.read_text(encoding="utf-8")
            source_file = str(Path(source_path.name) / rel)

            con.execute(
                """
                INSERT INTO financial_statement_reports (ticker, year, content, source_file)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (ticker, year) DO UPDATE SET
                    content = EXCLUDED.content,
                    source_file = EXCLUDED.source_file,
                    created_at = get_current_timestamp()
                """,
                [ticker, year, content, source_file],
            )
            loaded += 1
        except Exception:
            failed += 1

    return {"loaded": loaded, "failed": failed}


def embed_financial_statement_report(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    year: int,
    *,
    replace: bool = False,
    model: str = EMBEDDING_MODEL,
    dimensions: int = EMBEDDING_DIMENSIONS,
    chunk_size: int = EMBEDDING_CHUNK_SIZE,
    chunk_overlap: int = EMBEDDING_CHUNK_OVERLAP,
) -> dict[str, int | str | bool]:
    """Embed one financial statement into financial_statement_document_embeddings."""
    ticker_u = ticker.upper()
    row = con.execute(
        "SELECT content FROM financial_statement_reports WHERE ticker = ? AND year = ?",
        [ticker_u, year],
    ).fetchone()
    if not row:
        return {
            "ticker": ticker_u,
            "year": year,
            "embedded": 0,
            "skipped": 1,
            "failed": 0,
            "reason": "financial_statement_report_not_found",
        }

    existing = con.execute(
        "SELECT COUNT(*) FROM financial_statement_document_embeddings WHERE ticker = ? AND year = ?",
        [ticker_u, year],
    ).fetchone()[0]
    if existing > 0 and not replace:
        return {
            "ticker": ticker_u,
            "year": year,
            "embedded": 0,
            "skipped": 1,
            "failed": 0,
            "reason": "already_embedded",
        }

    if replace:
        con.execute(
            "DELETE FROM financial_statement_document_embeddings WHERE ticker = ? AND year = ?",
            [ticker_u, year],
        )

    chunks = chunk_markdown_content(
        str(row[0] or ""),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        return {
            "ticker": ticker_u,
            "year": year,
            "embedded": 0,
            "skipped": 1,
            "failed": 0,
            "reason": "empty_content",
        }

    vectors = get_embeddings(
        [c.chunk_text for c in chunks],
        model=model,
        dimensions=dimensions,
    )

    con.executemany(
        """
        INSERT INTO financial_statement_document_embeddings
            (ticker, year, chunk_index, chunk_text, token_count, embedding, model)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (ticker, year, chunk_index) DO UPDATE SET
            chunk_text = EXCLUDED.chunk_text,
            token_count = EXCLUDED.token_count,
            embedding = EXCLUDED.embedding,
            model = EXCLUDED.model,
            created_at = get_current_timestamp()
        """,
        [
            (
                ticker_u,
                year,
                chunks[i].chunk_index,
                chunks[i].chunk_text,
                chunks[i].token_count,
                vectors[i],
                model,
            )
            for i in range(len(chunks))
        ],
    )

    return {
        "ticker": ticker_u,
        "year": year,
        "embedded": len(chunks),
        "skipped": 0,
        "failed": 0,
        "reason": "",
    }


def retrieve_financial_statement_chunks_for_audit(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    year: int,
    *,
    top_k: int | None = None,
    embedding_model: str | None = None,
    dimensions: int | None = None,
) -> list[dict[str, Any]]:
    """Retrieve top-k financial-statement chunks for audit extraction."""
    qvec = _get_audit_query_embedding(
        model=embedding_model,
        dimensions=dimensions,
    )

    rows = con.execute(
        """
        SELECT
            chunk_index,
            chunk_text,
            token_count,
            list_cosine_distance(embedding::FLOAT[], ?::FLOAT[]) AS distance
        FROM financial_statement_document_embeddings
        WHERE ticker = ? AND year = ?
        ORDER BY distance ASC
        LIMIT ?
        """,
        [qvec, ticker.upper(), year, top_k or INFERENCE_TOP_K],
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


def _evaluate_audit_from_chunks(
    chunks: list[dict[str, Any]],
    *,
    inference_model: str,
) -> dict[str, Any]:
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

    prompt = _FINANCIAL_STATEMENT_AUDIT_PROMPT.format(content="\n".join(selected_chunks))

    client = _get_client()
    raw = run_chat_json_batch(
        client=client,
        model=inference_model,
        prompts={"audit": prompt},
        is_reasoning=inference_model.startswith(("o1", "o3", "o4", "gpt-5")),
        temperature=INFERENCE_TEMPERATURE,
    )["audit"]

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "found": False,
            "external_audit_firm": None,
            "external_audit_firm_en": None,
            "audit_opinion": None,
            "signing_auditor_names": [],
            "details": [],
            "reason": f"Failed to parse LLM response: {raw}",
        }

    if not isinstance(result.get("details"), list):
        result["details"] = []
    if not isinstance(result.get("signing_auditor_names"), list):
        result["signing_auditor_names"] = []

    found = bool(result.get("found", False))
    if not found:
        found = bool(result.get("external_audit_firm") or result.get("audit_opinion"))
        result["found"] = found

    return result


def get_financial_statement_audit_result(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    year: int,
    *,
    model: str | None = None,
) -> dict[str, Any] | None:
    """Read one stored financial-statement audit result."""
    row = con.execute(
        """
        SELECT
            ticker,
            year,
            found,
            audit_firm,
            audit_opinion,
            signing_auditor_names,
            value_json,
            details_json,
            reason,
            top_chunks,
            similarities,
            model,
            created_at
        FROM financial_statement_audit_results
        WHERE ticker = ? AND year = ? AND model = ?
        """,
        [ticker.upper(), year, model or INFERENCE_MODEL],
    ).fetchone()
    if not row:
        return None

    return {
        "ticker": row[0],
        "year": row[1],
        "found": bool(row[2]),
        "audit_firm": row[3],
        "audit_opinion": row[4],
        "signing_auditor_names": _safe_json_load(row[5], []),
        "value": _safe_json_load(row[6], None),
        "details": _safe_json_load(row[7], []),
        "reason": row[8],
        "top_chunks": _safe_json_load(row[9], []),
        "similarities": _safe_json_load(row[10], []),
        "model": row[11],
        "created_at": str(row[12]) if row[12] is not None else None,
    }


def extract_financial_statement_audit_info(
    ticker: str,
    year: int,
    con: duckdb.DuckDBPyConnection | None = None,
    *,
    replace: bool = False,
    top_k: int | None = None,
    inference_model: str | None = None,
    embedding_model: str | None = None,
    dimensions: int | None = None,
) -> dict[str, Any]:
    """Extract audit firm and opinion for one financial statement."""
    own_con = con is None
    if own_con:
        con = get_connection()

    assert con is not None
    ensure_vss_loaded(con)

    ticker_u = ticker.upper()
    model_name = inference_model or INFERENCE_MODEL

    report_row = con.execute(
        "SELECT source_file FROM financial_statement_reports WHERE ticker = ? AND year = ?",
        [ticker_u, year],
    ).fetchone()
    if not report_row:
        if own_con:
            con.close()
        raise ValueError(
            "No financial statement in financial_statement_reports for "
            f"{ticker_u}/{year}. Run sync_financial_statement_reports_from_markdown first."
        )

    if replace:
        con.execute(
            "DELETE FROM financial_statement_audit_results WHERE ticker = ? AND year = ? AND model = ?",
            [ticker_u, year, model_name],
        )
    else:
        existing = get_financial_statement_audit_result(
            con,
            ticker_u,
            year,
            model=model_name,
        )
        if existing is not None:
            if own_con:
                con.close()
            return existing

    emb_count = con.execute(
        "SELECT COUNT(*) FROM financial_statement_document_embeddings WHERE ticker = ? AND year = ?",
        [ticker_u, year],
    ).fetchone()[0]
    if int(emb_count) <= 0:
        if own_con:
            con.close()
        raise ValueError(
            "No embeddings in financial_statement_document_embeddings for "
            f"{ticker_u}/{year}. Run embed_financial_statement_report first."
        )

    chunks = retrieve_financial_statement_chunks_for_audit(
        con,
        ticker_u,
        year,
        top_k=top_k,
        embedding_model=embedding_model,
        dimensions=dimensions,
    )

    result = _evaluate_audit_from_chunks(
        chunks,
        inference_model=model_name,
    )

    value_json = json.dumps(
        {
            "external_audit_firm": result.get("external_audit_firm"),
            "external_audit_firm_en": result.get("external_audit_firm_en"),
            "audit_opinion": result.get("audit_opinion"),
            "signing_auditor_names": result.get("signing_auditor_names", []),
        },
        ensure_ascii=False,
    )
    details_json = json.dumps(result.get("details", []), ensure_ascii=False)
    top_chunks_json = json.dumps([c["chunk_index"] for c in chunks])
    similarities_json = json.dumps([round(float(c["distance"]), 6) for c in chunks])

    con.execute(
        """
        INSERT INTO financial_statement_audit_results
            (ticker, year, found, audit_firm, audit_opinion, signing_auditor_names,
             value_json, details_json, reason, top_chunks, similarities, model)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (ticker, year, model) DO UPDATE SET
            found = EXCLUDED.found,
            audit_firm = EXCLUDED.audit_firm,
            audit_opinion = EXCLUDED.audit_opinion,
            signing_auditor_names = EXCLUDED.signing_auditor_names,
            value_json = EXCLUDED.value_json,
            details_json = EXCLUDED.details_json,
            reason = EXCLUDED.reason,
            top_chunks = EXCLUDED.top_chunks,
            similarities = EXCLUDED.similarities,
            created_at = get_current_timestamp()
        """,
        [
            ticker_u,
            year,
            bool(result.get("found", False)),
            result.get("external_audit_firm") or result.get("external_audit_firm_en"),
            result.get("audit_opinion"),
            json.dumps(result.get("signing_auditor_names", []), ensure_ascii=False),
            value_json,
            details_json,
            str(result.get("reason", "")),
            top_chunks_json,
            similarities_json,
            model_name,
        ],
    )

    out = get_financial_statement_audit_result(
        con,
        ticker_u,
        year,
        model=model_name,
    )
    if own_con:
        con.close()
    return out or {
        "ticker": ticker_u,
        "year": year,
        "found": bool(result.get("found", False)),
        "audit_firm": result.get("external_audit_firm"),
        "audit_opinion": result.get("audit_opinion"),
        "signing_auditor_names": result.get("signing_auditor_names", []),
        "value": _safe_json_load(value_json, None),
        "details": _safe_json_load(details_json, []),
        "reason": str(result.get("reason", "")),
        "top_chunks": _safe_json_load(top_chunks_json, []),
        "similarities": _safe_json_load(similarities_json, []),
        "model": model_name,
        "created_at": None,
    }


def extract_financial_statement_audit_info_all_reports(
    con: duckdb.DuckDBPyConnection | None = None,
    *,
    tickers: list[str] | None = None,
    years: list[int] | None = None,
    replace: bool = False,
    top_k: int | None = None,
    inference_model: str | None = None,
    embedding_model: str | None = None,
    dimensions: int | None = None,
) -> list[dict[str, Any]]:
    """Run audit extraction for all embedded financial statements."""
    own_con = con is None
    if own_con:
        con = get_connection()

    assert con is not None
    ensure_vss_loaded(con)

    rows = con.execute(
        "SELECT DISTINCT ticker, year FROM financial_statement_document_embeddings ORDER BY ticker, year"
    ).fetchall()
    reports = [(str(r[0]).upper(), int(r[1])) for r in rows]

    if tickers:
        allowed = {t.upper() for t in tickers}
        reports = [(t, y) for t, y in reports if t in allowed]
    if years:
        allowed_years = set(years)
        reports = [(t, y) for t, y in reports if y in allowed_years]

    out: list[dict[str, Any]] = []
    for ticker, year in reports:
        try:
            out.append(
                extract_financial_statement_audit_info(
                    ticker,
                    year,
                    con=con,
                    replace=replace,
                    top_k=top_k,
                    inference_model=inference_model,
                    embedding_model=embedding_model,
                    dimensions=dimensions,
                )
            )
        except Exception as exc:
            out.append(
                {
                    "ticker": ticker,
                    "year": year,
                    "status": "failed",
                    "found": False,
                    "audit_firm": None,
                    "audit_opinion": None,
                    "signing_auditor_names": [],
                    "reason": str(exc),
                }
            )

    if own_con:
        con.close()
    return out
