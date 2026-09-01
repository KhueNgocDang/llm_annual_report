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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

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
from financial_statement_chunk_metadata import (
    LLM_METADATA_SYSTEM_PROMPT,
    build_chunk_metadata,
    build_chunk_metadata_json,
    merge_metadata,
    normalize_llm_metadata,
    parse_chunk_metadata,
)
from llm_batch_api import (
    get_batch_output_details,
    get_batch_output_map,
    get_batch_status,
    run_chat_json_batch,
    submit_chat_json_batch,
)
from llm_embeddings import chunk_markdown_content

logger = logging.getLogger(__name__)

try:
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None


MAX_PROMPT_CONTENT_TOKENS = 120000
FINANCIAL_STATEMENT_AUDIT_RESULTS_TABLE = "financial_statement_audit_results"
FINANCIAL_STATEMENT_AUDIT_JOBS_TABLE = "financial_statement_audit_jobs"
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
- Search the full text for audit-report headings and paragraphs such as "Báo cáo kiểm toán độc lập", "Ý kiến của kiểm toán viên", "Công ty kiểm toán", "Kiểm toán viên", or signature blocks.
- Also inspect signature blocks and the sentence immediately before/after them, because signing auditor names may appear there even if the opinion paragraph uses a different heading.
- Extract the audit firm name even when it appears on a separate line from the opinion paragraph, in a preamble sentence before the report, or in a signature block.
- Extract signing auditor names from the signature section, even if the firm is named separately or the report uses a different heading style.
- Treat short table rows or single-line entries naming a firm or person in an audit context as evidence when they appear near audit-report headings or opinion text.
- Treat phrases such as "đã được kiểm toán bởi...", "được kiểm toán bởi...", "thực hiện kiểm toán", or "kiểm toán bởi" as evidence of an external audit engagement.
- If the text names a firm in the audit context but does not explicitly include an opinion sentence, still set "found": true and preserve the firm name if the surrounding context clearly indicates an audit report.
- Map Vietnamese opinion phrases to the allowed values:
  - "chấp nhận toàn phần" or "unqualified" -> "unqualified"
  - "ngoại trừ" or "qualified" -> "qualified"
  - "từ chối" or "disclaimer" -> "disclaimer"
  - "không đưa ra ý kiến" or "adverse" -> "adverse"
- If the text only discusses audit procedures or standards without naming a firm, opinion, or signer, set "found": false and leave the other fields as null or [] rather than guessing.
- Do not infer missing information from unrelated sections, prior years, or generic company information.

TEXT DATA:
```
{content}
```
"""

_AUDIT_RETRIEVAL_TERMS: tuple[str, ...] = (
    "báo cáo kiểm toán",
    "bao cao kiem toan",
    "kiểm toán độc lập",
    "kiem toan doc lap",
    "ý kiến của kiểm toán viên",
    "y kien cua kiem toan vien",
    "công ty kiểm toán",
    "cong ty kiem toan",
    "đơn vị kiểm toán",
    "don vi kiem toan",
    "kiểm toán viên",
    "kiem toan vien",
    "ký báo cáo",
    "ky bao cao",
    "chữ ký",
    "chu ky",
    "người ký",
    "nguoi ky",
    "auditor",
    "audit report",
    "opinion",
)


def _build_financial_statement_audit_prompt(content: str) -> str:
    return _FINANCIAL_STATEMENT_AUDIT_PROMPT.format(content=content)


def _score_financial_statement_audit_chunk(
    chunk_text: str,
    metadata: dict[str, Any] | None = None,
) -> float:
    """Score a chunk higher when it looks like an independent audit-report section."""
    text = str(chunk_text or "").lower()
    metadata = metadata or {}
    section_tags = {str(tag).lower() for tag in metadata.get("section_tags", [])}
    role_tags = {str(tag).lower() for tag in metadata.get("role_tags", [])}

    score = 0.0
    for term in _AUDIT_RETRIEVAL_TERMS:
        if term in text:
            score += 0.7

    if "audit_report" in section_tags:
        score += 3.2
    if "gov_audit" in role_tags:
        score += 2.8

    if re.search(r"\b(chấp nhận toàn phần|ngoại trừ|từ chối|không đưa ra ý kiến|unqualified|qualified|adverse|disclaimer)\b", text):
        score += 1.1
    if re.search(r"\b(công ty kiểm toán|cong ty kiem toan|đơn vị kiểm toán|don vi kiem toan|kiểm toán viên|kiem toan vien)\b", text):
        score += 0.9
    if re.search(r"\b(ký báo cáo|ky bao cao|chữ ký|chu ky|người ký|nguoi ky|signature|signing)\b", text):
        score += 0.8
    if re.search(r"\b(nguyễn|nguyen|trần|tran|phạm|pham|lê|le)\b", text):
        score += 0.3

    return score


_audit_query_embedding: list[float] | None = None
_FS_EMBED_META_COLUMN_AVAILABLE: bool | None = None


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


def _has_fs_embedding_metadata_column(con: duckdb.DuckDBPyConnection) -> bool:
    global _FS_EMBED_META_COLUMN_AVAILABLE
    if _FS_EMBED_META_COLUMN_AVAILABLE is not None:
        return _FS_EMBED_META_COLUMN_AVAILABLE

    row = con.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'financial_statement_document_embeddings'
          AND column_name = 'chunk_metadata_json'
        LIMIT 1
        """
    ).fetchone()
    _FS_EMBED_META_COLUMN_AVAILABLE = row is not None
    return _FS_EMBED_META_COLUMN_AVAILABLE


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_async_run_id() -> str:
    return f"fs_meta_async_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"


def _ensure_fs_chunk_metadata_async_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS fs_chunk_metadata_async_runs (
            run_id VARCHAR PRIMARY KEY,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            model VARCHAR,
            temperature DOUBLE,
            prompt_batch_size INTEGER,
            replace_existing BOOLEAN,
            only_non_llm BOOLEAN,
            status VARCHAR,
            total_items INTEGER,
            submitted_items INTEGER,
            completed_items INTEGER,
            failed_items INTEGER
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS fs_chunk_metadata_async_items (
            run_id VARCHAR,
            ticker VARCHAR,
            year INTEGER,
            chunk_index INTEGER,
            custom_id VARCHAR,
            prompt_text VARCHAR,
            batch_id VARCHAR,
            status VARCHAR,
            attempt_count INTEGER,
            last_error VARCHAR,
            llm_raw_output VARCHAR,
            updated_at TIMESTAMP,
            PRIMARY KEY (run_id, ticker, year, chunk_index)
        )
        """
    )


def _refresh_fs_async_run_counters(con: duckdb.DuckDBPyConnection, run_id: str) -> dict[str, int]:
    total_items = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM fs_chunk_metadata_async_items
            WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()[0]
    )
    submitted_items = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM fs_chunk_metadata_async_items
            WHERE run_id = ? AND status = 'submitted'
            """,
            [run_id],
        ).fetchone()[0]
    )
    completed_items = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM fs_chunk_metadata_async_items
            WHERE run_id = ? AND status = 'completed'
            """,
            [run_id],
        ).fetchone()[0]
    )
    failed_items = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM fs_chunk_metadata_async_items
            WHERE run_id = ? AND status = 'failed'
            """,
            [run_id],
        ).fetchone()[0]
    )

    if submitted_items > 0:
        status = "running"
    elif completed_items == total_items and total_items > 0:
        status = "completed"
    elif completed_items > 0 and failed_items > 0:
        status = "partial_failed"
    elif failed_items > 0:
        status = "failed"
    elif total_items == 0:
        status = "empty"
    else:
        status = "pending"

    con.execute(
        """
        UPDATE fs_chunk_metadata_async_runs
        SET
            updated_at = ?,
            status = ?,
            total_items = ?,
            submitted_items = ?,
            completed_items = ?,
            failed_items = ?
        WHERE run_id = ?
        """,
        [
            _utc_now_iso(),
            status,
            total_items,
            submitted_items,
            completed_items,
            failed_items,
            run_id,
        ],
    )

    return {
        "total_items": total_items,
        "submitted_items": submitted_items,
        "completed_items": completed_items,
        "failed_items": failed_items,
    }


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


def _build_chunk_metadata_prompt(chunk_text: str) -> str:
    chunk = str(chunk_text or "")[:7000]
    return (
        f"{LLM_METADATA_SYSTEM_PROMPT}\n\n"
        "Chunk text:\n"
        "```\n"
        f"{chunk}\n"
        "```\n"
    )


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

    has_meta_col = _has_fs_embedding_metadata_column(con)

    if has_meta_col:
        con.executemany(
            """
            INSERT INTO financial_statement_document_embeddings
                (ticker, year, chunk_index, chunk_text, token_count, chunk_metadata_json, embedding, model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (ticker, year, chunk_index) DO UPDATE SET
                chunk_text = EXCLUDED.chunk_text,
                token_count = EXCLUDED.token_count,
                chunk_metadata_json = EXCLUDED.chunk_metadata_json,
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
                    build_chunk_metadata_json(chunks[i].chunk_text),
                    vectors[i],
                    model,
                )
                for i in range(len(chunks))
            ],
        )
    else:
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


def backfill_financial_statement_chunk_metadata(
    con: duckdb.DuckDBPyConnection,
    *,
    tickers: list[str] | None = None,
    years: list[int] | None = None,
) -> dict[str, int]:
    """Populate chunk_metadata_json from existing chunk_text rows.

    This avoids re-embedding vectors when only metadata enrichment is needed.
    """
    if not _has_fs_embedding_metadata_column(con):
        raise RuntimeError(
            "chunk_metadata_json column is missing in financial_statement_document_embeddings. "
            "Run init_db(con) with a writable connection first."
        )

    params: list[Any] = []
    where = ["1=1"]
    if tickers:
        upper_tickers = [t.upper() for t in tickers]
        where.append(f"ticker IN ({','.join(['?' for _ in upper_tickers])})")
        params.extend(upper_tickers)
    if years:
        where.append(f"year IN ({','.join(['?' for _ in years])})")
        params.extend(years)

    rows = con.execute(
        f"""
        SELECT ticker, year, chunk_index, chunk_text
        FROM financial_statement_document_embeddings
        WHERE {' AND '.join(where)}
          AND (chunk_metadata_json IS NULL OR trim(chunk_metadata_json) = '')
        ORDER BY ticker, year, chunk_index
        """,
        params,
    ).fetchall()

    if not rows:
        return {"updated_chunks": 0, "checked_chunks": 0}

    con.executemany(
        """
        UPDATE financial_statement_document_embeddings
        SET chunk_metadata_json = ?
        WHERE ticker = ? AND year = ? AND chunk_index = ?
        """,
        [
            (
                build_chunk_metadata_json(str(chunk_text or "")),
                str(ticker),
                int(year),
                int(chunk_index),
            )
            for ticker, year, chunk_index, chunk_text in rows
        ],
    )

    return {
        "updated_chunks": len(rows),
        "checked_chunks": len(rows),
    }


def regenerate_financial_statement_chunk_metadata_with_llm(
    con: duckdb.DuckDBPyConnection,
    *,
    tickers: list[str] | None = None,
    years: list[int] | None = None,
    model: str = INFERENCE_MODEL,
    temperature: float = INFERENCE_TEMPERATURE,
    prompt_batch_size: int = 120,
    replace_existing: bool = True,
    only_non_llm: bool = False,
    loop_per_firm_year: bool = False,
    checkpoint_per_firm_year: bool = False,
    max_chunks: int | None = None,
) -> dict[str, int]:
    """Regenerate chunk metadata with OpenAI prompt-engineered JSON outputs.

    This pipeline keeps deterministic metadata as a fallback and merges LLM tags
    for better retrieval signals.
    """
    if prompt_batch_size <= 0:
        raise ValueError("prompt_batch_size must be > 0")
    if not _has_fs_embedding_metadata_column(con):
        raise RuntimeError(
            "chunk_metadata_json column is missing in financial_statement_document_embeddings. "
            "Run init_db(con) with a writable connection first."
        )

    params: list[Any] = []
    where = ["1=1"]
    if tickers:
        upper_tickers = [t.upper() for t in tickers]
        where.append(f"ticker IN ({','.join(['?' for _ in upper_tickers])})")
        params.extend(upper_tickers)
    if years:
        where.append(f"year IN ({','.join(['?' for _ in years])})")
        params.extend(years)
    if not replace_existing:
        where.append("(chunk_metadata_json IS NULL OR trim(chunk_metadata_json) = '')")
    if only_non_llm:
        where.append(
            "COALESCE(json_extract_string(chunk_metadata_json, '$.metadata_source'), 'missing') <> 'hybrid'"
        )

    if loop_per_firm_year:
        firm_year_rows = con.execute(
            f"""
            SELECT DISTINCT ticker, year
            FROM financial_statement_document_embeddings
            WHERE {' AND '.join(where)}
            ORDER BY ticker, year
            """,
            params,
        ).fetchall()

        remaining_chunks = int(max_chunks) if max_chunks is not None and max_chunks > 0 else None
        summary = {
            "target_firm_years": len(firm_year_rows),
            "processed_firm_years": 0,
            "target_chunks": 0,
            "updated_chunks": 0,
            "llm_success": 0,
            "llm_failed": 0,
            "batches": 0,
        }

        for ticker, year in firm_year_rows:
            if remaining_chunks is not None and remaining_chunks <= 0:
                break

            child_max_chunks = remaining_chunks
            child_summary = regenerate_financial_statement_chunk_metadata_with_llm(
                con,
                tickers=[str(ticker)],
                years=[int(year)],
                model=model,
                temperature=temperature,
                prompt_batch_size=prompt_batch_size,
                replace_existing=replace_existing,
                only_non_llm=only_non_llm,
                loop_per_firm_year=False,
                checkpoint_per_firm_year=False,
                max_chunks=child_max_chunks,
            )

            summary["processed_firm_years"] += 1
            summary["target_chunks"] += int(child_summary["target_chunks"])
            summary["updated_chunks"] += int(child_summary["updated_chunks"])
            summary["llm_success"] += int(child_summary["llm_success"])
            summary["llm_failed"] += int(child_summary["llm_failed"])
            summary["batches"] += int(child_summary["batches"])

            if remaining_chunks is not None:
                remaining_chunks -= int(child_summary["updated_chunks"])

            if checkpoint_per_firm_year:
                con.execute("CHECKPOINT")

        return summary

    limit_sql = ""
    if max_chunks is not None and max_chunks > 0:
        limit_sql = f" LIMIT {int(max_chunks)}"

    rows = con.execute(
        f"""
        SELECT ticker, year, chunk_index, chunk_text
        FROM financial_statement_document_embeddings
        WHERE {' AND '.join(where)}
        ORDER BY ticker, year, chunk_index
        {limit_sql}
        """,
        params,
    ).fetchall()

    summary = {
        "target_chunks": len(rows),
        "updated_chunks": 0,
        "llm_success": 0,
        "llm_failed": 0,
        "batches": 0,
    }
    if not rows:
        return summary

    client = _get_client()
    is_reasoning = model.startswith(("o1", "o3", "o4", "gpt-5"))

    for start in range(0, len(rows), prompt_batch_size):
        batch_rows = rows[start : start + prompt_batch_size]
        prompts: dict[str, str] = {}
        row_map: dict[str, tuple[str, int, int, str]] = {}

        for ticker, year, chunk_index, chunk_text in batch_rows:
            custom_id = f"{ticker}|{int(year)}|{int(chunk_index)}"
            prompts[custom_id] = _build_chunk_metadata_prompt(str(chunk_text or ""))
            row_map[custom_id] = (str(ticker), int(year), int(chunk_index), str(chunk_text or ""))

        outputs: dict[str, str] = {}
        try:
            outputs = run_chat_json_batch(
                client=client,
                model=model,
                prompts=prompts,
                is_reasoning=is_reasoning,
                temperature=temperature,
                timeout_seconds=1800.0,
            )
        except Exception:
            # If a whole batch fails, fallback deterministic for all rows.
            outputs = {}

        updates: list[tuple[str, str, int, int]] = []
        for custom_id, (ticker, year, chunk_index, chunk_text) in row_map.items():
            base_meta = build_chunk_metadata(chunk_text)
            llm_raw = outputs.get(custom_id)

            if llm_raw is None:
                final_meta = {
                    **base_meta,
                    "schema_version": 1,
                    "metadata_source": "deterministic_fallback",
                }
                summary["llm_failed"] += 1
            else:
                llm_meta = normalize_llm_metadata(llm_raw)
                final_meta = merge_metadata(base_meta, llm_meta)
                summary["llm_success"] += 1

            updates.append(
                (
                    json.dumps(final_meta, ensure_ascii=False),
                    ticker,
                    year,
                    chunk_index,
                )
            )

        con.executemany(
            """
            UPDATE financial_statement_document_embeddings
            SET chunk_metadata_json = ?
            WHERE ticker = ? AND year = ? AND chunk_index = ?
            """,
            updates,
        )

        summary["updated_chunks"] += len(updates)
        summary["batches"] += 1

    return summary


def submit_financial_statement_chunk_metadata_regeneration_async(
    con: duckdb.DuckDBPyConnection,
    *,
    tickers: list[str] | None = None,
    years: list[int] | None = None,
    model: str = INFERENCE_MODEL,
    temperature: float = INFERENCE_TEMPERATURE,
    prompt_batch_size: int = 120,
    replace_existing: bool = True,
    only_non_llm: bool = True,
    loop_per_firm_year: bool = True,
    max_chunks: int | None = None,
) -> dict[str, Any]:
    """Submit asynchronous OpenAI batches for FS chunk metadata regeneration.

    This function only submits batch jobs and records progress state in DuckDB.
    Use ``collect_financial_statement_chunk_metadata_regeneration_async`` later
    to apply completed outputs.
    """
    if prompt_batch_size <= 0:
        raise ValueError("prompt_batch_size must be > 0")
    if not _has_fs_embedding_metadata_column(con):
        raise RuntimeError(
            "chunk_metadata_json column is missing in financial_statement_document_embeddings. "
            "Run init_db(con) with a writable connection first."
        )

    _ensure_fs_chunk_metadata_async_tables(con)

    params: list[Any] = []
    where = ["1=1"]
    if tickers:
        upper_tickers = [t.upper() for t in tickers]
        where.append(f"ticker IN ({','.join(['?' for _ in upper_tickers])})")
        params.extend(upper_tickers)
    if years:
        where.append(f"year IN ({','.join(['?' for _ in years])})")
        params.extend(years)
    if not replace_existing:
        where.append("(chunk_metadata_json IS NULL OR trim(chunk_metadata_json) = '')")
    if only_non_llm:
        where.append(
            "COALESCE(json_extract_string(chunk_metadata_json, '$.metadata_source'), 'missing') <> 'hybrid'"
        )

    limit_sql = ""
    if max_chunks is not None and max_chunks > 0:
        limit_sql = f" LIMIT {int(max_chunks)}"

    rows = con.execute(
        f"""
        SELECT ticker, year, chunk_index, chunk_text
        FROM financial_statement_document_embeddings
        WHERE {' AND '.join(where)}
        ORDER BY ticker, year, chunk_index
        {limit_sql}
        """,
        params,
    ).fetchall()

    run_id = _new_async_run_id()
    now = _utc_now_iso()
    con.execute(
        """
        INSERT INTO fs_chunk_metadata_async_runs (
            run_id, created_at, updated_at, model, temperature, prompt_batch_size,
            replace_existing, only_non_llm, status,
            total_items, submitted_items, completed_items, failed_items
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0)
        """,
        [
            run_id,
            now,
            now,
            model,
            float(temperature),
            int(prompt_batch_size),
            bool(replace_existing),
            bool(only_non_llm),
            "pending",
            len(rows),
        ],
    )

    if not rows:
        _refresh_fs_async_run_counters(con, run_id)
        return {
            "run_id": run_id,
            "total_items": 0,
            "submitted_items": 0,
            "submitted_batches": 0,
            "failed_to_submit_items": 0,
            "status": "empty",
        }

    con.executemany(
        """
        INSERT INTO fs_chunk_metadata_async_items (
            run_id, ticker, year, chunk_index, custom_id, prompt_text,
            batch_id, status, attempt_count, last_error, llm_raw_output, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, NULL, 'pending', 0, NULL, NULL, ?)
        """,
        [
            (
                run_id,
                str(ticker),
                int(year),
                int(chunk_index),
                f"{ticker}|{int(year)}|{int(chunk_index)}",
                _build_chunk_metadata_prompt(str(chunk_text or "")),
                now,
            )
            for ticker, year, chunk_index, chunk_text in rows
        ],
    )

    groups: dict[tuple[str, int], list[tuple[str, int, int, str, str]]] = defaultdict(list)
    for ticker, year, chunk_index, chunk_text in rows:
        custom_id = f"{ticker}|{int(year)}|{int(chunk_index)}"
        groups[(str(ticker), int(year))].append(
            (str(ticker), int(year), int(chunk_index), custom_id, _build_chunk_metadata_prompt(str(chunk_text or "")))
        )

    client = _get_client()
    is_reasoning = model.startswith(("o1", "o3", "o4", "gpt-5"))
    submitted_batches = 0
    failed_to_submit_items = 0

    if loop_per_firm_year:
        group_items = list(groups.items())
    else:
        flattened: list[tuple[str, int, int, str, str]] = []
        for items in groups.values():
            flattened.extend(items)
        group_items = [(("ALL", 0), flattened)]

    for (_ticker, _year), items in group_items:
        for start in range(0, len(items), prompt_batch_size):
            batch_items = items[start : start + prompt_batch_size]
            prompts = {custom_id: prompt for _, _, _, custom_id, prompt in batch_items}
            custom_ids = [custom_id for _, _, _, custom_id, _ in batch_items]

            try:
                batch_id = submit_chat_json_batch(
                    client=client,
                    model=model,
                    prompts=prompts,
                    is_reasoning=is_reasoning,
                    temperature=temperature,
                )
                con.executemany(
                    """
                    UPDATE fs_chunk_metadata_async_items
                    SET
                        status = 'submitted',
                        batch_id = ?,
                        attempt_count = attempt_count + 1,
                        last_error = NULL,
                        updated_at = ?
                    WHERE run_id = ? AND custom_id = ?
                    """,
                    [(batch_id, _utc_now_iso(), run_id, cid) for cid in custom_ids],
                )
                submitted_batches += 1
            except Exception as e:
                failed_to_submit_items += len(custom_ids)
                err = str(e)[:4000]
                con.executemany(
                    """
                    UPDATE fs_chunk_metadata_async_items
                    SET
                        status = 'failed',
                        last_error = ?,
                        updated_at = ?
                    WHERE run_id = ? AND custom_id = ?
                    """,
                    [(err, _utc_now_iso(), run_id, cid) for cid in custom_ids],
                )

    counters = _refresh_fs_async_run_counters(con, run_id)
    return {
        "run_id": run_id,
        "total_items": counters["total_items"],
        "submitted_items": counters["submitted_items"],
        "completed_items": counters["completed_items"],
        "failed_items": counters["failed_items"],
        "submitted_batches": submitted_batches,
        "failed_to_submit_items": failed_to_submit_items,
    }


def collect_financial_statement_chunk_metadata_regeneration_async(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    checkpoint_every_batches: int = 1,
) -> dict[str, Any]:
    """Collect completed async batch outputs, apply updates, and refresh progress."""
    if checkpoint_every_batches <= 0:
        checkpoint_every_batches = 1

    _ensure_fs_chunk_metadata_async_tables(con)

    run_exists = con.execute(
        """
        SELECT 1
        FROM fs_chunk_metadata_async_runs
        WHERE run_id = ?
        """,
        [run_id],
    ).fetchone()
    if run_exists is None:
        raise ValueError(f"Unknown async run_id: {run_id}")

    client = _get_client()
    pending_batches = [
        str(r[0])
        for r in con.execute(
            """
            SELECT DISTINCT batch_id
            FROM fs_chunk_metadata_async_items
            WHERE run_id = ? AND status = 'submitted' AND batch_id IS NOT NULL
            ORDER BY batch_id
            """,
            [run_id],
        ).fetchall()
    ]

    processed_batches = 0
    completed_batches = 0
    failed_batches = 0
    applied_chunks = 0
    batch_status_counts: dict[str, int] = defaultdict(int)

    for batch_id in pending_batches:
        status = get_batch_status(client, batch_id)
        batch_status_counts[status] += 1

        if status in {"validating", "in_progress", "finalizing", "cancelling"}:
            continue

        processed_batches += 1

        batch_items = con.execute(
            """
            SELECT ticker, year, chunk_index, custom_id
            FROM fs_chunk_metadata_async_items
            WHERE run_id = ? AND batch_id = ? AND status = 'submitted'
            """,
            [run_id, batch_id],
        ).fetchall()
        if not batch_items:
            continue

        if status != "completed":
            failed_batches += 1
            err = f"OpenAI batch status={status}"
            con.executemany(
                """
                UPDATE fs_chunk_metadata_async_items
                SET status = 'failed', last_error = ?, updated_at = ?
                WHERE run_id = ? AND batch_id = ? AND custom_id = ?
                """,
                [
                    (err, _utc_now_iso(), run_id, batch_id, str(custom_id))
                    for _, _, _, custom_id in batch_items
                ],
            )
            continue

        parsed = get_batch_output_details(client, batch_id)
        outputs: dict[str, str] = parsed["results"]
        raw_failures: dict[str, dict[str, Any]] = {
            str(f.get("custom_id") or ""): f for f in parsed.get("raw_failures", [])
        }

        completed_batches += 1

        # Load chunk text for deterministic fallback merge.
        chunk_text_map = {
            (str(t), int(y), int(i)): str(txt or "")
            for t, y, i, txt in con.execute(
                """
                SELECT e.ticker, e.year, e.chunk_index, e.chunk_text
                FROM financial_statement_document_embeddings e
                JOIN fs_chunk_metadata_async_items i
                  ON i.run_id = ?
                 AND i.batch_id = ?
                 AND i.status = 'submitted'
                 AND i.ticker = e.ticker
                 AND i.year = e.year
                 AND i.chunk_index = e.chunk_index
                """,
                [run_id, batch_id],
            ).fetchall()
        }

        embedding_updates: list[tuple[str, str, int, int]] = []
        item_updates: list[tuple[str, str | None, str | None, str, str, str, int, int]] = []

        for ticker, year, chunk_index, custom_id in batch_items:
            key = (str(ticker), int(year), int(chunk_index))
            chunk_text = chunk_text_map.get(key, "")
            llm_raw = outputs.get(str(custom_id))

            if llm_raw is None:
                failure = raw_failures.get(str(custom_id), {})
                status_code = int(failure.get("status_code") or 0)
                body = failure.get("body")
                err = f"missing output"
                if status_code > 0:
                    err = f"status={status_code} body={json.dumps(body, ensure_ascii=False)[:500]}"
                item_updates.append((
                    "failed",
                    err[:4000],
                    None,
                    _utc_now_iso(),
                    run_id,
                    str(ticker),
                    int(year),
                    int(chunk_index),
                ))
                continue

            try:
                base_meta = build_chunk_metadata(chunk_text)
                llm_meta = normalize_llm_metadata(llm_raw)
                final_meta = merge_metadata(base_meta, llm_meta)
                embedding_updates.append(
                    (
                        json.dumps(final_meta, ensure_ascii=False),
                        str(ticker),
                        int(year),
                        int(chunk_index),
                    )
                )
                item_updates.append((
                    "completed",
                    None,
                    str(llm_raw),
                    _utc_now_iso(),
                    run_id,
                    str(ticker),
                    int(year),
                    int(chunk_index),
                ))
            except Exception as e:
                item_updates.append((
                    "failed",
                    str(e)[:4000],
                    str(llm_raw),
                    _utc_now_iso(),
                    run_id,
                    str(ticker),
                    int(year),
                    int(chunk_index),
                ))

        if embedding_updates:
            con.executemany(
                """
                UPDATE financial_statement_document_embeddings
                SET chunk_metadata_json = ?
                WHERE ticker = ? AND year = ? AND chunk_index = ?
                """,
                embedding_updates,
            )
            applied_chunks += len(embedding_updates)

        if item_updates:
            con.executemany(
                """
                UPDATE fs_chunk_metadata_async_items
                SET status = ?, last_error = ?, llm_raw_output = ?, updated_at = ?
                WHERE run_id = ? AND ticker = ? AND year = ? AND chunk_index = ?
                """,
                item_updates,
            )

        if processed_batches % checkpoint_every_batches == 0:
            con.execute("CHECKPOINT")

    counters = _refresh_fs_async_run_counters(con, run_id)
    return {
        "run_id": run_id,
        "processed_batches": processed_batches,
        "completed_batches": completed_batches,
        "failed_batches": failed_batches,
        "applied_chunks": applied_chunks,
        "batch_status_counts": dict(batch_status_counts),
        **counters,
    }


def retry_financial_statement_chunk_metadata_regeneration_async(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    prompt_batch_size: int | None = None,
    max_items: int | None = None,
    max_attempts: int = 5,
    loop_per_firm_year: bool = True,
) -> dict[str, Any]:
    """Retry failed async metadata items by re-submitting fresh OpenAI batches."""
    _ensure_fs_chunk_metadata_async_tables(con)

    run_row = con.execute(
        """
        SELECT model, temperature, prompt_batch_size
        FROM fs_chunk_metadata_async_runs
        WHERE run_id = ?
        """,
        [run_id],
    ).fetchone()
    if run_row is None:
        raise ValueError(f"Unknown async run_id: {run_id}")

    model = str(run_row[0])
    temperature = float(run_row[1])
    eff_prompt_batch_size = int(prompt_batch_size or int(run_row[2] or 120))
    if eff_prompt_batch_size <= 0:
        raise ValueError("prompt_batch_size must be > 0")

    limit_sql = ""
    limit_params: list[Any] = [run_id, max_attempts]
    if max_items is not None and max_items > 0:
        limit_sql = f" LIMIT {int(max_items)}"

    failed_rows = con.execute(
        f"""
        SELECT ticker, year, chunk_index, custom_id, prompt_text
        FROM fs_chunk_metadata_async_items
        WHERE run_id = ?
          AND status = 'failed'
          AND attempt_count < ?
        ORDER BY ticker, year, chunk_index
        {limit_sql}
        """,
        limit_params,
    ).fetchall()

    if not failed_rows:
        counters = _refresh_fs_async_run_counters(con, run_id)
        return {
            "run_id": run_id,
            "retried_items": 0,
            "retried_batches": 0,
            **counters,
        }

    groups: dict[tuple[str, int], list[tuple[str, int, int, str, str]]] = defaultdict(list)
    for ticker, year, chunk_index, custom_id, prompt_text in failed_rows:
        groups[(str(ticker), int(year))].append(
            (str(ticker), int(year), int(chunk_index), str(custom_id), str(prompt_text or ""))
        )

    if loop_per_firm_year:
        group_items = list(groups.items())
    else:
        flattened: list[tuple[str, int, int, str, str]] = []
        for items in groups.values():
            flattened.extend(items)
        group_items = [(("ALL", 0), flattened)]

    client = _get_client()
    is_reasoning = model.startswith(("o1", "o3", "o4", "gpt-5"))
    retried_batches = 0
    retried_items = 0

    for (_ticker, _year), items in group_items:
        for start in range(0, len(items), eff_prompt_batch_size):
            batch_items = items[start : start + eff_prompt_batch_size]
            prompts = {custom_id: prompt for _, _, _, custom_id, prompt in batch_items}
            custom_ids = [custom_id for _, _, _, custom_id, _ in batch_items]

            try:
                batch_id = submit_chat_json_batch(
                    client=client,
                    model=model,
                    prompts=prompts,
                    is_reasoning=is_reasoning,
                    temperature=temperature,
                )
                con.executemany(
                    """
                    UPDATE fs_chunk_metadata_async_items
                    SET
                        status = 'submitted',
                        batch_id = ?,
                        attempt_count = attempt_count + 1,
                        last_error = NULL,
                        updated_at = ?
                    WHERE run_id = ? AND custom_id = ?
                    """,
                    [(batch_id, _utc_now_iso(), run_id, cid) for cid in custom_ids],
                )
                retried_batches += 1
                retried_items += len(custom_ids)
            except Exception as e:
                err = str(e)[:4000]
                con.executemany(
                    """
                    UPDATE fs_chunk_metadata_async_items
                    SET
                        status = 'failed',
                        last_error = ?,
                        updated_at = ?
                    WHERE run_id = ? AND custom_id = ?
                    """,
                    [(err, _utc_now_iso(), run_id, cid) for cid in custom_ids],
                )

    counters = _refresh_fs_async_run_counters(con, run_id)
    return {
        "run_id": run_id,
        "retried_items": retried_items,
        "retried_batches": retried_batches,
        **counters,
    }


def get_financial_statement_chunk_metadata_regeneration_progress(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
) -> dict[str, Any]:
    """Return persisted progress and completion ratio for an async run."""
    _ensure_fs_chunk_metadata_async_tables(con)
    counters = _refresh_fs_async_run_counters(con, run_id)

    run_row = con.execute(
        """
        SELECT created_at, updated_at, model, temperature, prompt_batch_size, status
        FROM fs_chunk_metadata_async_runs
        WHERE run_id = ?
        """,
        [run_id],
    ).fetchone()
    if run_row is None:
        raise ValueError(f"Unknown async run_id: {run_id}")

    status_counts = {
        str(k): int(v)
        for k, v in con.execute(
            """
            SELECT status, COUNT(*)
            FROM fs_chunk_metadata_async_items
            WHERE run_id = ?
            GROUP BY status
            ORDER BY COUNT(*) DESC
            """,
            [run_id],
        ).fetchall()
    }

    firm_year_done = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT ticker, year,
                       SUM(CASE WHEN status IN ('completed', 'failed') THEN 1 ELSE 0 END) AS done_cnt,
                       COUNT(*) AS total_cnt
                FROM fs_chunk_metadata_async_items
                WHERE run_id = ?
                GROUP BY ticker, year
            ) t
            WHERE done_cnt = total_cnt
            """,
            [run_id],
        ).fetchone()[0]
    )
    firm_year_total = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT ticker, year
                FROM fs_chunk_metadata_async_items
                WHERE run_id = ?
                GROUP BY ticker, year
            ) t
            """,
            [run_id],
        ).fetchone()[0]
    )

    total = counters["total_items"]
    done = counters["completed_items"] + counters["failed_items"]
    progress_pct = (100.0 * done / total) if total > 0 else 100.0

    return {
        "run_id": run_id,
        "created_at": str(run_row[0]),
        "updated_at": str(run_row[1]),
        "model": str(run_row[2]),
        "temperature": float(run_row[3]),
        "prompt_batch_size": int(run_row[4]),
        "status": str(run_row[5]),
        "progress_pct": progress_pct,
        "firm_year_done": firm_year_done,
        "firm_year_total": firm_year_total,
        "status_counts": status_counts,
        **counters,
    }


def list_financial_statement_chunk_metadata_regeneration_async_runs(
    con: duckdb.DuckDBPyConnection,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """List recent async metadata regeneration runs for later collection/review."""
    _ensure_fs_chunk_metadata_async_tables(con)
    if limit <= 0:
        limit = 20

    rows = con.execute(
        """
        SELECT
            run_id,
            created_at,
            updated_at,
            status,
            total_items,
            submitted_items,
            completed_items,
            failed_items,
            model,
            temperature,
            prompt_batch_size
        FROM fs_chunk_metadata_async_runs
        ORDER BY created_at DESC
        LIMIT ?
        """,
        [int(limit)],
    ).fetchall()

    runs: list[dict[str, Any]] = []
    for r in rows:
        total_items = int(r[4] or 0)
        completed_items = int(r[6] or 0)
        failed_items = int(r[7] or 0)
        done = completed_items + failed_items
        progress_pct = (100.0 * done / total_items) if total_items > 0 else 100.0
        runs.append(
            {
                "run_id": str(r[0]),
                "created_at": str(r[1]),
                "updated_at": str(r[2]),
                "status": str(r[3]),
                "total_items": total_items,
                "submitted_items": int(r[5] or 0),
                "completed_items": completed_items,
                "failed_items": failed_items,
                "model": str(r[8]),
                "temperature": float(r[9]),
                "prompt_batch_size": int(r[10] or 0),
                "progress_pct": progress_pct,
            }
        )
    return runs


def review_financial_statement_chunk_metadata_regeneration_async(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    status: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Review run outputs/errors with per-item detail for auditing and retries."""
    _ensure_fs_chunk_metadata_async_tables(con)
    if limit <= 0:
        limit = 20

    where = ["run_id = ?"]
    params: list[Any] = [run_id]
    if status:
        where.append("status = ?")
        params.append(status)

    rows = con.execute(
        f"""
        SELECT ticker, year, chunk_index, status, attempt_count, batch_id, last_error, llm_raw_output
        FROM fs_chunk_metadata_async_items
        WHERE {' AND '.join(where)}
        ORDER BY ticker, year, chunk_index
        LIMIT ?
        """,
        [*params, int(limit)],
    ).fetchall()

    sample_items: list[dict[str, Any]] = []
    for ticker, year, chunk_index, st, attempt_count, batch_id, last_error, llm_raw_output in rows:
        preview = None
        if llm_raw_output:
            parsed = _safe_json_load(llm_raw_output, {})
            if isinstance(parsed, dict):
                preview = {
                    "summary": parsed.get("summary"),
                    "confidence": parsed.get("confidence"),
                    "section_tags": parsed.get("section_tags"),
                    "role_tags": parsed.get("role_tags"),
                }
        sample_items.append(
            {
                "ticker": str(ticker),
                "year": int(year),
                "chunk_index": int(chunk_index),
                "status": str(st),
                "attempt_count": int(attempt_count or 0),
                "batch_id": str(batch_id) if batch_id is not None else None,
                "last_error": str(last_error) if last_error is not None else None,
                "output_preview": preview,
            }
        )

    progress = get_financial_statement_chunk_metadata_regeneration_progress(con, run_id=run_id)
    progress["sample_items"] = sample_items
    return progress


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

    top_k_final = top_k or INFERENCE_TOP_K
    candidate_k = max(top_k_final, top_k_final * 6)
    has_meta_col = _has_fs_embedding_metadata_column(con)
    metadata_select = "chunk_metadata_json" if has_meta_col else "NULL AS chunk_metadata_json"

    rows = con.execute(
        f"""
        SELECT
            chunk_index,
            chunk_text,
            token_count,
            {metadata_select},
            list_cosine_distance(embedding::FLOAT[], ?::FLOAT[]) AS distance
        FROM financial_statement_document_embeddings
        WHERE ticker = ? AND year = ?
        ORDER BY distance ASC
        LIMIT ?
        """,
        [qvec, ticker.upper(), year, candidate_k],
    ).fetchall()

    scored: list[dict[str, Any]] = []
    for r in rows:
        metadata = parse_chunk_metadata(r[3])
        section_tags = set(str(s) for s in metadata.get("section_tags", []))
        role_tags = set(str(s) for s in metadata.get("role_tags", []))
        signals = metadata.get("signals", {}) if isinstance(metadata, dict) else {}

        metadata_prior = 0.0
        if "audit_report" in section_tags:
            metadata_prior += 0.35
        if "gov_audit" in role_tags:
            metadata_prior += 0.35
        if bool(signals.get("has_table_like")):
            metadata_prior += 0.05

        lexical = _score_financial_statement_audit_chunk(r[1], metadata)
        semantic = 1.0 / (1.0 + max(float(r[4] or 0.0), 0.0))
        score = semantic + metadata_prior + lexical

        scored.append(
            {
                "chunk_index": r[0],
                "chunk_text": r[1],
                "token_count": r[2],
                "distance": r[4],
                "_meta_score": score,
            }
        )

    scored.sort(key=lambda c: float(c.get("_meta_score", 0.0)), reverse=True)

    return [
        {
            "chunk_index": c["chunk_index"],
            "chunk_text": c["chunk_text"],
            "token_count": c["token_count"],
            "distance": c["distance"],
        }
        for c in scored[:top_k_final]
    ]


def _normalize_audit_result(result: dict[str, Any], source_text: str) -> dict[str, Any]:
    """Recover audit firm/opinion values when the model leaves top-level fields empty."""
    normalized = dict(result or {})
    normalized.setdefault("found", False)
    normalized.setdefault("external_audit_firm", None)
    normalized.setdefault("external_audit_firm_en", None)
    normalized.setdefault("audit_opinion", None)
    normalized.setdefault("signing_auditor_names", [])
    normalized.setdefault("details", [])
    normalized.setdefault("reason", "")

    if not isinstance(normalized.get("details"), list):
        normalized["details"] = []
    if not isinstance(normalized.get("signing_auditor_names"), list):
        normalized["signing_auditor_names"] = []

    source_text_lc = str(source_text or "").lower()
    details = normalized.get("details") or []

    firm_candidates: list[str] = []
    for detail in details:
        if not isinstance(detail, dict):
            continue
        name = str(detail.get("name") or detail.get("firm") or detail.get("auditor") or "").strip()
        if name:
            firm_candidates.append(name)
        evidence = str(detail.get("evidence") or detail.get("notes") or "").strip()
        if evidence:
            firm_candidates.append(evidence)

    if not normalized.get("external_audit_firm") and not normalized.get("external_audit_firm_en"):
        for candidate in firm_candidates:
            if re.search(r"\b(?:deloitte|ernst|kpmg|pwc|ey|grant thornton|rsm)\b", candidate, re.I):
                cleaned = str(candidate).strip()
                for _ in range(3):
                    cleaned = re.sub(
                        r"^(?:công ty|cong ty|đơn vị|don vi|kiểm toán|kiem toan|audit|auditor|chi nhánh|chi nhanh|tnhh|limited|co\.?|ltd\.?|company)\s+",
                        "",
                        cleaned,
                        flags=re.I,
                    )
                    cleaned = re.sub(
                        r"\s+(?:công ty|cong ty|đơn vị|don vi|kiểm toán|kiem toan|audit|auditor|chi nhánh|chi nhanh|tnhh|limited|co\.?|ltd\.?|company)$",
                        "",
                        cleaned,
                        flags=re.I,
                    )
                normalized["external_audit_firm"] = cleaned.strip() or candidate
                break
        if not normalized.get("external_audit_firm"):
            for candidate in firm_candidates:
                if re.search(r"(công ty|cong ty|đơn vị|don vi|kiểm toán|kiem toan|auditor|firm)", candidate, re.I):
                    normalized["external_audit_firm"] = candidate
                    break

    if not normalized.get("audit_opinion"):
        opinion_from_details = None
        for detail in details:
            if not isinstance(detail, dict):
                continue
            opinion = detail.get("opinion") or detail.get("audit_opinion") or ""
            if opinion:
                opinion_from_details = str(opinion).strip().lower()
                break
        if opinion_from_details:
            normalized["audit_opinion"] = {
                "chấp nhận toàn phần": "unqualified",
                "unqualified": "unqualified",
                "ngoại trừ": "qualified",
                "qualified": "qualified",
                "từ chối": "disclaimer",
                "disclaimer": "disclaimer",
                "không đưa ra ý kiến": "adverse",
                "adverse": "adverse",
            }.get(opinion_from_details, opinion_from_details)
        else:
            opinion_match = re.search(
                r"\b(chấp nhận toàn phần|ngoại trừ|từ chối|không đưa ra ý kiến|unqualified|qualified|adverse|disclaimer)\b",
                source_text_lc,
            )
            if opinion_match:
                normalized["audit_opinion"] = {
                    "chấp nhận toàn phần": "unqualified",
                    "unqualified": "unqualified",
                    "ngoại trừ": "qualified",
                    "qualified": "qualified",
                    "từ chối": "disclaimer",
                    "disclaimer": "disclaimer",
                    "không đưa ra ý kiến": "adverse",
                    "adverse": "adverse",
                }[opinion_match.group(1).lower()]

    if not normalized.get("signing_auditor_names"):
        signer_candidates: list[str] = []
        for detail in details:
            if not isinstance(detail, dict):
                continue
            name = str(detail.get("signing_auditor") or detail.get("name") or detail.get("auditor") or "").strip()
            if name:
                signer_candidates.append(name)
        if signer_candidates:
            normalized["signing_auditor_names"] = signer_candidates

    found = bool(normalized.get("found", False))
    if not found:
        found = bool(
            normalized.get("external_audit_firm")
            or normalized.get("external_audit_firm_en")
            or normalized.get("audit_opinion")
            or normalized.get("signing_auditor_names")
            or re.search(r"(báo cáo kiểm toán|bao cao kiem toan|kiểm toán độc lập|kiem toan doc lap|kiểm toán viên|kiem toan vien|công ty kiểm toán|cong ty kiem toan)", source_text_lc, re.I)
        )
    normalized["found"] = found
    return normalized


def _evaluate_audit_from_chunks(
    chunks: list[dict[str, Any]],
    *,
    inference_model: str,
) -> dict[str, Any]:
    source_text, prompt = _build_audit_prompt_from_chunks(chunks)

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

    result = _normalize_audit_result(result, source_text)
    return result


def _build_audit_prompt_from_chunks(chunks: list[dict[str, Any]]) -> tuple[str, str]:
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

    source_text = "\n".join(selected_chunks)
    prompt = _build_financial_statement_audit_prompt(source_text)
    return source_text, prompt


def create_financial_statement_audit_jobs(
    con: duckdb.DuckDBPyConnection,
    *,
    tickers: list[str] | None = None,
    years: list[int] | None = None,
    replace: bool = False,
    inference_model: str | None = None,
) -> int:
    """Create pending financial-statement audit jobs for embedded reports."""
    inference_model = inference_model or INFERENCE_MODEL

    rows = con.execute(
        "SELECT DISTINCT ticker, year FROM financial_statement_document_embeddings"
    ).fetchall()
    embedded = [(str(r[0]).upper(), int(r[1])) for r in rows]

    if tickers:
        upper = {t.upper() for t in tickers}
        embedded = [(t, y) for t, y in embedded if t in upper]
    if years:
        embedded = [(t, y) for t, y in embedded if y in years]

    created = 0
    for ticker, year in embedded:
        existing = con.execute(
            "SELECT status FROM financial_statement_audit_jobs "
            "WHERE ticker = ? AND year = ? AND model = ?",
            [ticker, year, inference_model],
        ).fetchone()

        if existing is None:
            con.execute(
                """
                INSERT INTO financial_statement_audit_jobs
                    (id, ticker, year, model, status)
                VALUES (nextval('financial_statement_audit_jobs_id_seq'), ?, ?, ?, 'pending')
                """,
                [ticker, year, inference_model],
            )
            created += 1
        elif replace and str(existing[0]) in {"completed", "failed"}:
            con.execute(
                """
                UPDATE financial_statement_audit_jobs
                SET status = 'pending', error_message = NULL,
                    started_at = NULL, completed_at = NULL,
                    batch_id = NULL,
                    batch_submitted_at = NULL,
                    batch_checked_at = NULL,
                    top_k = NULL
                WHERE ticker = ? AND year = ? AND model = ?
                """,
                [ticker, year, inference_model],
            )
            created += 1

    return created


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
    """Extract audit firm and opinion for one financial statement.

    Uses async OpenAI Batch API lifecycle:
    - First run submits batch and stores batch_id in financial_statement_audit_jobs.
    - Later runs reconcile completed batch outputs and persist final results.
    """
    own_con = con is None
    if own_con:
        con = get_connection()

    assert con is not None
    ensure_vss_loaded(con)

    ticker_u = ticker.upper()
    model_name = inference_model or INFERENCE_MODEL
    effective_top_k = top_k or INFERENCE_TOP_K

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

    existing_job = con.execute(
        "SELECT status, batch_id FROM financial_statement_audit_jobs "
        "WHERE ticker = ? AND year = ? AND model = ?",
        [ticker_u, year, model_name],
    ).fetchone()
    if existing_job is None:
        con.execute(
            """
            INSERT INTO financial_statement_audit_jobs
                (id, ticker, year, model, status)
            VALUES (nextval('financial_statement_audit_jobs_id_seq'), ?, ?, ?, 'pending')
            """,
            [ticker_u, year, model_name],
        )
        existing_job = ("pending", None)

    if replace:
        con.execute(
            "DELETE FROM financial_statement_audit_results WHERE ticker = ? AND year = ? AND model = ?",
            [ticker_u, year, model_name],
        )
        con.execute(
            """
            UPDATE financial_statement_audit_jobs
            SET status = 'pending',
                error_message = NULL,
                started_at = NULL,
                completed_at = NULL,
                batch_id = NULL,
                batch_submitted_at = NULL,
                batch_checked_at = NULL,
                top_k = NULL
            WHERE ticker = ? AND year = ? AND model = ?
            """,
            [ticker_u, year, model_name],
        )
        existing_job = ("pending", None)
    else:
        existing = get_financial_statement_audit_result(
            con,
            ticker_u,
            year,
            model=model_name,
        )
        if existing is not None and not existing_job[1]:
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

    con.execute(
        """
        UPDATE financial_statement_audit_jobs
        SET status = 'running', started_at = get_current_timestamp(),
            error_message = NULL, top_k = ?
        WHERE ticker = ? AND year = ? AND model = ?
        """,
        [effective_top_k, ticker_u, year, model_name],
    )

    client = _get_client()

    # Reconcile running batch first.
    batch_id = str(existing_job[1]) if existing_job and existing_job[1] else None
    if batch_id:
        batch_status = get_batch_status(client, batch_id)
        con.execute(
            """
            UPDATE financial_statement_audit_jobs
            SET batch_checked_at = get_current_timestamp(),
                status = CASE
                    WHEN ? = 'completed' THEN status
                    WHEN ? IN ('failed', 'expired', 'cancelled') THEN 'failed'
                    ELSE 'running'
                END
            WHERE ticker = ? AND year = ? AND model = ?
            """,
            [batch_status, batch_status, ticker_u, year, model_name],
        )

        if batch_status in {"validating", "in_progress", "finalizing", "cancelling"}:
            if own_con:
                con.close()
            return {
                "ticker": ticker_u,
                "year": year,
                "status": "running",
                "batch_id": batch_id,
                "found": False,
                "audit_firm": None,
                "audit_opinion": None,
                "signing_auditor_names": [],
                "reason": f"Batch still running on OpenAI (status={batch_status})",
                "model": model_name,
            }

        if batch_status in {"failed", "expired", "cancelled"}:
            msg = f"Batch {batch_id} ended with status={batch_status}"
            con.execute(
                """
                UPDATE financial_statement_audit_jobs
                SET status = 'failed',
                    completed_at = get_current_timestamp(),
                    error_message = ?,
                    batch_id = NULL
                WHERE ticker = ? AND year = ? AND model = ?
                """,
                [msg, ticker_u, year, model_name],
            )
            if own_con:
                con.close()
            raise RuntimeError(msg)

        output_map = get_batch_output_map(client, batch_id)
        raw = output_map.get("audit")
        if raw is None:
            raw = output_map.get(f"audit|{ticker_u}|{year}")
        if raw is None:
            raise RuntimeError(f"Batch {batch_id} missing audit output for {ticker_u}/{year}")

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

        chunks = retrieve_financial_statement_chunks_for_audit(
            con,
            ticker_u,
            year,
            top_k=effective_top_k,
            embedding_model=embedding_model,
            dimensions=dimensions,
        )
        source_text, _ = _build_audit_prompt_from_chunks(chunks)
        result = _normalize_audit_result(result, source_text)

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

        con.execute(
            """
            UPDATE financial_statement_audit_jobs
            SET status = 'completed',
                completed_at = get_current_timestamp(),
                error_message = NULL,
                batch_id = NULL
            WHERE ticker = ? AND year = ? AND model = ?
            """,
            [ticker_u, year, model_name],
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

    # Submit new async batch when no in-flight batch exists.
    chunks = retrieve_financial_statement_chunks_for_audit(
        con,
        ticker_u,
        year,
        top_k=effective_top_k,
        embedding_model=embedding_model,
        dimensions=dimensions,
    )
    _, prompt = _build_audit_prompt_from_chunks(chunks)
    batch_id = submit_chat_json_batch(
        client=client,
        model=model_name,
        prompts={"audit": prompt},
        is_reasoning=model_name.startswith(("o1", "o3", "o4", "gpt-5")),
        temperature=INFERENCE_TEMPERATURE,
    )

    con.execute(
        """
        UPDATE financial_statement_audit_jobs
        SET status = 'running',
            batch_id = ?,
            batch_submitted_at = get_current_timestamp(),
            batch_checked_at = get_current_timestamp(),
            top_k = ?
        WHERE ticker = ? AND year = ? AND model = ?
        """,
        [batch_id, effective_top_k, ticker_u, year, model_name],
    )

    if own_con:
        con.close()
    return {
        "ticker": ticker_u,
        "year": year,
        "status": "submitted",
        "batch_id": batch_id,
        "found": False,
        "audit_firm": None,
        "audit_opinion": None,
        "signing_auditor_names": [],
        "reason": "Submitted batch to OpenAI; rerun later to sync completed output.",
        "model": model_name,
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
