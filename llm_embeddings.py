"""Embedding ingestion pipeline for annual_reports -> document_embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import duckdb

from config import (
    EMBEDDING_CHUNK_OVERLAP,
    EMBEDDING_CHUNK_SIZE,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
)
from database import get_connection
from embedder import get_embeddings

try:
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None


@dataclass
class Chunk:
    chunk_index: int
    chunk_text: str
    token_count: int


def _encode_tokens(text: str) -> list[int]:
    if tiktoken is not None:
        enc = tiktoken.get_encoding("cl100k_base")
        return enc.encode(text)
    return [hash(tok) % 1000003 for tok in text.split()]


def _decode_tokens(tokens: list[int]) -> str:
    if tiktoken is not None:
        enc = tiktoken.get_encoding("cl100k_base")
        return enc.decode(tokens)
    return " ".join(str(tok) for tok in tokens)


def chunk_markdown_content(
    content: str,
    *,
    chunk_size: int = EMBEDDING_CHUNK_SIZE,
    chunk_overlap: int = EMBEDDING_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split markdown content into overlapping token chunks."""
    tokens = _encode_tokens(content)
    if not tokens:
        return []

    step = max(chunk_size - chunk_overlap, 1)
    chunks: list[Chunk] = []

    index = 0
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        if not window:
            continue

        text = _decode_tokens(window).strip()
        if not text:
            continue

        chunks.append(
            Chunk(
                chunk_index=index,
                chunk_text=text,
                token_count=len(window),
            )
        )
        index += 1

        if start + chunk_size >= len(tokens):
            break

    return chunks


def embed_report(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    year: int,
    *,
    replace: bool = False,
    model: str = EMBEDDING_MODEL,
    dimensions: int = EMBEDDING_DIMENSIONS,
    chunk_size: int = EMBEDDING_CHUNK_SIZE,
    chunk_overlap: int = EMBEDDING_CHUNK_OVERLAP,
) -> dict[str, int | bool | str]:
    """Embed one annual report and store vectors in document_embeddings."""
    t = ticker.upper()

    row = con.execute(
        "SELECT content FROM annual_reports WHERE ticker = ? AND year = ?",
        [t, year],
    ).fetchone()
    if not row:
        return {
            "ticker": t,
            "year": year,
            "embedded": 0,
            "chunks": 0,
            "skipped": 1,
            "failed": 0,
            "reason": "annual_report_not_found",
        }

    existing = con.execute(
        "SELECT COUNT(*) FROM document_embeddings WHERE ticker = ? AND year = ?",
        [t, year],
    ).fetchone()[0]

    if existing > 0 and not replace:
        return {
            "ticker": t,
            "year": year,
            "embedded": 0,
            "chunks": int(existing),
            "skipped": 1,
            "failed": 0,
            "reason": "already_embedded",
        }

    if replace:
        con.execute(
            "DELETE FROM document_embeddings WHERE ticker = ? AND year = ?",
            [t, year],
        )

    content = row[0] or ""
    chunks = chunk_markdown_content(
        content,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        return {
            "ticker": t,
            "year": year,
            "embedded": 0,
            "chunks": 0,
            "skipped": 1,
            "failed": 0,
            "reason": "empty_content",
        }

    vectors = get_embeddings(
        [chunk.chunk_text for chunk in chunks],
        model=model,
        dimensions=dimensions,
    )

    con.executemany(
        """
        INSERT INTO document_embeddings (
            ticker,
            year,
            chunk_index,
            chunk_text,
            token_count,
            embedding,
            model
        )
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
                t,
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
        "ticker": t,
        "year": year,
        "embedded": len(chunks),
        "chunks": len(chunks),
        "skipped": 0,
        "failed": 0,
        "reason": "",
    }


def embed_all_reports(
    con: duckdb.DuckDBPyConnection | None = None,
    *,
    tickers: list[str] | None = None,
    years: list[int] | None = None,
    replace: bool = False,
    model: str = EMBEDDING_MODEL,
    dimensions: int = EMBEDDING_DIMENSIONS,
    chunk_size: int = EMBEDDING_CHUNK_SIZE,
    chunk_overlap: int = EMBEDDING_CHUNK_OVERLAP,
    progress_callback=None,
) -> dict[str, int]:
    """Embed all filtered annual reports into document_embeddings."""
    own_con = con is None
    if own_con:
        con = get_connection()

    try:
        rows = con.execute(
            "SELECT ticker, year FROM annual_reports ORDER BY ticker, year"
        ).fetchall()
        reports = [(r[0], r[1]) for r in rows]

        if tickers:
            tset = {t.upper() for t in tickers}
            reports = [(t, y) for t, y in reports if t in tset]
        if years:
            yset = set(years)
            reports = [(t, y) for t, y in reports if y in yset]

        results = {
            "embedded_reports": 0,
            "embedded_chunks": 0,
            "skipped_reports": 0,
            "failed_reports": 0,
            "total_reports": len(reports),
        }

        for idx, (ticker, year) in enumerate(reports):
            try:
                out = embed_report(
                    con,
                    ticker,
                    year,
                    replace=replace,
                    model=model,
                    dimensions=dimensions,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                if out["skipped"]:
                    results["skipped_reports"] += 1
                else:
                    results["embedded_reports"] += 1
                    results["embedded_chunks"] += int(out["embedded"])
                if progress_callback:
                    progress_callback(
                        ticker,
                        year,
                        out,
                        idx + 1,
                        len(reports),
                    )
            except Exception as exc:
                results["failed_reports"] += 1
                if progress_callback:
                    progress_callback(
                        ticker,
                        year,
                        {
                            "embedded": 0,
                            "chunks": 0,
                            "skipped": 0,
                            "failed": 1,
                            "reason": str(exc),
                        },
                        idx + 1,
                        len(reports),
                    )

        return results
    finally:
        if own_con:
            con.close()
