"""RAG + LLM helpers for reference methodology retrieval.

This module builds a small reference knowledge base from PRD_LLM.md and
METHODOLOGY.md, then provides retrieval and structured generation utilities
for EDC and PROPER-VN criteria.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from config import BASE_DIR, INFERENCE_TEMPERATURE
from database import get_connection, init_db

try:
    import tiktoken
except Exception:  # pragma: no cover - optional at runtime
    tiktoken = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional at runtime
    OpenAI = None


REFERENCE_FILES = ["PRD_LLM.md", "METHODOLOGY.md"]
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-4.1-mini"
CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 128
EMBED_BATCH_SIZE = 128

EMBEDDING_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

EDC_INDICATORS = [
    "CC1",
    "CC2",
    "GHG1",
    "GHG2",
    "GHG3",
    "GHG4",
    "GHG5",
    "GHG6",
    "GHG7",
    "EC1",
    "EC2",
    "EC3",
    "RC1",
    "RC2",
    "RC3",
    "RC4",
    "ACC1",
    "ACC2",
]

PROPER_VN_INDICATORS = [
    "S1_VIOLATION",
    "S1_MINOR_NC",
    "S1_COMPLIANCE",
    "S2_ISO14001",
    "S2_CARBON_DISC",
    "S2_REDUCTION",
    "S2_EFFICIENCY",
]


@dataclass
class _Chunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    chunk_text: str
    token_count: int


def _get_openai_client() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("openai package is required. Install dependencies first.")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_framework(framework: str | None) -> str | None:
    if framework is None:
        return None
    normalized = framework.strip().upper()
    if normalized in {"PROPER", "PROPER-VN", "PROPER_VN"}:
        return "PROPER_VN"
    if normalized == "EDC":
        return "EDC"
    raise ValueError(f"Unsupported framework: {framework}")


def _required_indicators(framework: str) -> list[str]:
    if framework == "EDC":
        return EDC_INDICATORS
    if framework == "PROPER_VN":
        return PROPER_VN_INDICATORS
    raise ValueError(f"Unsupported framework: {framework}")


def _scoring_rule_for_framework(framework: str) -> str:
    if framework == "EDC":
        return "Binary scoring: 1 if valid disclosure evidence exists, else 0."
    return (
        "Stage 1 priority rules (S1_VIOLATION -> Black, S1_MINOR_NC without "
        "S1_COMPLIANCE -> Red). Stage 2 evidence levels: none=0, basic_mention=1, "
        "quantified=2; color thresholds: Blue < 3, Green 3-5, Gold >= 6."
    )


def _tokenize_text(text: str) -> list[int]:
    if tiktoken is not None:
        enc = tiktoken.get_encoding("cl100k_base")
        return enc.encode(text)
    # Fallback if tiktoken is unavailable at runtime.
    return [hash(tok) % 1000003 for tok in text.split()]


def _detokenize_tokens(tokens: list[int]) -> str:
    if tiktoken is not None:
        enc = tiktoken.get_encoding("cl100k_base")
        return enc.decode(tokens)
    # Fallback cannot reconstruct exact text, use placeholder join by id.
    return " ".join(str(t) for t in tokens)


def _chunk_text(doc_id: str, text: str) -> list[_Chunk]:
    tokens = _tokenize_text(text)
    if not tokens:
        return []

    chunks: list[_Chunk] = []
    step = max(CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS, 1)
    chunk_index = 0
    for start in range(0, len(tokens), step):
        window = tokens[start : start + CHUNK_SIZE_TOKENS]
        if not window:
            continue
        chunk_text = _detokenize_tokens(window).strip()
        if not chunk_text:
            continue
        chunk_id = f"{doc_id}:{chunk_index}"
        chunks.append(
            _Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                chunk_index=chunk_index,
                chunk_text=chunk_text,
                token_count=len(window),
            )
        )
        chunk_index += 1
        if start + CHUNK_SIZE_TOKENS >= len(tokens):
            break
    return chunks


def _iter_batches(values: list[str], batch_size: int) -> Iterable[list[str]]:
    for i in range(0, len(values), batch_size):
        yield values[i : i + batch_size]


def _embed_texts(texts: list[str], model: str = DEFAULT_EMBEDDING_MODEL) -> list[list[float]]:
    if not texts:
        return []
    client = _get_openai_client()
    vectors: list[list[float]] = []
    for batch in _iter_batches(texts, EMBED_BATCH_SIZE):
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend([item.embedding for item in response.data])
    return vectors


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return -1.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return -1.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _is_reasoning_model(model: str) -> bool:
    m = model.lower()
    return m.startswith("o") or m.startswith("gpt-5")


def _parse_json_or_raise(raw_text: str) -> dict:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON: {exc}") from exc


def _build_generation_prompt(query: str, contexts: list[dict]) -> str:
    context_lines: list[str] = []
    for idx, ctx in enumerate(contexts, 1):
        context_lines.append(
            f"[{idx}] source={ctx['source_file']} chunk_id={ctx['chunk_id']} "
            f"score={ctx['similarity_score']:.4f}\n{ctx['chunk_text']}"
        )

    context_blob = "\n\n".join(context_lines)
    return (
        "You are a strict information extraction assistant. "
        "Use ONLY the provided context and respond with JSON only.\n\n"
        "Task query:\n"
        f"{query}\n\n"
        "Context:\n"
        f"{context_blob}\n\n"
        "Return a JSON object with keys: "
        "summary, key_points, scoring_rule, constraints, assumptions, citations. "
        "`citations` must be an array of objects with keys: source_file, chunk_id."
    )


def build_reference_kb(force_rebuild: bool = False) -> dict:
    """Build/refresh a reference KB from PRD_LLM.md and METHODOLOGY.md.

    Args:
        force_rebuild: Rebuild even if file hash has not changed.

    Returns:
        Summary counts for documents, chunks, embeddings, and skipped docs.
    """

    con = get_connection()
    try:
        init_db(con)
        source_paths = [BASE_DIR / name for name in REFERENCE_FILES]
        missing = [str(p.name) for p in source_paths if not p.exists()]
        if missing:
            missing_str = ", ".join(missing)
            raise FileNotFoundError(
                f"Missing required reference file(s): {missing_str}"
            )

        inserted_docs = 0
        skipped_docs = 0
        total_chunks = 0
        total_embeddings = 0

        for path in source_paths:
            source_file = path.name
            doc_id = path.stem.lower().replace(" ", "_")
            content = path.read_text(encoding="utf-8")
            content_hash = _sha256_text(content)

            existing = con.execute(
                "SELECT content_hash FROM reference_documents WHERE doc_id = ?",
                [doc_id],
            ).fetchone()

            unchanged = existing is not None and existing[0] == content_hash
            if unchanged and not force_rebuild:
                skipped_docs += 1
                continue

            con.execute(
                """
                INSERT INTO reference_documents (doc_id, source_file, title, content, content_hash, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT (doc_id) DO UPDATE SET
                    source_file = EXCLUDED.source_file,
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    content_hash = EXCLUDED.content_hash,
                    updated_at = CURRENT_TIMESTAMP
                """,
                [doc_id, source_file, path.stem, content, content_hash],
            )

            con.execute(
                "DELETE FROM reference_embeddings WHERE chunk_id IN "
                "(SELECT chunk_id FROM reference_chunks WHERE doc_id = ?)",
                [doc_id],
            )
            con.execute("DELETE FROM reference_chunks WHERE doc_id = ?", [doc_id])

            chunks = _chunk_text(doc_id=doc_id, text=content)
            if chunks:
                con.executemany(
                    """
                    INSERT INTO reference_chunks (chunk_id, doc_id, chunk_index, chunk_text, token_count)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            c.chunk_id,
                            c.doc_id,
                            c.chunk_index,
                            c.chunk_text,
                            c.token_count,
                        )
                        for c in chunks
                    ],
                )

                vectors = _embed_texts(
                    [c.chunk_text for c in chunks],
                    model=DEFAULT_EMBEDDING_MODEL,
                )
                dims = EMBEDDING_DIMS.get(DEFAULT_EMBEDDING_MODEL, len(vectors[0])) if vectors else 0
                con.executemany(
                    """
                    INSERT INTO reference_embeddings (chunk_id, model, dimensions, embedding)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        model = EXCLUDED.model,
                        dimensions = EXCLUDED.dimensions,
                        embedding = EXCLUDED.embedding,
                        created_at = CURRENT_TIMESTAMP
                    """,
                    [
                        (
                            chunks[idx].chunk_id,
                            DEFAULT_EMBEDDING_MODEL,
                            dims,
                            vectors[idx],
                        )
                        for idx in range(len(chunks))
                    ],
                )
                total_chunks += len(chunks)
                total_embeddings += len(vectors)

            inserted_docs += 1

        return {
            "documents": inserted_docs,
            "chunks": total_chunks,
            "embeddings": total_embeddings,
            "skipped": skipped_docs,
        }
    finally:
        con.close()


def retrieve_required_information(
    query: str,
    framework: str | None = None,
    indicator_code: str | None = None,
    topic: str | None = None,
    top_k: int = 5,
    min_score: float = 0.20,
) -> list[dict]:
    """Retrieve ranked reference chunks for the given requirement query."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if not query.strip():
        raise ValueError("query must not be empty")

    normalized_framework = _normalize_framework(framework)
    query_vector = _embed_texts([query], model=DEFAULT_EMBEDDING_MODEL)[0]

    con = get_connection()
    try:
        init_db(con)
        rows = con.execute(
            """
            SELECT
                rc.chunk_id,
                rd.source_file,
                rc.chunk_text,
                re.embedding
            FROM reference_chunks rc
            JOIN reference_documents rd ON rd.doc_id = rc.doc_id
            JOIN reference_embeddings re ON re.chunk_id = rc.chunk_id
            WHERE re.model = ?
            """,
            [DEFAULT_EMBEDDING_MODEL],
        ).fetchall()
    finally:
        con.close()

    if not rows:
        return []

    needle_parts: list[str] = [query]
    if indicator_code:
        needle_parts.append(indicator_code)
    if topic:
        needle_parts.append(topic)
    if normalized_framework == "EDC":
        needle_parts.extend(["checklist", "edc", "disclosure"])
    if normalized_framework == "PROPER_VN":
        needle_parts.extend(["proper-vn", "stage 1", "stage 2", "color"])

    needles = [part.strip().lower() for part in needle_parts if part and part.strip()]

    scored: list[dict] = []
    for chunk_id, source_file, chunk_text, vector in rows:
        text_lower = chunk_text.lower()
        if indicator_code and indicator_code.lower() not in text_lower:
            # Hard filter on explicit indicator code when provided.
            continue
        if topic and topic.lower() not in text_lower:
            continue

        sim = _cosine_similarity(query_vector, vector)
        if sim < min_score:
            continue

        bonus = 0.0
        for needle in needles:
            if needle in text_lower:
                bonus += 0.01

        scored.append(
            {
                "chunk_id": chunk_id,
                "source_file": source_file,
                "chunk_text": chunk_text,
                "similarity_score": sim + bonus,
            }
        )

    scored.sort(key=lambda item: item["similarity_score"], reverse=True)
    return scored[:top_k]


def get_indicator_criteria(indicator_code: str, framework: str) -> dict:
    """Return framework indicator criteria with scoring rule and citations."""

    normalized_framework = _normalize_framework(framework)
    if normalized_framework is None:
        raise ValueError("framework is required")

    indicator = indicator_code.strip().upper()
    required = _required_indicators(normalized_framework)
    if indicator not in required:
        raise ValueError(
            f"Indicator {indicator} is not valid for framework {normalized_framework}"
        )

    contexts = retrieve_required_information(
        query=f"Criteria and scoring for indicator {indicator} in {normalized_framework}",
        framework=normalized_framework,
        indicator_code=indicator,
        top_k=5,
        min_score=0.0,
    )

    criteria_text = "\n\n".join(ctx["chunk_text"] for ctx in contexts[:3]).strip()
    if not criteria_text:
        criteria_text = f"No criteria text found for {indicator} in {normalized_framework}."

    citations = [
        {
            "source_file": ctx["source_file"],
            "chunk_id": ctx["chunk_id"],
            "score": round(float(ctx["similarity_score"]), 6),
        }
        for ctx in contexts
    ]

    return {
        "indicator_code": indicator,
        "framework": normalized_framework,
        "criteria_text": criteria_text,
        "scoring_rule": _scoring_rule_for_framework(normalized_framework),
        "source_citations": citations,
    }


def generate_structured_requirement(
    query: str,
    contexts: list[dict],
    llm_model: str = DEFAULT_LLM_MODEL,
) -> dict:
    """Generate strict JSON requirement output from retrieved contexts."""

    if not contexts:
        raise ValueError("contexts must not be empty")

    client = _get_openai_client()
    prompt = _build_generation_prompt(query=query, contexts=contexts)

    request_kwargs = {
        "model": llm_model,
        "messages": [
            {
                "role": "system",
                "content": "Return valid JSON only. Do not add markdown fences.",
            },
            {"role": "user", "content": prompt},
        ],
    }

    if _is_reasoning_model(llm_model):
        request_kwargs["reasoning_effort"] = "low"
    else:
        request_kwargs["temperature"] = INFERENCE_TEMPERATURE

    response = client.chat.completions.create(**request_kwargs)
    content = (response.choices[0].message.content or "").strip()
    try:
        return _parse_json_or_raise(content)
    except ValueError:
        # Single strict retry when first response is not valid JSON.
        retry_messages = request_kwargs["messages"] + [
            {
                "role": "assistant",
                "content": content,
            },
            {
                "role": "user",
                "content": "Your previous response was invalid JSON. Return only a valid JSON object matching the required schema.",
            },
        ]
        retry_kwargs = dict(request_kwargs)
        retry_kwargs["messages"] = retry_messages
        retry = client.chat.completions.create(**retry_kwargs)
        retry_content = (retry.choices[0].message.content or "").strip()
        return _parse_json_or_raise(retry_content)


def retrieve_and_generate_requirement(
    query: str,
    framework: str | None = None,
    indicator_code: str | None = None,
    top_k: int = 5,
    llm_model: str = DEFAULT_LLM_MODEL,
) -> dict:
    """Orchestrate retrieval + JSON generation and persist an audit log."""

    start = time.perf_counter()
    normalized_framework = _normalize_framework(framework)
    contexts = retrieve_required_information(
        query=query,
        framework=normalized_framework,
        indicator_code=indicator_code,
        top_k=top_k,
    )
    answer_json = generate_structured_requirement(
        query=query,
        contexts=contexts,
        llm_model=llm_model,
    )
    latency_ms = int((time.perf_counter() - start) * 1000)

    request_id = str(uuid.uuid4())
    con = get_connection()
    try:
        init_db(con)
        con.execute(
            """
            INSERT INTO reference_retrieval_logs (
                request_id,
                query,
                framework,
                indicator_code,
                top_k,
                embedding_model,
                llm_model,
                response_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                request_id,
                query,
                normalized_framework,
                indicator_code,
                top_k,
                DEFAULT_EMBEDDING_MODEL,
                llm_model,
                json.dumps(answer_json, ensure_ascii=False),
            ],
        )
    finally:
        con.close()

    return {
        "request_id": request_id,
        "query": query,
        "contexts": contexts,
        "answer_json": answer_json,
        "latency_ms": latency_ms,
    }


def validate_requirement_coverage(framework: str) -> dict:
    """Validate whether required indicator criteria are retrievable with confidence."""

    normalized_framework = _normalize_framework(framework)
    if normalized_framework is None:
        raise ValueError("framework is required")

    indicators = _required_indicators(normalized_framework)
    missing: list[str] = []
    low_confidence: list[dict] = []
    coverage_details: list[dict] = []

    for indicator in indicators:
        contexts = retrieve_required_information(
            query=f"{normalized_framework} {indicator} criteria",
            framework=normalized_framework,
            indicator_code=indicator,
            top_k=3,
            min_score=0.0,
        )
        top_score = float(contexts[0]["similarity_score"]) if contexts else 0.0
        coverage_details.append(
            {
                "indicator_code": indicator,
                "contexts_found": len(contexts),
                "top_score": round(top_score, 6),
            }
        )

        if not contexts:
            missing.append(indicator)
            continue
        if top_score < 0.35:
            low_confidence.append(
                {
                    "indicator_code": indicator,
                    "top_score": round(top_score, 6),
                }
            )

    covered = len(indicators) - len(missing)
    suggestions: list[str] = []
    if missing:
        suggestions.append(
            "Review PRD_LLM.md and METHODOLOGY.md for missing indicator language, then rebuild reference KB."
        )
    if low_confidence:
        suggestions.append(
            "Add clearer indicator headings/keywords in reference docs or reduce retrieval threshold for these indicators."
        )
    if not suggestions:
        suggestions.append("Coverage is sufficient for inference preflight.")

    return {
        "framework": normalized_framework,
        "total_indicators": len(indicators),
        "covered_indicators": covered,
        "missing_indicators": missing,
        "low_confidence_indicators": low_confidence,
        "coverage_details": coverage_details,
        "is_ready": not missing,
        "suggested_remediation": suggestions,
    }
