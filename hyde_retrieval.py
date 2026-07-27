"""HyDe-2 query embedding helper for inference retrieval tasks."""

from __future__ import annotations

import json
import logging
import threading
from functools import lru_cache
from pathlib import Path

from config import (
    DATA_DIR,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    HYDE2_MODEL,
    HYDE2_SYNTHETIC_DOC_COUNT,
    INFERENCE_TEMPERATURE,
)
from embedder import _get_client, get_embeddings
from llm_batch_api import run_chat_json_batch

logger = logging.getLogger(__name__)

_HYDE_HYPOTHESES_CACHE: dict[tuple[str, str, int], tuple[str, ...]] = {}
_HYDE_EMBEDDING_CACHE: dict[tuple[str, str, int, str, int], tuple[float, ...]] = {}
_CACHE_LOCK = threading.RLock()
_CACHE_LOADED = False
_CACHE_PATH = Path(DATA_DIR) / "cache" / "hyde2_cache.json"


def _ensure_cache_loaded() -> None:
    """Load persisted HyDE caches once per process."""
    global _CACHE_LOADED
    with _CACHE_LOCK:
        if _CACHE_LOADED:
            return
        _CACHE_LOADED = True

        if not _CACHE_PATH.exists():
            return

        try:
            payload = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load HyDE cache file %s: %s", _CACHE_PATH, exc)
            return

        hypotheses_rows = payload.get("hypotheses", [])
        for row in hypotheses_rows:
            try:
                key = (
                    str(row["criterion"]),
                    str(row["llm_model"]),
                    int(row["synthetic_doc_count"]),
                )
                values = tuple(str(v) for v in row.get("hypotheses", []))
                if values:
                    _HYDE_HYPOTHESES_CACHE[key] = values
            except Exception:
                continue

        embedding_rows = payload.get("embeddings", [])
        for row in embedding_rows:
            try:
                key = (
                    str(row["criterion"]),
                    str(row["embedding_model"]),
                    int(row["dimensions"]),
                    str(row["llm_model"]),
                    int(row["synthetic_doc_count"]),
                )
                vec = tuple(float(v) for v in row.get("embedding", []))
                if vec:
                    _HYDE_EMBEDDING_CACHE[key] = vec
            except Exception:
                continue


def _save_cache() -> None:
    """Persist in-memory HyDE caches for reuse on next run."""
    with _CACHE_LOCK:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "hypotheses": [
                {
                    "criterion": key[0],
                    "llm_model": key[1],
                    "synthetic_doc_count": key[2],
                    "hypotheses": list(value),
                }
                for key, value in _HYDE_HYPOTHESES_CACHE.items()
            ],
            "embeddings": [
                {
                    "criterion": key[0],
                    "embedding_model": key[1],
                    "dimensions": key[2],
                    "llm_model": key[3],
                    "synthetic_doc_count": key[4],
                    "embedding": list(value),
                }
                for key, value in _HYDE_EMBEDDING_CACHE.items()
            ],
        }
        _CACHE_PATH.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

_HYDE_PROMPT = """You generate a hypothetical report excerpt used for semantic retrieval.
Write a concise paragraph in Vietnamese that could appear in an annual report and would satisfy the criterion below.

Return only a JSON object with a single field:
- \"hypothesis\": string

CRITERION:
{criterion}
"""


@lru_cache(maxsize=1024)
def _cached_hyde_hypotheses(
    criterion: str,
    llm_model: str,
    synthetic_doc_count: int,
) -> tuple[str, ...]:
    _ensure_cache_loaded()
    cache_key = (criterion, llm_model, synthetic_doc_count)
    cached = _HYDE_HYPOTHESES_CACHE.get(cache_key)
    if cached is not None:
        return cached

    count = max(1, synthetic_doc_count)
    prompts = {
        f"h{i}": _HYDE_PROMPT.format(criterion=criterion)
        for i in range(1, count + 1)
    }
    client = _get_client()
    is_reasoning = llm_model.startswith(("o1", "o3", "o4", "gpt-5"))

    response_map = run_chat_json_batch(
        client=client,
        model=llm_model,
        prompts=prompts,
        is_reasoning=is_reasoning,
        temperature=INFERENCE_TEMPERATURE,
    )

    hypotheses: list[str] = []
    for key in sorted(response_map.keys()):
        raw = response_map[key]
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"hypothesis": str(raw)}
        text = str(payload.get("hypothesis", "")).strip()
        if text:
            hypotheses.append(text)
    resolved = tuple(hypotheses)
    if resolved:
        _HYDE_HYPOTHESES_CACHE[cache_key] = resolved
        _save_cache()
    return resolved


def _batch_generate_hypotheses(
    criteria: list[str],
    *,
    llm_model: str,
    synthetic_doc_count: int,
) -> dict[str, tuple[str, ...]]:
    """Generate hypotheses for many criteria in a single OpenAI batch call."""
    count = max(1, synthetic_doc_count)
    prompts: dict[str, str] = {}
    for cidx, criterion in enumerate(criteria):
        for hidx in range(1, count + 1):
            prompts[f"c{cidx}_h{hidx}"] = _HYDE_PROMPT.format(criterion=criterion)

    client = _get_client()
    is_reasoning = llm_model.startswith(("o1", "o3", "o4", "gpt-5"))
    response_map = run_chat_json_batch(
        client=client,
        model=llm_model,
        prompts=prompts,
        is_reasoning=is_reasoning,
        temperature=INFERENCE_TEMPERATURE,
    )

    by_criterion: dict[str, list[str]] = {criterion: [] for criterion in criteria}
    for cidx, criterion in enumerate(criteria):
        for hidx in range(1, count + 1):
            key = f"c{cidx}_h{hidx}"
            raw = response_map.get(key, "")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"hypothesis": str(raw)}
            text = str(payload.get("hypothesis", "")).strip()
            if text:
                by_criterion[criterion].append(text)

    return {k: tuple(v) for k, v in by_criterion.items()}


@lru_cache(maxsize=1024)
def _cached_hyde_embedding(
    criterion: str,
    embedding_model: str,
    dimensions: int,
    llm_model: str,
    synthetic_doc_count: int,
) -> tuple[float, ...]:
    hypothesis_docs = _cached_hyde_hypotheses(
        criterion,
        llm_model,
        synthetic_doc_count,
    )

    texts = [criterion, *hypothesis_docs]
    vectors = get_embeddings(
        texts,
        model=embedding_model,
        dimensions=dimensions,
    )
    if not vectors:
        raise RuntimeError("No embeddings generated for HyDe-2")

    dim = len(vectors[0])
    accumulator = [0.0] * dim
    for vec in vectors:
        for i, value in enumerate(vec):
            accumulator[i] += float(value)
    factor = 1.0 / float(len(vectors))
    return tuple(value * factor for value in accumulator)


def build_hyde2_embedding(
    criterion: str,
    *,
    embedding_model: str | None = None,
    dimensions: int | None = None,
    llm_model: str | None = None,
    synthetic_doc_count: int | None = None,
) -> list[float]:
    """Build a HyDe-2 embedding by averaging criterion + synthetic hypotheses."""
    _ensure_cache_loaded()
    embedding_model = embedding_model or EMBEDDING_MODEL
    dimensions = dimensions or EMBEDDING_DIMENSIONS
    llm_model = llm_model or HYDE2_MODEL
    synthetic_doc_count = synthetic_doc_count or HYDE2_SYNTHETIC_DOC_COUNT

    cache_key = (
        criterion,
        embedding_model,
        dimensions,
        llm_model,
        synthetic_doc_count,
    )

    try:
        if cache_key in _HYDE_EMBEDDING_CACHE:
            return list(_HYDE_EMBEDDING_CACHE[cache_key])

        vec = _cached_hyde_embedding(
            criterion,
            embedding_model,
            dimensions,
            llm_model,
            synthetic_doc_count,
        )
        _HYDE_EMBEDDING_CACHE[cache_key] = vec
        _save_cache()
        return list(vec)
    except Exception as exc:
        logger.warning("HyDe-2 fallback to plain embedding: %s", exc)
        return get_embeddings(
            [criterion],
            model=embedding_model,
            dimensions=dimensions,
        )[0]


def precompute_hyde2_embeddings(
    criteria: list[str],
    *,
    embedding_model: str | None = None,
    dimensions: int | None = None,
    llm_model: str | None = None,
    synthetic_doc_count: int | None = None,
) -> None:
    """Pre-generate HyDe hypotheses/embeddings for criterion-only prompts.

    Use this for prompts that do not depend on ticker/year so retrieval can
    run without first-call LLM latency.
    """
    _ensure_cache_loaded()
    embedding_model = embedding_model or EMBEDDING_MODEL
    dimensions = dimensions or EMBEDDING_DIMENSIONS
    llm_model = llm_model or HYDE2_MODEL
    synthetic_doc_count = synthetic_doc_count or HYDE2_SYNTHETIC_DOC_COUNT

    normalized = [c.strip() for c in criteria if str(c).strip()]
    unique_criteria = list(dict.fromkeys(normalized))
    if not unique_criteria:
        return

    missing_hypotheses: list[str] = []
    for criterion in unique_criteria:
        hyp_key = (criterion, llm_model, synthetic_doc_count)
        if hyp_key not in _HYDE_HYPOTHESES_CACHE:
            missing_hypotheses.append(criterion)

    if missing_hypotheses:
        batch_hyp = _batch_generate_hypotheses(
            missing_hypotheses,
            llm_model=llm_model,
            synthetic_doc_count=synthetic_doc_count,
        )
        for criterion, hypotheses in batch_hyp.items():
            _HYDE_HYPOTHESES_CACHE[(criterion, llm_model, synthetic_doc_count)] = hypotheses
        _save_cache()

    embed_targets: list[str] = []
    for criterion in unique_criteria:
        emb_key = (
            criterion,
            embedding_model,
            dimensions,
            llm_model,
            synthetic_doc_count,
        )
        if emb_key not in _HYDE_EMBEDDING_CACHE:
            embed_targets.append(criterion)

    if not embed_targets:
        return

    all_texts: list[str] = []
    ranges: list[tuple[str, int, int]] = []
    for criterion in embed_targets:
        hyp_key = (criterion, llm_model, synthetic_doc_count)
        hypotheses = _HYDE_HYPOTHESES_CACHE.get(hyp_key)
        if hypotheses is None:
            hypotheses = _cached_hyde_hypotheses(
                criterion,
                llm_model,
                synthetic_doc_count,
            )
            _HYDE_HYPOTHESES_CACHE[hyp_key] = hypotheses

        texts = [criterion, *hypotheses]
        start = len(all_texts)
        all_texts.extend(texts)
        ranges.append((criterion, start, len(texts)))

    vectors = get_embeddings(
        all_texts,
        model=embedding_model,
        dimensions=dimensions,
    )

    for criterion, start, count in ranges:
        subset = vectors[start : start + count]
        if not subset:
            continue
        dim = len(subset[0])
        accumulator = [0.0] * dim
        for vec in subset:
            for i, value in enumerate(vec):
                accumulator[i] += float(value)
        factor = 1.0 / float(len(subset))
        emb_key = (
            criterion,
            embedding_model,
            dimensions,
            llm_model,
            synthetic_doc_count,
        )
        _HYDE_EMBEDDING_CACHE[emb_key] = tuple(value * factor for value in accumulator)

    _save_cache()
