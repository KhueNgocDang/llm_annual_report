"""Embedding helpers for annual report and criteria retrieval workflows."""

from __future__ import annotations

import os
from typing import Sequence

from openai import OpenAI

from config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL

try:
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None


MAX_EMBED_REQUEST_TOKENS = 240000
MAX_EMBED_INPUT_TOKENS = 8000


def _is_token_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "requested" in msg
        and "token" in msg
        and ("max" in msg or "maximum" in msg)
    )


def _estimate_tokens(text: str) -> int:
    if tiktoken is not None:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    return max(1, len(text) // 4)


def _truncate_to_token_limit(text: str, max_tokens: int) -> str:
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


def _get_client() -> OpenAI:
    """Create an OpenAI client from environment configuration."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI()


def get_embeddings(
    texts: Sequence[str],
    *,
    model: str | None = None,
    dimensions: int | None = None,
    batch_size: int | None = None,
) -> list[list[float]]:
    """Return embedding vectors for input texts.

    The order of returned vectors always matches the input text order.
    """
    if not texts:
        return []

    selected_model = model or EMBEDDING_MODEL
    selected_batch = batch_size or EMBEDDING_BATCH_SIZE
    client = _get_client()

    vectors: list[list[float]] = []
    normalized = [
        _truncate_to_token_limit(str(text or ""), MAX_EMBED_INPUT_TOKENS)
        for text in texts
    ]

    idx = 0
    total = len(normalized)
    while idx < total:
        batch: list[str] = []
        batch_tokens = 0
        while idx < total and len(batch) < selected_batch:
            text = normalized[idx]
            token_count = _estimate_tokens(text)
            if batch and batch_tokens + token_count > MAX_EMBED_REQUEST_TOKENS:
                break
            batch.append(text)
            batch_tokens += token_count
            idx += 1

        pending_batches: list[list[str]] = [batch]
        while pending_batches:
            current = pending_batches.pop(0)
            payload: dict = {
                "model": selected_model,
                "input": current,
            }
            if dimensions is not None:
                payload["dimensions"] = dimensions

            try:
                response = client.embeddings.create(**payload)
                vectors.extend([item.embedding for item in response.data])
                continue
            except Exception as exc:
                if not _is_token_limit_error(exc):
                    raise

                if len(current) > 1:
                    mid = len(current) // 2
                    pending_batches = [current[:mid], current[mid:]] + pending_batches
                    continue

                text = current[0]
                current_tokens = _estimate_tokens(text)
                new_limit = max(512, int(current_tokens * 0.7))
                smaller = _truncate_to_token_limit(text, new_limit)
                if not smaller or smaller == text:
                    raise
                pending_batches = [[smaller]] + pending_batches

    return vectors
