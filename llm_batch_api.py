"""Shared OpenAI Batch API helpers for chat-completion JSON tasks."""

from __future__ import annotations

import json
import time
from typing import Any

from openai import OpenAI


def _read_file_text(client: OpenAI, file_id: str) -> str:
    """Read a files API object as UTF-8 text across SDK variants."""
    retrieve_content = getattr(client.files, "retrieve_content", None)
    if callable(retrieve_content):
        data = retrieve_content(file_id)
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="replace")
        return str(data)

    content_resp = client.files.content(file_id)

    text_attr = getattr(content_resp, "text", None)
    if callable(text_attr):
        return str(text_attr())
    if isinstance(text_attr, str):
        return text_attr

    read_fn = getattr(content_resp, "read", None)
    if callable(read_fn):
        data = read_fn()
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="replace")
        return str(data)

    content_attr = getattr(content_resp, "content", None)
    if callable(content_attr):
        data = content_attr()
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="replace")
        return str(data)
    if isinstance(content_attr, bytes):
        return content_attr.decode("utf-8", errors="replace")

    return str(content_resp)


def run_chat_json_batch(
    *,
    client: OpenAI,
    model: str,
    prompts: dict[str, str],
    is_reasoning: bool,
    temperature: float,
    reasoning_effort: str = "low",
    poll_interval_seconds: float = 2.0,
    timeout_seconds: float = 900.0,
) -> dict[str, str]:
    """Run chat-completion JSON requests via OpenAI Batch API.

    Returns a mapping of ``custom_id`` to raw JSON-string model content.
    Raises RuntimeError on batch or per-request failures.
    """
    if not prompts:
        return {}

    lines: list[str] = []
    for custom_id, prompt in prompts.items():
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        if is_reasoning:
            body["reasoning_effort"] = reasoning_effort
        else:
            body["temperature"] = temperature

        lines.append(
            json.dumps(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                },
                ensure_ascii=False,
            )
        )

    payload = ("\n".join(lines) + "\n").encode("utf-8")

    input_file = client.files.create(
        file=("batch_input.jsonl", payload, "application/jsonl"),
        purpose="batch",
    )

    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    start = time.time()
    while batch.status in {
        "validating",
        "in_progress",
        "finalizing",
        "cancelling",
    }:
        if timeout_seconds > 0 and (time.time() - start) > timeout_seconds:
            raise TimeoutError(
                f"Batch {batch.id} did not complete within {timeout_seconds} seconds"
            )
        time.sleep(max(0.5, poll_interval_seconds))
        batch = client.batches.retrieve(batch.id)

    if batch.status != "completed":
        error_text = ""
        error_file_id = getattr(batch, "error_file_id", None)
        if isinstance(error_file_id, str) and error_file_id:
            error_text = _read_file_text(client, error_file_id)
        raise RuntimeError(
            f"Batch {batch.id} failed with status={batch.status}. {error_text[:4000]}"
        )

    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        raise RuntimeError(f"Batch {batch.id} completed but has no output file")

    output_text = _read_file_text(client, output_file_id)

    results: dict[str, str] = {}
    failures: list[str] = []

    for raw_line in output_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        record = json.loads(line)
        custom_id = str(record.get("custom_id") or "")
        response = record.get("response") or {}
        status_code = int(response.get("status_code") or 0)
        body = response.get("body") or {}

        if status_code != 200:
            failures.append(
                f"{custom_id}: status={status_code} body={json.dumps(body, ensure_ascii=False)[:500]}"
            )
            continue

        choices = body.get("choices") or []
        message = choices[0].get("message") if choices else {}
        content = (message or {}).get("content") or ""
        results[custom_id] = str(content)

    missing = sorted(set(prompts.keys()) - set(results.keys()))
    if missing:
        failures.append(f"Missing outputs for custom_ids: {', '.join(missing)}")

    if failures:
        raise RuntimeError("Batch request failures: " + " | ".join(failures)[:4000])

    return results


def submit_chat_json_batch(
    *,
    client: OpenAI,
    model: str,
    prompts: dict[str, str],
    is_reasoning: bool,
    temperature: float,
    reasoning_effort: str = "low",
) -> str:
    """Submit a chat-completion batch and return OpenAI batch_id without waiting."""
    if not prompts:
        raise ValueError("prompts must not be empty")

    lines: list[str] = []
    for custom_id, prompt in prompts.items():
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        if is_reasoning:
            body["reasoning_effort"] = reasoning_effort
        else:
            body["temperature"] = temperature

        lines.append(
            json.dumps(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                },
                ensure_ascii=False,
            )
        )

    payload = ("\n".join(lines) + "\n").encode("utf-8")
    input_file = client.files.create(
        file=("batch_input.jsonl", payload, "application/jsonl"),
        purpose="batch",
    )
    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    return str(batch.id)


def get_batch_status(client: OpenAI, batch_id: str) -> str:
    """Return current OpenAI batch status."""
    batch = client.batches.retrieve(batch_id)
    return str(batch.status)


def get_batch_output_map(client: OpenAI, batch_id: str) -> dict[str, str]:
    """Fetch completed batch outputs as ``custom_id -> raw content`` map."""
    batch = client.batches.retrieve(batch_id)
    if str(batch.status) != "completed":
        raise RuntimeError(f"Batch {batch_id} is not completed (status={batch.status})")

    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        raise RuntimeError(f"Batch {batch_id} completed but has no output file")

    output_text = _read_file_text(client, str(output_file_id))
    results: dict[str, str] = {}
    failures: list[str] = []

    for raw_line in output_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        record = json.loads(line)
        custom_id = str(record.get("custom_id") or "")
        response = record.get("response") or {}
        status_code = int(response.get("status_code") or 0)
        body = response.get("body") or {}

        if status_code != 200:
            failures.append(
                f"{custom_id}: status={status_code} body={json.dumps(body, ensure_ascii=False)[:500]}"
            )
            continue

        choices = body.get("choices") or []
        message = choices[0].get("message") if choices else {}
        content = (message or {}).get("content") or ""
        results[custom_id] = str(content)

    if failures:
        raise RuntimeError("Batch request failures: " + " | ".join(failures)[:4000])

    return results
