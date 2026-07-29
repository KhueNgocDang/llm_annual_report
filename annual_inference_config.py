"""Shared configuration helpers for annual-report inference tasks."""

from __future__ import annotations

from config import (
    CHECKLIST_ITEMS,
    CHECKLIST_ITEMS_ALT,
    CHECKLIST_ITEMS_ALT_TWO,
    GOVERNANCE_EXTRACTION_ITEMS,
    PROPER_VN_ALL_ITEMS,
)

# GOV_AUDIT belongs to BCTC/financial-statement extraction, not annual reports.
GOVERNANCE_ANNUAL_ITEMS: list[dict[str, str]] = [
    item for item in GOVERNANCE_EXTRACTION_ITEMS if item.get("code") != "GOV_AUDIT"
]

ANNUAL_INFERENCE_TASK_ITEMS: dict[str, list[dict[str, str]]] = {
    "edc": CHECKLIST_ITEMS,
    "edc_alt": CHECKLIST_ITEMS_ALT,
    "edc_alt_two": CHECKLIST_ITEMS_ALT_TWO,
    "proper_vn": PROPER_VN_ALL_ITEMS,
    "governance": GOVERNANCE_ANNUAL_ITEMS,
}

ANNUAL_INFERENCE_ALIASES: dict[str, dict[str, str]] = {
    "governance": {
        "GOV_BOARD": "GOV_DIRECTORY",
    }
}


def get_task_items(
    task_type: str,
    item_configs: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Return validated item configuration for a task.

    If ``item_configs`` is provided, it is used as-is after schema validation.
    Otherwise defaults from ``ANNUAL_INFERENCE_TASK_ITEMS`` are returned.
    """
    if item_configs is None:
        if task_type not in ANNUAL_INFERENCE_TASK_ITEMS:
            raise ValueError(f"Unknown annual inference task_type: {task_type}")
        return ANNUAL_INFERENCE_TASK_ITEMS[task_type]

    normalized: list[dict[str, str]] = []
    for idx, item in enumerate(item_configs):
        code = str(item.get("code", "")).strip()
        description = str(item.get("description", "")).strip()
        if not code or not description:
            raise ValueError(
                f"Invalid item_configs[{idx}] for task {task_type}: "
                "both 'code' and 'description' are required"
            )
        merged = dict(item)
        merged["code"] = code
        merged["description"] = description
        normalized.append(merged)
    return normalized


def normalize_item_codes(task_type: str, item_codes: list[str]) -> list[str]:
    """Normalize alias codes for a task while preserving input order."""
    alias_map = ANNUAL_INFERENCE_ALIASES.get(task_type, {})
    return [alias_map.get(code, code) for code in item_codes]
