"""Chunk metadata extraction helpers for financial-statement RAG retrieval."""

from __future__ import annotations

import json
import re
from typing import Any


_SECTION_TERMS: dict[str, tuple[str, ...]] = {
    "leadership_overview": (
        "thong tin chung",
        "thông tin chung",
        "gioi thieu",
        "giới thiệu",
    ),
    "board_of_directors": (
        "hoi dong quan tri",
        "hội đồng quản trị",
        "hdqt",
        "hđqt",
    ),
    "executive_board": (
        "ban dieu hanh",
        "ban điều hành",
        "ban tong giam doc",
        "ban tổng giám đốc",
    ),
    "supervisory_board": (
        "ban kiem soat",
        "ban kiểm soát",
        "bks",
    ),
    "audit_report": (
        "bao cao kiem toan",
        "báo cáo kiểm toán",
        "kiem toan doc lap",
        "kiểm toán độc lập",
        "cong ty kiem toan",
        "công ty kiểm toán",
        "don vi kiem toan",
        "đơn vị kiểm toán",
    ),
    "shareholders": (
        "co dong",
        "cổ đông",
        "ty le so huu",
        "tỷ lệ sở hữu",
        "co cau co dong",
        "cơ cấu cổ đông",
    ),
}

LLM_METADATA_SYSTEM_PROMPT = """You are a metadata tagging assistant for Vietnamese financial-statement chunks used by a RAG pipeline.

Return ONLY valid JSON object with this exact schema:
{
    "section_tags": ["leadership_overview"|"board_of_directors"|"executive_board"|"supervisory_board"|"audit_report"|"shareholders"],
    "role_tags": ["gov_directory"|"gov_executive"|"gov_supervisory"|"gov_audit"|"gov_shareholders"],
    "signals": {
        "has_person_title": boolean,
        "has_member_role": boolean,
        "has_table_like": boolean
    },
    "summary": string,
    "confidence": number
}

Rules:
- Keep only tags supported by the schema.
- If uncertain, return fewer tags, not more.
- confidence in [0,1].
- summary must be short (< 160 chars) and factual.
"""

LLM_ALLOWED_SECTION_TAGS: set[str] = {
        "leadership_overview",
        "board_of_directors",
        "executive_board",
        "supervisory_board",
        "audit_report",
        "shareholders",
}

LLM_ALLOWED_ROLE_TAGS: set[str] = {
        "gov_directory",
        "gov_executive",
        "gov_supervisory",
        "gov_audit",
        "gov_shareholders",
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _collect_heading_hints(chunk_text: str, limit: int = 4) -> list[str]:
    hints: list[str] = []
    for raw in str(chunk_text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#") or re.match(r"^\d+(?:\.\d+)*\s+", line):
            hints.append(line[:140])
        if len(hints) >= limit:
            break
    return hints


def build_chunk_metadata(chunk_text: str) -> dict[str, Any]:
    """Generate compact retrieval metadata for one chunk text."""
    normalized = _normalize_text(chunk_text)

    section_tags = [
        tag for tag, terms in _SECTION_TERMS.items() if _contains_any(normalized, terms)
    ]

    role_tags: list[str] = []
    if "board_of_directors" in section_tags:
        role_tags.append("gov_directory")
    if "executive_board" in section_tags:
        role_tags.append("gov_executive")
    if "supervisory_board" in section_tags:
        role_tags.append("gov_supervisory")
    if "audit_report" in section_tags:
        role_tags.append("gov_audit")
    if "shareholders" in section_tags:
        role_tags.append("gov_shareholders")

    has_person_title = bool(re.search(r"\b(ong|ông|ba|bà)\b", normalized))
    has_member_role = bool(
        re.search(
            r"\b(thanh vien|thành viên|chu tich|chủ tịch|tong giam doc|tổng giám đốc|truong ban|trưởng ban)\b",
            normalized,
        )
    )
    has_table_like = "|" in str(chunk_text or "")

    metadata = {
        "schema_version": 1,
        "section_tags": section_tags,
        "role_tags": role_tags,
        "heading_hints": _collect_heading_hints(chunk_text),
        "signals": {
            "has_person_title": has_person_title,
            "has_member_role": has_member_role,
            "has_table_like": has_table_like,
        },
    }
    return metadata


def build_chunk_metadata_json(chunk_text: str) -> str:
    return json.dumps(build_chunk_metadata(chunk_text), ensure_ascii=False)


def parse_chunk_metadata(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    txt = str(raw).strip()
    if not txt:
        return {}
    try:
        out = json.loads(txt)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def normalize_llm_metadata(raw: Any) -> dict[str, Any]:
    """Normalize LLM JSON metadata to pipeline-safe shape."""
    meta = parse_chunk_metadata(raw)

    section_tags_raw = meta.get("section_tags", [])
    role_tags_raw = meta.get("role_tags", [])
    signals_raw = meta.get("signals", {})

    section_tags: list[str] = []
    for tag in section_tags_raw if isinstance(section_tags_raw, list) else []:
        t = str(tag).strip()
        if t in LLM_ALLOWED_SECTION_TAGS and t not in section_tags:
            section_tags.append(t)

    role_tags: list[str] = []
    for tag in role_tags_raw if isinstance(role_tags_raw, list) else []:
        t = str(tag).strip()
        if t in LLM_ALLOWED_ROLE_TAGS and t not in role_tags:
            role_tags.append(t)

    signals = {
        "has_person_title": bool((signals_raw or {}).get("has_person_title", False)),
        "has_member_role": bool((signals_raw or {}).get("has_member_role", False)),
        "has_table_like": bool((signals_raw or {}).get("has_table_like", False)),
    }

    summary = str(meta.get("summary") or "").strip()[:200]
    try:
        confidence = float(meta.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "schema_version": 2,
        "section_tags": section_tags,
        "role_tags": role_tags,
        "signals": signals,
        "summary": summary,
        "confidence": confidence,
        "metadata_source": "llm",
    }


def merge_metadata(
    base_meta: dict[str, Any],
    llm_meta: dict[str, Any],
) -> dict[str, Any]:
    """Merge deterministic and LLM metadata, favoring union for tags and OR for signals."""
    base_sections = [str(s) for s in base_meta.get("section_tags", [])]
    llm_sections = [str(s) for s in llm_meta.get("section_tags", [])]
    section_tags = []
    for tag in [*base_sections, *llm_sections]:
        if tag and tag not in section_tags:
            section_tags.append(tag)

    base_roles = [str(s) for s in base_meta.get("role_tags", [])]
    llm_roles = [str(s) for s in llm_meta.get("role_tags", [])]
    role_tags = []
    for tag in [*base_roles, *llm_roles]:
        if tag and tag not in role_tags:
            role_tags.append(tag)

    base_signals = base_meta.get("signals", {}) if isinstance(base_meta, dict) else {}
    llm_signals = llm_meta.get("signals", {}) if isinstance(llm_meta, dict) else {}
    signals = {
        "has_person_title": bool(base_signals.get("has_person_title", False) or llm_signals.get("has_person_title", False)),
        "has_member_role": bool(base_signals.get("has_member_role", False) or llm_signals.get("has_member_role", False)),
        "has_table_like": bool(base_signals.get("has_table_like", False) or llm_signals.get("has_table_like", False)),
    }

    merged = {
        "schema_version": 2,
        "section_tags": section_tags,
        "role_tags": role_tags,
        "heading_hints": base_meta.get("heading_hints", []),
        "signals": signals,
        "summary": str(llm_meta.get("summary") or "")[:200],
        "confidence": float(llm_meta.get("confidence", 0.0)),
        "metadata_source": "hybrid",
    }
    return merged
