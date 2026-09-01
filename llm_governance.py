"""
Governance extraction module — RAG-based extraction of shareholders,
board of directors, and audit information from annual reports.

Extracts structured data that can be used to compute:
  - Audit firm
  - Proportion of ownership held by foreign shareholders
  - Proportion of ownership held by state shareholders
  - Number of shares owned by institutions
  - Number of outstanding shares
  - Number of woman board members
  - Number of foreign board members
  - Number of independent board members
  - Chair of the board cum director (CEO duality)
  - Total members of the board of directors and commissioners
  - Existence of an internal audit committee

Results are stored in ``governance_results``; batch progress is tracked
in ``governance_jobs``.
"""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata

import duckdb

from annual_inference_config import get_task_items, normalize_item_codes
from config import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    INFERENCE_MODEL,
    INFERENCE_RETRIEVAL_ALPHA,
    INFERENCE_RETRIEVAL_BETA,
    INFERENCE_RETRIEVAL_CANDIDATE_MULTIPLIER,
    INFERENCE_RETRIEVAL_GAMMA,
    INFERENCE_TEMPERATURE,
    INFERENCE_TOP_K,
)
from database import ensure_vss_loaded, get_connection
from embedder import _get_client, get_embeddings
from financial_statement_chunk_metadata import parse_chunk_metadata
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
GOV_RESULTS_TABLE = "governance_results"
GOV_RESULTS_WRITE_TABLE = "governance_results_hyde2"
GOVERNANCE_ITEMS = get_task_items("governance")

# Governance items that should retrieve context from financial statements.
_FS_SOURCE_GOV_ITEMS: set[str] = {
    "GOV_DIRECTORY",
    "GOV_EXECUTIVE",
    "GOV_SUPERVISORY",
    "GOV_AUDIT",
}

_FS_GOV_ITEM_SUFFIX = "_FS"
_FS_GOV_ITEM_CODES: dict[str, str] = {
    "GOV_DIRECTORY": "GOV_DIRECTORY_FS",
    "GOV_EXECUTIVE": "GOV_EXECUTIVE_FS",
    "GOV_SUPERVISORY": "GOV_SUPERVISORY_FS",
}


def _get_governance_result_item_code(item_code: str, source_table: str) -> str:
    if source_table == "financial_statement_document_embeddings":
        return _FS_GOV_ITEM_CODES.get(item_code, item_code)
    return item_code


def _governance_item_code_for_source(item_code: str, source_table: str) -> str:
    return _get_governance_result_item_code(item_code, source_table)

_LEADERSHIP_ROSTER_ITEMS: set[str] = {
    "GOV_DIRECTORY",
    "GOV_EXECUTIVE",
    "GOV_SUPERVISORY",
}

_LEADERSHIP_SECTION_ANCHORS: list[str] = [
    "thông tin chung",
    "thong tin chung",
    "hội đồng quản trị",
    "hoi dong quan tri",
    "hđqt",
    "hdqt",
    "ban tổng giám đốc",
    "ban tong giam doc",
    "ban điều hành",
    "ban dieu hanh",
    "ban kiểm soát",
    "ban kiem soat",
    "ban kièm soát",
    "thành viên hội đồng quản trị, ban kiểm soát và ban tổng giám đốc",
    "thanh vien hoi dong quan tri, ban kiem soat va ban tong giam doc",
]

_EMBED_METADATA_COLUMN_CACHE: dict[str, bool] = {}

_GOV_METADATA_TAGS: dict[str, set[str]] = {
    "GOV_DIRECTORY": {"board_of_directors", "leadership_overview"},
    "GOV_EXECUTIVE": {"executive_board", "leadership_overview"},
    "GOV_SUPERVISORY": {"supervisory_board", "leadership_overview"},
    "GOV_AUDIT": {"audit_report"},
    "GOV_SHAREHOLDERS": {"shareholders"},
}

_GOV_CONFLICT_TAGS: dict[str, set[str]] = {
    "GOV_DIRECTORY": {"executive_board", "supervisory_board"},
    "GOV_EXECUTIVE": {"board_of_directors", "supervisory_board"},
    "GOV_SUPERVISORY": {"board_of_directors", "executive_board"},
    "GOV_AUDIT": {"board_of_directors", "executive_board", "supervisory_board"},
}


def _embedding_table_has_metadata_column(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
) -> bool:
    cached = _EMBED_METADATA_COLUMN_CACHE.get(table_name)
    if cached is not None:
        return cached

    row = con.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = ?
          AND column_name = 'chunk_metadata_json'
        LIMIT 1
        """,
        [table_name],
    ).fetchone()
    has_col = row is not None
    _EMBED_METADATA_COLUMN_CACHE[table_name] = has_col
    return has_col


def _embedding_table_for_item(item_code: str) -> str:
    if item_code in _FS_SOURCE_GOV_ITEMS:
        return "financial_statement_document_embeddings"
    return "document_embeddings"


def _report_pairs_for_tables(
    con: duckdb.DuckDBPyConnection,
    tables: set[str],
) -> list[tuple[str, int]]:
    """Return available ticker/year pairs for required embedding table(s)."""
    if not tables:
        return []

    if tables == {"document_embeddings"}:
        rows = con.execute(
            "SELECT DISTINCT ticker, year FROM document_embeddings ORDER BY ticker, year"
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    if tables == {"financial_statement_document_embeddings"}:
        rows = con.execute(
            "SELECT DISTINCT ticker, year FROM financial_statement_document_embeddings ORDER BY ticker, year"
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    rows = con.execute(
        """
        SELECT d.ticker, d.year
        FROM (
            SELECT DISTINCT ticker, year FROM document_embeddings
        ) d
        INNER JOIN (
            SELECT DISTINCT ticker, year
            FROM financial_statement_document_embeddings
        ) f
          ON f.ticker = d.ticker AND f.year = d.year
        ORDER BY d.ticker, d.year
        """
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def _clean_chunk_text_for_prompt(text: str) -> str:
    """Normalize noisy OCR/HTML artifacts before sending content to the LLM."""
    # Many markdown chunks contain inline HTML line-break tags from OCR tables.
    return re.sub(r"(?i)<br\s*/?>", " ", text)


def _is_token_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "requested" in msg
        and "token" in msg
        and ("max" in msg or "maximum" in msg)
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_governance_items() -> list[dict[str, str]]:
    """Return all governance extraction items."""
    return get_task_items("governance")


def get_governance_groups() -> dict[str, list[dict]]:
    """Return governance items grouped by their group name."""
    governance_items = get_task_items("governance")
    groups: dict[str, list[dict]] = {}
    for item in governance_items:
        groups.setdefault(item["group"], []).append(item)
    return groups


# ---------------------------------------------------------------------------
# Category embedding cache
# ---------------------------------------------------------------------------

_gov_category_embeddings: dict[str, list[float]] | None = None


def _get_gov_category_embeddings(
    model: str | None = None,
    dimensions: int | None = None,
    item_configs: list[dict[str, str]] | None = None,
) -> dict[str, list[float]]:
    global _gov_category_embeddings
    if item_configs is None and _gov_category_embeddings is not None:
        return _gov_category_embeddings

    model = model or EMBEDDING_MODEL
    dimensions = dimensions or EMBEDDING_DIMENSIONS
    governance_items = get_task_items("governance", item_configs)

    # Criteria are static (not ticker/year-specific), so precompute HyDE upfront.
    precompute_hyde2_embeddings(
        [item["description"] for item in governance_items],
        embedding_model=model,
        dimensions=dimensions,
    )

    embeddings: dict[str, list[float]] = {}
    for item in governance_items:
        code = item["code"]
        description = item["description"]
        embeddings[code] = build_hyde2_embedding(
            description,
            embedding_model=model,
            dimensions=dimensions,
        )
    if item_configs is None:
        _gov_category_embeddings = embeddings
    return embeddings


def reset_gov_category_embeddings_cache() -> None:
    global _gov_category_embeddings
    _gov_category_embeddings = None


_GOV_RETRIEVAL_PROFILES: dict[str, dict[str, list[str]]] = {
    "GOV_SHAREHOLDERS": {
        "include_terms": [
            "cổ đông",
            "co dong",
            "sở hữu",
            "so huu",
            "tỷ lệ sở hữu",
            "ty le so huu",
            "cổ phiếu",
            "co phieu",
            "cổ đông lớn",
            "co dong lon",
        ],
        "exclude_terms": ["hđqt", "hdqt", "ban kiểm soát", "ban kiem soat"],
        "section_terms": ["cơ cấu cổ đông", "co cau co dong", "thông tin cổ phiếu", "thong tin co phieu"],
    },
    "GOV_DIRECTORY": {
        "include_terms": [
            "hội đồng quản trị",
            "hoi dong quan tri",
            "hội đồng quẩn tri",
            "hoi dong quan tri",
            "hđqt",
            "hdqt",
            "thành viên hđqt",
            "thanh vien hdqt",
            "chủ tịch hđqt",
            "chu tich hdqt",
        ],
        "exclude_terms": ["ban kiểm soát", "ban kiem soat", "thành viên bks", "thanh vien bks"],
        "anchor_terms": [
            "hội đồng quản trị",
            "hoi dong quan tri",
            "hội đồng quẩn tri",
            "các thành viên hội đồng quản trị",
            "cac thanh vien hoi dong quan tri",
        ],
        "section_terms": [
            "giới thiệu hội đồng quản trị",
            "gioi thieu hoi dong quan tri",
            "hoạt động của hđqt",
            "hoat dong cua hdqt",
            "thông tin chung",
            "thong tin chung",
        ],
    },
    "GOV_EXECUTIVE": {
        "include_terms": [
            "ban điều hành",
            "ban dieu hanh",
            "ban tổng giám đốc",
            "ban tong giam doc",
            "ban tổng giám dóc",
            "giới thiệu ban điều hành",
            "gioi thieu ban dieu hanh",
            "tổng giám đốc",
            "tong giam doc",
            "tổng giám dóc",
            "ceo",
            "phó tổng giám đốc",
            "pho tong giam doc",
            "kế toán trưởng",
            "ke toan truong",
        ],
        "exclude_terms": [
            "ban kiểm soát",
            "ban kiem soat",
            "thành viên bks",
            "thanh vien bks",
            "thành viên hđqt độc lập",
            "thanh vien hdqt doc lap",
            "hđqt độc lập",
            "hdqt doc lap",
        ],
        "anchor_terms": [
            "1.9.2. giới thiệu ban điều hành",
            "1.9.2 giới thiệu ban điều hành",
            "giới thiệu ban điều hành",
            "gioi thieu ban dieu hanh",
            "ban tổng giám đốc",
            "ban tong giam doc",
            "các thành viên ban tổng giám đốc",
            "cac thanh vien ban tong giam doc",
        ],
        "section_terms": [
            "giới thiệu ban điều hành",
            "gioi thieu ban dieu hanh",
            "ban điều hành",
            "ban dieu hanh",
            "ban tổng giám đốc",
            "ban tong giam doc",
            "thông tin chung",
            "thong tin chung",
        ],
    },
    "GOV_AUDIT": {
        "include_terms": [
            "kiểm toán",
            "kiem toan",
            "báo cáo kiểm toán",
            "bao cao kiem toan",
            "ý kiến kiểm toán",
            "y kien kiem toan",
            "công ty kiểm toán",
            "cong ty kiem toan",
            "kiểm toán độc lập",
            "kiem toan doc lap",
            "đơn vị kiểm toán",
            "don vi kiem toan",
            "ernst & young",
            "ey",
        ],
        "exclude_terms": [
            "hđqt",
            "hdqt",
            "ban điều hành",
            "ban dieu hanh",
            "ban kiểm soát",
            "ban kiem soat",
            "bks",
            "thành viên bks",
            "thanh vien bks",
            "ủy ban kiểm toán",
            "uy ban kiem toan",
            "kiểm toán nội bộ",
            "kiem toan noi bo",
        ],
        "section_terms": [
            "báo cáo kiểm toán",
            "bao cao kiem toan",
            "ý kiến kiểm toán",
            "y kien kiem toan",
            "kiểm toán độc lập",
            "kiem toan doc lap",
        ],
    },
    "GOV_SUPERVISORY": {
        "include_terms": [
            "ban kiểm soát",
            "ban kiem soat",
            "ban kièm soát",
            "ban kiem soát",
            "bks",
            "trưởng bks",
            "truong bks",
            "trưởng ban",
            "truong ban",
            "thành viên bks",
            "thanh vien bks",
            "thành viên",
            "thanh vien",
        ],
        "exclude_terms": [
            "kế toán trưởng",
            "ke toan truong",
            "ban điều hành",
            "ban dieu hanh",
            "hđqt",
            "hdqt",
            "phó tổng giám đốc",
            "pho tong giam doc",
            "tổng giám đốc",
            "tong giam doc",
            "người phụ trách quản trị",
            "nguoi phu trach quan tri",
            "ủy quyền cbtt",
            "uy quyen cbtt",
        ],
        "anchor_terms": [
            "ban kiểm soát",
            "ban kiem soat",
            "ban kièm soát",
            "các thành viên ban kiểm soát",
            "cac thanh vien ban kiem soat",
        ],
        "section_terms": ["giới thiệu ban kiểm soát", "gioi thieu ban kiem soat", "hoạt động của bks", "hoat dong cua bks"],
    },
}


def _normalize_financial_statement_governance_result(
    item_code: str,
    result: dict,
    chunks: list[dict],
) -> dict:
    """Normalize governance results produced from financial-statement sources."""
    normalized = dict(result or {})
    normalized.setdefault("found", False)
    normalized.setdefault("value", {})
    normalized.setdefault("details", [])
    normalized.setdefault("reason", "")

    if not isinstance(normalized.get("details"), list):
        normalized["details"] = []
    if not isinstance(normalized.get("value"), dict):
        normalized["value"] = {}

    if item_code in _FS_GOV_ITEM_CODES.values():
        normalized["value"] = {}

        cleaned_details: list[dict] = []
        seen_names: set[str] = set()
        for detail in normalized.get("details", []):
            if not isinstance(detail, dict):
                continue
            raw_name = str(detail.get("name") or "")
            cleaned_name = _extract_person_name_from_text(raw_name)
            if not cleaned_name:
                continue
            key = cleaned_name.lower()
            if key in seen_names:
                continue
            seen_names.add(key)
            cleaned_details.append(
                {
                    "name": cleaned_name,
                    "gender": _normalize_gender_label(detail.get("gender")),
                }
            )

        if item_code == "GOV_SUPERVISORY_FS":
            fallback_details = _fallback_parse_supervisory_members_from_text(chunks)
            if not fallback_details:
                fallback_details = _fallback_parse_audit_committee_members_from_text(
                    chunks
                )

            fallback_cleaned: list[dict] = []
            fallback_seen: set[str] = set()
            for detail in fallback_details:
                if not isinstance(detail, dict):
                    continue
                cleaned_name = _extract_person_name_from_text(
                    str(detail.get("name") or "")
                )
                if not cleaned_name:
                    continue
                key = cleaned_name.lower()
                if key in fallback_seen:
                    continue
                fallback_seen.add(key)
                fallback_cleaned.append(
                    {
                        "name": cleaned_name,
                        "gender": _normalize_gender_label(detail.get("gender")),
                    }
                )

            if fallback_cleaned and len(fallback_cleaned) > len(cleaned_details):
                cleaned_details = fallback_cleaned
                reason_text = str(normalized.get("reason") or "").strip()
                fallback_note = (
                    "Used fallback roster parsing from financial statement text."
                )
                normalized["reason"] = (
                    f"{reason_text} ({fallback_note})"
                    if reason_text
                    else fallback_note
                )

        normalized["details"] = cleaned_details
        normalized["found"] = bool(cleaned_details)

    return normalized


def _normalize_gender_label(gender: object) -> str:
    gender_text = str(gender or "").strip().lower()
    if gender_text in {"female", "bà", "ba", "f", "nu"}:
        return "Bà"
    if gender_text in {"male", "ông", "ong", "m", "nam"}:
        return "Ông"
    return ""


def _extract_person_name_from_text(raw_name: str) -> str:
    text = re.sub(r"\s+", " ", str(raw_name or "")).strip(" .,:;-")
    if not text:
        return ""

    # Drop obvious OCR/table payloads early.
    if any(tok in text for tok in ["|", "<", ">", "#", "http://", "https://"]):
        return ""

    honorific_name_pattern = re.compile(
        r"(?:Ông|Bà)\s+([A-ZÀ-ỴĐ][A-Za-zÀ-ỹĐđ]+(?:\s+[A-ZÀ-ỴĐ][A-Za-zÀ-ỹĐđ]+){1,5})"
    )
    match = honorific_name_pattern.search(text)
    candidate = match.group(1).strip() if match else text
    candidate = re.sub(r"\s+", " ", candidate).strip(" .,:;-")

    tokens = [tok for tok in candidate.split(" ") if tok]
    if len(tokens) < 2:
        return ""
    if len(tokens) > 6:
        tokens = tokens[:6]
    candidate = " ".join(tokens)

    if any(ch.isdigit() for ch in candidate):
        return ""

    low = candidate.lower()
    banned_terms = [
        "công ty",
        "bao cao",
        "báo cáo",
        "hội đồng",
        "ban kiểm",
        "kiểm toán",
        "thuyết minh",
        "tài sản",
        "nguồn vốn",
        "doanh thu",
        "lợi nhuận",
    ]
    if any(term in low for term in banned_terms):
        return ""

    return candidate


def _strip_accents(text: str) -> str:
    return "".join(
        ch
        for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )


def _fallback_parse_supervisory_members_from_text(chunks: list[dict]) -> list[dict]:
    """Parse compact BKS roster text when LLM returns empty details.

    Handles OCR-joined lines like:
    'Ba Nguyen Thi A Truong ban Ba Nguyen Thi B Thanh vien ...'
    """
    if not chunks:
        return []

    merged = "\n".join(
        _clean_chunk_text_for_prompt(str(c.get("chunk_text") or ""))
        for c in chunks
    )
    text = re.sub(r"\s+", " ", merged).strip()
    if not text:
        return []

    ascii_text = _strip_accents(text).lower()
    start = ascii_text.find("ban kiem soat")
    if start < 0:
        return []

    stop_terms = [
        "uy ban kiem toan",
        "nguoi dai dien theo phap luat",
        "kiem toan vien",
        "bao cao kiem toan",
        "bang can doi",
        "thuyet minh",
    ]
    end = len(text)
    for term in stop_terms:
        idx = ascii_text.find(term, start + 1)
        if idx >= 0:
            end = min(end, idx)

    section = text[start:end][:2000]

    pattern = re.compile(
        r"(Ông|Bà)\s+([A-ZÀ-ỴĐ][A-Za-zÀ-ỹĐđ]+(?:\s+[A-ZÀ-ỴĐ][A-Za-zÀ-ỹĐđ]+){1,5})\s+(Trưởng\s+ban|Thành\s+viên)",
        flags=re.IGNORECASE,
    )
    members: list[dict] = []
    for m in pattern.finditer(section):
        honorific = str(m.group(1) or "").strip()
        raw_name = str(m.group(2) or "").strip(" .,:;-")
        role = str(m.group(3) or "").strip()
        if not raw_name:
            continue
        members.append(
            {
                "name": raw_name,
                "position": role,
                "gender": "female" if honorific.lower() == "bà" else "male",
                "is_independent": None,
                "date_of_birth": None,
                "appointment_date": None,
                "term_end": None,
                "education": None,
                "shares_owned": None,
                "notes": "Parsed from compact BKS roster line in financial statement.",
            }
        )

    deduped: list[dict] = []
    seen: set[str] = set()
    for member in members:
        key = str(member.get("name") or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(member)
    return deduped


def _fallback_parse_audit_committee_members_from_text(
    chunks: list[dict],
) -> list[dict]:
    """Parse Audit Committee roster as supervisory proxy when BKS is absent.

    Some financial statements disclose governance through "Uy ban kiem toan"
    instead of an explicit "Ban Kiem soat" section.
    """
    if not chunks:
        return []

    merged = "\n".join(
        _clean_chunk_text_for_prompt(str(c.get("chunk_text") or ""))
        for c in chunks
    )
    text = re.sub(r"\s+", " ", merged).strip()
    if not text:
        return []

    ascii_text = _strip_accents(text).lower()
    start = ascii_text.find("uy ban kiem toan")
    if start < 0:
        return []

    stop_terms = [
        "nguoi dai dien theo phap luat",
        "kiem toan vien",
        "bao cao kiem toan",
        "bang can doi",
        "thuyet minh",
    ]
    end = len(text)
    for term in stop_terms:
        idx = ascii_text.find(term, start + 1)
        if idx >= 0:
            end = min(end, idx)

    section = text[start:end][:2500]

    member_pattern = re.compile(
        r"(Ông|Bà)\s+([A-ZÀ-ỴĐ][A-Za-zÀ-ỹĐđ]+(?:\s+[A-ZÀ-ỴĐ][A-Za-zÀ-ỹĐđ]+){1,5})\s+(Chủ tịch|Thành viên)",
        flags=re.IGNORECASE,
    )
    backup_pattern = re.compile(
        r"(Ông|Bà)\s+([A-ZÀ-ỴĐ][A-Za-zÀ-ỹĐđ]+(?:\s+[A-ZÀ-ỴĐ][A-Za-zÀ-ỹĐđ]+){1,5})(?=\s+(?:Ông|Bà)\s+|$)",
        flags=re.IGNORECASE,
    )

    members: list[dict] = []
    seen: set[str] = set()

    def _clean_name(raw: str) -> str:
        name = re.sub(r"\s+", " ", raw).strip(" .,:;-")
        name = re.sub(r"\b(Chủ\s+tịch|Thành\s+viên)\b$", "", name, flags=re.IGNORECASE).strip(" .,:;-")
        # Keep only person-like token spans and avoid OCR tails.
        tokens = [tok for tok in name.split(" ") if tok]
        if len(tokens) < 2:
            return ""
        if len(tokens) > 6:
            tokens = tokens[:6]
        cleaned = " ".join(tokens)
        if any(ch.isdigit() for ch in cleaned):
            return ""
        low = cleaned.lower()
        if any(
            bad in low
            for bad in [
                "hội đồng",
                "ban kiểm soát",
                "ban tổng",
                "tổng giám đốc",
                "công ty",
                "báo cáo",
                "kiểm toán",
            ]
        ):
            return ""
        return cleaned

    for m in member_pattern.finditer(section):
        honorific = str(m.group(1) or "").strip()
        raw_name = str(m.group(2) or "")
        role = str(m.group(3) or "").strip()
        name = _clean_name(raw_name)
        if not name:
            continue

        key = name.lower()
        if key in seen:
            continue
        seen.add(key)

        members.append(
            {
                "name": name,
                "position": f"Ủy ban kiểm toán - {role}",
                "gender": "female" if honorific.lower() == "bà" else "male",
                "is_independent": None,
                "date_of_birth": None,
                "appointment_date": None,
                "term_end": None,
                "education": None,
                "shares_owned": None,
                "notes": (
                    "Parsed from Audit Committee roster in financial statement "
                    "as supervisory proxy when BKS roster is absent."
                ),
            }
        )

    if members:
        return members

    for m in backup_pattern.finditer(section):
        honorific = str(m.group(1) or "").strip()
        raw_name = str(m.group(2) or "")
        name = _clean_name(raw_name)
        if not name:
            continue

        key = name.lower()
        if key in seen:
            continue
        seen.add(key)

        members.append(
            {
                "name": name,
                "position": "Ủy ban kiểm toán - Thành viên",
                "gender": "female" if honorific.lower() == "bà" else "male",
                "is_independent": None,
                "date_of_birth": None,
                "appointment_date": None,
                "term_end": None,
                "education": None,
                "shares_owned": None,
                "notes": (
                    "Parsed from Audit Committee roster in financial statement "
                    "as supervisory proxy when BKS roster is absent."
                ),
            }
        )

    return members


def _count_hits(text: str, terms: list[str]) -> int:
    return sum(1 for term in terms if term in text)


def _semantic_score_from_distance(distance: float) -> float:
    # Convert cosine distance to a bounded relevance score (higher is better).
    return 1.0 / (1.0 + max(distance, 0.0))


def _hybrid_score_for_chunk(chunk: dict, item_code: str) -> tuple[float, dict[str, float]]:
    text = str(chunk.get("chunk_text") or "").lower()
    distance = float(chunk.get("distance") or 0.0)

    profile = _GOV_RETRIEVAL_PROFILES.get(item_code, {})
    include_terms = profile.get("include_terms", [])
    exclude_terms = profile.get("exclude_terms", [])
    section_terms = profile.get("section_terms", [])

    include_hits = _count_hits(text, include_terms)
    exclude_hits = _count_hits(text, exclude_terms)
    section_hits = _count_hits(text, section_terms)

    semantic_score = _semantic_score_from_distance(distance)

    metadata = parse_chunk_metadata(chunk.get("chunk_metadata_json"))
    section_tags = set(str(s) for s in metadata.get("section_tags", []))
    role_tags = set(str(s) for s in metadata.get("role_tags", []))
    expected_tags = _GOV_METADATA_TAGS.get(item_code, set())
    conflict_tags = _GOV_CONFLICT_TAGS.get(item_code, set())

    matched_tags = len(section_tags.intersection(expected_tags))
    role_matches = len(role_tags.intersection({
        "gov_directory" if item_code == "GOV_DIRECTORY" else "",
        "gov_executive" if item_code == "GOV_EXECUTIVE" else "",
        "gov_supervisory" if item_code == "GOV_SUPERVISORY" else "",
        "gov_audit" if item_code == "GOV_AUDIT" else "",
        "gov_shareholders" if item_code == "GOV_SHAREHOLDERS" else "",
    }))
    conflict_hits = len(section_tags.intersection(conflict_tags))

    metadata_prior = min(1.0, 0.45 * matched_tags + 0.35 * role_matches)
    metadata_penalty = min(1.0, 0.35 * conflict_hits)

    # Lexical score is normalized to [-1, 1] to keep it stable across chunk sizes.
    lexical_raw = include_hits - exclude_hits
    lexical_norm = lexical_raw / max(1, include_hits + exclude_hits)

    # Section prior is a bounded hint until structured heading metadata is available.
    section_prior = min(1.0, section_hits / 2.0)

    final_score = (
        INFERENCE_RETRIEVAL_ALPHA * semantic_score
        + INFERENCE_RETRIEVAL_BETA * lexical_norm
        + INFERENCE_RETRIEVAL_GAMMA * section_prior
        + 0.20 * metadata_prior
        - 0.15 * metadata_penalty
    )
    return final_score, {
        "semantic": semantic_score,
        "lexical": lexical_norm,
        "section_prior": section_prior,
        "metadata_prior": metadata_prior,
        "metadata_penalty": metadata_penalty,
        "include_hits": float(include_hits),
        "exclude_hits": float(exclude_hits),
    }


def _get_section_window_candidates(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    year: int,
    query_emb: list[float],
    *,
    anchor_terms: list[str] | None = None,
    section_terms: list[str],
    embeddings_table: str = "document_embeddings",
    window_before: int = 1,
    window_after: int = 24,
    max_chunks: int = 120,
) -> list[dict]:
    """Get candidate chunks around section-heading anchors for hard OCR cases."""
    if not section_terms:
        return []

    def _find_anchor_rows(terms: list[str]) -> list[tuple]:
        anchor_clauses = " OR ".join(["lower(chunk_text) LIKE ?" for _ in terms])
        anchor_params = [f"%{term.lower()}%" for term in terms]
        anchor_sql = f"""
            SELECT DISTINCT chunk_index
                        FROM {embeddings_table}
            WHERE ticker = ? AND year = ?
              AND ({anchor_clauses})
            ORDER BY chunk_index
            LIMIT 8
        """
        return con.execute(
            anchor_sql,
            [ticker.upper(), year, *anchor_params],
        ).fetchall()

    anchor_rows: list[tuple] = []
    if anchor_terms:
        anchor_rows = _find_anchor_rows(anchor_terms)
    if not anchor_rows:
        anchor_rows = _find_anchor_rows(section_terms)
    if not anchor_rows:
        return []

    ranges = [
        (max(0, int(row[0]) - window_before), int(row[0]) + window_after)
        for row in anchor_rows
    ]
    range_clauses = " OR ".join(["(chunk_index BETWEEN ? AND ?)" for _ in ranges])
    range_params: list[int] = []
    for start_idx, end_idx in ranges:
        range_params.extend([start_idx, end_idx])

    has_meta_col = _embedding_table_has_metadata_column(con, embeddings_table)
    metadata_select = "chunk_metadata_json" if has_meta_col else "NULL AS chunk_metadata_json"

    sql = f"""
        SELECT
            chunk_index,
            chunk_text,
            token_count,
            {metadata_select},
            list_cosine_distance(embedding::FLOAT[], ?::FLOAT[]) AS distance
                FROM {embeddings_table}
        WHERE ticker = ? AND year = ?
          AND ({range_clauses})
        ORDER BY chunk_index ASC
        LIMIT ?
    """
    rows = con.execute(
        sql,
        [query_emb, ticker.upper(), year, *range_params, max_chunks],
    ).fetchall()
    return [
        {
            "chunk_index": r[0],
            "chunk_text": r[1],
            "token_count": r[2],
            "chunk_metadata_json": r[3],
            "distance": r[4],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# RAG retrieval
# ---------------------------------------------------------------------------


def retrieve_chunks_for_gov_item(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    year: int,
    item_code: str,
    *,
    top_k: int | None = None,
    model: str | None = None,
    dimensions: int | None = None,
    item_configs: list[dict[str, str]] | None = None,
) -> list[dict]:
    """Retrieve top-k most relevant chunks for a governance extraction item."""
    top_k = top_k or INFERENCE_TOP_K
    dimensions = dimensions or EMBEDDING_DIMENSIONS

    cat_embeddings = _get_gov_category_embeddings(
        model=model, dimensions=dimensions, item_configs=item_configs
    )
    if item_code not in cat_embeddings:
        raise ValueError(f"Unknown governance item code: {item_code}")

    query_emb = cat_embeddings[item_code]
    embeddings_table = _embedding_table_for_item(item_code)
    has_meta_col = _embedding_table_has_metadata_column(con, embeddings_table)
    metadata_select = "chunk_metadata_json" if has_meta_col else "NULL AS chunk_metadata_json"

    # Phase 1 hybrid retrieval:
    # 1) fetch wider semantic candidates,
    # 2) rerank with lexical + section priors.
    candidate_k = max(top_k, top_k * INFERENCE_RETRIEVAL_CANDIDATE_MULTIPLIER)
    sql = f"""
        SELECT
            chunk_index,
            chunk_text,
            token_count,
            {metadata_select},
            list_cosine_distance(embedding::FLOAT[], ?::FLOAT[]) AS distance
        FROM {embeddings_table}
        WHERE ticker = ? AND year = ?
        ORDER BY distance ASC
        LIMIT ?
    """
    rows = con.execute(
        sql, [query_emb, ticker.upper(), year, candidate_k]
    ).fetchall()
    chunks = [
        {
            "chunk_index": r[0],
            "chunk_text": r[1],
            "token_count": r[2],
            "chunk_metadata_json": r[3],
            "distance": r[4],
        }
        for r in rows
    ]

    executive_section_candidates: list[dict] = []
    rescored: list[dict] = []
    for chunk in chunks:
        score, components = _hybrid_score_for_chunk(chunk, item_code)
        rescored.append({**chunk, "_hybrid_score": score, "_score_parts": components})

    if item_code in _LEADERSHIP_ROSTER_ITEMS:
        profile = _GOV_RETRIEVAL_PROFILES.get(item_code, {})
        executive_section_candidates = _get_section_window_candidates(
            con,
            ticker,
            year,
            query_emb,
            anchor_terms=profile.get("anchor_terms", []),
            section_terms=profile.get("section_terms", []),
            embeddings_table=embeddings_table,
        )

        # In financial statements, board/executive/supervisory rosters are often
        # contiguous in the same "Thông tin chung" leadership section.
        shared_candidates = _get_section_window_candidates(
            con,
            ticker,
            year,
            query_emb,
            anchor_terms=_LEADERSHIP_SECTION_ANCHORS,
            section_terms=_LEADERSHIP_SECTION_ANCHORS,
            embeddings_table=embeddings_table,
            window_before=2,
            window_after=28,
            max_chunks=140,
        )
        if shared_candidates:
            by_index: dict[int, dict] = {
                int(c["chunk_index"]): c for c in executive_section_candidates
            }
            for chunk in shared_candidates:
                idx = int(chunk["chunk_index"])
                if idx not in by_index:
                    by_index[idx] = chunk
            executive_section_candidates = [
                by_index[idx] for idx in sorted(by_index)
            ]

        if executive_section_candidates:
            by_index: dict[int, dict] = {
                int(c["chunk_index"]): c for c in rescored
            }
            for chunk in executive_section_candidates:
                score, components = _hybrid_score_for_chunk(chunk, item_code)
                # Strongly prefer chunks from the executive-intro section window.
                section_boost = 0.35 if components.get("section_prior", 0.0) > 0 else 0.15
                boosted = {**chunk, "_hybrid_score": score + section_boost, "_score_parts": components}
                idx = int(chunk["chunk_index"])
                if idx not in by_index or boosted["_hybrid_score"] > by_index[idx]["_hybrid_score"]:
                    by_index[idx] = boosted
            rescored = list(by_index.values())

    rescored.sort(key=lambda c: c["_hybrid_score"], reverse=True)

    if item_code == "GOV_EXECUTIVE":
        gated = [
            c
            for c in rescored
            if c["_score_parts"].get("include_hits", 0.0) > 0
            and c["_score_parts"].get("exclude_hits", 0.0) == 0
        ]
        if len(gated) >= min(5, top_k):
            rescored = gated

    if item_code == "GOV_AUDIT":
        # For audit tasks, prefer chunks that explicitly mention external-audit
        # signals and do not contain excluded governance-role terms.
        gated = [
            c
            for c in rescored
            if c["_score_parts"].get("include_hits", 0.0) > 0
            and c["_score_parts"].get("exclude_hits", 0.0) == 0
        ]
        if gated:
            rescored = gated

    logger.debug(
        "Hybrid retrieval reranked %s/%s/%s with alpha=%.3f beta=%.3f gamma=%.3f",
        ticker.upper(),
        year,
        item_code,
        INFERENCE_RETRIEVAL_ALPHA,
        INFERENCE_RETRIEVAL_BETA,
        INFERENCE_RETRIEVAL_GAMMA,
    )
    selected = rescored
    if item_code == "GOV_EXECUTIVE" and executive_section_candidates:
        # Preserve contiguous section context for OCR-heavy roster tables.
        section_by_idx = {
            int(c["chunk_index"]): c for c in executive_section_candidates
        }
        selected = [section_by_idx[idx] for idx in sorted(section_by_idx)]
        if len(selected) < top_k:
            seen = {int(c["chunk_index"]) for c in selected}
            selected.extend(
                c for c in rescored if int(c["chunk_index"]) not in seen
            )

    return [
        {
            "chunk_index": c["chunk_index"],
            "chunk_text": c["chunk_text"],
            "token_count": c["token_count"],
            "distance": c["distance"],
        }
        for c in selected[:top_k]
    ]


# ---------------------------------------------------------------------------
# LLM evaluation prompts
# ---------------------------------------------------------------------------

_GOV_EXTRACT_PROMPT = """You are a structured data extraction assistant specializing in Vietnamese corporate reports (annual reports and financial statements). You will be given text from a report and a specific extraction task.

Your job is to extract ALL information related to the task — list every person, entity, and data point you can find. Be exhaustive: if 7 board members are mentioned, list all 7 with all available details for each.

Rules:
- Extract every individual/entity mentioned that is relevant to the task.
- For each person, extract ALL available fields (name, position, gender, shares, dates, etc.).
- If a field is not mentioned for a person, use `null` for that field — do NOT skip the person.
- Translate Vietnamese names to their original form (keep Vietnamese diacritics).
- Infer gender from Vietnamese names when not explicitly stated (e.g., "Nguyễn Thị ..." → female, "Nguyễn Văn ..." → male).
- For financial statements, leadership rosters often appear in compact sections such as "Thông tin chung", "Hội đồng quản trị", "Ban điều hành", or "Ban kiểm soát"; treat those sections as authoritative even if the names are listed in short lines or tables.
- Combined leadership sections that mention several bodies in one heading (for example "Thành viên Hội đồng Quản trị, Ban Kiểm soát và Ban Tổng Giám đốc") should be treated as a source for all relevant rosters in that section.
- A roster may appear as a short line, table row, or multi-line list after the heading, so do not require a long paragraph to extract names and roles.
- Do not omit people just because their title is brief or the roster is split across multiple lines.
- If the text contains a list of names followed by titles or roles, extract each person individually.
- Return ONLY a valid JSON object — no additional text.

The JSON must have these properties:

- `"value"`: an object with summary/aggregate statistics computed from the extracted details.
- `"details"`: an array of ALL extracted records. Each record is an object with relevant fields. Every person/entity must be a separate entry.
- `"reason"`: brief summary of what was extracted and from which section of the report.

All JSON properties must always be present.

TEXT DATA:
```
{content}
```

EXTRACTION TASK:
```
{criteria}
```

EXPECTED OUTPUT FORMAT:
```
{output_format}
```"""

# Per-item output format hints
_OUTPUT_FORMATS: dict[str, str] = {
    "GOV_SHAREHOLDERS": """{
  "value": {
    "total_outstanding_shares": 123456789,
    "total_treasury_shares": 0,
    "foreign_ownership_pct": 5.2,
    "state_ownership_pct": 30.0,
    "institutional_shares": 50000000,
    "individual_shares": 73456789
  },
  "details": [
        {"name": "Nguyen Van A", "shares": 1000000, "ownership_pct": 2.5, "type": "individual", "notes": "Major shareholder"},
        {"name": "SCIC", "shares": 30000000, "ownership_pct": 30.0, "type": "state", "notes": "State Capital Investment Corporation"},
        {"name": "Dragon Capital", "shares": 5000000, "ownership_pct": 5.2, "type": "foreign_institution", "notes": "Foreign fund"}
  ],
  "reason": "Extracted from section 'Cơ cấu cổ đông' ..."
}""",
        "GOV_DIRECTORY": """{
  "value": {
        "total_members": 5,
    "women_count": 2,
        "men_count": 3,
        "foreign_count": 0,
        "independent_count": 1,
        "chair_name": "Nguyen Van A"
  },
  "details": [
            {"name": "Nguyen Van A", "position": "Chủ tịch HĐQT / Chairman", "gender": "male", "is_independent": false, "independent_reason": null, "date_of_birth": "1965-03-15", "appointment_date": "2020-06-15", "term_end": "2025-06-15", "education": "MBA", "shares_owned": 500000, "ownership_pct": 1.2, "notes": ""},
            {"name": "Tran Thi C", "position": "Thành viên HĐQT độc lập / Independent Member", "gender": "female", "is_independent": true, "independent_reason": "Position/title explicitly states 'Thành viên HĐQT độc lập'.", "date_of_birth": null, "appointment_date": "2021-04-20", "term_end": null, "education": null, "shares_owned": 0, "ownership_pct": 0, "notes": ""}
  ],
    "reason": "Extracted board members from section 'Hội đồng quản trị' ..."
}""",
        "GOV_EXECUTIVE": """{
    "value": {
        "total_members": 4,
        "women_count": 2,
        "men_count": 2,
        "ceo_name": "Tran Van B",
        "chief_accountant_name": "Le Thi C",
        "board_member_count": 2
    },
    "details": [
            {"name": "Tran Van B", "position": "Tổng Giám đốc / CEO", "gender": "male", "date_of_birth": null, "appointment_date": "2022-06-01", "education": null, "is_executive": true, "executive_reason": "Position/title contains executive role evidence: 'tổng giám đốc'.", "is_board_member": false, "notes": ""},
            {"name": "Le Thi C", "position": "Kế toán trưởng", "gender": "female", "date_of_birth": null, "appointment_date": "2020-01-01", "education": null, "is_executive": true, "executive_reason": "Position/title contains executive role evidence: 'kế toán trưởng'.", "is_board_member": false, "notes": ""}
    ],
    "reason": "Extracted executive management members from section 'Ban Điều hành' ..."
}""",
    "GOV_AUDIT": """{
  "value": {
    "external_audit_firm": "Công ty TNHH Deloitte Việt Nam",
    "external_audit_firm_en": "Deloitte Vietnam Co., Ltd.",
        "audit_opinion": "unqualified",
                "signing_auditor_names": ["Pham Van F"]
  },
  "details": [
        {"name": "Công ty TNHH Deloitte Việt Nam", "role": "external_auditor_firm", "organization": "Công ty TNHH Deloitte Việt Nam", "notes": "Independent external auditor"},
        {"name": "Pham Van F", "role": "signing_auditor", "organization": "Công ty TNHH Deloitte Việt Nam", "notes": "Signed the audit report"}
  ],
    "reason": "Extracted from 'Báo cáo kiểm toán' / independent auditor report sections ..."
}""",
    "GOV_SUPERVISORY": """{
  "value": {
    "total_members": 3,
    "women_count": 1,
    "men_count": 2,
    "independent_count": 2
  },
  "details": [
    {"name": "Nguyen Van G", "position": "Trưởng Ban Kiểm soát / Head", "gender": "male", "is_independent": true, "date_of_birth": null, "appointment_date": "2020-06-15", "term_end": null, "education": null, "shares_owned": 0, "notes": ""},
    {"name": "Pham Thi H", "position": "Thành viên / Member", "gender": "female", "is_independent": true, "date_of_birth": null, "appointment_date": "2020-06-15", "term_end": null, "education": null, "shares_owned": 1000, "notes": ""}
  ],
  "reason": "Extracted from 'Ban Kiểm soát' section ..."
}""",
}

_GOV_ITEM_OUTPUT_TEMPLATES: dict[str, str] = {
    item["code"]: item.get("json_template", "")
    for item in GOVERNANCE_ITEMS
    if item.get("json_template")
}

_FS_GOV_ITEM_OUTPUT_TEMPLATES: dict[str, str] = {
    "GOV_DIRECTORY_FS": """{
  "value": {},
  "details": [
    {"name": "Nguyễn Văn A", "gender": "Ông"}
  ],
  "reason": "Extracted from financial-statement leadership roster."
}""",
    "GOV_EXECUTIVE_FS": """{
  "value": {},
  "details": [
    {"name": "Nguyễn Văn A", "gender": "Ông"}
  ],
  "reason": "Extracted from financial-statement executive roster."
}""",
    "GOV_SUPERVISORY_FS": """{
  "value": {},
  "details": [
    {"name": "Nguyễn Thị B", "gender": "Bà"}
  ],
  "reason": "Extracted from financial-statement supervisory roster."
}""",
}


def evaluate_gov_item(
    chunks: list[dict],
    item_code: str,
    item_description: str,
    *,
    model: str | None = None,
) -> dict:
    """Send retrieved chunks to the LLM for governance data extraction.

    Returns dict with keys: ``found``, ``value``, ``details``, ``reason``.
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
        text = _clean_chunk_text_for_prompt(
            str(chunk.get("chunk_text") or "")
        )
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
    output_format = _FS_GOV_ITEM_OUTPUT_TEMPLATES.get(
        item_code,
        _GOV_ITEM_OUTPUT_TEMPLATES.get(
            item_code,
            _OUTPUT_FORMATS.get(
                item_code,
                '{"found": true/false, "value": ..., "details": [...], "reason": "..."}',
            ),
        ),
    )
    extra_constraint = ""
    if item_code == "GOV_DIRECTORY":
        extra_constraint = (
            "\n\nSTRICT SCOPE for this task:\n"
            "- Extract the board-of-directors roster from financial-statement leadership sections such as 'Hội đồng quản trị', 'HĐQT', or 'Thông tin chung'.\n"
            "- Mark is_independent=true ONLY when the source explicitly indicates independent board member status (e.g., 'Thanh vien HDQT doc lap' / 'Thành viên HĐQT độc lập').\n"
            "- independent_reason must cite that explicit phrase from position/title or nearby text.\n"
            "- If no explicit evidence exists, set is_independent=false and independent_reason=null.\n"
            "- Exclude executive-only roster fields from Ban Dieu hanh in this task.\n"
            "- If the section combines multiple bodies in one heading, extract the relevant board roster from that section rather than skipping it."
        )
    elif item_code == "GOV_EXECUTIVE":
        extra_constraint = (
            "\n\nSTRICT SCOPE for this task:\n"
            "- Extract the executive management roster from financial-statement sections such as 'Ban điều hành', 'Ban Điều hành', 'Tổng giám đốc', or 'Thông tin chung'.\n"
            "- Extract ONLY executive management roles (Tong Giam doc/CEO, Pho Tong Giam doc, Ke toan truong, and equivalent executive roles).\n"
            "- Mark is_executive=true ONLY when position/title explicitly contains executive role evidence (e.g., CEO/Tong Giam doc/Pho Tong Giam doc/Ke toan truong).\n"
            "- If no explicit executive evidence exists, set is_executive=false and executive_reason=null.\n"
            "- Exclude independent board-only profiles (e.g., 'Thanh vien HDQT doc lap') unless they also explicitly hold executive title.\n"
            "- Set is_board_member=true only when the person is explicitly also a board member.\n"
            "- If the section combines multiple bodies in one heading, extract the relevant executive roster from that section rather than skipping it."
        )
    elif item_code == "GOV_SUPERVISORY":
        extra_constraint = (
            "\n\nSTRICT SCOPE for this task:\n"
            "- Extract the supervisory-board roster from financial-statement sections such as 'Ban Kiểm soát', 'BKS', or 'Thông tin chung'.\n"
            "- Include ONLY members explicitly belonging to Ban Kiem soat (BKS) / Supervisory Board.\n"
            "- Exclude Board of Directors, Executive team, and accounting/secretary roles (e.g., Ke toan truong).\n"
            "- If a person is not clearly identified as BKS member, do not include them in details.\n"
            "- If the section combines multiple bodies in one heading, extract the relevant supervisory roster from that section rather than skipping it."
        )
    elif item_code == "GOV_AUDIT":
        extra_constraint = (
            "\n\nSTRICT SCOPE for this task:\n"
            "- Include ONLY independent external audit information: audit firm, audit opinion, and signing auditor(s).\n"
            "- Exclude Ban Kiem soat (BKS), Hoi dong quan tri, executive team, and internal audit committee members.\n"
            "- If an entity/person is not part of external auditor report context, do not include in details."
        )

    prompt = _GOV_EXTRACT_PROMPT.format(
        content=content,
        criteria=item_description + extra_constraint,
        output_format=output_format,
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
                prompt = _GOV_EXTRACT_PROMPT.format(
                    content=content,
                    criteria=item_description,
                    output_format=output_format,
                )
                logger.warning(
                    "Prompt too large for %s; shrinking and retrying (attempt %d/3)",
                    model,
                    attempt + 1,
                )
                continue
            if (
                attempt < 2
                and "could not parse the JSON body" in str(exc).lower()
            ):
                logger.warning(
                    "Transient API error (attempt %d/3), retrying: %s",
                    attempt + 1,
                    exc,
                )
                time.sleep(2**attempt)
            else:
                raise
    else:
        raise last_exc  # type: ignore[misc]

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "value": None,
            "details": [],
            "reason": f"Failed to parse LLM response: {raw}",
        }

    details = result.get("details", [])
    value = result.get("value")

    if item_code in _FS_GOV_ITEM_CODES.values():
        result = _normalize_financial_statement_governance_result(item_code, result, chunks)
        details = result.get("details", [])
        value = result.get("value")

    if item_code in _FS_GOV_ITEM_CODES.values():
        result = _normalize_financial_statement_governance_result(item_code, result, chunks)
        details = result.get("details", [])
        value = result.get("value")
    elif item_code == "GOV_DIRECTORY" and isinstance(details, list):
        explicit_independent_terms = [
            "thành viên hđqt độc lập",
            "thanh vien hdqt doc lap",
            "thành viên hội đồng quản trị độc lập",
            "thanh vien hoi dong quan tri doc lap",
            "tv hdqt độc lập",
            "tv hdqt doc lap",
            "independent member",
        ]
        for detail in details:
            if not isinstance(detail, dict):
                continue
            independence_evidence_text = " ".join(
                [
                    str(detail.get("position") or ""),
                    str(detail.get("role") or ""),
                    str(detail.get("notes") or ""),
                ]
            ).lower()
            has_explicit_independent_evidence = any(
                term in independence_evidence_text
                for term in explicit_independent_terms
            )

            if not has_explicit_independent_evidence:
                detail["is_independent"] = False
                detail["independent_reason"] = None
            else:
                detail["is_independent"] = True
                detail["independent_reason"] = (
                    "Position/title explicitly states 'Thành viên HĐQT độc lập'."
                )

        if isinstance(value, dict):
            value["independent_count"] = sum(
                1 for d in details if bool(d.get("is_independent"))
            )
            value["total_members"] = len(details)
            if not value.get("chair_name"):
                for d in details:
                    if "chủ tịch" in str(d.get("position") or "").lower() or "chu tich" in str(d.get("position") or "").lower():
                        value["chair_name"] = d.get("name")
                        break

    if item_code == "GOV_EXECUTIVE" and isinstance(details, list):
        executive_evidence_terms = [
            "ceo",
            "tổng giám đốc",
            "tong giam doc",
            "phó tổng giám đốc",
            "pho tong giam doc",
            "kế toán trưởng",
            "ke toan truong",
        ]
        board_terms = [
            "hđqt",
            "hdqt",
            "hội đồng quản trị",
            "hoi dong quan tri",
            "thành viên hđqt",
            "thanh vien hdqt",
        ]
        bks_terms = [
            "ban kiểm soát",
            "ban kiem soat",
            "bks",
            "thành viên bks",
            "thanh vien bks",
        ]
        filtered_details = []
        for detail in details:
            if not isinstance(detail, dict):
                continue
            name = str(detail.get("name") or "").strip()
            if not name:
                continue
            position_text = " ".join(
                [
                    str(detail.get("position") or ""),
                    str(detail.get("role") or ""),
                ]
            ).lower()
            notes_text = str(detail.get("notes") or "").lower()
            evidence_text = f"{position_text} {notes_text}".strip()
            matched_executive_term = next(
                (
                    term
                    for term in executive_evidence_terms
                    if term in evidence_text
                ),
                None,
            )
            has_bks_signal = any(term in evidence_text for term in bks_terms)
            is_board_independent_only = (
                ("độc lập" in position_text or "doc lap" in position_text)
                and matched_executive_term is None
            )

            # Drop likely supervisory records that are not executive roles.
            if (has_bks_signal and matched_executive_term is None) or is_board_independent_only:
                continue

            if matched_executive_term is None:
                continue

            detail["is_executive"] = True
            detail["executive_reason"] = (
                f"Position/title contains executive role evidence: '{matched_executive_term}'."
            )
            detail["is_board_member"] = any(
                term in evidence_text for term in board_terms
            )
            filtered_details.append(detail)

        details = filtered_details
        if isinstance(value, dict):
            value["total_members"] = len(details)
            value["women_count"] = sum(
                1
                for d in details
                if str(d.get("gender") or "").strip().lower() == "female"
            )
            value["men_count"] = sum(
                1
                for d in details
                if str(d.get("gender") or "").strip().lower() == "male"
            )
            value["board_member_count"] = sum(
                1 for d in details if bool(d.get("is_board_member"))
            )
            value["ceo_name"] = next(
                (
                    d.get("name")
                    for d in details
                    if "ceo" in (
                        f"{str(d.get('position') or '').lower()} {str(d.get('role') or '').lower()}"
                    )
                    or "tổng giám đốc" in (
                        f"{str(d.get('position') or '').lower()} {str(d.get('role') or '').lower()}"
                    )
                    or "tong giam doc" in (
                        f"{str(d.get('position') or '').lower()} {str(d.get('role') or '').lower()}"
                    )
                ),
                value.get("ceo_name"),
            )
            value["chief_accountant_name"] = next(
                (
                    d.get("name")
                    for d in details
                    if "kế toán trưởng" in (
                        f"{str(d.get('position') or '').lower()} {str(d.get('role') or '').lower()}"
                    )
                    or "ke toan truong" in (
                        f"{str(d.get('position') or '').lower()} {str(d.get('role') or '').lower()}"
                    )
                ),
                value.get("chief_accountant_name"),
            )

    if item_code == "GOV_SUPERVISORY" and isinstance(details, list):
        filtered_details = []
        for detail in details:
            if not isinstance(detail, dict):
                continue
            text = " ".join(
                [
                    str(detail.get("position") or ""),
                    str(detail.get("role") or ""),
                    str(detail.get("notes") or ""),
                ]
            ).lower()
            if any(
                term in text
                for term in [
                    "kế toán trưởng",
                    "ke toan truong",
                    "tổng giám đốc",
                    "tong giam doc",
                    "hđqt",
                    "hdqt",
                    "ban điều hành",
                    "ban dieu hanh",
                ]
            ):
                continue
            filtered_details.append(detail)

        if not filtered_details:
            fallback_details = _fallback_parse_supervisory_members_from_text(chunks)
            if fallback_details:
                filtered_details = fallback_details
                result_reason = str(result.get("reason", "")).strip()
                if result_reason:
                    result["reason"] = (
                        f"{result_reason} (Applied fallback parsing for compact BKS roster text.)"
                    )
                else:
                    result["reason"] = (
                        "Applied fallback parsing for compact BKS roster text."
                    )

        if not filtered_details:
            audit_committee_details = _fallback_parse_audit_committee_members_from_text(
                chunks
            )
            if audit_committee_details:
                filtered_details = audit_committee_details
                result_reason = str(result.get("reason", "")).strip()
                if result_reason:
                    result["reason"] = (
                        f"{result_reason} "
                        "(Used Audit Committee roster as supervisory proxy for financial statement.)"
                    )
                else:
                    result["reason"] = (
                        "Used Audit Committee roster as supervisory proxy for financial statement."
                    )

        if len(filtered_details) != len(details):
            details = filtered_details
            result_reason = str(result.get("reason", "")).strip()
            if result_reason:
                result["reason"] = (
                    f"{result_reason} (Filtered to BKS-only entries.)"
                )
            else:
                result["reason"] = "Filtered to BKS-only entries."

        if isinstance(value, dict):
            value["total_members"] = len(details)
            value["women_count"] = sum(
                1
                for d in details
                if str(d.get("gender") or "").strip().lower() == "female"
            )
            value["men_count"] = sum(
                1
                for d in details
                if str(d.get("gender") or "").strip().lower() == "male"
            )
            value["independent_count"] = sum(
                1 for d in details if bool(d.get("is_independent"))
            )

    if item_code == "GOV_AUDIT" and isinstance(details, list):
        filtered_details = []
        for detail in details:
            if not isinstance(detail, dict):
                continue
            text = " ".join(
                [
                    str(detail.get("name") or ""),
                    str(detail.get("role") or ""),
                    str(detail.get("organization") or ""),
                    str(detail.get("notes") or ""),
                ]
            ).lower()
            if any(
                term in text
                for term in [
                    "ban kiểm soát",
                    "ban kiem soat",
                    "bks",
                    "thành viên bks",
                    "thanh vien bks",
                    "ủy ban kiểm toán",
                    "uy ban kiem toan",
                    "kiểm toán nội bộ",
                    "kiem toan noi bo",
                    "hđqt",
                    "hdqt",
                    "ban điều hành",
                    "ban dieu hanh",
                ]
            ):
                continue
            filtered_details.append(detail)

        if len(filtered_details) != len(details):
            details = filtered_details
            result_reason = str(result.get("reason", "")).strip()
            if result_reason:
                result["reason"] = (
                    f"{result_reason} (Filtered to external-auditor-only entries.)"
                )
            else:
                result["reason"] = "Filtered to external-auditor-only entries."

        if isinstance(value, dict):
            signing_auditors = []
            for d in details:
                role = str(d.get("role") or "").strip().lower()
                name = str(d.get("name") or "").strip()
                if "signing" in role and name:
                    signing_auditors.append(name)
            value["signing_auditor_names"] = signing_auditors
            value.pop("has_internal_audit_committee", None)
            value.pop("internal_audit_committee_size", None)

    # Derive 'found' from whether we got any details or non-null value
    found = bool(details) or (value is not None and value != {})

    return {
        "found": found,
        "value": value,
        "details": details,
        "reason": str(result.get("reason", "")),
    }


def _build_gov_prompt(
    chunks: list[dict], item_code: str, item_description: str
) -> str:
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
        text = _clean_chunk_text_for_prompt(str(chunk.get("chunk_text") or ""))
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
    output_format = _GOV_ITEM_OUTPUT_TEMPLATES.get(
        item_code,
        _OUTPUT_FORMATS.get(
            item_code,
            '{"found": true/false, "value": ..., "details": [...], "reason": "..."}',
        ),
    )

    extra_constraint = ""
    if item_code == "GOV_DIRECTORY":
        extra_constraint = (
            "\n\nSTRICT SCOPE for this task:\n"
            "- Mark is_independent=true ONLY when the source explicitly indicates independent board member status (e.g., 'Thanh vien HDQT doc lap' / 'Thành viên HĐQT độc lập').\n"
            "- independent_reason must cite that explicit phrase from position/title or nearby text.\n"
            "- If no explicit evidence exists, set is_independent=false and independent_reason=null.\n"
            "- Exclude executive-only roster fields from Ban Dieu hanh in this task."
        )
    elif item_code == "GOV_EXECUTIVE":
        extra_constraint = (
            "\n\nSTRICT SCOPE for this task:\n"
            "- Extract ONLY executive management roster (Ban Dieu hanh): Tong Giam doc/CEO, Pho Tong Giam doc, Ke toan truong, and equivalent executive roles.\n"
            "- Mark is_executive=true ONLY when position/title explicitly contains executive role evidence (e.g., CEO/Tong Giam doc/Pho Tong Giam doc/Ke toan truong).\n"
            "- If no explicit executive evidence exists, set is_executive=false and executive_reason=null.\n"
            "- Exclude independent board-only profiles (e.g., 'Thanh vien HDQT doc lap') unless they also explicitly hold executive title.\n"
            "- Set is_board_member=true only when the person is explicitly also a board member."
        )
    elif item_code == "GOV_SUPERVISORY":
        extra_constraint = (
            "\n\nSTRICT SCOPE for this task:\n"
            "- Include ONLY members explicitly belonging to Ban Kiem soat (BKS) / Supervisory Board.\n"
            "- Exclude Board of Directors, Executive team, and accounting/secretary roles (e.g., Ke toan truong).\n"
            "- If a person is not clearly identified as BKS member, do not include them in details."
        )
    elif item_code == "GOV_AUDIT":
        extra_constraint = (
            "\n\nSTRICT SCOPE for this task:\n"
            "- Include ONLY independent external audit information: audit firm, audit opinion, and signing auditor(s).\n"
            "- Exclude Ban Kiem soat (BKS), Hoi dong quan tri, executive team, and internal audit committee members.\n"
            "- If an entity/person is not part of external auditor report context, do not include in details."
        )

    return _GOV_EXTRACT_PROMPT.format(
        content=content,
        criteria=item_description + extra_constraint,
        output_format=output_format,
    )


def _save_governance_request_inputs(
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
                "governance",
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


# ---------------------------------------------------------------------------
# Computed governance variables
# ---------------------------------------------------------------------------


def compute_governance_variables(
    results: dict[str, dict],
) -> dict[str, object]:
    """Compute derived governance variables from extraction results.

    Parameters
    ----------
    results : dict
        Mapping of item code → extraction result dict.

    Returns
    -------
    dict
        Computed variables ready for analysis/export.
    """
    variables: dict[str, object] = {}

    # --- From GOV_SHAREHOLDERS ---
    sh = results.get("GOV_SHAREHOLDERS", {})
    sh_val = sh.get("value") or {}
    sh_details = sh.get("details") or []
    if isinstance(sh_val, dict):
        variables["total_outstanding_shares"] = sh_val.get(
            "total_outstanding_shares"
        )
        variables["total_treasury_shares"] = sh_val.get(
            "total_treasury_shares"
        )
        variables["foreign_ownership_pct"] = sh_val.get(
            "foreign_ownership_pct"
        )
        variables["state_ownership_pct"] = sh_val.get("state_ownership_pct")
        variables["institutional_shares"] = sh_val.get("institutional_shares")
        variables["individual_shares"] = sh_val.get("individual_shares")
    variables["shareholder_count"] = len(sh_details) if sh_details else None

    # --- From GOV_DIRECTORY (fallback GOV_BOARD for backward compatibility) ---
    bd = results.get("GOV_DIRECTORY", results.get("GOV_BOARD", {}))
    bd_val = bd.get("value") or {}
    bd_details = bd.get("details") or []
    if isinstance(bd_val, dict):
        variables["total_board_members"] = bd_val.get("total_members")
        variables["women_board_members"] = bd_val.get("women_count")
        variables["men_board_members"] = bd_val.get("men_count")
        variables["foreign_board_members"] = bd_val.get("foreign_count")
        variables["independent_board_members"] = bd_val.get(
            "independent_count"
        )
        variables["chair_name"] = bd_val.get("chair_name")
    if bd_details and not variables.get("total_board_members"):
        variables["total_board_members"] = len(bd_details)
    if bd_details and variables.get("women_board_members") is None:
        variables["women_board_members"] = sum(
            1
            for d in bd_details
            if str(d.get("gender", "")).lower() == "female"
        )
    if bd_details and variables.get("foreign_board_members") is None:
        variables["foreign_board_members"] = sum(
            1 for d in bd_details if d.get("is_foreign")
        )
    if bd_details and variables.get("independent_board_members") is None:
        variables["independent_board_members"] = sum(
            1 for d in bd_details if d.get("is_independent")
        )

    # --- From GOV_EXECUTIVE ---
    ex = results.get("GOV_EXECUTIVE", {})
    ex_val = ex.get("value") or {}
    ex_details = ex.get("details") or []
    if isinstance(ex_val, dict):
        variables["ceo_name"] = ex_val.get("ceo_name")
        variables["chief_accountant_name"] = ex_val.get(
            "chief_accountant_name"
        )
        variables["executive_member_count"] = ex_val.get("total_members")
        variables["executive_board_overlap_count"] = ex_val.get(
            "board_member_count"
        )
    if ex_details and variables.get("executive_member_count") is None:
        variables["executive_member_count"] = len(ex_details)
    if ex_details and variables.get("executive_board_overlap_count") is None:
        variables["executive_board_overlap_count"] = sum(
            1 for d in ex_details if d.get("is_board_member")
        )

    if (
        variables.get("chair_name")
        and variables.get("ceo_name")
        and variables.get("chair_is_ceo") is None
    ):
        variables["chair_is_ceo"] = (
            str(variables.get("chair_name")).strip().lower()
            == str(variables.get("ceo_name")).strip().lower()
        )

    # --- From GOV_AUDIT ---
    au = results.get("GOV_AUDIT", {})
    au_val = au.get("value") or {}
    if isinstance(au_val, dict):
        variables["audit_firm"] = (
            au_val.get("external_audit_firm")
            or au_val.get("external_audit_firm_en")
            or au_val.get("audit_firm")
        )
        variables["audit_opinion"] = au_val.get("audit_opinion")
        variables["has_internal_audit_committee"] = au_val.get(
            "has_internal_audit_committee"
        )
        variables["internal_audit_committee_size"] = au_val.get(
            "internal_audit_committee_size"
        )

    # --- From GOV_SUPERVISORY ---
    sv = results.get("GOV_SUPERVISORY", {})
    sv_val = sv.get("value") or {}
    sv_details = sv.get("details") or []
    if isinstance(sv_val, dict):
        variables["total_supervisory_members"] = sv_val.get("total_members")
        variables["supervisory_women_count"] = sv_val.get("women_count")
        variables["supervisory_independent_count"] = sv_val.get(
            "independent_count"
        )
    if sv_details and not variables.get("total_supervisory_members"):
        variables["total_supervisory_members"] = len(sv_details)

    return variables


# ---------------------------------------------------------------------------
# Job management
# ---------------------------------------------------------------------------


def create_governance_jobs(
    con: duckdb.DuckDBPyConnection,
    *,
    tickers: list[str] | None = None,
    years: list[int] | None = None,
    replace: bool = False,
    inference_model: str | None = None,
    item_configs: list[dict[str, str]] | None = None,
) -> int:
    """Create pending governance extraction jobs for embedded reports."""
    inference_model = inference_model or INFERENCE_MODEL
    governance_items = get_task_items("governance", item_configs)
    required_tables = {
        _embedding_table_for_item(item["code"])
        for item in governance_items
    }

    embedded = _report_pairs_for_tables(con, required_tables)

    if tickers:
        upper = {t.upper() for t in tickers}
        embedded = [(t, y) for t, y in embedded if t in upper]
    if years:
        embedded = [(t, y) for t, y in embedded if y in years]

    total_items = len(governance_items)
    created = 0
    for ticker, year in embedded:
        existing = con.execute(
            "SELECT status FROM governance_jobs "
            "WHERE ticker = ? AND year = ? AND model = ?",
            [ticker, year, inference_model],
        ).fetchone()

        if existing is None:
            con.execute(
                """
                INSERT INTO governance_jobs
                    (id, ticker, year, model, status, items_total)
                VALUES (nextval('governance_jobs_id_seq'), ?, ?, ?, 'pending', ?)
                """,
                [ticker, year, inference_model, total_items],
            )
            created += 1
        elif replace and existing[0] in ("completed", "failed"):
            con.execute(
                """
                UPDATE governance_jobs
                SET status = 'pending', error_message = NULL,
                    started_at = NULL, completed_at = NULL,
                    items_done = 0,
                    batch_id = NULL,
                    batch_submitted_at = NULL,
                    batch_checked_at = NULL
                WHERE ticker = ? AND year = ? AND model = ?
                """,
                [ticker, year, inference_model],
            )
            created += 1

    return created


def get_governance_jobs(
    con: duckdb.DuckDBPyConnection,
    *,
    status: str | None = None,
    model: str | None = None,
) -> list[dict]:
    """Return governance jobs, optionally filtered."""
    query = "SELECT * FROM governance_jobs"
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
        "items_done",
        "items_total",
        "batch_id",
        "batch_submitted_at",
        "batch_checked_at",
        "started_at",
        "completed_at",
        "error_message",
        "created_at",
    ]
    return [dict(zip(columns, row)) for row in rows]


def get_governance_job_counts(
    con: duckdb.DuckDBPyConnection,
    *,
    model: str | None = None,
) -> dict[str, int]:
    query = "SELECT status, COUNT(*) FROM governance_jobs"
    params: list = []
    if model:
        query += " WHERE model = ?"
        params.append(model)
    query += " GROUP BY status"
    rows = con.execute(query, params).fetchall()
    return {r[0]: r[1] for r in rows}


def reset_failed_governance_jobs(
    con: duckdb.DuckDBPyConnection,
    *,
    model: str | None = None,
) -> int:
    query = (
        "UPDATE governance_jobs SET status = 'pending', error_message = NULL, "
        "started_at = NULL, completed_at = NULL, items_done = 0, "
        "batch_id = NULL, batch_submitted_at = NULL, batch_checked_at = NULL "
        "WHERE status = 'failed'"
    )
    params: list = []
    if model:
        query += " AND model = ?"
        params.append(model)
    con.execute(query, params)
    return 1


def sync_governance_job_states(
    con: duckdb.DuckDBPyConnection,
    *,
    model: str | None = None,
    item_configs: list[dict[str, str]] | None = None,
) -> dict[str, int]:
    """Sync governance job statuses with actual results in DB."""
    total_items = len(get_task_items("governance", item_configs))

    query = (
        "SELECT id, ticker, year, model, status, items_done "
        "FROM governance_jobs WHERE status != 'completed'"
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
            f"SELECT COUNT(*) FROM {GOV_RESULTS_WRITE_TABLE} "
            "WHERE ticker = ? AND year = ? AND model = ?",
            [ticker, year, job_model],
        ).fetchone()
        result_count = row[0] if row else 0

        if result_count >= total_items:
            con.execute(
                """
                UPDATE governance_jobs
                SET status = 'completed',
                    completed_at = get_current_timestamp(),
                    items_done = ?, items_total = ?,
                    error_message = NULL
                WHERE id = ?
                """,
                [total_items, total_items, job_id],
            )
            n_completed += 1
        elif result_count > 0 and result_count != done:
            con.execute(
                """
                UPDATE governance_jobs
                SET items_done = ?, items_total = ?
                WHERE id = ?
                """,
                [result_count, total_items, job_id],
            )
            n_updated += 1

    return {"completed": n_completed, "updated": n_updated}


def delete_governance_jobs(
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

    con.execute(f"DELETE FROM {GOV_RESULTS_WRITE_TABLE}{where}", params)
    con.execute(f"DELETE FROM governance_jobs{where}", params)


# ---------------------------------------------------------------------------
# Results helpers
# ---------------------------------------------------------------------------


def get_governance_results(
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
        SELECT id, ticker, year, item_code, found, value_json,
               details_json, reason, top_chunks, similarities,
               model, created_at
        FROM {GOV_RESULTS_TABLE}
        {where}
        ORDER BY ticker, year, item_code
        """,
        params,
    ).fetchall()

    columns = [
        "id",
        "ticker",
        "year",
        "item_code",
        "found",
        "value_json",
        "details_json",
        "reason",
        "top_chunks",
        "similarities",
        "model",
        "created_at",
    ]
    return [dict(zip(columns, row)) for row in rows]


# ---------------------------------------------------------------------------
# Core: extract governance data for a single report
# ---------------------------------------------------------------------------


def extract_governance(
    ticker: str,
    year: int,
    con: duckdb.DuckDBPyConnection | None = None,
    *,
    replace: bool = False,
    top_k: int | None = None,
    inference_model: str | None = None,
    embedding_model: str | None = None,
    dimensions: int | None = None,
    item_codes: list[str] | None = None,
    item_configs: list[dict[str, str]] | None = None,
) -> int:
    """Run governance extraction for a single annual report.

    For each governance item:
    1. Retrieve top-k chunks via vector similarity (RAG)
    2. Send to LLM for structured extraction
    3. Store result in ``governance_results``

    Returns the number of items extracted.
    """
    own_con = con is None
    if own_con:
        con = get_connection()
        ensure_vss_loaded(con)

    ticker = ticker.upper()
    top_k = top_k or INFERENCE_TOP_K
    inference_model = inference_model or INFERENCE_MODEL
    governance_items = get_task_items("governance", item_configs)

    # Determine items to extract
    if item_codes:
        normalized_item_codes = normalize_item_codes("governance", item_codes)
        valid_codes = {it["code"] for it in governance_items}
        unknown = set(normalized_item_codes) - valid_codes
        if unknown:
            raise ValueError(f"Unknown governance item codes: {unknown}")
        items_to_eval = [
            it
            for it in governance_items
            if it["code"] in normalized_item_codes
        ]
        item_codes = normalized_item_codes
    else:
        items_to_eval = governance_items

    # Ensure job row exists
    existing_job = con.execute(
        "SELECT status, batch_id FROM governance_jobs "
        "WHERE ticker = ? AND year = ? AND model = ?",
        [ticker, year, inference_model],
    ).fetchone()

    total_items = len(governance_items)

    if existing_job is None:
        con.execute(
            """
            INSERT INTO governance_jobs
                (id, ticker, year, model, status, items_total)
            VALUES (nextval('governance_jobs_id_seq'), ?, ?, ?, 'pending', ?)
            """,
            [ticker, year, inference_model, total_items],
        )

    # Mark job as running
    con.execute(
        """
        UPDATE governance_jobs
        SET status = 'running', started_at = get_current_timestamp(),
            top_k = ?, items_total = ?,
            items_done = 0, error_message = NULL
        WHERE ticker = ? AND year = ? AND model = ?
        """,
        [top_k, total_items, ticker, year, inference_model],
    )

    try:
        batch_id = str(existing_job[1]) if existing_job and existing_job[1] else None
        client = _get_client()

        if batch_id:
            batch_status = get_batch_status(client, batch_id)
            con.execute(
                """
                UPDATE governance_jobs
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
                return 0

            if batch_status in {"failed", "expired", "cancelled"}:
                msg = f"Batch {batch_id} ended with status={batch_status}"
                con.execute(
                    """
                    UPDATE governance_jobs
                    SET status = 'failed', completed_at = get_current_timestamp(),
                        error_message = ?, batch_id = NULL
                    WHERE ticker = ? AND year = ? AND model = ?
                    """,
                    [msg, ticker, year, inference_model],
                )
                raise RuntimeError(msg)

            output_map = get_batch_output_map(client, batch_id)
            by_code = {it["code"]: it for it in governance_items}
            by_code.update(
                {
                    _governance_item_code_for_source(it["code"], "financial_statement_document_embeddings"): it
                    for it in governance_items
                }
            )
            done_now = 0
            for code, raw in output_map.items():
                item = by_code.get(code)
                if item is None:
                    continue

                # Preserve similarity/top chunk metadata by repeating retrieval.
                gov_top_k = max(top_k, 10)
                source_table = _embedding_table_for_item(item["code"])
                storage_code = _governance_item_code_for_source(item["code"], source_table)
                chunks = retrieve_chunks_for_gov_item(
                    con,
                    ticker,
                    year,
                    item["code"],
                    top_k=gov_top_k,
                    model=embedding_model,
                    dimensions=dimensions,
                    item_configs=governance_items,
                )
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = {
                        "value": None,
                        "details": [],
                        "reason": f"Failed to parse LLM response: {raw}",
                    }

                # Re-apply core post-processing for batch outputs.
                if item["code"] == "GOV_SUPERVISORY":
                    details_candidate = parsed.get("details", [])
                    if not isinstance(details_candidate, list):
                        details_candidate = []
                    if not details_candidate:
                        fallback_details = _fallback_parse_supervisory_members_from_text(
                            chunks
                        )
                        if not fallback_details:
                            fallback_details = _fallback_parse_audit_committee_members_from_text(
                                chunks
                            )
                        if fallback_details:
                            parsed["details"] = fallback_details
                            parsed["value"] = {
                                "total_members": len(fallback_details),
                                "women_count": sum(
                                    1
                                    for d in fallback_details
                                    if str(d.get("gender") or "").strip().lower()
                                    == "female"
                                ),
                                "men_count": sum(
                                    1
                                    for d in fallback_details
                                    if str(d.get("gender") or "").strip().lower()
                                    == "male"
                                ),
                                "independent_count": sum(
                                    1
                                    for d in fallback_details
                                    if bool(d.get("is_independent"))
                                ),
                            }
                            result_reason = str(parsed.get("reason", "")).strip()
                            if result_reason:
                                parsed["reason"] = (
                                    f"{result_reason} "
                                    "(Applied supervisory fallback parsing from financial-statement roster text.)"
                                )
                            else:
                                parsed["reason"] = (
                                    "Applied supervisory fallback parsing from financial-statement roster text."
                                )

                if storage_code in _FS_GOV_ITEM_CODES.values():
                    parsed = _normalize_financial_statement_governance_result(
                        storage_code,
                        parsed,
                        chunks,
                    )

                details = parsed.get("details", [])
                value = parsed.get("value")
                found = bool(details) or (value is not None and value != {})

                similarities_json = json.dumps([round(c["distance"], 6) for c in chunks])
                top_chunks_json = json.dumps([c["chunk_index"] for c in chunks])
                value_json = json.dumps(value, ensure_ascii=False)
                details_json = json.dumps(details, ensure_ascii=False)

                con.execute(
                    """
                    INSERT INTO governance_results_hyde2
                        (id, ticker, year, item_code, found, value_json,
                         details_json, reason, top_chunks, similarities, model)
                    VALUES
                        (nextval('governance_results_id_seq'),
                         ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (ticker, year, item_code, model) DO UPDATE SET
                        found = EXCLUDED.found,
                        value_json = EXCLUDED.value_json,
                        details_json = EXCLUDED.details_json,
                        reason = EXCLUDED.reason,
                        top_chunks = EXCLUDED.top_chunks,
                        similarities = EXCLUDED.similarities,
                        created_at = get_current_timestamp()
                    """,
                    [
                        ticker,
                        year,
                        storage_code,
                        found,
                        value_json,
                        details_json,
                        str(parsed.get("reason", "")),
                        top_chunks_json,
                        similarities_json,
                        inference_model,
                    ],
                )
                done_now += 1

            total_results = con.execute(
                f"SELECT COUNT(*) FROM {GOV_RESULTS_WRITE_TABLE} "
                "WHERE ticker = ? AND year = ? AND model = ?",
                [ticker, year, inference_model],
            ).fetchone()
            total_done = total_results[0] if total_results else 0

            if total_done >= total_items:
                con.execute(
                    """
                    UPDATE governance_jobs
                    SET status = 'completed',
                        completed_at = get_current_timestamp(),
                        items_done = ?,
                        batch_id = NULL
                    WHERE ticker = ? AND year = ? AND model = ?
                    """,
                    [total_done, ticker, year, inference_model],
                )
            else:
                con.execute(
                    """
                    UPDATE governance_jobs
                    SET status = 'pending', items_done = ?, batch_id = NULL
                    WHERE ticker = ? AND year = ? AND model = ?
                    """,
                    [total_done, ticker, year, inference_model],
                )
            if own_con:
                con.close()
            return done_now

        if replace:
            if item_codes:
                placeholders = ",".join(["?"] * len(item_codes))
                con.execute(
                    f"DELETE FROM {GOV_RESULTS_WRITE_TABLE} "
                    "WHERE ticker = ? AND year = ? AND model = ? "
                    f"AND item_code IN ({placeholders})",
                    [ticker, year, inference_model, *item_codes],
                )
            else:
                con.execute(
                    f"DELETE FROM {GOV_RESULTS_WRITE_TABLE} "
                    "WHERE ticker = ? AND year = ? AND model = ?",
                    [ticker, year, inference_model],
                )

        done = 0
        prompts: dict[str, str] = {}
        for item in items_to_eval:
            code = item["code"]
            source_table = _embedding_table_for_item(code)
            storage_code = _governance_item_code_for_source(code, source_table)

            if not replace:
                existing = con.execute(
                    f"SELECT 1 FROM {GOV_RESULTS_WRITE_TABLE} "
                    "WHERE ticker = ? AND year = ? AND item_code = ? "
                    "AND model = ?",
                    [ticker, year, storage_code, inference_model],
                ).fetchone()
                if existing:
                    done += 1
                    continue

            row = con.execute(
                f"SELECT COUNT(*) FROM {source_table} "
                "WHERE ticker = ? AND year = ?",
                [ticker, year],
            ).fetchone()
            emb_count = row[0] if row else 0
            if emb_count == 0:
                result = {
                    "found": False,
                    "value": None,
                    "details": [],
                    "reason": (
                        "No embeddings found in "
                        f"{source_table} for {ticker}/{year}. "
                        "Embed the report first."
                    ),
                }
                similarities_json = json.dumps([])
                top_chunks_json = json.dumps([])
                value_json = json.dumps(result.get("value"), ensure_ascii=False)
                details_json = json.dumps(result.get("details", []), ensure_ascii=False)

                con.execute(
                    """
                    INSERT INTO governance_results_hyde2
                        (id, ticker, year, item_code, found, value_json,
                         details_json, reason, top_chunks, similarities, model)
                    VALUES
                        (nextval('governance_results_id_seq'),
                         ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (ticker, year, item_code, model) DO UPDATE SET
                        found = EXCLUDED.found,
                        value_json = EXCLUDED.value_json,
                        details_json = EXCLUDED.details_json,
                        reason = EXCLUDED.reason,
                        top_chunks = EXCLUDED.top_chunks,
                        similarities = EXCLUDED.similarities,
                        created_at = get_current_timestamp()
                    """,
                    [
                        ticker,
                        year,
                        storage_code,
                        result["found"],
                        value_json,
                        details_json,
                        result["reason"],
                        top_chunks_json,
                        similarities_json,
                        inference_model,
                    ],
                )
                done += 1
                continue

            # Use more chunks for governance extraction (richer context)
            gov_top_k = max(top_k, 10)
            chunks = retrieve_chunks_for_gov_item(
                con,
                ticker,
                year,
                code,
                top_k=gov_top_k,
                model=embedding_model,
                dimensions=dimensions,
                item_configs=governance_items,
            )

            if not chunks:
                result = {
                    "found": False,
                    "value": None,
                    "details": [],
                    "reason": "No relevant text chunks found in the report.",
                }
                similarities_json = json.dumps([round(c["distance"], 6) for c in chunks])
                top_chunks_json = json.dumps([c["chunk_index"] for c in chunks])
                value_json = json.dumps(result.get("value"), ensure_ascii=False)
                details_json = json.dumps(result.get("details", []), ensure_ascii=False)

                con.execute(
                    """
                    INSERT INTO governance_results_hyde2
                        (id, ticker, year, item_code, found, value_json,
                         details_json, reason, top_chunks, similarities, model)
                    VALUES
                        (nextval('governance_results_id_seq'),
                         ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (ticker, year, item_code, model) DO UPDATE SET
                        found = EXCLUDED.found,
                        value_json = EXCLUDED.value_json,
                        details_json = EXCLUDED.details_json,
                        reason = EXCLUDED.reason,
                        top_chunks = EXCLUDED.top_chunks,
                        similarities = EXCLUDED.similarities,
                        created_at = get_current_timestamp()
                    """,
                    [
                        ticker,
                        year,
                        storage_code,
                        result["found"],
                        value_json,
                        details_json,
                        result["reason"],
                        top_chunks_json,
                        similarities_json,
                        inference_model,
                    ],
                )
                done += 1
                continue

            prompts[storage_code] = _build_gov_prompt(chunks, code, item["description"])

        if prompts:
            is_reasoning = inference_model.startswith(("o1", "o3", "o4", "gpt-5"))
            new_batch_id = submit_chat_json_batch(
                client=client,
                model=inference_model,
                prompts=prompts,
                is_reasoning=is_reasoning,
                temperature=INFERENCE_TEMPERATURE,
            )
            _save_governance_request_inputs(
                con,
                ticker=ticker,
                year=year,
                model=inference_model,
                batch_id=new_batch_id,
                prompts=prompts,
            )
            con.execute(
                """
                UPDATE governance_jobs
                SET status = 'running', items_done = ?,
                    batch_id = ?,
                    batch_submitted_at = get_current_timestamp(),
                    batch_checked_at = get_current_timestamp()
                WHERE ticker = ? AND year = ? AND model = ?
                """,
                [done, new_batch_id, ticker, year, inference_model],
            )
            logger.info(
                "Submitted governance batch %s for %s/%d (%d items)",
                new_batch_id,
                ticker,
                year,
                len(prompts),
            )
            if own_con:
                con.close()
            return done + len(prompts)

        # Check completion
        total_results = con.execute(
            f"SELECT COUNT(*) FROM {GOV_RESULTS_WRITE_TABLE} "
            "WHERE ticker = ? AND year = ? AND model = ?",
            [ticker, year, inference_model],
        ).fetchone()
        total_done = total_results[0] if total_results else 0

        if total_done >= total_items:
            con.execute(
                """
                UPDATE governance_jobs
                SET status = 'completed',
                    completed_at = get_current_timestamp(),
                    items_done = ?,
                    batch_id = NULL
                WHERE ticker = ? AND year = ? AND model = ?
                """,
                [total_done, ticker, year, inference_model],
            )
        else:
            con.execute(
                """
                UPDATE governance_jobs
                SET items_done = ?,
                    status = CASE WHEN status = 'running' THEN 'pending'
                                  ELSE status END,
                    batch_id = NULL
                WHERE ticker = ? AND year = ? AND model = ?
                """,
                [total_done, ticker, year, inference_model],
            )

        logger.info(
            "Governance extraction complete for %s/%d — %d items extracted",
            ticker,
            year,
            done,
        )

        if own_con:
            con.close()
        return done

    except Exception as exc:
        con.execute(
            """
            UPDATE governance_jobs
            SET status = 'failed', completed_at = get_current_timestamp(),
                error_message = ?
            WHERE ticker = ? AND year = ? AND model = ?
            """,
            [str(exc), ticker, year, inference_model],
        )
        logger.error(
            "Governance extraction failed for %s/%d: %s", ticker, year, exc
        )
        if own_con:
            con.close()
        raise


# ---------------------------------------------------------------------------
# Batch operations
# ---------------------------------------------------------------------------


def extract_governance_all(
    *,
    replace: bool = False,
    progress_callback=None,
    top_k: int | None = None,
    inference_model: str | None = None,
    embedding_model: str | None = None,
    dimensions: int | None = None,
    item_configs: list[dict[str, str]] | None = None,
) -> dict:
    """Run governance extraction for every pending job."""
    con = get_connection()
    ensure_vss_loaded(con)

    _model = inference_model or INFERENCE_MODEL

    create_governance_jobs(
        con,
        replace=replace,
        inference_model=_model,
        item_configs=item_configs,
    )

    pending = get_governance_jobs(con, status="pending", model=_model)
    results: dict = {"evaluated": [], "skipped": [], "failed": []}
    total = len(pending)

    for i, job in enumerate(pending):
        ticker, year = job["ticker"], job["year"]
        n = 0

        try:
            n = extract_governance(
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
            if n > 0:
                results["evaluated"].append((ticker, year, n))
            else:
                results["skipped"].append((ticker, year))
        except Exception as exc:
            logger.error(
                "Governance extraction failed for %s/%d: %s",
                ticker,
                year,
                exc,
            )
            results["failed"].append((ticker, year, str(exc)))

        if progress_callback:
            progress_callback(ticker, year, n, i + 1, total)

    con.close()
    return results


def extract_governance_item_all_reports(
    item_codes: list[str],
    *,
    tickers: list[str] | None = None,
    years: list[int] | None = None,
    replace: bool = False,
    progress_callback=None,
    top_k: int | None = None,
    inference_model: str | None = None,
    embedding_model: str | None = None,
    dimensions: int | None = None,
) -> dict:
    """Run specific governance items across all embedded reports."""
    con = get_connection()
    ensure_vss_loaded(con)

    normalized_item_codes = normalize_item_codes("governance", item_codes)
    required_tables = {
        _embedding_table_for_item(code)
        for code in normalized_item_codes
    }

    reports = _report_pairs_for_tables(con, required_tables)

    if tickers:
        upper = {t.upper() for t in tickers}
        reports = [(t, y) for t, y in reports if t in upper]
    if years:
        reports = [(t, y) for t, y in reports if y in years]

    results: dict = {"evaluated": [], "skipped": [], "failed": []}
    total = len(reports)

    for i, (ticker, year) in enumerate(reports):
        n = 0
        try:
            n = extract_governance(
                ticker,
                year,
                con=con,
                replace=replace,
                top_k=top_k,
                inference_model=inference_model,
                embedding_model=embedding_model,
                dimensions=dimensions,
                item_codes=normalized_item_codes,
            )
            if n > 0:
                results["evaluated"].append((ticker, year, n))
            else:
                results["skipped"].append((ticker, year))
        except Exception as exc:
            logger.error(
                "Governance extraction failed for %s/%d: %s",
                ticker,
                year,
                exc,
            )
            results["failed"].append((ticker, year, str(exc)))

        if progress_callback:
            progress_callback(ticker, year, n, i + 1, total)

    con.close()
    return results
