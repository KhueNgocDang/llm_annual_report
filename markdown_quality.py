"""Helpers to score converted markdown for broken Vietnamese text patterns."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re

_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+", re.UNICODE)
_NONSPACE_TOKEN_RE = re.compile(r"\S+", re.UNICODE)
_BROKEN_SPACING_RE = re.compile(
    r"\b(?:[A-Za-zÀ-ỹĐđ]\s+){3,}[A-Za-zÀ-ỹĐđ]\b",
    re.UNICODE,
)
_VIETNAMESE_DIACRITIC_RE = re.compile(r"[À-ỹ]", re.UNICODE)
_LETTER_RE = re.compile(r"[A-Za-zÀ-ỹĐđ]", re.UNICODE)
_GARBLED_TOKEN_CHARS = set("0123456789,./%*+#&@\\()[]{}|^~+=")
_MARKDOWN_ARTIFACT_RE = re.compile(
    r"(?:!\[\]\(|\]\(#page-|_page_\d+_picture_\d+\.|<span|</span>|id=\"page-|^#page-)",
    re.IGNORECASE,
)
_LATEX_ARTIFACT_RE = re.compile(r"[\\{}$^]|\\[A-Za-z]+")
_FILE_EXTENSION_RE = re.compile(
    r"\.(?:jpeg|jpg|png|gif|webp|svg|pdf)$",
    re.IGNORECASE,
)
_DOMAIN_RE = re.compile(
    r"^[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+$",
    re.IGNORECASE,
)
_ABBREVIATED_PLACE_RE = re.compile(
    r"^\(?[A-Za-zÀ-ỹĐđ]{1,4}\.[A-ZÀ-ỸĐ][A-Za-zÀ-ỹĐđ-]+$",
    re.UNICODE,
)

_SINGLE_CHAR_RATIO_WEIGHT = 0.45
_BROKEN_SPACING_WEIGHT = 0.35
_AVERAGE_TOKEN_LENGTH_WEIGHT = 0.20
_GARBLED_VIETNAMESE_WEIGHT = 0.35
_SUSPICIOUS_SCORE_THRESHOLD = 0.45
_FAIL_SUSPICIOUS_SCORE_THRESHOLD = 0.6
_SUSPICIOUS_AVERAGE_TOKEN_LENGTH = 2.2
_SUSPICIOUS_GARBLED_TOKEN_RATIO = 0.015
_SUSPICIOUS_BROKEN_SPACING_COUNT = 3
_FAIL_BROKEN_SPACING_COUNT = 8
_FAIL_AFFECTED_REGION_COUNT = 2
_FAIL_AFFECTED_LINE_RATIO = 0.01
_REGION_GAP_LINES = 5
_MIN_BROKEN_SPACING_CHARS = 25


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


@dataclass(slots=True)
class MarkdownQualityScore:
    token_count: int
    single_char_token_count: int
    single_char_token_ratio: float
    broken_spacing_pattern_count: int
    average_token_length: float
    isolated_diacritic_token_count: int
    garbled_vietnamese_token_count: int
    garbled_vietnamese_token_ratio: float
    affected_line_count: int
    affected_line_ratio: float
    affected_region_count: int
    quality_status: str
    quality_evidence: str
    suspicious_score: float
    suspicious: bool
    suspicious_reason: str

    def to_dict(self) -> dict[str, int | float | bool | str]:
        return asdict(self)


def tokenize_vietnamese_text(text: str) -> list[str]:
    """Return Unicode-aware word-like tokens for Vietnamese text."""
    return _TOKEN_RE.findall(text)


def _looks_like_garbled_vietnamese_token(token: str) -> bool:
    raw_token = token
    if _MARKDOWN_ARTIFACT_RE.search(raw_token):
        return False
    if _LATEX_ARTIFACT_RE.search(raw_token):
        return False
    if "**" in raw_token:
        return False
    if "@" in raw_token:
        return False
    if "<" in raw_token or ">" in raw_token:
        return False

    token = token.strip("'\"`*_~“”‘’<>[]{}().,;:!?")
    if len(token) < 3:
        return False
    if "http://" in token.lower() or "https://" in token.lower():
        return False
    if _DOMAIN_RE.fullmatch(token):
        return False
    if _FILE_EXTENSION_RE.search(token):
        return False
    if _ABBREVIATED_PLACE_RE.fullmatch(token):
        return False
    if "_" in token:
        return False
    if "/" in token:
        slash_parts = [part.strip("()") for part in token.split("/") if part]
        if slash_parts and all(
            re.fullmatch(r"[A-Za-zÀ-ỹĐđ0-9μ.-]+", part)
            for part in slash_parts
        ):
            return False

    suspicious_chars = {ch for ch in token if ch in _GARBLED_TOKEN_CHARS}
    if suspicious_chars == {"/"}:
        return False

    slash_parts = [part for part in re.split(r"[/.]", token) if part]
    if any(
        part.isupper() and any(ch.isalpha() for ch in part)
        for part in slash_parts
    ):
        return False

    letter_count = sum(1 for ch in token if _LETTER_RE.fullmatch(ch))
    if letter_count < 2:
        return False

    suspicious_indexes = [
        index for index, ch in enumerate(token) if ch in _GARBLED_TOKEN_CHARS
    ]
    if not suspicious_indexes:
        return False

    for index in suspicious_indexes:
        if _LETTER_RE.search(token[:index]) and _LETTER_RE.search(
            token[index + 1 :]
        ):
            return True

    return False


def count_garbled_vietnamese_tokens(text: str) -> int:
    """Count whitespace-delimited tokens that look like garbled OCR Vietnamese."""
    return sum(
        1
        for token in _NONSPACE_TOKEN_RE.findall(text)
        if _looks_like_garbled_vietnamese_token(token)
    )


def _find_broken_spacing_matches(text: str) -> list[re.Match[str]]:
    return [
        match
        for match in _BROKEN_SPACING_RE.finditer(text)
        if len(match.group(0)) >= _MIN_BROKEN_SPACING_CHARS
    ]


def _find_garbled_vietnamese_line_examples(
    text: str,
    max_examples: int = 5,
) -> list[str]:
    examples: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        tokens = [
            token
            for token in _NONSPACE_TOKEN_RE.findall(line)
            if _looks_like_garbled_vietnamese_token(token)
        ]
        if not tokens:
            continue
        unique_tokens = list(dict.fromkeys(tokens))
        snippet = line.strip()
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        examples.append(
            f"L{line_number}: {', '.join(unique_tokens[:5])} | {snippet}"
        )
        if len(examples) >= max_examples:
            break
    return examples


def _find_garbled_vietnamese_line_numbers(text: str) -> list[int]:
    line_numbers: list[int] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if any(
            _looks_like_garbled_vietnamese_token(token)
            for token in _NONSPACE_TOKEN_RE.findall(line)
        ):
            line_numbers.append(line_number)
    return line_numbers


def _find_broken_spacing_line_examples(
    text: str,
    max_examples: int = 5,
) -> list[str]:
    examples: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        matches = _find_broken_spacing_matches(line)
        match = matches[0] if matches else None
        if not match:
            continue
        snippet = line.strip()
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        examples.append(f"L{line_number}: {match.group(0)} | {snippet}")
        if len(examples) >= max_examples:
            break
    return examples


def _find_broken_spacing_line_numbers(text: str) -> list[int]:
    return [
        line_number
        for line_number, line in enumerate(text.splitlines(), start=1)
        if _find_broken_spacing_matches(line)
    ]


def _count_affected_regions(line_numbers: list[int]) -> int:
    if not line_numbers:
        return 0

    regions = 1
    previous_line = line_numbers[0]
    for current_line in line_numbers[1:]:
        if current_line - previous_line > _REGION_GAP_LINES:
            regions += 1
        previous_line = current_line
    return regions


def summarize_quality_evidence(
    text: str,
    max_examples: int = 5,
    *,
    include_broken_spacing: bool = True,
    include_garbled: bool = True,
) -> str:
    """Summarize suspicious lines so audits can show where problems occurred."""
    evidence_parts: list[str] = []
    broken_spacing = _find_broken_spacing_line_examples(text, max_examples)
    if include_broken_spacing and broken_spacing:
        evidence_parts.append("broken_spacing=" + " ; ".join(broken_spacing))
    garbled_lines = _find_garbled_vietnamese_line_examples(text, max_examples)
    if include_garbled and garbled_lines:
        evidence_parts.append("garbled=" + " ; ".join(garbled_lines))
    return " || ".join(evidence_parts)


def score_markdown_quality(
    text: str,
    suspicious_score_threshold: float = _SUSPICIOUS_SCORE_THRESHOLD,
) -> MarkdownQualityScore:
    """Score a markdown document for symptoms of broken Vietnamese OCR output.

    The score combines three primary signals:
    - High single-character token ratio
    - Repeated broken spacing patterns such as "V i e t N a m"
    - Unusually short average token length
    """
    tokens = tokenize_vietnamese_text(text)
    token_count = len(tokens)
    if token_count == 0:
        return MarkdownQualityScore(
            token_count=0,
            single_char_token_count=0,
            single_char_token_ratio=0.0,
            broken_spacing_pattern_count=0,
            average_token_length=0.0,
            isolated_diacritic_token_count=0,
            garbled_vietnamese_token_count=0,
            garbled_vietnamese_token_ratio=0.0,
            affected_line_count=0,
            affected_line_ratio=0.0,
            affected_region_count=0,
            quality_status="pass",
            quality_evidence="",
            suspicious_score=0.0,
            suspicious=False,
            suspicious_reason="",
        )

    single_char_tokens = [token for token in tokens if len(token) == 1]
    single_char_token_count = len(single_char_tokens)
    single_char_token_ratio = single_char_token_count / token_count
    broken_spacing_pattern_count = len(_find_broken_spacing_matches(text))
    average_token_length = sum(len(token) for token in tokens) / token_count
    isolated_diacritic_token_count = sum(
        1
        for token in single_char_tokens
        if _VIETNAMESE_DIACRITIC_RE.fullmatch(token)
    )
    garbled_vietnamese_token_count = count_garbled_vietnamese_tokens(text)
    garbled_vietnamese_token_ratio = (
        garbled_vietnamese_token_count / token_count
    )
    broken_spacing_lines = _find_broken_spacing_line_numbers(text)
    garbled_lines = _find_garbled_vietnamese_line_numbers(text)
    affected_lines: list[int] = []
    if broken_spacing_pattern_count >= _SUSPICIOUS_BROKEN_SPACING_COUNT:
        affected_lines.extend(broken_spacing_lines)
    if garbled_vietnamese_token_ratio >= _SUSPICIOUS_GARBLED_TOKEN_RATIO:
        affected_lines.extend(garbled_lines)
    affected_lines = sorted(set(affected_lines))
    total_nonempty_lines = max(
        1, sum(1 for line in text.splitlines() if line.strip())
    )
    affected_line_count = len(affected_lines)
    affected_line_ratio = affected_line_count / total_nonempty_lines
    affected_region_count = _count_affected_regions(affected_lines)
    quality_evidence = summarize_quality_evidence(
        text,
        include_broken_spacing=(
            broken_spacing_pattern_count >= _SUSPICIOUS_BROKEN_SPACING_COUNT
        ),
        include_garbled=(
            garbled_vietnamese_token_ratio >= _SUSPICIOUS_GARBLED_TOKEN_RATIO
        ),
    )

    ratio_signal = _clamp(single_char_token_ratio / 0.18)
    spacing_signal = _clamp(broken_spacing_pattern_count / 3.0)
    average_length_signal = _clamp(
        (_SUSPICIOUS_AVERAGE_TOKEN_LENGTH - average_token_length) / 1.2
    )
    garbled_signal = _clamp(
        garbled_vietnamese_token_ratio / _SUSPICIOUS_GARBLED_TOKEN_RATIO
    )
    suspicious_score = (
        ratio_signal * _SINGLE_CHAR_RATIO_WEIGHT
        + spacing_signal * _BROKEN_SPACING_WEIGHT
        + average_length_signal * _AVERAGE_TOKEN_LENGTH_WEIGHT
        + garbled_signal * _GARBLED_VIETNAMESE_WEIGHT
    )
    reasons: list[str] = []
    if single_char_token_ratio >= 0.18:
        reasons.append("high_single_char_ratio")
    if broken_spacing_pattern_count >= _SUSPICIOUS_BROKEN_SPACING_COUNT:
        reasons.append("broken_spacing_patterns")
    if average_token_length < _SUSPICIOUS_AVERAGE_TOKEN_LENGTH:
        reasons.append("short_average_token_length")
    if garbled_vietnamese_token_ratio >= _SUSPICIOUS_GARBLED_TOKEN_RATIO:
        reasons.append("garbled_vietnamese_tokens")

    quality_status = "pass"
    if (
        single_char_token_ratio >= 0.18
        or average_token_length < _SUSPICIOUS_AVERAGE_TOKEN_LENGTH
        or garbled_vietnamese_token_ratio >= _SUSPICIOUS_GARBLED_TOKEN_RATIO
        or suspicious_score >= _FAIL_SUSPICIOUS_SCORE_THRESHOLD
        or (
            broken_spacing_pattern_count >= _SUSPICIOUS_BROKEN_SPACING_COUNT
            and (
                affected_region_count >= _FAIL_AFFECTED_REGION_COUNT
                or affected_line_ratio >= _FAIL_AFFECTED_LINE_RATIO
                or broken_spacing_pattern_count >= _FAIL_BROKEN_SPACING_COUNT
            )
        )
    ):
        quality_status = "fail"
    elif (
        suspicious_score >= suspicious_score_threshold
        or broken_spacing_pattern_count >= _SUSPICIOUS_BROKEN_SPACING_COUNT
    ):
        quality_status = "warning"

    return MarkdownQualityScore(
        token_count=token_count,
        single_char_token_count=single_char_token_count,
        single_char_token_ratio=single_char_token_ratio,
        broken_spacing_pattern_count=broken_spacing_pattern_count,
        average_token_length=average_token_length,
        isolated_diacritic_token_count=isolated_diacritic_token_count,
        garbled_vietnamese_token_count=garbled_vietnamese_token_count,
        garbled_vietnamese_token_ratio=garbled_vietnamese_token_ratio,
        affected_line_count=affected_line_count,
        affected_line_ratio=affected_line_ratio,
        affected_region_count=affected_region_count,
        quality_status=quality_status,
        quality_evidence=quality_evidence,
        suspicious_score=suspicious_score,
        suspicious=quality_status != "pass",
        suspicious_reason=",".join(reasons),
    )


def score_markdown_file(
    path: str | Path,
    suspicious_score_threshold: float = _SUSPICIOUS_SCORE_THRESHOLD,
) -> MarkdownQualityScore:
    """Read and score a markdown file from disk."""
    content = Path(path).read_text(encoding="utf-8")
    return score_markdown_quality(
        content,
        suspicious_score_threshold=suspicious_score_threshold,
    )
