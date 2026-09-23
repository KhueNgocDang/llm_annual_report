"""Phase 0 benchmark evaluator for governance RAG retrieval/extraction quality."""

from __future__ import annotations

import argparse
import json
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from config import INFERENCE_MODEL, INFERENCE_TOP_K, ensure_env_loaded
from database import ensure_vss_loaded, get_connection, init_db
from llm_governance import extract_governance, retrieve_chunks_for_gov_item


def _normalize_text(text: str) -> str:
    text = " ".join(str(text).strip().split()).lower()
    # Remove accents to make comparisons robust to OCR and output variation.
    text = "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )
    return text


def _load_cases(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("cases", []))


def _evaluate_retrieval_case(case: dict, chunks: list[dict]) -> dict:
    required_terms = [_normalize_text(t) for t in case.get("required_terms", [])]
    forbidden_terms = [_normalize_text(t) for t in case.get("forbidden_terms", [])]

    relevant = 0
    forbidden_hits = 0
    for chunk in chunks:
        text = _normalize_text(chunk.get("chunk_text") or "")
        has_required = any(term in text for term in required_terms) if required_terms else True
        has_forbidden = any(term in text for term in forbidden_terms) if forbidden_terms else False
        if has_required:
            relevant += 1
        if has_forbidden:
            forbidden_hits += 1

    k = max(1, len(chunks))
    return {
        "retrieval_precision_at_k": relevant / k,
        "retrieval_forbidden_hit_rate": forbidden_hits / k,
        "retrieval_any_relevant": relevant > 0,
    }


def _extract_names_from_details(details_json: str | None) -> list[str]:
    if not details_json:
        return []
    try:
        details = json.loads(details_json)
    except json.JSONDecodeError:
        return []
    if not isinstance(details, list):
        return []

    names: list[str] = []
    for detail in details:
        if not isinstance(detail, dict):
            continue
        name = str(detail.get("name") or "").strip()
        if name:
            names.append(name)
    return names


def _evaluate_extraction_case(case: dict, predicted_names: list[str]) -> dict:
    expected_names = [
        _normalize_text(name) for name in case.get("expected_entities", []) if str(name).strip()
    ]
    predicted_norm = [_normalize_text(name) for name in predicted_names if str(name).strip()]

    expected_set = set(expected_names)
    predicted_set = set(predicted_norm)

    expected_count = case.get("expected_count")
    exact_match = bool(expected_set) and expected_set == predicted_set

    return {
        "expected_count": expected_count,
        "predicted_count": len(predicted_set),
        "extraction_exact_entity_match": exact_match,
        "missing_entities": sorted(expected_set - predicted_set),
        "unexpected_entities": sorted(predicted_set - expected_set),
    }


def evaluate_governance_benchmark(
    benchmark_file: Path,
    *,
    top_k: int,
    model: str,
    run_extraction: bool,
) -> dict:
    ensure_env_loaded()
    con = get_connection()
    ensure_vss_loaded(con)
    init_db(con)

    cases = _load_cases(benchmark_file)
    results: list[dict] = []

    precision_values: list[float] = []
    forbidden_rates: list[float] = []
    relevant_hit_count = 0
    evaluated_retrieval_cases = 0
    extraction_eval_cases = 0
    extraction_exact_matches = 0

    for case in cases:
        ticker = str(case["ticker"]).upper()
        year = int(case["year"])
        item_code = str(case["item_code"])

        chunks = retrieve_chunks_for_gov_item(
            con,
            ticker,
            year,
            item_code,
            top_k=top_k,
        )

        retrieval_eval = _evaluate_retrieval_case(case, chunks)
        precision_values.append(retrieval_eval["retrieval_precision_at_k"])
        forbidden_rates.append(retrieval_eval["retrieval_forbidden_hit_rate"])
        evaluated_retrieval_cases += 1
        if retrieval_eval["retrieval_any_relevant"]:
            relevant_hit_count += 1

        case_result = {
            "id": case.get("id"),
            "ticker": ticker,
            "year": year,
            "item_code": item_code,
            "retrieval": retrieval_eval,
            "top_chunk_indices": [c.get("chunk_index") for c in chunks],
        }

        if run_extraction and case.get("enabled_for_extraction_eval"):
            extract_governance(
                ticker,
                year,
                con=con,
                replace=False,
                item_codes=[item_code],
                inference_model=model,
                top_k=top_k,
            )
            row = con.execute(
                "SELECT details_json FROM governance_results "
                "WHERE ticker = ? AND year = ? AND item_code = ? AND model = ?",
                [ticker, year, item_code, model],
            ).fetchone()
            predicted_names = _extract_names_from_details(row[0] if row else None)
            extraction_eval = _evaluate_extraction_case(case, predicted_names)
            case_result["extraction"] = extraction_eval
            extraction_eval_cases += 1
            if extraction_eval["extraction_exact_entity_match"]:
                extraction_exact_matches += 1

        results.append(case_result)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_file": str(benchmark_file),
        "model": model,
        "top_k": top_k,
        "run_extraction": run_extraction,
        "cases_total": len(cases),
        "retrieval_cases_evaluated": evaluated_retrieval_cases,
        "retrieval_precision_at_k_avg": (
            sum(precision_values) / len(precision_values) if precision_values else 0.0
        ),
        "retrieval_forbidden_hit_rate_avg": (
            sum(forbidden_rates) / len(forbidden_rates) if forbidden_rates else 0.0
        ),
        "retrieval_any_relevant_rate": (
            relevant_hit_count / evaluated_retrieval_cases if evaluated_retrieval_cases else 0.0
        ),
        "extraction_cases_evaluated": extraction_eval_cases,
        "extraction_exact_entity_match_rate": (
            extraction_exact_matches / extraction_eval_cases if extraction_eval_cases else None
        ),
    }

    con.close()
    return {"summary": summary, "cases": results}


def _render_markdown_report(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Governance RAG Benchmark Report",
        "",
        f"- Timestamp: {summary['timestamp_utc']}",
        f"- Benchmark file: {summary['benchmark_file']}",
        f"- Model: {summary['model']}",
        f"- Top-K: {summary['top_k']}",
        f"- Run extraction: {summary['run_extraction']}",
        "",
        "## Summary Metrics",
        "",
        f"- Cases total: {summary['cases_total']}",
        f"- Retrieval cases evaluated: {summary['retrieval_cases_evaluated']}",
        f"- Avg Precision@K: {summary['retrieval_precision_at_k_avg']:.4f}",
        f"- Avg Forbidden-hit rate: {summary['retrieval_forbidden_hit_rate_avg']:.4f}",
        f"- Any-relevant retrieval rate: {summary['retrieval_any_relevant_rate']:.4f}",
        f"- Extraction cases evaluated: {summary['extraction_cases_evaluated']}",
        f"- Extraction exact-entity-match rate: {summary['extraction_exact_entity_match_rate']}",
        "",
        "## Case Results",
        "",
    ]

    for case in payload["cases"]:
        lines.append(f"### {case['id']}")
        lines.append(
            f"- {case['ticker']}/{case['year']} {case['item_code']} | Precision@K={case['retrieval']['retrieval_precision_at_k']:.4f}, Forbidden={case['retrieval']['retrieval_forbidden_hit_rate']:.4f}"
        )
        if "extraction" in case:
            ext = case["extraction"]
            lines.append(
                f"- Extraction exact match: {ext['extraction_exact_entity_match']} | predicted_count={ext['predicted_count']} expected_count={ext['expected_count']}"
            )
            if ext["missing_entities"]:
                lines.append(f"- Missing: {', '.join(ext['missing_entities'])}")
            if ext["unexpected_entities"]:
                lines.append(f"- Unexpected: {', '.join(ext['unexpected_entities'])}")
        lines.append("")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 0 governance RAG benchmark")
    parser.add_argument(
        "--benchmark-file",
        default="benchmarks/rag_governance_benchmark.seed.json",
        help="Path to benchmark cases JSON file",
    )
    parser.add_argument("--top-k", type=int, default=INFERENCE_TOP_K)
    parser.add_argument("--model", default=INFERENCE_MODEL)
    parser.add_argument(
        "--run-extraction",
        action="store_true",
        help="Also evaluate extraction entity matches for enabled cases",
    )
    parser.add_argument(
        "--output-json",
        default="benchmarks/reports/rag_benchmark.latest.json",
        help="Path to write JSON report",
    )
    parser.add_argument(
        "--output-md",
        default="benchmarks/reports/rag_benchmark.latest.md",
        help="Path to write markdown report",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    payload = evaluate_governance_benchmark(
        Path(args.benchmark_file),
        top_k=args.top_k,
        model=args.model,
        run_extraction=args.run_extraction,
    )

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    output_md.write_text(_render_markdown_report(payload), encoding="utf-8")

    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
