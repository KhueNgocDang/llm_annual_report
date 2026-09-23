# Product Requirements Document (PRD)

## RAG Retrieval Improvement Plan for Governance and Disclosure Extraction

**Version:** 1.0  
**Date:** 2026-06-25  
**Status:** Proposed (Implementation in phases)

---

## How to Read This PRD

This PRD is organized for incremental delivery.

1. Start with Phase 0 to measure current quality.
2. Implement one phase at a time.
3. After each phase, run the benchmark and verify acceptance criteria.
4. Only move forward when metrics improve or stay stable on key quality targets.

Why this matters:
- Retrieval quality problems are often subtle and can regress silently.
- A phased approach keeps changes measurable and reversible.

---

## 1. Problem Statement

The current RAG pipeline can return semantically similar but contextually wrong chunks, especially in governance extraction tasks where role boundaries are strict.

Example failure class:
- `GOV_SUPERVISORY` (Ban Kiem soat / BKS) can include non-BKS people such as executive or accounting roles when retrieval mixes governance sections.

This causes downstream extraction hallucination and role contamination even when prompts ask for structured outputs.

---

## 2. Objectives

1. Increase retrieval precision for each governance item without reducing coverage.
2. Reduce false-positive entity extraction caused by mixed-section chunks.
3. Make retrieval behavior auditable and testable with explicit metrics.
4. Implement improvements incrementally so each change is verifiable before the next.

### Non-Technical Summary

Today, the system sometimes finds the right topic but from the wrong section. This PRD improves retrieval so the model reads the right evidence before generating outputs.

In simple terms:
1. Find candidate chunks by meaning (semantic search).
2. Re-score candidates by exact wording and section context.
3. Keep only the best chunks for extraction.
4. Reject role-inconsistent outputs before final save.

---

## 3. Scope

### In Scope

- Retrieval and reranking logic for governance items.
- Prompt constraints tied to retrieval scope.
- Post-filter safeguards for item-specific role boundaries.
- Retrieval evaluation harness and regression checks.
- Chunking strategy improvements for governance-heavy tables.

### Out of Scope

- Replacing the LLM provider or core model APIs.
- Rebuilding the full database schema from scratch.
- Redesigning unrelated ETL stages (document fetch/download).

---

## 4. Target Outcomes and Metrics

| Metric | Baseline | Target |
|---|---:|---:|
| Precision@10 (GOV_SUPERVISORY retrieval) | TBD | >= 0.90 |
| False inclusion rate (non-BKS in GOV_SUPERVISORY) | TBD | <= 1% |
| Exact member-list match on labeled supervisory cases | TBD | >= 0.95 |
| Governance extraction retry rate due to anomaly checks | TBD | <= 5% |

Notes:
- Baseline is measured in Phase 0 from current production behavior.
- Targets apply after Phase 4 completion.

Metric interpretation:
1. Precision@10 retrieval: among top 10 retrieved chunks, how many are truly relevant.
2. False inclusion rate: how often wrong-role entities appear in final details.
3. Exact member-list match: whether extracted list exactly matches expected entities.
4. Retry rate: percentage of cases needing strict retry due to anomaly checks.

---

## 5. Phased Implementation Plan

### Phase 0: Baseline and Benchmark Dataset

Goal:
- Establish current retrieval/extraction quality before further changes.

Requirements:
1. Create a labeled benchmark set of 20-50 ticker/year/item cases.
2. For each case, store:
   - expected entities,
   - expected counts,
   - forbidden entities (if any),
   - source section hints (optional).
3. Add an evaluation script to compute:
   - Precision@k for retrieved chunks,
   - extraction exact-match / partial-match,
   - false inclusion rate.

Acceptance Criteria:
1. Benchmark file exists and is versioned in repo.
2. Evaluation command outputs all baseline metrics.

---

### Phase 1: Hybrid Retrieval Scoring

Goal:
- Move from mostly semantic retrieval to hybrid scoring.

Plain-language meaning:
- Hybrid retrieval combines multiple signals instead of relying only on embeddings.
- Typical combination:
   1. Semantic similarity (embedding-based)
   2. Lexical relevance (BM25 or keyword relevance)
   3. Section prior (is this chunk from the likely section)

Requirements:
1. Keep semantic candidate generation (top-N).
2. Add lexical score from item-specific keywords (BM25-compatible design).
3. Add section prior score (when section metadata is available).
4. Compute combined score:

```text
final_score = alpha * semantic_score + beta * lexical_score + gamma * section_prior
```

5. Make `alpha`, `beta`, and `gamma` configurable constants.

Implementation note:
- Lexical score can start with keyword-hit heuristics and later be replaced by BM25 without changing the scoring interface.

Acceptance Criteria:
1. Hybrid scoring is active for all governance items.
2. Metrics improve over Phase 0 baseline for at least:
   - Precision@10,
   - false inclusion rate.

Example:
- For GOV_SUPERVISORY, chunks containing Ban Kiem soat or BKS should outrank chunks that only mention Ke toan truong or Ban dieu hanh.

---

### Phase 2: Section-Aware Metadata and Filtering

Goal:
- Filter retrieval candidates by likely source sections per item.

Requirements:
1. During markdown/chunk ingestion, persist chunk metadata fields:
   - heading path,
   - section title,
   - table or paragraph marker,
   - page marker (if available).
2. Define per-item section allowlists and optional deny lists.
3. Apply section-aware filtering before final reranking.

Acceptance Criteria:
1. Metadata fields are queryable for embedded chunks.
2. Retrieval logs include applied section filters.
3. Benchmark shows reduced cross-section contamination.

Example section rules:
1. GOV_SUPERVISORY allowlist: sections containing Ban Kiem soat, BKS.
2. GOV_BOARD allowlist: sections containing HDQT, Hoi dong quan tri.
3. GOV_AUDIT allowlist: sections containing kiem toan, bao cao kiem toan, uy ban kiem toan.

---

### Phase 3: Two-Stage Retrieval and Lightweight Reranking

Goal:
- Improve candidate quality before prompt construction.

Requirements:
1. Candidate generation: top 40-80 by semantic score.
2. Reranking stage:
   - hybrid score,
   - item-specific include/exclude term boosts,
   - optional lightweight reranker model hook.
3. Pass only top 8-12 chunks to LLM prompt.

Design guidance:
1. Keep stage-1 broad for recall.
2. Make stage-2 strict for precision.
3. Log both stages for audit and debugging.

Acceptance Criteria:
1. Reranking stage is enabled and logged.
2. End-to-end extraction accuracy improves on benchmark set.

---

### Phase 4: Prompt Constraints + Post-Filter Generalization

Goal:
- Apply strict task boundaries to all governance item types.

Requirements:
1. Add item-specific strict scope clauses to prompts.
2. Add post-filter validators to reject role-inconsistent details.
3. Recompute summary `value` fields from filtered `details`.
4. Persist filter action annotations in `reason` for traceability.

Rationale:
- Retrieval can still include borderline chunks.
- Prompt constraints reduce confusion.
- Post-filters provide deterministic safety nets for role boundaries.

Acceptance Criteria:
1. All governance item codes have explicit scope constraints.
2. Post-filter changes are deterministic and test-covered.
3. False inclusion rate meets target in Section 4.

---

### Phase 5: Table-Aware Chunking Improvements

Goal:
- Preserve row-level governance context in noisy OCR markdown.

Requirements:
1. Keep table headers attached to row chunks.
2. Avoid splitting a single row across unrelated chunks where possible.
3. Tune chunk size/overlap for governance sections separately.

Why this phase exists:
- Governance member data is often in long OCR tables.
- If rows are split incorrectly, retrieval and extraction both degrade.

Acceptance Criteria:
1. Table extraction quality improves on benchmark cases with long personnel tables.
2. Reduced missing-role or merged-row errors in extraction output.

---

### Phase 6: Pre-Persist Anomaly Check and Auto-Retry

Goal:
- Catch likely wrong outputs before writing final results.

Requirements:
1. Add anomaly scorer based on role/item consistency.
2. If anomaly score exceeds threshold:
   - re-run extraction once with stricter retrieval filters.
3. Log both attempts and final selected output.

Example anomaly triggers:
1. GOV_SUPERVISORY contains roles like Tong giam doc, Ke toan truong, or HDQT.
2. Extracted member count is far outside expected range for that item.

Acceptance Criteria:
1. Anomaly logic is configurable and observable.
2. Retry path reduces final wrong-row persistence rate.

---

## 6. Execution Order (One-by-One)

Implement phases in this order:

1. Phase 0 (benchmark first)
2. Phase 1 (hybrid scoring)
3. Phase 2 (section metadata/filtering)
4. Phase 3 (two-stage reranking)
5. Phase 4 (prompt + post-filter generalization)
6. Phase 5 (table-aware chunking)
7. Phase 6 (anomaly check + auto-retry)

Gate rule:
- Do not start next phase until current phase metrics and acceptance criteria are met.

Recommended sprint shape:
1. One phase per sprint.
2. End each sprint with benchmark report and go/no-go decision.

---

## 7. Deliverables by Phase

| Phase | Primary Deliverables |
|---|---|
| 0 | Benchmark dataset + evaluation CLI/report |
| 1 | Hybrid retrieval implementation + config constants |
| 2 | Metadata-enriched chunks + section filter rules |
| 3 | Two-stage retrieval/rerank module + logs |
| 4 | Item-specific scope constraints + post-filter validators |
| 5 | Table-aware chunking logic + chunk QA examples |
| 6 | Anomaly detector + strict retry policy |

---

## 8. Risks and Mitigations

1. Risk: Over-filtering reduces recall.
   - Mitigation: Keep semantic candidate pool large before filtering; monitor recall metrics.
2. Risk: OCR noise breaks section detection.
   - Mitigation: Add fuzzy heading matching and fallback lexical heuristics.
3. Risk: Increased latency from reranking and retry logic.
   - Mitigation: Limit reranker candidate count and max retry to one pass.

---

## 9. Rollout Strategy

1. Enable each phase behind feature flags.
2. Run benchmark and a limited ticker-year subset first.
3. Compare against baseline.
4. Promote phase to full runs once metrics pass.

---

## 10. Initial Backlog (Ready-to-Implement)

1. Create benchmark JSON schema and sample cases.
2. Build evaluation script for retrieval/extraction metrics.
3. Extract keyword dictionaries and section priors per governance item.
4. Implement weighted hybrid scorer.
5. Add retrieval debug logging table or structured logs.

Suggested first ticket breakdown:
1. Add retrieval benchmark schema file.
2. Add 10 seed benchmark cases (including AAA 2025 GOV_SUPERVISORY).
3. Build evaluate command that outputs markdown and JSON summary.
4. Add hybrid scorer interface with pluggable lexical module.

---

## 11. Definition of Done (Overall)

The RAG improvement program is complete when:

1. All phases are implemented and validated.
2. Target metrics in Section 4 are met or exceeded.
3. Governance extraction errors like non-BKS leakage in `GOV_SUPERVISORY` are consistently prevented in benchmark and production spot checks.
