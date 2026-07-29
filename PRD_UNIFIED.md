# Product Requirements Document (Unified)

## Annual Report Intelligence Platform (Rebuild Edition)

Version: 2.0 (Rebuild)
Date: 2026-07-29
Status: Planning for Greenfield Implementation

## 1. Purpose

This PRD defines how to recreate the project from scratch while preserving
business goals, data contracts, and research outputs.

This document is now the primary build blueprint for:

1. Product scope and architecture.
2. Functional and non-functional requirements.
3. Phased implementation tasks.
4. Acceptance criteria for release readiness.

## 2. Product Vision

Build a local-first platform that ingests Vietnamese listed-company reports,
processes them into queryable text, and evaluates disclosure quality with
auditable LLM + RAG pipelines.

Core outcomes:

1. Reproducible company-year dataset from raw filings.
2. Deterministic, model-keyed extraction/scoring outputs.
3. Analyst-friendly query, browsing, and export workflows.

## 3. Users and Primary Jobs

Primary users:

1. Researchers and analysts.
2. Policy and compliance reviewers.
3. Pipeline operators and maintainers.

Primary jobs to be done:

1. Register target companies and ingest source data.
2. Build annual-report and BCTC content corpus.
3. Run embeddings and LLM inference at scale.
4. Audit prompts, evidence, and output quality.
5. Export structured results for downstream analysis.

## 4. Scope for Rebuild

In scope:

1. Full reimplementation of backend modules and UI shell.
2. DuckDB schema creation and migration bootstrap.
3. End-to-end pipeline orchestration and retry logic.
4. LLM/RAG modules for EDC, PROPER-VN, governance, and BCTC audit.
5. Query, audit, and reporting tools in NiceGUI.

Out of scope:

1. Distributed/multi-tenant deployment.
2. Replacing DuckDB with external OLTP/warehouse services.
3. Human-free correction of OCR/source-document defects.

## 5. Target Architecture (From Scratch)

High-level modules:

1. Source adapters:
   - VNDirect adapter for stocks, models, statements, ratios.
   - Vietstock adapter for listings and downloads.
2. File-processing layer:
   - Raw storage, staging, conversion, and lineage tracking.
3. Data layer:
   - DuckDB schema, repositories, and read/write contracts.
4. Intelligence layer:
   - Chunking, embedding, retrieval, reranking, and inference.
5. App layer:
   - NiceGUI pages, task-state management, logs, and actions.

Primary storage directories:

1. data/raw
2. data/staging
3. data/output
4. data/markdown
5. data/markdown_bctc
6. data/logs

Runtime configuration model:

1. Environment-driven config for models and retrieval weights.
2. Database path and data directories configurable at startup.
3. Safe defaults for deterministic inference behavior.

## 6. Functional Requirements

### FR-1. Project Bootstrap and Environment

1. Initialize reproducible Python environment and dependency management.
2. Provide startup commands for app, smoke tasks, and validation checks.
3. Ensure all required directories and DB schema can be created from zero.

Acceptance:

1. Fresh clone can run initialization and start app without manual patching.
2. First-run bootstrap creates required tables and directories.

### FR-2. Company and Market Data Ingestion

1. Sync stocks universe (HOSE/HNX/UPCOM).
2. Maintain user-controlled company target registry.
3. Sync financial models, statements, and ratios for target tickers.

Acceptance:

1. Per-task run status and row counts are displayed in UI.
2. Partial failures do not block other tickers.

### FR-3. Document Listing and Download

1. Sync annual and BCTC listings from Vietstock.
2. Download unsynced files into ticker-scoped raw paths.
3. Handle pagination and metadata deduplication.

Acceptance:

1. Listings and download states remain consistent across reruns.
2. Failed downloads are logged and retryable.

### FR-4. Conversion and Content Loading

1. Convert raw annual/BCTC files to markdown using marker-based OCR.
2. Support archive/unstructured-file resolution heuristics.
3. Load markdown content into normalized DB tables with idempotency.

Acceptance:

1. Converted files map back to source documents and ticker/year keys.
2. Reloading does not duplicate logical records.

### FR-5. Embedding and Retrieval

1. Token-based chunking with overlap for annual and BCTC content.
2. Embedding generation with model/dimension compatibility checks.
3. Retrieval by task/item with top-k selection and scored evidence.
4. Configurable hybrid retrieval weights and candidate expansion.

Acceptance:

1. Evidence chunks and scores are inspectable per inference request.
2. Retrieval settings are traceable in run metadata.

### FR-6. Inference and Extraction

1. EDC inference supports:
   - edc
   - edc_alt
   - edc_alt_two
2. PROPER-VN inference supports stage logic and final class output.
3. Governance extraction supports annual governance item set.
4. BCTC audit extraction supports financial-statement governance items.
5. Batch submission and sync workflows reconcile pending/running tasks.

Acceptance:

1. Outputs are model-keyed and version-safe.
2. Item-level reasoning fields are persisted and queryable.

### FR-7. Query, Audit, and Export

1. Filter outputs by dataset, ticker, year, model, and item code.
2. Inspect request-input logs by task type.
3. Export tabular outputs for research pipelines.
4. Preserve strict separation among edc families in queries.

Acceptance:

1. User can trace output to prompt/evidence history.
2. Exported data supports paper/BI workflows without manual transformation.

## 7. Non-Functional Requirements

1. Reliability: retryable jobs, resumable pipelines, and durable state.
2. Reproducibility: model-keyed outputs and deterministic defaults.
3. Observability: task logs, status tables, and audit trails.
4. Performance: acceptable throughput on local workstation scale.
5. Maintainability: modular code boundaries and explicit interfaces.

## 8. Data Contracts

Required core domains:

1. Company/financial data tables.
2. Document metadata and file lineage.
3. Markdown content stores.
4. Embedding/vector tables.
5. Inference result and job tables.
6. Request-input audit logging tables.

Output contract guarantees:

1. Results preserve item code granularity.
2. EDC strict/alt/alt_two segmentation is query-safe.
3. All outputs include sufficient metadata for reproducibility.

## 9. UX Requirements

Rebuild must include these operational surfaces:

1. Home and company management.
2. Jobs and job matrix.
3. LLM tasks and output query.
4. Input audit and extraction browser.
5. Converter, data studio, and report/document browsers.

UX behavior requirements:

1. Task cards with run/progress/error/summary states.
2. Prevent conflicting concurrent pipeline runs.
3. Show actionable error details and retry paths.

## 10. Rebuild Delivery Plan

### Phase 0: Foundation

1. Repository skeleton, config module, and environment bootstrap.
2. DB initialization and migration baseline.
3. Shared task-state and logging framework.

Exit criteria:

1. Fresh setup can launch app shell and initialize empty DB.

### Phase 1: Data Ingestion MVP

1. Stocks and company registry workflow.
2. Financial models/statements/ratios sync.
3. Vietstock listings + download pipeline.

Exit criteria:

1. Target tickers can be fully synced and raw documents downloaded.

### Phase 2: Conversion and Loading MVP

1. Annual and BCTC conversion tasks.
2. Content loading into DB with idempotent logic.
3. File lineage and failure categorization.

Exit criteria:

1. Queryable markdown corpus exists for selected ticker/year set.

### Phase 3: Intelligence MVP

1. Embedding jobs and retrieval layer.
2. EDC, PROPER-VN, governance, BCTC inference tasks.
3. LLM request-input audit logging.

Exit criteria:

1. End-to-end run produces auditable outputs for all task families.

### Phase 4: Quality and Hardening

1. Retrieval benchmark dataset and metrics harness.
2. Section-aware filtering and reranking improvements.
3. Anomaly detection and controlled retry policy.

Exit criteria:

1. Benchmark targets met without regression in core throughput.

### Phase 5: Analyst Experience

1. Saved SQL templates and export profiles.
2. Reproducibility report (settings, model, run IDs).
3. Final UX polish for browse/query workflows.

Exit criteria:

1. Analyst can run from ingestion to export without code changes.

## 11. Rebuild Backlog (Single Source)

Priority bands:

1. P0: blocking for end-to-end MVP.
2. P1: required for robust production-like operation.
3. P2: optimization and advanced analysis.

P0 tasks:

1. Bootstrap and schema initialization from empty workspace.
2. Source adapters for VNDirect and Vietstock with retry-safe pagination.
3. Annual/BCTC conversion and DB loading with idempotency.
4. Embedding + EDC/PROPER/governance/BCTC inference base flows.
5. UI task cards and run-state orchestration.

P1 tasks:

1. Retrieval quality benchmark harness.
2. Section-aware filtering and hybrid reranking hardening.
3. Batch sync reconciliation and failure recovery tools.
4. Export-ready SQL templates and audit-centric query views.

P2 tasks:

1. Trend dashboards and sector comparisons.
2. Ensemble scoring and confidence calibration.
3. Additional disclosure frameworks beyond EDC/PROPER-VN.

## 12. Risks and Mitigations

1. External API behavior changes.
   - Mitigation: adapter abstraction, schema validation, and guarded parsing.
2. OCR variability in scanned or archived files.
   - Mitigation: conversion fallback rules and quality diagnostics.
3. Retrieval cross-section contamination.
   - Mitigation: benchmark-driven reranking and post-filter validators.
4. Runtime cost and latency growth.
   - Mitigation: staged runs, model controls, and caching policies.

## 13. Acceptance Criteria

The rebuilt project is accepted when:

1. A clean environment can run end-to-end pipeline for a test ticker set.
2. All task families produce queryable, model-keyed outputs.
3. Audit trail includes request inputs and retrieval evidence.
4. Reproducibility package (config + run metadata + exports) is generated.

## 14. Source-of-Truth Policy

1. This PRD defines the rebuild plan and supersedes legacy fragmented PRDs.
2. Implementation details live in code, but must satisfy this PRD.
3. Any scope change must update this PRD and its backlog section.
