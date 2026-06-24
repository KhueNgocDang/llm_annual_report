# Product Requirements Document (PRD)

## Environment Annual Report — Carbon Disclosure Grading System

**Version:** 1.0
**Date:** March 2026
**Status:** In Production

---

## 1. Product Overview

### 1.1 Problem Statement

Vietnamese listed companies are increasingly expected to disclose environmental
and climate-related information in their annual reports, yet there is no
standardized, scalable method to evaluate the quality and completeness of these
disclosures across the market. Manual review of hundreds of annual reports — each
written in Vietnamese and often published as scanned PDFs — is impractical for
researchers and regulators.

### 1.2 Solution

An end-to-end automated pipeline and interactive dashboard that:

1. **Collects** annual report PDFs from Vietstock for Vietnamese listed companies.
2. **Converts** PDFs to structured text via OCR.
3. **Embeds** the text as vectors for semantic search.
4. **Evaluates** each report against two environmental disclosure frameworks
   using RAG (Retrieval-Augmented Generation) with large language models.
5. **Presents** results in an interactive Streamlit dashboard with export
   capabilities for academic analysis.

### 1.3 Target Users

- **Academic researchers** studying corporate environmental disclosure in Vietnam.
- **Policy analysts** evaluating market-wide compliance with environmental
  reporting standards.
- **Students and thesis writers** needing structured environmental disclosure
  data across companies and years.

### 1.4 Scope

- **Companies:** Vietnamese companies listed on HOSE, HNX, and UPCOM exchanges.
- **Document type:** Annual reports (Báo cáo thường niên).
- **Time period:** Fiscal years 2019–2024.
- **Language:** Vietnamese (reports); English (evaluation output and UI).
- **Evaluation frameworks:** Environmental Disclosure Checklist (EDC) and
  PROPER-VN Environmental Rating.

---

## 2. Goals and Success Metrics

### 2.1 Goals

| # | Goal | Description |
|---|------|-------------|
| G1 | Automate data collection | Eliminate manual PDF downloading by integrating with the Vietstock API. |
| G2 | Reliable text extraction | Convert Vietnamese PDF reports — including scanned documents — to machine-readable text with high fidelity. |
| G3 | Scalable evaluation | Evaluate 50+ companies × 6 years × 18 checklist items without human intervention. |
| G4 | Reproducible scoring | Produce deterministic, model-keyed results that can be compared across LLM configurations. |
| G5 | Research-ready output | Provide structured data (SQL views, CSV export) suitable for statistical analysis and paper writing. |

### 2.2 Success Metrics

| Metric | Target |
|--------|--------|
| Report coverage | ≥ 80% of expected company-year slots have a raw PDF |
| OCR conversion rate | ≥ 95% of staged PDFs successfully converted to markdown |
| Embedding completion | 100% of converted reports chunked and embedded |
| Inference completion | 100% of embedded reports evaluated against both frameworks |
| Cross-model comparison | ≥ 2 LLM models evaluated per company for result validation |

---

## 3. System Architecture

### 3.1 Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | ≥ 3.13 |
| Package manager | uv | — |
| Web framework | Streamlit | ≥ 1.50 |
| Database | DuckDB (embedded, single-file) | ≥ 1.4.4 |
| Vector search | DuckDB VSS extension | Built-in |
| PDF → Markdown | marker-pdf | 1.8.5 |
| Embedding API | OpenAI (text-embedding-3-small/large) | — |
| LLM API | OpenAI (gpt-4.1-mini, gpt-4.1, o3-mini, etc.) | — |
| Token counting | tiktoken | ≥ 0.12 |
| HTML parsing | BeautifulSoup4 | ≥ 4.13 |
| Data manipulation | pandas | < 3.0 |

### 3.2 Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION                              │
│                                                                     │
│  ┌──────────┐    ┌─────────────┐    ┌──────────┐    ┌───────────┐  │
│  │ Register │───▶│   Vietstock │───▶│ Download │───▶│   File    │  │
│  │Companies │    │  API Fetch  │    │   PDFs   │    │ Inventory │  │
│  └──────────┘    └─────────────┘    └──────────┘    └───────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         PROCESSING                                  │
│                                                                     │
│  ┌──────────┐    ┌─────────────┐    ┌──────────┐    ┌───────────┐  │
│  │  Stage   │───▶│  marker-pdf │───▶│  Chunk   │───▶│  OpenAI   │  │
│  │  PDFs    │    │  OCR + Parse│    │   Text   │    │ Embedding │  │
│  └──────────┘    └─────────────┘    └──────────┘    └───────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          ANALYSIS                                   │
│                                                                     │
│  ┌──────────────────────┐    ┌──────────────────────────────────┐   │
│  │  EDC Inference       │    │  PROPER-VN Rating                │   │
│  │  (18 binary items)   │    │  (2-stage, 5-color)             │   │
│  │  RAG + LLM → 0/1    │    │  RAG + LLM → Black…Gold         │   │
│  └──────────────────────┘    └──────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│              ┌───────────────────────────────┐                      │
│              │  DuckDB Score Views + Export  │                      │
│              └───────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 Directory Structure

```
data/
├── raw/           # Downloaded PDFs, organized by ticker (e.g., raw/VNM/)
├── staging/       # Standardized PDFs (annual_report_<YEAR>.pdf per ticker)
├── markdown/      # OCR output per ticker/year (e.g., markdown/VNM/annual_report_2023/)
├── logs/          # Job execution logs (job_<ID>.log)
└── environment_reports.duckdb   # Single-file database (all tables, embeddings, results)
```

### 3.4 Database Schema Summary

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `companies` | Company registry | ticker (PK), name, sector, exchange |
| `vietstock_documents` | Fetched document metadata | ticker, doc_type, file_url, file_info_id, synced_to_raw |
| `file_inventory` | Raw file scan results | ticker, file_path, year, is_recognized, is_corrected, is_ignored |
| `pipeline_files` | File tracking across stages | ticker, year, stage (raw/staging/markdown), file_path |
| `conversion_jobs` | OCR conversion job tracking | ticker, year, status, pid, log_path |
| `document_embeddings` | Vector store | ticker, year, chunk_index, chunk_text, embedding (FLOAT[1536]) |
| `embedding_jobs` | Embedding job tracking | ticker, year, status, model, num_chunks |
| `inference_results` | EDC evaluation results | ticker, year, category_code, is_valid, reason, model |
| `inference_jobs` | EDC job tracking | ticker, year, status, model, categories_done/total |
| `proper_vn_results` | PROPER-VN indicator results | ticker, year, indicator_code, is_present, evidence_level, model |
| `proper_vn_jobs` | PROPER-VN job tracking | ticker, year, status, model, color, s2_score |

**Key design decisions:**
- All results are keyed by `(ticker, year, model)` — running a different LLM
  never overwrites previous results.
- Embeddings and raw results are stored together in a single DuckDB file for
  portability (no external database server required).
- DuckDB VSS extension enables in-database cosine similarity search without an
  external vector store.

---

## 4. Feature Requirements

### 4.1 Data Collection

#### FR-1: Company Registry

| Requirement | Description |
|-------------|-------------|
| FR-1.1 | Add individual companies by ticker, name, sector, and exchange. |
| FR-1.2 | Bulk-import companies via comma-separated text or CSV upload. |
| FR-1.3 | Auto-register companies detected from existing files on disk. |
| FR-1.4 | Delete a company and all its associated data (with confirmation). |

#### FR-2: Document Fetching (Vietstock Integration)

| Requirement | Description |
|-------------|-------------|
| FR-2.1 | Fetch annual report and financial statement metadata from the Vietstock API. |
| FR-2.2 | Preview fetched documents before saving to the database. |
| FR-2.3 | Deduplicate documents by `file_info_id` to prevent duplicate entries. |
| FR-2.4 | Download PDFs to the raw directory, organized by ticker. |
| FR-2.5 | Batch-fetch documents for all registered companies in one operation. |
| FR-2.6 | Batch-download all unsynced documents to the raw layer. |

#### FR-3: File Inventory

| Requirement | Description |
|-------------|-------------|
| FR-3.1 | Scan `data/raw/` and classify files as Recognized, Multi-part, Corrected, Ignored, or Unrecognized. |
| FR-3.2 | Extract year from Vietnamese filenames via regex pattern matching. |
| FR-3.3 | Support PDF and RAR file formats. |
| FR-3.4 | Apply corrected-file priority: when a "điều chỉnh" (adjusted) version exists, mark the original as ignored. |
| FR-3.5 | Filter inventory by ticker, year, file type, and tag status. |

#### FR-4: Missing Reports

| Requirement | Description |
|-------------|-------------|
| FR-4.1 | Compute the expected company × year grid based on configured year range. |
| FR-4.2 | Identify gaps where no recognized raw PDF exists. |
| FR-4.3 | Display coverage metrics (total slots, missing count, coverage %). |
| FR-4.4 | Filter missing reports by ticker and year. |

### 4.2 Document Processing

#### FR-5: PDF Staging

| Requirement | Description |
|-------------|-------------|
| FR-5.1 | Copy recognized raw PDFs to `data/staging/` with standardized names (`annual_report_<YEAR>.pdf`). |
| FR-5.2 | One staged file per ticker+year; corrected files take priority. |
| FR-5.3 | Filter staging by ticker and year range. |
| FR-5.4 | Option to overwrite existing staged files. |
| FR-5.5 | Clear all staged files. |

#### FR-6: PDF → Markdown Conversion (OCR)

| Requirement | Description |
|-------------|-------------|
| FR-6.1 | Convert staged PDFs to structured markdown using `marker_single` with `--force_ocr`. |
| FR-6.2 | Track each conversion as a database job with status lifecycle: pending → running → completed / failed / cancelled. |
| FR-6.3 | Stream stdout/stderr to per-job log files with live viewing in the dashboard. |
| FR-6.4 | Record process ID (PID) for running jobs. |
| FR-6.5 | Configurable batch sizes for marker-pdf's internal ML models (layout, detection, recognition, equation, table). |
| FR-6.6 | Skip creating duplicate jobs for the same ticker+year. |
| FR-6.7 | Reset failed jobs back to pending; cancel running jobs. |

**marker-pdf OCR Configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `force_ocr` | True | Re-OCR every page regardless of existing text |
| `layout_batch_size` | 12 | Page-level layout analysis batch |
| `detection_batch_size` | 8 | Text-line and table detection batch |
| `ocr_error_batch_size` | 12 | OCR error correction batch |
| `recognition_batch_size` | 32 | Character recognition batch |
| `equation_batch_size` | 16 | Math formula recognition batch |
| `table_rec_batch_size` | 12 | Table structure recognition batch |

### 4.3 Embedding Pipeline

#### FR-7: Document Embedding

| Requirement | Description |
|-------------|-------------|
| FR-7.1 | Split markdown documents into overlapping chunks (configurable token size and overlap). |
| FR-7.2 | Generate embeddings via the OpenAI API and store them in the DuckDB vector store. |
| FR-7.3 | Track embedding jobs with status lifecycle: pending → running → completed / failed. |
| FR-7.4 | Support model selection between `text-embedding-3-small` (1,536 dims) and `text-embedding-3-large` (3,072 dims). |
| FR-7.5 | Batch embed all pending reports with progress tracking. |
| FR-7.6 | Embed or re-embed individual reports. |
| FR-7.7 | Perform semantic search across embedded documents with optional ticker/year filters. |
| FR-7.8 | Delete embeddings per report or full reset. |

**Embedding Configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| Model | `text-embedding-3-small` | OpenAI embedding model |
| Dimensions | 1,536 | Vector dimensionality |
| Chunk size | 512 tokens | Maximum tokens per chunk |
| Chunk overlap | 128 tokens | Sliding window overlap |
| Batch size | 512 | Max texts per API call |
| Similarity metric | Cosine similarity | Nearest-neighbor search metric |

### 4.4 Evaluation Frameworks

#### FR-8: Environmental Disclosure Checklist (EDC)

| Requirement | Description |
|-------------|-------------|
| FR-8.1 | Evaluate each company-year report against 18 checklist items across 5 groups (CC, GHG, EC, RC, ACC). |
| FR-8.2 | For each item, retrieve top-*k* relevant chunks via vector similarity search. |
| FR-8.3 | Send retrieved text + item criteria to the LLM; parse a structured JSON response with `is_valid` (boolean) and `reason`. |
| FR-8.4 | Score each item as binary 0 (not disclosed) or 1 (disclosed). |
| FR-8.5 | Compute aggregate disclosure score (0–18) per company-year. |
| FR-8.6 | Track inference jobs with progress (categories_done / categories_total). |
| FR-8.7 | Support model selection: run the same report against different LLMs and compare results. |
| FR-8.8 | Run inference per-item across all reports (e.g., evaluate GHG3 for every company). |
| FR-8.9 | Provide database views in both long and wide (pivoted) format. |

**EDC Checklist (18 items):**

| Group | Items | Description |
|-------|-------|-------------|
| Climate Change | CC1, CC2 | Risk assessment and financial implications of climate change |
| GHG Emissions | GHG1–GHG7 | Emission calculation methodology, verification, totals, scopes, sources, facilities, historical comparison |
| Energy Consumption | EC1–EC3 | Total energy, renewables, breakdown by type/facility |
| GHG Reduction | RC1–RC4 | Strategies, targets, achievements, cost planning |
| GHG Accountability | ACC1, ACC2 | Governance responsibility and executive compensation alignment |

#### FR-9: PROPER-VN Environmental Rating

| Requirement | Description |
|-------------|-------------|
| FR-9.1 | Evaluate each company-year report using a two-stage framework adapted from Indonesia's PROPER system. |
| FR-9.2 | **Stage 1 — Regulatory Compliance:** Check 3 indicators (serious violation, minor non-compliance, stated compliance) with priority-based decision logic. |
| FR-9.3 | **Stage 2 — Beyond-Compliance:** Score 4 indicators on a three-level evidence scale (none=0, basic_mention=1, quantified=2). |
| FR-9.4 | Assign a final color rating: Black, Red, Blue, Green, or Gold based on threshold rules. |
| FR-9.5 | Track jobs with progress and store the final color + S2 score. |
| FR-9.6 | Support model selection for cross-model comparison. |
| FR-9.7 | Provide database views in both long and wide format. |

**Color Classification:**

| Color | Condition |
|-------|-----------|
| Black | Serious violation detected (S1_VIOLATION) |
| Red | Minor non-compliance without stated compliance |
| Blue | Compliant, S2 score < 37.5% of max (< 3/8) |
| Green | Compliant, S2 score ≥ 37.5% of max (≥ 3/8) |
| Gold | Compliant, S2 score ≥ 75% of max (≥ 6/8) |

### 4.5 LLM Configuration

#### FR-10: Model and Inference Settings

| Requirement | Description |
|-------------|-------------|
| FR-10.1 | Default LLM: `gpt-4.1-mini`. |
| FR-10.2 | Changeable per-run via sidebar (supported: gpt-4.1-mini, gpt-4.1, gpt-4o, o3-mini, o3, o4-mini, gpt-5-mini, gpt-5-nano, gpt-5, etc.). |
| FR-10.3 | Temperature = 0 for deterministic output on standard models. |
| FR-10.4 | Reasoning models (o-series, GPT-5) use `reasoning_effort = "low"` instead of temperature. |
| FR-10.5 | Top-*k* retrieval configurable per-run (default: 5). |

### 4.6 Dashboard and Tools

#### FR-11: Pipeline Dashboard

| Requirement | Description |
|-------------|-------------|
| FR-11.1 | Display KPIs: company count, raw PDFs, staged PDFs, markdown files, coverage %. |
| FR-11.2 | Show a completion matrix (company × year) with status indicators for each pipeline stage. |

#### FR-12: Query Editor

| Requirement | Description |
|-------------|-------------|
| FR-12.1 | Execute arbitrary read-only SQL queries against the DuckDB database. |
| FR-12.2 | Browse database schema (tables, views, columns) in a sidebar explorer. |
| FR-12.3 | Provide saved query templates (companies, inference results, PROPER-VN ratings, embedding stats). |
| FR-12.4 | Export query results to CSV. |

#### FR-13: Database Management

| Requirement | Description |
|-------------|-------------|
| FR-13.1 | Rebuild database: drop all tables, recreate schema, and rescan data directories (sidebar action with confirmation). |
| FR-13.2 | Full reset: delete all data and files (danger zone with confirmation). |

---

## 5. Non-Functional Requirements

### 5.1 Performance

| Requirement | Description |
|-------------|-------------|
| NFR-1 | PDF conversion: process 1 report within 2–10 minutes depending on page count and GPU availability. |
| NFR-2 | Embedding: process 1 report (100–500 chunks) within 30 seconds via batched API calls. |
| NFR-3 | Inference: evaluate 1 report against 18 EDC items within 60 seconds (parallel-capable). |
| NFR-4 | Dashboard: load and render completion matrix for 50+ companies within 2 seconds. |

### 5.2 Reliability

| Requirement | Description |
|-------------|-------------|
| NFR-5 | Job recovery: failed conversion/embedding/inference jobs can be reset and re-run without data corruption. |
| NFR-6 | Job idempotency: re-creating jobs for an existing ticker+year is safely skipped. |
| NFR-7 | Data deduplication: Vietstock documents are deduplicated by `file_info_id`; corrected files supersede originals. |

### 5.3 Portability

| Requirement | Description |
|-------------|-------------|
| NFR-8 | Single-file database: all data, embeddings, and results stored in one portable DuckDB file. |
| NFR-9 | No external services beyond the OpenAI API — no database server, no vector store server. |
| NFR-10 | Runs locally on Linux with GPU (for marker-pdf) or CPU-only (slower). |

### 5.4 Reproducibility

| Requirement | Description |
|-------------|-------------|
| NFR-11 | Model-keyed results: switching LLM models creates new result sets without overwriting previous runs. |
| NFR-12 | Deterministic inference: temperature=0 / reasoning_effort="low" for consistent output. |
| NFR-13 | All inference results include the retrieved chunks and similarity scores for auditability. |

### 5.5 Extensibility

| Requirement | Description |
|-------------|-------------|
| NFR-14 | Checklist items and PROPER-VN indicators are defined in `config.py` — adding or modifying items requires no code changes to the inference engine. |
| NFR-15 | New embedding models can be added by selecting them in the dashboard sidebar. |
| NFR-16 | New LLM models can be used by adding them to the model list in the page configuration. |

---

## 6. Data Outputs for Research

### 6.1 EDC Output Tables

**`disclosure_scores` (long format):**
One row per (ticker, year, model, category_code) with columns: `ticker`,
`company_name`, `year`, `model`, `category_code`, `score` (0/1), `reason`.

**`disclosure_scores_wide` (pivoted):**
One row per (ticker, year, model) with columns for each checklist code
(CC1, CC2, GHG1–GHG7, EC1–EC3, RC1–RC4, ACC1, ACC2) plus `total_score`.

### 6.2 PROPER-VN Output Tables

**`proper_vn_scores` (long format):**
One row per (ticker, year, model, indicator_code) with columns: `ticker`,
`company_name`, `year`, `model`, `indicator_code`, `score` (0/1),
`evidence_level`, `reason`.

**`proper_vn_scores_wide` (pivoted):**
One row per (ticker, year, model) with columns for each indicator
(S1_VIOLATION, S1_MINOR_NC, S1_COMPLIANCE, S2_ISO14001, S2_CARBON_DISC,
S2_REDUCTION, S2_EFFICIENCY) plus `color`, `s2_score`, `s2_max_score`.

### 6.3 Export Capabilities

- Any SQL query result can be exported to CSV via the Query Editor.
- All tables and views are accessible through standard DuckDB SQL for
  integration with R, Python (pandas), or statistical software.

---

## 7. Constraints and Assumptions

| # | Constraint / Assumption |
|---|------------------------|
| C1 | Annual reports are in Vietnamese and follow the naming convention "Báo cáo thường niên năm \<YEAR\>.pdf". |
| C2 | Reports for fiscal year Y are expected to be published by March Y+1. |
| C3 | The Vietstock API is available and returns document metadata in the expected format. |
| C4 | An OpenAI API key with access to embedding and chat completion models is required. |
| C5 | marker-pdf requires Python ≥ 3.13 and benefits significantly from GPU availability (CUDA). |
| C6 | The system evaluates self-reported disclosures only — it does not independently verify environmental claims. |
| C7 | OCR quality may degrade for heavily scanned or low-resolution PDFs, affecting downstream evaluation accuracy. |
| C8 | LLM outputs are non-deterministic across model versions even with temperature=0; results should be interpreted within the context of the model used. |

---

## 8. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Vietstock API rate limiting or downtime | Cannot fetch new documents | Batch fetching with retry logic; local file inventory as fallback |
| OCR errors in tables and numbers | Incorrect data extraction for GHG totals, energy figures | Force OCR with marker-pdf's error correction pipeline; manual review of key metrics |
| LLM hallucination or misinterpretation | False positive/negative scores | Store retrieved chunks alongside results for human verification; cross-model comparison |
| Vietnamese text nuance | Criteria may be interpreted differently by the LLM | Criteria written in Vietnamese to minimize translation loss; reason field for auditability |
| Large-scale API costs | High embedding + inference costs for many companies × years | Batch size optimization; model-keyed results prevent redundant re-runs |
| DuckDB file corruption | Loss of all data | Regular backups of the `.duckdb` file (manual or scripted) |

---

## 9. Future Considerations

The following are not in current scope but represent natural extensions:

- **Multi-year trend analysis:** Dashboard visualizations showing disclosure
  improvement or regression over the 2019–2024 period.
- **Sector-level benchmarking:** Aggregate scores by industry sector for
  comparative analysis.
- **Additional disclosure frameworks:** Extend beyond EDC and PROPER-VN to
  support TCFD, GRI, or ISSB standards.
- **Automated report generation:** Generate per-company or market-wide summary
  reports in PDF or Word format.
- **Ensemble scoring:** Combine results from multiple LLMs to produce
  consensus scores with confidence intervals.
- **Financial data integration:** Correlate environmental disclosure scores
  with financial performance metrics.
- **Expanded document types:** Include sustainability reports and ESG reports
  in addition to annual reports.
