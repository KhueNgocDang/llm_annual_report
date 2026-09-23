# Methodology: Grading Carbon Disclosure in Vietnamese Company Annual Reports

## Short Methodology for Paper

This study uses secondary data from Vietnamese listed transportation companies'
annual reports to assess carbon-emission disclosure across road, rail,
maritime, and aviation subsectors. Annual reports collected from Vietstock are
converted from PDF to markdown, segmented into text chunks, and processed with
a retrieval-augmented generation pipeline to identify disclosure evidence. Each
company-year observation is then evaluated using two measures developed in this
project.

First, a **Disclosure Score** is constructed from an 18-item Environmental
Disclosure Checklist covering climate-related risks, greenhouse gas emissions,
energy consumption, emission-reduction actions, and governance accountability.
Each item is coded **1** if the report contains relevant disclosure and **0**
otherwise, and the total score is summed from **0 to 18**, with higher values
indicating more extensive carbon and environmental disclosure.

Second, a **PROPER-VN Score** is used to classify the quality of environmental
performance disclosure. In Stage 1, reports are screened for serious violations,
minor non-compliance, and stated regulatory compliance. Firms are assigned
**Black** for serious violations and **Red** for minor non-compliance without
evidence of compliance. Firms passing Stage 1 proceed to Stage 2, where four
beyond-compliance indicators are scored on a 0-2 scale: ISO 14001 or equivalent
environmental management systems, quantified carbon disclosure, emission
reduction targets or achievements, and energy/resource-efficiency initiatives.
The Stage 2 total ranges from **0 to 8** and is converted into **Blue**
(<3), **Green** (3-5), or **Gold** (6-8).

The resulting Disclosure Score and PROPER-VN rating are used to compare carbon
disclosure practices across transport modes and identify gaps in transparency
and environmental accountability in Vietnam's transportation sector. To situate
the Vietnam evidence in the broader literature, the study also complements this
analysis with a bibliometric review of Web of Science publications from the last
five years on carbon emission disclosure and transportation sustainability.

## 1. Introduction

This document describes the methodology used to evaluate and grade the quality
of carbon and environmental disclosure in Vietnamese listed companies' annual
reports. Two complementary frameworks are applied:

1. **Environmental Disclosure Checklist (EDC)** — a binary (0/1) assessment
   across 18 disclosure items grouped into five thematic categories.
2. **PROPER-VN Environmental Rating** — a five-color rating adapted from
   Indonesia's PROPER system, combining regulatory compliance assessment with
   beyond-compliance scoring.

Both frameworks use **Retrieval-Augmented Generation (RAG)** over the full text
of annual reports, with vector-based semantic search for evidence retrieval and
a large language model (LLM) for structured evaluation.

---

## 2. Data Collection and Processing Pipeline

### 2.1 Source Data

Annual reports are collected from **Vietstock** for Vietnamese listed companies
across the HOSE, HNX, and UPCOM exchanges. The target period covers fiscal
years **2019–2024**, with reports typically published by the end of March of the
following year.

### 2.2 Document Processing Pipeline

```
Raw PDF → Staging → Markdown → Chunking → Embedding → Inference
```

| Stage | Description |
|-------|-------------|
| **Raw PDF** | Original annual report PDFs downloaded from Vietstock. Files named in Vietnamese (e.g., "Báo cáo thường niên năm 2023.pdf"). |
| **Staging** | Recognized PDFs copied with standardized names (`annual_report_<YEAR>.pdf`) for consistent processing. |
| **Markdown** | PDFs converted to structured markdown using **marker-pdf** with forced OCR (`--force_ocr`) to handle scanned documents. |
| **Chunking** | Markdown text split into overlapping chunks of **512 tokens** with **128-token overlap** to preserve context across chunk boundaries. |
| **Embedding** | Each chunk embedded using OpenAI's **text-embedding-3-small** model (1,536 dimensions) and stored in a DuckDB vector store. |

### 2.3 File Recognition

Raw files are classified using pattern matching on Vietnamese filenames:

- **Recognized**: Matches the standard annual report naming convention, with year extracted.
- **Corrected/Revised**: Files containing "điều chỉnh" (adjusted) in the filename take priority over originals for the same ticker-year pair.
- **Multi-part / Unrecognized**: Non-standard filenames are flagged for manual review.

Supported file formats: PDF and RAR archives.

---

## 3. Framework 1: Environmental Disclosure Checklist (EDC)

### 3.1 Overview

The Environmental Disclosure Checklist evaluates whether a company's annual
report discloses specific climate- and environment-related information. Each of
the **18 checklist items** is scored as a **binary 0 or 1**.

### 3.2 Checklist Items

The 18 items are organized into five thematic groups:

#### Climate Change (CC)

| Code | Item | Criteria (Vietnamese) |
|------|------|-----------------------|
| CC1 | Assessment of climate-related risks and opportunities | Đánh giá các rủi ro (các quy định, tác động vật lý hoặc các tác động chung) liên quan đến biến đổi khí hậu và các hành động đã hoặc sẽ thực hiện để quản lý rủi ro. |
| CC2 | Financial implications of climate change | Đánh giá các tác động tài chính hiện tại (và tương lai), tác động kinh doanh và cơ hội của biến đổi khí hậu. |

#### GHG Emissions (GHG)

| Code | Item | Criteria (Vietnamese) |
|------|------|-----------------------|
| GHG1 | Methodology for GHG emission calculation | Mô tả các phương pháp sử dụng để tính toán khí thải nhà kính. |
| GHG2 | External verification of GHG emissions | Có sự xác nhận bởi yếu tố bên ngoài về lượng phát thải khí nhà kính hay không? – nếu có bởi ai và trên cơ sở gì. |
| GHG3 | Total GHG emissions disclosed | Lượng phát thải khí nhà kính tính bằng đơn vị MtCO2e (hệ mét tấn CO2 thải ra). |
| GHG4 | Disclosure of emissions by scope (Scope 1, 2, 3) | Việc công bố liên quan đến Phạm vi 1, Phạm vi 2, Phạm vi 3 và liên quan trực tiếp đến phát thải khí nhà kính. |
| GHG5 | Disclosure of emissions by source | Công bố phát thải nhà kính dựa trên nguồn phát thải nào? |
| GHG6 | Disclosure of emissions by facility or segment | Công bố phát thải khí nhà kính dựa trên cơ sở vật chất hoặc cấp độ nào? |
| GHG7 | Historical comparison of emissions over time | So sánh lượng phát thải khí nhà kính trong năm hiện tại và năm trước. |

#### Energy Consumption (EC)

| Code | Item | Criteria (Vietnamese) |
|------|------|-----------------------|
| EC1 | Total energy consumed | Lượng năng lượng tiêu thụ. |
| EC2 | Disclosure of energy consumption from renewable sources | Lượng năng lượng tiêu thụ có nguồn gốc từ nguồn năng lượng tái tạo. |
| EC3 | Disclosure of energy consumption by type, facility, or segment | Tiết lộ dựa trên loại khí thải, cơ sở vật chất hoặc cấp độ nào? |

#### GHG Reduction (RC)

| Code | Item | Criteria (Vietnamese) |
|------|------|-----------------------|
| RC1 | Plans or strategies to reduce GHG emissions | Giải thích chi tiết về chiến lược và kế hoạch giảm thiểu phát thải khí nhà kính. |
| RC2 | Specific targets for GHG emission reduction | Mục tiêu cụ thể về lượng giảm thiểu phát thải nhà kính. |
| RC3 | Reductions achieved to date | Lượng giảm phát thải tối đa và các chi phí hoặc tiết kiệm liên quan đến giảm phát khí thải nhà kính tính đến thời điểm báo cáo. |
| RC4 | Costs of future emissions factored into capital expenditure planning | Mức độ chi phí liên quan đến phát thải khí nhà kính trong tương lai vì chi phí này được bao gồm trong kế hoạch sử dụng vốn của công ty. |

#### GHG Accountability (ACC)

| Code | Item | Criteria (Vietnamese) |
|------|------|-----------------------|
| ACC1 | Responsibility for climate change policy and action | Giải thích ai là Ủy ban hoặc Giám đốc chịu trách nhiệm về các chính sách liên quan đến biến đổi khí hậu. |
| ACC2 | Consideration of climate-related goals in executive compensation | Giải thích cơ chế xem xét khi đạt được mục tiêu công ty liên quan đến biến đổi khí hậu bởi Ủy ban hoặc Hội đồng quản trị. |

### 3.3 Scoring Process

For each checklist item and each company-year annual report:

**Step 1 — Semantic Retrieval.** The item's Vietnamese description is used as a
query against the company's embedded annual report. The top-*k* most relevant
text chunks are retrieved via cosine similarity search over the vector store
(default *k* = 5).

**Step 2 — LLM Evaluation.** The retrieved chunks are concatenated and sent to
the LLM (default: `gpt-4.1-mini`) with the following prompt:

> You are a helpful assistant designed to validate the content of sections in an
> annual report. The content to validate is related to climate change. You will
> be given a single row of Vietnamese text, and your task is to determine
> whether the data matches the provided criteria.
>
> - Translate the text to English.
>
> **Return only a JSON object** with the following two properties:
>
> - `"is_valid"`: a boolean (`true` or `false`) indicating whether the text
>   matches the criteria.
> - `"reason"`: Provide a brief explanation on why the text data is valid or not.

**Step 3 — Binary Scoring.** Each item receives:

- **1** if `is_valid = true` (the report contains sufficient evidence).
- **0** if `is_valid = false` (no sufficient evidence found).

### 3.4 Aggregate Disclosure Score

The **total disclosure score** for a company–year is the sum of all 18 item
scores, ranging from **0** (no disclosure) to **18** (full disclosure).

Group-level sub-scores can also be computed:

| Group | Max Score |
|-------|-----------|
| Climate Change (CC1–CC2) | 2 |
| GHG Emissions (GHG1–GHG7) | 7 |
| Energy Consumption (EC1–EC3) | 3 |
| GHG Reduction (RC1–RC4) | 4 |
| GHG Accountability (ACC1–ACC2) | 2 |
| **Total** | **18** |

---

## 4. Framework 2: PROPER-VN Environmental Rating

### 4.1 Overview

PROPER-VN (Program for Pollution Control, Evaluation, and Rating — Vietnam
Adaptation) assigns a **color rating** to each company based on a two-stage
assessment of environmental performance extracted from annual reports. It is
adapted from Indonesia's PROPER system for applicability to Vietnamese
corporate disclosures.

### 4.2 Color Rating Scale

| Color | Symbol | Meaning |
|-------|--------|---------|
| **Gold** | 🥇 | Compliant with strong beyond-compliance environmental engagement |
| **Green** | 🟢 | Compliant with moderate beyond-compliance engagement |
| **Blue** | 🔵 | Compliant with regulations, low beyond-compliance engagement |
| **Red** | 🔴 | Minor regulatory non-compliance without stated compliance |
| **Black** | ⚫ | Serious environmental violations detected |

### 4.3 Stage 1: Regulatory Compliance Assessment

Stage 1 determines whether the company has violated environmental regulations.
Three indicators are evaluated:

| Code | Indicator | Criteria (Vietnamese) | If Present → |
|------|-----------|----------------------|--------------|
| S1_VIOLATION | Serious environmental violations | Có bằng chứng hoặc đề cập đến vi phạm nghiêm trọng về môi trường, bị xử phạt hành chính nặng, gây ô nhiễm, hoặc bị đình chỉ hoạt động vì lý do môi trường. | **Black** (stop) |
| S1_MINOR_NC | Minor regulatory non-compliance | Có đề cập đến vi phạm nhỏ về quy định môi trường, chưa tuân thủ đầy đủ các tiêu chuẩn phát thải, xử lý chất thải hoặc quản lý môi trường theo quy định hiện hành. | **Red** (only if S1_COMPLIANCE absent) |
| S1_COMPLIANCE | Stated regulatory compliance | Công ty tuyên bố tuân thủ các quy định về môi trường, có giấy phép môi trường, đánh giá tác động môi trường (ĐTM/EIA), hoặc báo cáo tuân thủ quy chuẩn kỹ thuật môi trường. | Proceed to Stage 2 |

**Decision logic:**

```
IF S1_VIOLATION is present       → Black (stop)
IF S1_MINOR_NC and NOT S1_COMPLIANCE → Red (stop)
OTHERWISE                        → proceed to Stage 2
```

### 4.4 Stage 2: Beyond-Compliance Scoring

Stage 2 evaluates voluntary environmental initiatives. Each indicator is scored
on **evidence level**:

| Evidence Level | Score | Description |
|----------------|-------|-------------|
| `none` | 0 | No relevant information found |
| `basic_mention` | 1 | Topic is mentioned but without specific data |
| `quantified` | 2 | Specific numbers, targets, or detailed evidence provided |

**Stage 2 Indicators (4 items):**

| Code | Indicator | Criteria (Vietnamese) |
|------|-----------|----------------------|
| S2_ISO14001 | ISO 14001 certification or equivalent EMS | Công ty có chứng nhận ISO 14001 hoặc hệ thống quản lý môi trường tương đương (EMS). Bao gồm việc đề cập đến việc đạt, duy trì hoặc tái chứng nhận ISO 14001. |
| S2_CARBON_DISC | Carbon/GHG emission disclosure | Công ty công bố lượng phát thải carbon hoặc khí nhà kính (GHG) cụ thể, bao gồm Scope 1, Scope 2, hoặc Scope 3, hoặc tổng phát thải tính bằng tấn CO2 tương đương. |
| S2_REDUCTION | Emission reduction targets or achievements | Công ty có đặt mục tiêu giảm phát thải khí nhà kính cụ thể (ví dụ: giảm X% vào năm Y), hoặc đã đạt được kết quả giảm phát thải so với năm gốc, hoặc cam kết Net Zero/Carbon Neutral. |
| S2_EFFICIENCY | Energy or resource efficiency initiatives | Công ty thực hiện các sáng kiến tiết kiệm năng lượng, sử dụng năng lượng tái tạo, tái chế tài nguyên, giảm tiêu thụ nước, hoặc các chương trình cải thiện hiệu suất sử dụng tài nguyên. |

**Maximum Stage 2 score:** 4 indicators × 2 points = **8 points**

### 4.5 Color Classification Thresholds

| Color | Condition |
|-------|-----------|
| **Gold** | S2 score ≥ 75% of max (≥ 6 out of 8) |
| **Green** | S2 score ≥ 37.5% of max (≥ 3 out of 8) |
| **Blue** | S2 score < 37.5% of max (< 3 out of 8) |

### 4.6 Evaluation Process

For each PROPER-VN indicator and each company-year annual report:

**Step 1 — Semantic Retrieval.** Same as EDC: top-*k* chunks retrieved via
cosine similarity from the vector store (default *k* = 5).

**Step 2 — LLM Evaluation.** The retrieved chunks are sent to the LLM with the
following prompt:

> You are a helpful assistant designed to evaluate annual reports against
> PROPER-VN environmental indicators. You will be given Vietnamese text from an
> annual report and a specific environmental indicator to check.
>
> - Translate the text to English.
>
> **Return only a JSON object** with the following properties:
>
> - `"is_present"`: a boolean (`true` or `false`) indicating whether evidence
>   for this indicator is found in the text.
> - `"evidence_level"`: one of `"none"`, `"basic_mention"`, or `"quantified"`.
>   - `"none"` — no relevant information found.
>   - `"basic_mention"` — the topic is mentioned but without specific data.
>   - `"quantified"` — specific numbers, targets, or detailed evidence provided.
> - `"reason"`: A brief explanation supporting your assessment.

**Step 3 — Stage 1 Check.** Apply the violation/compliance decision logic.

**Step 4 — Stage 2 Scoring.** Sum evidence-level scores for the 4
beyond-compliance indicators.

**Step 5 — Color Assignment.** Apply threshold rules to determine the final
color rating.

---

## 5. Technical Implementation Details

### 5.1 PDF-to-Markdown Conversion (marker-pdf OCR)

Annual report PDFs are converted to structured markdown using
**[marker-pdf](https://github.com/VikParuchuri/marker)**, invoked via the
`marker_single` CLI. Each PDF is processed as an individual job.

**Key setting:** `--force_ocr` is enabled on all files, meaning every page is
re-OCR'd regardless of whether the PDF already contains an extractable text
layer. This ensures consistent output quality across natively-digital PDFs and
scanned documents.

#### Marker OCR Batch-Size Configuration

Marker internally runs multiple deep-learning models in sequence. The batch
sizes below control GPU/CPU memory usage and throughput at each stage:

| Flag | Value | Processing Stage |
|------|-------|------------------|
| `--force_ocr` | `True` | Force OCR on all pages |
| `--layout_batch_size` | 12 | **LayoutBuilder** — page-level layout analysis |
| `--detection_batch_size` | 8 | **LineBuilder / TableProcessor** — text-line and table detection |
| `--ocr_error_batch_size` | 12 | **LineBuilder** — OCR error correction |
| `--recognition_batch_size` | 32 | **OcrBuilder / TableProcessor** — character recognition |
| `--equation_batch_size` | 16 | **EquationProcessor** — mathematical formula recognition |
| `--table_rec_batch_size` | 12 | **TableProcessor** — table structure recognition |

The output is a markdown file per report, preserving headings, paragraphs,
tables, and lists. Output files are organized by ticker in the markdown
directory (e.g., `data/markdown/VNM/annual_report_2023/`).

### 5.2 RAG Pipeline Configuration

The Retrieval-Augmented Generation (RAG) pipeline consists of three stages:
chunking, embedding, and inference. The full parameter set is listed below.

#### 5.2.1 Text Chunking

Markdown documents are split into overlapping chunks using a token-based
sliding window:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Chunk size | 512 tokens | Maximum tokens per chunk |
| Chunk overlap | 128 tokens | Overlapping tokens between consecutive chunks to preserve cross-boundary context |

#### 5.2.2 Embedding

Each text chunk is embedded into a dense vector for similarity search:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model | OpenAI `text-embedding-3-small` | Embedding model |
| Dimensions | 1,536 | Output vector dimensionality |
| Batch size | 512 | Max texts per OpenAI API call |
| Similarity metric | Cosine similarity | Used for nearest-neighbor retrieval |
| Storage | DuckDB (in-database vector store) | Vectors stored alongside metadata |

Alternative model `text-embedding-3-large` (3,072 dimensions) is supported and
can be selected per-run via the dashboard.

#### 5.2.3 LLM Inference

Retrieved chunks are evaluated by a large language model:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Default LLM | `gpt-4.1-mini` | Primary evaluation model |
| Top-*k* retrieval | 5 chunks per checklist item | Number of most-similar chunks retrieved per query |
| Temperature | 0 | Deterministic output for reproducibility |
| Reasoning models (o-series, GPT-5) | `reasoning_effort = "low"` | Replaces temperature for reasoning-class models |
| Response format | Structured JSON | Enforced via prompt instructions |

The model and top-*k* values can be changed per-run through the dashboard
sidebar, allowing comparison across different configurations.

### 5.3 Model-Keyed Results

All scores are keyed by `(ticker, year, model)`. Running different LLM models
does not overwrite previous results, enabling cross-model comparison and
ensemble approaches.

### 5.4 Database Output

Results are stored in DuckDB with both long and wide format views:

**EDC Results:**

| View | Format | Description |
|------|--------|-------------|
| `disclosure_scores` | Long | One row per (ticker, year, model, category_code) with binary `score` |
| `disclosure_scores_wide` | Wide | One row per (ticker, year, model) with a column per checklist code + `total_score` |

**PROPER-VN Results:**

| View | Format | Description |
|------|--------|-------------|
| `proper_vn_scores` | Long | One row per (ticker, year, model, indicator_code) with `score` and `evidence_level` |
| `proper_vn_scores_wide` | Wide | One row per (ticker, year, model) with a column per indicator + `color`, `s2_score`, `s2_max_score` |

---

## 6. Scope and Limitations

- **Language**: Annual reports are in Vietnamese. The LLM is instructed to
  translate text to English before evaluation, which may introduce translation
  artifacts.
- **OCR quality**: Reports processed with forced OCR may contain recognition
  errors, particularly in tables and numerical data.
- **Report availability**: Not all companies publish annual reports for all
  fiscal years, leading to missing data points.
- **Self-reported data**: All evaluated disclosures are based on self-reported
  information in annual reports; independent verification is not performed by
  this system (except where GHG2 checks for external verification statements).
- **Binary simplification (EDC)**: The 0/1 scoring does not capture the depth
  or quality of disclosure — only its presence or absence.
- **Threshold sensitivity (PROPER-VN)**: The Green/Gold boundary (37.5% / 75%)
  is based on the adapted framework and may require calibration for the
  Vietnamese context.
