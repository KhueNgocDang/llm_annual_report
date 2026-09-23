# Governance RAG Benchmark Report

- Timestamp: 2026-06-25T13:54:11.051154+00:00
- Benchmark file: benchmarks/rag_governance_benchmark.seed.json
- Model: gpt-4.1-mini
- Top-K: 5
- Run extraction: True

## Summary Metrics

- Cases total: 8
- Retrieval cases evaluated: 8
- Avg Precision@K: 0.5500
- Avg Forbidden-hit rate: 0.0500
- Any-relevant retrieval rate: 0.6250
- Extraction cases evaluated: 4
- Extraction exact-entity-match rate: 0.5

## Case Results

### AAA_2025_GOV_SUPERVISORY_TERMS
- AAA/2025 GOV_SUPERVISORY | Precision@K=1.0000, Forbidden=0.0000

### AAA_2025_GOV_SUPERVISORY_ENTITIES
- AAA/2025 GOV_SUPERVISORY | Precision@K=0.8000, Forbidden=0.0000
- Extraction exact match: True | predicted_count=3 expected_count=3

### AAA_2025_GOV_BOARD_TERMS
- AAA/2025 GOV_BOARD | Precision@K=0.0000, Forbidden=0.0000

### AAA_2025_GOV_BOARD_ENTITIES
- AAA/2025 GOV_BOARD | Precision@K=0.0000, Forbidden=0.0000
- Extraction exact match: True | predicted_count=5 expected_count=5

### AAA_2025_GOV_SHAREHOLDERS_TERMS
- AAA/2025 GOV_SHAREHOLDERS | Precision@K=0.6000, Forbidden=0.0000

### AAA_2025_GOV_SHAREHOLDERS_ENTITY
- AAA/2025 GOV_SHAREHOLDERS | Precision@K=0.0000, Forbidden=0.0000
- Extraction exact match: False | predicted_count=10 expected_count=None
- Missing: ctcp tap doan an phat holdings
- Unexpected: ctcp tap đoan an phat holdings, hoa thi thu ha, nguyen le thang long, nguyen le trung, nguyen thi giang, nguyen thi tien, nguyen xuan co, phan tri nghia, tran thi thoan, van thi lan anh

### AAA_2025_GOV_AUDIT_TERMS
- AAA/2025 GOV_AUDIT | Precision@K=1.0000, Forbidden=0.4000

### AAA_2025_GOV_AUDIT_ENTITY
- AAA/2025 GOV_AUDIT | Precision@K=1.0000, Forbidden=0.0000
- Extraction exact match: False | predicted_count=0 expected_count=None
- Missing: cong ty tnhh ey viet nam
