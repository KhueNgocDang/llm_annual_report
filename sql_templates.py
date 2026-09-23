"""Centralized SQL templates for Data Studio."""

from __future__ import annotations

DEFAULT_DATA_STUDIO_SQL_TEMPLATE = "Governance Board Metrics"

DATA_STUDIO_SQL_TEMPLATES: dict[str, str] = {
    "Governance Board Metrics": """
WITH base AS (
    SELECT DISTINCT ticker, year, model
    FROM governance_results_hyde2
),
dir AS (
    SELECT ticker, year, model, value_json, details_json
    FROM governance_results_hyde2
    WHERE item_code = 'GOV_DIRECTORY'
),
exec AS (
    SELECT ticker, year, model, value_json, details_json
    FROM governance_results_hyde2
    WHERE item_code = 'GOV_EXECUTIVE'
),
sup AS (
    SELECT ticker, year, model, value_json, details_json
    FROM governance_results_hyde2
    WHERE item_code = 'GOV_SUPERVISORY'
),
aud AS (
    SELECT ticker, year, model, value_json
    FROM financial_statement_audit_results
)
SELECT
    b.ticker,
    b.year,
    b.model,
    COALESCE((
        SELECT COUNT(*)
        FROM json_each(COALESCE(d.details_json, '[]')) jd
        WHERE lower(trim(COALESCE(json_extract_string(jd.value, '$.gender'), ''))) IN ('female', 'nữ', 'nu')
    ), 0) AS directory_board_female_members,
    COALESCE((
        SELECT COUNT(*)
        FROM json_each(COALESCE(e.details_json, '[]')) je
        WHERE lower(trim(COALESCE(json_extract_string(je.value, '$.gender'), ''))) IN ('female', 'nữ', 'nu')
    ), 0) AS executive_board_female_members,
    COALESCE((
        SELECT COUNT(*)
        FROM json_each(COALESCE(d.details_json, '[]')) jd2
        WHERE lower(trim(COALESCE(json_extract_string(jd2.value, '$.is_independent'), ''))) IN ('true', '1', 'yes', 'y', 'co', 'có')
    ), 0) AS directory_board_independent_members,
    COALESCE(json_array_length(COALESCE(d.details_json, '[]')), 0) AS directory_board_size,
    COALESCE(json_array_length(COALESCE(e.details_json, '[]')), 0) AS executive_board_size,
    COALESCE(json_array_length(COALESCE(s.details_json, '[]')), 0) AS supervisory_board_committee_size,
    COALESCE(
        json_extract_string(a.value_json, '$.external_audit_firm'),
        json_extract_string(a.value_json, '$.external_audit_firm_en')
    ) AS audit_firm
FROM base b
LEFT JOIN dir d ON d.ticker = b.ticker AND d.year = b.year AND d.model = b.model
LEFT JOIN exec e ON e.ticker = b.ticker AND e.year = b.year AND e.model = b.model
LEFT JOIN sup s ON s.ticker = b.ticker AND s.year = b.year AND s.model = b.model
LEFT JOIN aud a ON a.ticker = b.ticker AND a.year = b.year AND a.model = b.model
ORDER BY b.ticker, b.year DESC, b.model
LIMIT 200
""".strip(),
    "Latest Governance Results": """
SELECT ticker, year, item_code, model, created_at
FROM governance_results_hyde2
ORDER BY created_at DESC
LIMIT 100
""".strip(),
    "Governance Item Coverage": """
SELECT item_code, COUNT(*) AS rows_count, COUNT(DISTINCT ticker) AS ticker_count
FROM governance_results_hyde2
GROUP BY item_code
ORDER BY rows_count DESC
""".strip(),
    "Firm Metrics (SIZE/ROA/LOA)": """
WITH base AS (
    SELECT
        code AS ticker,
        TRY_CAST(SUBSTR(report_date, 1, 4) AS INTEGER) AS year,
        report_date,
        ratio_code,
        value
    FROM financial_ratios
    WHERE ratio_code IN (
        'TOTAL_ASSETS_AQ',
        'ROAA_TR_AVG5Q',
        'LOANS_TO_ASSET_AQ'
    )
      AND value IS NOT NULL
),
latest_per_ratio AS (
    SELECT
        ticker,
        year,
        ratio_code,
        value
    FROM base
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY ticker, year, ratio_code
        ORDER BY report_date DESC
    ) = 1
),
pivoted AS (
    SELECT
        ticker,
        year,
        MAX(CASE WHEN ratio_code = 'TOTAL_ASSETS_AQ' THEN value END) AS total_assets,
        MAX(CASE WHEN ratio_code = 'ROAA_TR_AVG5Q' THEN value END) AS roa,
        MAX(CASE WHEN ratio_code = 'LOANS_TO_ASSET_AQ' THEN value END) AS loa
    FROM latest_per_ratio
    GROUP BY ticker, year
)
SELECT
    ticker,
    year,
    LN(NULLIF(total_assets, 0)) AS size,
    roa,
    loa,
    total_assets
FROM pivoted
ORDER BY ticker, year DESC
""".strip(),
    "Board/Governance Profile (EXE/DIR/AGE/ISO/WOMAN/SUPERVISOR)": """
WITH base AS (
    SELECT DISTINCT ticker, year, model
    FROM governance_results_hyde2
),
dir AS (
    SELECT ticker, year, model, details_json
    FROM governance_results_hyde2
    WHERE item_code = 'GOV_DIRECTORY'
),
exe AS (
    SELECT ticker, year, model, details_json
    FROM governance_results_hyde2
    WHERE item_code = 'GOV_EXECUTIVE'
),
sup AS (
    SELECT ticker, year, model, details_json
    FROM governance_results_hyde2
    WHERE item_code = 'GOV_SUPERVISORY'
),
iso AS (
    SELECT ticker, year, model, MAX(CASE WHEN is_present THEN 1 ELSE 0 END) AS iso_flag
    FROM proper_vn_results_hyde2
    WHERE indicator_code = 'S2_ISO14001'
    GROUP BY ticker, year, model
)
SELECT
    b.ticker,
    b.year,
    b.model,
    COALESCE(json_array_length(COALESCE(e.details_json, '[]')), 0) AS EXE_BOARDSIZE,
    COALESCE(json_array_length(COALESCE(d.details_json, '[]')), 0) AS DIR_BOARDSIZE,
    CASE
        WHEN ch.first_event_year IS NOT NULL THEN b.year - ch.first_event_year
        ELSE NULL
    END AS AGE,
    COALESCE(i.iso_flag, 0) AS ISO,
    COALESCE((
        SELECT COUNT(*)
        FROM json_each(COALESCE(d.details_json, '[]')) jd
        WHERE lower(trim(COALESCE(json_extract_string(jd.value, '$.gender'), ''))) IN ('female', 'nữ', 'nu')
    ), 0) AS WOMAN,
    COALESCE(json_array_length(COALESCE(s.details_json, '[]')), 0) AS SUPERVISOR,
    COALESCE((
        SELECT COUNT(*)
        FROM json_each(COALESCE(s.details_json, '[]')) js
        WHERE lower(trim(COALESCE(json_extract_string(js.value, '$.gender'), ''))) IN ('female', 'nữ', 'nu')
    ), 0) AS SUPERVISOR_WOMAN
FROM base b
LEFT JOIN dir d ON d.ticker = b.ticker AND d.year = b.year AND d.model = b.model
LEFT JOIN exe e ON e.ticker = b.ticker AND e.year = b.year AND e.model = b.model
LEFT JOIN sup s ON s.ticker = b.ticker AND s.year = b.year AND s.model = b.model
LEFT JOIN iso i ON i.ticker = b.ticker AND i.year = b.year AND i.model = b.model
LEFT JOIN company_history_summary ch ON ch.ticker = b.ticker
ORDER BY b.ticker, b.year DESC, b.model
LIMIT 500
""".strip(),
    "Carbon Disclosure + PROPER-VN": """
WITH proper AS (
    SELECT
        ticker,
        year,
        model,
        MAX(CASE WHEN indicator_code = 'S2_CARBON_DISC' AND is_present THEN 1 ELSE 0 END) AS proper_carbon_disclosure,
        MAX(CASE WHEN indicator_code = 'S2_REDUCTION' AND is_present THEN 1 ELSE 0 END) AS proper_emission_reduction,
        MAX(CASE WHEN indicator_code = 'S2_EFFICIENCY' AND is_present THEN 1 ELSE 0 END) AS proper_resource_efficiency,
        MAX(CASE WHEN indicator_code = 'S2_ISO14001' AND is_present THEN 1 ELSE 0 END) AS proper_iso14001,
        MAX(CASE WHEN indicator_code = 'S1_COMPLIANCE' AND is_present THEN 1 ELSE 0 END) AS proper_env_compliance,
        MAX(CASE WHEN indicator_code = 'S1_MINOR_NC' AND is_present THEN 1 ELSE 0 END) AS proper_minor_non_compliance,
        MAX(CASE WHEN indicator_code = 'S1_VIOLATION' AND is_present THEN 1 ELSE 0 END) AS proper_env_violation,
        SUM(CASE WHEN is_present THEN 1 ELSE 0 END) AS proper_present_count,
        COUNT(*) AS proper_total_indicators
    FROM proper_vn_results_hyde2
    GROUP BY ticker, year, model
),
edc_ghg AS (
    SELECT
        ticker,
        year,
        model,
        SUM(CASE WHEN is_valid THEN 1 ELSE 0 END) AS ghg_valid_count,
        COUNT(*) AS ghg_total_count,
        MAX(CASE WHEN category_code = 'GHG1' AND is_valid THEN 1 ELSE 0 END) AS ghg1_methodology,
        MAX(CASE WHEN category_code = 'GHG2' AND is_valid THEN 1 ELSE 0 END) AS ghg2_external_verification,
        MAX(CASE WHEN category_code = 'GHG3' AND is_valid THEN 1 ELSE 0 END) AS ghg3_total_emissions,
        MAX(CASE WHEN category_code = 'GHG4' AND is_valid THEN 1 ELSE 0 END) AS ghg4_scope_disclosure,
        MAX(CASE WHEN category_code = 'GHG5' AND is_valid THEN 1 ELSE 0 END) AS ghg5_by_source,
        MAX(CASE WHEN category_code = 'GHG6' AND is_valid THEN 1 ELSE 0 END) AS ghg6_by_facility,
        MAX(CASE WHEN category_code = 'GHG7' AND is_valid THEN 1 ELSE 0 END) AS ghg7_historical_comparison
    FROM inference_results_hyde2
    WHERE category_code IN ('GHG1', 'GHG2', 'GHG3', 'GHG4', 'GHG5', 'GHG6', 'GHG7')
    GROUP BY ticker, year, model
),
base AS (
    SELECT ticker, year, model FROM proper
    UNION
    SELECT ticker, year, model FROM edc_ghg
)
SELECT
    b.ticker,
    b.year,
    b.model,
    COALESCE(p.proper_carbon_disclosure, 0) AS proper_carbon_disclosure,
    COALESCE(p.proper_emission_reduction, 0) AS proper_emission_reduction,
    COALESCE(p.proper_resource_efficiency, 0) AS proper_resource_efficiency,
    COALESCE(p.proper_iso14001, 0) AS proper_iso14001,
    COALESCE(p.proper_env_compliance, 0) AS proper_env_compliance,
    COALESCE(p.proper_minor_non_compliance, 0) AS proper_minor_non_compliance,
    COALESCE(p.proper_env_violation, 0) AS proper_env_violation,
    COALESCE(p.proper_present_count, 0) AS proper_present_count,
    COALESCE(p.proper_total_indicators, 0) AS proper_total_indicators,
    COALESCE(g.ghg_valid_count, 0) AS edc_ghg_valid_count,
    COALESCE(g.ghg_total_count, 0) AS edc_ghg_total_count,
    COALESCE(g.ghg1_methodology, 0) AS ghg1_methodology,
    COALESCE(g.ghg2_external_verification, 0) AS ghg2_external_verification,
    COALESCE(g.ghg3_total_emissions, 0) AS ghg3_total_emissions,
    COALESCE(g.ghg4_scope_disclosure, 0) AS ghg4_scope_disclosure,
    COALESCE(g.ghg5_by_source, 0) AS ghg5_by_source,
    COALESCE(g.ghg6_by_facility, 0) AS ghg6_by_facility,
    COALESCE(g.ghg7_historical_comparison, 0) AS ghg7_historical_comparison
FROM base b
LEFT JOIN proper p ON p.ticker = b.ticker AND p.year = b.year AND p.model = b.model
LEFT JOIN edc_ghg g ON g.ticker = b.ticker AND g.year = b.year AND g.model = b.model
ORDER BY b.ticker, b.year DESC, b.model
LIMIT 500
""".strip(),
}
