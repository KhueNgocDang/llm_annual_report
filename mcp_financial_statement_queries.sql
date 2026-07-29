-- MCP SQL templates for financial statement analysis
-- Use with DuckDB MCP execute_query on db.db

-- 1) Discover likely item codes by keyword (example: total assets)
WITH model_names AS (
    SELECT
        item_code,
        MAX(NULLIF(item_vn_name, '')) AS item_vn_name,
        MAX(NULLIF(item_en_name, '')) AS item_en_name
    FROM financial_models
    GROUP BY item_code
)
SELECT
    item_code,
    item_vn_name,
    item_en_name
FROM model_names
WHERE LOWER(COALESCE(item_vn_name, '') || ' ' || COALESCE(item_en_name, '') || ' ' || item_code)
      LIKE '%tong tai san%'
   OR LOWER(COALESCE(item_vn_name, '') || ' ' || COALESCE(item_en_name, '') || ' ' || item_code)
      LIKE '%total asset%'
ORDER BY item_code;

-- 2) Trend for one ticker + one item code
SELECT
    code AS ticker,
    item_code,
    TRY_CAST(SUBSTR(fiscal_date, 1, 4) AS INTEGER) AS year,
    fiscal_date,
    numeric_value
FROM financial_statements
WHERE code = 'VNM'
  AND item_code = 'BS_270'
  AND numeric_value IS NOT NULL
ORDER BY year;

-- 3) Multi-item trend for one ticker using keyword matching
WITH model_names AS (
    SELECT
        item_code,
        MAX(NULLIF(item_vn_name, '')) AS item_vn_name,
        MAX(NULLIF(item_en_name, '')) AS item_en_name
    FROM financial_models
    GROUP BY item_code
)
SELECT
    fs.code AS ticker,
    fs.item_code,
    COALESCE(mn.item_en_name, mn.item_vn_name, fs.item_code) AS item_name,
    TRY_CAST(SUBSTR(fs.fiscal_date, 1, 4) AS INTEGER) AS year,
    fs.numeric_value
FROM financial_statements fs
LEFT JOIN model_names mn ON mn.item_code = fs.item_code
WHERE fs.code = 'VNM'
  AND fs.numeric_value IS NOT NULL
  AND LOWER(COALESCE(mn.item_vn_name, '') || ' ' || COALESCE(mn.item_en_name, '') || ' ' || COALESCE(fs.item_code, ''))
      LIKE '%tai san%'
ORDER BY year, fs.item_code;

-- 4) Latest snapshot for key balance-sheet metrics by ticker
WITH model_names AS (
    SELECT
        item_code,
        MAX(NULLIF(item_vn_name, '')) AS item_vn_name,
        MAX(NULLIF(item_en_name, '')) AS item_en_name
    FROM financial_models
    GROUP BY item_code
),
base AS (
    SELECT
        fs.code AS ticker,
        fs.item_code,
        COALESCE(mn.item_en_name, mn.item_vn_name, fs.item_code) AS item_name,
        TRY_CAST(SUBSTR(fs.fiscal_date, 1, 4) AS INTEGER) AS year,
        fs.numeric_value
    FROM financial_statements fs
    LEFT JOIN model_names mn ON mn.item_code = fs.item_code
    WHERE fs.numeric_value IS NOT NULL
      AND fs.code = 'VNM'
      AND (
          LOWER(COALESCE(mn.item_vn_name, '') || ' ' || COALESCE(mn.item_en_name, '') || ' ' || COALESCE(fs.item_code, '')) LIKE '%tai san%'
          OR LOWER(COALESCE(mn.item_vn_name, '') || ' ' || COALESCE(mn.item_en_name, '') || ' ' || COALESCE(fs.item_code, '')) LIKE '%no phai tra%'
          OR LOWER(COALESCE(mn.item_vn_name, '') || ' ' || COALESCE(mn.item_en_name, '') || ' ' || COALESCE(fs.item_code, '')) LIKE '%von chu so huu%'
      )
),
latest AS (
    SELECT
        ticker,
        item_code,
        item_name,
        ARG_MAX(numeric_value, year) AS latest_value,
        MAX(year) AS latest_year
    FROM base
    GROUP BY ticker, item_code, item_name
)
SELECT *
FROM latest
ORDER BY latest_value DESC;
