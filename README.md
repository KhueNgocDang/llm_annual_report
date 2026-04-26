# Annual Report Pipeline

Vietnamese stock data & annual report pipeline with NiceGUI web UI.

## Database Schema

`companies` is the central control table. All pipeline operations are scoped to tickers in this table. Deleting a company cascades to all related data.

```mermaid
erDiagram
    companies {
        VARCHAR ticker PK
    }

    stocks {
        VARCHAR code PK
        VARCHAR type
        VARCHAR floor
        VARCHAR status
        VARCHAR company_name
        VARCHAR company_name_eng
        VARCHAR short_name
        VARCHAR listed_date
        VARCHAR delisted_date
        VARCHAR company_id
        VARCHAR tax_code
        VARCHAR isin
    }

    financial_models {
        VARCHAR model_type
        VARCHAR item_code
        VARCHAR model_type_name
        VARCHAR model_vn_desc
        VARCHAR model_en_desc
        VARCHAR company_form
        VARCHAR note
        VARCHAR code_list
        VARCHAR item_vn_name
        VARCHAR item_en_name
        INTEGER display_order
        INTEGER display_level
        VARCHAR form_type
    }

    financial_statements {
        VARCHAR code
        VARCHAR item_code
        VARCHAR report_type
        VARCHAR model_type
        DOUBLE numeric_value
        VARCHAR fiscal_date
        VARCHAR created_date
        VARCHAR modified_date
    }

    financial_ratios {
        VARCHAR code
        VARCHAR ratio_group
        VARCHAR report_date
        VARCHAR item_code
        VARCHAR ratio_code
        VARCHAR item_name
        DOUBLE value
    }

    vietstock_documents {
        BIGINT id PK
        VARCHAR ticker
        VARCHAR doc_type
        VARCHAR title
        VARCHAR full_name
        VARCHAR source
        VARCHAR published_date
        VARCHAR file_url
        BIGINT file_info_id
        BOOLEAN synced_to_raw
        VARCHAR raw_path
    }

    conversion_jobs {
        INTEGER id PK
        VARCHAR ticker
        INTEGER year
        INTEGER start_year
        INTEGER end_year
        VARCHAR source_path
        VARCHAR output_dir
        VARCHAR status
        VARCHAR command
        VARCHAR log_path
        INTEGER pid
        VARCHAR error_message
        VARCHAR failed_step
        TIMESTAMP started_at
        TIMESTAMP completed_at
        TIMESTAMP created_at
    }

    annual_reports {
        VARCHAR ticker PK
        INTEGER year PK
        VARCHAR content
        VARCHAR source_file
        TIMESTAMP created_at
    }

    pipeline_files {
        VARCHAR ticker PK
        INTEGER year PK
        VARCHAR stage PK
        VARCHAR file_path
        TIMESTAMP created_at
    }

    companies ||--o{ financial_statements : "ticker → code"
    companies ||--o{ financial_ratios : "ticker → code"
    companies ||--o{ vietstock_documents : "ticker"
    companies ||--o{ conversion_jobs : "ticker"
    companies ||--o{ annual_reports : "ticker"
    companies ||--o{ pipeline_files : "ticker"
    financial_models ||--o{ financial_statements : "item_code"
```
