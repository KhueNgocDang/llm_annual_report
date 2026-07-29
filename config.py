from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"


def ensure_env_loaded() -> None:
    """Load environment variables from local .env once."""
    load_dotenv(ENV_PATH, override=False)


ensure_env_loaded()

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
STAGING_DIR = DATA_DIR / "staging"
OUTPUT_DIR = DATA_DIR / "output"
MARKDOWN_DIR = DATA_DIR / "markdown"
FINANCIAL_STATEMENT_MARKDOWN_DIR = DATA_DIR / "markdown_bctc"
BCTC_MARKDOWN_DIR = FINANCIAL_STATEMENT_MARKDOWN_DIR
LOGS_DIR = DATA_DIR / "logs"

DB_PATH = BASE_DIR / os.getenv("DB_PATH", "db.db")

DEFAULT_START_YEAR = int(os.getenv("DEFAULT_START_YEAR", "2015"))
DEFAULT_END_YEAR = int(os.getenv("DEFAULT_END_YEAR", "2025"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
EMBEDDING_CHUNK_SIZE = int(os.getenv("EMBEDDING_CHUNK_SIZE", "512"))
EMBEDDING_CHUNK_OVERLAP = int(os.getenv("EMBEDDING_CHUNK_OVERLAP", "128"))

INFERENCE_MODEL = os.getenv("INFERENCE_MODEL", "gpt-4.1-mini")
INFERENCE_TOP_K = int(os.getenv("INFERENCE_TOP_K", "20"))
INFERENCE_TEMPERATURE = float(os.getenv("INFERENCE_TEMPERATURE", "0"))

INFERENCE_RETRIEVAL_ALPHA = float(os.getenv("INFERENCE_RETRIEVAL_ALPHA", "1.0"))
INFERENCE_RETRIEVAL_BETA = float(os.getenv("INFERENCE_RETRIEVAL_BETA", "0.12"))
INFERENCE_RETRIEVAL_GAMMA = float(os.getenv("INFERENCE_RETRIEVAL_GAMMA", "0.08"))
INFERENCE_RETRIEVAL_CANDIDATE_MULTIPLIER = int(
    os.getenv("INFERENCE_RETRIEVAL_CANDIDATE_MULTIPLIER", "4")
)


def bootstrap_directories() -> list[Path]:
    """Create all required directories for a fresh environment."""
    created: list[Path] = []
    for path in (
        DATA_DIR,
        RAW_DIR,
        STAGING_DIR,
        OUTPUT_DIR,
        MARKDOWN_DIR,
        FINANCIAL_STATEMENT_MARKDOWN_DIR,
        LOGS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)
    return created
