from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "output"
DB_PATH = BASE_DIR / "db.db"

DEFAULT_START_YEAR = 2015
DEFAULT_END_YEAR = 2025
