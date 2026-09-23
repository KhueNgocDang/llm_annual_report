"""
Central configuration for the Environment Annual Report pipeline.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
STAGING_DIR = DATA_DIR / "staging"
MARKDOWN_DIR = DATA_DIR / "markdown"
LOGS_DIR = DATA_DIR / "logs"

# ---------------------------------------------------------------------------
# marker_single CLI settings
# ---------------------------------------------------------------------------
# Extra arguments passed to every marker_single invocation.
# Keys are CLI flag names (without leading "--"); values are their arguments.
# Use True for boolean flags that take no value (e.g. "force_ocr": True).
MARKER_EXTRA_ARGS: dict[str, str | int | bool] = {
    "force_ocr": True,
    # --- Batch sizes (tuned for ~8GB VRAM GPU) ---
    "layout_batch_size": 8,
    "detection_batch_size": 4,
    "ocr_error_batch_size": 4,
    "recognition_batch_size": 16,
    "equation_batch_size": 4,
    "table_rec_batch_size": 4,
}
