"""
Central configuration for the Surgical Instrument Classifier.

Loads settings from environment variables (via .env file) and provides
sensible defaults for all pipeline parameters.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
RAW_DIR = DATASET_DIR / "raw"
PROCESSED_DIR = DATASET_DIR / "processed"
UPLOADS_DIR = BASE_DIR / "uploads"
METADATA_CSV = DATASET_DIR / "metadata.csv"
CLASS_MAPPING_JSON = DATASET_DIR / "class_mapping.json"
MODEL_WEIGHTS = DATASET_DIR / "classifier_resnet50.pt"
ENV_FILE = BASE_DIR / ".env"

# Ensure critical directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# AI Configuration (Gemini / Groq)
# ---------------------------------------------------------------------------
def _load_env():
    """Load key=value pairs from .env file into os.environ (simple parser)."""
    if ENV_FILE.exists():
        with open(ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

_load_env()

# Groq settings
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

# ---------------------------------------------------------------------------
# Pipeline Settings
# ---------------------------------------------------------------------------
TRAINING_EPOCHS = int(os.environ.get("TRAINING_EPOCHS", "50"))
MAX_PDF_PAGES = int(os.environ.get("MAX_PDF_PAGES", "100"))

# Image filtering thresholds
MIN_IMAGE_WIDTH = 30   # pixels — ignore tiny decorations
MIN_IMAGE_HEIGHT = 30
MAX_LOGO_DUPLICATES = 3  # if an image hash appears >N times across pages, it's a logo
