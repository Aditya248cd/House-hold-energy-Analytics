"""
config.py — Central configuration for all pipeline scripts.

All paths are derived dynamically from this file's location so the
project works on any machine without editing paths by hand.
MONGO_URI is read from the .env file in the project root — never
hardcode secrets in source files.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project-root directory (one level up from data/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # …/House Hold energy Analytics/

# ---------------------------------------------------------------------------
# Load secrets from .env file (must sit in project root)
# ---------------------------------------------------------------------------
load_dotenv(PROJECT_ROOT / ".env")

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise EnvironmentError(
        "MONGO_URI is not set. Copy .env.example to .env and add your "
        "MongoDB Atlas connection string."
    )

# ---------------------------------------------------------------------------
# Hadoop / Spark environment (Windows-specific, read from .env with fallback)
# ---------------------------------------------------------------------------
HADOOP_HOME = os.getenv("HADOOP_HOME", "C:/hadoop")
os.environ["HADOOP_HOME"] = HADOOP_HOME
os.environ["hadoop.home.dir"] = HADOOP_HOME
os.environ["PATH"] = str(Path(HADOOP_HOME) / "bin") + os.pathsep + os.environ.get("PATH", "")
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# ---------------------------------------------------------------------------
# Data paths — all relative to project root
# ---------------------------------------------------------------------------
DATA_DIR            = PROJECT_ROOT / "data"
RAW_DATA            = DATA_DIR / "household_power_consumption.txt"
CLEANED_PARQUET     = DATA_DIR / "cleaned_parquet"
HOURLY_PARQUET      = DATA_DIR / "hourly_parquet"   # ← reused by step4 & step5

# String versions (Spark read/write needs strings, not Path objects)
RAW_DATA_STR        = str(RAW_DATA)
CLEANED_PARQUET_STR = str(CLEANED_PARQUET)
HOURLY_PARQUET_STR  = str(HOURLY_PARQUET)
