"""
check.py — Connectivity and data verification script.
Reads credentials from .env via config.py (no hardcoded secrets).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

from pymongo import MongoClient

# ---------- 1. MongoDB Atlas check ----------
try:
    client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
    print("MongoDB connected")

    col = client["energy"]["test"]
    col.insert_one({"msg": "hello from energy-bda"})
    print("Test document:", col.find_one({}, {"_id": 0}))
    col.drop()
    print("Databases:", client.list_database_names())
except Exception as e:
    print("MongoDB failed:", e)

# ---------- 2. Spark + dataset check ----------
try:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("check").getOrCreate()
    df = spark.read.csv(config.RAW_DATA_STR, sep=";", header=True)
    print("Rows:", df.count())
    df.show(5)
    spark.stop()
except Exception as e:
    print("Spark or dataset failed:", e)