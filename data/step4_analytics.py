"""
step4_analytics.py
───────────────────
PySpark job:
  Reads the pre-aggregated hourly Parquet (not 2 M raw rows) and
  produces five analytics collections in MongoDB Atlas:

    monthly_trend      — energy by year-month
    peak_hours         — average power per hour of day
    dow_pattern        — average power per day of week
    appliance_share    — kitchen / laundry / heater-AC / other breakdown
    anomalies          — hourly readings with z-score > 3
    dow_hour_heatmap   — [P1] 7×24 grid pre-aggregated for the dashboard heatmap

Paths and secrets are sourced from config.py (which reads .env).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, date_trunc, avg, stddev, sum as _sum,
    round as _round, year, month, dayofweek, hour, lit,
)
from pymongo import MongoClient

# ---------------------------------------------------------------------------
spark = SparkSession.builder.appName("Step4Analytics").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# [P1] Read the pre-aggregated hourly Parquet — not the 2 M raw rows
hourly = spark.read.parquet(config.HOURLY_PARQUET_STR).cache()
print("Hourly rows loaded:", hourly.count())

# For daily appliance share we still need the minute-level data briefly
raw = spark.read.parquet(config.CLEANED_PARQUET_STR)

client = MongoClient(config.MONGO_URI)
db = client["energy"]

def save(sdf, name):
    pdf = sdf.toPandas()
    db[name].drop()
    db[name].insert_many(pdf.to_dict("records"))
    print(f"\n=== {name} ({len(pdf)} docs) ===")
    print(pdf.head(10).to_string(index=False))

# ── 1. Monthly trend ────────────────────────────────────────────────────────
# hourly already has year and month columns
monthly = (
    hourly.groupBy("year", "month")
          .agg(
              _round(_sum("avg_power") / 60, 2).alias("total_kwh"),
              _round(avg("avg_power"), 3).alias("avg_power"),
          )
          .orderBy("year", "month")
)
save(monthly, "monthly_trend")

# ── 2. Peak hours (average power by hour of day) ────────────────────────────
peak = (
    hourly.groupBy("hour")
          .agg(
              _round(avg("avg_power"), 3).alias("avg_power"),
              _round(_sum("avg_power"), 3).alias("max_power"),
          )
          .orderBy("hour")
)
save(peak, "peak_hours")

# ── 3. Day-of-week pattern ──────────────────────────────────────────────────
dow = (
    hourly.groupBy("dow")
          .agg(_round(avg("avg_power"), 3).alias("avg_power"))
          .orderBy("dow")
)
save(dow, "dow_pattern")

# ── 4. Appliance share (needs minute-level Wh values) ───────────────────────
tot = raw.agg(
    _sum(col("active_power") * 1000 / 60).alias("total"),
    _sum("kitchen").alias("kitchen"),
    _sum("laundry").alias("laundry"),
    _sum("heater_ac").alias("heater_ac"),
).collect()[0]
other = tot["total"] - tot["kitchen"] - tot["laundry"] - tot["heater_ac"]
rows = [
    ("Kitchen",            tot["kitchen"]),
    ("Laundry",            tot["laundry"]),
    ("Water heater / AC",  tot["heater_ac"]),
    ("Other appliances",   other),
]
share_pdf = pd.DataFrame(
    [(n, round(v / 1000, 1), round(100 * v / tot["total"], 1)) for n, v in rows],
    columns=["appliance", "kwh", "percent"],
)
db["appliance_share"].drop()
db["appliance_share"].insert_many(share_pdf.to_dict("records"))
print("\n=== appliance_share (4 docs) ===")
print(share_pdf.to_string(index=False))

# ── 5. Anomaly detection (hourly z-score, grouped by hour of day) ───────────
# Using hour already in the hourly dataset
stats = hourly.groupBy("hour").agg(
    avg("avg_power").alias("mu"),
    stddev("avg_power").alias("sd"),
)
anom = (
    hourly.join(stats, "hour")
          .withColumn("z_score", _round((col("avg_power") - col("mu")) / col("sd"), 2))
          .filter(col("z_score") > 3)
          .select(
              "hour_ts", "hour",
              _round("avg_power", 3).alias("avg_power"),
              _round("mu", 3).alias("normal_power"),
              "z_score",
          )
          .orderBy(col("z_score").desc())
)
save(anom, "anomalies")
db["anomalies"].create_index("hour_ts")

# ── 6. [P1] Pre-aggregate 7×24 heatmap (168 docs, not 35k) ─────────────────
# The dashboard used to download all 35k hourly docs to compute this in pandas.
# Now it reads this tiny collection of 168 cells directly.
heatmap = (
    hourly.groupBy("dow", "hour")
          .agg(_round(avg("avg_power"), 3).alias("avg_power"))
          .orderBy("dow", "hour")
)
save(heatmap, "dow_hour_heatmap")

print("\nAll analytics saved to MongoDB.")
spark.stop()