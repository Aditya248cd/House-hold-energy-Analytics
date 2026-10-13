"""
step3_load_mongo.py
────────────────────
PySpark job:
  1. Reads cleaned Parquet (2 M minute-level rows).
  2. Aggregates to hourly and daily granularity.
  3. Writes both to MongoDB Atlas (reads_hourly, readings_daily).
  4. [P1] Also saves the hourly aggregation as Parquet (data/hourly_parquet)
     so that step4 and step5 can read it directly instead of re-processing
     the 2 M raw rows again from scratch.

Paths and secrets are sourced from config.py (which reads .env).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, date_trunc, to_date, avg, sum as _sum, max as _max,
    count, round as _round, year, month, dayofweek, hour,
)
from pymongo import MongoClient

# ---------------------------------------------------------------------------
spark = SparkSession.builder.appName("Step3Load").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet(config.CLEANED_PARQUET_STR)
print("Minute-level rows:", df.count())

# ---------- Hourly aggregation ----------
hourly = (
    df.withColumn("hour_ts", date_trunc("hour", col("ts")))
      .groupBy("hour_ts")
      .agg(
          _round(avg("active_power"), 3).alias("avg_power"),
          _round(_max("active_power"), 3).alias("max_power"),
          _round(avg("voltage"), 2).alias("avg_voltage"),
          _round(_sum("kitchen"), 1).alias("kitchen"),
          _round(_sum("laundry"), 1).alias("laundry"),
          _round(_sum("heater_ac"), 1).alias("heater_ac"),
          count("*").alias("readings"),
      )
      .withColumn("year",  year("hour_ts"))
      .withColumn("month", month("hour_ts"))
      .withColumn("dow",   dayofweek("hour_ts"))
      .withColumn("hour",  hour("hour_ts"))
      .orderBy("hour_ts")
)

# ---------- Daily aggregation ----------
daily = (
    df.withColumn("day", to_date("ts"))
      .groupBy("day")
      .agg(
          _round(_sum("active_power") / 60, 3).alias("total_kwh"),
          _round(avg("active_power"), 3).alias("avg_power"),
          _round(_max("active_power"), 3).alias("max_power"),
          _round(_sum("kitchen") / 1000, 3).alias("kitchen_kwh"),
          _round(_sum("laundry") / 1000, 3).alias("laundry_kwh"),
          _round(_sum("heater_ac") / 1000, 3).alias("heater_ac_kwh"),
      )
      .orderBy("day")
)

print("Hourly rows:", hourly.count())
print("Daily rows:",  daily.count())

# ---------- [P1] Save hourly Parquet for reuse in step4 & step5 ----------
# This eliminates re-aggregating 2 M rows in every downstream Spark job.
hourly.write.mode("overwrite").parquet(config.HOURLY_PARQUET_STR)
print(f"Hourly Parquet saved to: {config.HOURLY_PARQUET_STR}")

# ---------- Write to MongoDB Atlas ----------
client = MongoClient(config.MONGO_URI)
db = client["energy"]

def load_to_mongo(pdf, name, batch=5000):
    col_ = db[name]
    col_.drop()
    records = pdf.to_dict("records")
    for i in range(0, len(records), batch):
        col_.insert_many(records[i:i + batch])
    print(f"Inserted {len(records)} docs into {name}")

hp = hourly.toPandas()
dp = daily.toPandas()
dp["day"] = dp["day"].astype("datetime64[ns]")   # Mongo needs datetime not date

load_to_mongo(hp, "readings_hourly")
load_to_mongo(dp, "readings_daily")

# ---------- Indexes ----------
db["readings_hourly"].create_index("hour_ts")
db["readings_daily"].create_index("day")
print("Indexes created.")

print("Sample hourly doc:", db["readings_hourly"].find_one({}, {"_id": 0}))
print("Sample daily doc:",  db["readings_daily"].find_one({},  {"_id": 0}))

spark.stop()