"""
step2_clean.py
──────────────
PySpark job: reads raw household power consumption CSV, cleans it,
and writes a Parquet file for the next pipeline steps.

Paths and secrets are sourced from config.py (which reads .env).
No hardcoded paths or credentials in this file.
"""

import sys
from pathlib import Path

# Make data/ importable when run from project root or from data/
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402  — sets Hadoop env vars as a side effect

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, to_timestamp, concat_ws, sum as _sum, min as _min, max as _max,
)

# ---------------------------------------------------------------------------
spark = SparkSession.builder.appName("Step2Clean").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# 1. Read raw data
df = spark.read.csv(config.RAW_DATA_STR, sep=";", header=True)
print("Raw rows:", df.count())

# 2. Count missing ('?') values per column
print("\nMissing values per column:")
df.select([
    _sum(when(col(c) == "?", 1).otherwise(0)).alias(c) for c in df.columns
]).show()

# 3. Combine Date + Time into one timestamp
df = df.withColumn(
    "ts",
    to_timestamp(concat_ws(" ", col("Date"), col("Time")), "d/M/yyyy HH:mm:ss")
)

# 4. Rename and cast numeric columns ('?' becomes null on cast)
df = df.select(
    "ts",
    col("Global_active_power").cast("double").alias("active_power"),
    col("Global_reactive_power").cast("double").alias("reactive_power"),
    col("Voltage").cast("double").alias("voltage"),
    col("Global_intensity").cast("double").alias("intensity"),
    col("Sub_metering_1").cast("double").alias("kitchen"),
    col("Sub_metering_2").cast("double").alias("laundry"),
    col("Sub_metering_3").cast("double").alias("heater_ac"),
)

# 5. Drop rows with any missing value
df = df.dropna()
print("Rows after cleaning:", df.count())

# 6. Basic statistics
print("\nSummary statistics:")
df.describe("active_power", "voltage", "kitchen", "laundry", "heater_ac").show()

# 7. Date range
df.select(_min("ts").alias("first_reading"), _max("ts").alias("last_reading")).show()

# 8. Preview
df.show(5)
df.printSchema()

# 9. Save cleaned data as Parquet (faster than CSV for downstream steps)
out = config.CLEANED_PARQUET_STR
df.write.mode("overwrite").parquet(out)
print(f"Cleaned Parquet saved to: {out}")

spark.stop()