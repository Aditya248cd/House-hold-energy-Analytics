"""
step5_forecast.py
──────────────────
PySpark MLlib job:
  Reads the pre-aggregated hourly Parquet (not 2 M raw rows),
  engineers lag features, trains a Baseline / Linear Regression /
  Random Forest model, and stores predictions + metrics in MongoDB Atlas.

Paths and secrets are sourced from config.py (which reads .env).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg, expr, hour, dayofweek, month, round as _round,
)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pymongo import MongoClient

SPLIT = "2010-01-01"

# ---------------------------------------------------------------------------
spark = SparkSession.builder.appName("Step5Forecast").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ── 1. [P1] Read pre-aggregated hourly Parquet (not 2 M raw rows) ───────────
# step3 produced this file — it is ~35k rows vs 2 M raw minute readings.
hourly_raw = spark.read.parquet(config.HOURLY_PARQUET_STR)

# We need only (hour_ts, avg_power) for the model; rename avg_power → power
hourly = (
    hourly_raw.select("hour_ts", col("avg_power").alias("power"))
              .orderBy("hour_ts")
)

# ── 2. Lag features: power 1h, 24h, 168h (1 week) earlier ──────────────────
data = hourly
for k in [1, 24, 168]:
    shifted = (
        hourly
        .withColumn("hour_ts", expr(f"hour_ts + interval {k} hours"))
        .withColumnRenamed("power", f"lag_{k}")
    )
    data = data.join(shifted, "hour_ts", "inner")

data = (
    data.withColumn("hour",  hour("hour_ts"))
        .withColumn("dow",   dayofweek("hour_ts"))
        .withColumn("month", month("hour_ts"))
        .cache()
)
print("Usable hourly rows:", data.count())

features = ["hour", "dow", "month", "lag_1", "lag_24", "lag_168"]
data = VectorAssembler(inputCols=features, outputCol="features").transform(data)

# ── 3. Chronological train / test split ─────────────────────────────────────
train = data.filter(col("hour_ts") < SPLIT)
test  = data.filter(col("hour_ts") >= SPLIT)
print("Train rows:", train.count(), "| Test rows:", test.count())

ev = {
    m: RegressionEvaluator(labelCol="power", predictionCol="prediction", metricName=m)
    for m in ["rmse", "mae", "r2"]
}

# ── 4. Baseline: "same hour yesterday" ──────────────────────────────────────
base = test.withColumn("prediction", col("lag_24"))
metrics = [{
    "model": "Baseline (same hour yesterday)",
    "rmse": round(ev["rmse"].evaluate(base), 4),
    "mae":  round(ev["mae"].evaluate(base), 4),
    "r2":   round(ev["r2"].evaluate(base), 4),
}]

# ── 5. Train models ──────────────────────────────────────────────────────────
lr = LinearRegression(featuresCol="features", labelCol="power").fit(train)
rf = RandomForestRegressor(
    featuresCol="features", labelCol="power",
    numTrees=50, maxDepth=8, seed=42,
).fit(train)

preds = {}
for name, model in [("Linear Regression", lr), ("Random Forest", rf)]:
    p = model.transform(test)
    preds[name] = p
    metrics.append({
        "model": name,
        "rmse": round(ev["rmse"].evaluate(p), 4),
        "mae":  round(ev["mae"].evaluate(p), 4),
        "r2":   round(ev["r2"].evaluate(p), 4),
    })

mdf = pd.DataFrame(metrics)
print("\n=== Model comparison (test set: 2010) ===")
print(mdf.to_string(index=False))

best_name = mdf.iloc[1:].sort_values("rmse").iloc[0]["model"]
print("\nBest model:", best_name)

imp = pd.DataFrame({
    "feature":    features,
    "importance": [round(float(x), 4) for x in rf.featureImportances.toArray()],
}).sort_values("importance", ascending=False)
print("\n=== Random Forest feature importance ===")
print(imp.to_string(index=False))

# ── 6. Save to MongoDB Atlas ─────────────────────────────────────────────────
fc = (
    preds[best_name]
    .select(
        "hour_ts",
        _round("power",      3).alias("actual"),
        _round("prediction", 3).alias("predicted"),
    )
    .orderBy("hour_ts")
    .toPandas()
)

db = MongoClient(config.MONGO_URI)["energy"]
for name, pdf in [("forecasts", fc), ("model_metrics", mdf), ("feature_importance", imp)]:
    db[name].drop()
    db[name].insert_many(pdf.to_dict("records"))
    print(f"Saved {len(pdf)} docs into {name}")

db["forecasts"].create_index("hour_ts")
print("Forecast model used:", best_name)

spark.stop()