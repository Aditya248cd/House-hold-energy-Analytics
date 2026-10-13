# ⚡ Household Energy Analytics

An end-to-end **Big Data Analytics** project built with **PySpark**, **MongoDB Atlas**, and **Streamlit**.

It processes ~2 million smart-meter readings (Dec 2006 – Nov 2010) from one household, extracts deep consumption insights, detects anomalies, and forecasts future energy demand using Spark MLlib.

---

## 🏗️ Architecture

```
Raw CSV (2M rows)
      │
      ▼
┌─────────────┐     PySpark      ┌──────────────────┐
│  step2_     │  ─── cleans ──▶  │ cleaned_parquet   │
│  clean.py   │                  └──────────────────┘
└─────────────┘                          │
                                         ▼
┌─────────────┐     PySpark      ┌──────────────────┐
│  step3_     │  ─ aggregates ─▶ │ hourly_parquet    │
│  load_mongo │  ─ loads DB ──▶  │ MongoDB Atlas     │
└─────────────┘                  └──────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                     ▼
             ┌────────────┐      ┌────────────┐      ┌────────────────┐
             │  step4_    │      │  step5_    │      │  dashboard.py  │
             │  analytics │      │  forecast  │      │  (Streamlit)   │
             └────────────┘      └────────────┘      └────────────────┘
             Patterns, peaks,    Linear Regression   Interactive charts
             anomalies, heatmap  Random Forest       Forecasts, insights
```

---

## 📦 Tech Stack

| Layer | Technology |
|:---|:---|
| **Big Data Processing** | PySpark 3.5 (Spark MLlib) |
| **Storage** | MongoDB Atlas (NoSQL document store) |
| **Dashboard** | Streamlit + Plotly |
| **Language** | Python 3.x |

---

## 🗂️ Project Structure

```
House-hold-energy-Analytics/
├── .env.example              # Template — copy to .env and fill in real values
├── .gitignore                # Prevents secrets, venv, large data files
├── README.md
├── dashboard.py              # Streamlit dashboard (reads from MongoDB)
└── data/
    ├── config.py             # Central config — paths + secrets via .env
    ├── check.py              # Connectivity verification script
    ├── step2_clean.py        # PySpark: ingest & clean raw CSV → Parquet
    ├── step3_load_mongo.py   # PySpark: aggregate → MongoDB + hourly Parquet
    ├── step4_analytics.py    # PySpark: patterns, anomalies, heatmap → MongoDB
    └── step5_forecast.py     # PySpark MLlib: LR + RF forecasting → MongoDB
```

> **Note:** Raw data (`household_power_consumption.txt`), processed Parquet files, and `.env` secrets are gitignored and must be set up locally.

---

## 🚀 Setup & Run

### 1. Prerequisites
- Python 3.9+
- Java 11+ (required by PySpark)
- [Hadoop winutils](https://github.com/cdarlint/winutils) for Windows → set `HADOOP_HOME` in `.env`
- MongoDB Atlas account (free tier works)

### 2. Clone & install dependencies
```bash
git clone https://github.com/Aditya248cd/House-hold-energy-Analytics.git
cd House-hold-energy-Analytics
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install pyspark pymongo python-dotenv streamlit plotly certifi pandas pyarrow
```

### 3. Configure secrets
```bash
# Copy the template and fill in your MongoDB Atlas URI
copy .env.example .env
```
Edit `.env`:
```
MONGO_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?appName=<App>
HADOOP_HOME=C:/hadoop
```

Also create `.streamlit/secrets.toml` for the dashboard:
```toml
MONGO_URI = "mongodb+srv://..."
```

### 4. Download the dataset
Download [Individual household electric power consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) from UCI ML Repository and place the `.txt` file at:
```
data/household_power_consumption.txt
```

### 5. Run the pipeline (in order)
```bash
python data/step2_clean.py        # Clean raw data → cleaned_parquet
python data/step3_load_mongo.py   # Aggregate → MongoDB + hourly_parquet
python data/step4_analytics.py    # Compute analytics → MongoDB
python data/step5_forecast.py     # Train models → MongoDB
```

### 6. Launch the dashboard
```bash
streamlit run dashboard.py
```
Open **http://localhost:8501** in your browser.

---

## 📊 Dashboard Tabs

| Tab | What it shows |
|:---|:---|
| 📊 **Overview** | KPI metrics, daily trend, energy-saving recommendations |
| 🕒 **When is energy used?** | Hourly bars, day-of-week bars, monthly year-over-year lines, weekday/hour heatmap |
| 🔌 **Which appliances?** | Donut + bar chart of appliance share (kitchen, laundry, heater/AC, other) |
| 🚨 **Unusual usage** | Anomaly scatter plot, anomaly breakdown by hour & year, full table |
| 🔮 **Forecast** | Model comparison (Baseline vs LR vs RF), feature importance, interactive actual vs predicted chart |

---

## 📈 Model Results (test set: 2010)

| Model | RMSE (kW) | MAE (kW) | R² |
|:---|:---|:---|:---|
| Baseline (same hour yesterday) | 0.821 | 0.556 | -0.11 |
| Linear Regression | 0.535 | 0.380 | 0.53 |
| **Random Forest** ⭐ | **0.505** | **0.357** | **0.58** |

Top predictive feature: **lag_1** (power 1 hour ago) at 58.4% importance.

---

## 🌍 UN Sustainable Development Goals

This project directly supports:
- **SDG 7** — Affordable and Clean Energy
- **SDG 12** — Responsible Consumption and Production
- **SDG 13** — Climate Action

---

## 📄 Dataset

UCI ML Repository — [Individual household electric power consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
- ~2 million minute-by-minute readings
- Period: December 2006 – November 2010
- One household in Sceaux, France
