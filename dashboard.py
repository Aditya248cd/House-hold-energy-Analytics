import calendar

import certifi
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pymongo import MongoClient

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
# URI is read from .streamlit/secrets.toml (gitignored).
# Never hardcode credentials in source files.
URI = st.secrets["MONGO_URI"]

st.set_page_config(page_title="Household Energy Analytics", page_icon="⚡", layout="wide")

BLUE, RED, GREEN, ORANGE, GREY = "#2E86AB", "#E4572E", "#3BB273", "#F5A623", "#9AA5B1"
DOW_NAMES = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday",
             5: "Thursday", 6: "Friday", 7: "Saturday"}
DOW_ORDER = [2, 3, 4, 5, 6, 7, 1]  # Monday first
FEATURE_NAMES = {
    "lag_1": "Usage 1 hour ago",
    "lag_24": "Usage at same hour yesterday",
    "lag_168": "Usage at same hour last week",
    "hour": "Hour of day",
    "month": "Month of year",
    "dow": "Day of week",
}


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def hour_label(h: int) -> str:
    return f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}"


def show(fig):
    """Display a Plotly chart (works on old and new Streamlit versions)."""
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), title_font_size=18,
                      font=dict(size=13), legend_title_text="")
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def table(df: pd.DataFrame, height=None):
    kwargs = {"hide_index": True}
    if height is not None:
        kwargs["height"] = height
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)


def explain(text: str):
    st.markdown(f"<div style='color:#555;font-size:15px;margin-bottom:6px'>ℹ️ {text}</div>",
                unsafe_allow_html=True)


@st.cache_resource
def get_db():
    return MongoClient(URI, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=15000)["energy"]


@st.cache_data(ttl=600, show_spinner="Loading data from MongoDB...")
def load(name, cols=None):
    proj = {"_id": 0}
    if cols:
        proj.update({c: 1 for c in cols})
    return pd.DataFrame(list(get_db()[name].find({}, proj)))


# ----------------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------------
try:
    daily   = load("readings_daily")
    monthly = load("monthly_trend")
    peak    = load("peak_hours")
    dow     = load("dow_pattern")
    share   = load("appliance_share")
    anom    = load("anomalies")
    fc      = load("forecasts")
    metrics = load("model_metrics")
    imp     = load("feature_importance")
    # [P1] Load the pre-aggregated 7×24 heatmap (168 docs) instead of the
    # full readings_hourly collection (~35,000 docs) — much faster over network.
    heatmap = load("dow_hour_heatmap", ["dow", "hour", "avg_power"])
except Exception as e:
    st.error("Could not connect to MongoDB. Check Atlas Network Access (0.0.0.0/0), "
             "that the cluster is not paused, and try a mobile hotspot.")
    st.code(str(e)[:300])
    st.stop()

daily["day"] = pd.to_datetime(daily["day"])
daily = daily.sort_values("day")
fc["hour_ts"] = pd.to_datetime(fc["hour_ts"])
fc = fc.sort_values("hour_ts")
anom["hour_ts"] = pd.to_datetime(anom["hour_ts"])

# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header(" Project Details")
    st.write(
        "This dashboard analyses **4 years of smart-meter data** from one household "
        "(Dec 2006 - Nov 2010, about 2 million minute-by-minute readings)."
    )
    st.markdown(
        "**Pipeline**\n\n"
        "1. PySpark cleans and processes the data\n"
        "2. MongoDB stores the results\n"
        "3. This dashboard reads from MongoDB"
    )
    st.markdown("**SDG 7:** Affordable and Clean Energy  \n**SDG 12 / 13:** Responsible Consumption, Climate Action")
    with st.expander("📖 Simple glossary"):
        st.markdown(
            "- **kW (kilowatt):** how much power the house is using *right now*.\n"
            "- **kWh (kilowatt-hour):** total energy used over time. Using 1 kW for 1 hour = 1 kWh. "
            "This is what your electricity bill counts.\n"
            "- **Peak hour:** the time of day with the highest average power use.\n"
            "- **Anomaly:** an hour with unusually high power use compared with normal for that hour.\n"
            "- **Z-score:** how unusual a reading is. 3 or more = very unusual.\n"
            "- **Forecast:** a prediction of future power use made by a machine learning model.\n"
            "- **RMSE / MAE:** the model's typical error in kW. Lower is better.\n"
            "- **R²:** how much of the ups and downs the model explains (1.0 = perfect, 0 = no better than the average)."
        )
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.title("⚡ Household Energy Consumption Analytics")
st.markdown(
    "An interactive analysis of household electricity consumption, providing insights into usage patterns, appliance-level energy consumption, unusual usage trends, and electricity demand forecasting. "
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "🕒 When is energy used?", "🔌 Which appliances?", "🚨 Unusual usage", "🔮 Forecast"]
)

# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================
with tab1:
    top = peak.loc[peak["avg_power"].idxmax()]
    low = peak.loc[peak["avg_power"].idxmin()]
    total_kwh = daily["total_kwh"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total energy used", f"{total_kwh:,.0f} kWh", help="Sum of all daily consumption in the dataset.")
    c2.metric("Average per day", f"{daily['total_kwh'].mean():.1f} kWh", help="Average energy used in one day.")
    c3.metric("Busiest hour", hour_label(int(top["hour"])),
              f"{top['avg_power']:.2f} kW on average", delta_color="off",
              help="Hour of the day with the highest average power use.")
    c4.metric("Unusual hours found", f"{len(anom):,}", help="Hours with power use far above normal (z-score > 3).")

    st.subheader("Energy used each day")
    explain("Each faint line point is one day. The thick line is the 30-day average, which smooths out "
            "day-to-day noise so the seasons are easy to see. Usage rises in winter and falls in summer.")
    d = daily.copy()
    d["avg30"] = d["total_kwh"].rolling(30, min_periods=7).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["day"], y=d["total_kwh"], mode="lines", name="Daily usage",
                             line=dict(color=GREY, width=1), opacity=0.6,
                             hovertemplate="%{x|%d %b %Y}<br>%{y:.1f} kWh<extra>Daily</extra>"))
    fig.add_trace(go.Scatter(x=d["day"], y=d["avg30"], mode="lines", name="30-day average",
                             line=dict(color=BLUE, width=3),
                             hovertemplate="%{x|%d %b %Y}<br>%{y:.1f} kWh<extra>30-day avg</extra>"))
    fig.update_layout(title="Daily electricity consumption (2006 - 2010)",
                      xaxis_title="Date", yaxis_title="Energy used per day (kWh)", hovermode="x unified")
    show(fig)

    st.subheader("💡 Energy-saving recommendations")
    explain("These tips are generated automatically from the results in this dashboard.")
    heater = share.loc[share["appliance"] == "Water heater / AC", "percent"].iloc[0]
    heater_kwh = share.loc[share["appliance"] == "Water heater / AC", "kwh"].iloc[0]
    wm = monthly[monthly["month"].isin([12, 1, 2])]["total_kwh"].mean()
    sm = monthly[monthly["month"].isin([6, 7, 8])]["total_kwh"].mean()
    r1, r2, r3 = st.columns(3)
    r1.info(f"**1. Shift heavy appliances away from peak time**\n\n"
            f"Usage is highest around **{hour_label(int(top['hour']))}** ({top['avg_power']:.2f} kW) and lowest at "
            f"**{hour_label(int(low['hour']))}** ({low['avg_power']:.2f} kW), about "
            f"{top['avg_power'] / low['avg_power']:.1f}x difference. Run washing machines and dishwashers off-peak.")
    r2.info(f"**2. Target water heating and air conditioning**\n\n"
            f"They use **{heater:.1f}%** of all energy, the biggest controllable load. "
            f"Cutting it by just 10% would save about **{heater_kwh * 0.1:,.0f} kWh** over the period "
            f"(illustrative estimate).")
    r3.info(f"**3. Reduce winter heating demand**\n\n"
            f"Winter months average **{wm:,.0f} kWh** vs **{sm:,.0f} kWh** in summer "
            f"(**{wm / sm:.1f}x** more). Better insulation and smart thermostats help most.")

# ============================================================================
# TAB 2: WHEN IS ENERGY USED
# ============================================================================
with tab2:
    st.subheader("Usage through the day")
    explain("Average power at each hour of the day, over all 4 years. "
            "<b>Red bars = the 3 busiest hours.</b> Notice the low use at night and the jump in the morning.")
    p = peak.sort_values("hour").copy()
    p["label"] = p["hour"].apply(hour_label)
    top3 = set(p.nlargest(3, "avg_power")["hour"])
    fig = go.Figure(go.Bar(
        x=p["label"], y=p["avg_power"],
        marker_color=[RED if h in top3 else BLUE for h in p["hour"]],
        hovertemplate="<b>%{x}</b><br>Average power: %{y:.2f} kW<extra></extra>"))
    fig.update_xaxes(categoryorder="array", categoryarray=p["label"].tolist())
    fig.update_layout(title="Average power by hour of day",
                      xaxis_title="Hour of day", yaxis_title="Average power (kW)")
    show(fig)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Usage by day of the week")
        explain("Average power on each weekday. Weekends are higher because people are home.")
        dd = dow.copy()
        dd["day"] = dd["dow"].map(DOW_NAMES)
        dd["order"] = dd["dow"].map({d: i for i, d in enumerate(DOW_ORDER)})
        dd = dd.sort_values("order")
        hi_day = dd.loc[dd["avg_power"].idxmax(), "day"]
        fig = go.Figure(go.Bar(
            x=dd["day"], y=dd["avg_power"],
            marker_color=[RED if x == hi_day else BLUE for x in dd["day"]],
            hovertemplate="<b>%{x}</b><br>Average power: %{y:.2f} kW<extra></extra>"))
        fig.update_layout(title="Average power by day of week",
                          xaxis_title="Day of week", yaxis_title="Average power (kW)")
        show(fig)

    with col_b:
        st.subheader("Usage by month, year by year")
        explain("Each line is one year. All years follow the same shape: high in winter, low in summer. "
                "(Dec 2006 and Nov 2010 are partial months.)")
        m = monthly.copy()
        m["Month"] = m["month"].apply(lambda x: calendar.month_abbr[x])
        m["Year"] = m["year"].astype(str)
        fig = px.line(m, x="Month", y="total_kwh", color="Year", markers=True,
                      category_orders={"Month": list(calendar.month_abbr)[1:]},
                      labels={"total_kwh": "Energy used (kWh)"},
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(hovertemplate="%{x}<br>%{y:,.0f} kWh<extra>%{fullData.name}</extra>")
        fig.update_layout(title="Monthly energy consumption by year",
                          xaxis_title="Month", yaxis_title="Energy used per month (kWh)")
        show(fig)

    st.subheader("Heatmap: the busiest times of the week")
    explain("Darker red = higher average power. Read it like a calendar: each row is a weekday and "
            "each column is an hour of the day. The deep red blocks show the daily peak periods.")
    hm = heatmap.groupby(["dow", "hour"])["avg_power"].mean().unstack()
    hm = hm.reindex(DOW_ORDER)
    fig = go.Figure(go.Heatmap(
        z=hm.values, x=[hour_label(h) for h in hm.columns], y=[DOW_NAMES[i] for i in hm.index],
        colorscale="YlOrRd", colorbar=dict(title="kW"),
        hovertemplate="%{y}, %{x}<br>Average power: %{z:.2f} kW<extra></extra>"))
    fig.update_layout(title="Average power by weekday and hour",
                      xaxis_title="Hour of day", yaxis_title="Day of week")
    fig.update_yaxes(autorange="reversed")
    show(fig)

# ============================================================================
# TAB 3: APPLIANCES
# ============================================================================
with tab3:
    st.subheader("Where does the energy go?")
    explain("The dataset has 3 separately metered areas (kitchen, laundry, water heater/AC). "
            "Everything else, such as lighting, heating and electronics, is grouped as <b>Other appliances</b>.")
    cmap = {"Kitchen": ORANGE, "Laundry": GREEN, "Water heater / AC": RED, "Other appliances": GREY}
    a, b = st.columns(2)
    with a:
        fig = px.pie(share, names="appliance", values="kwh", hole=0.45,
                     color="appliance", color_discrete_map=cmap)
        fig.update_traces(textinfo="label+percent", textposition="outside",
                          hovertemplate="<b>%{label}</b><br>%{value:,.0f} kWh (%{percent})<extra></extra>")
        fig.update_layout(title="Share of total energy by appliance group", showlegend=False)
        show(fig)
    with b:
        s = share.sort_values("kwh")
        fig = go.Figure(go.Bar(
            x=s["kwh"], y=s["appliance"], orientation="h",
            marker_color=[cmap[x] for x in s["appliance"]],
            text=[f"{v:,.0f} kWh" for v in s["kwh"]], textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x:,.0f} kWh<extra></extra>"))
        fig.update_layout(title="Total energy used by appliance group",
                          xaxis_title="Energy used (kWh)", yaxis_title="")
        fig.update_xaxes(range=[0, s["kwh"].max() * 1.25])
        show(fig)

    st.subheader("Appliance usage over time")
    explain("Monthly energy of the three metered areas. The water heater / AC is consistently the largest, "
            "and rises in winter.")
    mm = daily.copy()
    mm["Month"] = mm["day"].dt.to_period("M").dt.to_timestamp()
    mm = mm.groupby("Month")[["kitchen_kwh", "laundry_kwh", "heater_ac_kwh"]].sum().reset_index()
    mm = mm.rename(columns={"kitchen_kwh": "Kitchen", "laundry_kwh": "Laundry",
                            "heater_ac_kwh": "Water heater / AC"})
    long = mm.melt("Month", var_name="Appliance", value_name="kWh")
    fig = px.bar(long, x="Month", y="kWh", color="Appliance", color_discrete_map=cmap)
    fig.update_traces(hovertemplate="%{x|%b %Y}<br>%{y:,.0f} kWh<extra>%{fullData.name}</extra>")
    fig.update_layout(title="Monthly energy by metered appliance group (stacked)",
                      xaxis_title="Month", yaxis_title="Energy used (kWh)", barmode="stack")
    show(fig)

    st.subheader("Summary table")
    t = share.rename(columns={"appliance": "Appliance group", "kwh": "Energy (kWh)", "percent": "Share of total (%)"})
    table(t)

# ============================================================================
# TAB 4: ANOMALIES
# ============================================================================
with tab4:
    st.subheader("Hours with unusually high usage")
    explain(f"We compared every hour with what is normal for that time of day. "
            f"<b>{len(anom):,} hours</b> were far above normal (z-score above 3). "
            f"These could be heating left on, faulty appliances, or unusual events like a holiday gathering.")

    big = anom.sort_values("z_score", ascending=False).iloc[0]
    k1, k2, k3 = st.columns(3)
    k1.metric("Most unusual hour", big["hour_ts"].strftime("%d %b %Y, %I %p"))
    k2.metric("Power at that hour", f"{big['avg_power']:.2f} kW",
              f"normal is {big['normal_power']:.2f} kW", delta_color="off")
    k3.metric("How unusual (z-score)", f"{big['z_score']:.1f}", help="3 or more is very unusual.")

    fig = px.scatter(anom, x="hour_ts", y="avg_power", size="z_score", color="z_score",
                     color_continuous_scale="OrRd",
                     labels={"hour_ts": "Date and time", "avg_power": "Power (kW)", "z_score": "Z-score"})
    fig.update_traces(hovertemplate="%{x|%d %b %Y, %I %p}<br>Power: %{y:.2f} kW<br>"
                                    "Z-score: %{marker.color:.1f}<extra></extra>")
    fig.update_layout(title="When the anomalies happened (bigger and darker dot = more unusual)",
                      xaxis_title="Date", yaxis_title="Average power in that hour (kW)")
    show(fig)

    a, b = st.columns(2)
    with a:
        by_hour = anom.groupby("hour").size().reindex(range(24), fill_value=0)
        fig = go.Figure(go.Bar(x=[hour_label(h) for h in by_hour.index], y=by_hour.values,
                               marker_color=ORANGE,
                               hovertemplate="<b>%{x}</b><br>%{y} anomalies<extra></extra>"))
        fig.update_xaxes(categoryorder="array", categoryarray=[hour_label(h) for h in range(24)])
        fig.update_layout(title="Number of anomalies by hour of day",
                          xaxis_title="Hour of day", yaxis_title="Number of anomalous hours")
        show(fig)
    with b:
        by_year = anom.groupby(anom["hour_ts"].dt.year).size()
        fig = go.Figure(go.Bar(x=by_year.index.astype(str), y=by_year.values, marker_color=BLUE,
                               hovertemplate="<b>%{x}</b><br>%{y} anomalies<extra></extra>"))
        fig.update_layout(title="Number of anomalies by year",
                          xaxis_title="Year", yaxis_title="Number of anomalous hours")
        show(fig)

    st.subheader("Full list (sorted by most unusual)")
    t = anom.sort_values("z_score", ascending=False).copy()
    t["hour_ts"] = t["hour_ts"].dt.strftime("%d %b %Y, %I:%M %p")
    t = t[["hour_ts", "avg_power", "normal_power", "z_score"]].rename(columns={
        "hour_ts": "Date and time", "avg_power": "Actual power (kW)",
        "normal_power": "Normal for this hour (kW)", "z_score": "Z-score"})
    table(t, height=350)

# ============================================================================
# TAB 5: FORECAST
# ============================================================================
with tab5:
    st.subheader("Can we predict energy use?")
    explain("We trained machine-learning models (using Spark MLlib) on data up to the end of 2009, then asked them "
            "to predict every hour of 2010, data they had never seen. Predictions are compared with what actually happened.")

    mt = metrics.copy()
    base = mt[mt["model"].str.startswith("Baseline")].iloc[0]
    ml = mt[~mt["model"].str.startswith("Baseline")].sort_values("rmse")
    best = ml.iloc[0]
    improve = 100 * (base["rmse"] - best["rmse"]) / base["rmse"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Best model", best["model"])
    k2.metric("Typical error (RMSE)", f"{best['rmse']:.2f} kW",
              f"{improve:.0f}% better than baseline", help="Lower is better.")
    k3.metric("Average miss (MAE)", f"{best['mae']:.2f} kW", help="On average the prediction is this far off.")
    k4.metric("R² score", f"{best['r2']:.2f}", help="1.0 = perfect. Hourly household power is noisy, so about 0.6 is reasonable.")

    a, b = st.columns(2)
    with a:
        explain("<b>Baseline</b> = simply guessing 'same as this hour yesterday'. "
                "A good model must beat it. Shorter bar = smaller error = better.")
        mt = mt.sort_values("rmse", ascending=False)
        fig = go.Figure(go.Bar(
            x=mt["rmse"], y=mt["model"], orientation="h",
            marker_color=[GREY if m.startswith("Baseline") else (GREEN if m == best["model"] else BLUE)
                          for m in mt["model"]],
            text=[f"{v:.3f}" for v in mt["rmse"]], textposition="outside",
            hovertemplate="<b>%{y}</b><br>RMSE: %{x:.3f} kW<extra></extra>"))
        fig.update_layout(title="Prediction error by model (lower is better)",
                          xaxis_title="RMSE (kW)", yaxis_title="")
        fig.update_xaxes(range=[0, mt["rmse"].max() * 1.2])
        show(fig)
    with b:
        explain("Which information helped the model most? Recent usage is the strongest clue.")
        im = imp.copy()
        im["name"] = im["feature"].map(FEATURE_NAMES).fillna(im["feature"])
        im = im.sort_values("importance")
        fig = go.Figure(go.Bar(
            x=im["importance"] * 100, y=im["name"], orientation="h", marker_color=GREEN,
            text=[f"{v * 100:.0f}%" for v in im["importance"]], textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x:.1f}% importance<extra></extra>"))
        fig.update_layout(title="What the model relies on when predicting",
                          xaxis_title="Importance (%)", yaxis_title="")
        fig.update_xaxes(range=[0, im["importance"].max() * 120])
        show(fig)

    st.subheader("Predicted vs actual usage")
    explain("Pick a start date and how many days to view. The closer the orange line follows the blue line, "
            "the better the prediction.")
    lo, hi = fc["hour_ts"].min().date(), fc["hour_ts"].max().date()
    c1, c2 = st.columns(2)
    start = c1.date_input("Start date", lo, min_value=lo, max_value=hi)
    days = c2.slider("Number of days to show", 1, 30, 7)
    sel = fc[(fc["hour_ts"] >= pd.Timestamp(start)) &
             (fc["hour_ts"] < pd.Timestamp(start) + pd.Timedelta(days=days))]
    if sel.empty:
        st.warning("No forecast data for this range. Pick another date.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sel["hour_ts"], y=sel["actual"], name="Actual", mode="lines",
                                 line=dict(color=BLUE, width=2.5),
                                 hovertemplate="%{x|%d %b, %I %p}<br>Actual: %{y:.2f} kW<extra></extra>"))
        fig.add_trace(go.Scatter(x=sel["hour_ts"], y=sel["predicted"], name="Predicted", mode="lines",
                                 line=dict(color=ORANGE, width=2.5, dash="dot"),
                                 hovertemplate="%{x|%d %b, %I %p}<br>Predicted: %{y:.2f} kW<extra></extra>"))
        fig.update_layout(title=f"{best['model']}: actual vs predicted power ({days} day(s) from {start})",
                          xaxis_title="Date and time", yaxis_title="Average power (kW)", hovermode="x unified")
        show(fig)

    st.subheader("All model results")
    t = metrics.rename(columns={"model": "Model", "rmse": "RMSE (kW)", "mae": "MAE (kW)", "r2": "R² score"})
    table(t)