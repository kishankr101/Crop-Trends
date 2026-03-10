import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from data_utils import load_and_prepare_data
from model_utils import train_model, predict_trend

# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="wide")
st.title("🌾 AI-Based Crop Production Trend Prediction")

# ---------------- LOAD DATA ----------------
@st.cache_data
def get_data():
    df = load_data("crop_data.csv")
    df.columns = (
        df.columns
        .str.replace('ï»¿', '', regex=False)
        .str.replace('"', '', regex=False)
        .str.strip()
    )
    return df

df = get_data()

# ---------------- PREPROCESS ----------------
@st.cache_data
def preprocess_data(df, crop, state):
    df = df[(df["Crop"] == crop) & (df["State"] == state)]
    if df.empty:
        return pd.DataFrame()

    prod_cols = [c for c in df.columns if "Production-" in c]

    long_df = df.melt(
        id_vars=["Crop", "State", "Season"],
        value_vars=prod_cols,
        var_name="Year_Col",
        value_name="Production"
    )

    long_df["Year"] = long_df["Year_Col"].str.extract(r"(\d{4})").astype(int)
    long_df = long_df.dropna(subset=["Production"])
    return long_df.sort_values("Year")

# ---------------- MODEL ----------------
def train_models(df_long):
    results = {}
    for season in df_long["Season"].unique():
        s_df = df_long[df_long["Season"] == season]
        if len(s_df) < 2:
            continue

        X = s_df[["Year"]]
        y = s_df["Production"]

        model = LinearRegression()
        model.fit(X, y)

        results[season] = {
            "model": model,
            "last": y.iloc[-1],
            "r2": model.score(X, y)
        }
    return results

def predict_future(models):
    years = pd.DataFrame({"Year": [2026, 2027, 2028, 2029, 2030]})
    rows = []

    for season, info in models.items():
        preds = info["model"].predict(years)
        for y, p in zip(years["Year"], preds):
            rows.append({"Season": season, "Year": y, "Production": p})

    return pd.DataFrame(rows)

# ---------------- UI ----------------
st.sidebar.header("Configuration")

crop = st.sidebar.selectbox("Select Crop", sorted(df["Crop"].unique()))
state = st.sidebar.selectbox(
    "Select State",
    sorted(df[df["Crop"] == crop]["State"].unique())
)

df_long = preprocess_data(df, crop, state)

if df_long.empty:
    st.warning("No data available.")
    st.stop()

models = train_models(df_long)
future_df = predict_future(models)

st.subheader("📊 Historical + Forecast Trends")

plot_df = pd.concat([
    df_long.assign(Type="Historical"),
    future_df.assign(Type="Forecast")
])

fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(
    data=plot_df,
    x="Year",
    y="Production",
    hue="Season",
    style="Type",
    markers=True,
    ax=ax
)
st.pyplot(fig)

st.subheader("🧠 Insights")
for season, info in models.items():
    avg_future = future_df[future_df["Season"] == season]["Production"].mean()
    if avg_future > info["last"] * 1.05:
        st.success(f"{season}: Likely Surplus")
    elif avg_future < info["last"] * 0.95:
        st.warning(f"{season}: Likely Shortage")
    else:
        st.info(f"{season}: Stable")
