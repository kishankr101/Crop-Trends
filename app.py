import streamlit as st
import pandas as pd
from model_utils import *

st.set_page_config(
    page_title="Crop Trend Predictor",
    layout="wide"
)

st.title("🌾 AI‑Based Crop Production Trend Predictor")
st.caption(
    "Predictive decision‑support to anticipate crop surplus and shortages"
)

# Load data
df_raw = load_data("data/crop_data.csv")
df = preprocess_data(df_raw)

# Sidebar
st.sidebar.header("Select Inputs")
state = st.sidebar.selectbox("State", sorted(df["State"].unique()))
crop = st.sidebar.selectbox("Crop", sorted(df["Crop"].unique()))

if st.sidebar.button("Predict Trend"):
    model, filtered = train_model(df, state, crop)

    last_year = filtered["Year"].max()
    years, predictions = predict_future(model, last_year)

    trend, risk, recommendation = generate_insight(predictions)

    # --- OUTPUTS ---
    st.subheader(f"📍 {crop} in {state}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Trend", trend)
    col2.metric("Risk Indicator", risk)
    col3.metric(
        "Predicted Production (Next Year)",
        f"{predictions[-1]:.2f}"
    )

    st.markdown("### 📈 Production Trend")
    chart_df = pd.DataFrame({
        "Year": list(filtered["Year"]) + list(years),
        "Production": list(filtered["Production"]) + list(predictions)
    })

    st.line_chart(chart_df.set_index("Year"))

    st.markdown("### 🧠 Recommendation")
    st.success(recommendation)

st.markdown("---")
st.caption(
    "⚠️ Disclaimer: Predictions are trend‑based and indicative. "
    "Use alongside local agronomic expertise."
)
