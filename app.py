import streamlit as st
import matplotlib.pyplot as plt

from data_utils import load_and_prepare_data, filter_data
from model_utils import train_model, forecast_future

# --------------------------------------------------
st.set_page_config(page_title="Crop Production Predictor", layout="centered")

st.title("🌾 AI-Based Crop Production Trend Predictor")
st.markdown(
    "Predict future crop production trends to anticipate surplus or shortages."
)

# --------------------------------------------------
@st.cache_data
def load_data():
    return load_and_prepare_data()

df = load_data()

# --------------------------------------------------
states = sorted(df["State"].unique())
seasons = sorted(df["Season"].unique())

state = st.selectbox("Select State", states)
season = st.selectbox("Select Season", seasons)

# --------------------------------------------------
if st.button("Generate Prediction"):
    df_filt = filter_data(df, state, season)

    if len(df_filt) < 3:
        st.warning("Not enough data for prediction.")
    else:
        model, last_val = train_model(df_filt)
        years, preds, status, recommendation = forecast_future(model, last_val)

        # ------------------ Plot ------------------
        fig, ax = plt.subplots()
        ax.plot(df_filt["Year"], df_filt["Production"], marker="o", label="Historical")
        ax.plot(years, preds, linestyle="--", marker="o", label="Forecast")

        ax.set_title(f"Production Trend: {state} ({season})")
        ax.set_xlabel("Year")
        ax.set_ylabel("Production")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

        # ------------------ Output ------------------
        st.subheader("📌 Prediction Result")
        st.success(status)
        st.markdown(recommendation)
