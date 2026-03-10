import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_utils import load_and_prepare_data
from model_utils import preprocess_data, train_models, predict_future


st.set_page_config(layout="wide")

st.title("AI Crop Production Prediction")


@st.cache_data
def get_data():

    df = load_and_prepare_data("crop_data.csv")

    return df


df = get_data()


st.sidebar.header("Configuration")

crop = st.sidebar.selectbox(
    "Select Crop",
    sorted(df["Crop"].unique())
)

state = st.sidebar.selectbox(
    "Select State",
    sorted(df[df["Crop"]==crop]["State"].unique())
)


df_long = preprocess_data(df, crop, state)

if df_long.empty:

    st.warning("No data available")
    st.stop()


results = train_models(df_long)

last_year = df_long["Year"].max()

future_df = predict_future(results, last_year)


st.subheader("Production Trend")

plot_df = pd.concat([
    df_long.assign(Type="Historical"),
    future_df.assign(Type="Forecast")
])

fig, ax = plt.subplots()

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


st.subheader("Insights")

for season, info in results.items():

    avg_future = future_df[future_df["Season"]==season]["Production"].mean()

    if avg_future > info["last"]*1.05:

        st.success(f"{season} : Surplus expected")

    elif avg_future < info["last"]*0.95:

        st.warning(f"{season} : Shortage expected")

    else:

        st.info(f"{season} : Stable")
