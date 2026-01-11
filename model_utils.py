import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    # Clean column names
    df.columns = df.columns.str.strip()

    # Convert wide → long (example for production)
    production_cols = [c for c in df.columns if "Production" in c]

    long_df = df.melt(
        id_vars=["Crop", "State", "Season"],
        value_vars=production_cols,
        var_name="Year",
        value_name="Production"
    )

    long_df["Year"] = long_df["Year"].str.extract(r'(\d{4})').astype(int)
    long_df.dropna(inplace=True)

    return long_df

def train_model(df, state, crop):
    filtered = df[(df["State"] == state) & (df["Crop"] == crop)]

    X = filtered[["Year"]]
    y = filtered["Production"]

    model = LinearRegression()
    model.fit(X, y)

    return model, filtered

def predict_future(model, last_year, years_ahead=1):
    future_years = np.array(
        [last_year + i for i in range(1, years_ahead + 1)]
    ).reshape(-1, 1)

    predictions = model.predict(future_years)
    return future_years.flatten(), predictions

def generate_insight(predictions):
    trend = "Increasing" if predictions[-1] > predictions[0] else "Decreasing"

    if trend == "Increasing":
        risk = "Potential surplus"
        recommendation = "Plan storage, export, or crop diversification"
    else:
        risk = "Potential shortage"
        recommendation = "Encourage increased cultivation or alternatives"

    return trend, risk, recommendation
