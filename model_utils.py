import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def preprocess_data(df, crop, state):
    """
    Converts wide production columns into long time-series format
    for a selected crop and state.
    """
    df = df[(df["Crop"] == crop) & (df["State"] == state)].copy()

    if df.empty:
        return pd.DataFrame()

    # Clean column names
    df.columns = df.columns.str.replace('ï»¿', '', regex=False).str.strip()

    production_cols = [c for c in df.columns if "Production" in c]

    long_df = df.melt(
        id_vars=["Crop", "State", "Season"],
        value_vars=production_cols,
        var_name="Year",
        value_name="Production"
    )

    long_df["Year"] = long_df["Year"].str.extract(r"(\d{4})").astype(int)
    long_df.dropna(subset=["Production"], inplace=True)
    long_df.sort_values(["Season", "Year"], inplace=True)

    return long_df


def train_models(df_long):
    """
    Trains one Linear Regression model per season.
    """
    results = {}

    for season in df_long["Season"].unique():
        season_df = df_long[df_long["Season"] == season]

        if len(season_df) < 2:
            continue

        X = season_df[["Year"]]
        y = season_df["Production"]

        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)

        results[season] = {
            "model": model,
            "mae": mean_absolute_error(y, y_pred),
            "r2": r2_score(y, y_pred),
            "last_production": y.iloc[-1]
        }

    return results


def predict_future(results, start_year, years_ahead=5):
    """
    Predicts future production for each season.
    """
    future_years = np.array(
        [start_year + i for i in range(1, years_ahead + 1)]
    ).reshape(-1, 1)

    predictions = []

    for season, info in results.items():
        preds = info["model"].predict(future_years)

        for year, value in zip(future_years.flatten(), preds):
            predictions.append({
                "Season": season,
                "Year": year,
                "Predicted_Production": round(value, 2)
            })

    return pd.DataFrame(predictions)


def generate_insights(results, predictions_df):
    """
    Generates surplus / shortage insights.
    """
    insights = []

    for season, info in results.items():
        season_preds = predictions_df[predictions_df["Season"] == season]

        if season_preds.empty:
            continue

        avg_future = season_preds["Predicted_Production"].mean()
        last_val = info["last_production"]

        if avg_future > last_val * 1.05:
            status = "Likely Surplus"
            recommendation = "Plan storage, exports, or diversification"
        elif avg_future < last_val * 0.95:
            status = "Likely Shortage"
            recommendation = "Encourage increased cultivation"
        else:
            status = "Balanced"
            recommendation = "Maintain current strategy"

        insights.append({
            "Season": season,
            "Avg_Future_Production": round(avg_future, 2),
            "Status": status,
            "Recommendation": recommendation
        })

    return pd.DataFrame(insights)
