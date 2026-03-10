import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def preprocess_data(df, crop, state):

    df = df[(df["Crop"] == crop) & (df["State"] == state)]

    if df.empty:
        return pd.DataFrame()

    production_cols = [c for c in df.columns if "Production" in c]

    long_df = df.melt(
        id_vars=["Crop", "State", "Season"],
        value_vars=production_cols,
        var_name="Year_Col",
        value_name="Production"
    )

    long_df["Year"] = long_df["Year_Col"].str.extract(r"(\d{4})").astype(int)

    long_df.dropna(subset=["Production"], inplace=True)

    long_df.sort_values(["Season", "Year"], inplace=True)

    return long_df


def train_models(df_long):

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
            "last": y.iloc[-1]
        }

    return results


def predict_future(results, start_year, years_ahead=5):

    future_years = np.arange(start_year+1, start_year+years_ahead+1).reshape(-1,1)

    rows = []

    for season, info in results.items():

        preds = info["model"].predict(future_years)

        for year, value in zip(future_years.flatten(), preds):

            rows.append({
                "Season": season,
                "Year": int(year),
                "Production": float(value)
            })

    return pd.DataFrame(rows)
