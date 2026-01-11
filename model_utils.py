import numpy as np
from sklearn.linear_model import LinearRegression

def train_model(df):
    """
    Trains regression model and returns model + last production value.
    """
    X = df[["Year"]]
    y = df["Production"]

    model = LinearRegression()
    model.fit(X, y)

    return model, y.iloc[-1]


def forecast_future(model, last_value):
    """
    Forecasts future production and classifies result.
    """
    future_years = np.array([2026, 2027, 2028]).reshape(-1, 1)
    preds = model.predict(future_years)
    avg_future = preds.mean()

    if avg_future > last_value * 1.05:
        status = "Likely Surplus"
    elif avg_future < last_value * 0.95:
        status = "Likely Shortage"
    else:
        status = "Balanced"

    recommendation = (
        f"Future production trend indicates **{status}**. "
        f"Average expected production ≈ {avg_future:.2f}."
    )

    return future_years.flatten(), preds, status, recommendation
