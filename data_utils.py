import pandas as pd
import os

DATA_PATH = os.path.join(
    "data",
    "crop_data.csv"
)

def load_and_prepare_data():
    """
    Loads raw crop data and converts wide production columns
    into long (year-wise) format.
    """
    df = pd.read_csv(DATA_PATH)

    # Clean column names (BOM + quotes)
    df.columns = (
        df.columns
        .str.replace('ï»¿', '', regex=False)
        .str.replace('"', '', regex=False)
        .str.strip()
    )

    # Identify production columns
    prod_cols = [c for c in df.columns if "Production-" in c]

    df_long = df.melt(
        id_vars=["Crop", "State", "Season"],
        value_vars=prod_cols,
        var_name="Year_Col",
        value_name="Production"
    )

    df_long["Year"] = (
        df_long["Year_Col"]
        .str.extract(r"(\d{4})")
        .astype(int)
    )

    df_long = df_long.dropna(subset=["Production"])

    return df_long


def filter_data(df_long, state, season):
    """
    Filters long-format data by state and season.
    """
    return df_long[
        (df_long["State"] == state) &
        (df_long["Season"] == season)
    ]
