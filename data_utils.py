import pandas as pd
import numpy as np

def load_and_prepare_data():
    """
    Loads raw data and converts wide format to long format.
    Returns cleaned dataframe.
    """
    # ---- paste your Colab data loading + reshaping logic here ----
    return df_long


def filter_data(df, state, season):
    """
    Filters data by state and season.
    """
    return df[(df["State"] == state) & (df["Season"] == season)]
