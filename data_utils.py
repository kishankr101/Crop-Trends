import pandas as pd

def load_and_prepare_data(csv_file):

    df = pd.read_csv(csv_file)

    df.columns = (
        df.columns
        .str.replace('ï»¿', '', regex=False)
        .str.replace('"', '', regex=False)
        .str.strip()
    )

    return df
