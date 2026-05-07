import pandas as pd


def save_dataframe(df, path):

    df.to_csv(path, index=False)

    print(f"Data saved at: {path}")


def load_processed_data(path):

    df = pd.read_csv(path)

    return df