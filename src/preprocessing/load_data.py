import pandas as pd

def load_data(path):
    df = pd.read_excel(path)

    print(df.head())
    print(df.info())

    return df


if __name__ == "__main__":
    df = load_data("data/raw/sales_data.xlsx")