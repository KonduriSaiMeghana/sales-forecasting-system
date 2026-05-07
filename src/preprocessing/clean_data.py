import pandas as pd


def clean_data(df):

    # Standardize columns
    df.columns = df.columns.str.strip().str.lower()

    # Rename total -> sales
    df.rename(columns={
        'total': 'sales'
    }, inplace=True)

    # Convert date column
    df['date'] = pd.to_datetime(df['date'])

    # Sort values
    df = df.sort_values(['state', 'date'])

    # Fill missing sales
    df['sales'] = df['sales'].ffill()

    return df