import pandas as pd
import holidays


def create_features(df):

    india_holidays = holidays.India()

    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)

    df = df.sort_values(['state', 'date'])

    # Lag Features
    df['lag_1'] = df.groupby('state')['sales'].shift(1)
    df['lag_7'] = df.groupby('state')['sales'].shift(7)
    df['lag_30'] = df.groupby('state')['sales'].shift(30)

    # Rolling Features
    df['rolling_mean_7'] = (
        df.groupby('state')['sales']
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )

    df['rolling_std_7'] = (
        df.groupby('state')['sales']
        .transform(lambda x: x.rolling(7, min_periods=1).std())
    )

    # Date Features
    df['day_of_week'] = df['date'].dt.dayofweek.astype(int)
    df['month'] = df['date'].dt.month.astype(int)

    # Holiday Flag
    df['is_holiday'] = df['date'].apply(
        lambda x: 1 if pd.Timestamp(x).date() in india_holidays else 0
    ).astype(int)

    # Drop NaN from lagging
    df = df.dropna()
    
    # Ensure correct data types for model input
    for col in ['lag_1', 'lag_7', 'lag_30', 'rolling_mean_7', 'rolling_std_7']:
        df[col] = df[col].astype(float)

    return df