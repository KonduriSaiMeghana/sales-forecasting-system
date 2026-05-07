from xgboost import XGBRegressor
import numpy as np


FEATURES = [
    'lag_1',
    'lag_7',
    'lag_30',
    'rolling_mean_7',
    'rolling_std_7',
    'day_of_week',
    'month',
    'is_holiday'
]


def train_xgboost(train, test):

    X_train = train[FEATURES].astype(float).fillna(0)
    y_train = train['sales'].astype(float).fillna(train['sales'].mean())

    X_test = test[FEATURES].astype(float).fillna(0)
    y_test = test['sales'].astype(float).fillna(test['sales'].mean())
    
    # Handle any remaining NaN or inf values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        verbosity=0
    )

    model.fit(X_train, y_train, verbose=False)

    preds = model.predict(X_test)

    return model, preds