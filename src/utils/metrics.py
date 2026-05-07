import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)


def evaluate_model(y_true, preds):

    mae = mean_absolute_error(y_true, preds)

    rmse = np.sqrt(
        mean_squared_error(y_true, preds)
    )

    mape = mean_absolute_percentage_error(
        y_true,
        preds
    )

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2)
    }