from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings('ignore')


def train_sarima(train, test):
    """
    Train a SARIMA model for time series forecasting.
    
    Args:
        train: Training data with 'sales' column
        test: Test data with 'sales' column
        
    Returns:
        Tuple of (fitted model, predictions)
    """

    try:
        model = SARIMAX(
            train['sales'],
            order=(1,1,1),
            seasonal_order=(1,1,1,7),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        fitted = model.fit(disp=False, maxiter=200)

        preds = fitted.forecast(len(test))
        
        # Ensure predictions are valid
        preds = preds.fillna(preds.mean())

        return fitted, preds
    
    except Exception as e:
        print(f"SARIMA training error: {str(e)}")
        # Return fallback prediction (mean of training data)
        fallback_pred = [train['sales'].mean()] * len(test)
        return None, fallback_pred