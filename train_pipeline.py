import joblib
import os

from src.preprocessing.load_data import load_data
from src.preprocessing.clean_data import clean_data
from src.preprocessing.missing_dates import fill_missing_dates
from src.features.feature_engineering import create_features
from src.preprocessing.train_test_split import time_series_split

from src.models.arima_model import train_sarima
# from src.models.prophet_model import train_prophet  # Commented out due to compatibility issues
from src.models.xgboost_model import train_xgboost
# from src.models.lstm_model import train_lstm  # Commented out due to TensorFlow compatibility issues

from src.evaluation.model_selection import select_best_model

from src.utils.logger import logger
from src.utils.metrics import evaluate_model
from src.utils.helpers import save_dataframe


# ==========================================
# CREATE SAVED MODELS DIRECTORY
# ==========================================

os.makedirs("saved_models", exist_ok=True)


# ==========================================
# LOAD DATA
# ==========================================

logger.info("Loading dataset...")

df = load_data("data/raw/sales_data.xlsx")


# ==========================================
# PREPROCESSING
# ==========================================

logger.info("Cleaning data...")

df = clean_data(df)

df = fill_missing_dates(df)

df = create_features(df)


# ==========================================
# SAVE PROCESSED DATA
# ==========================================

os.makedirs("data/processed", exist_ok=True)

save_dataframe(
    df,
    "data/processed/final_features.csv"
)


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

train, test = time_series_split(df)

results = {}
trained_models = {}


# ==========================================
# SARIMA
# ==========================================

logger.info("Training SARIMA model...")

try:
    sarima_model, sarima_preds = train_sarima(train, test)
    
    if sarima_model is not None:
        sarima_metrics = evaluate_model(
            test['sales'],
            sarima_preds
        )
        print("SARIMA Metrics:", sarima_metrics)
        results['SARIMA'] = sarima_metrics['MAE']
        trained_models['SARIMA'] = sarima_model
except Exception as e:
    logger.error(f"SARIMA training failed: {str(e)}")


# ==========================================
# XGBOOST
# ==========================================

logger.info("Training XGBoost model...")

try:
    xgb_model, xgb_preds = train_xgboost(train, test)
    
    xgb_metrics = evaluate_model(
        test['sales'],
        xgb_preds
    )
    print("XGBoost Metrics:", xgb_metrics)
    results['XGBoost'] = xgb_metrics['MAE']
    trained_models['XGBoost'] = xgb_model
except Exception as e:
    logger.error(f"XGBoost training failed: {str(e)}")


# ==========================================
# MODEL COMPARISON
# ==========================================

if len(results) == 0:
    logger.error("No models trained successfully!")
    print("ERROR: No models trained successfully. Please check the logs.")
    exit(1)

print("\nModel Comparison:")
print(results)

best_model_name = select_best_model(results)

print(f"\nBest Model: {best_model_name}")


# ==========================================
# SAVE BEST MODEL
# ==========================================

if best_model_name in trained_models:
    best_model = trained_models[best_model_name]
    
    model_data = {
        "model": best_model,
        "model_name": best_model_name,
        "metrics": results[best_model_name]
    }
    
    joblib.dump(model_data, "saved_models/model.pkl")
    
    print("\nBest model saved successfully!")
    print(f"Model: {best_model_name}")
    print(f"MAE Score: {results[best_model_name]}")
else:
    logger.error(f"Best model '{best_model_name}' not found in trained models!")
    print("ERROR: Could not save best model.")