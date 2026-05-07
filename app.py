from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
import os
import numpy as np
from src.api.schemas import PredictionRequest

app = FastAPI(
    title="Sales Forecasting API",
    description="Machine Learning API for sales predictions",
    version="1.0.0"
)


# ==========================================
# LOAD SAVED MODEL
# ==========================================

model = None
model_name = None
model_loaded = False

try:
    if os.path.exists("saved_models/model.pkl"):
        saved_data = joblib.load("saved_models/model.pkl")
        model = saved_data.get("model")
        model_name = saved_data.get("model_name")
        model_loaded = True
except Exception as e:
    print(f"Error loading model: {str(e)}")


# ==========================================
# HEALTH CHECK ROUTE
# ==========================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "model_name": model_name if model_loaded else "None"
    }


# ==========================================
# HOME ROUTE
# ==========================================

@app.get("/")
def home():
    return {
        "message": "Forecasting API Running",
        "best_model": model_name if model_loaded else "No model loaded",
        "model_status": "ready" if model_loaded else "not_ready",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


# ==========================================
# PREDICTION ROUTE
# ==========================================

@app.post("/predict")
def predict(data: PredictionRequest):
    """
    Make a sales prediction using the trained model.
    
    Args:
        data: PredictionRequest containing all required features
        
    Returns:
        Dictionary with prediction value and model metadata
    """
    
    if not model_loaded or model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the pipeline first using train_pipeline.py"
        )

    try:
        # Convert request data to DataFrame
        input_dict = data.model_dump()
        df = pd.DataFrame([input_dict])
        
        # Ensure all values are numeric
        df = df.astype(float)
        
        # Make prediction
        prediction = model.predict(df)
        prediction_value = float(prediction[0])
        
        # Check for invalid predictions
        if np.isnan(prediction_value) or np.isinf(prediction_value):
            raise ValueError("Model returned invalid prediction (NaN or Inf)")
        
        return {
            "status": "success",
            "model_used": model_name,
            "prediction": round(prediction_value, 2),
            "input_features": input_dict
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")