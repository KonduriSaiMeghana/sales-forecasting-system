from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):

    lag_1: float = Field(..., description="1-day lag of sales")
    lag_7: float = Field(..., description="7-day lag of sales")
    lag_30: float = Field(..., description="30-day lag of sales")

    rolling_mean_7: float = Field(..., description="7-day rolling mean")
    rolling_std_7: float = Field(..., description="7-day rolling std")

    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0-6)")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")

    is_holiday: int = Field(..., ge=0, le=1, description="Is holiday flag (0 or 1)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "lag_1": 5000000.0,
                "lag_7": 4800000.0,
                "lag_30": 4900000.0,
                "rolling_mean_7": 4950000.0,
                "rolling_std_7": 150000.0,
                "day_of_week": 2,
                "month": 5,
                "is_holiday": 0
            }
        }