"""
Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional
from datetime import datetime

class TransactionFeatures(BaseModel):
    """
    Input features for a single transaction.
    """
    
    # Original features
    Time: float = Field(..., description="Seconds elapsed since first transaction")
    V1: float = Field(..., description="PCA transformed feature 1")
    V2: float = Field(..., description="PCA transformed feature 2")
    V3: float = Field(..., description="PCA transformed feature 3")
    V4: float = Field(..., description="PCA transformed feature 4")
    V5: float = Field(..., description="PCA transformed feature 5")
    V6: float = Field(..., description="PCA transformed feature 6")
    V7: float = Field(..., description="PCA transformed feature 7")
    V8: float = Field(..., description="PCA transformed feature 8")
    V9: float = Field(..., description="PCA transformed feature 9")
    V10: float = Field(..., description="PCA transformed feature 10")
    V11: float = Field(..., description="PCA transformed feature 11")
    V12: float = Field(..., description="PCA transformed feature 12")
    V13: float = Field(..., description="PCA transformed feature 13")
    V14: float = Field(..., description="PCA transformed feature 14")
    V15: float = Field(..., description="PCA transformed feature 15")
    V16: float = Field(..., description="PCA transformed feature 16")
    V17: float = Field(..., description="PCA transformed feature 17")
    V18: float = Field(..., description="PCA transformed feature 18")
    V19: float = Field(..., description="PCA transformed feature 19")
    V20: float = Field(..., description="PCA transformed feature 20")
    V21: float = Field(..., description="PCA transformed feature 21")
    V22: float = Field(..., description="PCA transformed feature 22")
    V23: float = Field(..., description="PCA transformed feature 23")
    V24: float = Field(..., description="PCA transformed feature 24")
    V25: float = Field(..., description="PCA transformed feature 25")
    V26: float = Field(..., description="PCA transformed feature 26")
    V27: float = Field(..., description="PCA transformed feature 27")
    V28: float = Field(..., description="PCA transformed feature 28")
    Amount: float = Field(..., ge=0, description="Transaction amount")
    
    @field_validator('Amount')
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError('Amount must be non-negative')
        return v
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "Time": 0.0,
                "V1": -1.359807,
                "V2": -0.072781,
                "V3": 2.536347,
                "V4": 1.378155,
                "V5": -0.338321,
                "V6": 0.462388,
                "V7": 0.239599,
                "V8": 0.098698,
                "V9": 0.363787,
                "V10": 0.090794,
                "V11": -0.551600,
                "V12": -0.617801,
                "V13": -0.991390,
                "V14": -0.311169,
                "V15": 1.468177,
                "V16": -0.470401,
                "V17": 0.207971,
                "V18": 0.025791,
                "V19": 0.403993,
                "V20": 0.251412,
                "V21": -0.018307,
                "V22": 0.277838,
                "V23": -0.110474,
                "V24": 0.066928,
                "V25": 0.128539,
                "V26": -0.189115,
                "V27": 0.133558,
                "V28": -0.021053,
                "Amount": 149.62
            }
        }
    }
    
class PredictionResponse(BaseModel):
    """
    Response model for fraud prediction.
    """
    
    is_fraud: bool = Field(..., description="Whether transaction is classified as fraud")
    fraud_probability: float = Field(..., ge=0, le=1, description="Probability of fraud (0-1)")
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, or HIGH")
    threshold_used: float = Field(..., description="Decision threshold applied")
    model_version: str = Field(..., description="Model version used for prediction")
    prediction_id: str = Field(..., description="Unique ID for this prediction")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "is_fraud": False,
                "fraud_probability": 0.125,
                "risk_level": "LOW",
                "threshold_used": 0.704,
                "model_version": "ensemble_v1",
                "prediction_id": "pred_123e4567-e89b-12d3-a456-426614174000",
                "timestamp": "2025-07-15T10:30:00Z"
            }
        }
    }
    
class BatchPredictionRequest(BaseModel):
    """
    Request model for batch predictions.
    """
    transactions: List[TransactionFeatures]
    
    @field_validator('transactions')
    @classmethod
    def validate_batch_size(cls, v: List[TransactionFeatures]) -> List[TransactionFeatures]:
        if len(v) > 100:
            raise ValueError('Batch size cannot exceed 100 transactions')
        if len(v) == 0:
            raise ValueError('Batch must contain at least one transaction')
        return v
    
class HealthResponse(BaseModel):
    """
    Health check response.
    """
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    
class ErrorResponse(BaseModel):
    """
    Error response model.
    """
    error: str
    detail: str
    status_code: int
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "Validation Error",
                "detail": "Amount must be non-negative",
                "status_code": 422
            }
        }
    }