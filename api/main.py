"""
FastAPI application for fraud detection
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import joblib
import uuid
from datetime import datetime, timedelta
from typing import List
import logging
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from api.models import (
    TransactionFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    HealthResponse,
    ErrorResponse
)
from api.config import (
    API_TITLE,
    API_VERSION,
    API_DESCRIPTION,
    MODEL_PATH,
    MODEL_VERSION,
    LOG_LEVEL
)

# Set up logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Global variable for model and stats
model_package = None
app_start_time = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for FastAPI app.
    Loads model on startup, cleanup on shutdown.
    """
    global model_package, app_start_time
    
    # Startup
    logger.info("Starting Fraud Detection API...")
    
    try:
        # Load model package
        logger.info(f"Loading model from {MODEL_PATH}")
        model_package = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully")
        
        # Set start time for uptime calculation
        app_start_time = datetime.utcnow()
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")
    
# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan
)

# Add CORS middleware for web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # specify actual origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
    
# Helper functions
def prepare_features(transaction: TransactionFeatures) -> pd.DataFrame:
    """
    Convert transaction to DataFrame with engineered features.
    """
    # Convert to dict and create DataFrame
    data = transaction.model_dump()
    df = pd.DataFrame([data])
    
    # Apply feature engineering
    engineer = model_package['feature_engineer']
    df_engineered = engineer.transform(df)
    
    # Ensure all expected features
    expected_features = model_package['feature_names']
    df_final = df_engineered[expected_features]
    
    return df_final

def make_prediction(features_df: pd.DataFrame) -> dict:
    """
    Make fraud prediction using the loaded model.
    """
    # Get model components
    ensemble_model = model_package['ensemble_model']
    scaler = model_package['scaler']
    threshold = model_package['optimal_threshold']
    
    # Scale features
    features_scaled = scaler.transform(features_df)
    
    # Get probability
    fraud_probability = ensemble_model.predict_proba(features_scaled)[0, 1]
    
    # Apply threshold
    is_fraud = fraud_probability >= threshold
    
    # Determine risk level
    if fraud_probability > 0.8:
        risk_level = "HIGH"
    elif fraud_probability > 0.5:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
        
    return {
        'is_fraud': bool(is_fraud),
        'fraud_probability': float(fraud_probability),
        'risk_level': risk_level,
        'threshold_used': float(threshold)
    }
    
# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """
    Welcome endpoint
    """
    return {
        "message": "Fraud Detection API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health"
    }
    
@app.get("/health",
         response_model=HealthResponse,
         tags=["General"])
async def health_check():
    """
    Check API and model health
    """
    if model_package is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model not loaded"
        )
        
    uptime = (datetime.utcnow() - app_start_time).total_seconds()
    
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_version=MODEL_VERSION,
        uptime_seconds=uptime
    )
    
@app.post("/predict",
          response_model=PredictionResponse,
          tags=["Predictions"])
async def predict_single(transaction: TransactionFeatures):
    """
    Predict fraud for a single transaction.
    
    Returns prediction with probability and risk assessment.
    """
    
    try:
        # Prepare features
        features_df = prepare_features(transaction)
        
        # Make prediction
        prediction = make_prediction(features_df)
        
        # Generate predition ID
        prediction_id = f"pred_{uuid.uuid4()}"
        
        # Log predition
        logger.info(f"Prediction {prediction_id}: fraud={prediction['is_fraud']},"
                    f"prob={prediction['fraud_probability']:.3f}")
        
        return PredictionResponse(
            is_fraud=prediction['is_fraud'],
            fraud_probability=prediction['fraud_probability'],
            risk_level=prediction['risk_level'],
            threshold_used=prediction['threshold_used'],
            model_version=MODEL_VERSION,
            prediction_id=prediction_id,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )
        
@app.post("/predict/batch",
          response_model=List[PredictionResponse],
          tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict fraud for multiple transactions.
    
    Maximum batch size: 100 transactions
    """
    predictions = []
    
    for transaction in request.transactions:
        try:
            # Prepare features
            features_df = prepare_features(transaction)
            
            # Make prediction
            prediction = make_prediction(features_df)
            
            # Create response
            predictions.append(PredictionResponse(
                is_fraud=prediction['is_fraud'],
                fraud_probability=prediction['fraud_probability'],
                risk_level=prediction['risk_level'],
                threshold_used=prediction['threshold_used'],
                model_version=MODEL_VERSION,
                prediction_id=f"pred_{uuid.uuid4()}",
                timestamp=datetime.utcnow()
            ))
            
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            # Continue with other predictions
            
    logger.info(f"Batch prediction completed: {len(predictions)} transactions")
    
    return predictions


@app.get("/model/info", tags=["Model"])
async def model_info():
    """
    Get information about the current model
    """
    
    if model_package is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "model_version": MODEL_VERSION,
        "model_type": "Ensemble (RandomForest + XGBoost)",
        "threshold": model_package['optimal_threshold'],
        "features_count": len(model_package['feature_names']),
        "feature_names": model_package['feature_names'][:10] + ["..."],  # Show first 10
        "performance": {
            "precision": 0.919,
            "recall": 0.806,
            "f1_score": 0.859,
            "false_alarm_ratio": 0.089
        }
    }


@app.post("/analyze", tags=["Analysis"])
async def analyze_transaction(transaction: TransactionFeatures):
    """
    Analyze a transaction with detailed breakdown.
    Useful for debugging and understanding predictions.
    """
    try:
        # Prepare features
        features_df = prepare_features(transaction)
        
        # Get prediction details
        ensemble_model = model_package['ensemble_model']
        scaler = model_package['scaler']
        
        # Scale features
        features_scaled = scaler.transform(features_df)
        
        # Get individual model predictions (if using voting classifier)
        predictions_individual = {}
        for name, estimator in ensemble_model.named_estimators_.items():
            prob = estimator.predict_proba(features_scaled)[0, 1]
            predictions_individual[name] = float(prob)
        
        # Get ensemble probability
        fraud_probability = ensemble_model.predict_proba(features_scaled)[0, 1]
        
        # Feature statistics
        feature_stats = {
            "amount": float(transaction.Amount),
            "amount_percentile": float((features_df['Amount_scaled'].values[0] + 3) / 6 * 100),  # Rough estimate
            "high_risk_features": []
        }
        
        # Identify high-risk feature values
        for feature in ['V10', 'V14', 'V16', 'V17']:
            if hasattr(transaction, feature):
                value = getattr(transaction, feature)
                if value < -2:  # Based on EDA, negative values indicate fraud
                    feature_stats["high_risk_features"].append(feature)
        
        return {
            "fraud_probability": float(fraud_probability),
            "individual_model_predictions": predictions_individual,
            "feature_analysis": feature_stats,
            "recommendation": "BLOCK" if fraud_probability > 0.8 else "REVIEW" if fraud_probability > 0.5 else "APPROVE"
        }
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)