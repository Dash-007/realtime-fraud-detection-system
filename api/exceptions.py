"""
Custom exceptions for the Fraud Detection API
"""

from fastapi import HTTPException, status


class ModelNotLoadedError(HTTPException):
    """Raised when model is not loaded but prediction is requested"""
    
    def __init__(self, detail: str = "Model not loaded. Please check server logs."):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail
        )


class InvalidTransactionError(HTTPException):
    """Raised when transaction data is invalid"""
    
    def __init__(self, detail: str = "Invalid transaction data"):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail
        )


class PredictionError(HTTPException):
    """Raised when prediction fails"""
    
    def __init__(self, detail: str = "Prediction failed"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )


class FeatureEngineeringError(HTTPException):
    """Raised when feature engineering fails"""
    
    def __init__(self, detail: str = "Feature engineering failed"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )